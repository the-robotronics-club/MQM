"""Photo analysis and live campus dining congestion dashboard."""

from __future__ import annotations

import hashlib
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import plotly.express as px
import streamlit as st
from dotenv import load_dotenv
from streamlit_autorefresh import st_autorefresh

from dashboard.client import DashboardAPIError, fetch_history, fetch_status
from dashboard.messes import (
    CAMPUS_FACILITIES,
    DiningFacility,
    crowd_badge_for_level,
    estimate_wait_minutes,
    fallback_metrics_for_mess,
    get_all_canteens,
    get_all_facilities,
    get_all_messes,
    get_current_meal_slot,
    get_facility,
)
from dashboard.student_auth import (
    get_current_student,
    is_authenticated,
    render_login_view,
    render_student_badge,
)

load_dotenv()

CROWD_COLOURS = {"green": "#16a34a", "amber": "#d97706", "red": "#dc2626"}
REPO_ROOT = Path(__file__).resolve().parent.parent
LOCAL_SAMPLE_DIR = REPO_ROOT / "data" / "samples" / "local"
_TEST_FIXTURES = {"mess_hall_dense.jpg", "mess_hall_alt.jpg"}


def local_sample_images() -> list[Path]:
    """Return user-provided local images without exposing test fixtures in the UI."""
    if not LOCAL_SAMPLE_DIR.is_dir():
        return []
    return sorted(
        (
            path
            for path in LOCAL_SAMPLE_DIR.iterdir()
            if path.is_file()
            and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
            and path.name not in _TEST_FIXTURES
        ),
        key=lambda path: path.name.lower(),
    )


def render_upload() -> None:
    """Render photo upload and local sample analysis view."""
    from dashboard.image_analysis import analyze_photo, decode_photo

    st.caption("Upload a photo or choose a local test image to count people and analyze mess zones.")
    upload_col, sample_col = st.columns(2)
    with upload_col:
        uploaded = st.file_uploader(
            "Choose an image",
            type=["jpg", "jpeg", "png"],
            key="photo_upload",
        )
    with sample_col:
        samples = local_sample_images()
        selected_sample = st.selectbox(
            "Or choose a local test image",
            options=[None, *samples],
            format_func=lambda path: "Select a local image" if path is None else path.name,
            help="Images in data/samples/local are only for this machine and are not uploaded.",
        )

    image_data: bytes | None = uploaded.getvalue() if uploaded else None
    image_name: str | None = uploaded.name if uploaded else None
    if image_data is None and selected_sample is not None:
        try:
            image_data = selected_sample.read_bytes()
            image_name = selected_sample.name
        except OSError as exc:
            st.error(f"Could not read {selected_sample.name}: {exc}")
            return

    calibrated = st.toggle(
        "Use saved mess layout for queue and seating estimates",
        help="Only for photos from the same viewpoint and resolution as the saved camera layout.",
    )
    camera_id = None
    if calibrated:
        from inference.zones import DEFAULT_CONFIG_PATH

        cameras = json.loads(DEFAULT_CONFIG_PATH.read_text())["cameras"]
        camera_id = st.selectbox("Saved layout", list(cameras))
        camera = cameras[camera_id]
        st.info(
            f"This layout requires {camera['frame_width']} × {camera['frame_height']} pixels "
            "and the same camera viewpoint. Zones and thresholds are calibrated estimates."
        )

    rotation = 0
    if image_data is not None:
        rotation = (
            st.segmented_control(
                "Rotate photo until people are upright",
                options=[0, 90, 180, 270],
                default=0,
                format_func=lambda value: {
                    0: "No rotation",
                    90: "90° left",
                    180: "180°",
                    270: "90° right",
                }[value],
                selection_mode="single",
            )
            or 0
        )
        try:
            preview = decode_photo(image_data, rotation)
        except ValueError as exc:
            st.session_state.pop("photo_result", None)
            st.error(str(exc))
            return
        st.image(preview, caption=image_name, width="stretch")

    analyze = st.button("Analyze image", type="primary", disabled=image_data is None)
    selection = (
        hashlib.sha256(image_data).hexdigest() if image_data else None,
        camera_id,
        rotation,
    )
    if st.session_state.get("photo_selection") != selection:
        st.session_state.pop("photo_result", None)
        st.session_state["photo_selection"] = selection

    if analyze and image_data is not None:
        st.session_state.pop("photo_result", None)
        try:
            with st.spinner("Detecting people… The first analysis may take a little longer."):
                analyzed_result = analyze_photo(image_data, camera_id, rotation)
                analyzed_result["name"] = image_name
                st.session_state["photo_result"] = analyzed_result
        except ValueError as exc:
            st.error(str(exc))
        except Exception:
            st.error(
                "Image analysis is unavailable. Check that local model weights are installed."
            )
            logging.getLogger(__name__).exception("Photo analysis failed")

    result = st.session_state.get("photo_result")
    if result is None:
        return

    st.subheader(f"Analysis: {result['name']}")
    with st.container(border=True):
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("People detected", result["headcount"], border=True)
        if result.get("sitting") is not None:
            col2.metric("Queue estimate", result["queue_count"], border=True)
            col3.metric("Sitting estimate", result["sitting"], border=True)
            col4.metric("Other standing", result["standing_other_count"], border=True)

    if result.get("scene") is not None:
        from inference.annotations import LEGEND

        st.caption(LEGEND)
        st.caption("Distance-based queue estimate. Nearby groups can be mistaken for a queue.")
        reassigned = result["scene"].get("rescued_count", 0)
        if reassigned:
            st.caption(
                f"Queue grouping reassigned {reassigned} uncertain sitting predictions to the queue."
            )
    if result.get("posture_error"):
        st.warning(result["posture_error"])

    metrics = result["metrics"]
    if metrics is not None:
        with st.container(border=True):
            queue, wait, seats, occupancy = st.columns(4)
            wait_time = estimate_wait_minutes(metrics["queue_count"])
            queue.metric("Queue inside layout", metrics["queue_count"], border=True)
            wait.metric("Est. wait time", f"{wait_time} min", border=True)
            seats.metric(
                "Seats occupied",
                f"{metrics['seats_occupied']}/{metrics['seats_total']}",
                border=True,
            )
            occupancy.metric("Occupancy", f"{metrics['seat_occupancy_pct']:.1f}%", border=True)
            badge_text, _ = crowd_badge_for_level(metrics["crowd_level"])
            st.markdown(f"Crowd status: {badge_text}")
    else:
        st.info("Seat occupancy and calibrated crowd levels need a saved mess layout for this view.")

    st.image(result["image"], caption="Predicted queue and posture categories", width="stretch")
    st.download_button(
        "Download annotated image",
        result["image"],
        "people-detected.png",
        "image/png",
        icon=":material/download:",
    )


def api_base_url() -> str | None:
    """Return API base URL or None if not configured."""
    try:
        secret_value = st.secrets.get("READ_API_BASE_URL")
    except (FileNotFoundError, KeyError):
        secret_value = None
    value = str(secret_value or os.environ.get("READ_API_BASE_URL", "")).strip()
    if not value and os.environ.get("READ_API_HOST"):
        host = os.environ["READ_API_HOST"]
        port = int(os.environ.get("READ_API_PORT", "10000"))
        value = f"http://{host}:{port}"
    return value.rstrip("/") if value else None


@st.cache_data(ttl=5, show_spinner=False)
def cached_status(base_url: str) -> dict:
    """Fetch camera statuses from Read API."""
    return fetch_status(base_url)


@st.cache_data(ttl=5, show_spinner=False)
def cached_history(base_url: str, camera_id: str) -> list[dict]:
    """Fetch 60-minute congestion history from Read API."""
    return fetch_history(base_url, camera_id, minutes=60)


def render_calibrated_metrics(metrics: dict, facility: DiningFacility | None = None) -> None:
    """Render structured metrics cards with wait times and occupancy."""
    crowd_level = str(metrics.get("crowd_level", "green"))
    badge_md, _ = crowd_badge_for_level(crowd_level)
    queue_count = int(metrics.get("queue_count", 0))
    facility_type = facility.facility_type if facility else "mess"
    wait_time = estimate_wait_minutes(queue_count, facility_type)
    headcount = int(metrics.get("headcount", 0))
    seats_occ = metrics.get("seats_occupied")
    seats_tot = metrics.get("seats_total") or (facility.capacity if facility else 130)
    occ_pct = metrics.get("seat_occupancy_pct")
    if occ_pct is None and seats_occ is not None and seats_tot:
        occ_pct = round((seats_occ / seats_tot) * 100, 1)

    with st.container(border=True):
        header_l, header_r = st.columns([3, 1], vertical_alignment="center")
        with header_l:
            st.markdown(f"**Current crowd status:** {badge_md}")
        with header_r:
            ts = metrics.get("timestamp", datetime.now().strftime("%H:%M:%S"))
            if "T" in str(ts):
                ts = str(ts).split("T")[1][:8]
            st.caption(f":material/schedule: Updated: {ts}")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Est. wait time", f"{wait_time} min", border=True)
        c2.metric("Queue count", f"{queue_count} people", border=True)
        c3.metric("Total in venue", f"{headcount} people", border=True)
        if seats_occ is not None:
            c4.metric(
                "Seat occupancy",
                f"{occ_pct:.0f}%",
                f"{seats_occ}/{seats_tot} seats",
                border=True,
            )
        else:
            c4.metric("Seating capacity", f"{seats_tot} seats", border=True)

    if metrics.get("queue_visible") is not None:
        with st.expander(":material/analytics: Research & queue diagnostics"):
            diag1, diag2, diag3 = st.columns(3)
            diag1.metric("Visible in queue", metrics["queue_visible"], border=True)
            diag2.metric(
                "Occlusion correction",
                f"+{metrics.get('queue_hidden_estimate', 0)}",
                border=True,
            )
            if metrics.get("queue_ci_low") is not None:
                diag3.metric(
                    "Confidence range",
                    f"{metrics['queue_ci_low']}–{metrics['queue_ci_high']}",
                    border=True,
                )
            st.caption(
                f"Method: {metrics.get('queue_method', 'calibrated')} · "
                f"Model: {metrics.get('queue_model_version', 'yolo11n-seg')}"
            )


def render_history_chart(base_url: str | None, camera_id: str) -> None:
    """Render historical line chart for congestion trends."""
    points: list[dict[str, Any]] = []
    if base_url:
        try:
            points = cached_history(base_url, camera_id)
        except DashboardAPIError:
            points = []

    if not points:
        st.info("Live history will plot automatically as readings are logged to InfluxDB.")
        return

    figure = px.line(
        points,
        x="timestamp",
        y=["headcount", "queue_count"],
        labels={"value": "People", "timestamp": "Time", "variable": "Metric"},
        title="Last 60 Minutes Queue & Crowd History",
    )
    figure.update_layout(margin=dict(l=8, r=8, t=40, b=8), legend_title_text="")
    st.plotly_chart(figure, width="stretch")


def select_facility(facility_id: str) -> None:
    """Set selected dining facility and switch to live cameras view."""
    st.session_state["selected_dining_hall"] = facility_id
    st.session_state["dashboard_mode"] = "Live cameras"


def render_facility_card(
    facility: DiningFacility,
    live_cameras: dict[str, Any],
) -> None:
    """Render a card for one dining facility."""
    camera_data = live_cameras.get(facility.camera_id, {})
    is_live = bool(camera_data.get("online") and camera_data.get("metrics"))
    metrics = camera_data.get("metrics") if is_live else fallback_metrics_for_mess(facility.id)

    queue_count = int(metrics.get("queue_count", 0))
    wait_time = estimate_wait_minutes(queue_count, facility.facility_type)
    crowd_level = str(metrics.get("crowd_level", "green"))
    badge_md, _ = crowd_badge_for_level(crowd_level)

    with st.container(border=True):
        icon = ":material/restaurant:" if facility.facility_type == "mess" else ":material/local_cafe:"
        st.markdown(f"#### {icon} {facility.name}")
        st.caption(f":material/location_on: {facility.location}")
        st.markdown(f"Status: {badge_md}")

        col_a, col_b = st.columns(2)
        col_a.metric("Est. wait", f"{wait_time} min", border=True)
        col_b.metric("Queue", f"{queue_count}", border=True)

        st.caption(f"*{facility.specialty}*")
        slot_name, _, _ = get_current_meal_slot()
        menu_highlight = facility.current_menu.get(
            slot_name, facility.current_menu.get("All-Day", "Daily Specials")
        )
        with st.expander(f":material/menu_book: {slot_name} menu"):
            st.write(menu_highlight)

        st.button(
            "View live",
            key=f"select_btn_{facility.id}",
            icon=":material/visibility:",
            on_click=select_facility,
            args=(facility.id,),
        )


def render_all_messes_comparison(base_url: str | None, live_cameras: dict[str, Any]) -> None:
    """Render grouped comparison across all campus dining halls and canteens."""
    st.subheader(":material/compare_arrows: Campus Dining Comparison")
    st.caption("Compare live congestion across all hostel messes and canteens to pick the fastest queue.")

    tab_messes, tab_canteens, tab_fastest = st.tabs([
        ":material/restaurant: Hostel messes (Tulsi, Pine, Oak, Alder, Ganga, Saraswati)",
        ":material/local_cafe: Campus canteens (Monal, Drongo, Bulbul, Bake-O-Mocha)",
        ":material/speed: Shortest queues right now",
    ])

    with tab_messes:
        messes = get_all_messes()
        # 2 rows of 3 columns
        cols1 = st.columns(3)
        for idx, mess in enumerate(messes[:3]):
            with cols1[idx]:
                render_facility_card(mess, live_cameras)
        cols2 = st.columns(3)
        for idx, mess in enumerate(messes[3:6]):
            with cols2[idx]:
                render_facility_card(mess, live_cameras)

    with tab_canteens:
        canteens = get_all_canteens()
        cols_c = st.columns(4)
        for idx, canteen in enumerate(canteens):
            with cols_c[idx]:
                render_facility_card(canteen, live_cameras)

    with tab_fastest:
        st.markdown("##### ⚡ Quickest dining options right now")
        all_facs = get_all_facilities()
        ranked = []
        for fac in all_facs:
            camera_data = live_cameras.get(fac.camera_id, {})
            is_live = bool(camera_data.get("online") and camera_data.get("metrics"))
            metrics = camera_data.get("metrics") if is_live else fallback_metrics_for_mess(fac.id)
            q = int(metrics.get("queue_count", 0))
            w = estimate_wait_minutes(q, fac.facility_type)
            ranked.append((w, q, fac, metrics))

        ranked.sort(key=lambda item: item[0])

        fast_cols = st.columns(min(4, len(ranked)))
        for i in range(min(4, len(ranked))):
            w, q, fac, met = ranked[i]
            with fast_cols[i]:
                with st.container(border=True):
                    badge_md, _ = crowd_badge_for_level(met.get("crowd_level", "green"))
                    st.markdown(f"**#{i+1} {fac.name}**")
                    st.caption(f":material/location_on: {fac.location}")
                    st.markdown(badge_md)
                    st.metric("Estimated wait", f"{w} min", border=True)
                    st.button(
                        "Go to live feed",
                        key=f"fast_btn_{fac.id}",
                        icon=":material/arrow_forward:",
                        on_click=select_facility,
                        args=(fac.id,),
                    )


def render_single_mess_view(
    facility: DiningFacility,
    base_url: str | None,
    live_cameras: dict[str, Any],
) -> None:
    """Render focused live view for a specific mess or canteen."""
    icon = ":material/restaurant:" if facility.facility_type == "mess" else ":material/local_cafe:"
    st.subheader(f"{icon} {facility.name}")
    st.caption(
        f":material/location_on: {facility.location} · "
        f":material/group: Serving: {facility.hostels_served} · "
        f":material/restaurant_menu: Specialty: *{facility.specialty}*"
    )

    camera_data = live_cameras.get(facility.camera_id, {})
    is_live = bool(camera_data.get("online") and camera_data.get("metrics"))

    if is_live:
        metrics = camera_data["metrics"]
    else:
        metrics = fallback_metrics_for_mess(facility.id)
        if base_url:
            st.caption(
                ":orange-badge[Camera Offline / Dev Mode] Displaying baseline queue data. "
                "Camera will sync automatically once online."
            )

    render_calibrated_metrics(metrics, facility)

    slot_name, slot_time, _ = get_current_meal_slot()
    today_menu = facility.current_menu.get(slot_name) or facility.current_menu.get("All-Day")
    if today_menu:
        with st.expander(f":material/menu_book: Today's menu highlights ({slot_name} · {slot_time})"):
            st.info(f"**Specialties & counter items:** {today_menu}")

    render_history_chart(base_url, facility.camera_id)


def main() -> None:
    """Main Streamlit application entrypoint."""
    st.set_page_config(
        page_title="MQM · Mess Queue Monitor",
        page_icon=":material/restaurant:",
        layout="wide",
    )

    # Top Header & Institute Branding
    header_col, info_col = st.columns([3, 2], vertical_alignment="center")
    with header_col:
        st.title(":material/restaurant: MQM · Mess & Canteen Queue Monitor")
        st.caption("Real-time crowd detection, queue estimation, and dining hall occupancy.")

    with info_col:
        slot_name, slot_time, is_active = get_current_meal_slot()
        status_color = "green" if is_active else "orange"
        st.markdown(
            f":{status_color}-badge[{slot_name} ({slot_time})] :blue-badge[Auto-refresh 8s]"
        )

    if snapshot_only := os.environ.get("DASHBOARD_SNAPSHOT_ONLY", "false").lower() == "true":
        st.caption("Free demo hosting: services may sleep; history resets on server restart.")

    # Student Authentication Banner & Login Handling
    student = get_current_student()
    if student:
        render_student_badge()
    else:
        with st.expander(":material/account_circle: Student login & demo profiles", expanded=False):
            render_login_view()

    # Determine default facility based on student profile or session
    all_facilities = get_all_facilities()
    default_facility_id = "tulsi"
    if student and student.assigned_mess_id in CAMPUS_FACILITIES:
        default_facility_id = student.assigned_mess_id

    st.session_state.setdefault("selected_dining_hall", default_facility_id)

    # Navigation Modes
    modes = ["Live cameras", "All messes comparison"]
    if not snapshot_only:
        modes += ["Upload image", "Phone camera"]
    default_mode = os.environ.get("DASHBOARD_DEFAULT_MODE", "Live cameras")
    if st.session_state.get("dashboard_mode", default_mode) not in modes:
        st.session_state["dashboard_mode"] = "Live cameras"
    mode = (
        st.segmented_control(
            "Dashboard view",
            modes,
            default=default_mode if default_mode in modes else "Live cameras",
            key="dashboard_mode",
        )
        or "Live cameras"
    )

    # Route views
    if mode == "Phone camera":
        from dashboard.live_camera import render_live_camera

        render_live_camera()
        return

    if mode == "Upload image":
        render_upload()
        return

    st_autorefresh(interval=8_000, key="mess-status-refresh")

    # Fetch live camera statuses from Read API (graceful offline fallback)
    base_url = api_base_url()
    live_cameras: dict[str, Any] = {}
    if base_url:
        try:
            live_cameras = cached_status(base_url)
        except (RuntimeError, DashboardAPIError):
            live_cameras = {}

    if mode == "All messes comparison":
        render_all_messes_comparison(base_url, live_cameras)
        return

    # "Live cameras" Mode - With "Which Mess / Canteen" selection
    st.markdown("---")
    filter_col, select_col = st.columns([1, 3], vertical_alignment="bottom")

    with filter_col:
        category_choice = st.segmented_control(
            "Category",
            options=["All (10)", "Messes (6)", "Canteens (4)"],
            default="All (10)",
            key="category_filter",
        ) or "All (10)"

    if category_choice == "Messes (6)":
        selectable = get_all_messes()
    elif category_choice == "Canteens (4)":
        selectable = get_all_canteens()
    else:
        selectable = all_facilities

    selectable_ids = [f.id for f in selectable]
    current_selected = st.session_state.get("selected_dining_hall", default_facility_id)
    if current_selected not in selectable_ids:
        current_selected = selectable_ids[0]
        st.session_state["selected_dining_hall"] = current_selected

    with select_col:
        selected_id = (
            st.segmented_control(
                "Select mess or canteen",
                options=selectable_ids,
                format_func=lambda fid: get_facility(fid).name,
                key="selected_dining_hall",
            )
            or current_selected
        )

    active_facility = get_facility(selected_id)
    render_single_mess_view(active_facility, base_url, live_cameras)


if __name__ == "__main__":
    main()
