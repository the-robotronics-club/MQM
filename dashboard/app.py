"""Photo analysis and live campus dining congestion dashboard."""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any

import plotly.express as px
import streamlit as st
from dotenv import load_dotenv
from streamlit_autorefresh import st_autorefresh

from dashboard.client import DashboardAPIError, fetch_history, fetch_status
from dashboard.messes import (
    CAMPUS_MESSES,
    MessInfo,
    crowd_badge_for_level,
    estimate_wait_minutes,
    get_all_messes,
    get_current_meal_slot,
    get_mess,
)
from dashboard.student_auth import (
    get_current_student,
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

    st.caption(
        "Upload a photo or choose a local test image to detect people and analyze crowd zones."
    )
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
            "and the same camera viewpoint. Zones and thresholds are sample estimates."
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
            st.error("Image analysis is unavailable. Check that local model weights are installed.")
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
                f"Queue grouping reassigned {reassigned} uncertain sitting predictions "
                "to the queue."
            )
    if result.get("posture_error"):
        st.warning(result["posture_error"])

    metrics = result["metrics"]
    if metrics is not None:
        with st.container(border=True):
            queue, seats, occupancy, wait = st.columns(4)
            wait_time = estimate_wait_minutes(metrics["queue_count"])
            queue.metric("Queue inside layout", metrics["queue_count"], border=True)
            wait.metric("Illustrative wait", f"{wait_time} min", border=True)
            st.caption("Assumes 1.3 minutes per person; service rate is not calibrated.")
            seats.metric(
                "Seats occupied",
                f"{metrics['seats_occupied']}/{metrics['seats_total']}",
                border=True,
            )
            occupancy.metric("Occupancy", f"{metrics['seat_occupancy_pct']:.1f}%", border=True)
            badge_text, _ = crowd_badge_for_level(metrics["crowd_level"])
            st.markdown(f"Crowd status: {badge_text}")
    else:
        st.info(
            "Seat occupancy and calibrated crowd levels need a saved mess layout for this view."
        )

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


def render_calibrated_metrics(metrics: dict) -> None:
    """Display only measured categories and calibrated seat information."""
    with st.container(border=True):
        if metrics.get("scene_method") == "distance":
            with st.container(horizontal=True):
                st.metric("People detected", metrics["headcount"], border=True)
                st.metric("Queue estimate", metrics["queue_count"], border=True)
                st.metric("Sitting estimate", metrics["sitting"], border=True)
                st.metric("Other standing", metrics["standing_other_count"], border=True)
            st.caption("Snapshot estimates. Nearby groups can be mistaken for queues.")
        else:
            badge_md, _ = crowd_badge_for_level(metrics.get("crowd_level", "unknown"))
            st.markdown(f"**Current crowd status:** {badge_md}")
            with st.container(horizontal=True):
                st.metric("Headcount", metrics["headcount"], border=True)
                st.metric("Queue", metrics["queue_count"], border=True)
                occupied, total = metrics.get("seats_occupied"), metrics.get("seats_total")
                if occupied is not None and total is not None:
                    st.metric("Seats", f"{occupied}/{total}", border=True)
                percentage = metrics.get("seat_occupancy_pct")
                if percentage is not None:
                    st.metric("Occupied", f"{percentage:.1f}%", border=True)
        st.caption(f"Last update: {metrics['timestamp']}")
        st.caption(
            f"Illustrative wait: {estimate_wait_minutes(metrics['queue_count'])} min "
            "at an assumed 1.3 minutes per person. Service rate is not calibrated."
        )

    if metrics.get("queue_visible") is not None:
        with st.expander(":material/analytics: Queue diagnostics"):
            st.write(
                {
                    key: metrics.get(key)
                    for key in (
                        "queue_visible",
                        "queue_hidden_estimate",
                        "queue_ci_low",
                        "queue_ci_high",
                        "queue_method",
                        "queue_model_version",
                    )
                }
            )


def render_history_chart(base_url: str | None, camera_id: str) -> None:
    """Render historical line chart for congestion trends."""
    points: list[dict[str, Any]] = []
    if base_url:
        try:
            points = cached_history(base_url, camera_id)
        except DashboardAPIError as exc:
            st.warning(f"History unavailable: {exc}")
            return

    if not points:
        st.info("No history is available yet. It will appear after the first stored readings.")
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


def select_mess(mess_id: str) -> None:
    """Run before widgets are instantiated when a comparison card is selected."""
    st.session_state["selected_dining_hall"] = mess_id
    st.session_state["dashboard_mode"] = "Live cameras"


def render_all_messes_comparison(base_url: str | None, live_cameras: dict[str, Any]) -> None:
    """Render side-by-side cards comparing all campus dining halls."""
    st.subheader(":material/compare_arrows: Campus Mess Comparison")
    st.caption("Compare live congestion across all dining halls to choose the fastest queue.")

    messes = get_all_messes()
    cols = st.columns(len(messes))

    for idx, mess in enumerate(messes):
        with cols[idx]:
            camera_data = live_cameras.get(mess.camera_id, {})
            metrics = camera_data.get("metrics") if camera_data.get("online") else None

            with st.container(border=True):
                st.markdown(f"#### {mess.name}")
                st.caption(f":material/location_on: {mess.location}")
                if metrics:
                    st.success("Receiving snapshots")
                    col_a, col_b = st.columns(2)
                    col_a.metric("People detected", metrics["headcount"], border=True)
                    col_b.metric("Queue estimate", metrics["queue_count"], border=True)
                    st.caption(f"Last update: {metrics['timestamp']}")
                else:
                    st.warning("Camera offline — no fresh snapshot available.")

                st.caption(f":material/restaurant: *{mess.specialty}*")
                slot_name, _, _ = get_current_meal_slot()
                menu_highlight = mess.current_menu.get(slot_name, "Daily Special")
                with st.expander(f":material/menu_book: Sample {slot_name} menu"):
                    st.write(menu_highlight)

                st.button(
                    "View this mess live",
                    key=f"select_mess_btn_{mess.id}",
                    icon=":material/visibility:",
                    on_click=select_mess,
                    args=(mess.id,),
                )


def render_single_mess_view(
    mess: MessInfo,
    base_url: str | None,
    live_cameras: dict[str, Any],
) -> None:
    """Render focused live view for a specific mess."""
    st.subheader(f":material/restaurant: {mess.name}")
    st.caption(f":material/location_on: {mess.location} · Serving {mess.hostels_served}")

    camera_data = live_cameras.get(mess.camera_id, {})
    is_live = bool(camera_data.get("online") and camera_data.get("metrics"))

    if is_live:
        render_calibrated_metrics(camera_data["metrics"])
    else:
        st.error("Camera offline — no fresh reading was received in the configured TTL window.")
        st.caption("Start the Raspberry Pi uploader and wait for a Stored acknowledgement.")
        return

    slot_name, slot_time, _ = get_current_meal_slot()
    today_menu = mess.current_menu.get(slot_name)
    if today_menu:
        with st.expander(f":material/menu_book: Sample {slot_name} menu ({slot_time})"):
            st.info(f"**Menu highlights:** {today_menu}")

    render_history_chart(base_url, mess.camera_id)


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
        st.title(":material/restaurant: MQM · Mess Queue Management")
        st.caption("Real-time person detection, queue estimation, and dining hall occupancy.")

    with info_col:
        slot_name, slot_time, is_active = get_current_meal_slot()
        status_color = "green" if is_active else "orange"
        st.markdown(
            f":{status_color}-badge[{slot_name} ({slot_time})] :blue-badge[Auto-refresh 8s]"
        )

    if snapshot_only := os.environ.get("DASHBOARD_SNAPSHOT_ONLY", "false").lower() == "true":
        st.caption("Free demo hosting: services may sleep; history resets on server restart.")
    st.caption(
        "Dining hall names and menus are sample content. Only connected cameras show counts."
    )

    # Student Authentication Banner & Login Handling
    student = get_current_student()
    if student:
        render_student_badge()
    else:
        with st.expander(":material/account_circle: Demo student profiles", expanded=False):
            render_login_view()

    # Determine default mess based on student profile or session
    all_messes = get_all_messes()
    default_mess_id = all_messes[0].id
    if student and student.assigned_mess_id in CAMPUS_MESSES:
        default_mess_id = student.assigned_mess_id

    st.session_state.setdefault("selected_dining_hall", default_mess_id)

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
    if not base_url:
        st.error("READ_API_BASE_URL is not configured. Set it to the hosted API URL.")
    else:
        try:
            live_cameras = cached_status(base_url)
        except (RuntimeError, DashboardAPIError) as exc:
            st.error(f"Live API unavailable: {exc}")

    if mode == "All messes comparison":
        render_all_messes_comparison(base_url, live_cameras)
        return

    # "Live cameras" Mode - With "Which Mess" selection
    st.markdown("---")
    select_col, _ = st.columns([2, 1], vertical_alignment="center")
    with select_col:
        selected_id = (
            st.segmented_control(
                "Select dining hall",
                options=[m.id for m in all_messes],
                format_func=lambda mid: get_mess(mid).name,
                key="selected_dining_hall",
            )
            or default_mess_id
        )

    active_mess = get_mess(selected_id)
    render_single_mess_view(active_mess, base_url, live_cameras)


if __name__ == "__main__":
    main()
