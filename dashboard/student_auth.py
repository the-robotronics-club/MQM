"""Student authentication and session management for the MQM dashboard."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import streamlit as st


@dataclass(frozen=True)
class StudentProfile:
    """Authenticated student details."""

    student_id: str
    name: str
    hostel: str
    room: str
    assigned_mess_id: str
    assigned_mess_name: str
    meal_plan: str
    dietary_pref: str


# Demo student registry for instant campus validation and testing
DEMO_STUDENTS: dict[str, dict[str, Any]] = {
    "2024CS101": {
        "pin": "1234",
        "name": "Aarav Sharma",
        "hostel": "Bhabha Hostel (H-2)",
        "room": "312",
        "assigned_mess_id": "north_mess",
        "assigned_mess_name": "North Dining Hall",
        "meal_plan": "Standard Full Plan",
        "dietary_pref": "Vegetarian",
    },
    "2024EE205": {
        "pin": "1234",
        "name": "Priya Patel",
        "hostel": "Sarojini Hostel (H-5)",
        "room": "204",
        "assigned_mess_id": "south_mess",
        "assigned_mess_name": "South Dining Hall",
        "meal_plan": "Standard Full Plan",
        "dietary_pref": "Non-Vegetarian",
    },
    "2024ME310": {
        "pin": "1234",
        "name": "Rohan Sen",
        "hostel": "Aryabhatta Hostel (H-8)",
        "room": "115",
        "assigned_mess_id": "central_mess",
        "assigned_mess_name": "Central Food Court",
        "meal_plan": "Flexible Meal Passes",
        "dietary_pref": "All Diets",
    },
}


def init_auth_state() -> None:
    """Ensure authentication keys exist in session state."""
    st.session_state.setdefault("is_authenticated", False)
    st.session_state.setdefault("student_profile", None)
    st.session_state.setdefault("active_mess_override", None)


def get_current_student() -> StudentProfile | None:
    """Return the active student profile if logged in."""
    init_auth_state()
    raw = st.session_state.get("student_profile")
    if isinstance(raw, StudentProfile):
        return raw
    return None


def is_authenticated() -> bool:
    """Return True if a student is actively authenticated."""
    init_auth_state()
    return bool(st.session_state.get("is_authenticated"))


def login_student(student_id: str, pin: str) -> tuple[bool, str]:
    """Authenticate student ID and PIN."""
    init_auth_state()
    clean_id = student_id.strip().upper()
    clean_pin = pin.strip()

    if not clean_id:
        return False, "Please enter your student roll number."
    if not clean_pin:
        return False, "Please enter your 4-digit security PIN."

    account = DEMO_STUDENTS.get(clean_id)
    if account is not None:
        if clean_pin == account["pin"]:
            profile = StudentProfile(
                student_id=clean_id,
                name=account["name"],
                hostel=account["hostel"],
                room=account["room"],
                assigned_mess_id=account["assigned_mess_id"],
                assigned_mess_name=account["assigned_mess_name"],
                meal_plan=account["meal_plan"],
                dietary_pref=account["dietary_pref"],
            )
            st.session_state["is_authenticated"] = True
            st.session_state["student_profile"] = profile
            st.session_state["active_mess_override"] = profile.assigned_mess_id
            st.session_state["selected_dining_hall"] = profile.assigned_mess_id
            return True, f"Welcome back, {profile.name}!"
        return False, "Incorrect PIN. (For demo accounts, use 1234)"

    return False, "Unknown demo profile. Select one of the sample accounts below."


def quick_login(student_id: str) -> None:
    """One-click login for testing demo profiles."""
    account = DEMO_STUDENTS.get(student_id)
    if account:
        login_student(student_id, account["pin"])


def logout_student() -> None:
    """Terminate the current student session."""
    st.session_state["is_authenticated"] = False
    st.session_state["student_profile"] = None
    st.session_state["active_mess_override"] = None


def render_login_view() -> None:
    """Render a modern student login portal."""
    with st.container(horizontal_alignment="center"):
        st.markdown("### :material/account_circle: Demo student profiles")
        st.caption("Sample accounts only. This demo does not verify institute identity.")

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        with st.container(border=True):
            with st.form("student_login_form", border=False):
                roll_input = st.text_input(
                    "Roll number / Student ID",
                    placeholder="e.g. 2024CS101",
                    help="Use one of the demo profiles below.",
                )
                pin_input = st.text_input(
                    "Demo PIN (1234)",
                    type="password",
                    placeholder="••••",
                    max_chars=4,
                )
                submitted = st.form_submit_button(
                    "Open demo profile",
                    type="primary",
                    icon=":material/login:",
                )
                if submitted:
                    success, message = login_student(roll_input, pin_input)
                    if success:
                        st.toast(message, icon="✅")
                        st.rerun()
                    else:
                        st.error(message)

        st.caption("Quick test accounts (click to test immediately):")
        demo_cols = st.columns(len(DEMO_STUDENTS))
        for idx, (demo_id, data) in enumerate(DEMO_STUDENTS.items()):
            with demo_cols[idx]:
                if st.button(
                    f"{data['name'].split()[0]}\n({demo_id})",
                    key=f"demo_btn_{demo_id}",
                    help=f"Login as {data['name']} ({data['assigned_mess_name']})",
                ):
                    quick_login(demo_id)
                    st.rerun()


def render_student_badge() -> None:
    """Render the authenticated student pill and controls in the header/sidebar."""
    student = get_current_student()
    if not student:
        return

    with st.container(border=True):
        left, right = st.columns([4, 1], vertical_alignment="center")
        with left:
            st.markdown(
                f":material/badge: **{student.name}** (`{student.student_id}`) · "
                f":material/home: {student.hostel} · "
                f":material/restaurant: Assigned: **{student.assigned_mess_name}**"
            )
        with right:
            if st.button("Log out", icon=":material/logout:", key="student_logout_btn"):
                logout_student()
                st.rerun()
