"""Student and campus authentication with IIT Mandi email support and session state."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import streamlit as st

from dashboard.messes import get_all_messes, get_facility


@dataclass(frozen=True)
class StudentProfile:
    """Authenticated student or campus user details."""

    email: str
    student_id: str
    name: str
    login_type: str  # "Student", "Faculty/Staff", "Guest"
    hostel: str
    room: str
    assigned_mess_id: str
    assigned_mess_name: str
    meal_plan: str
    dietary_pref: str


# Demo accounts mapping to real IIT Mandi student emails and hostel messes
DEMO_STUDENTS: dict[str, dict[str, Any]] = {
    "b25201@students.iitmandi.ac.in": {
        "pin": "1234",
        "roll": "B25201",
        "name": "Aarav Sharma",
        "login_type": "Student",
        "hostel": "Pine Hostel",
        "room": "P-204",
        "assigned_mess_id": "pine",
        "assigned_mess_name": "Pine Mess",
        "meal_plan": "Regular 4-Meal Plan",
        "dietary_pref": "Vegetarian",
    },
    "b25104@students.iitmandi.ac.in": {
        "pin": "1234",
        "roll": "B25104",
        "name": "Priya Patel",
        "login_type": "Student",
        "hostel": "Tulsi Hostel",
        "room": "T-312",
        "assigned_mess_id": "tulsi",
        "assigned_mess_name": "Tulsi Mess",
        "meal_plan": "Regular 4-Meal Plan",
        "dietary_pref": "Non-Vegetarian",
    },
    "b24089@students.iitmandi.ac.in": {
        "pin": "1234",
        "roll": "B24089",
        "name": "Rohan Sen",
        "login_type": "Student",
        "hostel": "Oak Hostel",
        "room": "O-115",
        "assigned_mess_id": "oak",
        "assigned_mess_name": "Oak Mess",
        "meal_plan": "Regular 4-Meal Plan",
        "dietary_pref": "Vegetarian",
    },
    "b24150@students.iitmandi.ac.in": {
        "pin": "1234",
        "roll": "B24150",
        "name": "Devansh Roy",
        "login_type": "Student",
        "hostel": "Alder Hostel",
        "room": "A-302",
        "assigned_mess_id": "alder",
        "assigned_mess_name": "Alder Mess",
        "meal_plan": "Regular 4-Meal Plan",
        "dietary_pref": "Vegetarian",
    },
    "b23210@students.iitmandi.ac.in": {
        "pin": "1234",
        "roll": "B23210",
        "name": "Kabir Mehta",
        "login_type": "Student",
        "hostel": "Ganga Hostel",
        "room": "G-408",
        "assigned_mess_id": "ganga",
        "assigned_mess_name": "Ganga Mess",
        "meal_plan": "Regular 4-Meal Plan",
        "dietary_pref": "Non-Vegetarian",
    },
    "b25312@students.iitmandi.ac.in": {
        "pin": "1234",
        "roll": "B25312",
        "name": "Ananya Verma",
        "login_type": "Student",
        "hostel": "Saraswati Hostel",
        "room": "S-102",
        "assigned_mess_id": "saraswati",
        "assigned_mess_name": "Saraswati Mess",
        "meal_plan": "Dietary Choice Plan",
        "dietary_pref": "Jain / Pure Veg",
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
    """Return True if a student or user is actively authenticated."""
    init_auth_state()
    return bool(st.session_state.get("is_authenticated"))


def extract_roll_from_email(email: str) -> str:
    """Extract roll number from college email (e.g. b25201@students.iitmandi.ac.in -> B25201)."""
    user_part = email.strip().split("@")[0]
    return user_part.upper()


def login_with_email(
    email: str,
    pin: str,
    selected_mess_id: str | None = None,
) -> tuple[bool, str]:
    """Authenticate via college email."""
    init_auth_state()
    clean_email = email.strip().lower()
    clean_pin = pin.strip()

    if not clean_email:
        return False, "Please enter your college email address."
    if not clean_pin:
        return False, "Please enter your 4-digit security PIN."

    # Validate college email domain or format
    if "@" not in clean_email:
        clean_email = f"{clean_email}@students.iitmandi.ac.in"

    # Match predefined test profiles
    account = DEMO_STUDENTS.get(clean_email)
    if account is not None:
        if clean_pin == account["pin"]:
            profile = StudentProfile(
                email=clean_email,
                student_id=account["roll"],
                name=account["name"],
                login_type=account["login_type"],
                hostel=account["hostel"],
                room=account["room"],
                assigned_mess_id=account["assigned_mess_id"],
                assigned_mess_name=account["assigned_mess_name"],
                meal_plan=account["meal_plan"],
                dietary_pref=account["dietary_pref"],
            )
            st.session_state["is_authenticated"] = True
            st.session_state["student_profile"] = profile
            st.session_state["selected_dining_hall"] = profile.assigned_mess_id
            return True, f"Welcome back, {profile.name} ({profile.student_id})!"
        return False, "Incorrect PIN. (Demo accounts use PIN: 1234)"

    # Fallback registration for any institute email (e.g. any b25xxx@students.iitmandi.ac.in)
    roll = extract_roll_from_email(clean_email)
    assigned_mess = selected_mess_id or "pine"
    mess_info = get_facility(assigned_mess)

    profile = StudentProfile(
        email=clean_email,
        student_id=roll,
        name=f"Student {roll}",
        login_type="Student",
        hostel=f"{mess_info.short_name} Hostel",
        room="Res-Room",
        assigned_mess_id=assigned_mess,
        assigned_mess_name=mess_info.name,
        meal_plan="Regular 4-Meal Plan",
        dietary_pref="Vegetarian",
    )
    st.session_state["is_authenticated"] = True
    st.session_state["student_profile"] = profile
    st.session_state["selected_dining_hall"] = profile.assigned_mess_id
    return True, f"Logged in as {clean_email} (Assigned: {mess_info.name})"


def login_as_guest(name: str, chosen_mess_id: str) -> tuple[bool, str]:
    """Instant guest login without credentials."""
    init_auth_state()
    clean_name = name.strip() or "Campus Visitor"
    mess_info = get_facility(chosen_mess_id)

    profile = StudentProfile(
        email="guest@iitmandi.ac.in",
        student_id="GUEST",
        name=clean_name,
        login_type="Guest",
        hostel="Visitor / Day Scholar",
        room="N/A",
        assigned_mess_id=chosen_mess_id,
        assigned_mess_name=mess_info.name,
        meal_plan="Pay Per Meal / Canteen Pass",
        dietary_pref="All Diets",
    )
    st.session_state["is_authenticated"] = True
    st.session_state["student_profile"] = profile
    st.session_state["selected_dining_hall"] = chosen_mess_id
    return True, f"Welcome, {clean_name}!"


def quick_login(email: str) -> None:
    """One-click login helper for test profiles."""
    account = DEMO_STUDENTS.get(email)
    if account:
        login_with_email(email, account["pin"])


def logout_student() -> None:
    """Terminate the active session."""
    st.session_state["is_authenticated"] = False
    st.session_state["student_profile"] = None


def render_login_view() -> None:
    """Render modern login portal with college email input and login type selector."""
    with st.container(horizontal_alignment="center"):
        st.markdown("### :material/account_circle: Campus dining login")
        st.caption("Sign in with your institute email or roll number to personalize your mess queue.")

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        with st.container(border=True):
            login_type = st.segmented_control(
                "Login type",
                options=["College email", "Roll number", "Guest / Explorer"],
                default="College email",
                key="login_type_choice",
            ) or "College email"

            messes = get_all_messes()

            if login_type == "College email":
                with st.form("college_email_login_form", border=False):
                    email_input = st.text_input(
                        "College email",
                        placeholder="e.g. b25201@students.iitmandi.ac.in",
                        help="Enter your institute email (e.g. b25201@students.iitmandi.ac.in)",
                    )
                    pin_input = st.text_input(
                        "Security PIN (4 digits)",
                        type="password",
                        placeholder="••••",
                        max_chars=4,
                        help="Default demo PIN is 1234",
                    )
                    assigned_mess_select = st.selectbox(
                        "Your assigned hostel mess",
                        options=[m.id for m in messes],
                        format_func=lambda mid: get_facility(mid).name,
                        help="If this is your first sign-in, select your allocated hostel mess.",
                    )
                    submitted = st.form_submit_button(
                        "Sign in with college email",
                        type="primary",
                        icon=":material/login:",
                    )
                    if submitted:
                        success, message = login_with_email(
                            email_input, pin_input, assigned_mess_select
                        )
                        if success:
                            st.toast(message, icon="✅")
                            st.rerun()
                        else:
                            st.error(message)

            elif login_type == "Roll number":
                with st.form("roll_login_form", border=False):
                    roll_input = st.text_input(
                        "Student roll number",
                        placeholder="e.g. B25201",
                        help="Enter your roll number (e.g. B25201, B25104)",
                    )
                    pin_input = st.text_input(
                        "Security PIN (4 digits)",
                        type="password",
                        placeholder="••••",
                        max_chars=4,
                    )
                    assigned_mess_select = st.selectbox(
                        "Your assigned hostel mess",
                        options=[m.id for m in messes],
                        format_func=lambda mid: get_facility(mid).name,
                    )
                    submitted = st.form_submit_button(
                        "Sign in with roll number",
                        type="primary",
                        icon=":material/badge:",
                    )
                    if submitted:
                        email_equiv = (
                            f"{roll_input.strip().lower()}@students.iitmandi.ac.in"
                            if roll_input
                            else ""
                        )
                        success, message = login_with_email(
                            email_equiv, pin_input, assigned_mess_select
                        )
                        if success:
                            st.toast(message, icon="✅")
                            st.rerun()
                        else:
                            st.error(message)

            else:  # Guest / Explorer
                with st.form("guest_login_form", border=False):
                    guest_name = st.text_input("Your name", placeholder="e.g. Rahul Sharma")
                    guest_mess = st.selectbox(
                        "Preferred dining hall or canteen",
                        options=[m.id for m in messes],
                        format_func=lambda mid: get_facility(mid).name,
                    )
                    submitted = st.form_submit_button(
                        "Explore as guest",
                        type="primary",
                        icon=":material/person:",
                    )
                    if submitted:
                        success, message = login_as_guest(guest_name, guest_mess)
                        st.toast(message, icon="✅")
                        st.rerun()

        # One-click test student profiles
        st.caption("Quick test accounts (IIT Mandi emails):")
        demo_cols = st.columns(3)
        for idx, (demo_email, data) in enumerate(list(DEMO_STUDENTS.items())[:6]):
            with demo_cols[idx % 3]:
                roll = data["roll"]
                mess_short = data["assigned_mess_name"].split()[0]
                if st.button(
                    f"{roll} ({mess_short})\n{data['name'].split()[0]}",
                    key=f"demo_btn_{roll}",
                    help=f"{demo_email} · {data['name']} · {data['assigned_mess_name']}",
                ):
                    quick_login(demo_email)
                    st.rerun()


def render_student_badge() -> None:
    """Render the active student/user profile card."""
    student = get_current_student()
    if not student:
        return

    with st.container(border=True):
        left, right = st.columns([4, 1], vertical_alignment="center")
        with left:
            if student.login_type == "Guest":
                st.markdown(
                    f":material/person: **{student.name}** (Guest / Visitor) · "
                    f":material/restaurant: Selected: **{student.assigned_mess_name}**"
                )
            else:
                st.markdown(
                    f":material/school: **{student.name}** (`{student.email}`) · "
                    f":material/badge: Roll: `{student.student_id}` · "
                    f":material/home: {student.hostel} · "
                    f":material/restaurant: Assigned: **{student.assigned_mess_name}**"
                )
        with right:
            if st.button("Log out", icon=":material/logout:", key="user_logout_btn"):
                logout_student()
                st.rerun()
