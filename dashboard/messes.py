"""Campus dining halls configuration, meal timings, and congestion calculations."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from zoneinfo import ZoneInfo


@dataclass(frozen=True)
class MessInfo:
    """Information for a campus dining facility."""

    id: str
    name: str
    short_name: str
    camera_id: str
    capacity: int
    location: str
    hostels_served: str
    specialty: str
    current_menu: dict[str, str]


CAMPUS_MESSES: dict[str, MessInfo] = {
    "north_mess": MessInfo(
        id="north_mess",
        name="North Dining Hall",
        short_name="North Mess",
        camera_id="mess_main",  # maps to real inference camera
        capacity=140,
        location="North Campus, near Bhabha & Raman Hostels",
        hostels_served="Hostels 1, 2, 3",
        specialty="North Indian Thali, Paneer Butter Masala, Fresh Rotis",
        current_menu={
            "Breakfast": "Idli Sambar, Poha, Boiled Eggs / Banana, Tea / Coffee",
            "Lunch": "Paneer Makhani, Dal Tadka, Seasonal Veg, Jeera Rice, Phulka, Curd",
            "Evening Snacks": "Veg Puff, Samosa, Masala Chai",
            "Dinner": "Egg Curry / Malai Kofta, Dal Makhani, Pulao, Gulab Jamun",
        },
    ),
    "south_mess": MessInfo(
        id="south_mess",
        name="South Dining Hall",
        short_name="South Mess",
        camera_id="mess_south",
        capacity=110,
        location="South Campus, adjacent to Sarojini Hostel",
        hostels_served="Hostels 4, 5, 6",
        specialty="South Indian Delicacies, Healthy Salad & Millet Bowls",
        current_menu={
            "Breakfast": "Masala Dosa, Medu Vada, Upma, Filter Coffee",
            "Lunch": "Sambar, Rasam, Avial, Lemon Rice, Chapati, Curd",
            "Evening Snacks": "Bhel Puri, Onion Pakoda, Filter Coffee",
            "Dinner": "Chicken Biryani / Veg Biryani, Mirchi Ka Salan, Raita, Kheer",
        },
    ),
    "central_mess": MessInfo(
        id="central_mess",
        name="Central Food Court",
        short_name="Central Court",
        camera_id="mess_central",
        capacity=200,
        location="Student Activity Center (SAC) Ground Floor",
        hostels_served="Hostels 7, 8, 9 & Day Scholars",
        specialty="Multi-cuisine Express Counter, Quick Turnover",
        current_menu={
            "Breakfast": "Aloo Paratha with Curd, Bread Omelette, Fresh Juice",
            "Lunch": "Rajma Chawal, Kadhi Pakora, Mixed Veg, Tandoori Roti",
            "Evening Snacks": "Grilled Sandwiches, Cold Coffee, French Fries",
            "Dinner": "Hakka Noodles / Fried Rice, Chilli Paneer / Manchurian, Ice Cream",
        },
    ),
}


def get_all_messes() -> list[MessInfo]:
    """Return all configured campus messes."""
    return list(CAMPUS_MESSES.values())


def get_mess(mess_id: str) -> MessInfo:
    """Return mess info by ID, defaulting to North Mess if unknown."""
    return CAMPUS_MESSES.get(mess_id, CAMPUS_MESSES["north_mess"])


def estimate_wait_minutes(queue_count: int) -> int:
    """Estimate queue wait time in minutes (approx ~1.2 to 1.5 min per person)."""
    if queue_count <= 0:
        return 0
    return max(1, round(queue_count * 1.3))


def get_current_meal_slot(now: datetime | None = None) -> tuple[str, str, bool]:
    """
    Determine the current meal period based on local time.
    Returns (slot_name, time_window_str, is_active).
    """
    dt = now or datetime.now(ZoneInfo("Asia/Kolkata"))
    if dt.tzinfo is not None:
        dt = dt.astimezone(ZoneInfo("Asia/Kolkata"))
    hour = dt.hour + (dt.minute / 60.0)

    if 7.5 <= hour <= 9.75:
        return "Breakfast", "7:30 AM – 9:45 AM", True
    if 12.25 <= hour <= 14.75:
        return "Lunch", "12:15 PM – 2:45 PM", True
    if 17.0 <= hour <= 18.75:
        return "Evening Snacks", "5:00 PM – 6:45 PM", True
    if 19.5 <= hour <= 21.75:
        return "Dinner", "7:30 PM – 9:45 PM", True

    if hour < 7.5:
        return "Breakfast", "Next: 7:30 AM", False
    if hour < 12.25:
        return "Lunch", "Next: 12:15 PM", False
    if hour < 17.0:
        return "Evening Snacks", "Next: 5:00 PM", False
    if hour < 19.5:
        return "Dinner", "Next: 7:30 PM", False
    return "Dining Closed", "Opens at 7:30 AM tomorrow", False


def crowd_badge_for_level(level: str) -> tuple[str, str]:
    """Return markdown badge and human-friendly label for crowd level."""
    clean = (level or "green").lower()
    if clean == "red":
        return ":red-badge[Heavy Queue]", "red"
    if clean in ("amber", "yellow", "orange"):
        return ":orange-badge[Moderate Crowd]", "orange"
    if clean == "green":
        return ":green-badge[Low crowd]", "green"
    return ":gray-badge[Unknown]", "gray"
