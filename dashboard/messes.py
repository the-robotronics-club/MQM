"""Campus dining halls and canteens configuration, meal timings, and congestion calculations."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal


@dataclass(frozen=True)
class DiningFacility:
    """Information for a campus mess or canteen."""

    id: str
    name: str
    short_name: str
    facility_type: Literal["mess", "canteen"]
    camera_id: str
    capacity: int
    location: str
    hostels_served: str
    specialty: str
    current_menu: dict[str, str]


CAMPUS_FACILITIES: dict[str, DiningFacility] = {
    # --- Hostel Messes ---
    "tulsi": DiningFacility(
        id="tulsi",
        name="Tulsi Mess",
        short_name="Tulsi",
        facility_type="mess",
        camera_id="mess_main",  # Primary calibrated camera feed
        capacity=160,
        location="South Campus (near Tulsi Hostel)",
        hostels_served="Tulsi, Mandakini & South Hostels",
        specialty="North & South Indian Thali, Paneer Butter Masala, Fresh Rotis",
        current_menu={
            "Breakfast": "Poha, Boiled Eggs / Banana, Milk & Tea",
            "Lunch": "Rajma Chawal, Aloo Gobi, Roti, Curd, Salad",
            "Evening Snacks": "Samosa & Masala Chai",
            "Dinner": "Egg Curry / Shahi Paneer, Dal Tadka, Jeera Rice, Gulab Jamun",
        },
    ),
    "pine": DiningFacility(
        id="pine",
        name="Pine Mess",
        short_name="Pine",
        facility_type="mess",
        camera_id="mess_pine",
        capacity=140,
        location="North Campus (near Pine Hostel)",
        hostels_served="Pine Hostel & North Quad",
        specialty="Tandoori Breads, Dal Makhani, Seasonal Sabzi",
        current_menu={
            "Breakfast": "Aloo Paratha, Curd, Pickle, Tea / Coffee",
            "Lunch": "Kadhi Pakora, Rice, Bhindi Masala, Phulka, Papad",
            "Evening Snacks": "Bread Pakoda with Green Chutney",
            "Dinner": "Chicken Curry / Matar Paneer, Dal Fry, Roti, Rice, Halwa",
        },
    ),
    "oak": DiningFacility(
        id="oak",
        name="Oak Mess",
        short_name="Oak",
        facility_type="mess",
        camera_id="mess_oak",
        capacity=130,
        location="North Campus (near Oak Hostel)",
        hostels_served="Oak & Chinar Hostels",
        specialty="South Indian Specials, Rasam, Medu Vada",
        current_menu={
            "Breakfast": "Masala Dosa, Sambar, Coconut Chutney, Coffee",
            "Lunch": "Chole Bhature / Chole Rice, Boondi Raita, Salad",
            "Evening Snacks": "Veg Cutlet & Tea",
            "Dinner": "Veg Pulao, Paneer Do Pyaza, Dal Palak, Chapati, Kheer",
        },
    ),
    "alder": DiningFacility(
        id="alder",
        name="Alder Mess",
        short_name="Alder",
        facility_type="mess",
        camera_id="mess_alder",
        capacity=120,
        location="North Campus (near Alder Hostel)",
        hostels_served="Alder Hostel",
        specialty="Homestyle Dal, Multigrain Rotis, Healthy Diet Counter",
        current_menu={
            "Breakfast": "Idli Vada Sambar, Sprouts, Milk & Tea",
            "Lunch": "Dal Tadka, Mix Veg, Steamed Rice, Phulka, Buttermilk",
            "Evening Snacks": "Bhel Puri & Lemon Tea",
            "Dinner": "Soya Chaap Masala, Yellow Dal, Phulka, Rice, Ice Cream",
        },
    ),
    "ganga": DiningFacility(
        id="ganga",
        name="Ganga Mess",
        short_name="Ganga",
        facility_type="mess",
        camera_id="mess_ganga",
        capacity=150,
        location="Campus Central (near Ganga Hostel)",
        hostels_served="Ganga & Yamuna Hostels",
        specialty="Biryani Special Nights, Rich Gravies, Salad Bar",
        current_menu={
            "Breakfast": "Uttapam with Sambar, Boiled Eggs, Tea",
            "Lunch": "Paneer Lababdar, Dal Makhani, Peas Pulao, Roti",
            "Evening Snacks": "Patties / Puff & Masala Chai",
            "Dinner": "Chicken Biryani / Veg Biryani, Mirchi Salan, Raita",
        },
    ),
    "saraswati": DiningFacility(
        id="saraswati",
        name="Saraswati Mess",
        short_name="Saraswati",
        facility_type="mess",
        camera_id="mess_saraswati",
        capacity=130,
        location="South Campus (near Saraswati Hostel)",
        hostels_served="Saraswati Hostel & Research Scholars",
        specialty="Balanced Diet, Fresh Phulkas, Curd Rice",
        current_menu={
            "Breakfast": "Methi Thepla, Curd, Pickle, Coffee",
            "Lunch": "Dum Aloo, Dal Fry, Jeera Rice, Chapati, Curd",
            "Evening Snacks": "Onion Pakoda & Filter Coffee",
            "Dinner": "Kadhai Paneer, Dal Tadka, Rice, Roti, Sewaiyan",
        },
    ),
    # --- Campus Canteens & Eateries ---
    "monal": DiningFacility(
        id="monal",
        name="Monal Canteen",
        short_name="Monal",
        facility_type="canteen",
        camera_id="canteen_monal",
        capacity=70,
        location="South Campus Student Hub",
        hostels_served="All Students, Faculty & Visitors",
        specialty="Quick Bites, Maggi, Momos, Parathas, Fresh Fruit Juices",
        current_menu={
            "All-Day": "Steamed / Fried Momos, Cheese Maggi, Egg Roll, Cold Coffee, Fresh Juices",
            "Breakfast": "Bun Maska, Omelette, Chai",
            "Lunch": "Mini Thali, Fried Rice with Manchurian",
            "Evening Snacks": "Chicken Roll, Paneer Roll, French Fries",
            "Dinner": "Noodles, Chilli Chicken, Biryani Plates",
        },
    ),
    "drongo": DiningFacility(
        id="drongo",
        name="Drongo Canteen",
        short_name="Drongo",
        facility_type="canteen",
        camera_id="canteen_drongo",
        capacity=60,
        location="North Campus Academic Complex",
        hostels_served="All Students & Research Quad",
        specialty="Grilled Sandwiches, Dosa Varieties, Filter Coffee, Shakes",
        current_menu={
            "All-Day": "Cheese Grilled Sandwich, Mysore Masala Dosa, Filter Coffee, Milkshakes",
            "Breakfast": "Butter Toast, Egg Bhurji, Tea",
            "Lunch": "Pasta (Red / White Sauce), Garlic Bread",
            "Evening Snacks": "Peri Peri Fries, Veg Burger, Iced Tea",
            "Dinner": "Paneer Tikka Roll, Chowmein",
        },
    ),
    "bulbul": DiningFacility(
        id="bulbul",
        name="Bulbul Canteen",
        short_name="Bulbul",
        facility_type="canteen",
        camera_id="canteen_bulbul",
        capacity=65,
        location="Sports Complex Ground Level",
        hostels_served="All Students & Sports Area",
        specialty="Protein Shakes, Fresh Fruit Bowls, Energy Drinks, Fast Snacks",
        current_menu={
            "All-Day": "Whey Protein Shakes, Fruit Bowls, Peanut Butter Sandwiches, Energy Bars",
            "Breakfast": "Oats Porridge, Boiled Eggs, Banana Shake",
            "Lunch": "Chicken Rice Bowl, Paneer Salad",
            "Evening Snacks": "Sprouts Chaat, Sweet Corn, Green Tea",
            "Dinner": "Egg Rice, Grilled Chicken / Paneer Plates",
        },
    ),
    "bake_o_mocha": DiningFacility(
        id="bake_o_mocha",
        name="Bake-O-Mocha",
        short_name="Bake-O-Mocha",
        facility_type="canteen",
        camera_id="canteen_bake_o_mocha",
        capacity=50,
        location="North Campus Promenade",
        hostels_served="All Students, Visitors & Campus Community",
        specialty="Artisanal Coffee, Pastries, Cheesecakes, Croissants, Waffles",
        current_menu={
            "All-Day": "Espresso, Cappuccino, Mocha, Blueberry Cheesecake, Choco-Lava Cake, Belgian Waffles",
            "Breakfast": "Butter Croissants, Muffins, Hot Chocolate",
            "Lunch": "Subway-style Subs, Paneer Panini",
            "Evening Snacks": "Brownie with Ice Cream, Frappes",
            "Dinner": "Thin Crust Pizza Slices, Pasta",
        },
    ),
}

# Backward compatibility alias
MessInfo = DiningFacility
CAMPUS_MESSES = CAMPUS_FACILITIES


def get_all_facilities() -> list[DiningFacility]:
    """Return all configured campus messes and canteens."""
    return list(CAMPUS_FACILITIES.values())


def get_all_messes() -> list[DiningFacility]:
    """Return only hostel messes."""
    return [f for f in CAMPUS_FACILITIES.values() if f.facility_type == "mess"]


def get_all_canteens() -> list[DiningFacility]:
    """Return only campus canteens."""
    return [f for f in CAMPUS_FACILITIES.values() if f.facility_type == "canteen"]


def get_facility(facility_id: str) -> DiningFacility:
    """Return dining facility by ID, defaulting to Tulsi if unknown."""
    return CAMPUS_FACILITIES.get(facility_id, CAMPUS_FACILITIES["tulsi"])


# Backward compatibility
get_mess = get_facility


def estimate_wait_minutes(queue_count: int, facility_type: str = "mess") -> int:
    """Estimate queue wait time in minutes (messes: ~1.2 min/person, canteens: ~2.0 min/order)."""
    if queue_count <= 0:
        return 0
    multiplier = 2.0 if facility_type == "canteen" else 1.2
    return max(1, round(queue_count * multiplier))


def get_current_meal_slot(now: datetime | None = None) -> tuple[str, str, bool]:
    """
    Determine the current meal period based on local time.
    Returns (slot_name, time_window_str, is_active).
    """
    dt = now or datetime.now()
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
    """Return markdown badge and color tag for crowd level."""
    clean = (level or "green").lower()
    if clean == "red":
        return ":red-badge[Heavy Queue]", "red"
    if clean in ("amber", "yellow", "orange"):
        return ":orange-badge[Moderate Crowd]", "orange"
    return ":green-badge[Low Crowd / Fast]", "green"


def fallback_metrics_for_mess(facility_id: str) -> dict[str, Any]:
    """Realistic simulated readings for facilities when camera feed is offline."""
    now_iso = datetime.now().isoformat()
    profiles: dict[str, dict[str, Any]] = {
        # Messes
        "tulsi": {
            "headcount": 19,
            "queue_count": 6,
            "seats_occupied": 13,
            "seats_total": 160,
            "seat_occupancy_pct": 8.1,
            "crowd_level": "green",
            "timestamp": now_iso,
        },
        "pine": {
            "headcount": 36,
            "queue_count": 14,
            "seats_occupied": 22,
            "seats_total": 140,
            "seat_occupancy_pct": 15.7,
            "crowd_level": "amber",
            "timestamp": now_iso,
        },
        "oak": {
            "headcount": 12,
            "queue_count": 3,
            "seats_occupied": 9,
            "seats_total": 130,
            "seat_occupancy_pct": 6.9,
            "crowd_level": "green",
            "timestamp": now_iso,
        },
        "alder": {
            "headcount": 28,
            "queue_count": 9,
            "seats_occupied": 19,
            "seats_total": 120,
            "seat_occupancy_pct": 15.8,
            "crowd_level": "green",
            "timestamp": now_iso,
        },
        "ganga": {
            "headcount": 48,
            "queue_count": 18,
            "seats_occupied": 30,
            "seats_total": 150,
            "seat_occupancy_pct": 20.0,
            "crowd_level": "amber",
            "timestamp": now_iso,
        },
        "saraswati": {
            "headcount": 15,
            "queue_count": 4,
            "seats_occupied": 11,
            "seats_total": 130,
            "seat_occupancy_pct": 8.5,
            "crowd_level": "green",
            "timestamp": now_iso,
        },
        # Canteens
        "monal": {
            "headcount": 22,
            "queue_count": 8,
            "seats_occupied": 14,
            "seats_total": 70,
            "seat_occupancy_pct": 20.0,
            "crowd_level": "amber",
            "timestamp": now_iso,
        },
        "drongo": {
            "headcount": 16,
            "queue_count": 5,
            "seats_occupied": 11,
            "seats_total": 60,
            "seat_occupancy_pct": 18.3,
            "crowd_level": "green",
            "timestamp": now_iso,
        },
        "bulbul": {
            "headcount": 10,
            "queue_count": 2,
            "seats_occupied": 8,
            "seats_total": 65,
            "seat_occupancy_pct": 12.3,
            "crowd_level": "green",
            "timestamp": now_iso,
        },
        "bake_o_mocha": {
            "headcount": 25,
            "queue_count": 7,
            "seats_occupied": 18,
            "seats_total": 50,
            "seat_occupancy_pct": 36.0,
            "crowd_level": "amber",
            "timestamp": now_iso,
        },
    }
    return profiles.get(facility_id, profiles["tulsi"])
