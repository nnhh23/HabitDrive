"""HabitDrive dataset generator — constants, pools, and configuration.

All decisions not specified by the paper are documented here and in
GENERATION_NOTES.md. The paper (HabitDrive) is authoritative for:
  - 100 households, 5,000 base history events,
  - 2,000 route-planning tasks and 1,000 in-cabin tasks,
  - member attributes (id, role, occupation, age group, driving eligibility),
  - six evolution types (no_change, content_update, condition_update,
    unconditional_to_conditional, conditional_to_unconditional, cancellation),
  - four challenge labels (user_binding, preference_update,
    conditional_application, temporary_request_interference),
  - route task stages (destination -> route -> parking when required),
  - five in-cabin device categories (climate, seats, media, windows,
    ambient lighting).
"""
from __future__ import annotations

MASTER_SEED = 42
NUM_HOUSEHOLDS = 100
BASE_EVENTS_PER_HOUSEHOLD = 50        # -> 5,000 base events
ROUTE_TASKS_PER_HOUSEHOLD = 20        # -> 2,000 route tasks
CABIN_TASKS_PER_HOUSEHOLD = 10        # -> 1,000 cabin tasks

# ---- Member composition (paper: "diverse households"; weights are ours) ----
COMPOSITION_WEIGHTS = [
    ("two_adults_one_child", 0.40),
    ("two_adults", 0.30),
    ("one_adult_one_child", 0.20),
    ("two_adults_two_children", 0.10),
]

ADULT_NAMES = [
    "Michael", "David", "James", "Daniel", "William", "Thomas", "Robert",
    "Andrew", "Matthew", "Joseph", "Charles", "Benjamin", "Christopher",
    "Anthony", "Jonathan", "Nicholas", "Samuel", "Alexander", "Ryan", "Kevin",
    "Emily", "Sarah", "Jessica", "Sophia", "Olivia", "Emma", "Grace", "Laura",
    "Hannah", "Amelia", "Charlotte", "Chloe", "Victoria", "Natalie",
    "Rebecca", "Rachel", "Julia", "Anna", "Megan", "Claire",
]
CHILD_NAMES = [
    "Ethan", "Noah", "Liam", "Lucas", "Leo", "Mason", "Logan", "Jacob",
    "Dylan", "Henry", "Owen", "Jack", "Luke", "Caleb", "Isaac", "Mia", "Lily",
    "Ella", "Lucy", "Zoe", "Ava", "Ruby", "Alice", "Sophie", "Maya", "Eva",
    "Nora", "Leah", "Ivy", "Clara",
]

ADULT_OCCUPATIONS = [
    "software_engineer", "teacher", "doctor", "researcher", "accountant",
    "salesperson", "designer", "civil_servant", "lawyer", "nurse",
    "shop_owner", "freelancer", "journalist", "architect",
    "financial_analyst", "university_lecturer", "human_resources_specialist",
    "marketing_specialist", "retired",
]
CHILD_OCCUPATIONS = [
    "primary_school_student", "middle_school_student", "high_school_student",
]

DRIVING_PROB_DEFAULT_ADULT = 0.85
DRIVING_PROB_RETIRED = 0.65

# ---- Time horizon ----
TIME = {
    "creation_start": "2025-01-01T00:00:00",
    "creation_end": "2025-06-30T23:59:59",
    "evolution_end": "2025-11-30T23:59:59",
    "history_end": "2025-12-20T23:59:59",
    "minimum_gap_days": 10,          # between successive evolution nodes
}

# ---- Locations ----
RESIDENTIAL_PREFIXES = ["Maple", "Riverside", "Greenfield", "Lakeside",
                        "Oakwood", "Sunnyvale", "Hillcrest", "Westwood",
                        "Parkview", "Brookside"]
RESIDENTIAL_SUFFIXES = ["Gardens", "Residence", "Court", "Heights",
                        "Community", "Apartments", "Estate", "Manor"]
COMPANY_PREFIXES = ["Horizon", "Summit", "Riverside", "Central", "Evergreen",
                    "Northstar", "Bluewave", "Metro", "Innovation", "Pioneer"]
COMPANY_SUFFIXES = ["Technology Park", "Innovation Center", "Business Center",
                    "Software Park", "Research Campus", "Financial Center",
                    "Creative Hub", "Industrial Park"]
BUILDING_NAMES = ["Building A", "Building B", "Building C", "Tower 1",
                  "Tower 2", "Tower 3"]
SCHOOL_PREFIXES = ["Riverside", "Greenfield", "Oakwood", "Hillcrest",
                   "Westlake", "Sunrise", "Central", "Maple Grove",
                   "Northview", "Lakeside"]
SCHOOL_SUFFIXES = {
    "primary_school_student": ["Primary School", "Elementary School"],
    "middle_school_student": ["Middle School", "Junior High School"],
    "high_school_student": ["High School", "Senior High School"],
}
FREQUENT_POI_TYPES = {
    "supermarket": ["FreshMart", "Greenway Supermarket", "Daily Choice Market",
                    "Riverside Grocery", "Family Foods"],
    "hospital": ["Central General Hospital", "Riverside Medical Center",
                 "Greenfield Community Hospital", "Westlake People's Hospital",
                 "Northview Medical Center"],
    "gym": ["FitZone Gym", "Urban Fitness Club", "ActiveLife Gym",
            "PowerHouse Fitness", "Motion Fitness Center"],
    "shopping_mall": ["Riverside Mall", "Central Plaza",
                      "Westlake Shopping Center", "Horizon Mall",
                      "Greenfield Town Center"],
    "restaurant": ["Harbor Kitchen", "Maple Garden Restaurant",
                   "Riverside Bistro", "Oakwood Grill", "Green Table"],
    "train_station": ["Central Railway Station", "South Railway Station",
                      "West Railway Station", "North Railway Station"],
    "airport": ["Central International Airport", "Westfield International Airport"],
    "family_home": ["Parents' Home", "Grandparents' Home", "In-Laws' Home"],
}

# ---- Route preferences ----
ROUTE_PREF_OPTIONS = {
    "highway": ["avoid", "prefer"],
    "toll_road": ["avoid", "prefer"],
    "congestion": ["avoid"],
    "traffic_lights": ["minimize"],
    "lane_changes": ["minimize"],
    "smoothness": ["prefer"],
    "scenery": ["prefer"],
    "travel_time": ["minimize"],
    "distance": ["minimize"],
}
ROUTE_CONDITION_DISTRIBUTION = {
    "unconditional": 0.40,
    "destination": 0.20,
    "weather": 0.15,
    "time_period": 0.10,
    "trip_context": 0.15,
}
CONDITION_VALUES = {
    "weather": ["rainy", "snowy", "foggy"],
    "time_period": ["morning", "evening", "night"],
    "trip_context": ["commute", "daily_errand", "long_distance_trip"],
}
# Allowed (attribute, value) under each non-destination condition.
CONDITIONAL_RULES = {
    "weather": {
        "rainy": [("smoothness", "prefer"), ("highway", "avoid"),
                  ("lane_changes", "minimize"), ("congestion", "avoid")],
        "foggy": [("highway", "avoid"), ("lane_changes", "minimize"),
                  ("smoothness", "prefer")],
        "snowy": [("highway", "avoid"), ("smoothness", "prefer"),
                  ("lane_changes", "minimize")],
    },
    "time_period": {
        "morning": [("congestion", "avoid"), ("travel_time", "minimize"),
                    ("traffic_lights", "minimize")],
        "evening": [("congestion", "avoid"), ("travel_time", "minimize"),
                    ("traffic_lights", "minimize")],
        "night": [("highway", "prefer"), ("traffic_lights", "minimize"),
                  ("travel_time", "minimize")],
    },
    "trip_context": {
        "commute": [("travel_time", "minimize"), ("highway", "prefer"),
                    ("toll_road", "prefer")],
        "daily_errand": [("travel_time", "minimize"), ("highway", "prefer"),
                         ("toll_road", "prefer")],
        "long_distance_trip": [("highway", "prefer"), ("smoothness", "prefer"),
                               ("traffic_lights", "minimize"),
                               ("toll_road", "prefer")],
    },
}
CONDITIONAL_DESTINATION_ROLES = ["home", "workplace", "school", "frequent_poi"]

# ---- Parking preferences ----
PARKING_TYPE_WEIGHTS = [
    ("nearest", 0.22), ("low_cost", 0.20), ("indoor", 0.20),
    ("easy_to_park", 0.23), ("preferred_parking_lot", 0.15),
]
PARKING_CONDITION_DISTRIBUTION = {"unconditional": 0.40, "destination": 0.60}
PARKING_CONDITIONAL_ROLES = ["workplace", "school", "frequent_poi"]
PARKING_LOT_PREFIXES = ["Central", "Riverside", "North", "South", "East",
                        "West", "Main", "Visitor", "Metro", "Garden"]
PARKING_LOT_SUFFIXES = ["Parking Garage", "Underground Parking", "Parking Lot",
                        "Car Park", "Parking Center"]
PARKING_LOT_SECTIONS = ["Area A", "Area B", "Area C", "Level 1", "Level 2",
                        "Level B1", "Level B2"]

# ---- Passenger constraints ----
CONSTRAINT_ASSIGN_PROB = {"adult": 0.08, "child": 0.12, "retired": 0.15}
CONSTRAINT_TYPE_WEIGHTS = [("motion_sickness", 0.40),
                           ("limited_mobility", 0.25),
                           ("need_regular_breaks", 0.35)]
CONSTRAINT_EVAL_MAP = {
    "motion_sickness": {"smoothness": "prefer", "lane_changes": "minimize"},
    "limited_mobility": {"walking_distance": "minimize",
                         "accessible_pickup": "require"},
    "need_regular_breaks": {"rest_stop_available": "require"},
}
# Route-relevant vs parking-relevant effects of passenger constraints.
ROUTE_CONSTRAINT_EVAL_MAP = {
    "motion_sickness": {"smoothness": "prefer", "lane_changes": "minimize"},
    "need_regular_breaks": {"rest_stop_available": "require"},
}
PARKING_CONSTRAINT_EVAL_MAP = {
    "limited_mobility": {"walking_distance": "minimize",
                         "accessible_pickup": "require"},
}

# ---- In-cabin categories ----
CABIN_CATEGORIES = {
    "climate": {
        "temperature_c": [18, 20, 22, 24, 26, 28],
        "fan_level": ["low", "medium", "high", "auto"],
        "airflow_mode": ["face", "feet", "windshield", "auto"],
    },
    "seat": {
        "seat_heating": ["off", "low", "medium", "high"],
        "seat_ventilation": ["off", "low", "medium", "high"],
        "seat_position": ["upright", "standard", "relaxed"],
    },
    "media": {
        "content_type": ["music", "podcast", "news", "audiobook", "silence"],
        "music_genre": ["pop", "classical", "jazz", "rock"],
        "volume": ["mute", "low", "medium", "high"],
    },
    "window": {
        "side_window": ["closed", "slightly_open", "half_open"],
        "sunroof": ["closed", "tilted", "open"],
    },
    "ambient_light": {
        "light_state": ["off", "on"],
        "brightness": ["low", "medium", "high"],
        "color": ["warm_white", "cool_white", "blue"],
    },
}
CABIN_ZONES = {
    "climate": ["driver", "front_passenger", "rear", "whole_cabin"],
    "seat": ["driver", "front_passenger", "rear"],
    "media": ["front", "rear", "whole_cabin"],
    "window": ["driver", "front_passenger", "rear", "whole_cabin"],
    "ambient_light": ["front", "rear", "whole_cabin"],
}
CABIN_CONDITION_TYPES = ["weather", "outside_temperature_band", "time_period",
                         "trip_context", "passenger_id"]
CABIN_CONDITION_VALUES = {
    "weather": ["clear", "rainy", "snowy", "foggy"],
    "outside_temperature_band": ["cold", "mild", "hot"],
    "time_period": ["morning", "daytime", "evening", "night"],
    "trip_context": ["commute", "daily_errand", "family_trip",
                     "long_distance_trip"],
}

# ---- History expansion (our decisions; paper only fixes base events) ----
APPLICATION_COUNT = {"min": 20, "max": 30}     # recurring-use events / household
UNRELATED_COUNT = {"min": 5, "max": 8}         # time/traffic/casual exchanges
TEMPORARY_COUNT = {"min": 1, "max": 2}         # one-trip-only instructions

# One-trip-only instruction content (structured, so it can drive the
# temporary-request-interference challenge label).
TEMPORARY_ROUTE_OPTIONS = [
    {"attribute": "highway", "preference": "prefer"},
    {"attribute": "toll_road", "preference": "prefer"},
    {"attribute": "scenery", "preference": "prefer"},
]
TEMPORARY_CABIN_OPTIONS = [
    {"control_type": "climate", "target_zone": "whole_cabin",
     "attribute": "temperature_c", "value": 26},
    {"control_type": "media", "target_zone": "whole_cabin",
     "attribute": "volume", "value": "high"},
]

# ---- Route / parking candidate attributes ----
ROUTE_BASELINE = {
    "uses_highway": False, "uses_toll_road": False, "congestion_level": "low",
    "traffic_lights": 5, "lane_changes": 3, "smoothness": "high",
    "scenery": "medium", "travel_time_minutes": 30, "distance_km": 18.0,
    "rest_stop_available": True,
}
ROUTE_RANGES = {
    "traffic_lights": (3, 18), "lane_changes": (1, 12),
    "travel_time_minutes": (20, 65), "distance_km": (8.0, 45.0),
}
ROUTE_CATEGORICAL = {
    "congestion_level": ["low", "medium", "high"],
    "smoothness": ["low", "medium", "high"],
    "scenery": ["low", "medium", "high"],
}
PARKING_BASELINE = {
    "walking_distance_m": 120, "cost_level": "low", "indoor": True,
    "easy_to_park": True, "accessible_pickup": True,
}
PARKING_RANGES = {"walking_distance_m": (50, 800)}
PARKING_CATEGORICAL = {"cost_level": ["low", "medium", "high"]}

# ---- Context pools for tasks ----
TASK_WEATHER = ["clear", "rainy", "snowy", "foggy"]
TASK_TIME_PERIOD = ["morning", "afternoon", "evening", "night"]
TASK_TRIP_CONTEXT = ["commute", "daily_errand", "family_trip",
                     "long_distance_trip"]
# Parking is required for ~40% of route tasks (our decision).
PARKING_REQUIRED_PROB = 0.40

# Cabin seat assignment for occupants.
SEATS = ["driver", "front_passenger", "rear_left", "rear_right", "rear_middle"]

# Cabin constraints: heating and ventilation cannot be active simultaneously.
CABIN_MUTUAL_EXCLUSION = [("seat_heating", "seat_ventilation")]
