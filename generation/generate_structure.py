"""Deterministic structured generation for HabitDrive.

Produces households, members, structured preferences with evolution
trajectories, chronological interaction history (template dialogue that is
later LLM-verbalized), and the two evaluation task suites (route planning and
in-cabin settings) with reference answers, supporting evidence, and challenge
labels separated from model-visible input.

Paper-fixed targets:
  - 100 households
  - 5,000 base history events (establishment + evolution; 50 / household)
  - 2,000 route-planning tasks (20 / household)
  - 1,000 in-cabin tasks (10 / household)
"""
from __future__ import annotations

import copy
import hashlib
import json
import random
from datetime import datetime, timedelta

import constants as C


# --------------------------------------------------------------------------
# Deterministic RNG
# --------------------------------------------------------------------------
def rng_for(*parts) -> random.Random:
    key = "|".join(str(p) for p in parts)
    digest = hashlib.md5(f"{C.MASTER_SEED}:{key}".encode()).hexdigest()
    return random.Random(int(digest[:16], 16))


def parse_ts(ts: str) -> datetime:
    return datetime.fromisoformat(ts)


def fmt_ts(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%S")


def random_ts(rng, lo: str, hi: str) -> str:
    a, b = parse_ts(lo), parse_ts(hi)
    span = (b - a).total_seconds()
    return fmt_ts(a + timedelta(seconds=rng.uniform(0, span)))


def choice_weighted(rng, items):
    """items: list of (item, weight)."""
    total = sum(w for _, w in items)
    r = rng.uniform(0, total)
    acc = 0.0
    for item, w in items:
        acc += w
        if r <= acc:
            return item
    return items[-1][0]


# --------------------------------------------------------------------------
# Households and members
# --------------------------------------------------------------------------
def generate_households():
    households = []
    # Pre-shuffle name pools so names are globally unique.
    adult_names = list(C.ADULT_NAMES)
    child_names = list(C.CHILD_NAMES)
    rng = rng_for("names")
    rng.shuffle(adult_names)
    rng.shuffle(child_names)
    adult_i = 0
    child_i = 0

    for h in range(1, C.NUM_HOUSEHOLDS + 1):
        hid = f"hh_{h:04d}"
        rng = rng_for("household", h)
        comp = choice_weighted(rng, C.COMPOSITION_WEIGHTS)
        members = []

        n_adults = 2 if comp.startswith("two_adults") else 1
        n_children = 1 if comp.endswith("one_child") else (
            2 if comp == "two_adults_two_children" else 0)

        # Adults
        for a in range(n_adults):
            mid = f"{hid}_{chr(ord('A') + len(members))}"
            name = adult_names[adult_i % len(adult_names)]
            adult_i += 1
            occupation = rng.choice(C.ADULT_OCCUPATIONS)
            can_drive = rng.random() < (
                C.DRIVING_PROB_RETIRED if occupation == "retired"
                else C.DRIVING_PROB_DEFAULT_ADULT)
            members.append({
                "member_id": mid, "name": name,
                "role": "primary_user" if a == 0 else "spouse",
                "age_group": "adult", "occupation": occupation,
                "can_drive": can_drive,
            })
        # Children
        for ch in range(n_children):
            mid = f"{hid}_{chr(ord('A') + len(members))}"
            name = child_names[child_i % len(child_names)]
            child_i += 1
            occupation = rng.choice(C.CHILD_OCCUPATIONS)
            members.append({
                "member_id": mid, "name": name,
                "role": "child" if ch == 0 else "second_child",
                "age_group": "child", "occupation": occupation,
                "can_drive": False,
            })
        # Guarantee at least one driver.
        if not any(m["can_drive"] for m in members):
            for m in members:
                if m["age_group"] == "adult":
                    m["can_drive"] = True
                    break

        households.append({
            "household_id": hid, "namespace_id": hid,
            "composition": comp, "members": members,
        })
    return households


# --------------------------------------------------------------------------
# Name / place helpers
# --------------------------------------------------------------------------
def make_home_name(rng):
    return (rng.choice(C.RESIDENTIAL_PREFIXES) + " "
            + rng.choice(C.RESIDENTIAL_SUFFIXES))


def make_company_name(rng):
    return (rng.choice(C.COMPANY_PREFIXES) + " "
            + rng.choice(C.COMPANY_SUFFIXES) + " "
            + rng.choice(C.BUILDING_NAMES))


def make_school_name(rng, occupation):
    suffix = rng.choice(C.SCHOOL_SUFFIXES[occupation])
    return rng.choice(C.SCHOOL_PREFIXES) + " " + suffix


def make_poi_name(rng, poi_type):
    return rng.choice(C.FREQUENT_POI_TYPES[poi_type])


def make_parking_lot_name(rng):
    return (rng.choice(C.PARKING_LOT_PREFIXES) + " "
            + rng.choice(C.PARKING_LOT_SUFFIXES) + " "
            + rng.choice(C.PARKING_LOT_SECTIONS))


# --------------------------------------------------------------------------
# Memory / preference establishment
# --------------------------------------------------------------------------
def establish_memories(household):
    hid = household["household_id"]
    members = household["members"]
    rng = rng_for("establish", hid)
    memories = []
    mem_counter = [0]

    def new_id():
        mem_counter[0] += 1
        return f"{hid}_mem_{mem_counter[0]:04d}"

    adults = [m for m in members if m["age_group"] == "adult"]
    children = [m for m in members if m["age_group"] == "child"]
    drivers = [m for m in members if m["can_drive"]]
    primary = members[0]

    def add_memory(owner, category, content, condition=None):
        mid = new_id()
        memories.append({
            "memory_id": mid, "household_id": hid, "owner_id": owner["member_id"],
            "domain": "cabin" if category in C.CABIN_CATEGORIES else "route",
            "category": category,
            "created_at": random_ts(rng, C.TIME["creation_start"],
                                    C.TIME["creation_end"]),
            "initial_content": copy.deepcopy(content),
            "initial_condition": copy.deepcopy(condition or {}),
        })
        return memories[-1]

    # --- Location memories ---
    home_name = make_home_name(rng)
    # Home: shared, owned by primary adult (all members reference it).
    add_memory(primary, "location", {"semantic_role": "home",
                                     "place_name": home_name}, {})
    for m in adults:
        if m["occupation"] != "retired":
            add_memory(m, "location",
                       {"semantic_role": "workplace",
                        "place_name": make_company_name(rng)}, {})
    for m in children:
        add_memory(m, "location",
                   {"semantic_role": "school",
                    "place_name": make_school_name(rng, m["occupation"])}, {})
    # Frequent POIs: 1-3 per adult.
    for m in adults:
        n_poi = rng.randint(1, 3)
        for _ in range(n_poi):
            poi_type = rng.choice(list(C.FREQUENT_POI_TYPES.keys()))
            add_memory(m, "location",
                       {"semantic_role": "frequent_poi", "poi_type": poi_type,
                        "place_name": make_poi_name(rng, poi_type)}, {})

    # --- Route preferences (adults) ---
    for m in adults:
        n = rng.randint(2, 4)
        for _ in range(n):
            add_memory(m, "route_preference",
                       *make_route_pref(rng))

    # --- Parking preferences (drivers) ---
    for m in drivers:
        n = rng.randint(1, 3)
        for _ in range(n):
            add_memory(m, "parking_preference",
                       *make_parking_pref(rng))

    # --- Passenger constraints (rare) ---
    for m in members:
        prob = C.CONSTRAINT_ASSIGN_PROB.get(
            "retired" if m["occupation"] == "retired" else m["age_group"], 0.0)
        if rng.random() < prob:
            ct = choice_weighted(rng, C.CONSTRAINT_TYPE_WEIGHTS)
            # Eligibility.
            if ct == "motion_sickness" and m["can_drive"]:
                ct = "need_regular_breaks"
            if ct == "limited_mobility" and m["occupation"] != "retired":
                ct = "need_regular_breaks"
            cond = ({"trip_context": "long_distance_trip"}
                    if ct == "need_regular_breaks" else {})
            add_memory(m, "passenger_constraint", {"constraint_type": ct}, cond)

    # --- In-cabin preferences (cover all 5 categories, plus extras) ---
    # Ensure one per category, then add up to ~8 total.
    cat_order = list(C.CABIN_CATEGORIES.keys())
    rng.shuffle(cat_order)
    owners = members
    for cat in cat_order:
        owner = rng.choice(owners)
        add_memory(owner, cat, *make_cabin_pref(rng, cat))
    extra = rng.randint(2, 4)
    for _ in range(extra):
        cat = rng.choice(cat_order)
        owner = rng.choice(owners)
        add_memory(owner, cat, *make_cabin_pref(rng, cat))

    # Fill to a minimum establishment floor so the evolution budget stays
    # reasonable for every household size (and base events stay exactly 50).
    while len(memories) < 32:
        cat = rng.choice(cat_order)
        owner = rng.choice(owners)
        add_memory(owner, cat, *make_cabin_pref(rng, cat))

    return memories


def make_route_pref(rng):
    cond_type = choice_weighted(rng, list(C.ROUTE_CONDITION_DISTRIBUTION.items()))
    condition = {}
    if cond_type == "unconditional":
        attr = rng.choice(list(C.ROUTE_PREF_OPTIONS.keys()))
        val = rng.choice(C.ROUTE_PREF_OPTIONS[attr])
    elif cond_type == "destination":
        role = rng.choice(C.CONDITIONAL_DESTINATION_ROLES)
        condition = {"destination": role}
        attr = rng.choice(list(C.ROUTE_PREF_OPTIONS.keys()))
        val = rng.choice(C.ROUTE_PREF_OPTIONS[attr])
    else:
        cval = rng.choice(C.CONDITION_VALUES[cond_type])
        condition = {cond_type: cval}
        attr, val = rng.choice(C.CONDITIONAL_RULES[cond_type][cval])
    return {"attribute": attr, "preference": val}, condition


def make_parking_pref(rng):
    cond_type = choice_weighted(rng, list(C.PARKING_CONDITION_DISTRIBUTION.items()))
    ptype = choice_weighted(rng, C.PARKING_TYPE_WEIGHTS)
    condition = {}
    if cond_type == "destination":
        role = rng.choice(C.PARKING_CONDITIONAL_ROLES)
        condition = {"destination": role}
    content = {"preference_type": ptype}
    if ptype == "preferred_parking_lot":
        content["lot_name"] = make_parking_lot_name(rng)
    return content, condition


def make_cabin_pref(rng, category):
    zone = rng.choice(C.CABIN_ZONES[category])
    attr = rng.choice(list(C.CABIN_CATEGORIES[category].keys()))
    val = rng.choice(C.CABIN_CATEGORIES[category][attr])
    condition = {}
    if rng.random() < 0.40:
        ctype = rng.choice([t for t in C.CABIN_CONDITION_TYPES
                            if t != "passenger_id"])
        condition = {ctype: rng.choice(C.CABIN_CONDITION_VALUES[ctype])}
    content = {"control_type": category, "target_zone": zone,
               "attribute": attr, "value": val}
    return content, condition


# --------------------------------------------------------------------------
# Evolution
# --------------------------------------------------------------------------
def pick_evolution_op(rng, category, condition):
    """Return a valid next operation for a memory's current condition state."""
    has_cond = bool(condition)
    if category == "location":
        return rng.choice(["content_update", "content_update", "cancellation"])
    if category == "passenger_constraint":
        return "cancellation"
    # route_preference / parking_preference / cabin categories
    if has_cond:
        return rng.choice(["content_update", "condition_update",
                           "conditional_to_unconditional", "cancellation"])
    return rng.choice(["content_update", "unconditional_to_conditional",
                       "cancellation"])


def apply_evolution(rng, category, content, condition):
    """Apply one evolution op; return (operation, new_content, new_condition,
    active)."""
    op = pick_evolution_op(rng, category, condition)

    if op == "cancellation":
        return op, content, condition, False

    if category == "location":
        # content_update: move to a new place (same role).
        role = content.get("semantic_role", "frequent_poi")
        new_content = dict(content)
        if role == "home":
            new_content["place_name"] = make_home_name(rng)
        elif role == "workplace":
            new_content["place_name"] = make_company_name(rng)
        elif role == "school":
            new_content["place_name"] = rng.choice(
                C.SCHOOL_PREFIXES) + " " + rng.choice(
                C.SCHOOL_SUFFIXES.get("high_school_student",
                                      C.SCHOOL_SUFFIXES["high_school_student"]))
        else:
            poi_type = content.get("poi_type", "supermarket")
            new_content["place_name"] = make_poi_name(rng, poi_type)
        return op, new_content, condition, True

    if category == "passenger_constraint":
        return op, content, condition, False

    if category == "route_preference":
        return evolve_route_pref(rng, op, content, condition)

    if category == "parking_preference":
        return evolve_parking_pref(rng, op, content, condition)

    # cabin
    return evolve_cabin_pref(rng, op, content, condition)


def evolve_route_pref(rng, op, content, condition):
    if op == "content_update":
        new_content, new_condition = make_route_pref(rng)
        # Keep same condition type/value if conditional (content update only
        # changes value, not conditions).
        if condition:
            new_content, _ = make_route_pref_under_condition(rng, condition)
            new_condition = dict(condition)
        return op, new_content, new_condition, True
    if op == "condition_update":
        ctype = next(iter(condition))
        if ctype == "destination":
            role = rng.choice(C.CONDITIONAL_DESTINATION_ROLES)
            new_condition = {"destination": role}
            new_content = {"attribute": content["attribute"],
                           "preference": content["preference"]}
            return op, new_content, new_condition, True
        cval = rng.choice(C.CONDITION_VALUES[ctype])
        new_condition = {ctype: cval}
        new_content, _ = make_route_pref_under_condition(rng, new_condition)
        return op, new_content, new_condition, True
    if op == "unconditional_to_conditional":
        ctype = rng.choice(list(C.CONDITION_VALUES.keys()))
        cval = rng.choice(C.CONDITION_VALUES[ctype])
        new_condition = {ctype: cval}
        new_content, _ = make_route_pref_under_condition(rng, new_condition)
        return op, new_content, new_condition, True
    if op == "conditional_to_unconditional":
        new_content = {"attribute": content["attribute"],
                       "preference": content["preference"]}
        return op, new_content, {}, True
    return op, content, condition, True


def make_route_pref_under_condition(rng, condition):
    ctype = next(iter(condition))
    cval = condition[ctype]
    if ctype in ("weather", "time_period", "trip_context"):
        attr, val = rng.choice(C.CONDITIONAL_RULES[ctype][cval])
    else:  # destination
        attr = rng.choice(list(C.ROUTE_PREF_OPTIONS.keys()))
        val = rng.choice(C.ROUTE_PREF_OPTIONS[attr])
    return {"attribute": attr, "preference": val}, condition


def evolve_parking_pref(rng, op, content, condition):
    if op == "content_update":
        new_content, new_condition = make_parking_pref(rng)
        if condition:
            new_content, new_condition = make_parking_pref(rng)
            new_condition = dict(condition)
        return op, new_content, new_condition, True
    if op == "condition_update":
        role = rng.choice(C.PARKING_CONDITIONAL_ROLES)
        return op, content, {"destination": role}, True
    if op == "unconditional_to_conditional":
        role = rng.choice(C.PARKING_CONDITIONAL_ROLES)
        return op, content, {"destination": role}, True
    if op == "conditional_to_unconditional":
        return op, content, {}, True
    return op, content, condition, True


def evolve_cabin_pref(rng, op, content, condition):
    category = content["control_type"]
    if op == "content_update":
        attr = content["attribute"]
        val = rng.choice(C.CABIN_CATEGORIES[category][attr])
        new_content = dict(content)
        new_content["value"] = val
        return op, new_content, dict(condition), True
    if op == "condition_update":
        if not condition:
            return "conditional_to_unconditional", content, {}, True
        ctype = next(iter(condition))
        if ctype == "passenger_id":
            return op, content, dict(condition), True
        cval = rng.choice(C.CABIN_CONDITION_VALUES[ctype])
        return op, content, {ctype: cval}, True
    if op == "unconditional_to_conditional":
        ctype = rng.choice([t for t in C.CABIN_CONDITION_TYPES
                            if t != "passenger_id"])
        cval = rng.choice(C.CABIN_CONDITION_VALUES[ctype])
        return op, content, {ctype: cval}, True
    if op == "conditional_to_unconditional":
        return op, content, {}, True
    return op, content, condition, True


def build_trajectories(household, memories, base_budget):
    """Assign evolution steps so that establishments + evolutions ==
    base_budget exactly. Returns (memories_with_trajectory, base_events)."""
    hid = household["household_id"]
    rng = rng_for("evolution", hid)
    n_est = len(memories)
    n_evo = base_budget - n_est
    if n_evo < 0:
        raise ValueError(f"{hid}: establishments {n_est} exceed budget "
                         f"{base_budget}")

    # Initialize every memory with an establish node; keep an active pool of
    # memories that can still evolve (home location is never evolved).
    active_pool = []
    for m in memories:
        content = copy.deepcopy(m["initial_content"])
        condition = copy.deepcopy(m["initial_condition"])
        m["_content"] = content
        m["_condition"] = condition
        m["_prev_ts"] = m["created_at"]
        m["trajectory"] = [make_node(m, "establish", None, content, condition,
                                     m["created_at"])]
        if not (m["category"] == "location"
                and content.get("semantic_role") == "home"):
            active_pool.append(m)

    # Apply evolution steps one at a time until exactly n_evo nodes exist.
    for _ in range(n_evo):
        if not active_pool:
            raise ValueError(f"{hid}: ran out of evolvable memories")
        m = rng.choice(active_pool)
        op, new_content, new_condition, active = apply_evolution(
            rng, m["category"], m["_content"], m["_condition"])
        m["_prev_ts"] = advance_ts(rng, m["_prev_ts"])
        m["trajectory"].append(make_node(
            m, op, (m["_content"], m["_condition"]),
            new_content, new_condition, m["_prev_ts"]))
        m["_content"], m["_condition"] = new_content, new_condition
        if not active:
            active_pool.remove(m)  # cancellation ends the trajectory

    base_events = []
    for m in memories:
        m["evolution_types"] = [n["operation"] for n in m["trajectory"][1:]]
        for node in m["trajectory"]:
            base_events.append(make_event_from_node(household, m, node))
        # Remove ephemeral fields.
        for f in ("initial_content", "initial_condition", "created_at",
                  "_content", "_condition", "_prev_ts"):
            m.pop(f, None)

    base_events.sort(key=lambda e: e["timestamp"])
    return memories, base_events


def make_node(mem, operation, prev_state, content, condition, timestamp):
    return {
        "node_id": f"{mem['memory_id']}_node_{len(mem.get('trajectory', [])) + 1:03d}",
        "timestamp": timestamp,
        "operation": operation,
        "state": copy.deepcopy(content),
        "condition": copy.deepcopy(condition),
    }


def advance_ts(rng, prev_ts):
    dt = parse_ts(prev_ts) + timedelta(days=rng.randint(
        C.TIME["minimum_gap_days"], 60))
    if dt > parse_ts(C.TIME["evolution_end"]):
        dt = parse_ts(C.TIME["evolution_end"])
    return fmt_ts(dt)


# --------------------------------------------------------------------------
# Events
# --------------------------------------------------------------------------
def make_event_from_node(household, mem, node):
    return {
        "event_id": None,  # assigned after sorting
        "household_id": household["household_id"],
        "timestamp": node["timestamp"],
        "speaker_id": mem["owner_id"],
        "event_type": ("establishment" if node["operation"] == "establish"
                       else "evolution"),
        "domain": mem["domain"],
        "memory_ids": [mem["memory_id"]],
        "operation": node["operation"],
        "memory": mem,
        "context": {},
    }


def generate_expanded_events(household, memories):
    """Applications, temporary requests, unrelated exchanges."""
    hid = household["household_id"]
    members = {m["member_id"]: m for m in household["members"]}
    rng = rng_for("expanded", hid)
    events = []

    # Active (non-cancelled) memories at history end, for applications.
    active_mems = [m for m in memories
                   if m["trajectory"][-1]["operation"] != "cancellation"]

    n_app = rng.randint(C.APPLICATION_COUNT["min"], C.APPLICATION_COUNT["max"])
    for _ in range(n_app):
        mem = rng.choice(active_mems)
        node = mem["trajectory"][-1]
        ts = random_ts(rng, node["timestamp"], C.TIME["history_end"])
        events.append({
            "event_id": None,
            "household_id": hid,
            "timestamp": ts,
            "speaker_id": mem["owner_id"],
            "event_type": "application",
            "domain": mem["domain"],
            "memory_ids": [mem["memory_id"]],
            "operation": "application",
            "memory": mem,
            "context": {},
        })

    # Temporary (one-trip-only) instructions.
    n_temp = rng.randint(C.TEMPORARY_COUNT["min"], C.TEMPORARY_COUNT["max"])
    for _ in range(n_temp):
        owner = rng.choice(list(members.values()))
        ts = random_ts(rng, C.TIME["creation_start"], C.TIME["history_end"])
        domain = rng.choice(["route", "cabin"])
        instruction = (rng.choice(C.TEMPORARY_ROUTE_OPTIONS) if domain == "route"
                       else rng.choice(C.TEMPORARY_CABIN_OPTIONS))
        events.append({
            "event_id": None,
            "household_id": hid,
            "timestamp": ts,
            "speaker_id": owner["member_id"],
            "event_type": "temporary",
            "domain": domain,
            "memory_ids": [],
            "operation": "temporary",
            "instruction": instruction,
            "memory": None,
            "context": {},
        })

    # Unrelated exchanges.
    n_unrel = rng.randint(C.UNRELATED_COUNT["min"], C.UNRELATED_COUNT["max"])
    for _ in range(n_unrel):
        owner = rng.choice(list(members.values()))
        ts = random_ts(rng, C.TIME["creation_start"], C.TIME["history_end"])
        events.append({
            "event_id": None,
            "household_id": hid,
            "timestamp": ts,
            "speaker_id": owner["member_id"],
            "event_type": "unrelated",
            "domain": "unrelated",
            "memory_ids": [],
            "operation": "unrelated",
            "memory": None,
            "context": {},
        })

    return events


# --------------------------------------------------------------------------
# History assembly + template dialogue
# --------------------------------------------------------------------------
def assemble_history(household, base_events, expanded_events):
    hid = household["household_id"]
    members = {m["member_id"]: m for m in household["members"]}
    all_events = sorted(base_events + expanded_events,
                        key=lambda e: e["timestamp"])
    sessions = {}
    final_events = []
    for i, ev in enumerate(all_events, 1):
        ev["event_id"] = f"{hid}_event_{i:04d}"
        ev["speaker_name"] = members[ev["speaker_id"]]["name"]
        # Assign a session id by (speaker, day) so nearby turns share a session.
        day = ev["timestamp"][:10]
        skey = (ev["speaker_id"], day)
        sessions[skey] = sessions.get(skey, 0) + 1
        ev["session_id"] = f"{hid}_session_{sessions[skey]:04d}"
        ev["user_utterance"], ev["agent_response"] = template_dialogue(
            household, ev)
        # Keep a compact serializable fact + template; drop raw memory object.
        ev["fact"] = fact_for_event(ev)
        ev.pop("memory", None)
        final_events.append(ev)
    return final_events


def application_dialogue(ev):
    """Recurring-use request that applies the current preference state."""
    mem = ev["memory"]
    cat = mem["category"]
    node = mem["trajectory"][-1]
    state = node["state"]
    cond = node["condition"]
    if cat == "location":
        role = state["semantic_role"]
        place = state["place_name"]
        req = {"home": "Take me home.", "workplace": "Take me to work.",
               "school": "Take me to school."}.get(
            role, f"Take me to {place}.")
        return req, f"Navigating to {place}."
    if cat == "route_preference":
        attr = state["attribute"].replace("_", " ")
        val = state["preference"]
        return ("Let's head out — use my usual route preferences.",
                f"I'll {val} {attr} as you prefer.")
    if cat == "parking_preference":
        return ("Find me parking as usual.",
                f"I'll apply your parking preference.")
    if cat == "passenger_constraint":
        ct = state["constraint_type"]
        return ("Please make sure the ride is comfortable for everyone.",
                f"I'll keep your {ct.replace('_', ' ')} in mind.")
    # cabin
    zone = state["target_zone"]
    attr = display_attr(state["attribute"])
    val = state["value"]
    return (f"Set the {zone} {attr} to my usual preference.",
            f"Setting the {zone} {attr} to {val}.")


def fact_for_event(ev):
    """Canonical structured fact the dialogue must express (for verbalizer +
    validation)."""
    if ev["event_type"] in ("establishment", "evolution"):
        mem = ev["memory"]
        node = mem["trajectory"][-1]
        return {
            "memory_id": mem["memory_id"],
            "category": mem["category"],
            "owner_id": mem["owner_id"],
            "operation": ev["operation"],
            "state": node["state"],
            "condition": node["condition"],
        }
    if ev["event_type"] == "application":
        mem = ev["memory"]
        node = mem["trajectory"][-1]
        return {
            "memory_id": mem["memory_id"],
            "category": mem["category"],
            "owner_id": mem["owner_id"],
            "operation": "application",
            "state": node["state"],
            "condition": node["condition"],
        }
    if ev["event_type"] == "temporary":
        return {"operation": "temporary", "domain": ev["domain"],
                "instruction": ev.get("instruction", {})}
    return {"operation": ev["operation"]}


def template_dialogue(household, ev):
    """Deterministic template text (ground truth; replaced by LLM later)."""
    name = ev["speaker_name"]
    if ev["event_type"] == "application":
        return application_dialogue(ev)
    if ev["event_type"] in ("establishment", "evolution"):
        mem = ev["memory"]
        cat = mem["category"]
        node = mem["trajectory"][-1]
        state = node["state"]
        cond = node["condition"]
        op = ev["operation"]
        if cat == "location":
            role = state["semantic_role"]
            place = state["place_name"]
            if op == "establish":
                return (f"{state_for_loc(role, place)}",
                        f"Understood. I've set {place} as your "
                        f"{display_role(role)}.")
            if op == "content_update":
                return (f"My {display_role(role)} has changed to {place}.",
                        f"Understood. I've updated your {display_role(role)} "
                        f"to {place}.")
            if op == "cancellation":
                return (f"You can forget about {place}.",
                        f"Understood. I've removed {place}.")
        if cat == "route_preference":
            phrase = route_pref_phrase(state, cond)
            if op == "establish":
                return (phrase, f"Understood. {assistant_route_ack(state)}")
            if op == "content_update":
                return (f"Actually, {phrase.lower()}",
                        f"Understood. {assistant_route_ack(state)}")
            if op == "condition_update":
                return (f"Update: {phrase.lower()}",
                        f"Understood. {assistant_route_ack(state)}")
            if op == "unconditional_to_conditional":
                return (f"Only {phrase.lower()}",
                        f"Understood. {assistant_route_ack(state)}")
            if op == "conditional_to_unconditional":
                return (f"{phrase} from now on.",
                        f"Understood. {assistant_route_ack(state)}")
            if op == "cancellation":
                return ("Please stop applying that route preference.",
                        "Understood. I've removed that route preference.")
        if cat == "parking_preference":
            phrase = parking_pref_phrase(state, cond)
            if op == "establish":
                return (phrase, f"Understood. {assistant_parking_ack(state)}")
            if op == "content_update":
                return (f"Actually, {phrase.lower()}",
                        f"Understood. {assistant_parking_ack(state)}")
            if op == "cancellation":
                return ("Please stop applying that parking preference.",
                        "Understood. I've removed that parking preference.")
            return (phrase, f"Understood. {assistant_parking_ack(state)}")
        if cat == "passenger_constraint":
            phrase = constraint_phrase(state)
            if op == "establish":
                return (phrase, "Understood. I'll keep that in mind.")
            if op == "cancellation":
                return ("That passenger constraint no longer applies.",
                        "Understood. I've removed that constraint.")
        # cabin
        phrase = cabin_pref_phrase(state, cond)
        if op == "establish":
            return (phrase, f"Understood. {assistant_cabin_ack(state)}")
        if op == "content_update":
            return (f"Actually, {phrase.lower()}",
                    f"Understood. {assistant_cabin_ack(state)}")
        if op == "cancellation":
            return ("Please forget that cabin setting.",
                    "Understood. I've removed that cabin setting.")
        return (phrase, f"Understood. {assistant_cabin_ack(state)}")

    if ev["event_type"] == "temporary":
        instr = ev.get("instruction", {})
        if ev["domain"] == "route":
            attr = instr.get("attribute", "highway").replace("_", " ")
            pref = instr.get("preference", "prefer")
            return (f"Just for this trip, please {pref} {attr}.",
                    f"Understood. For this trip only, I'll {pref} {attr}.")
        attr = display_attr(instr.get("attribute", "temperature_c"))
        val = instr.get("value", 26)
        return (f"Just for now, set the {attr} to {val}.",
                f"Understood. For this trip only, I'll set the {attr} to {val}.")
    # unrelated
    return unrelated_exchange(ev)
    return ("", "")


def state_for_loc(role, place):
    if role == "home":
        return f"{place} is my home address."
    if role == "workplace":
        return f"My workplace is {place}."
    if role == "school":
        return f"My school is {place}."
    return f"{place} is one of my frequent places."


def display_role(role):
    return {"home": "home", "workplace": "workplace", "school": "school",
            "frequent_poi": "frequent place"}.get(role, role)


def display_attr(attr):
    return "temperature" if attr == "temperature_c" else attr.replace("_", " ")


def route_pref_phrase(state, cond):
    attr = state["attribute"]
    val = state["preference"]
    base = f"I prefer to {val} {attr.replace('_', ' ')}"
    if cond:
        ctype, cval = next(iter(cond.items()))
        return f"{base} when it's {cond_value_phrase(ctype, cval)}."
    return base + "."


def assistant_route_ack(state):
    attr = state["attribute"].replace("_", " ")
    return f"I'll {state['preference']} {attr} on your routes."


def parking_pref_phrase(state, cond):
    pt = state["preference_type"]
    base = {
        "nearest": "I prefer the nearest parking",
        "low_cost": "I prefer low-cost parking",
        "indoor": "I prefer indoor parking",
        "easy_to_park": "I prefer parking that is easy to get in and out of",
        "preferred_parking_lot": f"I prefer to park at {state.get('lot_name', 'my usual lot')}",
    }[pt]
    if cond:
        ctype, cval = next(iter(cond.items()))
        return f"{base} when going to {cond_value_phrase(ctype, cval)}."
    return base + "."


def assistant_parking_ack(state):
    pt = state["preference_type"]
    return {
        "nearest": "I'll pick the nearest parking.",
        "low_cost": "I'll pick low-cost parking.",
        "indoor": "I'll pick indoor parking.",
        "easy_to_park": "I'll pick parking that is easy to get in and out of.",
        "preferred_parking_lot": "I'll use your preferred parking lot.",
    }[pt]


def constraint_phrase(state):
    ct = state["constraint_type"]
    return {
        "motion_sickness": "I get motion sickness; please choose a smoother route.",
        "limited_mobility": "I have limited mobility; please minimize walking.",
        "need_regular_breaks": "On long trips, please choose routes with rest stops.",
    }[ct]


def cabin_pref_phrase(state, cond):
    zone = state["target_zone"]
    attr = display_attr(state["attribute"])
    val = state["value"]
    base = f"Please set the {zone} {attr} to {val}"
    if cond:
        ctype, cval = next(iter(cond.items()))
        return f"{base} when {cond_value_phrase(ctype, cval)}."
    return base + "."


def assistant_cabin_ack(state):
    zone = state["target_zone"]
    attr = display_attr(state["attribute"])
    return f"I'll set the {zone} {attr} to {state['value']}."


def cond_value_phrase(ctype, cval):
    if ctype == "destination":
        return f"going to the {cval}"
    if ctype == "weather":
        return {"rainy": "it's rainy", "foggy": "it's foggy",
                "snowy": "it's snowy"}.get(cval, cval)
    if ctype == "time_period":
        return {"morning": "the morning rush", "evening": "the evening rush",
                "night": "night time"}.get(cval, cval)
    if ctype == "trip_context":
        return {"commute": "my commute", "daily_errand": "a daily errand",
                "long_distance_trip": "a long trip"}.get(cval, cval)
    if ctype == "outside_temperature_band":
        return {"cold": "it's cold outside", "mild": "it's mild outside",
                "hot": "it's hot outside"}.get(cval, cval)
    return str(cval)


_UNRELATED = [
    ("What time is it right now?", "It's currently 3:24 PM."),
    ("How's the traffic on the ring road?",
     "Traffic on the ring road is light right now."),
    ("Can you check the weather for today?",
     "It will be partly cloudy with a high of 22 degrees."),
    ("Remind me to get gas this week.",
     "Sure, I've noted a reminder to refuel this week."),
    ("How far is the nearest charging station?",
     "The nearest charging station is about 2 km away."),
    ("Play something relaxing.",
     "Playing a calm playlist now."),
    ("Is there any news about the road closure?",
     "No major road closures reported on your usual routes."),
    ("Thanks, that's all for now.",
     "You're welcome. Drive safely."),
]


def unrelated_exchange(ev):
    idx = int(hashlib.md5(ev["event_id"].encode()).hexdigest()[:8], 16)
    return _UNRELATED[idx % len(_UNRELATED)]


# --------------------------------------------------------------------------
# Route tasks
# --------------------------------------------------------------------------
def final_active(memories):
    """Return {memory_id: (memory, last_node)} for non-cancelled memories."""
    out = {}
    for m in memories:
        node = m["trajectory"][-1]
        if node["operation"] != "cancellation":
            out[m["memory_id"]] = (m, node)
    return out


def resolve_route_requirements(active, requester_id, destination_role, context,
                               occupant_ids):
    """Collect active route requirements for a trip. Returns list of
    (source_memory, attribute, value). Conflicting same-attribute requirements
    are deduplicated, keeping the most recently updated memory."""
    cand = []
    for mid, (m, node) in active.items():
        st = node["state"]
        cond = node["condition"]
        ts = node["timestamp"]
        if m["category"] == "route_preference":
            # Route preferences belong to the driver.
            if m["owner_id"] == requester_id and \
                    condition_active(cond, destination_role, context):
                cand.append((m, st["attribute"], st["preference"], ts))
        elif m["category"] == "passenger_constraint":
            # Constraint applies only when the constrained person is present.
            if m["owner_id"] in occupant_ids and \
                    constraint_applies(m, node, context):
                ct = st["constraint_type"]
                for attr, val in C.ROUTE_CONSTRAINT_EVAL_MAP.get(ct, {}).items():
                    cand.append((m, attr, val, ts))
    best = {}
    for m, attr, val, ts in cand:
        if attr not in best or ts >= best[attr][1]:
            best[attr] = (m, ts, val)
    reqs = [(m, attr, val) for attr, (m, _ts, val) in best.items()]
    evidence = list(dict.fromkeys(m["memory_id"] for (m, _a, _v) in reqs))
    return reqs, evidence


def condition_active(cond, destination_role, context):
    if not cond:
        return True
    ctype, cval = next(iter(cond.items()))
    if ctype == "destination":
        return cval == destination_role
    if ctype == "weather":
        return context.get("weather") == cval
    if ctype == "time_period":
        return context.get("time_period") == cval
    if ctype == "trip_context":
        return context.get("trip_context") == cval
    return False


def constraint_applies(mem, node, context):
    cond = node["condition"]
    if not cond:
        return True
    ctype, cval = next(iter(cond.items()))
    if ctype == "trip_context":
        return context.get("trip_context") == cval
    return False


def generate_route_candidates(rng, reqs):
    """3 routes: exactly one gold (satisfies all reqs), 2 distractors."""
    attrs = {a: v for (_, a, v) in reqs}

    def satisfies(route):
        for a, v in attrs.items():
            if not route_attr_ok(route, a, v):
                return False
        return True

    # Gold: build a route satisfying all requirements.
    gold = build_route(rng)
    enforce_route(gold, attrs)
    keys = list(attrs.keys())
    # Distractor 1: violate the first requirement.
    d1 = build_route(rng)
    enforce_route(d1, attrs)
    violate_route(d1, attrs, keys[0] if keys else None)
    # Distractor 2: violate the last requirement (a different one).
    d2 = build_route(rng)
    enforce_route(d2, attrs)
    violate_route(d2, attrs, keys[-1] if keys else None)

    routes = [gold, d1, d2]
    rng.shuffle(routes)
    return routes, gold


def build_route(rng):
    r = dict(C.ROUTE_BASELINE)
    r["uses_highway"] = rng.random() < 0.5
    r["uses_toll_road"] = rng.random() < 0.5
    r["congestion_level"] = rng.choice(C.ROUTE_CATEGORICAL["congestion_level"])
    r["traffic_lights"] = rng.randint(*C.ROUTE_RANGES["traffic_lights"])
    r["lane_changes"] = rng.randint(*C.ROUTE_RANGES["lane_changes"])
    r["smoothness"] = rng.choice(C.ROUTE_CATEGORICAL["smoothness"])
    r["scenery"] = rng.choice(C.ROUTE_CATEGORICAL["scenery"])
    r["travel_time_minutes"] = rng.randint(*C.ROUTE_RANGES["travel_time_minutes"])
    r["distance_km"] = round(rng.uniform(*C.ROUTE_RANGES["distance_km"]), 1)
    r["rest_stop_available"] = rng.random() < 0.5
    return r


def route_attr_ok(route, attr, val):
    if attr == "highway":
        return (route["uses_highway"]) if val == "prefer" else (
            not route["uses_highway"])
    if attr == "toll_road":
        return (route["uses_toll_road"]) if val == "prefer" else (
            not route["uses_toll_road"])
    if attr == "congestion":
        return route["congestion_level"] == "low"
    if attr == "traffic_lights":
        return route["traffic_lights"] <= 8
    if attr == "lane_changes":
        return route["lane_changes"] <= 6
    if attr == "smoothness":
        return route["smoothness"] == "high"
    if attr == "scenery":
        return route["scenery"] == "high"
    if attr == "travel_time":
        return route["travel_time_minutes"] <= 35
    if attr == "distance":
        return route["distance_km"] <= 22.0
    if attr == "rest_stop_available":
        return route["rest_stop_available"] is True
    return True


def enforce_route(route, attrs):
    for a, v in attrs.items():
        # Set the attribute to satisfy v.
        if a == "highway":
            route["uses_highway"] = (v == "prefer")
        elif a == "toll_road":
            route["uses_toll_road"] = (v == "prefer")
        elif a == "congestion":
            route["congestion_level"] = "low"
        elif a == "traffic_lights":
            route["traffic_lights"] = 5
        elif a == "lane_changes":
            route["lane_changes"] = 3
        elif a == "smoothness":
            route["smoothness"] = "high"
        elif a == "scenery":
            route["scenery"] = "high"
        elif a == "travel_time":
            route["travel_time_minutes"] = 28
        elif a == "distance":
            route["distance_km"] = 15.0
        elif a == "rest_stop_available":
            route["rest_stop_available"] = True


def violate_route(route, attrs, key=None):
    """Violate one attribute to make the route a distractor."""
    if not attrs:
        route["travel_time_minutes"] = 60
        return
    if key is None:
        key = next(iter(attrs))
    v = attrs[key]
    if key == "highway":
        route["uses_highway"] = (v != "prefer")
    elif key == "toll_road":
        route["uses_toll_road"] = (v != "prefer")
    elif key == "congestion":
        route["congestion_level"] = "high"
    elif key == "traffic_lights":
        route["traffic_lights"] = 15
    elif key == "lane_changes":
        route["lane_changes"] = 10
    elif key == "smoothness":
        route["smoothness"] = "low"
    elif key == "scenery":
        route["scenery"] = "low"
    elif key == "travel_time":
        route["travel_time_minutes"] = 55
    elif key == "distance":
        route["distance_km"] = 40.0
    elif key == "rest_stop_available":
        route["rest_stop_available"] = False


def generate_parking_candidates(rng, reqs):
    attrs = {a: v for (_, a, v) in reqs}

    def satisfies(p):
        for a, v in attrs.items():
            if not parking_attr_ok(p, a, v):
                return False
        return True

    gold = build_parking(rng)
    enforce_parking(gold, attrs)
    keys = list(attrs.keys())
    d1 = build_parking(rng)
    enforce_parking(d1, attrs)
    violate_parking(d1, attrs, keys[0] if keys else None)
    d2 = build_parking(rng)
    enforce_parking(d2, attrs)
    violate_parking(d2, attrs, keys[-1] if keys else None)
    lots = [gold, d1, d2]
    rng.shuffle(lots)
    return lots, gold


def build_parking(rng):
    p = dict(C.PARKING_BASELINE)
    p["walking_distance_m"] = rng.randint(*C.PARKING_RANGES["walking_distance_m"])
    p["cost_level"] = rng.choice(C.PARKING_CATEGORICAL["cost_level"])
    p["indoor"] = rng.random() < 0.5
    p["easy_to_park"] = rng.random() < 0.5
    p["accessible_pickup"] = rng.random() < 0.5
    p["lot_name"] = make_parking_lot_name(rng)
    return p


def parking_attr_ok(p, attr, val):
    if attr == "walking_distance":
        return p["walking_distance_m"] <= 300
    if attr == "accessible_pickup":
        return p["accessible_pickup"] is True
    if attr == "indoor":
        return p["indoor"] is True
    if attr == "easy_to_park":
        return p["easy_to_park"] is True
    if attr == "low_cost":
        return p["cost_level"] == "low"
    if attr == "nearest":
        return p["walking_distance_m"] <= 150
    if attr == "preferred_parking_lot":
        return p.get("lot_name") == val
    return True


def enforce_parking(p, attrs):
    for a, v in attrs.items():
        if a == "walking_distance":
            p["walking_distance_m"] = 120
        elif a == "accessible_pickup":
            p["accessible_pickup"] = True
        elif a == "indoor":
            p["indoor"] = True
        elif a == "easy_to_park":
            p["easy_to_park"] = True
        elif a == "low_cost":
            p["cost_level"] = "low"
        elif a == "nearest":
            p["walking_distance_m"] = 100
        elif a == "preferred_parking_lot":
            p["lot_name"] = v


def violate_parking(p, attrs, key=None):
    if not attrs:
        p["walking_distance_m"] = 600
        return
    if key is None:
        key = next(iter(attrs))
    v = attrs[key]
    if key == "walking_distance":
        p["walking_distance_m"] = 500
    elif key == "accessible_pickup":
        p["accessible_pickup"] = False
    elif key == "indoor":
        p["indoor"] = False
    elif key == "easy_to_park":
        p["easy_to_park"] = False
    elif key == "low_cost":
        p["cost_level"] = "high"
    elif key == "nearest":
        p["walking_distance_m"] = 600
    elif key == "preferred_parking_lot":
        p["lot_name"] = make_parking_lot_name_for_distractor(p, v)


def generate_route_tasks(household, memories):
    hid = household["household_id"]
    members = {m["member_id"]: m for m in household["members"]}
    rng = rng_for("route_tasks", hid)
    active = final_active(memories)

    # Collect active locations by owner.
    locs_by_owner = {}
    for mid, (m, node) in active.items():
        if m["category"] == "location":
            locs_by_owner.setdefault(m["owner_id"], []).append((m, node))

    tasks = []
    for t in range(C.ROUTE_TASKS_PER_HOUSEHOLD):
        # Requesting user = a driver who has locations.
        drivers_with_locs = [uid for uid in members
                             if members[uid]["can_drive"] and uid in locs_by_owner]
        if not drivers_with_locs:
            drivers_with_locs = list(locs_by_owner.keys())
        requester = rng.choice(drivers_with_locs)
        loc_pool = locs_by_owner[requester]
        mem, node = rng.choice(loc_pool)
        role = node["state"]["semantic_role"]
        dest_name = node["state"]["place_name"]

        weather = rng.choice(C.TASK_WEATHER)
        time_period = rng.choice(C.TASK_TIME_PERIOD)
        trip_ctx = rng.choice(C.TASK_TRIP_CONTEXT)

        # Occupants: requester + optional passengers.
        passenger_ids = []
        others = [m for m in members.values() if m["member_id"] != requester]
        if others and rng.random() < 0.5:
            k = rng.randint(1, min(2, len(others)))
            passenger_ids = [m["member_id"] for m in rng.sample(others, k)]

        context = {"weather": weather, "time_period": time_period,
                   "trip_context": trip_ctx, "driver_id": requester,
                   "occupant_ids": [requester] + passenger_ids}

        reqs, evidence = resolve_route_requirements(
            active, requester, role, context, context["occupant_ids"])
        # Default objective ensures a unique gold even when no personal
        # preference applies (navigation still must pick a route).
        if not reqs:
            reqs = [(None, "travel_time", "minimize")]

        # Parking required only when the driver has a parking preference for
        # this destination and a seeded coin passes.
        parking_required = False
        parking_reqs, parking_evidence = [], []
        if role in C.PARKING_CONDITIONAL_ROLES or role == "workplace":
            parking_reqs, parking_evidence = resolve_parking_requirements(
                active, requester, role, context, context["occupant_ids"])
            if parking_reqs and rng.random() < C.PARKING_REQUIRED_PROB:
                parking_required = True

        routes, gold_route = generate_route_candidates(rng, reqs)
        gold_route_id = None
        for i, r in enumerate(routes, 1):
            r["route_id"] = f"{hid}_task_{t+1:03d}_route_{i}"
            if r is gold_route:
                gold_route_id = r["route_id"]

        parking_candidates = []
        gold_parking_id = None
        if parking_required:
            lots, gold_park = generate_parking_candidates(rng, parking_reqs)
            for i, p in enumerate(lots, 1):
                p["parking_id"] = f"{hid}_task_{t+1:03d}_parking_{i}"
                if p is gold_park:
                    gold_parking_id = p["parking_id"]
            parking_candidates = lots

        # Challenge labels (paper's four, overlapping).
        labels = []
        if user_binding_check(active, requester, context, role):
            labels.append("user_binding")
        if preference_update_check(memories, requester, role, evidence):
            labels.append("preference_update")
        if any(condition_active(m["trajectory"][-1]["condition"], role, context)
               for mid, (m, node) in active.items()
               if m["category"] in ("route_preference", "parking_preference")
               and node["condition"] and m["owner_id"] == requester):
            labels.append("conditional_application")
        req_attrs = {r[1] for r in reqs}
        if any(attr in req_attrs for attr in household.get("_route_temp_attrs", [])):
            labels.append("temporary_request_interference")

        request_text = route_request_text(role, dest_name, members,
                                          requester, passenger_ids,
                                          parking_required, rng)

        tasks.append({
            "task_id": f"{hid}_route_task_{t+1:03d}",
            "household_id": hid,
            "public": {
                "request": {"text": request_text,
                            "speaker_id": requester,
                            "speaker_name": members[requester]["name"]},
                "context": context,
                "parking_required": parking_required,
                "stages": {
                    "destination_resolution": {"stage_index": 1,
                                               "candidates": None},
                    "route_selection": {"stage_index": 2,
                                        "candidates": routes},
                    "parking_selection": {
                        "stage_index": 3,
                        "required": parking_required,
                        "candidates": parking_candidates,
                    },
                },
            },
            "evaluation": {
                "gold": {
                    "destination": {"memory_id": mem["memory_id"],
                                    "name": dest_name,
                                    "semantic_role": role},
                    "route_id": gold_route_id,
                    "parking_id": gold_parking_id,
                },
                "evidence": evidence + parking_evidence,
                "challenge_labels": labels,
            },
        })
    return tasks


def resolve_parking_requirements(active, requester_id, destination_role,
                                 context, occupant_ids):
    cand = []
    for mid, (m, node) in active.items():
        st = node["state"]
        ts = node["timestamp"]
        if m["category"] == "parking_preference" and m["owner_id"] == requester_id:
            if condition_active(node["condition"], destination_role, context):
                pt = st["preference_type"]
                if pt == "preferred_parking_lot":
                    cand.append((m, "preferred_parking_lot",
                                 st.get("lot_name"), ts))
                else:
                    cand.append((m, pt, True, ts))
        elif m["category"] == "passenger_constraint" and \
                m["owner_id"] in occupant_ids:
            ct = st["constraint_type"]
            if ct == "limited_mobility" and constraint_applies(m, node, context):
                for attr, val in C.PARKING_CONSTRAINT_EVAL_MAP[ct].items():
                    cand.append((m, attr, val, ts))
    best = {}
    for m, attr, val, ts in cand:
        if attr not in best or ts >= best[attr][1]:
            best[attr] = (m, ts, val)
    reqs = [(m, attr, val) for attr, (m, _ts, val) in best.items()]
    evidence = list(dict.fromkeys(m["memory_id"] for (m, _a, _v) in reqs))
    return reqs, evidence


def make_parking_lot_name_for_distractor(p, preferred_name):
    # Return a lot name different from the preferred one.
    name = make_parking_lot_name(rng_for("distractor_parking"))
    while name == preferred_name:
        name = make_parking_lot_name(rng_for("distractor_parking2"))
    return name


def user_binding_check(active, requester, context, role):
    """True if another member has a same-type preference (route preference or
    a location of the same semantic role) that could be confused with the
    requester's."""
    others = set()
    for mid, (m, node) in active.items():
        if m["owner_id"] == requester:
            continue
        if m["category"] == "route_preference":
            if condition_active(node["condition"], role, context):
                others.add(m["owner_id"])
        elif m["category"] == "location":
            if node["state"].get("semantic_role") == role:
                others.add(m["owner_id"])
    return len(others) > 0


def preference_update_check(memories, requester, role, evidence):
    """True if any supporting memory evolved (has >1 trajectory node) or the
    destination memory evolved."""
    for m in memories:
        if m["memory_id"] in set(evidence) or (
            m["category"] == "location" and m["owner_id"] == requester
            and m["trajectory"][-1]["state"].get("semantic_role") == role):
            if len(m["trajectory"]) > 1:
                return True
    return False


def route_request_text(role, dest_name, members, requester, passenger_ids,
                       parking_required, rng):
    passenger_names = [members[p]["name"] for p in passenger_ids]
    if not passenger_names:
        who = "me"
    elif len(passenger_names) == 1:
        who = passenger_names[0] + " and me"
    else:
        who = ", ".join(passenger_names) + ", and me"

    if role == "home":
        base = rng.choice([f"Take {who} home.", f"Let's head home.",
                           f"Navigate home.", "I'd like to go home."])
    elif role == "workplace":
        base = rng.choice([f"Take {who} to work.", "Let's head to my office.",
                           "Navigate to my workplace.",
                           "I need to get to work."])
    elif role == "school":
        base = rng.choice([f"Take {who} to school.", "Let's go to my school.",
                           "Navigate to school.", "I need to get to school."])
    else:
        base = rng.choice([f"Take {who} to {dest_name}.",
                           f"Let's head to {dest_name}.",
                           f"Navigate to {dest_name}.",
                           f"I need to go to {dest_name}."])
    if parking_required:
        base += rng.choice([" Please find parking as well.",
                            " I also need a parking spot.",
                            " Please include parking.",
                            " Find somewhere suitable to park."])
    return base


# --------------------------------------------------------------------------
# Cabin tasks
# --------------------------------------------------------------------------
def generate_cabin_tasks(household, memories):
    hid = household["household_id"]
    members = {m["member_id"]: m for m in household["members"]}
    rng = rng_for("cabin_tasks", hid)
    active = final_active(memories)
    cabin_mems = [m for m in memories
                  if m["category"] in C.CABIN_CATEGORIES
                  and m["trajectory"][-1]["operation"] != "cancellation"]

    tasks = []
    for t in range(C.CABIN_TASKS_PER_HOUSEHOLD):
        # Pick a requesting user with cabin preferences.
        owners_with_prefs = list({m["owner_id"] for m in cabin_mems})
        requester = rng.choice(owners_with_prefs)
        owner_prefs = [m for m in cabin_mems if m["owner_id"] == requester]

        weather = rng.choice(C.TASK_WEATHER)
        outside = rng.choice(C.CABIN_CONDITION_VALUES["outside_temperature_band"])
        time_period = rng.choice(C.CABIN_CONDITION_VALUES["time_period"])
        trip_ctx = rng.choice(C.CABIN_CONDITION_VALUES["trip_context"])

        # Occupants + seat assignment.
        occupant_ids = [requester]
        others = [m for m in members.values() if m["member_id"] != requester]
        if others and rng.random() < 0.6:
            occupant_ids += [m["member_id"] for m in
                             rng.sample(others, rng.randint(1, len(others)))]
        seats = dict(zip(occupant_ids[:len(C.SEATS)], C.SEATS[:len(occupant_ids)]))

        context = {"weather": weather,
                   "outside_temperature_band": outside,
                   "time_period": time_period, "trip_context": trip_ctx,
                   "occupants": [{"member_id": uid, "seat": seats[uid]}
                                 for uid in occupant_ids]}

        # Select 1-3 target preferences (owner-bound, condition-relevant),
        # avoiding mutually-exclusive combinations (heating + ventilation both
        # active on the same zone).
        target_mems = []
        target_fields = {}
        for _attempt in range(30):
            k = rng.randint(1, min(3, len(owner_prefs)))
            cand = rng.sample(owner_prefs, k)
            tf = {}
            for m in cand:
                node = m["trajectory"][-1]
                st = node["state"]
                tf[f"{st['target_zone']}.{st['attribute']}"] = st["value"]
            if not cabin_conflict(tf):
                target_mems, target_fields = cand, tf
                break
        if not target_mems:
            target_mems = [rng.choice(owner_prefs)]
            node = target_mems[0]["trajectory"][-1]
            st = node["state"]
            target_fields = {f"{st['target_zone']}.{st['attribute']}":
                             st["value"]}

        evidence = []
        labels = []
        for m in target_mems:
            node = m["trajectory"][-1]
            st = node["state"]
            cond = node["condition"]
            zone = st["target_zone"]
            attr = st["attribute"]
            evidence.append(m["memory_id"])
            if cond:
                if cabin_cond_active(cond, context):
                    labels.append("conditional_application")
            if len(m["trajectory"]) > 1:
                labels.append("preference_update")

        # Initial device state (random but different from targets where
        # possible so the task requires a change).
        initial_state = build_initial_cabin_state(rng, target_fields)

        # Protected fields = all adjustable fields not in target set.
        all_fields = all_cabin_fields()
        protected = [f for f in all_fields if f not in target_fields]

        # User binding: another member has a same-category preference.
        for m in cabin_mems:
            if m["owner_id"] != requester and m["category"] in {
                    mm["category"] for mm in target_mems}:
                labels.append("user_binding")
                break

        # Temporary interference: household has a cabin temporary request
        # touching one of the task's target attributes.
        target_attrs = {m["trajectory"][-1]["state"]["attribute"]
                        for m in target_mems}
        if any(a in target_attrs for a in household.get("_cabin_temp_attrs", [])):
            labels.append("temporary_request_interference")

        labels = list(dict.fromkeys(labels))

        request_text = cabin_request_text(target_mems, requester, members, rng)

        tasks.append({
            "task_id": f"{hid}_cabin_task_{t+1:03d}",
            "household_id": hid,
            "public": {
                "request": {"text": request_text,
                            "speaker_id": requester,
                            "speaker_name": members[requester]["name"]},
                "environment": {
                    "context": context,
                    "initial_device_state": initial_state,
                    "adjustable_fields": cabin_field_spec(),
                    "constraints": [{"type": "mutual_exclusion",
                                     "fields": list(p)}
                                    for p in C.CABIN_MUTUAL_EXCLUSION],
                },
            },
            "evaluation": {
                "target_fields": target_fields,
                "protected_fields": protected,
                "execution_constraints": [
                    {"type": "mutual_exclusion", "fields": list(p)}
                    for p in C.CABIN_MUTUAL_EXCLUSION],
                "evidence": evidence,
                "challenge_labels": labels,
            },
        })
    return tasks


def cabin_cond_active(cond, context):
    if not cond:
        return True
    ctype, cval = next(iter(cond.items()))
    if ctype == "passenger_id":
        return True
    return context.get(ctype) == cval


def all_cabin_fields():
    fields = []
    for cat, attrs in C.CABIN_CATEGORIES.items():
        for zone in C.CABIN_ZONES[cat]:
            for attr in attrs:
                fields.append(f"{zone}.{attr}")
    return fields


def cabin_field_spec():
    spec = {}
    for cat, attrs in C.CABIN_CATEGORIES.items():
        for zone in C.CABIN_ZONES[cat]:
            for attr, vals in attrs.items():
                spec[f"{zone}.{attr}"] = list(vals)
    return spec


def cabin_conflict(target_fields):
    """True if any mutual-exclusion pair is violated by the target fields."""
    for a, b in C.CABIN_MUTUAL_EXCLUSION:
        for zone in ("driver", "front_passenger", "rear"):
            va = target_fields.get(f"{zone}.{a}")
            vb = target_fields.get(f"{zone}.{b}")
            if va is not None and vb is not None and va != "off" and vb != "off":
                return True
    return False


def build_initial_cabin_state(rng, target_fields):
    state = {}
    spec = cabin_field_spec()
    for field, vals in spec.items():
        if field in target_fields:
            # Pick a different value than target (to require a change).
            target = target_fields[field]
            others = [v for v in vals if v != target]
            state[field] = rng.choice(others) if others else target
        else:
            state[field] = rng.choice(vals)
    # Enforce mutual exclusion so the initial (and gold final) state never has
    # heating and ventilation active on the same zone.
    for a, b in C.CABIN_MUTUAL_EXCLUSION:
        for zone in ("driver", "front_passenger", "rear"):
            fa, fb = f"{zone}.{a}", f"{zone}.{b}"
            ta = target_fields.get(fa)
            tb = target_fields.get(fb)
            if ta is not None and ta != "off":
                state[fb] = "off"
            if tb is not None and tb != "off":
                state[fa] = "off"
            if state.get(fa, "off") != "off" and state.get(fb, "off") != "off":
                state[fb] = "off"
    return state


def cabin_request_text(target_mems, requester, members, rng):
    name = members[requester]["name"]
    cats = sorted({m["category"] for m in target_mems})
    if len(cats) == 1:
        cat = cats[0]
        if cat == "climate":
            base = rng.choice(["Set the climate to my usual preference.",
                               "Adjust the air conditioning to my usual settings."])
        elif cat == "media":
            base = rng.choice(["Play my usual music.",
                               "Set the media to my usual preference."])
        elif cat == "seat":
            base = rng.choice(["Adjust the seat to my usual setting."])
        elif cat == "window":
            base = rng.choice(["Set the windows to my usual preference."])
        else:
            base = rng.choice(["Set the ambient lighting to my usual preference."])
    else:
        base = "Restore all of my usual cabin settings."
    return base


# --------------------------------------------------------------------------
# Top-level structure generation
# --------------------------------------------------------------------------
def generate_structure():
    households = generate_households()
    out_households = []
    out_memories = []
    out_histories = []
    out_route_tasks = []
    out_cabin_tasks = []

    for hh in households:
        hid = hh["household_id"]
        memories = establish_memories(hh)
        memories, base_events = build_trajectories(
            hh, memories, C.BASE_EVENTS_PER_HOUSEHOLD)
        expanded = generate_expanded_events(hh, memories)
        # Record temporary-instruction attributes (for interference labels).
        hh["_route_temp_attrs"] = [
            e["instruction"]["attribute"] for e in expanded
            if e["event_type"] == "temporary" and e["domain"] == "route"]
        hh["_cabin_temp_attrs"] = [
            e["instruction"]["attribute"] for e in expanded
            if e["event_type"] == "temporary" and e["domain"] == "cabin"]

        events = assemble_history(hh, base_events, expanded)
        route_tasks = generate_route_tasks(hh, memories)
        cabin_tasks = generate_cabin_tasks(hh, memories)

        out_households.append({k: v for k, v in hh.items()
                               if not k.startswith("_")})
        out_memories.extend(memories)
        out_histories.append({"household_id": hid, "namespace_id": hid,
                              "events": events})
        out_route_tasks.extend(route_tasks)
        out_cabin_tasks.extend(cabin_tasks)

    return {
        "households": out_households,
        "memories": out_memories,
        "histories": out_histories,
        "route_tasks": out_route_tasks,
        "cabin_tasks": out_cabin_tasks,
    }


if __name__ == "__main__":
    data = generate_structure()
    print("households:", len(data["households"]))
    print("memories:", len(data["memories"]))
    base_events = sum(
        1 for h in data["histories"]
        for e in h["events"]
        if e["event_type"] in ("establishment", "evolution"))
    print("base_events:", base_events)
    print("route_tasks:", len(data["route_tasks"]))
    print("cabin_tasks:", len(data["cabin_tasks"]))
