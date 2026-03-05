#!/usr/bin/env python3
"""Storage and analysis for builder caps, OVR formula, and skill linkages."""
import json
import os
import numpy as np
from constants import POSITIONS, ATTRIBUTES, ALL_ATTRIBUTES, POSITION_RANGES, inches_to_height_str

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
CAPS_PATH = os.path.join(DATA_DIR, "builder_caps.json")
OVR_PATH = os.path.join(DATA_DIR, "ovr_tests.json")
LINKS_PATH = os.path.join(DATA_DIR, "skill_links.json")


def _ensure():
    os.makedirs(DATA_DIR, exist_ok=True)


# ============================================
# BUILDER CAPS: max attributes by body type
# ============================================

def load_caps_data():
    if os.path.exists(CAPS_PATH):
        with open(CAPS_PATH) as f:
            return json.load(f)
    return []


def save_caps_entry(position, height, weight, wingspan, caps_dict):
    """Save builder caps for a body type."""
    _ensure()
    data = load_caps_data()
    entry = {
        "position": position, "height": height, "weight": weight,
        "wingspan": wingspan, "caps": caps_dict,
    }
    # Replace if same body exists
    data = [d for d in data if not (d["position"] == position and d["height"] == height
            and d["weight"] == weight and d["wingspan"] == wingspan)]
    data.append(entry)
    with open(CAPS_PATH, "w") as f:
        json.dump(data, f, indent=2)
    return entry


def enter_caps():
    """CLI to enter builder caps for a body type."""
    pos = input("Position (PG/SG/SF/PF/C): ").strip().upper()
    height = int(input("Height (inches): "))
    weight = int(input("Weight (lbs): "))
    wingspan = int(input("Wingspan (inches): "))

    print(f"\n{pos} | {inches_to_height_str(height)} | {weight}lbs | {inches_to_height_str(wingspan)} ws")
    print("Enter all 21 max attribute values in order, space-separated:")
    print(f"({', '.join(ALL_ATTRIBUTES)})\n")

    vals = input("Caps: ").strip().split()
    while len(vals) != len(ALL_ATTRIBUTES):
        print(f"Need {len(ALL_ATTRIBUTES)} values, got {len(vals)}")
        vals = input("Caps: ").strip().split()

    caps = {attr: int(v) for attr, v in zip(ALL_ATTRIBUTES, vals)}
    save_caps_entry(pos, height, weight, wingspan, caps)
    print(f"Saved caps for {pos} {inches_to_height_str(height)} {weight}lbs")


# ============================================
# OVR TESTS: what OVR do you get for given attributes
# ============================================

def load_ovr_data():
    if os.path.exists(OVR_PATH):
        with open(OVR_PATH) as f:
            return json.load(f)
    return []


def save_ovr_entry(position, height, weight, wingspan, chosen_dict, ovr):
    _ensure()
    data = load_ovr_data()
    data.append({
        "position": position, "height": height, "weight": weight,
        "wingspan": wingspan, "chosen": chosen_dict, "ovr": ovr,
    })
    with open(OVR_PATH, "w") as f:
        json.dump(data, f, indent=2)


# ============================================
# SKILL LINKS: what minimums are forced
# ============================================

def load_links_data():
    if os.path.exists(LINKS_PATH):
        with open(LINKS_PATH) as f:
            return json.load(f)
    return []


def save_link_entry(position, height, weight, wingspan, trigger_attr, trigger_val, forced_mins):
    """Save a skill linkage observation.
    
    trigger_attr: the attribute you raised
    trigger_val: what you set it to
    forced_mins: dict of {attr: min_value} for all attributes that got forced up
    """
    _ensure()
    data = load_links_data()
    data.append({
        "position": position, "height": height, "weight": weight,
        "wingspan": wingspan, "trigger_attr": trigger_attr,
        "trigger_val": trigger_val, "forced_mins": forced_mins,
    })
    with open(LINKS_PATH, "w") as f:
        json.dump(data, f, indent=2)


# ============================================
# ANALYSIS
# ============================================

def analyze_caps():
    """Analyze builder caps data to find patterns."""
    data = load_caps_data()
    if not data:
        print("No caps data. Run enter_caps() first.")
        return

    print(f"\n{'=' * 60}")
    print(f"BUILDER CAPS ANALYSIS ({len(data)} body types)")
    print(f"{'=' * 60}")

    # Group by position
    by_pos = {}
    for d in data:
        by_pos.setdefault(d["position"], []).append(d)

    for pos in POSITIONS:
        entries = by_pos.get(pos, [])
        if len(entries) < 2:
            continue

        print(f"\n--- {pos} ({len(entries)} body types) ---")
        # Sort by height
        entries.sort(key=lambda x: x["height"])

        for attr in ALL_ATTRIBUTES:
            caps = [(e["height"], e["weight"], e["wingspan"], e["caps"][attr]) for e in entries]
            cap_vals = [c[3] for c in caps]
            if max(cap_vals) == min(cap_vals):
                continue  # Same across all body types

            print(f"\n  {attr}:")
            for h, w, ws, cap in caps:
                print(f"    {inches_to_height_str(h)} {w}lbs {inches_to_height_str(ws)}ws: {cap}")

            # If we have 3+ points, try linear regression on height
            if len(caps) >= 3:
                heights = np.array([c[0] for c in caps])
                cap_arr = np.array([c[3] for c in caps])
                from scipy import stats
                slope, intercept, r, p, se = stats.linregress(heights, cap_arr)
                print(f"    Height effect: {slope:+.1f} per inch (R²={r**2:.2f})")


def analyze_links():
    """Analyze skill linkage data."""
    data = load_links_data()
    if not data:
        print("No linkage data.")
        return

    print(f"\n{'=' * 60}")
    print(f"SKILL LINKAGE ANALYSIS ({len(data)} observations)")
    print(f"{'=' * 60}")

    # Group by trigger attribute
    by_trigger = {}
    for d in data:
        by_trigger.setdefault(d["trigger_attr"], []).append(d)

    for trigger, entries in sorted(by_trigger.items()):
        print(f"\n  When {trigger} is raised:")
        # Find consistent forced attributes
        all_forced = {}
        for e in entries:
            for attr, val in e["forced_mins"].items():
                if attr != trigger and val > 25:
                    all_forced.setdefault(attr, []).append({
                        "min": val, "trigger_val": e["trigger_val"],
                        "pos": e["position"],
                    })

        for attr, observations in sorted(all_forced.items(), key=lambda x: -max(o["min"] for o in x[1])):
            vals = [o["min"] for o in observations]
            triggers = [o["trigger_val"] for o in observations]
            if len(observations) == 1:
                o = observations[0]
                pct = o["min"] / o["trigger_val"] * 100 if o["trigger_val"] > 0 else 0
                print(f"    → {attr}: min={o['min']} ({pct:.0f}% of trigger {o['trigger_val']})")
            else:
                print(f"    → {attr}: mins={vals} (triggers={triggers})")


def predict_caps(position, height, weight, wingspan):
    """Predict builder caps for any body type."""
    model_path = os.path.join(DATA_DIR, "caps_model.json")
    with open(model_path) as f:
        model = json.load(f)
    g = model['pos_groups'][position]
    pg, sg, fwc = int(g == 0), int(g == 1), int(g == 2)
    feat = [pg, sg, fwc, pg*height, sg*height, fwc*height,
            pg*height**2, sg*height**2, fwc*height**2, weight, wingspan]
    caps = {}
    for attr in ALL_ATTRIBUTES:
        w = model['coefficients'][attr]
        val = sum(f * c for f, c in zip(feat, w))
        caps[attr] = int(round(max(25, min(99, val))))
    return caps


if __name__ == "__main__":
    print("Builder Data Collection Tool")
    print("1. Enter builder caps (max attributes)")
    print("2. Analyze caps data")
    print("3. Analyze skill links")
    choice = input("Choice: ").strip()
    if choice == "1":
        while True:
            enter_caps()
            if input("Another? (y/n): ").strip().lower() != "y":
                break
    elif choice == "2":
        analyze_caps()
    elif choice == "3":
        analyze_links()
