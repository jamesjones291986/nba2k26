#!/usr/bin/env python3
"""Generate systematic test builds to isolate cap breaker variables."""
from constants import POSITIONS, ATTRIBUTES, POSITION_RANGES, POSITION_MEDIANS, inches_to_height_str

tests = []


def add(name, pos, height, weight, wingspan, notes="All attributes at minimum (25)"):
    tests.append({"name": name, "pos": pos, "height": height, "weight": weight,
                   "wingspan": wingspan, "notes": notes})


# --- Phase 1: Baselines (5 builds) ---
# One per position at median physicals, all attributes at 25
for pos in POSITIONS:
    m = POSITION_MEDIANS[pos]
    add(f"Baseline {pos}", pos, m["height"], m["weight"], m["wingspan"])

# --- Phase 2: Height isolation on PG (3 builds, 1 overlaps baseline) ---
r = POSITION_RANGES["PG"]
m = POSITION_MEDIANS["PG"]
for label, h in [("min", r["height"][0]), ("max", r["height"][1])]:
    add(f"PG Height {label}", "PG", h, m["weight"], m["wingspan"])

# --- Phase 3: Weight isolation on PG (2 builds) ---
for label, w in [("min", r["weight"][0]), ("max", r["weight"][1])]:
    add(f"PG Weight {label}", "PG", m["height"], w, m["wingspan"])

# --- Phase 4: Wingspan isolation on PG (2 builds) ---
for label, ws in [("min", r["wingspan"][0]), ("max", r["wingspan"][1])]:
    add(f"PG Wingspan {label}", "PG", m["height"], m["weight"], ws)

# --- Phase 5: Attribute allocation isolation on PG median (4 builds) ---
m = POSITION_MEDIANS["PG"]
alloc_tests = [
    ("All Shooting", "Max out all Shooting attributes first, rest at 25"),
    ("All Finishing", "Max out all Finishing attributes first, rest at 25"),
    ("All Playmaking", "Max out all Playmaking attributes first, rest at 25"),
    ("Balanced", "Spread points evenly across all categories to 99 OVR"),
]
for label, notes in alloc_tests:
    add(f"PG Alloc: {label}", "PG", m["height"], m["weight"], m["wingspan"], notes)

# --- Phase 6: Single attribute sweep on PG median (4 builds) ---
# Vary Three-Point Shot specifically while keeping everything else at 25
for val in [40, 55, 70, 85]:
    add(f"PG 3PT={val}", "PG", m["height"], m["weight"], m["wingspan"],
        f"Set Three-Point Shot to {val}, all other attributes at 25")

# --- Phase 7: Cross-position height extremes (2 builds) ---
for pos in ["SF", "C"]:
    r2 = POSITION_RANGES[pos]
    m2 = POSITION_MEDIANS[pos]
    add(f"{pos} Height min", pos, r2["height"][0], m2["weight"], m2["wingspan"])


def print_test_plan():
    print("=" * 70)
    print("NBA 2K26 CAP BREAKER TEST PLAN")
    print(f"Total test builds: {len(tests)}")
    print("=" * 70)

    phase_names = {
        "Baseline": "Phase 1: Baselines — one per position, all attrs at 25",
        "Height": "Phase 2-3: Physical isolation — vary one dimension",
        "Weight": "Phase 2-3: Physical isolation — vary one dimension",
        "Wingspan": "Phase 4: Wingspan isolation",
        "Alloc": "Phase 5: Attribute allocation patterns",
        "3PT": "Phase 6: Single attribute sweep (Three-Point Shot)",
    }
    last_phase = ""
    for i, t in enumerate(tests, 1):
        # Detect phase change
        for key, label in phase_names.items():
            if key in t["name"] and label != last_phase:
                last_phase = label
                print(f"\n{'─' * 70}")
                print(f"  {label}")
                print(f"{'─' * 70}")
                break

        ht = inches_to_height_str(t["height"])
        ws = inches_to_height_str(t["wingspan"])
        print(f"\n  Build #{i}: {t['name']}")
        print(f"    Position: {t['pos']}  |  Height: {ht}  |  Weight: {t['weight']}lbs  |  Wingspan: {ws}")
        print(f"    Instructions: {t['notes']}")

    print(f"\n{'=' * 70}")
    print("WORKFLOW: Create each build in-game → check cap breaker screen →")
    print("          run 'python3 entry.py' to record the data → delete build → next")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    print_test_plan()
