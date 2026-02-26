#!/usr/bin/env python3
"""Optimize builds to hit target attributes with minimal cap breakers."""
import json
import os
import numpy as np
from itertools import product
from constants import (POSITIONS, ATTRIBUTES, ALL_ATTRIBUTES, POSITION_RANGES,
                       POSITION_MEDIANS, inches_to_height_str)
from models import load_all_builds
from analyze import extract_features

FORMULA_PATH = os.path.join(os.path.dirname(__file__), "data", "formula.json")


def fit_formula():
    """Fit the best formula from all collected data and save it."""
    builds = load_all_builds()
    rows = extract_features(builds)
    if len(rows) < 5:
        print("Not enough data to fit formula. Need at least 5 attribute data points.")
        return None

    features = ["chosen", "builder_cap", "gap", "height", "weight", "wingspan"]
    X = np.column_stack([[r[f] for r in rows] for f in features]).T
    X_c = np.column_stack([np.ones(len(rows)), X])
    y = np.array([r["total_gain"] for r in rows])

    coeffs, _, _, _ = np.linalg.lstsq(X_c, y, rcond=None)

    # Also fit per-step models
    step_coeffs = []
    for step in range(5):
        step_gains = []
        step_X = []
        for r in rows:
            if len(r["gains"]) > step:
                step_gains.append(r["gains"][step])
                step_X.append([r[f] for f in features])
        if len(step_gains) >= 5:
            sX = np.column_stack([np.ones(len(step_gains)), np.array(step_X)])
            sc, _, _, _ = np.linalg.lstsq(sX, np.array(step_gains), rcond=None)
            step_coeffs.append(sc.tolist())
        else:
            step_coeffs.append(None)

    formula = {
        "features": features,
        "total_gain_coeffs": coeffs.tolist(),
        "step_coeffs": step_coeffs,
        "n_samples": len(rows),
    }

    os.makedirs(os.path.dirname(FORMULA_PATH), exist_ok=True)
    with open(FORMULA_PATH, "w") as f:
        json.dump(formula, f, indent=2)
    print(f"Formula saved to {FORMULA_PATH} (fitted on {len(rows)} samples)")
    return formula


def load_formula():
    if not os.path.exists(FORMULA_PATH):
        print("No formula found. Fitting from data...")
        return fit_formula()
    with open(FORMULA_PATH) as f:
        return json.load(f)


def predict_total_gain(formula, chosen, builder_cap, gap, height, weight, wingspan):
    """Predict total cap breaker gain for an attribute."""
    coeffs = formula["total_gain_coeffs"]
    x = [1, chosen, builder_cap, gap, height, weight, wingspan]
    return max(0, sum(c * v for c, v in zip(coeffs, x)))


def predict_steps(formula, chosen, builder_cap, gap, height, weight, wingspan):
    """Predict per-step gains."""
    steps = []
    current = chosen
    for sc in formula.get("step_coeffs", []):
        if sc is None:
            break
        x = [1, chosen, builder_cap, gap, height, weight, wingspan]
        gain = max(0, round(sum(c * v for c, v in zip(sc, x))))
        current += gain
        steps.append(current)
    return steps


def count_breakers_needed(formula, chosen, builder_cap, height, weight, wingspan, target):
    """How many cap breakers needed to reach target from chosen value."""
    if chosen >= target:
        return 0
    gap = builder_cap - chosen
    steps = predict_steps(formula, chosen, builder_cap, gap, height, weight, wingspan)
    if not steps:
        # Fallback: use total gain / 5
        total = predict_total_gain(formula, chosen, builder_cap, gap, height, weight, wingspan)
        per_step = total / 5 if total > 0 else 1
        needed = int(np.ceil((target - chosen) / per_step))
        return min(needed, 5)
    for i, val in enumerate(steps):
        if val >= target:
            return i + 1
    return 6  # Can't reach it with 5 breakers


def optimize(targets, max_breakers=10, position=None):
    """Find builds that hit all targets with minimal cap breakers.

    targets: dict of {attribute_name: minimum_value}
    max_breakers: total cap breakers available
    position: optional position constraint
    """
    formula = load_formula()
    if not formula:
        return []

    positions = [position] if position else POSITIONS
    results = []

    for pos in positions:
        r = POSITION_RANGES[pos]
        # Sample physical combinations (step by reasonable increments)
        heights = range(r["height"][0], r["height"][1] + 1, 2)
        weights = range(r["weight"][0], r["weight"][1] + 1, 10)
        wingspans = range(r["wingspan"][0], r["wingspan"][1] + 1, 2)

        for h, w, ws in product(heights, weights, wingspans):
            total_cb = 0
            feasible = True
            breakdown = {}

            for attr, target_val in targets.items():
                # Estimate builder cap (we'd need real data, use heuristic for now)
                # For attributes the build is good at, cap is high; otherwise low
                # This is a rough estimate — improves as more data is collected
                cap = 99  # Placeholder — will be refined with data
                chosen = 25  # Start at minimum

                # The user will set chosen values to reach 99 OVR
                # For now, estimate: if target is high, chosen is probably high too
                # This is the part that gets refined as we learn the formula
                cb_needed = count_breakers_needed(formula, chosen, cap, h, w, ws, target_val)
                if cb_needed > 5:
                    feasible = False
                    break
                total_cb += cb_needed
                breakdown[attr] = {"cb_needed": cb_needed, "chosen": chosen, "target": target_val}

            if feasible and total_cb <= max_breakers:
                results.append({
                    "position": pos, "height": h, "weight": w, "wingspan": ws,
                    "total_cb": total_cb, "breakdown": breakdown,
                })

    results.sort(key=lambda x: x["total_cb"])
    return results[:20]


def interactive_optimize():
    """Interactive CLI for build optimization."""
    formula = load_formula()
    if not formula:
        return

    print("\n" + "=" * 60)
    print("BUILD OPTIMIZER")
    print("=" * 60)
    print(f"Formula fitted on {formula['n_samples']} samples")

    print("\nAvailable attributes:")
    for cat, attrs in ATTRIBUTES.items():
        print(f"  {cat}: {', '.join(attrs)}")

    print("\nEnter target attributes (one per line, format: 'Attribute Name >= value')")
    print("Type 'done' when finished\n")

    targets = {}
    while True:
        line = input("  Target: ").strip()
        if line.lower() == "done":
            break
        try:
            parts = line.split(">=")
            attr = parts[0].strip()
            val = int(parts[1].strip())
            if attr not in ALL_ATTRIBUTES:
                # Fuzzy match
                matches = [a for a in ALL_ATTRIBUTES if attr.lower() in a.lower()]
                if len(matches) == 1:
                    attr = matches[0]
                    print(f"    → Matched to: {attr}")
                else:
                    print(f"    Unknown attribute. Did you mean: {matches}")
                    continue
            targets[attr] = val
            print(f"    ✓ {attr} >= {val}")
        except (IndexError, ValueError):
            print("    Format: Attribute Name >= value")

    if not targets:
        print("No targets set.")
        return

    max_cb = int(input("\nMax cap breakers available (default 10): ").strip() or "10")
    pos_input = input("Position constraint (PG/SG/SF/PF/C or blank for all): ").strip().upper()
    pos = pos_input if pos_input in POSITIONS else None

    print("\nSearching for optimal builds...")
    results = optimize(targets, max_cb, pos)

    if not results:
        print("\n❌ No builds found that meet all targets within cap breaker budget.")
        print("   Try relaxing targets or increasing cap breaker budget.")
        return

    print(f"\n✅ Found {len(results)} viable builds (showing top 10):\n")
    for i, r in enumerate(results[:10], 1):
        ht = inches_to_height_str(r["height"])
        ws = inches_to_height_str(r["wingspan"])
        print(f"  #{i}: {r['position']} | {ht} | {r['weight']}lbs | {ws} ws | "
              f"Cap Breakers: {r['total_cb']}/{max_cb}")
        for attr, info in r["breakdown"].items():
            print(f"       {attr}: {info['cb_needed']} CB → {info['target']}")
        print()


if __name__ == "__main__":
    interactive_optimize()
