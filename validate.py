#!/usr/bin/env python3
"""Validate discovered formula and suggest next test builds."""
import numpy as np
from scipy import stats
from constants import ALL_ATTRIBUTES, POSITIONS, POSITION_RANGES, POSITION_MEDIANS
from models import load_all_builds
from analyze import extract_features


def validate(test_ratio=0.3):
    builds = load_all_builds()
    rows = extract_features(builds)
    if len(rows) < 10:
        print(f"Only {len(rows)} data points — need at least 10 for validation.")
        print("Collect more data first.")
        return

    np.random.seed(42)
    indices = np.random.permutation(len(rows))
    split = int(len(rows) * (1 - test_ratio))
    train_idx, test_idx = indices[:split], indices[split:]
    train = [rows[i] for i in train_idx]
    test = [rows[i] for i in test_idx]

    # Fit on training data using best features
    features = ["chosen", "builder_cap", "gap", "height", "weight", "wingspan"]
    X_train = np.column_stack([[r[f] for r in train] for f in features]).T
    if X_train.ndim == 1:
        X_train = X_train.reshape(-1, 1)
    y_train = np.array([r["total_gain"] for r in train])

    X_train_c = np.column_stack([np.ones(len(y_train)), X_train])
    coeffs, _, _, _ = np.linalg.lstsq(X_train_c, y_train, rcond=None)

    # Predict on test
    X_test = np.column_stack([[r[f] for r in test] for f in features]).T
    if X_test.ndim == 1:
        X_test = X_test.reshape(-1, 1)
    y_test = np.array([r["total_gain"] for r in test])
    X_test_c = np.column_stack([np.ones(len(y_test)), X_test])
    y_pred = X_test_c @ coeffs

    # Metrics
    ss_res = np.sum((y_test - y_pred) ** 2)
    ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    mae = np.mean(np.abs(y_test - y_pred))
    rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))

    print("=" * 60)
    print("FORMULA VALIDATION (Train/Test Split)")
    print("=" * 60)
    print(f"  Training samples: {len(train)}")
    print(f"  Test samples:     {len(test)}")
    print(f"  R² (test):        {r2:.4f}")
    print(f"  MAE:              {mae:.2f}")
    print(f"  RMSE:             {rmse:.2f}")

    print(f"\n  Formula: total_gain = {coeffs[0]:.2f}", end="")
    for i, f in enumerate(features):
        sign = "+" if coeffs[i+1] >= 0 else ""
        print(f" {sign}{coeffs[i+1]:.4f}*{f}", end="")
    print()

    # Show worst predictions
    errors = np.abs(y_test - y_pred)
    worst = np.argsort(errors)[-min(5, len(errors)):][::-1]
    print("\n  Worst predictions:")
    for idx in worst:
        r = test[idx]
        print(f"    Build #{r['build_id']} {r['attr']}: "
              f"actual={y_test[idx]:.0f}, predicted={y_pred[idx]:.1f}, "
              f"error={errors[idx]:.1f}")

    # Suggest next builds based on gaps in coverage
    suggest_next_builds(rows)

    return r2


def suggest_next_builds(rows):
    """Suggest builds that would fill gaps in the dataset."""
    print("\n" + "=" * 60)
    print("SUGGESTED NEXT BUILDS")
    print("=" * 60)

    # Check position coverage
    pos_seen = set(POSITIONS.index(r["attr"]) if r["attr"] in POSITIONS else r["pos_idx"] for r in rows)
    positions_covered = set()
    for r in rows:
        positions_covered.add(r["pos_idx"])

    missing_pos = [POSITIONS[i] for i in range(len(POSITIONS)) if i not in positions_covered]
    if missing_pos:
        print(f"\n  Missing positions: {', '.join(missing_pos)}")
        for pos in missing_pos:
            m = POSITION_MEDIANS[pos]
            print(f"    → Create baseline {pos} at {m['height']}h/{m['weight']}w/{m['wingspan']}ws, all attrs 25")

    # Check chosen value range coverage
    chosens = [r["chosen"] for r in rows]
    if chosens:
        mn, mx = min(chosens), max(chosens)
        if mn > 35:
            print(f"\n  No data for very low chosen values (<35). Current min: {mn}")
            print("    → Create a build with attributes at 25-30")
        if mx < 80:
            print(f"\n  No data for high chosen values (>80). Current max: {mx}")
            print("    → Create a build with some attributes at 80+")

    # Check physical extremes
    heights = set(r["height"] for r in rows)
    weights = set(r["weight"] for r in rows)
    if len(heights) < 3:
        print(f"\n  Only {len(heights)} unique heights tested. Try more height variation.")
    if len(weights) < 3:
        print(f"\n  Only {len(weights)} unique weights tested. Try more weight variation.")

    if not missing_pos and len(heights) >= 3 and len(weights) >= 3:
        print("\n  ✅ Good coverage! Focus on builds where predictions are worst.")


if __name__ == "__main__":
    validate()
