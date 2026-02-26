#!/usr/bin/env python3
"""Analyze collected cap breaker data to discover the formula."""
import numpy as np
from scipy import stats
from itertools import combinations
from constants import ALL_ATTRIBUTES, ATTRIBUTES, POSITIONS
from models import load_all_builds


def extract_features(builds):
    """Extract feature rows from builds. One row per (build, attribute) pair that has CB data."""
    rows = []
    for b in builds:
        pos_idx = POSITIONS.index(b.position) if b.position in POSITIONS else 0
        for attr in ALL_ATTRIBUTES:
            steps = b.cb_steps.get(attr, [])
            if not steps or all(s == 0 for s in steps):
                continue
            # Find which category this attribute belongs to
            cat_idx = 0
            for i, (cat, attrs) in enumerate(ATTRIBUTES.items()):
                if attr in attrs:
                    cat_idx = i
                    break

            chosen = b.chosen[attr]
            cap = b.builder_caps[attr]
            gap = cap - chosen
            total_gain = steps[-1] - chosen
            gains = b.cb_gains(attr)

            rows.append({
                "build_id": b.id, "attr": attr, "cat_idx": cat_idx,
                "pos_idx": pos_idx, "height": b.height, "weight": b.weight,
                "wingspan": b.wingspan, "chosen": chosen, "builder_cap": cap,
                "gap": gap, "gap_pct": gap / max(cap, 1),
                "chosen_pct": chosen / max(cap, 1),
                "total_gain": total_gain, "final_value": steps[-1],
                "gains": gains, "steps": steps,
                "num_steps": len(steps),
            })
    return rows


def correlation_analysis(rows):
    """Print correlation between features and total CB gain."""
    if len(rows) < 3:
        print("Need at least 3 data points with cap breaker data.")
        return

    features = ["chosen", "builder_cap", "gap", "gap_pct", "chosen_pct",
                 "height", "weight", "wingspan", "pos_idx", "cat_idx"]
    target = np.array([r["total_gain"] for r in rows])

    print("\n" + "=" * 60)
    print("CORRELATION ANALYSIS: Feature vs Total CB Gain")
    print("=" * 60)
    print(f"{'Feature':<15} {'Pearson r':>10} {'p-value':>10} {'Spearman':>10}")
    print("-" * 50)

    for feat in features:
        vals = np.array([r[feat] for r in rows])
        if np.std(vals) == 0:
            continue
        pr, pp = stats.pearsonr(vals, target)
        sr, sp = stats.spearmanr(vals, target)
        marker = " ***" if abs(pr) > 0.7 else " **" if abs(pr) > 0.5 else ""
        print(f"{feat:<15} {pr:>10.4f} {pp:>10.4f} {sr:>10.4f}{marker}")

    print("\n*** = strong correlation (|r| > 0.7)")
    print("**  = moderate correlation (|r| > 0.5)")


def regression_analysis(rows):
    """Try various regression models and report R² scores."""
    if len(rows) < 5:
        print("\nNeed at least 5 data points for regression.")
        return

    target = np.array([r["total_gain"] for r in rows])
    all_features = {
        "chosen": np.array([r["chosen"] for r in rows]),
        "builder_cap": np.array([r["builder_cap"] for r in rows]),
        "gap": np.array([r["gap"] for r in rows]),
        "gap_pct": np.array([r["gap_pct"] for r in rows]),
        "height": np.array([r["height"] for r in rows]),
        "weight": np.array([r["weight"] for r in rows]),
        "wingspan": np.array([r["wingspan"] for r in rows]),
    }

    print("\n" + "=" * 60)
    print("REGRESSION ANALYSIS")
    print("=" * 60)

    # Single-feature linear regressions
    print("\n--- Single Feature Linear Regression ---")
    print(f"{'Feature':<15} {'R²':>8} {'Slope':>10} {'Intercept':>10}")
    print("-" * 48)
    best_r2 = -1
    best_feat = ""
    for name, vals in all_features.items():
        if np.std(vals) == 0:
            continue
        slope, intercept, r, p, se = stats.linregress(vals, target)
        r2 = r ** 2
        print(f"{name:<15} {r2:>8.4f} {slope:>10.4f} {intercept:>10.4f}")
        if r2 > best_r2:
            best_r2, best_feat = r2, name

    # Multi-feature regression
    print("\n--- Multi-Feature Linear Regression ---")
    feat_names = list(all_features.keys())
    X = np.column_stack([all_features[f] for f in feat_names])
    X_with_const = np.column_stack([np.ones(len(target)), X])

    try:
        coeffs, residuals, rank, sv = np.linalg.lstsq(X_with_const, target, rcond=None)
        predicted = X_with_const @ coeffs
        ss_res = np.sum((target - predicted) ** 2)
        ss_tot = np.sum((target - np.mean(target)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        print(f"R² = {r2:.4f}")
        print(f"{'Feature':<15} {'Coefficient':>12}")
        print("-" * 30)
        print(f"{'intercept':<15} {coeffs[0]:>12.4f}")
        for i, name in enumerate(feat_names):
            print(f"{name:<15} {coeffs[i+1]:>12.4f}")
    except Exception as e:
        print(f"Multi-feature regression failed: {e}")

    # Polynomial on best single feature
    if best_feat:
        vals = all_features[best_feat]
        print(f"\n--- Polynomial Regression on '{best_feat}' ---")
        for degree in [2, 3]:
            coeffs = np.polyfit(vals, target, degree)
            predicted = np.polyval(coeffs, vals)
            ss_res = np.sum((target - predicted) ** 2)
            ss_tot = np.sum((target - np.mean(target)) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
            poly_str = " + ".join(
                f"{c:.4f}*x^{degree-i}" if degree-i > 0 else f"{c:.4f}"
                for i, c in enumerate(coeffs)
            )
            print(f"  Degree {degree}: R² = {r2:.4f}")
            print(f"    {poly_str}")


def per_step_analysis(rows):
    """Analyze how each individual cap breaker step behaves."""
    if len(rows) < 3:
        return

    print("\n" + "=" * 60)
    print("PER-STEP ANALYSIS")
    print("=" * 60)

    for step_num in range(5):
        gains_at_step = []
        chosens = []
        for r in rows:
            if len(r["gains"]) > step_num:
                gains_at_step.append(r["gains"][step_num])
                chosens.append(r["chosen"])
        if len(gains_at_step) < 3:
            continue
        gains_arr = np.array(gains_at_step)
        print(f"\n  Step {step_num+1}: n={len(gains_arr)}, "
              f"mean={gains_arr.mean():.2f}, std={gains_arr.std():.2f}, "
              f"min={gains_arr.min()}, max={gains_arr.max()}")
        # Correlation with chosen value
        pr, _ = stats.pearsonr(np.array(chosens), gains_arr)
        print(f"    Correlation with chosen value: r={pr:.4f}")


def summary(rows):
    """Print a high-level summary of the dataset."""
    print("\n" + "=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)
    build_ids = set(r["build_id"] for r in rows)
    print(f"  Builds with CB data: {len(build_ids)}")
    print(f"  Total attribute data points: {len(rows)}")
    gains = [r["total_gain"] for r in rows]
    if gains:
        print(f"  Total gain range: {min(gains)} to {max(gains)}")
        print(f"  Mean total gain: {np.mean(gains):.1f}")
    # Coverage by position
    pos_counts = {}
    for r in rows:
        p = r["pos_idx"]
        pos_counts[p] = pos_counts.get(p, 0) + 1
    print(f"  Data points by position: {dict(sorted(pos_counts.items()))}")


def run_analysis():
    builds = load_all_builds()
    if not builds:
        print("No build data found. Run 'python3 entry.py' to enter data first.")
        return

    rows = extract_features(builds)
    if not rows:
        print("No cap breaker data found in builds. Make sure cb_steps are entered.")
        return

    summary(rows)
    correlation_analysis(rows)
    regression_analysis(rows)
    per_step_analysis(rows)


if __name__ == "__main__":
    run_analysis()
