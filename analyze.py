#!/usr/bin/env python3
"""Analyze collected cap breaker data to discover the formula."""
import numpy as np
from scipy import stats
from collections import defaultdict
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
                "pos_idx": pos_idx, "position": b.position,
                "height": b.height, "weight": b.weight, "wingspan": b.wingspan,
                "chosen": chosen, "builder_cap": cap,
                "gap": gap,
                "gap_pct": gap / max(cap, 1),
                "chosen_pct": chosen / max(cap, 1),
                "total_gain": total_gain,
                "recovery_ratio": total_gain / gap if gap > 0 else 1.0,
                "final_value": steps[-1],
                "gains": gains, "steps": steps,
                "num_steps": len(steps),
                "full_recovery": abs(total_gain - gap) < 1,
            })
    return rows


def correlation_analysis(rows):
    """Print correlation between features and total CB gain."""
    if len(rows) < 3:
        print("Need at least 3 data points with cap breaker data.")
        return

    features = ["chosen", "builder_cap", "gap", "gap_pct", "chosen_pct",
                 "height", "weight", "wingspan", "pos_idx", "cat_idx", "num_steps"]
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
        pr, _ = stats.pearsonr(np.array(chosens), gains_arr)
        print(f"    Correlation with chosen value: r={pr:.4f}")


def step_count_analysis(rows):
    """Analyze what determines the number of cap breaker steps."""
    print("\n" + "=" * 60)
    print("STEP COUNT ANALYSIS")
    print("=" * 60)
    print("What determines how many CB steps an attribute gets?\n")

    by_steps = defaultdict(list)
    for r in rows:
        by_steps[r["num_steps"]].append(r)

    for n in sorted(by_steps.keys()):
        group = by_steps[n]
        caps = [r["builder_cap"] for r in group]
        gaps = [r["gap"] for r in group]
        chosens = [r["chosen"] for r in group]
        recoveries = [r["recovery_ratio"] for r in group]
        full = sum(1 for r in group if r["full_recovery"])
        print(f"  {n} steps: n={len(group)}, full_recovery={full}/{len(group)}")
        print(f"    Cap range: {min(caps)}-{max(caps)}, mean={np.mean(caps):.0f}")
        print(f"    Gap range: {min(gaps)}-{max(gaps)}, mean={np.mean(gaps):.0f}")
        print(f"    Chosen range: {min(chosens)}-{max(chosens)}, mean={np.mean(chosens):.0f}")
        print(f"    Avg recovery: {np.mean(recoveries):.2f}")


def position_expectation_analysis(rows):
    """Analyze how 'position expectation' affects CB recovery.

    Theory: if a position is expected to have an attribute (high builder cap),
    leaving it low results in poor CB recovery. Unexpected attributes get
    generous recovery.
    """
    print("\n" + "=" * 60)
    print("POSITION EXPECTATION ANALYSIS")
    print("=" * 60)
    print("Theory: high cap = game expects you to invest there.")
    print("Leaving expected attributes low = poor CB recovery.\n")

    # Only look at 5-step attributes with meaningful gaps
    five_step = [r for r in rows if r["num_steps"] == 5 and r["gap"] >= 8]
    if len(five_step) < 5:
        print("Not enough 5-step data points.")
        return

    # Bucket by builder cap ranges
    print("--- Recovery ratio by builder cap range (5-step, gap >= 8) ---")
    buckets = [(60, 79, "Low cap (60-79)"),
               (80, 89, "Mid cap (80-89)"),
               (90, 95, "High cap (90-95)"),
               (96, 99, "Very high cap (96-99)")]

    for lo, hi, label in buckets:
        group = [r for r in five_step if lo <= r["builder_cap"] <= hi]
        if not group:
            continue
        ratios = [r["recovery_ratio"] for r in group]
        full = sum(1 for r in group if r["full_recovery"])
        print(f"\n  {label}: n={len(group)}, full={full}/{len(group)}")
        print(f"    Avg recovery: {np.mean(ratios):.2f}")
        print(f"    Recovery range: {min(ratios):.2f} - {max(ratios):.2f}")

    # Compare same attribute across different builds
    print("\n--- Same attribute, different builds (5-step, gap >= 8) ---")
    by_attr = defaultdict(list)
    for r in five_step:
        by_attr[r["attr"]].append(r)

    for attr in sorted(by_attr.keys()):
        group = by_attr[attr]
        if len(group) < 2:
            continue
        print(f"\n  {attr}:")
        for r in sorted(group, key=lambda x: x["builder_cap"]):
            print(f"    Build #{r['build_id']:>2} ({r['position']} {r['height']}in): "
                  f"cap={r['builder_cap']}, chosen={r['chosen']}, gap={r['gap']}, "
                  f"gain={r['total_gain']}, ratio={r['recovery_ratio']:.2f}")

    # Correlation: builder_cap vs recovery_ratio for 5-step attributes
    caps = np.array([r["builder_cap"] for r in five_step])
    ratios = np.array([r["recovery_ratio"] for r in five_step])
    gaps = np.array([r["gap"] for r in five_step])
    chosens = np.array([r["chosen"] for r in five_step])
    gap_pcts = np.array([r["gap_pct"] for r in five_step])

    print("\n--- Correlation with recovery ratio (5-step, gap >= 8) ---")
    for name, vals in [("builder_cap", caps), ("gap", gaps), ("chosen", chosens), ("gap_pct", gap_pcts)]:
        pr, pp = stats.pearsonr(vals, ratios)
        print(f"  {name:<15}: r={pr:.4f}, p={pp:.4f}")


def full_recovery_analysis(rows):
    """Analyze when full gap recovery occurs vs partial."""
    print("\n" + "=" * 60)
    print("FULL vs PARTIAL RECOVERY ANALYSIS")
    print("=" * 60)

    non_maxed = [r for r in rows if r["gap"] > 0]
    full = [r for r in non_maxed if r["full_recovery"]]
    partial = [r for r in non_maxed if not r["full_recovery"]]

    print(f"\n  Total: {len(non_maxed)} attributes with gap > 0")
    print(f"  Full recovery: {len(full)} ({len(full)/len(non_maxed)*100:.0f}%)")
    print(f"  Partial recovery: {len(partial)} ({len(partial)/len(non_maxed)*100:.0f}%)")

    if full:
        print(f"\n  FULL RECOVERY attributes:")
        print(f"    Gap range: {min(r['gap'] for r in full)}-{max(r['gap'] for r in full)}")
        print(f"    Cap range: {min(r['builder_cap'] for r in full)}-{max(r['builder_cap'] for r in full)}")
        print(f"    Chosen range: {min(r['chosen'] for r in full)}-{max(r['chosen'] for r in full)}")
        print(f"    Steps range: {min(r['num_steps'] for r in full)}-{max(r['num_steps'] for r in full)}")

    if partial:
        print(f"\n  PARTIAL RECOVERY attributes:")
        print(f"    Gap range: {min(r['gap'] for r in partial)}-{max(r['gap'] for r in partial)}")
        print(f"    Cap range: {min(r['builder_cap'] for r in partial)}-{max(r['builder_cap'] for r in partial)}")
        print(f"    Chosen range: {min(r['chosen'] for r in partial)}-{max(r['chosen'] for r in partial)}")
        print(f"    Steps range: {min(r['num_steps'] for r in partial)}-{max(r['num_steps'] for r in partial)}")

    # Find the boundary
    print("\n  --- Boundary analysis (sorted by gap) ---")
    print(f"  {'Attr':<20} {'Pos':<4} {'Cap':>4} {'Chosen':>7} {'Gap':>5} {'Gain':>6} {'Steps':>6} {'Full':>5}")
    print("  " + "-" * 60)
    for r in sorted(non_maxed, key=lambda x: x["gap"]):
        f = "YES" if r["full_recovery"] else "NO"
        print(f"  {r['attr']:<20} {r['position']:<4} {r['builder_cap']:>4} "
              f"{r['chosen']:>7} {r['gap']:>5} {r['total_gain']:>6} "
              f"{r['num_steps']:>6} {f:>5}")


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
    pos_counts = {}
    for r in rows:
        p = r["position"]
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
    step_count_analysis(rows)
    position_expectation_analysis(rows)
    full_recovery_analysis(rows)


if __name__ == "__main__":
    run_analysis()
