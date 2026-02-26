#!/usr/bin/env python3
"""Smarter cap breaker prediction model.

Key insights from data:
1. 1-4 step attributes: gain = gap (always full recovery, hits builder cap)
2. 5-step attributes: gain is throttled based on builder_cap
3. Higher builder_cap = more throttling (game expects you to invest)
4. Per-step gains decrease (step 1 biggest, step 5 smallest)
5. Lower chosen value = bigger per-step gains (but still throttled)
"""
import numpy as np
from scipy.optimize import minimize
from models import load_all_builds
from analyze import extract_features


def predict_num_steps(gap, builder_cap):
    """Predict number of cap breaker steps.
    
    From data: low gap or low cap = fewer steps.
    Steps 1-4 always give full recovery.
    Step 5 is where throttling happens.
    """
    if gap == 0:
        return 0
    # From observed data patterns:
    # The game seems to give enough steps to cover the gap if possible,
    # but caps at 5. Low-cap attributes get fewer steps because
    # fewer steps can already cover the full gap.
    # Rough thresholds from data:
    if gap <= 5 and builder_cap <= 80:
        return 1
    if gap <= 13 and builder_cap <= 65:
        return 2
    if gap <= 25 and builder_cap <= 75:
        return 2
    if gap <= 27 and builder_cap <= 72:
        return 3
    if gap <= 35 and builder_cap <= 65:
        return 3
    if gap <= 26 and builder_cap <= 82:
        return 3
    if gap <= 45 and builder_cap <= 80:
        return 3
    if gap <= 39 and builder_cap <= 68:
        return 4
    if gap <= 38 and builder_cap <= 65:
        return 4
    return 5


def predict_gain_simple(gap, builder_cap, chosen):
    """Simple model: predict total CB gain.
    
    For non-5-step: gain = gap (full recovery)
    For 5-step: gain depends on builder_cap and gap
    """
    steps = predict_num_steps(gap, builder_cap)
    if steps == 0:
        return 0
    if steps < 5:
        return gap  # Full recovery

    # 5-step throttled model
    # From data: recovery_ratio correlates with builder_cap (-0.51)
    # and weakly with gap_pct
    # Use a piecewise approach based on builder_cap
    gap_pct = gap / builder_cap if builder_cap > 0 else 0

    # Base recovery from gap_pct (quadratic fit from data)
    base_gain = 45.4 * gap_pct ** 2 + 29.4 * gap_pct + 2.4

    # Throttle based on builder_cap
    # Higher cap = more throttle
    if builder_cap >= 96:
        throttle = 0.85
    elif builder_cap >= 90:
        throttle = 0.95
    elif builder_cap >= 80:
        throttle = 1.0
    else:
        throttle = 1.05

    gain = base_gain * throttle
    return min(round(gain), gap)  # Can't exceed gap


class CBModel:
    """Fitted cap breaker model using per-step regression."""

    def __init__(self):
        self.step_models = []  # coefficients for each step
        self.fitted = False

    def fit(self, rows):
        """Fit per-step gain models from data."""
        five = [r for r in rows if r['num_steps'] == 5 and r['gap'] > 0]
        if len(five) < 10:
            return

        self.step_models = []
        for step in range(5):
            gains = []
            features = []
            for r in five:
                if len(r['gains']) > step:
                    gains.append(r['gains'][step])
                    features.append([
                        r['gap_pct'],
                        r['gap_pct'] ** 2,
                        r['builder_cap'] / 99,
                        r['chosen'] / 99,
                        (r['builder_cap'] / 99) * r['gap_pct'],  # interaction
                    ])
            if len(gains) < 5:
                break

            X = np.column_stack([np.ones(len(gains)), np.array(features)])
            y = np.array(gains)
            coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
            self.step_models.append(coeffs)

        self.fitted = True

    def predict_steps(self, gap, builder_cap, chosen):
        """Predict per-step gains."""
        if gap == 0:
            return []

        n_steps = predict_num_steps(gap, builder_cap)
        if n_steps < 5:
            # Full recovery, distribute across steps (decreasing)
            if n_steps == 1:
                return [gap]
            # Rough distribution: each step is ~60-80% of previous
            weights = [0.8 ** i for i in range(n_steps)]
            total_w = sum(weights)
            steps = [max(1, round(gap * w / total_w)) for w in weights]
            # Adjust to hit exact gap
            diff = gap - sum(steps)
            for i in range(abs(diff)):
                if diff > 0:
                    steps[i % n_steps] += 1
                elif diff < 0:
                    steps[-(i % n_steps) - 1] = max(1, steps[-(i % n_steps) - 1] - 1)
            return steps

        if not self.fitted:
            # Fallback
            gain = predict_gain_simple(gap, builder_cap, chosen)
            per = max(1, gain // 5)
            return [per] * 5

        # Use fitted per-step models
        gap_pct = gap / builder_cap if builder_cap > 0 else 0
        feats = [
            gap_pct,
            gap_pct ** 2,
            builder_cap / 99,
            chosen / 99,
            (builder_cap / 99) * gap_pct,
        ]
        x = np.array([1] + feats)

        gains = []
        for coeffs in self.step_models:
            g = max(1, round(x @ coeffs))
            gains.append(g)

        # Ensure gains are decreasing
        for i in range(1, len(gains)):
            gains[i] = min(gains[i], gains[i - 1])

        # Cap total at gap
        total = sum(gains)
        if total > gap:
            gains = [max(1, round(g * gap / total)) for g in gains]

        return gains

    def predict_total(self, gap, builder_cap, chosen):
        """Predict total CB gain."""
        return sum(self.predict_steps(gap, builder_cap, chosen))

    def evaluate(self, rows):
        """Evaluate model on data."""
        errors = []
        step_errors = []
        for r in rows:
            if r['gap'] == 0:
                continue
            pred_total = self.predict_total(r['gap'], r['builder_cap'], r['chosen'])
            actual_total = r['total_gain']
            errors.append(abs(pred_total - actual_total))

            # Per-step errors
            pred_steps = self.predict_steps(r['gap'], r['builder_cap'], r['chosen'])
            for i, (pg, ag) in enumerate(zip(pred_steps, r['gains'])):
                step_errors.append(abs(pg - ag))

        mae = np.mean(errors) if errors else 0
        within3 = sum(1 for e in errors if e <= 3) / len(errors) * 100 if errors else 0
        within5 = sum(1 for e in errors if e <= 5) / len(errors) * 100 if errors else 0
        step_mae = np.mean(step_errors) if step_errors else 0

        return {
            'mae': mae, 'within3': within3, 'within5': within5,
            'step_mae': step_mae, 'n': len(errors),
        }


def train_and_evaluate():
    builds = load_all_builds()
    rows = extract_features(builds)
    non_maxed = [r for r in rows if r['gap'] > 0]

    # Train on builds 1-12, test on 13-14
    train = [r for r in non_maxed if r['build_id'] <= 12]
    test = [r for r in non_maxed if r['build_id'] > 12]

    model = CBModel()
    model.fit(train)

    print("=" * 60)
    print("SMART CAP BREAKER MODEL")
    print("=" * 60)

    train_eval = model.evaluate(train)
    test_eval = model.evaluate(test)

    print(f"\nTraining ({train_eval['n']} points):")
    print(f"  MAE: {train_eval['mae']:.1f}")
    print(f"  Within 3: {train_eval['within3']:.0f}%")
    print(f"  Within 5: {train_eval['within5']:.0f}%")
    print(f"  Per-step MAE: {train_eval['step_mae']:.1f}")

    print(f"\nTest ({test_eval['n']} points):")
    print(f"  MAE: {test_eval['mae']:.1f}")
    print(f"  Within 3: {test_eval['within3']:.0f}%")
    print(f"  Within 5: {test_eval['within5']:.0f}%")
    print(f"  Per-step MAE: {test_eval['step_mae']:.1f}")

    # Show test predictions
    print(f"\n{'Attribute':<20} {'Gap':>5} {'Actual':>7} {'Pred':>7} {'Error':>7}")
    print("-" * 50)
    for r in test:
        pred = model.predict_total(r['gap'], r['builder_cap'], r['chosen'])
        err = pred - r['total_gain']
        marker = ' OK' if abs(err) <= 3 else ''
        print(f"{r['attr']:<20} {r['gap']:>5} {r['total_gain']:>7} {pred:>7} {err:>+7}{marker}")

    # Full model evaluation
    print(f"\nFull dataset ({len(non_maxed)} points):")
    full_eval = model.evaluate(non_maxed)
    print(f"  MAE: {full_eval['mae']:.1f}")
    print(f"  Within 3: {full_eval['within3']:.0f}%")
    print(f"  Within 5: {full_eval['within5']:.0f}%")

    return model


if __name__ == "__main__":
    train_and_evaluate()
