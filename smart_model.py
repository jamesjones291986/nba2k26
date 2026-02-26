#!/usr/bin/env python3
"""Cap breaker prediction model v2.

Formula: CB_gain = f(gap, builder_cap) + attribute_bias

Key findings:
- 1-4 step attributes: gain = gap (always full recovery to builder cap)
- 5-step attributes: gain = base_model(gap, cap) + per_attribute_correction
- Other attribute allocations have ZERO effect on CB gain
- Only chosen value and builder cap matter
"""
import json
import os
import numpy as np
from models import load_all_builds, DATA_DIR
from analyze import extract_features
from collections import defaultdict

MODEL_PATH = os.path.join(DATA_DIR, "cb_model_v2.json")


def predict_num_steps(gap, builder_cap):
    """Predict number of cap breaker steps from observed patterns."""
    if gap == 0:
        return 0
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


class CBModelV2:
    def __init__(self):
        self.base_coeffs = None  # [intercept, gap, cap_norm, gap*cap_norm]
        self.attr_bias = {}
        self.step_coeffs = []
        self.fitted = False

    def fit(self, builds=None):
        if builds is None:
            builds = load_all_builds()
        rows = extract_features(builds)
        five = [r for r in rows if r['num_steps'] == 5 and r['gap'] > 0]
        if len(five) < 20:
            print(f"Only {len(five)} 5-step points, need at least 20")
            return

        y = np.array([r['total_gain'] for r in five])
        gap = np.array([r['gap'] for r in five])
        cap_norm = np.array([r['builder_cap'] / 99 for r in five])

        # Base model: gain = a + b*gap + c*cap_norm + d*gap*cap_norm
        X = np.column_stack([np.ones(len(y)), gap, cap_norm, gap * cap_norm])
        self.base_coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)

        # Per-attribute bias
        base_pred = X @ self.base_coeffs
        residuals = y - base_pred
        by_attr = defaultdict(list)
        for i, r in enumerate(five):
            by_attr[r['attr']].append(residuals[i])
        self.attr_bias = {attr: float(np.mean(resids)) for attr, resids in by_attr.items()}

        # Per-step models
        self.step_coeffs = []
        for step in range(5):
            gains = []
            feats = []
            for r in five:
                if len(r['gains']) > step:
                    gains.append(r['gains'][step])
                    gp = r['gap_pct']
                    cn = r['builder_cap'] / 99
                    bias = self.attr_bias.get(r['attr'], 0) / 5  # spread bias across steps
                    feats.append([gp, gp**2, cn, bias])
            if len(gains) >= 10:
                Xs = np.column_stack([np.ones(len(gains)), np.array(feats)])
                ys = np.array(gains)
                sc, _, _, _ = np.linalg.lstsq(Xs, ys, rcond=None)
                self.step_coeffs.append(sc.tolist())

        self.fitted = True

    def predict_total(self, chosen, builder_cap, attr_name=None):
        gap = builder_cap - chosen
        if gap <= 0:
            return 0
        n_steps = predict_num_steps(gap, builder_cap)
        if n_steps < 5:
            return gap

        if not self.fitted:
            return gap  # fallback

        cap_norm = builder_cap / 99
        x = np.array([1, gap, cap_norm, gap * cap_norm])
        base = x @ self.base_coeffs
        bias = self.attr_bias.get(attr_name, 0) if attr_name else 0
        gain = max(1, round(base + bias))
        return min(gain, gap)

    def predict_steps(self, chosen, builder_cap, attr_name=None):
        gap = builder_cap - chosen
        if gap <= 0:
            return []
        n_steps = predict_num_steps(gap, builder_cap)

        if n_steps < 5:
            # Full recovery, distribute decreasingly
            if n_steps == 1:
                return [gap]
            weights = [0.75 ** i for i in range(n_steps)]
            total_w = sum(weights)
            steps = [max(1, round(gap * w / total_w)) for w in weights]
            diff = gap - sum(steps)
            for i in range(abs(diff)):
                idx = i % n_steps
                steps[idx] += 1 if diff > 0 else -1
                steps[idx] = max(1, steps[idx])
            return steps

        if not self.fitted or not self.step_coeffs:
            total = self.predict_total(chosen, builder_cap, attr_name)
            per = max(1, total // 5)
            return [per] * 5

        gap_pct = gap / builder_cap
        cap_norm = builder_cap / 99
        bias = (self.attr_bias.get(attr_name, 0) / 5) if attr_name else 0

        gains = []
        for sc in self.step_coeffs:
            x = np.array([1, gap_pct, gap_pct**2, cap_norm, bias])
            g = max(1, round(x @ np.array(sc)))
            gains.append(g)

        # Ensure decreasing
        for i in range(1, len(gains)):
            gains[i] = min(gains[i], gains[i-1])

        # Adjust to match predicted total
        total = self.predict_total(chosen, builder_cap, attr_name)
        current_total = sum(gains)
        if current_total > 0 and current_total != total:
            ratio = total / current_total
            gains = [max(1, round(g * ratio)) for g in gains]
            # Fine-tune
            diff = total - sum(gains)
            for i in range(abs(diff)):
                idx = i % 5
                gains[idx] += 1 if diff > 0 else -1
                gains[idx] = max(1, gains[idx])

        return gains

    def evaluate(self, rows):
        errors = []
        for r in rows:
            if r['gap'] == 0:
                continue
            pred = self.predict_total(r['chosen'], r['builder_cap'], r['attr'])
            errors.append(abs(pred - r['total_gain']))
        if not errors:
            return {}
        return {
            'mae': np.mean(errors),
            'within3': sum(1 for e in errors if e <= 3) / len(errors) * 100,
            'within5': sum(1 for e in errors if e <= 5) / len(errors) * 100,
            'n': len(errors),
        }

    def save(self):
        os.makedirs(DATA_DIR, exist_ok=True)
        data = {
            'base_coeffs': self.base_coeffs.tolist() if self.base_coeffs is not None else None,
            'attr_bias': self.attr_bias,
            'step_coeffs': self.step_coeffs,
        }
        with open(MODEL_PATH, 'w') as f:
            json.dump(data, f, indent=2)

    def load(self):
        with open(MODEL_PATH) as f:
            data = json.load(f)
        self.base_coeffs = np.array(data['base_coeffs'])
        self.attr_bias = data['attr_bias']
        self.step_coeffs = data['step_coeffs']
        self.fitted = True


def train_and_test():
    builds = load_all_builds()
    rows = extract_features(builds)
    non_maxed = [r for r in rows if r['gap'] > 0]

    # Train on 1-15, test on 16-17
    train_builds = [b for b in builds if b.id <= 15]
    test_rows = [r for r in non_maxed if r['build_id'] > 15]
    train_rows = [r for r in non_maxed if r['build_id'] <= 15]

    model = CBModelV2()
    model.fit(train_builds)

    print("=" * 60)
    print("CAP BREAKER MODEL V2 (with per-attribute correction)")
    print("=" * 60)

    print(f"\nBase formula: gain = {model.base_coeffs[0]:.2f} "
          f"+ {model.base_coeffs[1]:.4f}*gap "
          f"+ {model.base_coeffs[2]:.2f}*(cap/99) "
          f"+ {model.base_coeffs[3]:.4f}*gap*(cap/99)")

    print(f"\nPer-attribute corrections:")
    for attr, bias in sorted(model.attr_bias.items(), key=lambda x: x[1]):
        label = "STINGY" if bias < -3 else "GENEROUS" if bias > 3 else ""
        print(f"  {attr:<20}: {bias:>+6.1f}  {label}")

    train_eval = model.evaluate(train_rows)
    test_eval = model.evaluate(test_rows)

    print(f"\nTraining ({train_eval['n']} pts): MAE={train_eval['mae']:.1f}, "
          f"within 3={train_eval['within3']:.0f}%, within 5={train_eval['within5']:.0f}%")
    print(f"Test ({test_eval['n']} pts): MAE={test_eval['mae']:.1f}, "
          f"within 3={test_eval['within3']:.0f}%, within 5={test_eval['within5']:.0f}%")

    # Show test predictions
    print(f"\n{'Attribute':<20} {'Cap':>4} {'Cho':>4} {'Gap':>4} {'Actual':>7} {'Pred':>6} {'Err':>5}")
    print("-" * 55)
    for r in test_rows:
        pred = model.predict_total(r['chosen'], r['builder_cap'], r['attr'])
        err = pred - r['total_gain']
        marker = ' OK' if abs(err) <= 3 else ''
        print(f"{r['attr']:<20} {r['builder_cap']:>4} {r['chosen']:>4} {r['gap']:>4} "
              f"{r['total_gain']:>7} {pred:>6} {err:>+5}{marker}")

    # Full dataset
    model_full = CBModelV2()
    model_full.fit(builds)
    full_eval = model_full.evaluate(non_maxed)
    print(f"\nFull model ({full_eval['n']} pts): MAE={full_eval['mae']:.1f}, "
          f"within 3={full_eval['within3']:.0f}%, within 5={full_eval['within5']:.0f}%")

    model_full.save()
    print(f"\nModel saved to {MODEL_PATH}")
    return model_full


if __name__ == "__main__":
    train_and_test()
