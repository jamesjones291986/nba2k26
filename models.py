import json
import os
from dataclasses import dataclass, field
from constants import ALL_ATTRIBUTES

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


@dataclass
class Build:
    id: int = 0
    name: str = ""
    position: str = ""
    height: int = 0  # inches
    weight: int = 0  # lbs
    wingspan: int = 0  # inches
    # Builder ceiling per attribute (max you COULD set it to)
    builder_caps: dict = field(default_factory=lambda: {a: 25 for a in ALL_ATTRIBUTES})
    # What you actually set each attribute to
    chosen: dict = field(default_factory=lambda: {a: 25 for a in ALL_ATTRIBUTES})
    # Cap breaker steps: list of 5 values per attribute (the value AFTER each breaker)
    cb_steps: dict = field(default_factory=lambda: {a: [] for a in ALL_ATTRIBUTES})

    def cb_total_gain(self, attr):
        """Total gain from all cap breakers for an attribute."""
        steps = self.cb_steps.get(attr, [])
        return steps[-1] - self.chosen[attr] if steps else 0

    def cb_gains(self, attr):
        """Per-step gains for an attribute."""
        steps = self.cb_steps.get(attr, [])
        if not steps:
            return []
        prev = self.chosen[attr]
        gains = []
        for s in steps:
            gains.append(s - prev)
            prev = s
        return gains

    def gap(self, attr):
        """Gap between builder ceiling and chosen value."""
        return self.builder_caps[attr] - self.chosen[attr]

    def to_dict(self):
        return {
            "id": self.id, "name": self.name, "position": self.position,
            "height": self.height, "weight": self.weight, "wingspan": self.wingspan,
            "builder_caps": self.builder_caps, "chosen": self.chosen,
            "cb_steps": self.cb_steps,
        }

    @classmethod
    def from_dict(cls, d):
        return cls(**d)


def _ensure_data_dir():
    os.makedirs(DATA_DIR, exist_ok=True)


def save_build(build):
    _ensure_data_dir()
    path = os.path.join(DATA_DIR, f"build_{build.id:03d}.json")
    with open(path, "w") as f:
        json.dump(build.to_dict(), f, indent=2)
    return path


def load_build(build_id):
    path = os.path.join(DATA_DIR, f"build_{build_id:03d}.json")
    with open(path) as f:
        return Build.from_dict(json.load(f))


def load_all_builds():
    _ensure_data_dir()
    builds = []
    for fname in sorted(os.listdir(DATA_DIR)):
        if fname.startswith("build_") and fname.endswith(".json"):
            with open(os.path.join(DATA_DIR, fname)) as f:
                builds.append(Build.from_dict(json.load(f)))
    return builds


def next_build_id():
    _ensure_data_dir()
    ids = []
    for fname in os.listdir(DATA_DIR):
        if fname.startswith("build_") and fname.endswith(".json"):
            try:
                ids.append(int(fname[6:9]))
            except ValueError:
                pass
    return max(ids, default=0) + 1
