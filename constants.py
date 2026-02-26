POSITIONS = ["PG", "SG", "SF", "PF", "C"]

ATTRIBUTES = {
    "Finishing": ["Close Shot", "Driving Layup", "Driving Dunk", "Standing Dunk", "Post Control"],
    "Shooting": ["Mid-Range Shot", "Three-Point Shot", "Free Throw"],
    "Playmaking": ["Pass Accuracy", "Ball Handle", "Speed With Ball"],
    "Defense": ["Interior Defense", "Perimeter Defense", "Steal", "Block"],
    "Rebounding": ["Offensive Rebound", "Defensive Rebound"],
    "Physical": ["Speed", "Agility", "Strength", "Vertical"],
}

ALL_ATTRIBUTES = [a for attrs in ATTRIBUTES.values() for a in attrs]

# Height in inches, weight in lbs, wingspan in inches
POSITION_RANGES = {
    "PG": {"height": (70, 77), "weight": (160, 220), "wingspan": (66, 82)},
    "SG": {"height": (72, 79), "weight": (170, 230), "wingspan": (68, 84)},
    "SF": {"height": (75, 81), "weight": (180, 250), "wingspan": (71, 87)},
    "PF": {"height": (77, 83), "weight": (200, 270), "wingspan": (73, 89)},
    "C":  {"height": (80, 86), "weight": (220, 300), "wingspan": (76, 92)},
}

# Median build per position for baseline suggestions
POSITION_MEDIANS = {
    pos: {
        "height": (r["height"][0] + r["height"][1]) // 2,
        "weight": (r["weight"][0] + r["weight"][1]) // 2,
        "wingspan": (r["wingspan"][0] + r["wingspan"][1]) // 2,
    }
    for pos, r in POSITION_RANGES.items()
}


def inches_to_height_str(inches):
    return f"{inches // 12}'{inches % 12}\""


def height_str_to_inches(s):
    s = s.strip().replace('"', '').replace("'", " ").replace('′', ' ').replace('″', '')
    parts = s.split()
    return int(parts[0]) * 12 + int(parts[1])
