# NBA 2K26 Build Optimizer

Finds optimal MyPlayer builds by computing attribute/badge combinations that maximize effectiveness for given archetypes.

## Structure

- `entry.py` / `main.py` — Entry points
- `optimize.py` — Core build optimization engine
- `analyze.py` — Build comparison and analysis
- `models.py` — Data models for builds/attributes/badges
- `constants.py` — Game constants (attribute caps, costs)
- `builder_data.py` — Raw build data
- `smart_model.py` — Intelligent model selection
- `validate.py` — Build validation against game rules
- `data/` — Saved builds, reference data
- `docs/` — Guides (badge requirements, build minimums, jumpshots, animations)

## Key Concepts

- Badges have tiers: Bronze → Silver → Gold → HoF → Legend
- Badge boosts from gameplay allow reaching higher tiers with lower base attributes
- Must-have badges target HoF minimum, strong options target Gold minimum
- Build optimizer accounts for attribute caps, badge unlock thresholds, and boost allocations

## Tech Stack

Python
