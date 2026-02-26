#!/usr/bin/env python3
"""CLI tool for entering NBA 2K26 build data with cap breaker steps."""
from constants import POSITIONS, ATTRIBUTES, POSITION_RANGES, inches_to_height_str
from models import Build, save_build, next_build_id


def ask_int(prompt, lo=None, hi=None):
    while True:
        try:
            v = int(input(prompt))
            if (lo is not None and v < lo) or (hi is not None and v > hi):
                print(f"  Must be between {lo} and {hi}")
                continue
            return v
        except ValueError:
            print("  Enter a number")


def ask_choice(prompt, options):
    print(prompt)
    for i, o in enumerate(options, 1):
        print(f"  {i}. {o}")
    return options[ask_int("Choice: ", 1, len(options)) - 1]


def enter_build():
    bid = next_build_id()
    print(f"\n=== New Build (#{bid}) ===")
    name = input("Build name/description (optional): ").strip() or f"Test Build {bid}"
    pos = ask_choice("Position:", POSITIONS)
    r = POSITION_RANGES[pos]
    height = ask_int(f"Height (inches, {r['height'][0]}-{r['height'][1]}): ", *r["height"])
    weight = ask_int(f"Weight (lbs, {r['weight'][0]}-{r['weight'][1]}): ", *r["weight"])
    wingspan = ask_int(f"Wingspan (inches, {r['wingspan'][0]}-{r['wingspan'][1]}): ", *r["wingspan"])

    print(f"\n{pos} | {inches_to_height_str(height)} | {weight}lbs | {inches_to_height_str(wingspan)} ws")

    builder_caps = {}
    chosen = {}
    cb_steps = {}

    for cat, attrs in ATTRIBUTES.items():
        print(f"\n--- {cat} ---")
        print("For each attribute enter: builder_cap chosen cb1 cb2 cb3 cb4 cb5")
        print("(cap breaker values are the attribute value AFTER each breaker)")
        print("If no cap breakers, enter 0s for cb values\n")
        for attr in attrs:
            while True:
                raw = input(f"  {attr}: ").strip().split()
                if len(raw) == 7:
                    try:
                        vals = [int(x) for x in raw]
                        break
                    except ValueError:
                        pass
                elif len(raw) == 2:
                    # Allow just cap + chosen if no CB data yet
                    try:
                        vals = [int(x) for x in raw] + [0, 0, 0, 0, 0]
                        break
                    except ValueError:
                        pass
                print("    Enter 7 numbers: builder_cap chosen cb1 cb2 cb3 cb4 cb5")
                print("    Or 2 numbers: builder_cap chosen (if no CB data yet)")

            builder_caps[attr] = vals[0]
            chosen[attr] = vals[1]
            steps = [v for v in vals[2:] if v > 0]
            cb_steps[attr] = steps

    build = Build(
        id=bid, name=name, position=pos, height=height, weight=weight,
        wingspan=wingspan, builder_caps=builder_caps, chosen=chosen, cb_steps=cb_steps,
    )

    path = save_build(build)
    print(f"\n✅ Build #{bid} saved to {path}")
    print(f"   Total CB growth potential: {sum(build.cb_total_gain(a) for a in chosen)}")
    return build


if __name__ == "__main__":
    print("NBA 2K26 Cap Breaker Data Entry")
    print("================================")
    while True:
        enter_build()
        if input("\nEnter another build? (y/n): ").strip().lower() != "y":
            break
