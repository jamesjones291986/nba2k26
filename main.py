#!/usr/bin/env python3
"""NBA 2K26 Cap Breaker Reverse Engineering Tool — Main Menu."""
from models import load_all_builds


def menu():
    print("\n" + "=" * 50)
    print("  NBA 2K26 Cap Breaker Research Tool")
    print("=" * 50)

    builds = load_all_builds()
    n_builds = len(builds)
    n_with_cb = sum(1 for b in builds if any(b.cb_steps.get(a) for a in b.chosen))
    print(f"  Data: {n_builds} builds recorded, {n_with_cb} with CB data")

    print("""
  1. View test plan (which builds to create in-game)
  2. Enter build data
  3. Run analysis (find the formula)
  4. Validate formula (train/test split)
  5. Fit & save formula
  6. Optimize build (find best build for targets)
  7. List recorded builds
  0. Quit
""")
    return input("Choice: ").strip()


def list_builds():
    from constants import inches_to_height_str
    builds = load_all_builds()
    if not builds:
        print("No builds recorded yet.")
        return
    print(f"\n{'ID':>4} {'Name':<25} {'Pos':<4} {'Height':<7} {'Weight':<7} {'CB Attrs':>8}")
    print("-" * 60)
    for b in builds:
        ht = inches_to_height_str(b.height)
        cb_count = sum(1 for a in b.chosen if b.cb_steps.get(a))
        print(f"{b.id:>4} {b.name:<25} {b.position:<4} {ht:<7} {b.weight:<7} {cb_count:>8}")


def main():
    while True:
        choice = menu()
        if choice == "1":
            from testplan import print_test_plan
            print_test_plan()
        elif choice == "2":
            from entry import enter_build
            enter_build()
        elif choice == "3":
            from analyze import run_analysis
            run_analysis()
        elif choice == "4":
            from validate import validate
            validate()
        elif choice == "5":
            from optimize import fit_formula
            fit_formula()
        elif choice == "6":
            from optimize import interactive_optimize
            interactive_optimize()
        elif choice == "7":
            list_builds()
        elif choice == "0":
            print("✌️")
            break
        else:
            print("Invalid choice")


if __name__ == "__main__":
    main()
