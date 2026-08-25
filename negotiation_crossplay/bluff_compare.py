#!/usr/bin/env python3
"""Side-by-side gain-vs-leakage comparison across the bluff-prompt arms.

Reads the value-probe result dirs for the three arms (off / system / system_fewshot)
and, for the bluff model only (default qwen/qwen3.5-9b), prints per arm:

  n         = number of that model's seats in the arm
  lie_rate  = fraction of its seats with >=1 structural misrep hit (misrep_examples)
  leakage   = mean opponent post-game Spearman recovering its hidden values
              (LOW/negative = concealed/misled, HIGH = got read) -- Term-2
  gain      = mean value captured through trade under its OWN values -- Term-1
  intg      = mean game integrative_ratio (pie expansion)

Reuses misrep_examples.collect()/detect_misrep() so detection is identical to the
qualitative surfacer. A good bluff moves `gain` UP without moving `leakage` up.
"""
import argparse
from statistics import mean

from misrep_examples import collect

# arm label -> the {2p,3p} result dirs produced by launch_bluff_qwen35.sh
ARMS = [
    ("off",            ["bluff_off_2p", "bluff_off_3p"]),
    ("system",         ["bluff_system_2p", "bluff_system_3p"]),
    ("system_fewshot", ["bluff_fewshot_2p", "bluff_fewshot_3p"]),
]


def summarize(dirs, model_key):
    seats = [s for s in collect(dirs) if s["model"] == model_key]
    if not seats:
        return None

    def m(key):
        xs = [s[key] for s in seats if s.get(key) is not None]
        return mean(xs) if xs else None

    return {
        "n": len(seats),
        "lie_rate": mean(1.0 if s["hits"] else 0.0 for s in seats),
        "leakage": m("leakage"),
        "gain": m("gain"),
        "intg": m("integrative_ratio"),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="qwen/qwen3.5-9b")
    ap.add_argument("--arms", nargs="*",
                    help="override arms as name=dir1,dir2 ... (default: off/system/fewshot)")
    args = ap.parse_args()

    model_key = args.model.split("/")[-1]
    arms = ARMS
    if args.arms:
        arms = [(a.partition("=")[0], a.partition("=")[2].split(",")) for a in args.arms]

    def f(x, nd=3):
        return f"{x:.{nd}f}" if x is not None else "  -  "

    print(f"bluff-prompt comparison for {args.model}\n")
    print(f"{'arm':<16}{'n':>4}{'lie_rate':>10}{'leakage':>10}{'gain':>10}{'intg':>8}")
    print("-" * 58)
    for name, dirs in arms:
        r = summarize(dirs, model_key)
        if r is None:
            print(f"{name:<16}{'(no data)':>10}")
            continue
        print(f"{name:<16}{r['n']:>4}{f(r['lie_rate']):>10}{f(r['leakage']):>10}"
              f"{f(r['gain'], 1):>10}{f(r['intg']):>8}")
    print()


if __name__ == "__main__":
    main()
