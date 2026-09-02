"""Draw the fixed reference test set the teacher loop evaluates after every step.

A subset of MATH-500, stratified by the benchmark's own `level` distribution so
that 100 problems carry the same difficulty mix as the 500 they come from --
an unstratified draw of 100 can land several levels off, and a test set whose
difficulty differs from the benchmark's makes the curve it produces harder to
read, not easier.

The draw is deterministic (a fixed seed, and problems sorted by id before
sampling), so re-running this reproduces the file byte for byte. It exists as a
script rather than a one-off command because a committed data file whose
provenance is somebody's shell history is exactly how a stale baseline starts.

    python tools/make_test_set.py                    # 100 from MATH-500
    python tools/make_test_set.py --n 200 --seed 7
"""
import argparse
import collections
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def largest_remainder(counts: dict, total: int) -> dict:
    """Apportion `total` across strata in proportion to `counts`.

    Plain rounding of each share does not sum to `total`; largest-remainder does,
    and gives the seats to the strata that were rounded down hardest.
    """
    n = sum(counts.values())
    exact = {k: v * total / n for k, v in counts.items()}
    take = {k: int(v) for k, v in exact.items()}
    short = total - sum(take.values())
    for k in sorted(exact, key=lambda k: (-(exact[k] - take[k]), str(k)))[:short]:
        take[k] += 1
    return take


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--source", default="data/benchmark/math500.jsonl")
    ap.add_argument("--out", default="data/benchmark/math500_ref100.jsonl")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--stratify-by", default="level")
    a = ap.parse_args()

    src = ROOT / a.source
    rows = sorted((json.loads(l) for l in src.open() if l.strip()),
                  key=lambda r: r["id"])
    if a.n > len(rows):
        raise SystemExit(f"--n {a.n} exceeds the {len(rows)} problems in {src}")

    by = collections.defaultdict(list)
    for r in rows:
        by[r[a.stratify_by]].append(r)
    quota = largest_remainder({k: len(v) for k, v in by.items()}, a.n)

    rng = random.Random(a.seed)
    picked = []
    for k in sorted(by, key=str):
        picked += rng.sample(by[k], quota[k])
    picked.sort(key=lambda r: r["id"])

    out = ROOT / a.out
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in picked))

    got = collections.Counter(str(r[a.stratify_by]) for r in picked)
    src_dist = collections.Counter(str(r[a.stratify_by]) for r in rows)
    (out.parent / f"{out.stem}.manifest.json").write_text(json.dumps({
        "source": a.source, "n": a.n, "seed": a.seed,
        "stratified_by": a.stratify_by,
        "command": f"python tools/make_test_set.py --source {a.source} "
                   f"--out {a.out} --n {a.n} --seed {a.seed}",
        "distribution": {k: got[k] for k in sorted(got)},
        "source_distribution": {k: src_dist[k] for k in sorted(src_dist)},
        "ids": [r["id"] for r in picked],
    }, indent=2) + "\n")

    print(f"{len(picked)} problems -> {out}")
    for k in sorted(got):
        share = 100 * src_dist[k] / len(rows)
        print(f"  {a.stratify_by} {k}: {got[k]:3d}   "
              f"({share:.1f}% of the source, {100*got[k]/len(picked):.1f}% here)")


if __name__ == "__main__":
    raise SystemExit(main())
