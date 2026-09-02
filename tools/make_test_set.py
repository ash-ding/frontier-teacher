"""Draw a fixed reference test set for the teacher loop to evaluate every step.

A subset of one benchmark, stratified by a field of that benchmark so the subset
carries the same mix as the whole -- difficulty for MATH-500 (`level`), year for
AIME, contest for HMMT. An unstratified draw can land well off the parent
distribution, and a test set that is not representative of its benchmark makes
the curve it produces harder to read, not easier.

The draw is deterministic (a fixed seed, and problems sorted by id before
sampling), so re-running this reproduces the file byte for byte. It exists as a
script rather than a one-off command because a committed data file whose
provenance is somebody's shell history is exactly how a stale baseline starts.

    # the three the teacher loop uses, regenerated exactly as committed:
    python tools/make_test_set.py --source data/benchmark/math500.jsonl \
        --out data/benchmark/math500_ref100.jsonl --n 100 --stratify-by level
    python tools/make_test_set.py --source "data/benchmark/aime_20*.jsonl" \
        --out data/benchmark/aime_ref30.jsonl --n 30 --stratify-by year
    python tools/make_test_set.py --source "data/benchmark/hmmt_*_20*.jsonl" \
        --out data/benchmark/hmmt_ref19.jsonl --n 19 --stratify-by competition
"""
import argparse
import collections
import glob
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
    ap.add_argument("--source", required=True,
                    help="a jsonl path, or a glob for a benchmark split across "
                         "files (quote it so the shell does not expand it)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--stratify-by", required=True,
                    help="a field every row carries: level (MATH-500), "
                         "year (AIME), competition (HMMT)")
    a = ap.parse_args()

    files = sorted(glob.glob(str(ROOT / a.source)))
    if not files:
        raise SystemExit(f"--source {a.source} matched no files")
    rows = sorted((json.loads(l) for f in files for l in open(f) if l.strip()),
                  key=lambda r: r["id"])
    missing = [r["id"] for r in rows if a.stratify_by not in r]
    if missing:
        raise SystemExit(f"--stratify-by {a.stratify_by!r} is absent from "
                         f"{len(missing)} row(s), e.g. {missing[0]}")
    if a.n > len(rows):
        raise SystemExit(f"--n {a.n} exceeds the {len(rows)} problems in {a.source}")

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
        "source": a.source,
        "source_files": [str(Path(f).relative_to(ROOT)) for f in files],
        "source_n": len(rows), "n": a.n, "seed": a.seed,
        "stratified_by": a.stratify_by,
        "command": f"python tools/make_test_set.py --source '{a.source}' "
                   f"--out {a.out} --n {a.n} --seed {a.seed} "
                   f"--stratify-by {a.stratify_by}",
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
