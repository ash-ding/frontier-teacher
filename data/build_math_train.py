"""Build the MATH training pool, split by where each problem originally came from.

`nlile/hendrycks-MATH-benchmark`'s train split is 12,000 rows (11,996 unique):
7,498 from the original Hendrycks *train* split, plus 4,498 that the PRM800K
re-split moved over from the original *test* split. MATH-500 is the remaining
500 test problems, so the whole pool is disjoint from the eval set either way.

The two halves are written separately because they are not interchangeable. A
model may have been trained on the original train split and not on the test
split — Llama-3.2-3B demonstrably was (61.0% pass@1 on the train-derived half vs
40.3% on the test-derived half vs 38.8% on held-out MATH-500) — in which case
only the test-derived half measures anything about reasoning rather than recall.

  math_train_orig_train.jsonl   7,498   from the original Hendrycks train split
  math_train_orig_test.jsonl    4,498   moved over from the original test split
                                -----
                                11,996

Problem ids are assigned over the pool as a whole, in the upstream order, before
the split. Existing profiling records and curated subsets reference these ids, so
the numbering is stable across this partition and must stay that way.
"""
import collections
import json
from pathlib import Path

from datasets import concatenate_datasets, load_dataset

OUT = Path(__file__).parent / "training_set"
OUT.mkdir(parents=True, exist_ok=True)
BENCH = Path(__file__).parent / "benchmark"
CONFIGS = ["algebra", "counting_and_probability", "geometry", "intermediate_algebra",
           "number_theory", "prealgebra", "precalculus"]

norm = lambda s: "".join(s.split())

# provenance reference: the ORIGINAL Hendrycks train split
orig_train_txt = {
    norm(p) for p in
    concatenate_datasets([load_dataset("EleutherAI/hendrycks_math", c)["train"]
                          for c in CONFIGS])["problem"]
}

ds = load_dataset("nlile/hendrycks-MATH-benchmark")["train"]
seen, rows = set(), []
for r in ds:
    key = norm(r["problem"])
    if key in seen:            # 12,000 rows contain a few duplicate problem texts
        continue
    seen.add(key)
    rows.append({
        "id": f"mathtrain-{len(rows):05d}",     # assigned before the split - keep stable
        "problem": r["problem"],
        "answer": str(r["answer"]).strip(),
        "subject": r.get("subject"),
        "level": r.get("level"),
        "split_origin": "orig_train" if key in orig_train_txt else "orig_test",
        "source": "nlile/hendrycks-MATH-benchmark:train",
    })

# the eval set must never be reachable from the training pool
m500 = {norm(json.loads(l)["problem"]) for l in (BENCH / "math500.jsonl").open()}
overlap = sum(1 for r in rows if norm(r["problem"]) in m500)
assert overlap == 0, f"CONTAMINATION: {overlap} training problems appear in MATH-500"

parts = {
    "orig_train": ("math_train_orig_train.jsonl", 7498),
    "orig_test": ("math_train_orig_test.jsonl", 4498),
}
total = 0
for origin, (fname, expected) in parts.items():
    sel = [r for r in rows if r["split_origin"] == origin]
    assert len(sel) == expected, f"{origin}: got {len(sel)}, expected {expected}"
    with (OUT / fname).open("w") as f:
        for r in sel:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    total += len(sel)
    lv = dict(sorted(collections.Counter(str(r["level"]) for r in sel).items()))
    print(f"wrote {fname:30} n={len(sel):5}  levels={lv}")

assert total == len(rows) == 11996, f"partition lost rows: {total} vs {len(rows)}"
ids = [r["id"] for r in rows]
assert len(set(ids)) == len(ids), "duplicate ids"
print(f"\ntotal {total} (deduped from {len(ds)})   MATH-500 overlap: {overlap}")
print("  ids run mathtrain-00000..mathtrain-11995 across BOTH files, not per file")
