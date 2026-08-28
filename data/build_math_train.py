"""Build the MATH train pool (12k) used for difficulty profiling.

Uses nlile/hendrycks-MATH-benchmark's train split: 7498 unique problems from the
original Hendrycks train split + 4500 moved over from the original test split
(the PRM800K re-split). Verified disjoint from MATH-500, so a subset selected
here can be trained on while still reporting MATH-500 honestly.
"""
import json
from pathlib import Path

from datasets import load_dataset

OUT = Path(__file__).parent

ds = load_dataset("nlile/hendrycks-MATH-benchmark")["train"]
seen, rows = set(), []
for r in ds:
    key = "".join(r["problem"].split())
    if key in seen:          # 12000 rows contain a few duplicate problem texts
        continue
    seen.add(key)
    rows.append({
        "id": f"mathtrain-{len(rows):05d}",
        "problem": r["problem"],
        "answer": str(r["answer"]).strip(),
        "subject": r.get("subject"),
        "level": r.get("level"),
        "source": "nlile/hendrycks-MATH-benchmark:train",
    })

# guard against ever profiling on the eval set
m500 = {"".join(json.loads(l)["problem"].split()) for l in (OUT / "math500.jsonl").open()}
overlap = sum(1 for r in rows if "".join(r["problem"].split()) in m500)
assert overlap == 0, f"CONTAMINATION: {overlap} train problems appear in MATH-500"

p = OUT / "math_train_12k.jsonl"
with p.open("w") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

import collections
print(f"wrote {p}  n={len(rows)}  (deduped from {len(ds)})")
print("  MATH-500 overlap:", overlap, "(must be 0)")
print("  level dist:", dict(sorted(collections.Counter(str(r['level']) for r in rows).items())))
