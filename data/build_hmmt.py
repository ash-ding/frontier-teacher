"""Build the three HMMT competition sets from MathArena.

HMMT (Harvard-MIT Mathematics Tournament) runs twice a year — November at
Harvard, February at MIT. MathArena publishes each competition shortly after it
is held, which is what makes these useful as post-training-cutoff evaluation
sets rather than just as harder math problems.

  hmmt_feb_2025.jsonl   30 problems   February 2025
  hmmt_nov_2025.jsonl   30 problems   November 2025
  hmmt_feb_2026.jsonl   33 problems   February 2026
                        -- 93 total

Unlike AIME, HMMT answers are NOT restricted to integers 0-999: roughly half are
exact forms (fractions, radicals, expressions in pi). Scoring therefore has to
go through symbolic equivalence rather than integer comparison, and this script
reports the split so the grading path is chosen deliberately.
"""
import json
import re
from pathlib import Path

from datasets import load_dataset

OUT = Path(__file__).parent

SETS = [
    ("hmmt_feb_2025", "MathArena/hmmt_feb_2025", "HMMT February 2025", 30),
    ("hmmt_nov_2025", "MathArena/hmmt_nov_2025", "HMMT November 2025", 30),
    ("hmmt_feb_2026", "MathArena/hmmt_feb_2026", "HMMT February 2026", 33),
]

INT_RE = re.compile(r"-?\d+")


def build(slug, repo, competition, expected):
    ds = load_dataset(repo)["train"]
    assert len(ds) == expected, f"{repo}: got {len(ds)}, expected {expected}"

    cols = ds.column_names
    rows = []
    for i, r in enumerate(ds):
        problem = str(r["problem"]).strip()
        answer = str(r["answer"]).strip()
        assert problem, f"{slug}[{i}]: empty problem"
        assert answer, f"{slug}[{i}]: empty answer"
        rows.append({
            "id": f"{slug.replace('_', '-')}-{i:02d}",
            "problem": problem,
            "answer": answer,
            "competition": competition,
            "problem_idx": r.get("problem_idx"),
            # nov_2025 does not carry problem_type; keep the key for a stable schema
            "problem_type": r.get("problem_type") if "problem_type" in cols else None,
            "source": repo,
        })

    ids = {r["id"] for r in rows}
    assert len(ids) == len(rows), f"{slug}: duplicate ids"

    p = OUT / f"{slug}.jsonl"
    with p.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    ints = [r for r in rows if INT_RE.fullmatch(r["answer"])]
    multi = [r for r in rows if "," in r["answer"]]
    print(f"wrote {p.name:22} n={len(rows):3}  integer answers={len(ints):3}/{len(rows)}"
          f"  multi-answer={len(multi)}")
    return rows


if __name__ == "__main__":
    all_rows, per_set = [], {}
    for slug, repo, competition, expected in SETS:
        rows = build(slug, repo, competition, expected)
        per_set[slug] = rows
        all_rows += rows

    assert len(all_rows) == 93, f"expected 93 problems total, got {len(all_rows)}"

    # the three competitions must be disjoint, and none may collide with MATH-500
    texts = [" ".join(r["problem"].split()) for r in all_rows]
    assert len(set(texts)) == len(texts), "duplicate problem text across HMMT sets"

    m500 = OUT / "math500.jsonl"
    if m500.exists():
        eval_txt = {" ".join(json.loads(l)["problem"].split()) for l in m500.open()}
        overlap = sum(1 for t in texts if t in eval_txt)
        assert overlap == 0, f"CONTAMINATION: {overlap} HMMT problems appear in MATH-500"
        print("  MATH-500 overlap: 0 (checked)")

    n_int = sum(1 for r in all_rows if INT_RE.fullmatch(r["answer"]))
    print(f"\ntotal {len(all_rows)} problems across {len(SETS)} competitions")
    print(f"  integer answers      : {n_int}/{len(all_rows)} ({100*n_int/len(all_rows):.0f}%)")
    print(f"  need symbolic compare: {len(all_rows)-n_int}/{len(all_rows)}"
          f" ({100*(len(all_rows)-n_int)/len(all_rows):.0f}%)")
    print("  -> integer_answer must stay FALSE for these tasks")
