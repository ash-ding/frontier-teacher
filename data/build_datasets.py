"""Build the two eval sets: MATH-500 (500) and AIME 2020-2024 (150).

AIME needs assembling from two sources because neither covers 2020-2024 alone:
  AI-MO/aimo-validation-aime   -> 2022-2024 complete (90)
  di-zhang-fdu/AIME_1983_2024  -> 2020-2021 complete (60); its 2023/2024 are
                                  incomplete (29/14) so we do not use those.
"""
import json
import re
from pathlib import Path

from datasets import load_dataset

OUT = Path(__file__).parent / "benchmark"
OUT.mkdir(parents=True, exist_ok=True)


def build_math500():
    ds = load_dataset("HuggingFaceH4/MATH-500")["test"]
    rows = [
        {
            "id": f"math500-{i:04d}",
            "problem": r["problem"],
            "answer": str(r["answer"]).strip(),
            "subject": r.get("subject"),
            "level": r.get("level"),
            "source": "MATH-500",
        }
        for i, r in enumerate(ds)
    ]
    assert len(rows) == 500, f"expected 500, got {len(rows)}"
    return rows


def build_aime():
    rows = []

    dz = load_dataset("di-zhang-fdu/AIME_1983_2024")["train"]
    for r in dz:
        if int(r["Year"]) in (2020, 2021):
            rows.append(
                {
                    "problem": r["Question"],
                    "answer": str(r["Answer"]).strip(),
                    "year": int(r["Year"]),
                    "source": "di-zhang-fdu/AIME_1983_2024",
                }
            )

    aimo = load_dataset("AI-MO/aimo-validation-aime")["train"]
    for r in aimo:
        year = int(re.search(r"20\d\d", r["url"]).group(0))
        if year in (2022, 2023, 2024):
            rows.append(
                {
                    "problem": r["problem"],
                    "answer": str(r["answer"]).strip(),
                    "year": year,
                    "source": "AI-MO/aimo-validation-aime",
                }
            )

    rows.sort(key=lambda x: (x["year"], x["problem"][:40]))
    per_year = {}
    for r in rows:
        i = per_year.get(r["year"], 0)
        r["id"] = f"aime-{r['year']}-{i:02d}"
        per_year[r["year"]] = i + 1

    by_year = {}
    for r in rows:
        by_year[r["year"]] = by_year.get(r["year"], 0) + 1
    assert all(by_year.get(y) == 30 for y in range(2020, 2025)), by_year
    assert len(rows) == 150, f"expected 150, got {len(rows)}"

    # AIME answers are integers 0-999.
    for r in rows:
        assert re.fullmatch(r"\d{1,3}", r["answer"]), (r["id"], r["answer"])
    return rows


def write(rows, name):
    p = OUT / name
    with p.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {p}  n={len(rows)}")


if __name__ == "__main__":
    m = build_math500()
    write(m, "math500.jsonl")

    # One file per competition year, matching how the HMMT sets are stored: each
    # AIME year is its own contest with its own date, and the date is what makes
    # these useful for reasoning about training cutoffs.
    a = build_aime()
    for year in range(2020, 2025):
        rows = [r for r in a if r["year"] == year]
        assert len(rows) == 30, f"AIME {year}: got {len(rows)}, expected 30"
        write(rows, f"aime_{year}.jsonl")
    print(f"AIME total: {len(a)} across 5 years")
