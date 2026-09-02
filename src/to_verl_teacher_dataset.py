"""Convert teacher-authored problems into verl's training format.

Sibling of to_verl_dataset.py, kept separate so the validated curated-subset
converter stays untouched. The schema is identical (verl reads the same keys);
what differs is the source and the contract:

  * rows come from a teacher turn as {id, problem, answer} (level/subject/band
    optional), not from a difficulty-graded subset on disk;
  * the teacher's stated `answer` becomes `reward_model.ground_truth` DIRECTLY,
    with NO verification. This is deliberate: the loop is observational, so we do
    not impose our prior on the teacher. A wrong label flips GRPO's group-relative
    advantage sign for that problem -- a real poisoning risk, documented and
    accepted for this cycle (verification is a Deferred toggle).

The prompt text is imported from evaluate.py, exactly as to_verl_dataset.py does,
so training and evaluation always ask the model the same question.

usage:
  python src/to_verl_teacher_dataset.py --data <step>/data.jsonl --out <step>/train.verl.jsonl
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from evaluate import PROMPT  # noqa: E402  - the single source of the prompt

DATA_SOURCE = "teacher_curriculum"


def convert(rows):
    """Teacher rows -> verl rows. `answer` -> ground_truth verbatim (no verify)."""
    out = []
    for i, r in enumerate(rows):
        out.append({
            "data_source": DATA_SOURCE,
            "prompt": [{"role": "user", "content": PROMPT.format(problem=r["problem"])}],
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": r["answer"]},
            "extra_info": {
                "index": i,
                "id": r["id"],
                "level": r.get("level"),
                "subject": r.get("subject"),
                "band": r.get("band"),
                "source": "teacher",
            },
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="teacher data.jsonl with {id,problem,answer}")
    ap.add_argument("--out", default="", help="output path (default: alongside, .verl.jsonl)")
    a = ap.parse_args()

    src = Path(a.data)
    rows = [json.loads(l) for l in src.open() if l.strip()]
    conv = convert(rows)

    dst = Path(a.out) if a.out else src.with_suffix(".verl.jsonl")
    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("w") as f:
        for r in conv:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    assert len(conv) == len(rows), "row count changed during conversion"
    assert all(isinstance(r["prompt"], list) for r in conv), "prompt must be a message list"
    assert all(str(r["reward_model"]["ground_truth"]).strip() for r in conv), "empty ground truth"
    print(f"wrote {dst}  n={len(conv)}  data_source={DATA_SOURCE}")
    print(f"  prompt[0] head: {conv[0]['prompt'][0]['content'][:90]!r}")
    print(f"  ground_truth[0]: {conv[0]['reward_model']['ground_truth']!r}")


if __name__ == "__main__":
    main()
