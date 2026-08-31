"""Convert a curated difficulty-band subset into verl's training format.

verl accepts JSONL directly (it dispatches on file extension in
verl/utils/dataset/rl_dataset.py), so there is no parquet step. What it does
require is a specific schema:

  prompt        a CHAT MESSAGE LIST, not a string - the chat template is applied
                downstream by the rollout worker
  reward_model  a dict carrying ground_truth
  data_source   a string; the reward manager reads this key by name
  extra_info    free-form dict, passed verbatim to the reward function

The prompt text is imported from evaluate.py rather than copied. If training and
evaluation ever asked the model a different question, every curve in this project
would be comparing two different things while looking correct.

usage:
  python src/to_verl_dataset.py --subset data/further_improve/llama32-3b/<file>.jsonl
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from evaluate import PROMPT  # noqa: E402  - the single source of the prompt

DATA_SOURCE = "math_further_improve"


def convert(rows):
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
                "pass_at_1": r.get("pass_at_1"),
                "split_origin": r.get("split_origin"),
            },
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", required=True, help="path to a further_improve JSONL")
    ap.add_argument("--out", default="", help="output path (default: alongside, .verl.jsonl)")
    a = ap.parse_args()

    src = Path(a.subset)
    rows = [json.loads(l) for l in src.open()]

    # A handful of MATH problems carry an empty answer - the upstream extraction
    # failed on their multiple-choice form. They are ungradeable, so they can
    # never be scored correct and land in the p = 0 band by construction rather
    # than by being hard: 3 of the 11,996-problem pool, all three in a p = 0
    # subset and none anywhere else. Training on them is harmless (they are
    # always wrong, so the group is degenerate either way) but they would inflate
    # any count of "problems this model cannot solve", so drop them and say so.
    ungradeable = [r for r in rows if not str(r.get("answer", "")).strip()]
    if ungradeable:
        print(f"  dropping {len(ungradeable)} ungradeable row(s) with an empty answer: "
              f"{', '.join(r['id'] for r in ungradeable[:5])}")
        rows = [r for r in rows if str(r.get("answer", "")).strip()]
    conv = convert(rows)

    dst = Path(a.out) if a.out else src.with_suffix(".verl.jsonl")
    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("w") as f:
        for r in conv:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    assert len(conv) == len(rows), "row count changed during conversion"
    assert all(isinstance(r["prompt"], list) for r in conv), "prompt must be a message list"
    assert all(r["reward_model"]["ground_truth"] for r in conv), "empty ground truth"
    print(f"wrote {dst}  n={len(conv)}  band={rows[0].get('band')!r}")
    print(f"  prompt[0] head: {conv[0]['prompt'][0]['content'][:90]!r}")
    print(f"  ground_truth[0]: {conv[0]['reward_model']['ground_truth']!r}")


if __name__ == "__main__":
    main()
