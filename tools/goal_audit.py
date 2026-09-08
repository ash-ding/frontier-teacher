#!/usr/bin/env python3
"""Did the teacher leak a target into the curriculum? Run before believing a hit.

    GOAL_TEACHER_KEY=... python tools/goal_audit.py --run <dir>

The Q3 claim rests on the teacher not having that answer, and the Q1/Q2 claim
rests on it not having written the problem down. The first is structural -- the
key is not in its environment. The second is not, and this is the check for it.

Three tests, after curriculum-rl's audit.py:

  answers     the plaintext of any target appearing in anything the teacher wrote
  13-grams    shared 13-word spans with a target statement, which is how a
              paraphrase shows up when the wording has drifted
  similarity  the highest whole-statement character similarity, reported rather
              than thresholded -- a teacher drilling the right machinery writes
              problems of the same FORM as the target, and that is the intended
              behaviour, not contamination. Read the top match, do not trust a
              number.

Also greps the teacher's own tool calls for network access, which Bash makes
possible and no allowlist can prevent. curriculum-rl's rule applies unchanged: a
run whose audit shows a fetch is discarded rather than explained.
"""
import argparse
import difflib
import glob
import json
import os
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "train"))
from goal_teacher.secrets_box import load_answers  # noqa: E402

NET = ("curl", "wget", "urllib", "requests.get", "httpx", "huggingface",
       "datasets-server", "matharena", "arxiv.org", "snapshot_download",
       "load_dataset", "hf_hub")
N = 13


def norm(s):
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", str(s).lower())).strip()


def grams(s, n=N):
    w = norm(s).split()
    return {" ".join(w[i:i + n]) for i in range(max(0, len(w) - n + 1))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--targets", default="data/goal_targets/arxiv3.problems.jsonl")
    ap.add_argument("--sealed", default=".secrets/arxiv3.sealed")
    a = ap.parse_args()
    run = pathlib.Path(a.run)

    targets = [json.loads(l) for l in (ROOT / a.targets).open()]
    answers = load_answers(ROOT / a.sealed, os.environ.get("GOAL_TEACHER_KEY"))

    rows = []
    for f in sorted(glob.glob(str(run / "step_*/train/data.jsonl"))) + \
             sorted(glob.glob(str(run / "step_*/eval_*/data.jsonl"))):
        for line in open(f):
            try:
                rows.append((f.split("/")[-3], json.loads(line)))
            except json.JSONDecodeError:
                pass
    print(f"{run.name}: {len(rows)} problems written by the teacher\n")

    bad = [(s, r.get("id"), tid) for s, r in rows for tid, v in answers.items()
           if v and (v in str(r.get("problem", "")) or v in str(r.get("answer", "")))]
    print(f"answers in the curriculum:  {bad if bad else 'none'}")

    worst, where = 0, None
    for t in targets:
        tg = grams(t["problem"])
        for s, r in rows:
            ov = len(tg & grams(r.get("problem", "")))
            if ov > worst:
                worst, where = ov, (t["id"], s, r.get("id"))
    print(f"max shared {N}-grams:        {worst}" + (f"  {where}" if where else ""))

    for t in targets:
        q = norm(t["problem"])
        top = sorted(
            ((difflib.SequenceMatcher(None, q, norm(r.get("problem", ""))).ratio(),
              s, r.get("id"), r.get("answer"), r.get("problem", "")) for s, r in rows),
            key=lambda x: -x[0])[:2]
        print(f"\n{t.get('role', t['id'])} nearest curriculum items:")
        for sc, s, rid, ans, prob in top:
            print(f"  [{sc:.3f}] {s}/{rid}  answer={ans!r}")
            print(f"      {prob[:150].replace(chr(10), ' ')}")

    cmds = []
    for f in glob.glob(str(run / "step_*/*/teacher.log")):
        for line in open(f):
            try:
                o = json.loads(line)
            except json.JSONDecodeError:
                continue
            if o.get("type") != "assistant":
                continue
            for b in o.get("message", {}).get("content", []):
                if isinstance(b, dict) and b.get("type") == "tool_use":
                    cmds.append(str(b.get("input", {}).get("command", "")))
    net = [c[:120] for c in cmds if any(k in c.lower() for k in NET)]
    print(f"\nteacher tool calls: {len(cmds)}")
    print(f"network:            {len(net)} hit(s)")
    for c in net[:10]:
        print(f"  {c}")
    print("\n" + ("CLEAN" if not bad and not net and worst == 0
                  else "REVIEW THE ABOVE BEFORE USING THIS RUN"))


if __name__ == "__main__":
    main()
