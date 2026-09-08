#!/usr/bin/env python3
"""Re-fetch the ArXivMath answers and verify them against the hashes we already have.

The key that sealed curriculum-rl's copy was printed once at sealing time and
never written to disk, so that copy is unrecoverable by design. This refetches
the same rows from the same public endpoint and checks every one of the 49
answers against the answer_sha256 already recorded in
curriculum-rl/data/arxivmath-0626.json. A full match means this is byte-identical
to what that project sealed -- the hashes are the audit, not our good intentions.

Writes:
  data/goal_targets/arxiv3.problems.jsonl   the 3 targets, problem text only
  .secrets/arxiv3.answers.json              their answers, chmod 600 (seal next)
"""
import hashlib
import json
import os
import pathlib
import sys
import urllib.parse
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATASET = "MathArena/arxivmath-0626"
KNOWN = pathlib.Path.home() / "code/curriculum-rl/data/arxivmath-0626.json"
TARGETS = {"arxivmath-0626-003": "Q1", "arxivmath-0626-019": "Q2",
           "arxivmath-0626-030": "Q3"}


def fetch(dataset, split="train"):
    rows, offset = [], 0
    while True:
        q = urllib.parse.urlencode({"dataset": dataset, "config": "default",
                                    "split": split, "offset": offset, "length": 100})
        url = f"https://datasets-server.huggingface.co/rows?{q}"
        with urllib.request.urlopen(url, timeout=60) as r:
            d = json.load(r)
        if "error" in d:
            sys.exit(f"HF error: {d['error']}")
        batch = [x["row"] for x in d.get("rows", [])]
        rows += batch
        offset += len(batch)
        if len(batch) < 100 or offset >= d.get("num_rows_total", offset):
            break
    return rows


def norm(s):
    """Problem text, whitespace-collapsed. The join key.

    NOT problem_idx: the upstream set has lost a row since curriculum-rl fetched
    it (48 now, 49 then) and everything from index 18 on has shifted down by one,
    so index 019 today is the problem recorded as 020. Matching on the index
    would have handed Q2 and Q3 their neighbours' answers, silently. The text is
    what actually identifies a problem.
    """
    return " ".join(str(s).split())


def main():
    rows = fetch(DATASET)
    print(f"fetched {len(rows)} rows from {DATASET}")

    known = json.loads(KNOWN.read_text())
    by_text = {norm(k["problem"]): k for k in known}
    print(f"comparing against {len(known)} recorded problems in {KNOWN}")
    if len(by_text) != len(known):
        sys.exit("recorded problems are not unique by text; join key is unsafe")

    ok = bad = unknown = 0
    answers, problems = {}, []
    seen = set()
    for r in rows:
        ans = str(r["answer"]).strip()
        h = hashlib.sha256(ans.encode()).hexdigest()
        k = by_text.get(norm(r["problem"]))
        if k is None:
            unknown += 1
            continue
        seen.add(k["id"])
        if k["answer_sha256"] == h:
            ok += 1
        else:
            bad += 1
            print(f"  MISMATCH {k['id']}: recorded {k['answer_sha256'][:16]}… "
                  f"refetched {h[:16]}…")
        if k["id"] in TARGETS:
            answers[k["id"]] = ans
            # No answer_sha256 here, deliberately. The teacher can read this file
            # (it is given --add-dir on the repo), and a hash of a short answer is
            # a verification oracle: guess "1/8", hash it, and you know. The
            # candidate space for a limit constant is small enough to enumerate.
            # curriculum-rl's data/targets.json carries the hash; that is the one
            # part of its design not copied. The hashes are recoverable from the
            # sealed answers by anyone who holds the key, which is the only party
            # that should have them.
            problems.append({"id": k["id"], "role": TARGETS[k["id"]],
                             "problem": r["problem"],
                             "title": k.get("title"),
                             "source_arxiv": k.get("source_arxiv")})

    dropped = sorted({k["id"] for k in known} - seen)
    print(f"\nmatched by text: {ok} answers agree, {bad} disagree, "
          f"{unknown} refetched rows not in the recorded set")
    if dropped:
        print(f"recorded but no longer upstream: {', '.join(dropped)}")
    for pid, role in sorted(TARGETS.items(), key=lambda kv: kv[1]):
        print(f"  {role} {pid}: {'RECOVERED' if pid in answers else 'NOT FOUND'}")

    if bad or len(answers) != 3:
        sys.exit("refusing to write: could not recover all three targets intact")

    out = ROOT / "data" / "goal_targets"
    out.mkdir(parents=True, exist_ok=True)
    p = out / "arxiv3.problems.jsonl"
    with p.open("w") as f:
        for row in sorted(problems, key=lambda r: r["role"]):
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"  -> {p}  ({len(problems)} problems, no answers)")

    sec_dir = ROOT / ".secrets"
    sec_dir.mkdir(exist_ok=True)
    sec = sec_dir / "arxiv3.answers.json"
    sec.write_text(json.dumps(answers, indent=2, ensure_ascii=False))
    os.chmod(sec, 0o600)
    print(f"  -> {sec}  (chmod 600, plaintext -- seal it next)")


if __name__ == "__main__":
    main()
