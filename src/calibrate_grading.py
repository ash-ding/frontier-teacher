"""Calibrate the grading path on a dataset without spending any GPU time.

A score is only as trustworthy as the parser underneath it, and a parser that
fails silently looks exactly like a weak model. This exercises grade() directly
on known inputs, so a failure here is unambiguously a harness bug:

  identity     the gold answer, boxed, must grade as correct. 100% or the
               extractor/comparator cannot read this dataset's answer format.
  equivalence  a hand-written table of pairs that are the same value written
               differently. Failures here mean real correct answers get marked
               wrong, which depresses every score.
  specificity  another problem's answer must grade as WRONG. Without this a
               grader that returns True unconditionally passes the first two.
  thinking     the same identity check behind a </think> tag, since thinking
               models are graded only on what follows it.

usage:  python src/calibrate_grading.py --data data/benchmark/hmmt_feb_2025.jsonl ...
"""
import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from grading import grade  # noqa: E402

# (gold, variant, should_match) - written to be true by mathematics, not by string
EQUIVALENCE = [
    (r"\frac{1}{2}",  r"\dfrac{1}{2}",       True),
    (r"\frac{1}{2}",  r"0.5",                True),
    (r"\frac{1}{2}",  r"1/2",                True),
    (r"\frac{1}{2}",  r"\frac{2}{4}",        True),
    (r"\frac{1}{2}",  r"\frac{1}{3}",        False),
    (r"-\frac{1}{21}", r"-1/21",             True),
    (r"-\frac{1}{21}", r"\frac{-1}{21}",     True),
    (r"-\frac{1}{21}", r"\frac{1}{21}",      False),
    (r"\frac{7}{2}",  r"3.5",                True),
    (r"\frac{9 \sqrt{23}}{23}", r"\frac{9}{\sqrt{23}}", True),
    (r"\frac{9 \sqrt{23}}{23}", r"\frac{9\sqrt{23}}{23}", True),
    (r"1-\frac{2}{\pi}", r"1 - 2/\pi",       True),
    (r"1-\frac{2}{\pi}", r"\frac{\pi-2}{\pi}", True),
    (r"420261",       r"420{,}261",          True),
    (r"420261",       r"420261.0",           True),
    (r"48",           r"48",                 True),
    (r"48",           r"49",                 False),
    (r"3840",         r"\text{3840}",        True),

    # exact forms drawn from the actual HMMT answers, against the shapes a model
    # plausibly writes them in
    (r"\frac{-1+\sqrt{17}}{2}, \frac{-1-\sqrt{17}}{2}",
     r"\frac{-1-\sqrt{17}}{2}, \frac{-1+\sqrt{17}}{2}", True),   # reordered
    (r"\frac{-1+\sqrt{17}}{2}, \frac{-1-\sqrt{17}}{2}",
     r"\frac{-1\pm\sqrt{17}}{2}", True),                          # written with \pm
    (r"\frac{-1+\sqrt{17}}{2}, \frac{-1-\sqrt{17}}{2}",
     r"\frac{-1+\sqrt{17}}{2}", False),                           # only one root
    (r"(3+\sqrt{6})^{-1/3}", r"\frac{1}{\sqrt[3]{3+\sqrt{6}}}", True),
    (r"(3+\sqrt{6})^{-1/3}", r"(3+\sqrt{6})^{-\frac{1}{3}}", True),
    (r"50(1 - \frac{1}{2^{101} - 1})", r"50 - \frac{50}{2^{101}-1}", True),
    (r"50(1 - \frac{1}{2^{101} - 1})", r"\frac{50(2^{101}-2)}{2^{101}-1}", True),
    (r"\frac{2}{5}", r"0.4", True),
    (r"1-\frac{2}{\pi}", r"1-\frac{2}{\pi}\,", True),

    # DOCUMENTED BEHAVIOUR, not a defect: math_verify compares numerically within
    # a tolerance, so a truncated decimal matches an exact form. It makes the
    # grader more permissive, equally so for every model.
    (r"-\frac{1}{21}", r"-0.047619047619", True),
]


def boxed(x):
    return f"Some reasoning here.\n\nThe answer is $\\boxed{{{x}}}$."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--integer-answer", action="store_true")
    ap.add_argument("--show", type=int, default=12, help="max failures to print")
    a = ap.parse_args()

    rows = [json.loads(l) for f in a.data for l in open(f)]
    print(f"dataset: {len(rows)} problems from {len(a.data)} file(s)")
    ints = sum(1 for r in rows if re.fullmatch(r"-?\d+", r["answer"].strip()))
    print(f"  integer answers {ints}/{len(rows)}, exact forms {len(rows)-ints}\n")

    fails = {"identity": [], "specificity": [], "thinking": [], "equivalence": []}

    # 1. identity
    for r in rows:
        ok, ext = grade(boxed(r["answer"]), r["answer"], a.integer_answer)
        if not ok:
            fails["identity"].append((r["id"], r["answer"], ext))

    # 2. specificity - a sentinel that cannot be equivalent to any real answer.
    # Using a neighbouring problem's answer instead looks stronger but is not a
    # sound assertion: two problems can legitimately share a value, and a gold of
    # "5" against a neighbour's "x=5" is a correct match, not a grader fault.
    SENTINEL = "-999983"
    for r in rows:
        if r["answer"].strip() == SENTINEL:
            continue
        ok, _ = grade(boxed(SENTINEL), r["answer"], a.integer_answer)
        if ok:
            fails["specificity"].append((r["id"], r["answer"], SENTINEL))

    # informational: how often do two different problems' answers match each
    # other? A high rate means the comparator is loose, which inflates scores.
    loose = []
    for i, r in enumerate(rows):
        other = rows[(i + 1) % len(rows)]["answer"]
        if other.strip() == r["answer"].strip():
            continue
        ok, _ = grade(boxed(other), r["answer"], a.integer_answer)
        if ok:
            loose.append((r["id"], r["answer"], other))

    # 3. thinking-mode segment
    for r in rows:
        txt = f"<think>\nA wrong guess: \\boxed{{999999}}\n</think>\n{boxed(r['answer'])}"
        from evaluate import answer_segment
        ok, ext = grade(answer_segment(txt), r["answer"], a.integer_answer)
        if not ok:
            fails["thinking"].append((r["id"], r["answer"], ext))

    # 4. equivalence table
    for gold, variant, want in EQUIVALENCE:
        got, _ = grade(boxed(variant), gold, False)
        if bool(got) != want:
            fails["equivalence"].append((gold, variant, want, bool(got)))

    n = len(rows)
    totals = [("identity", n), ("specificity", n), ("thinking", n),
              ("equivalence", len(EQUIVALENCE))]
    print("=== results ===")
    worst = 0
    for name, total in totals:
        bad = len(fails[name])
        worst = max(worst, bad)
        rate = 100 * (total - bad) / total
        print(f"  {name:12} {total-bad:4}/{total:<4} pass  ({rate:5.1f}%)"
              f"{'' if not bad else '   <-- ' + str(bad) + ' FAILED'}")

    print(f"\n  cross-answer matches (informational, not a failure): "
          f"{len(loose)}/{n}")
    for x in loose[: a.show]:
        print("     ", x)

    for name, _ in totals:
        if not fails[name]:
            continue
        print(f"\n--- {name} failures (first {a.show}) ---")
        for f in fails[name][: a.show]:
            print("   ", f)

    print()
    print("VERDICT:", "grading path is sound for this dataset"
          if worst == 0 else "GRADING PATH IS NOT SAFE FOR THIS DATASET")
    return 1 if worst else 0


if __name__ == "__main__":
    raise SystemExit(main())
