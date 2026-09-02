"""Put each teacher run's curve next to the GRPO baseline bands it has to beat.

The two are directly comparable by construction: both spend 512 rollouts per
step for 20 steps, both are evaluated at 2,560 / 5,120 / 7,680 / 10,240 rollouts
on the full MATH-500, AIME and HMMT, and both are graded by eval/verifiers.
What differs is where the problems came from -- a fixed difficulty band, or a
frontier model choosing them step by step. That is the experiment.

    python tools/compare_teacher.py                      # every model
    python tools/compare_teacher.py --model llama32-3b

Reads outputs/grpo/<exp>__step<N>__<task>/summary.json for the bands,
outputs/benchmarks/<model>__<task>/summary.json for the 0-rollout point, and
outputs/frontier-model/run_*/final_eval/ for the teacher runs. A cell with no
summary prints as `--`; nothing is interpolated.
"""
import argparse
import json
import math
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"

# The metric each benchmark is reported on, per the project's convention.
HEADLINE = {"math500": "pass@1", "aime": "pass@4", "hmmt": "pass@4"}
TASKS = list(HEADLINE)
MILESTONES = [(4, 2560), (9, 5120), (14, 7680), (19, 10240)]

# Bands in the order they are worth reading: the three that span difficulty,
# then the matched control, then the two degenerate ends.
BAND_ORDER = ["05-15pct", "12-25pct", "40-60pct", "37-62pct",
              "40-60pct__g32", "37-62pct__g32", "85-95pct", "75-87pct",
              "eq_0", "eq_1"]


def _load(p):
    try:
        return json.loads(Path(p).read_text())
    except (OSError, json.JSONDecodeError):
        return None


def _cell(summary, task):
    if not summary:
        return None
    k = HEADLINE[task]
    return summary.get(k), summary.get(f"{k}_stderr")


def baseline_point(model, task):
    return _cell(_load(OUT / "benchmarks" / f"{model}__{task}" / "summary.json"), task)


def band_curves(model):
    """{band: {task: {rollouts: (value, stderr)}}} from outputs/grpo/."""
    out = {}
    pat = re.compile(rf"^{re.escape(model)}__pass1_(?P<band>.+?)__step(?P<step>\d+)__(?P<task>\w+)$")
    for d in sorted((OUT / "grpo").glob(f"{model}__pass1_*")):
        m = pat.match(d.name)
        if not m:
            continue
        step, task = int(m["step"]), m["task"]
        ro = dict((s, r) for s, r in MILESTONES).get(step)
        if ro is None or task not in HEADLINE:
            continue
        c = _cell(_load(d / "summary.json"), task)
        if c:
            out.setdefault(m["band"], {}).setdefault(task, {})[ro] = c
    return out


def teacher_curve(model):
    """{task: {rollouts: (value, stderr)}} for the newest run of this model."""
    runs = []
    for fe in sorted((OUT / "frontier-model").glob("run_*/final_eval")):
        if any(fe.glob(f"{model}__teacher__step*")):
            runs.append(fe)
    if not runs:
        return {}, None
    fe = runs[-1]
    out = {}
    for step, ro in MILESTONES:
        for task in TASKS:
            c = _cell(_load(fe / f"{model}__teacher__step{step}__{task}" /
                            "summary.json"), task)
            if c:
                out.setdefault(task, {})[ro] = c
    return out, fe.parent.name


def fmt(c):
    return "  --  " if not c else f"{100*c[0]:5.1f}"


def delta(end, base):
    """End-minus-base with the stderr of the difference, both in points."""
    if not end or not base:
        return "   --   "
    d = 100 * (end[0] - base[0])
    se = 100 * math.sqrt((end[1] or 0) ** 2 + (base[1] or 0) ** 2)
    sig = "*" if se and abs(d) > 2 * se else " "
    return f"{d:+5.1f}+-{se:4.1f}{sig}"


def report(model):
    bands = band_curves(model)
    teacher, run_name = teacher_curve(model)
    if not bands and not teacher:
        return
    print(f"\n=== {model} ===")
    if run_name:
        print(f"    teacher run: {run_name}")
    order = [b for b in BAND_ORDER if b in bands] + \
            [b for b in sorted(bands) if b not in BAND_ORDER]

    for task in TASKS:
        base = baseline_point(model, task)
        print(f"\n  {task}  ({HEADLINE[task]}, %)")
        print("    curriculum          " +
              "".join(f"{r:>8}" for _, r in MILESTONES) +
              "     end - base")
        print(f"    {'(base model)':<20}" + fmt(base).rjust(8) +
              " " * (8 * (len(MILESTONES) - 1)))
        for band in order:
            pts = bands[band].get(task, {})
            row = "".join(fmt(pts.get(r)).rjust(8) for _, r in MILESTONES)
            print(f"    {band:<20}{row}    {delta(pts.get(10240), base)}")
        if teacher.get(task):
            pts = teacher[task]
            row = "".join(fmt(pts.get(r)).rjust(8) for _, r in MILESTONES)
            print(f"    {'TEACHER':<20}{row}    {delta(pts.get(10240), base)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", action="append", default=None,
                    help="repeatable; default is all three")
    a = ap.parse_args()
    for m in a.model or ["llama32-3b", "qwen3-4b-nothink", "qwen3-4b-think"]:
        report(m)
    print("\n  * marks |change| > 2 standard errors of the change. "
          "Absent cells are runs that\n"
          "  have not been evaluated, not zeros.")


if __name__ == "__main__":
    main()
