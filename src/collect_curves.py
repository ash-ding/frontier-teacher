"""Collect the training curves from evaluation summaries.

Emits one JSON describing every (config, band, benchmark) series as points of
(student_rollouts, score). No plotting dependency - the figure is rendered
separately, so this can run inside the training environment without touching it.

Rollout count is derived from the checkpoint step: every run consumes 512
rollouts per step by construction, which is what puts all nine curves on a
common x-axis.

  baseline (step 0)  outputs/{config}__{task}.summary.json
  checkpoints        outputs/{config}__{band}__step{N}__{task}.summary.json
"""
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ROLLOUTS_PER_STEP = 512
BENCHMARKS = ["math500", "aime", "hmmt"]
HEADLINE = {"math500": "pass@1", "aime": "pass@4", "hmmt": "pass@4"}

CKPT = re.compile(r"^(?P<cfg>.+?)__(?P<band>pass1_[^_]+)__step(?P<step>\d+)__(?P<task>\w+)\.summary\.json$")
BASE = re.compile(r"^(?P<cfg>[\w.-]+)__(?P<task>math500|aime|hmmt)\.summary\.json$")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs", default=str(ROOT / "outputs"))
    ap.add_argument("--out", default=str(ROOT / "outputs" / "curves.json"))
    a = ap.parse_args()

    outdir = Path(a.outputs)
    baselines, points = {}, defaultdict(list)

    for p in sorted(outdir.glob("*.summary.json")):
        s = json.loads(p.read_text())
        m = CKPT.match(p.name)
        if m:
            task = m["task"]
            if task not in HEADLINE:
                continue
            metric = HEADLINE[task]
            points[(m["cfg"], m["band"], task)].append({
                "step": int(m["step"]),
                "rollouts": int(m["step"]) * ROLLOUTS_PER_STEP,
                "score": s.get(metric),
                "stderr": s.get(f"{metric}_stderr"),
                "pass@1": s.get("pass@1"),
                "truncation_rate": s.get("truncation_rate"),
                "no_answer_rate": s.get("no_answer_rate"),
            })
            continue
        b = BASE.match(p.name)
        if b:
            task = b["task"]
            baselines[(b["cfg"], task)] = {
                "rollouts": 0,
                "score": s.get(HEADLINE[task]),
                "stderr": s.get(f"{HEADLINE[task]}_stderr"),
                "pass@1": s.get("pass@1"),
                "truncation_rate": s.get("truncation_rate"),
                "no_answer_rate": s.get("no_answer_rate"),
            }

    series = []
    for (cfg, band, task), pts in sorted(points.items()):
        pts.sort(key=lambda x: x["rollouts"])
        base = baselines.get((cfg, task))
        # the rollout-0 point is the untrained model, shared by all bands of a
        # configuration; if it is missing the curve has no anchor
        full = ([dict(base, step=0)] if base else []) + pts
        series.append({"config": cfg, "band": band, "task": task,
                       "metric": HEADLINE[task], "has_baseline": base is not None,
                       "points": full})

    doc = {"rollouts_per_step": ROLLOUTS_PER_STEP,
           "benchmarks": BENCHMARKS,
           "headline_metric": HEADLINE,
           "baselines": {f"{c}__{t}": v for (c, t), v in sorted(baselines.items())},
           "series": series}
    Path(a.out).write_text(json.dumps(doc, indent=2))

    print(f"wrote {a.out}")
    print(f"  baselines: {len(baselines)}   series: {len(series)}")
    for s in series:
        xs = [p["rollouts"] for p in s["points"]]
        ys = [None if p["score"] is None else round(100 * p["score"], 1) for p in s["points"]]
        flag = "" if s["has_baseline"] else "  <-- NO BASELINE"
        print(f"  {s['config']:18} {s['band']:16} {s['task']:8} {s['metric']:7} "
              f"x={xs} y={ys}{flag}")


if __name__ == "__main__":
    main()
