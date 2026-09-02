"""Collect the training curves from evaluation summaries.

Emits one JSON describing every (config, band, benchmark) series as points of
(student_rollouts, score). No plotting dependency - the figure is rendered
separately, so this can run inside the training environment without touching it.

Rollout count is derived from the checkpoint step: every run consumes 512
rollouts per step by construction, which is what puts all nine curves on a
common x-axis.

  baseline (step 0)  outputs/benchmarks/{config}__{task}/summary.json
  checkpoints        outputs/grpo/{config}__{band}__step{N}__{task}/summary.json
  teacher runs       outputs/frontier-model/run_*/final_eval/
                       {config}__teacher__step{N}__{task}/summary.json

A teacher run is emitted as the pseudo-band `teacher` so it sits beside the
fixed bands on the same axes. Its checkpoints are numbered differently: a
baseline run is one verl job whose global_step counts optimiser steps from 1
(5/10/15/20), while a teacher run is twenty separate jobs each producing
global_step_1, so the milestone is the loop's own step index from 0 (4/9/14/19).
Both mean the same rollout counts, and both are converted here.

Identity comes from the directory name, not a filename: one evaluation is one
directory holding summary.json, records.jsonl and generations.jsonl.
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

# The optional variant suffix carries a deliberate re-run of the same band under
# different settings - `__g32` is the matched-group-size control, which exists
# because the default mapping ties group size to the band and so also ties how
# many distinct problems a step covers (512/G). Without capturing it here those
# runs are silently dropped, since the band group cannot span the extra field.
# The band group must be lazy up to __step, not [^_]+: band slugs contain
# underscores of their own (pass1_eq_0, pass1_eq_1), and a class that excludes
# them simply fails to match - the run then falls through to BASE, whose cfg
# group DOES admit underscores, and every checkpoint of that run is silently
# filed as a baseline. Anchoring BASE against __step is what stops that.
# The band group must be lazy up to __step: band slugs contain underscores of
# their own (pass1_eq_0), and a class that excludes them fails to match, sending
# the run to BASE - whose cfg group does admit them - so every checkpoint of that
# run is silently filed as a baseline.
CKPT = re.compile(r"^(?P<cfg>.+?)__(?P<band>pass1_.+?)(?:__(?P<variant>g\d+))?"
                  r"__step(?P<step>\d+)__(?P<task>\w+)$")
BASE = re.compile(r"^(?P<cfg>[\w.-]+?)__(?P<task>math500|aime|hmmt)$")
TEACHER = re.compile(r"^(?P<cfg>.+?)__teacher__step(?P<step>\d+)__(?P<task>\w+)$")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs", default=str(ROOT / "outputs"))
    ap.add_argument("--out", default=str(ROOT / "outputs" / "analysis" / "curves.json"))
    a = ap.parse_args()

    outdir = Path(a.outputs)
    baselines, points = {}, defaultdict(list)

    for f in sorted(outdir.glob("*/*/summary.json")):
        s = json.loads(f.read_text())
        name = f.parent.name
        m = CKPT.match(name)
        if m:
            task = m["task"]
            if task not in HEADLINE:
                continue
            metric = HEADLINE[task]
            points[(m["cfg"], m["band"], m["variant"] or "", task)].append({
                "step": int(m["step"]),
                "rollouts": int(m["step"]) * ROLLOUTS_PER_STEP,
                "score": s.get(metric),
                "stderr": s.get(f"{metric}_stderr"),
                "pass@1": s.get("pass@1"),
                "truncation_rate": s.get("truncation_rate"),
                "no_answer_rate": s.get("no_answer_rate"),
                "mean_gen_tokens": s.get("mean_gen_tokens"),
            })
            continue
        b = BASE.match(name)
        if b and "__step" not in name and "pass1_" not in name:
            task = b["task"]
            baselines[(b["cfg"], task)] = {
                "rollouts": 0,
                "score": s.get(HEADLINE[task]),
                "stderr": s.get(f"{HEADLINE[task]}_stderr"),
                "pass@1": s.get("pass@1"),
                "truncation_rate": s.get("truncation_rate"),
                "no_answer_rate": s.get("no_answer_rate"),
                "mean_gen_tokens": s.get("mean_gen_tokens"),
            }

    # Teacher runs live one level deeper, under the run directory that also
    # holds their transcripts and per-step reference tests.
    for f in sorted(outdir.glob("frontier-model/*/final_eval/*/summary.json")):
        m = TEACHER.match(f.parent.name)
        if not m or m["task"] not in HEADLINE:
            continue
        s_ = json.loads(f.read_text())
        metric = HEADLINE[m["task"]]
        points[(m["cfg"], "teacher", "", m["task"])].append({
            "step": int(m["step"]) + 1,          # loop index 0-based -> steps taken
            "rollouts": (int(m["step"]) + 1) * ROLLOUTS_PER_STEP,
            "score": s_.get(metric),
            "stderr": s_.get(f"{metric}_stderr"),
            "pass@1": s_.get("pass@1"),
            "truncation_rate": s_.get("truncation_rate"),
            "no_answer_rate": s_.get("no_answer_rate"),
            "mean_gen_tokens": s_.get("mean_gen_tokens"),
        })

    # The per-step reference tests: the same fifth of each benchmark, run after
    # every training update rather than only at the four milestones. Same
    # sampling, same verifier and same decoding as the milestone evaluations --
    # copied from configs/eval/ into the teacher config for exactly this reason
    # -- so the two are independent draws of one quantity, and this one has 21
    # points where the milestone split has 5.
    per_step = defaultdict(list)
    for run in sorted(outdir.glob("frontier-model/run_*")):
        cfg = None
        cj = run / "pipeline" / "config.resolved.json"
        if cj.exists():
            base = json.loads(cj.read_text())["student"]["base_model"]
            cfg = {"unsloth/Llama-3.2-3B-Instruct": "llama32-3b"}.get(base)
            if cfg is None:               # both Qwen runs share a base model
                think = json.loads(cj.read_text())["evaluation"].get("enable_thinking")
                cfg = "qwen3-4b-think" if think else "qwen3-4b-nothink"
        if not cfg:
            continue
        for task in BENCHMARKS:
            metric = HEADLINE[task]
            for d, ro in [(run / "test_base" / task, 0)] + [
                    (run / f"step_{k}" / "test" / task, (k + 1) * ROLLOUTS_PER_STEP)
                    for k in range(64)]:
                f = d / "summary.json"
                if not f.exists():
                    continue
                sm = json.loads(f.read_text())
                per_step[(cfg, task)].append({
                    "rollouts": ro, "score": sm.get(metric),
                    "stderr": sm.get(f"{metric}_stderr"),
                    "n_problems": sm.get("n_problems"),
                })
    for v in per_step.values():
        v.sort(key=lambda x: x["rollouts"])

    series = []
    for (cfg, band, variant, task), pts in sorted(points.items()):
        pts.sort(key=lambda x: x["rollouts"])
        base = baselines.get((cfg, task))
        # the rollout-0 point is the untrained model, shared by all bands of a
        # configuration; if it is missing the curve has no anchor
        full = ([dict(base, step=0)] if base else []) + pts
        # A run that reached fewer than 20 steps is not a shorter curve of the
        # same experiment - its last point is not the 10,240-rollout endpoint the
        # other cells report, and it must not be read as one.
        series.append({"config": cfg, "band": band, "variant": variant,
                       "task": task, "metric": HEADLINE[task],
                       "has_baseline": base is not None,
                       "reaches_10240": any(p["rollouts"] == 20 * ROLLOUTS_PER_STEP
                                            for p in pts),
                       "points": full})

    doc = {"rollouts_per_step": ROLLOUTS_PER_STEP,
           "per_step_reference": {f"{c}__{t}": v
                                  for (c, t), v in sorted(per_step.items())},
           "benchmarks": BENCHMARKS,
           "headline_metric": HEADLINE,
           "baselines": {f"{c}__{t}": v for (c, t), v in sorted(baselines.items())},
           "series": series}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(doc, indent=2))

    print(f"wrote {a.out}")
    print(f"  baselines: {len(baselines)}   series: {len(series)}")
    for k, v in sorted(per_step.items()):
        print(f"  per-step reference {k[0]}__{k[1]}: {len(v)} points")
    for s in series:
        xs = [p["rollouts"] for p in s["points"]]
        ys = [None if p["score"] is None else round(100 * p["score"], 1) for p in s["points"]]
        flag = "" if s["has_baseline"] else "  <-- NO BASELINE"
        if not s["reaches_10240"]:
            flag += "  <-- INCOMPLETE, no 10,240-rollout endpoint"
        name = s["band"] + ("/" + s["variant"] if s["variant"] else "")
        print(f"  {s['config']:18} {name:21} {s['task']:8} {s['metric']:7} "
              f"x={xs} y={ys}{flag}")


if __name__ == "__main__":
    main()
