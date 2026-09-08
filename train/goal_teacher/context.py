"""The teacher's per-turn task message, for the goal-directed arm.

Not a wrapper around `tools.render_context`: that function inlines the whole verl
metrics dictionary for every closed step, which by step 19 of an open-ended run
was 59 KB of a 74 KB message -- 79% timing, sequence-length and MFU keys that the
system prompt never asks anyone to read. The three keys it does name are in
there, once per step, buried.

The trajectory here therefore carries those three and points at `train.log` for
the rest, which the teacher opens anyway; the open-ended runs show it running
hundreds of Bash calls per run against its own run directory. Nothing is removed
from the run, only from the paste.

The other difference is the goal. There are no reference tests in this arm, so
the block that carried three benchmark curves is replaced by the one thing this
arm measures between steps: the student's score on the three targets.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "frontier_model"))
from teacher.tools import _current_step_evals_summary  # noqa: E402

# The keys the system prompt names, and what each is for.
TRAIN_KEYS = (("critic/score/mean", "reward"),
              ("critic/advantages/max", "adv_max"),
              ("response_length/clip_ratio", "clip"))


def _fmt(v, nd=3):
    return "n/a" if v is None else f"{v:.{nd}f}"


def _targets_line(d, indent):
    """One line of per-target pass@1, in the order the targets were given."""
    if not d:
        return []
    return [f"{indent}targets: " + "  ".join(
        f"{r['id'].rsplit('-', 1)[-1]} pass@1={_fmt(r.get('pass_at_1'))}" for r in d)]


def _trajectory(run_dir, step):
    lines = []
    base = run_dir / "targets" / "per_target.jsonl"
    if base.exists():
        lines.append("  base model, before any training:")
        lines += _targets_line([json.loads(l) for l in base.open()], " " * 6)
    for k in range(step):
        rp = run_dir / f"step_{k}" / "result.json"
        if not rp.exists():
            continue
        try:
            r = json.loads(rp.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        lines.append(f"  step {k}: status={r.get('status')} n_evals={r.get('n_evals')}")
        for e in (r.get("evals") or []):
            lines.append(
                f"      eval_{e.get('action_index')}: n={e.get('n_problems')} "
                f"pass@1={e.get('pass_at_1')} pass@k={e.get('pass_at_k')} "
                f"trunc={e.get('truncation_rate')} no_answer={e.get('no_answer_rate')}")
        t = r.get("train") or {}
        m = t.get("metrics") or {}
        if t:
            vals = "  ".join(f"{label}={_fmt(m.get(key))}" for key, label in TRAIN_KEYS)
            lines.append(f"      train: n={t.get('n_problems')}  {vals}")
            lines.append(f"          (every other verl metric is in "
                         f"step_{k}/train/train.log)")
        lines += _targets_line(r.get("target_eval"), " " * 6)
    return "\n".join(lines) if lines else "  (no prior steps -- this is step 0)"


def render(run_dir, step, config, latest_ckpt, *, cwd, action_index=0,
           current_step_evals=None, max_evals_per_step=10, targets_file=None,
           pipeline_dir=None):
    run_dir = Path(run_dir).resolve()
    cwd = Path(cwd).resolve()
    current_step_evals = current_step_evals or []
    remaining = max_evals_per_step - len(current_step_evals)
    tb = int(config["train_batch_size"])

    prior = [str(run_dir / f"step_{k}") for k in range(step)
             if (run_dir / f"step_{k}").exists()]
    cur_dir = run_dir / f"step_{step}"
    cur = [str(cur_dir / f"eval_{e.get('action_index')}")
           for e in current_step_evals
           if (cur_dir / f"eval_{e.get('action_index')}").exists()]

    msg = [
        f"STEP {step}, sub-action {action_index}.",
        "",
        f"The student's current checkpoint:\n  {latest_ckpt}",
        "Every evaluation in this step measures these same weights. Only a train",
        "advances them.",
        "",
        f"You have evaluated {len(current_step_evals)} time(s) in this step. "
        f"{max(remaining, 0)} of {max_evals_per_step} remain before this step must",
        "train.",
        "",
        "Results so far in THIS step:",
        _current_step_evals_summary(current_step_evals),
        "",
        "Every closed step before this one:",
        _trajectory(run_dir, step),
        "",
        "Directories you can open:",
        f"  {run_dir}/",
        "      this run",
        *([f"  {pipeline_dir}/",
           "      the frozen config, and the code that runs your decisions"]
          if pipeline_dir else []),
        *([f"  {targets_file}",
           "      the three problems, exactly as given to the student"]
          if targets_file else []),
        *[f"  {p}/" for p in prior],
        *[f"  {p}/" for p in cur],
        "",
        "  Each closed step also has targets/target_traces.json -- what the student",
        "  wrote on the three problems, with no verdict attached.",
        "",
        f"Now write decision.json and data.jsonl into {cwd}, then stop.",
        f'  decision.json: {{"step": {step}, "decision": "evaluation"}} or '
        f'{{"step": {step}, "decision": "train"}}',
        f"  data.jsonl:    any number of problems for an evaluation; for a train, "
        f"EXACTLY train_batch_size = {tb} problems.",
    ]
    return "\n".join(msg)
