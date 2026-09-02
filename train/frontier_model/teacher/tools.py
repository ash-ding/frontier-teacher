"""Orchestrator-side executors for the two teacher decisions, plus context.

Nothing here is Claude Code -- these are the Python actions the orchestrator runs
AFTER the teacher has written its decision.json / data.jsonl:

  render_context()      assemble the teacher's task message: read-only context +
                        the full trajectory of every prior step.
  rematerialize_ro()    `git checkout` the read-only source back to pristine before
                        any real action, so a teacher edit is always discarded.
  run_evaluation()      reuse evaluate.py UNMODIFIED on the teacher's data.jsonl.
  run_train()           convert (no verification) + one GRPO update via
                        run_teacher_step.sh; return the advanced checkpoint path.

Every executor returns a plain dict of the metrics the orchestrator writes into
result.json / metrics.jsonl. Full stdout+stderr of each subprocess is streamed to
the step's log file; we never fabricate a number we did not measure.
"""
import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path

import yaml

from . import protocol

ROOT = Path(__file__).resolve().parent.parent.parent.parent
# src/ split into eval/ (measurement) and train/ (learning) - this module
# reuses the evaluation harness unmodified, so it points at eval/.
EVAL_DIR = ROOT / "eval"
SCRIPTS = ROOT / "scripts"


# ----------------------------------------------------------- read-only context

def rematerialize_ro(repo_root, paths, log=print):
    """Discard any teacher edit to read-only source: `git checkout -- <paths>`.

    Run before EVERY real train/eval action (enforcement layer 2 of 3). Only
    touches tracked files; an untracked path is skipped with a note rather than
    aborting the run.
    """
    tracked = []
    for p in paths:
        rc = subprocess.run(["git", "ls-files", "--error-unmatch", p],
                            cwd=str(repo_root), stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL).returncode
        (tracked.append(p) if rc == 0 else log(f"  [ro] skip untracked {p}"))
    if tracked:
        subprocess.run(["git", "checkout", "--", *tracked], cwd=str(repo_root), check=True)
        log(f"  [ro] re-materialised {len(tracked)} read-only path(s) from git")


def _trajectory_summary(run_dir, step):
    """One compact block per CLOSED prior step, read from its result.json.

    A step now closes on a train sub-action, so its result.json is a STEP SUMMARY:
    the evaluations taken within it (`evals`), then the terminating `train`.
    """
    lines = []
    for k in range(step):
        rp = run_dir / f"step_{k}" / "result.json"
        if not rp.exists():
            continue
        try:
            r = json.loads(rp.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        lines.append(f"  step {k}: status={r.get('status')} "
                     f"n_evals={r.get('n_evals')}")
        for e in (r.get("evals") or []):
            lines.append(
                f"      eval_{e.get('action_index')}: n={e.get('n_problems')} "
                f"pass@1={e.get('pass_at_1')} pass@k={e.get('pass_at_k')} "
                f"trunc={e.get('truncation_rate')} no_answer={e.get('no_answer_rate')}")
        t = r.get("train")
        if t:
            lines.append(f"      train: n={t.get('n_problems')} metrics={t.get('metrics')}")
    return "\n".join(lines) if lines else "  (no prior steps -- this is step 0)"


def _current_step_evals_summary(evals):
    """Render the evaluations ALREADY taken within the current (open) step."""
    if not evals:
        return "  (none yet -- this is the first decision of the step)"
    lines = []
    for e in evals:
        lines.append(
            f"  eval_{e.get('action_index')}: n={e.get('n_problems')} "
            f"pass@1={e.get('pass_at_1')} pass@k={e.get('pass_at_k')} "
            f"trunc={e.get('truncation_rate')} no_answer={e.get('no_answer_rate')}")
    return "\n".join(lines)


def render_context(run_dir, step, config, latest_ckpt, repo_root, *,
                   cwd, action_index=0, current_step_evals=None,
                   max_evals_per_step=10):
    """Build the teacher's per-sub-action task message (a plain string).

    A step is an inner loop of sub-actions (evaluate* then train). This message is
    rendered once per sub-action and mounts, by absolute path so the teacher can
    Read/cat/grep them:
      * the curated reference training data (comparison point, context only),
      * the verifier + training/eval source (read-only),
      * the FULL trajectory of every CLOSED prior step (each step's evals + train),
      * the sub-actions ALREADY taken WITHIN THE CURRENT step (so the teacher can
        decide evaluate-again vs. train).
    Imposes no curriculum strategy -- the standing role/protocol lives in the
    system prompt; this message only supplies the step, the paths, and the eval cap.
    """
    run_dir = Path(run_dir).resolve()
    cwd = Path(cwd).resolve()
    current_step_evals = current_step_evals or []
    ref = (repo_root / config["reference_data"]).resolve()
    ro = [str((repo_root / p).resolve()) for p in config["read_only_paths"]]
    remaining = max_evals_per_step - len(current_step_evals)

    prior_paths = []
    for k in range(step):
        sd = run_dir / f"step_{k}"
        if sd.exists():
            prior_paths.append(str(sd) + "/   (browse: result.json, eval_*/ , train/)")
    # sub-action dirs of the CURRENT step already on disk
    cur_dir = run_dir / f"step_{step}"
    cur_paths = []
    if cur_dir.exists():
        for e in current_step_evals:
            ed = cur_dir / f"eval_{e.get('action_index')}"
            if ed.exists():
                cur_paths.append(str(ed) + "/   (decision.json, data.jsonl, "
                                 "eval.summary.json, eval.records.jsonl, eval.log)")

    msg = [
        f"You are on STEP {step} of the curriculum loop. One step is ONE training",
        "update: you may evaluate the current checkpoint any number of times (up to",
        "the cap below), and the step CLOSES only when you choose to train.",
        "",
        f"Within THIS step you have already evaluated {len(current_step_evals)} time(s). "
        f"You may evaluate at most {max_evals_per_step} time(s) per step; "
        f"{max(remaining, 0)} evaluation(s) remain before you MUST train.",
        "",
        "Make ONE decision (evaluation | train) for this turn and write your two",
        f"files into your current working directory ({cwd}):",
        "  - decision.json",
        "  - data.jsonl",
        f'Use step number {step} in decision.json: {{"step": {step}, "decision": "..."}}',
        "",
        "The student's current checkpoint (what an evaluation measures, and what a",
        f"train step would update) is:\n  {latest_ckpt}",
        "All evaluations within this step run on THIS same checkpoint; only a train",
        "advances it.",
        "",
        "READ-ONLY reference: curated, difficulty-graded training data already",
        "shipped with the project (the comparison point for standard GRPO). Browse",
        f"it for context only:\n  {ref}",
        "",
        "READ-ONLY project source (the verifier and the training/eval code that",
        "will run your decision). Context only -- any edit is discarded:",
        *[f"  {p}" for p in ro],
        "",
        "The FULL trajectory of every CLOSED prior step (evals + terminating train):",
        _trajectory_summary(run_dir, step),
        "",
        "Evaluations you have ALREADY run within the CURRENT (open) step:",
        _current_step_evals_summary(current_step_evals),
        "",
        "Prior-step directories you may open directly:",
        *([f"  {p}" for p in prior_paths] or ["  (none yet)"]),
        "",
        "Current-step evaluation directories you may open directly:",
        *([f"  {p}" for p in cur_paths] or ["  (none yet)"]),
        "",
        "Now write decision.json and data.jsonl, then stop.",
    ]
    return "\n".join(msg)


# --------------------------------------------------------------------- helpers

def _stream(argv, log_path, cwd=None, timeout_s=None, extra_env=None):
    """Run a subprocess, tee stdout+stderr to log_path, return (rc, timed_out)."""
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    t0 = time.time()
    with open(log_path, "w") as logf:
        logf.write(f"$ {' '.join(argv)}\n\n")
        logf.flush()
        proc = subprocess.Popen(argv, cwd=cwd, env=env, stdout=logf,
                                stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
        try:
            rc = proc.wait(timeout=timeout_s)
            timed_out = False
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            rc, timed_out = (proc.returncode if proc.returncode is not None else -9), True
    return rc, timed_out, time.time() - t0


_METRIC_RE = re.compile(r"([A-Za-z0-9_./@\-]+):\s*(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)")


def _parse_verl_metrics(log_path):
    """Best-effort parse of verl's last per-step metrics line from train.log.

    verl prints one `step:N - key:val - key:val ...` line per optimizer step. We
    take the LAST such line and return {key: float}. Absent (very young / failed
    run) -> {}. We never invent metrics; a missing key is simply not reported.
    """
    try:
        lines = Path(log_path).read_text(errors="replace").splitlines()
    except OSError:
        return {}
    for line in reversed(lines):
        if "step:" in line and " - " in line and ":" in line:
            pairs = dict((k, float(v)) for k, v in _METRIC_RE.findall(line))
            if len(pairs) >= 3:
                return pairs
    return {}


def _data_file_rel_to_data(step_data, repo_root):
    """evaluate.py resolves a task's data_file under ROOT/data/<name>, so express
    the workspace data.jsonl relative to that dir (a `..`-prefixed path). No
    symlink and no edit to evaluate.py required."""
    return os.path.relpath(Path(step_data).resolve(), (repo_root / "data").resolve())


# ------------------------------------------------------------------ evaluation

def write_eval_config(step_dir, step, model_path, config, step_data, repo_root,
                      action_index=0):
    """Clone the teacher_eval task template into a full evaluate.py config whose
    single task points at this sub-action's data.jsonl. Does NOT touch configs/*."""
    base = dict(config.get("eval_base", {}))
    tmpl = dict(config["teacher_eval"])
    data_rel = _data_file_rel_to_data(step_data, repo_root)
    cfg = {
        "name": f"teacher_step{step}_eval{action_index}",
        "model": str(model_path),
        "temperature": base.get("temperature", 0.6),
        "top_p": base.get("top_p", 0.9),
        "top_k": base.get("top_k", -1),
        "seed": base.get("seed", 1234),
        "tensor_parallel_size": base.get("tensor_parallel_size", 1),
        "gpu_memory_utilization": base.get("gpu_memory_utilization", 0.90),
        "tasks": {
            "teacher_eval": {
                "n": tmpl.get("n", 4),
                "max_tokens": tmpl.get("max_tokens", 4096),
                "max_model_len": tmpl.get("max_model_len", 8192),
                "data_file": data_rel,
                "integer_answer": tmpl.get("integer_answer", False),
                "headline_metric": "pass@1",
            }
        },
    }
    path = Path(step_dir) / "config.eval.yaml"
    protocol.atomic_write_text(path, yaml.safe_dump(cfg, sort_keys=False))
    return path


def run_evaluation(step_dir, step, model_path, config, repo_root, timeout_s,
                   action_index=0):
    """Evaluate the current checkpoint on the teacher's data via evaluate.py.

    `step_dir` is this eval sub-action's own dir (step_<N>/eval_<M>/). Reuses
    evaluate.py unmodified: `--limit 0` (uncapped); generations are always saved,
    `--out step_dir`, `--name teacher_step<N>_eval<M>`. Copies the tag-named
    summary/records to the canonical eval.summary.json / eval.records.jsonl in the
    sub-action dir. Returns a dict of eval stats (weights unchanged).
    """
    step_dir = Path(step_dir)
    step_data = step_dir / "data.jsonl"
    eval_cfg = write_eval_config(step_dir, step, model_path, config, step_data,
                                 repo_root, action_index)
    name = f"teacher_step{step}_eval{action_index}"
    tag = f"{name}__teacher_eval"

    argv = [
        "python", str(EVAL_DIR / "evaluate.py"),
        "--config", str(eval_cfg),
        "--weights", str(model_path),
        "--name", name,
        "--limit", "0",
        "--out", str(step_dir),
    ]
    rc, timed_out, wall = _stream(argv, step_dir / "eval.log", cwd=str(repo_root),
                                  timeout_s=timeout_s)

    summ_src = step_dir / f"{tag}.summary.json"
    recs_src = step_dir / f"{tag}.records.jsonl"
    stats = {"status": "ok" if (rc == 0 and not timed_out) else "error",
             "action_wallclock_s": round(wall, 1), "returncode": rc,
             "timed_out": timed_out}
    if summ_src.exists():
        shutil.copyfile(summ_src, step_dir / "eval.summary.json")
        if recs_src.exists():
            shutil.copyfile(recs_src, step_dir / "eval.records.jsonl")
        s = json.loads(summ_src.read_text())
        stats["eval"] = {
            "n_problems": s.get("n_problems"),
            "pass_at_1": s.get("pass@1"),
            "pass_at_k": s.get(f"pass@{s.get('samples_per_problem')}") or s.get("pass@4"),
            "truncation_rate": s.get("truncation_rate"),
            "no_answer_rate": s.get("no_answer_rate"),
        }
    else:
        stats["status"] = "error"
        stats["eval"] = None
        stats["error"] = "evaluate.py produced no summary.json"
    return stats


def eval_stats_from_summary(summary_path):
    """Reconstruct the compact eval-stats dict from a saved eval.summary.json.

    Used on idempotent restart to recover an already-completed eval sub-action's
    numbers without re-running evaluate.py. Mirrors the extraction in
    run_evaluation() exactly so a resumed step reads identically to a fresh one.
    """
    s = json.loads(Path(summary_path).read_text())
    return {
        "n_problems": s.get("n_problems"),
        "pass_at_1": s.get("pass@1"),
        "pass_at_k": s.get(f"pass@{s.get('samples_per_problem')}") or s.get("pass@4"),
        "truncation_rate": s.get("truncation_rate"),
        "no_answer_rate": s.get("no_answer_rate"),
    }


# ----------------------------------------------------------------------- train

def run_train(step_dir, step, model_path, config, repo_root, timeout_s, n_gpus,
              max_problems=16):
    """One real GRPO update on the teacher's problems.

    Truncate data.jsonl to the first `max_problems` rows -> convert via
    to_verl_teacher_dataset (teacher answer -> ground_truth, NO verification) ->
    run_teacher_step.sh for one optimizer step on the latest checkpoint. Saves the
    hf checkpoint, deletes the FSDP world-size shards, and returns the advanced
    checkpoint path.
    """
    step_dir = Path(step_dir)
    rows = protocol.read_jsonl(step_dir / "data.jsonl")
    truncated_extra = max(0, len(rows) - max_problems)
    rows = rows[:max_problems]

    # convert in-process (deterministic, no subprocess) so a converter error is a
    # clean ProtocolError rather than a shell exit code.
    import sys
    sys.path.insert(0, str(EVAL_DIR))
    from to_verl_teacher_dataset import convert  # noqa: E402
    verl_rows = convert(rows)
    train_file = step_dir / "train.verl.jsonl"
    protocol.atomic_write_jsonl(train_file, verl_rows)

    ckpt_dir = step_dir / "ckpt"
    argv = [
        "bash", str(SCRIPTS / "run_teacher_step.sh"),
        config.get("grpo_config", "llama32-3b"),
        str(train_file), str(model_path), str(ckpt_dir), str(n_gpus),
    ]
    rc, timed_out, wall = _stream(argv, step_dir / "train.log", cwd=str(repo_root),
                                  timeout_s=timeout_s)

    hf_path = ckpt_dir / "global_step_1" / "actor" / "huggingface"
    ok = rc == 0 and not timed_out and hf_path.exists()

    # disk discipline: drop the 26 GB FSDP shards, keep only huggingface/ (the next
    # step's model.path). resume_mode=disable means shards are never needed again.
    deleted_shards = 0
    if ok:
        for shard in (ckpt_dir / "global_step_1" / "actor").glob("model_world_size_*"):
            if shard.is_dir():
                shutil.rmtree(shard, ignore_errors=True)
            else:
                shard.unlink(missing_ok=True)
            deleted_shards += 1

    metrics = _parse_verl_metrics(step_dir / "train.log")
    stats = {
        "status": "ok" if ok else "error",
        "action_wallclock_s": round(wall, 1), "returncode": rc, "timed_out": timed_out,
        "train": {
            "n_problems": len(rows),
            "truncated_extra_rows": truncated_extra,
            # verl's clip_ratio is the fraction of rollouts that hit max length
            # (truncated). no_answer is not separable from the train reward, so we
            # report it only when eval measures it -- never fabricate it here.
            "truncation_rate": metrics.get("response_length/clip_ratio"),
            "no_answer_rate": None,
            "metrics": metrics or None,
            "deleted_shards": deleted_shards,
        },
        "new_checkpoint": str(hf_path) if ok else None,
    }
    if not ok:
        stats["error"] = (f"train failed rc={rc} timed_out={timed_out} "
                          f"hf_exists={hf_path.exists()}")
    return stats
