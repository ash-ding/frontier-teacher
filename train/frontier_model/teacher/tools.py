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
FM_DIR = ROOT / "train" / "frontier_model"   # the converter and run_teacher_step.sh


def snapshot_pipeline(run_dir, repo_root, config, resolved):
    """Copy the run's frozen configuration and the code it will run into
    <run_dir>/pipeline/, and return that directory.

    The teacher is asked to design a curriculum for a specific student under a
    specific training and evaluation setup. Describing that setup in a prompt
    invites the description and the code to drift apart; copying the code in
    cannot. The copy is also part of the run's record - months later, "what
    exactly ran" is answerable from the run directory alone.

    Source files keep their repo-relative paths under pipeline/, so which file is
    which is obvious without a manifest. These are copies: an edit to one changes
    nothing, and the originals are separately restored from git before every
    action.
    """
    run_dir, repo_root = Path(run_dir), Path(repo_root)
    pdir = run_dir / "pipeline"
    pdir.mkdir(parents=True, exist_ok=True)
    protocol.atomic_write_json(pdir / "config.resolved.json", resolved)

    copied = []
    for rel in config["pipeline_source"]:
        src = repo_root / rel
        if not src.is_file():
            raise SystemExit(f"pipeline_source: {src} does not exist")
        dst = pdir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied.append(rel)
    protocol.atomic_write_json(pdir / "manifest.json", {
        "copied_from": str(repo_root), "files": copied,
        "note": "read-only copies; editing them changes nothing",
    })
    return pdir


def teacher_convert(rows):
    """The teacher-data converter, loaded by path rather than by module name.

    train/frontier_model/ and train/grpo/ both hold a to_verl_dataset.py; an
    `import to_verl_dataset` would pick whichever sys.path entry came first, and
    the two disagree about where ground truth comes from (teacher-authored vs.
    verified). Naming the file removes the ambiguity.
    """
    import importlib.util
    src = FM_DIR / "to_verl_dataset.py"
    spec = importlib.util.spec_from_file_location("teacher_to_verl_dataset", src)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.convert(rows)


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
        rt = r.get("reference_test")
        if rt:
            lines.append(
                f"      reference test AFTER this train: n={rt.get('n_problems')} "
                f"pass@1={rt.get('pass_at_1')} pass@k={rt.get('pass_at_k')} "
                f"trunc={rt.get('truncation_rate')} no_answer={rt.get('no_answer_rate')}")
    return "\n".join(lines) if lines else "  (no prior steps -- this is step 0)"


def _reference_test_line(run_dir):
    """The base model's score on the reference test set, if it was measured."""
    sp = Path(run_dir) / "test_base" / "summary.json"
    if not sp.exists():
        return None
    try:
        s = json.loads(sp.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    k = s.get("samples_per_problem")
    return (f"  base model (before any training): n={s.get('n_problems')} "
            f"pass@1={s.get('pass@1')} pass@{k}={s.get(f'pass@{k}')}")


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
                   max_evals_per_step=10, reference_file=None, pipeline_dir=None):
    """Build the teacher's per-turn task message (a plain string).

    The standing explanation - what the loop is, what the run directory holds,
    what to write - lives in the system prompt, which does not change between
    turns. This message carries only what does change: which step and sub-action
    this is, how many evaluations remain, which checkpoint is current, and where
    the trajectory so far can be read.
    """
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
        *([_reference_test_line(run_dir)] if _reference_test_line(run_dir) else []),
        _trajectory_summary(run_dir, step),
        "",
        "Directories you can open:",
        f"  {run_dir}/",
        "      this run",
        *([f"  {pipeline_dir}/",
           "      the frozen config, and the code that runs your decisions"]
          if pipeline_dir else []),
        *([f"  {reference_file}",
           "      reference data, copied in for you to read"]
          if reference_file else []),
        *[f"  {p}/" for p in prior],
        *[f"  {p}/" for p in cur],
        "",
        f"Now write decision.json and data.jsonl into {cwd}, then stop.",
        f'  decision.json: {{"step": {step}, "decision": "evaluation"}} or '
        f'{{"step": {step}, "decision": "train"}}',
        f"  data.jsonl:    any number of problems for an evaluation; for a train, "
        f"EXACTLY train_batch_size = {tb} problems.",
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




# ------------------------------------------------------------------ evaluation



def run_test_evaluation(out_dir, model_path, config, repo_root, timeout_s):
    """Evaluate a checkpoint on the run's FIXED reference test set.

    Unlike run_evaluation, the problems are not the teacher's -- they are the
    same set on every step of every run, so the numbers form a curve that can be
    read across steps and across runs. Everything else (sampling, verifier,
    sample count) is the settings the config pins, so this differs from the
    teacher's own evaluations only in which problems it uses.

    Lands summary.json / records.jsonl / generations.jsonl in `out_dir`.
    """
    return _evaluate(out_dir, model_path, config, repo_root, timeout_s,
                     data=Path(repo_root) / config["test_data"],
                     label="reference_test", log_name="test.log")


def run_evaluation(step_dir, model_path, config, repo_root, timeout_s):
    """Evaluate the current checkpoint on the teacher's data via evaluate.py.

    `step_dir` is this eval sub-action's own dir (step_<N>/eval_<M>/), and is
    passed straight through as `--output-path`, so evaluate.py lands its
    summary.json / records.jsonl / generations.jsonl there. `--limit 0` is
    uncapped. Returns a dict of eval stats (weights unchanged).
    """
    return _evaluate(step_dir, model_path, config, repo_root, timeout_s,
                     data=Path(step_dir) / "data.jsonl",
                     label="teacher_eval", log_name="eval.log")


def _evaluate(out_dir, model_path, config, repo_root, timeout_s, *,
              data, label, log_name):
    """Shared body of the two evaluations: same settings, different problems.

    Pure command line: the data is written moments earlier or lives outside
    configs/, so there is no config to point at and nothing to register.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tmpl = dict(config.get("teacher_eval", {}))
    base = dict(config.get("eval_base", {}))
    argv = [
        "python", str(EVAL_DIR / "evaluate.py"),
        "--model", str(model_path),
        "--data", str(data),
        "--label", label,
        "--samples", str(tmpl.get("n", 4)),
        "--verifier", tmpl.get("verifier", "symbolic"),
        "--max-tokens", str(tmpl.get("max_tokens", 4096)),
        "--max-model-len", str(tmpl.get("max_model_len", 8192)),
        "--temperature", str(base.get("temperature", 0.6)),
        "--top-p", str(base.get("top_p", 0.9)),
        "--top-k", str(base.get("top_k", -1)),
        "--seed", str(base.get("seed", 1234)),
        "--tensor-parallel-size", str(base.get("tensor_parallel_size", 1)),
        "--gpu-memory-utilization", str(base.get("gpu_memory_utilization", 0.90)),
        "--limit", "0",
        "--output-path", str(out_dir),
    ]
    rc, timed_out, wall = _stream(argv, out_dir / log_name, cwd=str(repo_root),
                                  timeout_s=timeout_s)

    summ_src = out_dir / "summary.json"
    stats = {"status": "ok" if (rc == 0 and not timed_out) else "error",
             "action_wallclock_s": round(wall, 1), "returncode": rc,
             "timed_out": timed_out}
    if summ_src.exists():
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
    """Reconstruct the compact eval-stats dict from a saved summary.json.

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

def run_train(step_dir, model_path, config, repo_root, timeout_s):
    """One real GRPO update on the teacher's curriculum.

    Convert data.jsonl via train/frontier_model/to_verl_dataset.py (the teacher's
    answer becomes ground_truth, with no verification) -> run_teacher_step.sh for
    one optimizer step on the latest checkpoint. Saves the hf checkpoint, deletes
    the FSDP world-size shards, and returns the advanced checkpoint path.

    The group size, batch size and mini-batch come from the config and are passed
    as arguments; the curriculum's length does not influence them. The row count
    was validated as exactly train_batch_size before this was called.
    """
    step_dir = Path(step_dir)
    rows = protocol.read_jsonl(step_dir / "data.jsonl")

    # convert in-process (deterministic, no subprocess) so a converter error is a
    # clean ProtocolError rather than a shell exit code.
    verl_rows = teacher_convert(rows)
    train_file = step_dir / "train.verl.jsonl"
    protocol.atomic_write_jsonl(train_file, verl_rows)

    ckpt_dir = step_dir / "ckpt"
    argv = [
        "bash", str(FM_DIR / "run_teacher_step.sh"),
        str(config["grpo_config"]), str(train_file), str(model_path),
        str(ckpt_dir), str(config["n_gpus"]), str(config["group_size"]),
        str(config["train_batch_size"]), str(config["ppo_mini_batch_size"]),
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
            "rollouts": len(rows) * int(config["group_size"]),
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
