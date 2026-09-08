"""Grade the student on the sealed targets, and show the teacher only the score.

This is the piece the open-ended runs had no equivalent of. There, every
evaluation graded problems the teacher wrote, against answers the teacher
supplied -- so the teacher always knew the key. Here the whole point is that it
does not, for at least one of the three, and something else has to do the
grading.

The shape is curriculum-rl's: a grader that holds the key runs the evaluation,
and the teacher receives an aggregate plus the student's raw attempts, never a
key and never a per-sample verdict.

Three things leak if this is written the obvious way, and all three are why the
evaluation runs in a scratch directory and only a filtered subset is copied back:

  1. evaluate.py builds each record as `{**row, ...}`, so records.jsonl carries
     the gold `answer` for every problem.
  2. summary.json records `data_files`, the path to the plaintext.
  3. the unsealed file itself would sit inside a tree the teacher can read --
     it has --add-dir on the repo, and Bash besides.

The filter is a whitelist. A field added to evaluate.py's summary later will be
dropped here rather than inherited silently; that failure direction is the safe
one.
"""
from __future__ import annotations

import collections
import json
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "frontier_model"))
from teacher import protocol, tools  # noqa: E402

from .secrets_box import load_answers  # noqa: E402

# Whitelisted summary keys. pass@k is matched by prefix, since which k exist
# depends on the sample count.
SUMMARY_KEEP = {"n_problems", "samples_per_problem", "truncation_rate",
                "no_answer_rate", "mean_gen_tokens", "headline_metric"}


def _measure(path):
    files = [f for f in Path(path).rglob("*") if f.is_file()]
    return len(files), sum(f.stat().st_size for f in files)


def run_target_eval(run_dir, step, out_dir, model_path, config, repo_root,
                    targets, sealed_path, key, timeout_s):
    """Evaluate `model_path` on the targets; write a sanitised result to out_dir.

    Returns the per-target pass rates, or None if the evaluation did not produce
    a summary. Raising is the caller's choice, not ours -- the weights already
    exist and a failed grading must not end a run.
    """
    answers = load_answers(sealed_path, key)
    missing = [t["id"] for t in targets if t["id"] not in answers]
    if missing:
        raise KeyError(f"sealed answers do not cover {missing}")

    scratch = Path(tempfile.mkdtemp(prefix=f"goal_targets_step{step}_"))
    try:
        # The problems, joined to their answers, outside the run directory.
        # evaluate.py applies the project's own MATH prompt to `problem`; the
        # targets are therefore asked the way every other evaluation in this
        # repository asks, not the way MathArena asked them.
        plain = scratch / "targets.jsonl"
        protocol.atomic_write_jsonl(plain, [
            {"id": t["id"], "problem": t["problem"], "answer": answers[t["id"]]}
            for t in targets])
        plain.chmod(0o600)

        raw = scratch / "eval"
        spec = {"name": "targets",
                # absolute: tools joins this onto repo_root, and pathlib returns
                # the absolute operand unchanged
                "data": str(plain),
                **{k: v for k, v in (config.get("target_eval") or {}).items()
                   if k in ("samples", "verifier", "headline_metric")}}
        # Never shard three problems. `test_shards` is 8 for thinking mode, which
        # splits the reference sets across the node; round-robin over 3 problems
        # would hand five of those shards nothing to do.
        cfg = {**config, "test_shards": 1}
        stats = tools.run_test_evaluation(raw, model_path, cfg, repo_root,
                                          timeout_s, spec)
        if stats.get("status") != "ok" or not (raw / "summary.json").exists():
            return None, stats
        return _sanitise(raw, Path(out_dir)), stats
    finally:
        # The plaintext must not outlive the evaluation, including on the failure
        # path -- an exception here is exactly when it would otherwise be left.
        shutil.rmtree(scratch, ignore_errors=True)


def _sanitise(raw, out_dir):
    """Copy the evaluation out, minus anything that identifies an answer."""
    out_dir.mkdir(parents=True, exist_ok=True)

    s = json.loads((raw / "summary.json").read_text())
    protocol.atomic_write_json(out_dir / "summary.json", {
        k: v for k, v in s.items()
        if k in SUMMARY_KEEP or k.startswith("pass@")})

    rows = [json.loads(l) for l in (raw / "records.jsonl").open()]
    per = [{"id": r["id"], "n": r["n"], "pass_at_1": r["c"] / r["n"]}
           for r in rows]
    protocol.atomic_write_jsonl(out_dir / "per_target.jsonl", per)

    # The student's own attempts, with no verdict attached. Reading how an
    # attempt fails is the signal here -- curriculum-rl's teacher used exactly
    # this to work out that its curriculum had been handing the model the key
    # facts rather than teaching it to find them. Withholding `correct` is
    # deliberate: an attempt labelled correct is an answer.
    traces = collections.defaultdict(list)
    gen = raw / "generations.jsonl"
    if gen.exists():
        for line in gen.open():
            g = json.loads(line)
            traces[g["id"]].append(g["text"])
    protocol.atomic_write_json(out_dir / "target_traces.json", dict(traces))

    return per


# target_traces.json is exempt, and the exemption is the point. It holds what the
# STUDENT wrote, and a student that solves a target writes the answer into its own
# trace. That is the result we are hoping for, not a sanitiser failure -- checking
# it here would abort the run at the one moment it got interesting.
CHECKED = ("summary.json", "per_target.jsonl")


def assert_clean(out_dir, answers):
    """Fail loudly if an answer survived into the files WE construct.

    Cheap, and it tests the property the arm rests on directly. Runs every step,
    not only in the dry run: a sanitiser that breaks at step 12 should stop the
    run at step 12, because every step after it is unusable anyway.
    """
    out_dir = Path(out_dir)
    hay = "\n".join((out_dir / n).read_text(errors="ignore")
                    for n in CHECKED if (out_dir / n).exists())
    hits = [pid for pid, a in answers.items() if a and str(a) in hay]
    if hits:
        raise RuntimeError(
            f"target answers leaked into {out_dir} for {hits}; the sanitiser did "
            f"not hold and this run's Q3 claim would be void")
