"""The observational teacher-in-the-loop curriculum driver.

This is the external Python loop (NOT Claude Code). **One step is ONE training
update.** Within a step the teacher may EVALUATE the current checkpoint any number
of times (each eval measures the SAME latest checkpoint and does not change
weights); the step only CLOSES and advances when the teacher chooses to TRAIN.
`--steps N` therefore means N GRPO updates.

Per step, the inner loop is:

    repeat:
        teacher turn -> decision
        if decision == 'evaluation':
            run eval on the latest checkpoint, persist as step_<N>/eval_<M>/,
            STAY in this step, loop again
        if decision == 'train':
            run ONE GRPO update, save checkpoint, advance latest ckpt,
            persist as step_<N>/train/, CLOSE the step, go to step+1

If the teacher reaches `max_evals_per_step` evaluations without ever choosing
train, the run HALTS cleanly (descriptive halt event, rc=1) -- we never fabricate
a train. All evaluations within a step run on the SAME checkpoint (the one the
previous step's train produced; step 0 = base model).

Everything the run produces lives under one directory:

    run_<timestamp>/
      pipeline/                   THE FROZEN SETUP, written once at run start
        config.resolved.json      every setting: student, training_step,
                                  evaluation, loop
        manifest.json             which source files were copied, and from where
        eval/..., train/...       copies of the code that runs the decisions, at
                                  their repo-relative paths
      reference.jsonl             --reference-data, if given
      events.jsonl metrics.jsonl run_state.json
      step_<N>/
        config.json               resolved settings + the checkpoint this step
                                  started from
        eval_<M>/                 one per evaluation sub-action (M = 0,1,...)
          decision.json data.jsonl teacher.log
          summary.json records.jsonl generations.jsonl eval.log
        train/                    the terminating train sub-action
          decision.json data.jsonl teacher.log
          train.verl.jsonl train.log ckpt/.../huggingface
        result.json               STEP SUMMARY (written when the step closes on
                                  train): {step, n_evals, evals:[...], train:{...},
                                  latest_checkpoint_hf_path}

The teacher's ONLY interface to the student is the data it writes. The student,
the group size, the batch size, the sampling parameters and the GPU count are
settled from the config when the run starts and are identical on every step -
that is what makes a teacher run comparable to the no-teacher GRPO baseline. A
train curriculum must therefore be exactly `train_batch_size` problems; a
different count is a protocol error, not something to pad or truncate around.

Entry point:  python train/frontier_model/teacher/orchestrator.py --config <cfg>
              [--steps N] [--max-evals-per-step K] [--output-path DIR]
              [--reference-data FILE] [--dry-run] [--dry-always-eval]

The run directory is always <output-path>/run_<timestamp>/.

--dry-run is a fast, no-GPU self-test: it fabricates the teacher turns and stubs
the train/eval executors, exercising the inner loop (multiple evals then a train),
the protocol, converter, atomic handoffs, idempotent restart, and instrumentation.
The deliverable is the REAL run.
"""
import argparse
import datetime as _dt
import json
import os
import shutil
import sys
from pathlib import Path

import yaml

if __package__ in (None, ""):  # allow `python train/frontier_model/teacher/orchestrator.py`
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from teacher import claude_client, protocol, tools
else:
    from . import claude_client, protocol, tools

ROOT = Path(__file__).resolve().parent.parent.parent.parent
PROMPTS = Path(__file__).resolve().parent / "prompts"


def _now():
    return _dt.datetime.now().isoformat(timespec="seconds")


def _append_jsonl(path, obj):
    """Append one line. Not atomic-replace (append-only logs), but flushed+fsync'd."""
    with open(path, "a") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


class Orchestrator:
    # Command line, then config, then these - the same order eval/evaluate.py
    # uses, so one rule covers the whole repository.
    DEFAULTS = {
        "steps": 4,
        "max_evals_per_step": 10,
        "output_path": "outputs/frontier-model",
        "teacher_model": "opus-4.8",
        "teacher_timeout_s": 1800,
        "base_model": "unsloth/Llama-3.2-3B-Instruct",
        "grpo_config": "llama32-3b",
        "n_gpus": 8,
        "group_size": 32,
        "train_batch_size": 16,
        "ppo_mini_batch_size": 8,
        "train_step_timeout_s": 3600,
        "eval_step_timeout_s": 3600,
        "halt_on_duplicate": True,
        "consecutive_failure_abort": 2,
        "pipeline_source": [],
        "reference_data": None,
        "eval_base": {},
        "teacher_eval": {},
    }

    def __init__(self, config_path, dry_run, steps=None, max_evals_per_step=None,
                 output_path=None, reference_data=None, dry_always_eval=False):
        self.config_path = Path(config_path)
        cfg = yaml.safe_load(self.config_path.read_text()) or {}
        cli = {k: v for k, v in {"steps": steps,
                                 "max_evals_per_step": max_evals_per_step,
                                 "output_path": output_path,
                                 "reference_data": reference_data}.items()
               if v is not None}
        self.config = {**self.DEFAULTS, **cfg, **cli}
        self.dry_run = dry_run
        self.dry_always_eval = dry_always_eval
        self.steps = int(self.config["steps"])
        self.max_evals_per_step = int(self.config["max_evals_per_step"])

        # Runs are named by when they happened. A caller-supplied name is one more
        # thing to keep unique, and a repeat silently writes into a finished run.
        rid = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        out = Path(self.config["output_path"]).expanduser()
        if not out.is_absolute():
            out = ROOT / out
        self.run_dir = (out / f"run_{rid}").resolve()
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # The teacher sees one reference file, copied in rather than mounted, so
        # what it read is part of the run's record. Absent means it sees none.
        self.reference_file = None
        ref = self.config.get("reference_data")
        if ref:
            src = Path(ref).expanduser()
            if not src.is_file():
                raise SystemExit(f"reference_data is not a file: {src}")
            dst = self.run_dir / f"reference{src.suffix or '.jsonl'}"
            shutil.copy2(src, dst)
            self.reference_file = dst
        self.events = self.run_dir / "events.jsonl"
        self.metrics = self.run_dir / "metrics.jsonl"
        self.run_state = self.run_dir / "run_state.json"
        self.system_prompt = (PROMPTS / "teacher_system.md").read_text()
        self.base_model = self.config["base_model"]
        self.consecutive_failures = 0

        # The training shape is settled here, once, and never varies by step or by
        # what the teacher writes. Checked now rather than at the first train, so a
        # misconfigured run fails in the first second instead of an hour in.
        self.group_size = int(self.config["group_size"])
        self.train_batch_size = int(self.config["train_batch_size"])
        self.mini_batch_size = int(self.config["ppo_mini_batch_size"])
        n_gpus = int(self.config["n_gpus"])
        if self.train_batch_size % self.mini_batch_size:
            raise SystemExit(f"train_batch_size {self.train_batch_size} is not "
                             f"divisible by ppo_mini_batch_size {self.mini_batch_size}")
        if (self.train_batch_size * self.group_size) % n_gpus:
            raise SystemExit(
                f"train_batch_size * group_size "
                f"({self.train_batch_size * self.group_size}) is not divisible by "
                f"n_gpus {n_gpus}; verl requires this")

        # What will run the teacher's decisions, copied in so the teacher reads the
        # code that will actually execute rather than a description of it.
        self.pipeline_dir = tools.snapshot_pipeline(
            self.run_dir, ROOT, self.config, self.resolved_settings())

    def resolved_settings(self):
        """Every setting this run is pinned to, flat and JSON-safe.

        Written to pipeline/config.resolved.json and quoted back in each step's
        config.json. The teacher reads it: it is the difference between knowing
        the pipeline and guessing at it.
        """
        c = self.config
        return {
            "student": {
                "base_model": c["base_model"],
                "grpo_preset": c["grpo_config"],
                "n_gpus": int(c["n_gpus"]),
            },
            "training_step": {
                "group_size": self.group_size,
                "train_batch_size": self.train_batch_size,
                "ppo_mini_batch_size": self.mini_batch_size,
                "rollouts_per_step": self.train_batch_size * self.group_size,
                "gradient_updates_per_step": self.train_batch_size // self.mini_batch_size,
                "timeout_s": int(c["train_step_timeout_s"]),
            },
            "evaluation": {**dict(c["eval_base"]), **dict(c["teacher_eval"]),
                           "timeout_s": int(c["eval_step_timeout_s"])},
            "loop": {
                "steps": self.steps,
                "max_evals_per_step": self.max_evals_per_step,
                "halt_on_duplicate": bool(c["halt_on_duplicate"]),
                "teacher_model": c["teacher_model"],
                "teacher_timeout_s": int(c["teacher_timeout_s"]),
            },
        }

    # ------------------------------------------------------------- bookkeeping
    def log(self, *a):
        print(*a, flush=True)

    def event(self, step, event, action_index=None, **detail):
        tag = f"step {step}" + (f".{action_index}" if action_index is not None else "")
        self.log(f"[{_now()}] {tag}: {event} {detail if detail else ''}")
        rec = {"ts": _now(), "run": self.run_dir.name, "step": step, "event": event}
        if action_index is not None:
            rec["action_index"] = action_index
        rec.update(detail)
        _append_jsonl(self.events, rec)

    def recover_state(self):
        """Artifact-derived restart: scan CLOSED steps, rebuild latest ckpt.

        A step is a completed TRAINING step only when its result.json validates AND
        records a terminating train + advanced checkpoint. The checkpoint pointer is
        reconstructed from each closed step's result.json (not trusted solely from
        run_state.json), so a crashed/half-run restarts on the same weights it left
        off with. last_completed_step is the last completed TRAINING step.
        """
        latest = self.base_model
        start = 0
        for k in range(self.steps + 50):
            rp = self.run_dir / f"step_{k}" / "result.json"
            if not rp.exists():
                break
            try:
                r = protocol.validate_result(json.loads(rp.read_text()))
            except (protocol.ProtocolError, json.JSONDecodeError):
                break
            if not (r.get("train") and r.get("latest_checkpoint_hf_path")):
                break  # step opened but never closed on a train -> resume it
            latest = r["latest_checkpoint_hf_path"]
            start = k + 1
        return latest, start

    # -------------------------------------------------------------- teacher turn
    def _config_snapshot(self, step, step_dir, latest_ckpt):
        """Resolved config snapshot, written once at step start (guards drift)."""
        snapshot = {
            "step": step, "run": self.run_dir.name, "ts": _now(),
            "latest_checkpoint_hf_path": latest_ckpt,
            "dry_run": self.dry_run,
            "teacher_prompt_sha256": protocol.content_hash([self.system_prompt]),
            **self.resolved_settings(),
        }
        protocol.atomic_write_json(step_dir / "config.json", snapshot)

    def _teacher_subaction(self, step, action_index, step_dir, latest_ckpt, evals):
        """One teacher turn -> (decision, rows, target_dir, teacher_metrics).

        The teacher writes decision.json + data.jsonl into a per-step staging dir
        (we do not yet know eval vs train). After validating the decision we move
        the staging dir atomically to its final home -- step_<N>/eval_<M>/ for an
        evaluation (M = action_index), or step_<N>/train/ for the terminating train.
        """
        staging = step_dir / ".pending"
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
        staging.mkdir(parents=True)

        context = tools.render_context(
            self.run_dir, step, self.config, latest_ckpt, ROOT,
            cwd=staging, action_index=action_index, current_step_evals=evals,
            max_evals_per_step=self.max_evals_per_step,
            reference_file=self.reference_file,
            pipeline_dir=self.pipeline_dir)
        (staging / "task_context.txt").write_text(context)

        self.event(step, "teacher_turn_start", action_index=action_index)
        if self.dry_run:
            teacher = self._canned_teacher(step, action_index, staging)
        else:
            teacher = self._real_teacher(context, staging)
        self.event(step, "teacher_turn_done", action_index=action_index, **teacher)

        # strict validation -- missing/broken files halt cleanly (ProtocolError)
        decision = protocol.validate_decision(
            protocol.read_json_strict(staging / "decision.json"), step)["decision"]
        rows = protocol.validate_data_rows(
            protocol.read_jsonl_strict(staging / "data.jsonl"))
        if decision == "train":
            protocol.validate_train_batch(rows, self.train_batch_size)

        target = (step_dir / f"eval_{action_index}") if decision == "evaluation" \
            else (step_dir / "train")
        if target.exists():
            shutil.rmtree(target, ignore_errors=True)
        os.replace(staging, target)  # atomic rename within the step dir
        return decision, rows, target, teacher

    def _real_teacher(self, context, staging):
        res = claude_client.run_teacher_turn(
            prompt=context,
            system_prompt=self.system_prompt,
            cwd=str(staging),
            log_path=str(staging / "teacher.log"),
            repo_root=str(ROOT),
            model=self.config["teacher_model"],
            timeout_s=int(self.config["teacher_timeout_s"]),
        )
        if not res.ok:
            raise protocol.ProtocolError(
                f"teacher turn failed (rc={res.returncode} timed_out={res.timed_out} "
                f"err={res.error!r})")
        return {"wallclock_s": round(res.wallclock_s, 1),
                "input_tokens": res.input_tokens, "output_tokens": res.output_tokens,
                "cost_usd": res.cost_usd, "num_turns": res.num_turns,
                "session_id": res.session_id}

    # canned sequences per step exercise the inner loop across >1 training step:
    #   step 0: eval, eval, train    step 1: eval, train    step >=2: eval, train
    _CANNED_SEQUENCE = {0: ("evaluation", "evaluation", "train"),
                        1: ("evaluation", "train")}

    def _canned_decision(self, step, action_index):
        if self.dry_always_eval:
            return "evaluation"  # cap-test harness: never trains -> hits the cap
        seq = self._CANNED_SEQUENCE.get(step, ("evaluation", "train"))
        return seq[action_index] if action_index < len(seq) else "train"

    def _canned_teacher(self, step, action_index, staging):
        """Dry-run stand-in: a deterministic decision + tiny gradeable data set.

        The data's ids embed step AND action_index, so two consecutive evaluations
        in a step carry DIFFERENT data and never trip the duplicate circuit breaker.
        A train curriculum is exactly train_batch_size rows, because that is what
        the real protocol requires -- a dry run that skipped the batch-size rule
        would not be testing the protocol.
        """
        decision = self._canned_decision(step, action_index)
        n = self.train_batch_size if decision == "train" else 3
        probs = [{"id": f"dry-{step}-{action_index}-{i}",
                  "problem": f"What is {i} + {i}?", "answer": str(2 * i)}
                 for i in range(n)]
        protocol.atomic_write_json(staging / "decision.json",
                                   {"step": step, "decision": decision})
        protocol.atomic_write_jsonl(staging / "data.jsonl", probs)
        (staging / "teacher.log").write_text(
            f"[dry-run] canned step={step} action={action_index} decision={decision}\n")
        return {"wallclock_s": 0.0, "input_tokens": 0, "output_tokens": 0,
                "cost_usd": 0.0, "num_turns": 0, "session_id": "dry-run"}

    # ------------------------------------------------------------- step execution
    def run_step(self, step, latest_ckpt, prev):
        """Run one training step: eval* then a terminating train. Returns
        (new_ckpt, prev). Idempotent: completed eval_<M>/ sub-actions and an
        already-finished train are recovered from artifacts rather than re-run."""
        step_dir = self.run_dir / f"step_{step}"
        step_dir.mkdir(parents=True, exist_ok=True)
        self._config_snapshot(step, step_dir, latest_ckpt)

        # ---- idempotent restart: recover completed eval sub-actions in order ----
        evals = []
        action_index = 0
        while True:
            ed = step_dir / f"eval_{action_index}"
            if (ed / "summary.json").exists():
                stats = tools.eval_stats_from_summary(ed / "summary.json")
                prev = self._prev_from_dir(ed, "evaluation")
                evals.append({"action_index": action_index, **stats,
                              "teacher": self._teacher_from_log(ed), "resumed": True})
                self.event(step, "eval_skip_completed", action_index=action_index,
                           **stats)
                action_index += 1
            else:
                if ed.exists():  # half-run eval -> discard and redo at this index
                    shutil.rmtree(ed, ignore_errors=True)
                break

        # ---- if the terminating train already finished, just finalize ----------
        train_dir = step_dir / "train"
        train_hf = train_dir / "ckpt" / "global_step_1" / "actor" / "huggingface"
        if train_hf.exists():
            train_stats, train_teacher = self._train_from_artifacts(train_dir)
            new_ckpt = str(train_hf)
            self.event(step, "train_skip_completed", action_index=action_index)
            return self._close_step(step, step_dir, evals, train_stats,
                                    train_teacher, new_ckpt), prev
        if train_dir.exists():  # half-run train -> discard and redo
            shutil.rmtree(train_dir, ignore_errors=True)

        # ---- inner loop: evaluate* then train ----------------------------------
        while True:
            decision, rows, sub_dir, teacher = self._teacher_subaction(
                step, action_index, step_dir, latest_ckpt, evals)
            data_hash = protocol.content_hash(rows)

            # circuit breaker: identical (decision, data) as the immediately prior
            # sub-action (across step boundaries too). Consecutive DIFFERENT evals
            # are allowed; the max_evals_per_step cap is the main anti-loop guard.
            if self.config["halt_on_duplicate"] and prev and \
                    protocol.is_duplicate_step(prev["decision"], prev["data_hash"],
                                               decision, data_hash):
                raise protocol.ProtocolError(
                    f"step {step}.{action_index}: identical (decision={decision}, data) "
                    "as the immediately preceding sub-action -- halting a stuck teacher")
            prev = {"decision": decision, "data_hash": data_hash}
            self.event(step, "decision", action_index=action_index,
                       decision=decision, n_rows=len(rows))

            if not self.dry_run:  # discard any teacher edit to read-only source
                tools.rematerialize_ro(ROOT, self.config["pipeline_source"], log=self.log)

            if decision == "evaluation":
                if len(evals) >= self.max_evals_per_step:
                    # the teacher chose to evaluate again after already using its
                    # full eval budget for this step -> halt cleanly, never fabricate
                    # a train.
                    self.event(step, "eval_cap_reached", action_index=action_index,
                               n_evals=len(evals),
                               max_evals_per_step=self.max_evals_per_step)
                    raise protocol.ProtocolError(
                        f"step {step}: teacher chose evaluation after already "
                        f"evaluating {len(evals)} time(s) "
                        f"(max_evals_per_step={self.max_evals_per_step}); halting "
                        "-- not fabricating a train")
                stats = self._run_eval(step, action_index, sub_dir, latest_ckpt)
                self._metrics_row(step, action_index, "eval", teacher, stats,
                                  latest_ckpt)
                if stats["status"] != "ok":
                    raise protocol.ProtocolError(
                        f"step {step}.{action_index} evaluation failed: "
                        f"{stats.get('error')}")
                evals.append({"action_index": action_index,
                              **(stats.get("eval") or {}), "teacher": teacher})
                action_index += 1
                continue

            # decision == "train": run one GRPO update, close the step
            stats = self._run_train(step, action_index, sub_dir, latest_ckpt, len(rows))
            self._metrics_row(step, action_index, "train", teacher, stats,
                              stats.get("new_checkpoint") or latest_ckpt)
            if stats["status"] != "ok":
                raise protocol.ProtocolError(
                    f"step {step}.{action_index} train failed: {stats.get('error')}")
            new_ckpt = stats["new_checkpoint"]
            return self._close_step(step, step_dir, evals, stats.get("train"),
                                    teacher, new_ckpt), prev

    def _close_step(self, step, step_dir, evals, train_stats, train_teacher, new_ckpt):
        """Write the STEP SUMMARY result.json + run_state, emit step_done."""
        result = {
            "step": step, "status": "ok", "n_evals": len(evals),
            "evals": evals,
            "train": {**(train_stats or {}), "teacher": train_teacher},
            "latest_checkpoint_hf_path": new_ckpt,
        }
        protocol.atomic_write_json(step_dir / "result.json", result)
        protocol.atomic_write_json(self.run_state, {
            "last_completed_step": step, "latest_ckpt_hf_path": new_ckpt,
            "ts": _now(), "run": self.run_dir.name})
        self.event(step, "step_done", n_evals=len(evals), latest_ckpt=new_ckpt)
        return new_ckpt

    def _metrics_row(self, step, action_index, action_type, teacher, stats, latest_ckpt):
        sub = stats.get("eval") or stats.get("train") or {}
        _append_jsonl(self.metrics, {
            "ts": _now(), "step": step, "action_index": action_index,
            "type": action_type, "status": stats["status"],
            "teacher_wallclock_s": teacher["wallclock_s"],
            "teacher_input_tokens": teacher["input_tokens"],
            "teacher_output_tokens": teacher["output_tokens"],
            "teacher_cost_usd": teacher["cost_usd"],
            "action_wallclock_s": stats.get("action_wallclock_s"),
            "truncation_rate": sub.get("truncation_rate"),
            "no_answer_rate": sub.get("no_answer_rate"),
            "pass_at_1": (stats.get("eval") or {}).get("pass_at_1"),
            "pass_at_k": (stats.get("eval") or {}).get("pass_at_k"),
            "n_problems": sub.get("n_problems"),
            "latest_checkpoint_hf_path": latest_ckpt,
        })

    # ---- restart helpers -----------------------------------------------------
    def _prev_from_dir(self, sub_dir, decision):
        """Rebuild the circuit-breaker key (decision, data_hash) from a completed
        sub-action's data.jsonl, so the guard carries across a crash boundary."""
        try:
            rows = protocol.validate_data_rows(
                protocol.read_jsonl_strict(sub_dir / "data.jsonl"))
            return {"decision": decision, "data_hash": protocol.content_hash(rows)}
        except protocol.ProtocolError:
            return {"decision": decision, "data_hash": None}

    def _teacher_from_log(self, sub_dir):
        """Best-effort teacher metrics for a resumed sub-action (unknown -> zeros)."""
        return {"wallclock_s": None, "input_tokens": None, "output_tokens": None,
                "cost_usd": None, "num_turns": None, "session_id": "resumed"}

    def _train_from_artifacts(self, train_dir):
        """Reconstruct train stats for an already-finished train on restart."""
        try:
            rows = protocol.read_jsonl(train_dir / "data.jsonl")
        except (OSError, json.JSONDecodeError):
            rows = []
        metrics = tools._parse_verl_metrics(train_dir / "train.log")
        train_stats = {
            "n_problems": len(rows),
            "truncation_rate": metrics.get("response_length/clip_ratio"),
            "no_answer_rate": None,
            "metrics": metrics or None,
            "deleted_shards": None,
            "resumed": True,
        }
        return train_stats, self._teacher_from_log(train_dir)

    # ---- executors (real vs stubbed) -----------------------------------------
    def _run_eval(self, step, action_index, sub_dir, latest_ckpt):
        self.event(step, "eval_start", action_index=action_index, model=latest_ckpt)
        if self.dry_run:
            stats = self._stub_eval(step, sub_dir, latest_ckpt)
        else:
            stats = tools.run_evaluation(
                sub_dir, latest_ckpt, self.config, ROOT,
                int(self.config["eval_step_timeout_s"]))
        self.event(step, "eval_done", action_index=action_index,
                   **(stats.get("eval") or {"status": stats["status"]}))
        return stats

    def _run_train(self, step, action_index, sub_dir, latest_ckpt, n_rows):
        self.event(step, "train_start", action_index=action_index,
                   model=latest_ckpt, n_rows=n_rows)
        if self.dry_run:
            stats = self._stub_train(step, sub_dir, latest_ckpt)
        else:
            stats = tools.run_train(
                sub_dir, latest_ckpt, self.config, ROOT,
                int(self.config["train_step_timeout_s"]))
        self.event(step, "train_done", action_index=action_index,
                   **(stats.get("train") or {"status": stats["status"]}))
        return stats

    def _stub_eval(self, step, sub_dir, latest_ckpt):
        """No-GPU: synthesise a summary/records pair with the real schema."""
        step_data = sub_dir / "data.jsonl"
        rows = protocol.read_jsonl(step_data)
        recs = [{**r, "n": 4, "c": 4,
                 "samples": [{"correct": True, "extracted": r["answer"],
                              "truncated": False, "n_tokens": 10} for _ in range(4)]}
                for r in rows]
        protocol.atomic_write_jsonl(sub_dir / "records.jsonl", recs)
        summary = {"tag": f"teacher_step{step}__teacher_eval", "n_problems": len(rows),
                   "samples_per_problem": 4, "pass@1": 1.0, "pass@4": 1.0,
                   "truncation_rate": 0.0, "no_answer_rate": 0.0, "dry_run": True}
        protocol.atomic_write_json(sub_dir / "summary.json", summary)
        return {"status": "ok", "action_wallclock_s": 0.0,
                "eval": {"n_problems": len(rows), "pass_at_1": 1.0, "pass_at_k": 1.0,
                         "truncation_rate": 0.0, "no_answer_rate": 0.0}}

    def _stub_train(self, step, sub_dir, latest_ckpt):
        """No-GPU: run the REAL converter, then fabricate a checkpoint dir."""
        rows = protocol.read_jsonl(sub_dir / "data.jsonl")
        verl_rows = tools.teacher_convert(rows)
        protocol.atomic_write_jsonl(sub_dir / "train.verl.jsonl", verl_rows)
        # sanity: converted rows are verl-shaped with non-empty ground truth
        assert all(isinstance(r["prompt"], list) for r in verl_rows)
        assert all(str(r["reward_model"]["ground_truth"]).strip() for r in verl_rows)
        hf = sub_dir / "ckpt" / "global_step_1" / "actor" / "huggingface"
        hf.mkdir(parents=True, exist_ok=True)
        (hf / "DRY_RUN_STUB").write_text("stub checkpoint (no weights)\n")
        (sub_dir / "train.log").write_text("[dry-run] stubbed GRPO update\n")
        return {"status": "ok", "action_wallclock_s": 0.0,
                "train": {"n_problems": len(rows),
                          "rollouts": len(rows) * self.group_size,
                          "truncation_rate": None, "no_answer_rate": None,
                          "metrics": {"dry_run": 1.0}, "deleted_shards": 0},
                "new_checkpoint": str(hf)}

    # ----------------------------------------------------------------------- run
    def run(self):
        latest_ckpt, start = self.recover_state()
        self.event(-1, "run_start", steps=self.steps, dry_run=self.dry_run,
                   start_step=start, latest_ckpt=latest_ckpt,
                   max_evals_per_step=self.max_evals_per_step,
                   teacher_model=self.config["teacher_model"])
        if start:
            self.log(f"resuming: steps 0..{start-1} already complete; latest={latest_ckpt}")

        prev = None
        abort_at = int(self.config["consecutive_failure_abort"])
        for step in range(start, self.steps):
            rp = self.run_dir / f"step_{step}" / "result.json"
            if rp.exists():  # idempotent: a step is closed only when its train ran
                try:
                    r = protocol.validate_result(json.loads(rp.read_text()))
                    if r.get("train") and r.get("latest_checkpoint_hf_path"):
                        latest_ckpt = r["latest_checkpoint_hf_path"]
                        self.event(step, "skip_completed", status=r.get("status"),
                                   n_evals=r.get("n_evals"))
                        continue
                except (protocol.ProtocolError, json.JSONDecodeError):
                    pass  # malformed / not closed -> resume the step
            try:
                latest_ckpt, prev = self.run_step(step, latest_ckpt, prev)
                self.consecutive_failures = 0  # a clean step resets the streak
            except protocol.ProtocolError as e:
                # Every strict-validation failure -- teacher-turn failure, a missing
                # or broken decision.json/data.jsonl, a schema violation, an eval-cap
                # breach, or the duplicate breaker -- lands here: log the halt event,
                # count the failure, exit cleanly (rc=1). No best-effort parse/retry.
                self.consecutive_failures += 1
                self.event(step, "halt", error=str(e),
                           consecutive_failures=self.consecutive_failures)
                self.log(f"\nHALT at step {step}: {e}")
                if self.consecutive_failures >= abort_at:
                    self.log(f"consecutive teacher failures reached {abort_at}; aborting run")
                return 1
        self.event(-1, "run_done", latest_ckpt=latest_ckpt)
        self.log(f"\nrun complete: {self.run_dir}  latest_ckpt={latest_ckpt}")
        return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--steps", type=int, default=None,
                    help="N training (GRPO) updates; overrides the config")
    ap.add_argument("--max-evals-per-step", type=int, default=None,
                    help="cap on evaluations before a step must train; overrides the config")
    ap.add_argument("--output-path", default=None,
                    help="where run_<timestamp>/ is created; overrides the config")
    ap.add_argument("--reference-data", default=None,
                    help="a single data file to copy into the run directory for the "
                         "teacher to read; overrides the config. Omit for none.")
    ap.add_argument("--dry-run", action="store_true", help="no-GPU protocol self-test")
    ap.add_argument("--dry-always-eval", action="store_true",
                    help="dry-run cap harness: teacher only ever evaluates")
    a = ap.parse_args()
    orch = Orchestrator(a.config, a.dry_run, steps=a.steps,
                        max_evals_per_step=a.max_evals_per_step,
                        output_path=a.output_path,
                        reference_data=a.reference_data,
                        dry_always_eval=a.dry_always_eval)
    return orch.run()


if __name__ == "__main__":
    raise SystemExit(main())
