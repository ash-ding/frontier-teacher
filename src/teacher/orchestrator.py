"""The observational teacher-in-the-loop curriculum driver.

This is the external Python loop (NOT Claude Code). Each step it:
  1. skips the step if a valid result.json already exists (idempotent restart);
  2. snapshots the resolved config;
  3. builds the teacher's read-only context + full past trajectory;
  4. invokes `claude -p` once (Vertex auth inherited) to get decision.json + data.jsonl;
  5. validates strictly (malformed = halt, not retry-forever) and runs the
     same-(decision,data)-twice circuit breaker;
  6. re-materialises pristine read-only source (git checkout);
  7. executes the choice for real -- evaluate (weights unchanged) or one GRPO
     update (weights advance) -- on the LATEST checkpoint;
  8. persists full logs, appends metrics/events, atomic-writes result.json.

Entry point:  python -m src.teacher.orchestrator --config <cfg> [--run-name X]
              [--steps N] [--dry-run]

--dry-run is a fast, no-GPU self-test: it fabricates the teacher turn and stubs
the train/eval executors, exercising the protocol, converter, atomic handoffs,
idempotent restart, and instrumentation. The deliverable is the REAL run.
"""
import argparse
import datetime as _dt
import json
import sys
import time
from pathlib import Path

import yaml

if __package__ in (None, ""):  # allow `python src/teacher/orchestrator.py`
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from src.teacher import claude_client, protocol, tools
else:
    from . import claude_client, protocol, tools

ROOT = Path(__file__).resolve().parent.parent.parent
PROMPTS = Path(__file__).resolve().parent / "prompts"


def _now():
    return _dt.datetime.now().isoformat(timespec="seconds")


def _append_jsonl(path, obj):
    """Append one line. Not atomic-replace (append-only logs), but flushed+fsync'd."""
    import os
    with open(path, "a") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


class Orchestrator:
    def __init__(self, config_path, run_name, steps, dry_run):
        self.config_path = Path(config_path)
        self.config = yaml.safe_load(self.config_path.read_text())
        self.dry_run = dry_run
        self.steps = steps or int(self.config.get("steps", 4))
        rid = run_name or _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = (ROOT / self.config["out_root"] / f"run_{rid}").resolve()
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.events = self.run_dir / "events.jsonl"
        self.metrics = self.run_dir / "metrics.jsonl"
        self.run_state = self.run_dir / "run_state.json"
        self.system_prompt = (PROMPTS / "teacher_system.md").read_text()
        self.base_model = self.config.get("base_model", "unsloth/Llama-3.2-3B-Instruct")
        self.consecutive_failures = 0

    # ------------------------------------------------------------- bookkeeping
    def log(self, *a):
        print(*a, flush=True)

    def event(self, step, event, **detail):
        self.log(f"[{_now()}] step {step}: {event} {detail if detail else ''}")
        _append_jsonl(self.events, {"ts": _now(), "run": self.run_dir.name,
                                    "step": step, "event": event, **detail})

    def recover_state(self):
        """Artifact-derived restart: scan completed steps, rebuild latest ckpt.

        The checkpoint pointer is reconstructed from each step's result.json (not
        trusted solely from run_state.json), so a crashed/half-run restarts on the
        same weights it left off with.
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
            if r.get("latest_checkpoint_hf_path"):
                latest = r["latest_checkpoint_hf_path"]
            start = k + 1
        return latest, start

    # -------------------------------------------------------------- teacher turn
    def teacher_turn(self, step, step_dir, latest_ckpt):
        """Get decision.json + data.jsonl for this step (real claude -p or canned)."""
        context = tools.render_context(self.run_dir, step, self.config, latest_ckpt, ROOT)
        (step_dir / "task_context.txt").write_text(context)

        if self.dry_run:
            self._canned_teacher(step, step_dir)
            return {"wallclock_s": 0.0, "input_tokens": 0, "output_tokens": 0,
                    "cost_usd": 0.0, "num_turns": 0, "session_id": "dry-run"}

        res = claude_client.run_teacher_turn(
            prompt=context,
            system_prompt=self.system_prompt,
            cwd=str(step_dir),
            log_path=str(step_dir / "teacher.log"),
            repo_root=str(ROOT),
            model=self.config.get("teacher_model", ""),
            timeout_s=int(self.config.get("teacher_timeout_s", 900)),
            max_budget_usd=float(self.config.get("teacher_max_budget_usd", 0.0)),
        )
        if not res.ok:
            # The failure counter is owned by run() so every halt (teacher-turn
            # failure, malformed/missing file, schema violation) counts exactly once.
            raise protocol.ProtocolError(
                f"teacher turn failed (rc={res.returncode} timed_out={res.timed_out} "
                f"err={res.error!r})")
        return {"wallclock_s": round(res.wallclock_s, 1),
                "input_tokens": res.input_tokens, "output_tokens": res.output_tokens,
                "cost_usd": res.cost_usd, "num_turns": res.num_turns,
                "session_id": res.session_id}

    def _canned_teacher(self, step, step_dir):
        """Dry-run stand-in: a deterministic decision + tiny gradeable data set."""
        decision = "train" if step % 2 == 0 else "evaluation"
        probs = [
            {"id": f"dry-{step}-0", "problem": "What is 2 + 2?", "answer": "4"},
            {"id": f"dry-{step}-1", "problem": "Compute 3 * 7.", "answer": "21"},
            {"id": f"dry-{step}-2", "problem": "What is 10 - 6?", "answer": "4"},
        ]
        protocol.atomic_write_json(step_dir / "decision.json",
                                   {"step": step, "decision": decision})
        protocol.atomic_write_jsonl(step_dir / "data.jsonl", probs)
        (step_dir / "teacher.log").write_text(f"[dry-run] canned decision={decision}\n")

    # ------------------------------------------------------------- step execution
    def run_step(self, step, latest_ckpt, prev):
        step_dir = self.run_dir / f"step_{step}"
        step_dir.mkdir(parents=True, exist_ok=True)

        # (2) config snapshot BEFORE any work -- guards silent mid-run drift.
        snapshot = {
            "step": step, "run": self.run_dir.name, "ts": _now(),
            "latest_checkpoint_hf_path": latest_ckpt, "base_model": self.base_model,
            "teacher_model": self.config.get("teacher_model", ""),
            "G": 32, "TB": self.config.get("train_max_problems", 16),
            "dry_run": self.dry_run,
            "teacher_prompt_sha256": protocol.content_hash([self.system_prompt]),
        }
        protocol.atomic_write_json(step_dir / "config.json", snapshot)

        # (3,4) teacher turn -> decision.json + data.jsonl
        self.event(step, "teacher_turn_start")
        teacher = self.teacher_turn(step, step_dir, latest_ckpt)
        self.event(step, "teacher_turn_done", **teacher)

        # (5) strict validation -- a missing file or syntactically-broken JSON
        # halts cleanly (ProtocolError) exactly like a schema violation, rather
        # than escaping as an uncaught FileNotFoundError/JSONDecodeError traceback.
        decision = protocol.validate_decision(
            protocol.read_json_strict(step_dir / "decision.json"), step)["decision"]
        rows = protocol.validate_data_rows(
            protocol.read_jsonl_strict(step_dir / "data.jsonl"))
        data_hash = protocol.content_hash(rows)

        # circuit breaker: identical (decision, data) as the immediately prior step
        if self.config.get("halt_on_duplicate", True) and prev and protocol.is_duplicate_step(
                prev["decision"], prev["data_hash"], decision, data_hash):
            raise protocol.ProtocolError(
                f"step {step}: identical (decision={decision}, data) as step {step-1} "
                "-- halting a stuck teacher rather than looping")

        self.event(step, "decision", decision=decision, n_rows=len(rows))

        # (6) discard any teacher edit to read-only source before acting
        if not self.dry_run:
            tools.rematerialize_ro(ROOT, self.config["read_only_paths"], log=self.log)

        # (7) branch
        new_ckpt = latest_ckpt
        if decision == "evaluation":
            stats = self._do_eval(step, step_dir, latest_ckpt)
            eval_stats, train_stats = stats.get("eval"), None
        else:
            stats = self._do_train(step, step_dir, latest_ckpt, len(rows))
            eval_stats, train_stats = None, stats.get("train")
            if stats["status"] == "ok" and stats.get("new_checkpoint"):
                new_ckpt = stats["new_checkpoint"]

        # (9) result.json (atomic), metrics + run_state
        result = {
            "step": step, "decision": decision, "status": stats["status"],
            "teacher": teacher,
            "action_wallclock_s": stats.get("action_wallclock_s"),
            "eval": eval_stats, "train": train_stats,
            "latest_checkpoint_hf_path": new_ckpt,
        }
        if stats.get("error"):
            result["error"] = stats["error"]
        protocol.atomic_write_json(step_dir / "result.json", result)

        _append_jsonl(self.metrics, {
            "ts": _now(), "step": step, "decision": decision, "status": stats["status"],
            "teacher_wallclock_s": teacher["wallclock_s"],
            "teacher_input_tokens": teacher["input_tokens"],
            "teacher_output_tokens": teacher["output_tokens"],
            "teacher_cost_usd": teacher["cost_usd"],
            "action_wallclock_s": stats.get("action_wallclock_s"),
            "truncation_rate": (eval_stats or train_stats or {}).get("truncation_rate"),
            "no_answer_rate": (eval_stats or train_stats or {}).get("no_answer_rate"),
            "pass_at_1": (eval_stats or {}).get("pass_at_1"),
            "pass_at_k": (eval_stats or {}).get("pass_at_k"),
            "n_problems": (eval_stats or train_stats or {}).get("n_problems"),
            "latest_checkpoint_hf_path": new_ckpt,
        })
        protocol.atomic_write_json(self.run_state, {
            "last_completed_step": step, "latest_ckpt_hf_path": new_ckpt,
            "ts": _now(), "run": self.run_dir.name})
        self.event(step, "step_done", status=stats["status"], decision=decision,
                   latest_ckpt=new_ckpt)

        if stats["status"] != "ok":
            raise protocol.ProtocolError(
                f"step {step} {decision} failed: {stats.get('error')}")
        return new_ckpt, {"decision": decision, "data_hash": data_hash}

    # ---- executors (real vs stubbed) -----------------------------------------
    def _do_eval(self, step, step_dir, latest_ckpt):
        self.event(step, "eval_start", model=latest_ckpt)
        if self.dry_run:
            return self._stub_eval(step, step_dir, latest_ckpt)
        stats = tools.run_evaluation(step_dir, step, latest_ckpt, self.config, ROOT,
                                     int(self.config.get("eval_step_timeout_s", 3600)))
        self.event(step, "eval_done", **(stats.get("eval") or {"status": stats["status"]}))
        return stats

    def _do_train(self, step, step_dir, latest_ckpt, n_rows):
        self.event(step, "train_start", model=latest_ckpt, n_rows=n_rows)
        if self.dry_run:
            return self._stub_train(step, step_dir, latest_ckpt)
        stats = tools.run_train(
            step_dir, step, latest_ckpt, self.config, ROOT,
            int(self.config.get("train_step_timeout_s", 3600)),
            int(self.config.get("n_gpus", 8)),
            int(self.config.get("train_max_problems", 16)))
        self.event(step, "train_done", **(stats.get("train") or {"status": stats["status"]}))
        return stats

    def _stub_eval(self, step, step_dir, latest_ckpt):
        """No-GPU: exercise config.eval.yaml build + synthetic summary/records."""
        step_data = step_dir / "data.jsonl"
        tools.write_eval_config(step_dir, step, latest_ckpt, self.config, step_data, ROOT)
        rows = protocol.read_jsonl(step_data)
        recs = [{**r, "n": 4, "c": 4,
                 "samples": [{"correct": True, "extracted": r["answer"],
                              "truncated": False, "n_tokens": 10} for _ in range(4)]}
                for r in rows]
        protocol.atomic_write_jsonl(step_dir / "eval.records.jsonl", recs)
        summary = {"tag": f"teacher_step{step}__teacher_eval", "n_problems": len(rows),
                   "samples_per_problem": 4, "pass@1": 1.0, "pass@4": 1.0,
                   "truncation_rate": 0.0, "no_answer_rate": 0.0, "dry_run": True}
        protocol.atomic_write_json(step_dir / "eval.summary.json", summary)
        return {"status": "ok", "action_wallclock_s": 0.0,
                "eval": {"n_problems": len(rows), "pass_at_1": 1.0, "pass_at_k": 1.0,
                         "truncation_rate": 0.0, "no_answer_rate": 0.0}}

    def _stub_train(self, step, step_dir, latest_ckpt):
        """No-GPU: run the REAL converter, then fabricate a checkpoint dir."""
        rows = protocol.read_jsonl(step_dir / "data.jsonl")[
            : int(self.config.get("train_max_problems", 16))]
        sys.path.insert(0, str(ROOT / "src"))
        from to_verl_teacher_dataset import convert  # noqa: E402
        verl_rows = convert(rows)
        protocol.atomic_write_jsonl(step_dir / "train.verl.jsonl", verl_rows)
        # sanity: converted rows are verl-shaped with non-empty ground truth
        assert all(isinstance(r["prompt"], list) for r in verl_rows)
        assert all(str(r["reward_model"]["ground_truth"]).strip() for r in verl_rows)
        hf = step_dir / "ckpt" / "global_step_1" / "actor" / "huggingface"
        hf.mkdir(parents=True, exist_ok=True)
        (hf / "DRY_RUN_STUB").write_text("stub checkpoint (no weights)\n")
        (step_dir / "train.log").write_text("[dry-run] stubbed GRPO update\n")
        return {"status": "ok", "action_wallclock_s": 0.0,
                "train": {"n_problems": len(rows), "truncated_extra_rows": 0,
                          "truncation_rate": None, "no_answer_rate": None,
                          "metrics": {"dry_run": 1.0}, "deleted_shards": 0},
                "new_checkpoint": str(hf)}

    # ----------------------------------------------------------------------- run
    def run(self):
        latest_ckpt, start = self.recover_state()
        self.event(-1, "run_start", steps=self.steps, dry_run=self.dry_run,
                   start_step=start, latest_ckpt=latest_ckpt,
                   teacher_model=self.config.get("teacher_model", ""))
        if start:
            self.log(f"resuming: steps 0..{start-1} already complete; latest={latest_ckpt}")

        prev = None
        abort_at = int(self.config.get("consecutive_failure_abort", 2))
        for step in range(start, self.steps):
            rp = self.run_dir / f"step_{step}" / "result.json"
            if rp.exists():  # idempotent
                try:
                    r = protocol.validate_result(json.loads(rp.read_text()))
                    if r.get("latest_checkpoint_hf_path"):
                        latest_ckpt = r["latest_checkpoint_hf_path"]
                    self.event(step, "skip_completed", status=r.get("status"))
                    continue
                except (protocol.ProtocolError, json.JSONDecodeError):
                    pass  # malformed -> re-run
            try:
                latest_ckpt, prev = self.run_step(step, latest_ckpt, prev)
                self.consecutive_failures = 0  # a clean step resets the streak
            except protocol.ProtocolError as e:
                # Every strict-validation failure -- teacher-turn failure, missing
                # or syntactically-broken decision.json/data.jsonl, or a schema
                # violation -- lands here: log the halt event, count the failure,
                # and exit cleanly (rc=1). We do NOT best-effort parse or retry.
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
    ap.add_argument("--run-name", default="")
    ap.add_argument("--steps", type=int, default=0, help="0 -> config default")
    ap.add_argument("--dry-run", action="store_true", help="no-GPU protocol self-test")
    a = ap.parse_args()
    orch = Orchestrator(a.config, a.run_name, a.steps, a.dry_run)
    return orch.run()


if __name__ == "__main__":
    raise SystemExit(main())
