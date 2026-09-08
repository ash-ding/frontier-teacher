"""The goal-directed teacher loop: same GRPO recipe, three target problems.

    python train/goal_teacher/orchestrator.py --config configs/goal/<name>.yaml
        [--steps N] [--output-path DIR] [--resume DIR] [--dry-run]

Everything about the training step is the open-ended arm's, unchanged and
imported rather than copied: G = 32, 16 problems, 512 rollouts, one update per
step, `resume_mode=disable`, the same grader, the same protocol. What differs is
the situation the teacher is put in.

  * It is given three research problems and told its own answers do not count --
    only what the student produces when asked them directly. Two of the three
    are ones this teacher model solves on the public leaderboard; one it does
    not, and that asymmetry is the point.
  * It does not have the answers, and cannot get them. They are sealed under a
    key held only by this process, which pops it out of `os.environ` at startup
    precisely so the teacher subprocess cannot inherit it.
  * There are no reference tests. The open-ended arm showed the teacher three
    benchmark curves after every update; this one shows the three targets and
    nothing else, because the targets are what it is being asked to move.
  * The system prompt states the GRPO group mechanic -- G completions per
    problem, the group's scores determining the update -- and prescribes no
    strategy. That is the line this arm draws: mechanism yes, tactics no.

The three targets are fixed at run start and are recorded in the run's
`pipeline/`, like every other setting.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "frontier_model"))
from teacher import protocol  # noqa: E402
from teacher.orchestrator import ROOT, Orchestrator, _now  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from goal_teacher import client, context, target_eval  # noqa: E402
from goal_teacher.secrets_box import ENV_VAR, load_answers  # noqa: E402

PROMPTS = Path(__file__).resolve().parent / "prompts"


class GoalOrchestrator(Orchestrator):

    DEFAULTS = {**Orchestrator.DEFAULTS,
                "targets_file": None,
                "targets_sealed": None,
                "target_eval": {"samples": 8, "verifier": "symbolic",
                                "headline_metric": "pass@1"},
                # This arm has none, and the base default is already [] -- named
                # here so that reading this file tells you it is deliberate.
                "reference_tests": []}

    def __init__(self, *a, **kw):
        # Out of the environment before anything can spawn. The base client hands
        # the teacher `env=os.environ.copy()`; holding the key in memory instead
        # of in the environment removes the window entirely rather than relying
        # on every call site remembering to scrub it. client.run_teacher_turn
        # asserts the absence rather than establishing it.
        self._key = os.environ.pop(ENV_VAR, None)

        super().__init__(*a, **kw)

        if self.config.get("reference_tests"):
            raise SystemExit(
                "goal_teacher does not run reference tests; this arm shows the "
                "teacher the targets and nothing else. Remove reference_tests "
                "from the config or use the open-ended orchestrator.")

        tf = self.config.get("targets_file")
        if not tf:
            raise SystemExit("config must set targets_file")
        tf = Path(tf).expanduser()
        self.targets_file = tf if tf.is_absolute() else ROOT / tf
        if not self.targets_file.is_file():
            raise SystemExit(f"targets_file is not a file: {self.targets_file}")
        self.targets = [json.loads(l) for l in self.targets_file.open()]
        if not self.targets:
            raise SystemExit(f"{self.targets_file} is empty")

        sp = self.config.get("targets_sealed")
        if not sp:
            raise SystemExit("config must set targets_sealed")
        sp = Path(sp).expanduser()
        self.sealed_path = sp if sp.is_absolute() else ROOT / sp
        if not self.sealed_path.is_file():
            raise SystemExit(f"targets_sealed is not a file: {self.sealed_path}")

        # Fail now, not three hours in: a run that cannot grade its targets has
        # no read-out, and the only way to find out is to try.
        if not self.dry_run:
            if not self._key:
                raise SystemExit(
                    f"{ENV_VAR} is not set. Export it in the launching shell; it "
                    f"is removed from the environment before the teacher starts.")
            try:
                answers = load_answers(self.sealed_path, self._key)
            except ValueError as e:
                raise SystemExit(
                    f"cannot open {self.sealed_path}: {e}. {ENV_VAR} is set but "
                    f"does not match the key these answers were sealed under.")
            missing = [t["id"] for t in self.targets if t["id"] not in answers]
            if missing:
                raise SystemExit(f"sealed answers do not cover {missing}")

        self.system_prompt = self._build_system_prompt()

        # The base snapshots pipeline/ at the end of its __init__, before any of
        # the above exists, so config.resolved.json would record a run with no
        # targets and no goal. Rewrite it, and put the system prompt beside it:
        # the prompt carries the three problem statements and is the single
        # biggest thing that distinguishes this arm from the open-ended one. A
        # run whose prompt is not in its own record cannot be read later.
        protocol.atomic_write_json(
            self.pipeline_dir / "config.resolved.json", self.resolved_settings())
        protocol.atomic_write_text(
            self.pipeline_dir / "teacher_system.md", self.system_prompt)
        shutil.copy2(self.targets_file, self.pipeline_dir / self.targets_file.name)

    def resolved_settings(self):
        s = super().resolved_settings()
        s["goal"] = {
            "targets_file": str(getattr(self, "targets_file", "")),
            "targets_sealed": str(getattr(self, "sealed_path", "")),
            "target_ids": [t["id"] for t in getattr(self, "targets", [])],
            "target_eval": dict(self.config.get("target_eval") or {}),
            "answers_visible_to_teacher": False,
        }
        return s

    # ---------------------------------------------------------------- prompt
    def _build_system_prompt(self):
        """The standing prompt, with the three problems written into it.

        They belong here rather than behind a file path: they do not change
        between turns, they are the point of the run, and a goal mentioned as a
        directory listing reads like a footnote.
        """
        body = (PROMPTS / "teacher_system.md").read_text()
        block = []
        for t in self.targets:
            block += [f"### {t.get('role', t['id'])}  ({t['id']})", ""]
            if t.get("title"):
                block.append(f"*{t['title']}*")
                block.append("")
            block += [t["problem"].strip(), ""]
        return body.replace("{{TARGETS}}", "\n".join(block)) \
                   .replace("{{GROUP_SIZE}}", str(self.group_size)) \
                   .replace("{{N_TARGETS}}", str(len(self.targets)))

    # ------------------------------------------------------------ teacher turn
    def _real_teacher(self, ctx, staging):
        """Identical to the base, through the guarded client."""
        res = client.run_teacher_turn(
            prompt=ctx,
            system_prompt=self.system_prompt,
            cwd=str(staging),
            log_path=str(staging / "teacher.log"),
            repo_root=str(ROOT),
            model=self.config["teacher_model"],
            timeout_s=int(self.config["teacher_timeout_s"]),
        )
        if not res.ok:
            raise protocol.ProtocolError(
                f"teacher turn failed (rc={res.returncode} "
                f"timed_out={res.timed_out} err={res.error!r})")
        return {"wallclock_s": round(res.wallclock_s, 1),
                "input_tokens": res.input_tokens,
                "output_tokens": res.output_tokens,
                "cost_usd": res.cost_usd, "num_turns": res.num_turns,
                "session_id": res.session_id}

    def _teacher_subaction(self, step, action_index, step_dir, latest_ckpt, evals):
        """The base method with this arm's context renderer.

        Copied rather than extended because the base builds its message inline
        and there is no seam to pass a different renderer through. Everything
        else -- staging, strict validation, the atomic rename -- is the base's
        behaviour and must stay that way.
        """
        staging = step_dir / ".pending"
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
        staging.mkdir(parents=True)

        ctx = context.render(
            self.run_dir, step, self.config, latest_ckpt,
            cwd=staging, action_index=action_index, current_step_evals=evals,
            max_evals_per_step=self.max_evals_per_step,
            targets_file=self.reference_file or self.targets_file,
            pipeline_dir=self.pipeline_dir)
        (staging / "task_context.txt").write_text(ctx)

        self.event(step, "teacher_turn_start", action_index=action_index)
        if self.dry_run:
            teacher = self._canned_teacher(step, action_index, staging)
        else:
            teacher = self._real_teacher(ctx, staging)
        self.event(step, "teacher_turn_done", action_index=action_index, **teacher)

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
        os.replace(staging, target)
        return decision, rows, target, teacher

    # ------------------------------------------------------------ target eval
    def run_target_eval(self, step, out_dir, model_path):
        """Grade the checkpoint on the targets. Never fatal.

        The weights exist by the time this runs. A bucket hiccup, an OOM in the
        grader or a missing key must not end a run that has already produced
        what it was for -- the failure is recorded as an event and the step
        closes without a target row.
        """
        out_dir = Path(out_dir)
        try:
            if self.dry_run:
                per = [{"id": t["id"], "n": 8, "pass_at_1": 0.0}
                       for t in self.targets]
                out_dir.mkdir(parents=True, exist_ok=True)
                protocol.atomic_write_jsonl(out_dir / "per_target.jsonl", per)
                protocol.atomic_write_json(out_dir / "target_traces.json",
                                           {t["id"]: ["(dry run)"] for t in self.targets})
            else:
                per, stats = target_eval.run_target_eval(
                    self.run_dir, step, out_dir, model_path, self.config, ROOT,
                    self.targets, self.sealed_path, self._key,
                    int(self.config["eval_step_timeout_s"]))
                if per is None:
                    self.event(step, "target_eval_failed",
                               error=stats.get("error", stats.get("status")))
                    return None
                target_eval.assert_clean(
                    out_dir, load_answers(self.sealed_path, self._key))
            self.event(step, "target_eval_done",
                       **{r["id"].rsplit("-", 1)[-1]: r["pass_at_1"] for r in per})
            return per
        except Exception as e:  # noqa: BLE001 - see docstring; never fatal
            self.event(step, "target_eval_failed",
                       error=f"{type(e).__name__}: {e}")
            return None

    def run_reference_tests(self, step, out_dir, model_path):
        """There are none. The base calls this at step -1 and at every close;
        both are where the targets should be measured, so this is the seam."""
        where = Path(out_dir).parent / "targets"
        return self.run_target_eval(step, where, model_path)

    def _close_step(self, step, step_dir, evals, train_stats, train_teacher, new_ckpt):
        """The base method, with the target scores recorded under their own key."""
        per = self.run_reference_tests(step, step_dir / "test", new_ckpt)
        kept = self.prune_checkpoints(step)
        protocol.atomic_write_json(step_dir / "result.json", {
            "step": step, "status": "ok", "n_evals": len(evals), "evals": evals,
            "train": {**(train_stats or {}), "teacher": train_teacher},
            "target_eval": per,
            "latest_checkpoint_hf_path": new_ckpt,
            "checkpoints_kept": kept,
        })
        protocol.atomic_write_json(self.run_state, {
            "last_completed_step": step, "latest_ckpt_hf_path": new_ckpt,
            "ts": _now(), "run": self.run_dir.name})
        self.event(step, "step_done", n_evals=len(evals), latest_ckpt=new_ckpt)
        return new_ckpt


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--max-evals-per-step", type=int, default=None)
    ap.add_argument("--output-path", default=None)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--dry-always-eval", action="store_true")
    a = ap.parse_args()
    return GoalOrchestrator(a.config, a.dry_run, steps=a.steps,
                            max_evals_per_step=a.max_evals_per_step,
                            output_path=a.output_path,
                            dry_always_eval=a.dry_always_eval,
                            resume=a.resume).run()


if __name__ == "__main__":
    raise SystemExit(main())
