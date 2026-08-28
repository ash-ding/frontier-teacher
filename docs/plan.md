# Plan

Open work, ordered by what unblocks what. `docs/experiment.md` records what has
already been run; this file records what has not.

Everything below runs on three 8×H100 nodes under one shared environment
(vLLM 0.27.1 + verl 0.9.0 + flash-attn 2.8.3), verified identical across
`lumen-1`, `lumen-2`, `lumen-3`.

---

## 1. GRPO training — the actual experiment

Everything so far is preparation. The question this repository exists to answer
is whether a frontier model can teach a smaller one, and nothing has been trained
yet.

**Student:** Llama-3.2-3B-Instruct. It is the only candidate: 38.8% on MATH-500
leaves room to move, where Qwen3-4B at 95.5% does not.

**Training pool:** `data/further_improve/llama32-3b/`. The 5–15% band (400
problems, measured mean pass@1 8.7%) is the closest analogue to the reference
work's hard subset. The wider `0 < p ≤ 25%` slice of the clean pool holds 1,196
problems if 400 proves too few to sustain training.

- [ ] Write the verl GRPO config: rollout via vLLM, FSDP training, reward from
      `src/grading.py` so training and evaluation score identically.
- [ ] Baseline GRPO run on the hard band. This is the control the teacher has to
      beat; without it a teacher-side gain is unattributable.
- [ ] Evaluate the trained checkpoint on MATH-500 and AIME with the existing
      harness, unchanged, so the numbers join the baseline table directly.

**Design note.** The reward function must be the same code path as evaluation.
If training rewards and eval scores come from different graders, a gain can come
from the grader rather than the model, and nothing downstream distinguishes them.

**Watch for:** the p = 0 and p = 1 sets exist precisely because plain GRPO cannot
use them — they normalise to zero advantage. They are there for teacher-side
experiments, where a teacher's trajectory can supply signal that on-policy
sampling cannot. Do not put them in a plain GRPO run and expect anything.

---

## 2. Contamination: run the discriminating experiment

`docs/experiment.md` §3 establishes that the provenance test is *differential*
and cannot see uniform exposure. Both Qwen configurations pass it, which is the
absence of differential evidence rather than evidence of absence — and a 4B model
scoring 95% on every split of MATH is exactly the case where that distinction
matters.

Two independent probes. Either alone is suggestive; together they are close to
decisive.

- [ ] **Post-cutoff benchmark.** `MathArena/aime_2026`: 30 problems, all integer
      answers, zero integration cost against the existing AIME task. Qwen3's
      technical report is May 2025, so AIME 2026 (February 2026) is unambiguously
      after. Compare against AIME 2024 and HMMT Feb 2025 on the same models. A
      sharp drop is contamination evidence; parity weakens the hypothesis.
- [ ] **Verbatim-continuation probe.** Feed the first ~40% of a problem statement
      and measure overlap between the continuation and the true remainder, against
      a control set the model cannot have seen. This tests *exposure* directly and
      is independent of solving ability — a model that has not seen the text
      cannot reproduce it regardless of how good it is at math.

**Statistical limit, and it binds hard.** The standard error is set by the
problem count; extra sampling does not move it. At 30 problems it is ±8.9 points
around a 50% score, so two single competitions are distinguishable only past
roughly 30 points — larger than the 20.7-point effect measured on Llama. Any
date comparison must pool: HMMT Feb 2026 (33) + AIME 2026 (30) = 63 post-cutoff
against HMMT Feb 2025 (30) + HMMT Nov 2025 (30) + AIME 2024 (30) = 90 before.
Even pooled this resolves roughly 15 points, not 5.

---

## 3. Run HMMT

Configured and grading-calibrated; never executed.

- [ ] Run `hmmt` (93 problems pooled) on all three configurations, 16 samples,
      pass@4 headline. Also gives pass@1/@8/@16 from the same run.
- [ ] Run the three per-competition tasks for the date comparison in §2.

Cheap: 93 problems × 16 samples. Thinking mode is the long pole at ~30k tokens
per sample; shard it across 8 GPUs as with the profiling runs.

---

## 4. Save the benchmark generations

`run_all.sh` never passed `--save-generations`, so the reasoning traces behind
the six baseline runs were never written — the per-sample verdicts in `outputs/`
are all that survives. This is not a missing file; it is data that does not exist.

- [ ] Add `--save-generations` to `run_all.sh`.
- [ ] Re-run the six baselines with traces saved (~30 min sharded across 8 GPUs).

Needed before any error analysis, and before using teacher trajectories for
distillation — the premise of that work is that *the trajectory* is what the
student learns from, which correct/incorrect labels cannot supply.

---

## 5. Re-profile the zero-gradient problems at higher n

Both Qwen configurations have problems that are never solved in 8 samples: 779
for non-thinking, 295 for thinking. At `p = 0` they contribute no gradient, but
some fraction would land at 1/32 or 2/32 under more sampling — converting dead
weight into exactly the hard, trainable problems this project wants.

- [ ] Re-profile just those problems at n = 32. Cheap because the sets are small:
      ~10 min for non-thinking, ~1 h for thinking, against 1.5 h and 20 h for a
      full-pool re-profile.

Worth doing only if the Qwen configurations are wanted as students. As teachers
their pass@1 is what matters, and that is already measured.

---

## 6. Robustness and hygiene

Smaller items, each closing a real gap rather than tidying.

- [ ] **Incremental generation writes.** `evaluate.py` calls `llm.generate()`
      once and writes only after it returns, so a shard killed mid-run loses
      everything — as happened once already, costing a 12%-complete AIME shard.
      Batch the generate call and flush periodically.
- [ ] **A guard for the lumen-2 double mount.** `~/data` there has a local xfs
      mount shadowed by the rclone bucket. Writes currently reach the bucket
      correctly, but if rclone ever unmounts they would land on local disk with
      no error and no visibility from the other nodes. Any backup to `~/data`
      should verify by reading back from a different node (allow 90s for
      propagation).
- [ ] **Report the environment migration numbers in the results table.** The
      shared environment was validated by re-running Llama on MATH-500: 39.1%
      against 38.8%, inside the ±1.7 standard error. That evidence currently
      lives only in a commit message.

---

## Not planned

Recorded so the decisions are not revisited by accident.

**Qwen3-4B as the student.** Ruled out on headroom: 95.5% on MATH-500 in thinking
mode, and 92.9% of the training pool produces no gradient. Kept as a reference
point and a candidate teacher.

**A separate RL environment.** One environment evaluates and trains. The apparent
conflict — `math-verify[antlr4_13_2]` against verl's `hydra-core` pinning antlr4
to 4.9.\* — dissolves once the `antlr4_9_3` extra is used instead. Verified by
chat-template md5 equality, full grading calibration, and the re-run above.

**Committing the profiling records or raw generations.** 12–36 MB per record
file and ~1 GB of generations. Records rebuild from the shards via
`merge_shards.py`; generations live on the nodes and the shared bucket. Both are
copied identically to all three nodes under `outputs/math_profiling/` and
`~/data/frontier-teacher/generations/`.
