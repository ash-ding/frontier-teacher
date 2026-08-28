# Plan

Open work, ordered by what unblocks what. `docs/experiment.md` records what has
already been run; this file records what has not.

Everything below runs on three 8×H100 nodes under one shared environment
(vLLM 0.27.1 + verl 0.9.0 + flash-attn 2.8.3), verified identical across
`lumen-1`, `lumen-2`, `lumen-3`.

---

## 1. GRPO training — running

Nine runs (3 configurations × 3 difficulty bands) are in flight; see
`docs/experiment.md` §7 for the design and for two findings that came out of
setting it up. Remaining work once they land:

- [ ] `src/collect_curves.py` over all nine, then the 3×3 figure — one row per
      benchmark, one column per configuration, three band lines per panel, with
      the §1 baseline as each line's rollout = 0 point.
- [ ] Verify each line's rollout = 0 point equals the §1 baseline. If it does
      not, training and evaluation are not scoring the same thing.
- [ ] Fold the completed numbers into `docs/experiment.md` §7.

**Watch for:** the p = 0 and p = 1 sets exist precisely because plain GRPO cannot
use them — they normalise to zero advantage. They are there for teacher-side
experiments, where a teacher's trajectory can supply signal that on-policy
sampling cannot. Do not put them in a plain GRPO run and expect anything.

---

## 2. Teacher-side training — the actual experiment

§1 is the control: plain GRPO, no teacher. Nothing in the repository has yet used
a frontier model's trajectories, which is the question it exists to answer.

- [ ] Decide what the teacher supplies — full trajectories, hints, or a curriculum
      over the bands — and which of the nine control curves each variant is
      measured against.

---

## 3. Re-profile the zero-gradient problems at higher n

Both Qwen configurations have problems that are never solved in 8 samples: 779
for non-thinking, 295 for thinking. At `p = 0` they contribute no gradient, but
some fraction would land at 1/32 or 2/32 under more sampling — converting dead
weight into exactly the hard, trainable problems this project wants.

- [ ] Re-profile just those problems at n = 32. Cheap because the sets are small:
      ~10 min for non-thinking, ~1 h for thinking, against 1.5 h and 20 h for a
      full-pool re-profile.

Low value now. Qwen serves only as a reference point — not as the student, and
not as the teacher — and a reference point's pass@1 is already measured. Worth
revisiting only if a Qwen configuration is wanted as a training target after all.

---

## 4. Robustness and hygiene

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
- [ ] **Disable auto-resume everywhere else it could bite.** `run_grpo.sh` now
      sets `resume_mode=disable`, but any future trainer entry point inherits
      verl's default of resuming from whatever checkpoint directory it finds.
- [ ] **Strip the redundant FSDP shards at the source.** Every checkpoint is
      written twice — `model_world_size_*.pt` for resuming and `huggingface/` for
      loading — 31 GB where 15 GB is used. With resume disabled the shards are
      dead weight; `checkpoint.save_contents=[hf_model]` drops them. A background
      janitor currently deletes them after the fact, which is a workaround.

---

## 5. Contamination: run the discriminating experiment — deprioritised

`docs/experiment.md` §3 establishes that the provenance test is *differential*:
it detects contamination that is asymmetric across splits, which is what
Llama shows, and is blind to uniform exposure. Both Qwen configurations pass it,
which is the absence of differential evidence rather than evidence of absence.

**This was near the top of the plan while Qwen3-4B was a candidate teacher.** It
no longer is, and that removes the reason that made it urgent. The concern was
specific: if a teacher's solutions are recalled rather than derived, its
trajectories can be post-hoc rationalisations of a known answer — correct, fluent,
and containing exactly the unexplainable jumps that make a trace unlearnable.
The teacher's own metrics would look fine while the student failed to improve,
and no downstream number would separate that from the method not working.

With Qwen serving only as a reference point, what remains is weaker: any
capability claim made about it (*"Qwen3-4B reaches 95.5% on MATH-500"*) is
misleading if the number is recall. That is worth settling before publication,
not before training.

If it is run later, the two probes are independent:

- [ ] **Verbatim-continuation probe.** Feed the first ~40% of a problem statement
      and measure overlap between the continuation and the true remainder, against
      a control the model cannot have seen. Tests *exposure* directly, independent
      of solving ability, and can run over all 11,996 problems rather than 30 — far
      more statistical power than the temporal comparison, and cheaper, since it
      needs no long reasoning.
- [ ] **Post-cutoff benchmark.** `MathArena/aime_2026`: 30 problems, all integer
      answers, zero integration cost. Cheap enough to run alongside anything else,
      but see the limit below before reading much into it.

**Why the temporal probe alone would not settle it.** The standard error is set
by the problem count and extra sampling does not move it: ±8.9 points at 30
problems, so two single competitions are distinguishable only past roughly 30
points — larger than the 20.7-point effect measured on Llama. Pooling HMMT Feb
2026 (33) + AIME 2026 (30) against HMMT Feb 2025 (30) + HMMT Nov 2025 (30) +
AIME 2024 (30) gets that to about 15 points. A sharp drop would be strong
evidence; parity would prove very little. Prefer the continuation probe.

---

## Not planned

Recorded so the decisions are not revisited by accident.

**Qwen3-4B as the student.** Ruled out on headroom: 95.5% on MATH-500 in thinking
mode, and 92.9% of the training pool produces no gradient. It is kept purely as a
reference point — it will not be the teacher either.

**A separate RL environment.** One environment evaluates and trains. The apparent
conflict — `math-verify[antlr4_13_2]` against verl's `hydra-core` pinning antlr4
to 4.9.\* — dissolves once the `antlr4_9_3` extra is used instead. Verified by
chat-template md5 equality, full grading calibration, and the re-run above.

**Committing the profiling records or raw generations.** 12–36 MB per record
file and ~1 GB of generations. Records rebuild from the shards via
`merge_shards.py`; generations live on the nodes and the shared bucket. Both are
copied identically to all three nodes under `outputs/math_profiling/` and
`~/data/frontier-teacher/generations/`.
