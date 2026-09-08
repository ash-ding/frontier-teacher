# Plan

Open work, ordered by what unblocks what. `docs/experiment.md` records what has
already been run; this file records what has not.

Everything below runs on three 8×H100 nodes under one shared environment
(vLLM 0.27.1 + verl 0.9.0 + flash-attn 2.8.3), verified identical across
`lumen-1`, `lumen-2`, `lumen-3`.

---

## 1. GRPO training — done, with two loose ends

Nine runs (3 configurations × 3 difficulty bands) plus all three
matched-group-size controls are complete; `docs/experiment.md` §7 has the design, the results, and
the operational failures worth knowing about. The report is rendered by
`tools/render_report.py` from `outputs/curves.json`.

- [ ] **The thinking hard band has no 10,240-rollout endpoint.** It OOMed at step
      19 of 20 and its final checkpoint does not exist. Four points, ringed in the
      figure. Re-running costs ~6 h of training; the cell shows no movement at any
      of its four points, so the endpoint would very likely be another null.
- [ ] **Llama's hard band was still rising when the budget ran out.** Extending it
      to 40 steps (20,480 rollouts) is ~35 min of training plus ~45 min of
      evaluation and is the cheapest way to learn whether the one transfer effect
      in the study plateaus or continues. This is the highest-value open item.

**Watch for:** the p = 0 and p = 1 sets exist precisely because plain GRPO cannot
use them — they normalise to zero advantage. They are there for teacher-side
experiments, where a teacher's trajectory can supply signal that on-policy
sampling cannot. Do not put them in a plain GRPO run and expect anything.

---

## 2. Teacher-side training — run, but not yet readable

Three teacher runs completed 2026-09-02/03, one per student configuration, and
`docs/experiment.md` §8 has the design, the results and the operational record.
The teacher supplies a **curriculum** — problems and answers only, no trajectory,
no rationale, no logits — so what it changed cannot be imitation of its reasoning.

The result as it stands: on no (student, benchmark) cell does the teacher produce
the largest gain, except thinking AIME where every arm is inside noise. It ranks
2nd of seven on all three benchmarks for non-thinking, and 6th / 7th / 2nd for
Llama. §8 also measures why, from training reward: the teacher settles near a 60%
curriculum and lets it drift 20–30 points easier over the run, having never been
told the `p(1-p)` argument that §1 uses to set group size.

- [ ] **Extend `tools/paired_stats.py` to the teacher arm. This blocks reading
      §8 at all.** Every number in §8 is currently a point estimate with a
      per-point standard error, and nothing in it is bolded, because the
      two-level bootstrap §7's intervals depend on has never been run here. The
      existing `outputs/analysis/paired_stats.json` is not merely stale — the
      script **structurally cannot see these runs**: it globs `grpo/*/records.jsonl`
      where the teacher evaluations are under `frontier-model/<run>/final_eval/`,
      and its `CKPT` regex requires a band slug matching `pass1_*`, which
      `..__teacher__step19__math500` is not. Re-running it unchanged reproduces
      the same 108 cells. The fix is a second glob path and a widened regex; the
      resampling itself needs no change, since teacher `records.jsonl` carries the
      same `{id, n, c}` and the `benchmarks/<cfg>__<task>` baselines are shared.
      No GPU. Until this is done, §8 supports rankings and not differences.
- [ ] **The teacher's key-error rate.** Its `answer` becomes the reward target
      unverified, so a wrong key trains the student toward a wrong answer. 320
      training problems per run; grading them against an independent solve is
      cheap and currently nobody knows the rate.
- [ ] **Overlap between the teacher's problems and the benchmarks.** §8 argues
      from three read curricula that the 5,152 problems are written rather than
      recalled. It does not measure it, and a teacher that reproduced an AIME
      problem into the curriculum would have leaked the evaluation.
- [ ] The `p = 0` subsets remain the sharpest available test and were not used:
      800 problems for Llama, 200 for thinking. Plain GRPO provably cannot use
      them, so movement there is attributable to the teacher rather than to more
      compute. (§7 revises "provably" — see §3 — but the asymmetry survives.)

---

## 3. Re-profile the zero-gradient problems at higher n

Promoted from "low value". §7's headline for the thinking configuration is that
92.9% of its pool yields no gradient, and that number is measured at n = 8 — a
problem solved 1 time in 32 is indistinguishable from one never solved at all.
Some fraction of the 295 thinking and 779 non-thinking `p = 0` problems are
really at 1/32 or 2/32, i.e. exactly the hard-but-reachable problems this project
wants, and they are currently being written off.

- [ ] Re-profile the `p = 0` sets at n = 32. Cheap because they are small: ~10 min
      for non-thinking, ~1 h for thinking, against 1.5 h and 20 h for a full-pool
      re-profile.
- [ ] If it recruits enough problems, the thinking hard band stops being 100
      problems seen 3.2 times — the confound §7 currently cannot remove.

It also puts a number on the claim itself. "92.9% produces no gradient" is the
evidence for needing a teacher at all; it should not rest on n = 8.

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
- [ ] **Baseline files drift between nodes, and the failure is silent.** Three
      separate times a paired analysis either refused to run or ran against the
      wrong numbers because a node held an older copy of a `__{task}.records.jsonl`
      baseline — record ids `aime-0000` against the current `aime-2020-00`, or the
      HMMT baseline simply absent. Scores are identical either way, so nothing
      looks wrong until the ids fail to intersect. The baselines belong in one
      place with a checksum, not copied per node by hand.
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
