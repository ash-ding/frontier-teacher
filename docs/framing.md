# Framing

What question this repository is positioned to answer, what would make the answer
publishable, and where the current setup is weakest. `docs/experiment.md` is the
record of what ran; `docs/plan.md` is the queue. This file is neither — it is the
argument for what the queue should be for.

---

## 1. The question, stated so it can fail

> With the student, the optimiser, the group size, the rollout budget, the
> response cap and the grader all held fixed, does a frontier model choosing the
> problems spend that budget better than the best fixed difficulty band?

Everything in that sentence except *the problems* is already pinned by
`run_teacher_step.sh` and read from the run's frozen config, which is what makes
the comparison causal rather than suggestive. A teacher run and a band run are
twenty steps of 512 rollouts each, evaluated at the same four milestones by the
same `eval/evaluate.py`. One variable moves.

Two properties of that design are worth stating early, because they are the
paper's whole claim to cleanliness and they are easy to lose by accident:

**The channel is problems and answers, and nothing else.** The teacher writes
`{id, problem, answer}`. No solution trace, no rationale, no logits, no
preference labels. Whatever the student gains therefore cannot be explained as
imitation of the teacher's reasoning — the teacher's reasoning never reaches it.
That distinguishes this from every distillation result, and it sets up the
sharper version of the question: *how much of a frontier model's advantage can
be transmitted through problem selection alone?* Answering "some, measurably" is
interesting; answering "none" is also interesting, and both are papers. Do not
widen the channel to rescue a null.

**The control is a ladder, not a coin flip.** Fifteen cells and three
matched-group-size controls already exist. Almost every curriculum-RL result
compares against uniform sampling; this one can compare against *the best fixed
band*, chosen with hindsight from a measured sweep. That is a much stronger
statement and it is already paid for.

### The second question, which does not depend on the first

§7 established, on the control alone, that **in-domain gains are
curriculum-invariant and only transfer discriminates**: Llama's three bands land
at +5.0 / +6.3 / +6.0 on MATH-500 with almost fully overlapping intervals, while
on AIME only the hard band moves (+4.9 [0.3, 8.9], monotone over four
checkpoints). The non-thinking runs put their entire AIME gain in the second
half of training, after MATH-500 had already flattened — three independent
replications of a reversal that a normal early-stop rule would have hidden.

That is a methodological result about how curriculum experiments are read, and
it holds whether or not the teacher wins. It should be a section, not a footnote.

---

## 2. What a teacher can do that a band cannot

Five mechanisms. Each is a claim, so each is paired with the ablation that tests
it; a mechanism with no ablation beside it is decoration.

| Mechanism | Why a band cannot | The test |
|---|---|---|
| **Generation, not selection.** The teacher writes problems; a band is a subset of a finite pool. | Llama's 5–15% band holds 487 problems and 400 were sampled. Twenty steps at 16 problems/step is 320 — 0.80 epochs. The 40-step extension in `plan.md` §1 needs 640 and the pool cannot supply them. | Run long enough that the band repeats. Report epochs alongside rollouts. |
| **Diagnosis from traces, not scores.** The teacher reads `generations.jsonl`. | A pass rate says *hard*; the generation says *arithmetic slip* / *wrong method* / *ran out of tokens* / *solved it and the extractor missed it*. Those imply different next problems. | Ablate: give the teacher `summary.json` only, withhold generations. |
| **Non-stationarity.** Bands are profiled once, offline, before step 0. | A "5–15%" band measured on the base model is not 5–15% for the step-19 checkpoint. The label decays over exactly the run it is labelling. | Re-measure each band's pass rate on its own final checkpoint. Free: the training rollouts already contain it (`critic/rewards/mean`). |
| **Batch composition.** A curriculum is not i.i.d. draws — ordering, mixing, avoiding all-zero batches. | `run_teacher_step.sh` fixes the batch size, so composition is the only degree of freedom left, and it is exactly the one a band does not have. | Compare realised per-step reward variance, teacher against band. |
| **No labelled pool required.** The teacher supplies the answer key. | Where no curated set exists, a band cannot be defined at all. | §5 below — this is the mechanism that deserves its own experiment. |

The first four are advantages *at the margin* over an existing dataset. The
fifth is structural, and it is the one that decides where this method actually
matters.

---

## 3. The baseline ladder

The single most likely reviewer objection to a positive result is: *the gain
came from adaptivity, not from the frontier model.* One cheap arm removes it,
and the ladder as a whole is the spine of the paper — each rung isolates one
factor.

1. **Fixed band** — exists (three bands × three configurations, plus matched G).
2. **Uniform pool** — the conventional control; cheap, one run.
3. **Adaptive, no LLM** — resample each step toward the group-variance optimum
   using the student's own last-step rollouts, which verl already computes.
   `p(1−p)` learnability, or DAPO-style filtering of degenerate groups. This
   costs no extra generation and it is the arm that isolates *adaptivity*.
4. **Frontier teacher, blind** — writes all twenty curricula up front, never
   evaluates, never reads a trace. Isolates *the closed loop*.
5. **Frontier teacher, in the loop** — the method.
6. **Weaker teacher** — Qwen3-4B non-thinking under the identical protocol.
   Isolates *teacher strength*, and is the only arm that justifies the word
   *frontier* in the title.

Rungs 3 and 4 are the ones that make the result defensible. Rung 6 is what makes
the framing honest.

A seventh measurement, not an arm: **replay**. The teacher's twenty curricula are
a recorded artefact. Replay them on a fresh student with no teacher present. If
the gain survives, the deliverable is a dataset and the teacher is a one-time
cost; if it does not, the value is in the adaptation and cannot be amortised.
Either answer is worth having, and the experiment is one training run.

---

## 4. Is mathematics the right task

Separate two things the domain has to do: be a good *instrument* (can the
experiment be measured at all) and be a good *motivation* (does anyone need this
method here). Mathematics is excellent at the first and weak at the second, and
conflating them is how this project would end up with a technically sound paper
nobody needs.

A rubric, and where mathematics lands:

| | | |
|---|---|---|
| **(a) Verifiable reward** | essential — the teacher's labels must be checkable and RL must be stable | **yes**, and the graders here are calibrated (§5) |
| **(b) Student headroom under pass@k** | otherwise the curriculum reallocates mass instead of adding capability | **partial** — Llama has room (38.8 / 12.0 / 1.8), Qwen thinking has none |
| **(c) Labelled data scarce or mismatched** | the teacher's generation ability only earns its cost where selection cannot | **no** — an 11,996-problem curated pool sits in `data/` |
| **(d) Teacher can label what the student cannot solve** | a necessary condition, and unmeasured | **yes**, but the key-error rate is not known |
| **(e) Target distribution unknown or moving** | otherwise offline curation suffices | **partial** |
| **(f) Contamination knowable** | else no capability claim survives | **poor** — §3 measured 20.7 points of it on Llama |

Row (c) is the honest weakness, and it is the user's instinct restated: these
models were trained hard on mathematics before release, an enormous curated pool
already exists, and "another math RL result" competes with a crowded field. Row
(b) sharpens it — Llama's `p = 1` band gains +3.7 in domain and nothing out of
it, which is the repository's own demonstration that in-domain movement can be
consistency rather than capability. A ceiling set by what the base model can
already do at pass@k caps every arm equally.

None of that is a reason to leave mathematics. It is a reason to be explicit that
**mathematics is the instrument**: it is where the control is already run, where
grading is trustworthy, and where a result is legible to readers. Three changes
make it carry more:

- **Move the target off the saturated part of the distribution.** AIME and HMMT
  are already the transfer benchmarks and already show that MATH-500 cannot
  distinguish curricula. `MathArena/aime_2026` and HMMT Feb 2026 are post-cutoff
  and already in `data/benchmark/`. Note the power limit from §6 before leaning
  on them: ±8.9 points at 30 problems, so competitions must be pooled.
- **Consider a student whose training data is knowable.** An OLMo-family base
  model turns contamination from an inference into a lookup. That is a real
  publication advantage over inferring exposure from a train/test asymmetry.
- **Measure what the teacher writes.** Two numbers that do not exist yet and are
  cheap: the **key-error rate** (fraction of teacher-supplied answers that are
  wrong — a wrong key trains the student toward a wrong answer, with no
  verification anywhere in the loop), and the **benchmark overlap rate** (near-
  duplicate match of every teacher-written problem against MATH-500, AIME and
  HMMT). The second is not optional. A teacher that recalls AIME 2021 problems
  into the training set has leaked the evaluation into the curriculum, and that
  single objection would sink the result. Measure it, report it, and make the
  protocol reject matches.

---

## 5. Adaptation without a labelled set — the track with the better story

The setting the method is actually for: a target task with **no training set** —
a handful of unlabelled target instances, a way to check an answer, and a
compute budget. An in-house DSL or API, a codebase's own conventions, a new
specification, a fresh competition distribution, a formal library. There is no
band to define because there is no pool to slice. Selection is unavailable;
authoring is the only move. The teacher's advantage stops being marginal.

> Given a small set of unlabelled target instances and a fixed compute budget,
> can a frontier model author a training curriculum that adapts a small model to
> that target — and does training at test time beat spending the same compute on
> sampling at test time?

The second clause is the one reviewers reach for immediately, and it is a good
question rather than an obstacle. Best-of-n with the same FLOPs is the baseline
that matters; if a teacher-authored curriculum plus a few LoRA steps beats it,
that is a genuinely notable result, and if it does not, that too is worth
publishing plainly.

Three reasons this track is stronger than it first looks:

**It fixes the statistical power problem.** Today a run is one point on a curve
and costs hours; §7's headline effect (+4.9 on AIME) sits barely above the
harness's own run-to-run variance for an identical model (12.0 against 15.3),
and every cell is n = 1. Under per-target adaptation, *each target is a
replicate*. Fifty targets, each a few steps of LoRA, paired against a
no-adaptation control, is a paired test over fifty units — vastly more power than
comparing two five-point curves, and cheaper in total.

**It has a distinct neighbour in the literature.** Test-time RL with
majority-vote pseudo-labels is the obvious adaptive-without-labels baseline, and
it is exactly rung 3 of the ladder in a new setting: adaptive, no teacher, labels
manufactured by the student's own agreement. The frontier-teacher version differs
in one specific way — the teacher supplies *reliable labels for problems the
student cannot solve*, which self-consistency by construction cannot. That is a
one-sentence differentiation and a clean experiment.

**It keeps the existing harness.** `--data` takes any path and the teacher's
interface is data, so a per-target run needs no new registration and no config
synthesis. The pieces that would be new are a target-set builder and a
short-horizon (LoRA, few-step) training path.

Risk, stated: a 3B student may not move at all in a handful of steps, and a
single problem is a 0/1 measurement. Define a target as a small *cluster* —
one contest, one topic, one API surface — and score held-out instances from that
cluster. If the cluster has no held-out instances, it is not a measurable target.

---

## 6. What would sink the result, and what closes each

| Objection | Status | What closes it |
|---|---|---|
| Every cell is n = 1, at 20 steps where recipes use hundreds | **open, and the largest threat** | Seed replicates on the headline control cell and the teacher cell. Nothing else on this list matters if this is not done. |
| The teacher's compute is not counted | open | Report teacher tokens and wall clock beside GPU hours; state the rollout-matched and compute-matched comparisons separately rather than choosing one. |
| The teacher read the reference tests | **handled** | Held-out four-fifths scoring, plus §3 of the report which measures the leak directly. Keep it foregrounded — it is a strength, not an embarrassment. |
| Teacher-written problems duplicate the benchmarks | **open, and cheap** | Near-duplicate check of every written problem against all three benchmarks; reject at protocol level, report the rate. |
| "This is distillation" | structurally answered | The channel is problem and answer only. Say so early; do not widen it later. |
| Wrong answer keys train wrong behaviour | open | Independent verification of teacher keys; report the error rate as a finding, not a caveat. |
| One student, one domain | open | Rung 6 of the ladder, plus §5. |
| The gain is adaptivity, not the frontier model | open | Rung 3 of the ladder. |

Pre-register the primary endpoint before the teacher runs go out. Three
benchmarks × five checkpoints × several arms is a large enough grid that a
post-hoc choice of headline cell will not survive scrutiny, and the repository
has already been careful enough elsewhere that it would be a shame to lose it
here. The defensible choice given §7: **AIME pass@4 at 10,240 rollouts on the
held-out set, Llama student, two-level bootstrap**, with MATH-500 pass@1
secondary and HMMT reported for completeness only.

---

## 7. Sequencing

Ordered by what unblocks what, and priced against the existing record.

**Phase 0 — make the control worth comparing against.** The two open items in
`plan.md` §1 and §3 are not tidying; they are prerequisites. Extending Llama's
hard band to 40 steps (~35 min train, ~45 min eval) determines whether the only
transfer effect in the study plateaus, and a teacher compared against a truncated
control is comparing against an unknown. Re-profiling the `p = 0` sets at n = 32
(~10 min / ~1 h) settles whether the "untrainable" region is real. Add seeds.
Pre-register the endpoint. **Nothing else should start first.**

**Phase 1 — the ladder.** Rungs 2, 3, 4, 6 (rung 5 exists). Rung 3 is the
cheapest and the most load-bearing.

**Phase 2 — mechanism.** Instrument what the teacher does: realised difficulty of
each written batch on the current checkpoint, novelty against the pool,
key-error rate, benchmark overlap, and whether the batch's realised pass rate
tracks a moving target while the fixed band's drifts. Then the replay
experiment from §3. This is what turns a win into an explanation and a loss into
a finding.

**Phase 3 — the label-free track.** §5. Target-cluster builder, short-horizon
training path, best-of-n-at-equal-compute baseline, paired statistics over
targets.

**Phase 4 — write-up.** `tools/render_report.py` already produces the artefact;
sections 2 and 3 already have the teacher panels and the leak measurement.

---

## 8. What ships regardless

The risk in a single-hypothesis project is that a null leaves nothing to
deliver. That is not the position here. Three things exist independent of whether
the teacher wins:

- **The control study.** Fifteen cells, three matched controls, difficulty
  against transfer, with the finding that in-domain benchmarks cannot
  distinguish curricula and that the two model configurations disagree about
  which band transfers.
- **The testbed.** A budget-matched, interface-minimal harness for evaluating any
  curriculum policy — LLM or not — against a measured difficulty sweep, with one
  grader shared by training and evaluation. That is a contribution on its own,
  and it is the thing other people would actually reuse.
- **The negative results worth knowing**, which `docs/experiment.md` already
  keeps: that `p = 0` bands are not degenerate, that a response cap is part of
  the reward, that training on solved problems buys consistency rather than
  capability.

The teacher result is the headline. It is not the deliverable of last resort.
