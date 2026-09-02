# Experiment log

What has been run, what it measured, and what the numbers mean. The README covers
the repository and how to set it up; this file is the experimental record.

Everything below was produced on three 8×H100 nodes (`lumen-1`, `lumen-2`,
`lumen-3`) under one shared environment: vLLM 0.27.1, torch 2.13.0+cu130,
transformers 5.10.4, math-verify 0.9.0 with the `antlr4_9_3` extra, verl 0.9.0.
Decoding uses each vendor's recommended preset at seed 1234.

---

## 1. Baselines

Three model configurations on three held-out benchmarks. These are also the
rollout = 0 point of every training curve in §7, so they were re-run with
`--save-generations` and the traces are kept.

| Configuration | MATH-500 pass@1 | AIME 2020–2024 pass@4 | HMMT pass@4 |
|---|---:|---:|---:|
| Llama-3.2-3B-Instruct | 38.8% ± 1.7 | 15.3% ± 2.6 * | 1.8% ± 1.1 |
| Qwen3-4B non-thinking | 83.4% ± 1.4 | 36.0% ± 3.6 | 20.5% ± 3.7 |
| Qwen3-4B thinking | 95.9% ± 0.8 | 81.0% ± 3.0 | 57.0% ± 4.7 |

MATH-500: 4 samples per problem, pass@1 as the mean.
AIME: 8 samples, pass@4 via the unbiased estimator (Chen et al., 2021).
HMMT: 93 problems pooled over Feb 2025 / Nov 2025 / Feb 2026, 16 samples.

**HMMT does not resolve anything for Llama.** 1.8% pass@4 is 1.7 problems out of
93; the ±1.1 error bar reaches the floor. Any Llama gain on HMMT would have to be
enormous to clear noise, so for the student model HMMT is reported for
completeness and MATH-500 carries the signal. It separates the two Qwen
configurations well (20.5% against 57.0%), which is what it was added for.

The secondary metrics from the same runs: AIME pass@1 is 6.3% / 21.3% / 68.9%,
and HMMT pass@1 is 0.7% / 11.8% / 43.5%.

\* A second run of Llama's AIME returned 12.0%. See the variance note below.

**A check against the reference work.** Llama-3.2-3B lands on 38.8% where the
Pedagogical RL blog states 38%, which says the grading path is calibrated rather
than silently depressing scores.

**AIME has run-to-run variance worth stating before any AIME number is read.**
Two independent evaluations of the identical model over the identical 150
problems at n=8 (verified: the problem-text sets have the same md5) returned
pass@1 6.3% and 4.8%, pass@4 15.3% and 12.0%. A fixed seed does not make this
reproducible — vLLM's continuous batching changes the numerics with the shard
layout, and the first run was sharded across 8 GPUs where the second was not.
So Llama's AIME pass@4 is 12–15%, and a ±3-point move on AIME is the harness,
not the model. Comparisons against a checkpoint must therefore use a baseline
run under the same conditions as the checkpoint evaluations, which is what §7
does; the 8-GPU sharded original is kept in `outputs/superseded/`.

This is also why the curve analyses bootstrap over BOTH problems and
generations. Resampling problems alone treats each problem's measured c-of-n as
exact, and at n=8 that is not a safe assumption — it understated the intervals
on AIME and HMMT by roughly a third.

**Qwen3-4B cannot serve as the student.** At 95.5% on MATH-500 there are 4.5
points of headroom, so the ~4-point effect the paper reports (44.7% → 48.6%)
could not be resolved above noise. Both Qwen configurations are kept as
reference points and candidate teachers.

### Harness health

Reported beside every score, because a high value means the number is a parser
artefact rather than a model result.

| Run | Truncation | No answer |
|---|---:|---:|
| Llama MATH-500 | 3.2% | 10.7% |
| Llama AIME | 8.3% | 11.3% |
| Qwen non-think MATH-500 | 1.5% | 1.2% |
| Qwen non-think AIME | 15.3% | 12.4% |
| Qwen think MATH-500 | 1.1% | 0.5% |
| Qwen think AIME | 13.3% | 6.2% |

Llama emits no `\boxed{}` on about 10% of samples, and truncation accounts for
only 3.2% of that — the rest is weak format adherence at 3B. Thinking-mode AIME
truncates on 13.3% of samples against the 30,000-token cap, so its true AIME
score is somewhat above the figure reported.

**Not saved:** `run_all.sh` did not pass `--save-generations`, so the reasoning
traces behind these six runs were never written. The per-sample verdicts and
extracted answers in `outputs/` are all that exists of them.

---

## 2. Difficulty profiling

Every problem in the 11,996-problem MATH training pool, sampled directly to
measure its empirical pass@1. This is what decides which problems can train at
all: group-normalised RL produces gradient only when `0 < pass@1 < 1`, since an
all-correct or all-wrong group normalises to zero advantage.

| Model | Samples | Generations | Mean pass@1 | Wall clock |
|---|---:|---:|---:|---:|
| Llama-3.2-3B | **32** | 383,872 | 53.3% | ~26 min |
| Qwen3-4B non-thinking | 8 | 95,968 | 83.2% | ~21 min |
| Qwen3-4B thinking | 8 | 95,968 | 95.0% | ~5.1 h |

Llama gets 32 samples because the training subsets are selected from it, and the
paper's target of pass@1 ≈ 8% is not representable at 8 samples: pass@1 is
quantised to `c/n`, so 8 samples can only express 0, 12.5%, 25%… The Qwen runs
are comparisons, not selection pools, and 8 samples suffice.

### How much of the pool can train

| Model | All wrong (p=0) | Trainable | All correct (p=1) |
|---|---:|---:|---:|
| Llama-3.2-3B | 1,216 (10.1%) | **10,209 (85.1%)** | 571 (4.8%) |
| Qwen3-4B non-thinking | 779 (6.5%) | 3,018 (25.2%) | 8,199 (68.3%) |
| Qwen3-4B thinking | 295 (2.5%) | **851 (7.1%)** | 10,850 (90.4%) |

Running GRPO on this pool with Qwen thinking would spend more than nine tenths
of the compute on prompts that produce no gradient. The distributions differ in
shape, not only in level: Llama's is U-shaped with mass at both ends, Qwen
non-thinking's is a J, and Qwen thinking's is closer to an L.

### Full distribution, Qwen at 8 samples

| c | pass@1 | non-thinking | thinking |
|---:|---:|---:|---:|
| 0 | 0.0% | 779 | 295 |
| 1 | 12.5% | 336 | 71 |
| 2 | 25.0% | 288 | 65 |
| 3 | 37.5% | 308 | 72 |
| 4 | 50.0% | 294 | 75 |
| 5 | 62.5% | 419 | 88 |
| 6 | 75.0% | 510 | 142 |
| 7 | 87.5% | 863 | 338 |
| 8 | 100% | 8,199 | 10,850 |

Thinking empties the middle: every value from c=1 to c=5 holds under 90 problems,
against 288–419 without it.

### Llama at 32 samples

33 achievable values rather than 9. The distribution is U-shaped, and `p = 0` at
814 problems (of the clean pool) is the single largest bar — 2.6× the next.
Notably `p = 1` (131) sits *below* c=30 (162) and c=31 (145): solving a problem
on all 32 draws demands a per-sample reliability few problems reach.

Mean pass@1 by MATH level, full pool:

| Level | Problems | Llama | Qwen non-think | Qwen think |
|---|---:|---:|---:|---:|
| 1 | 958 | 77.2% | 97.0% | 99.3% |
| 2 | 2,151 | 67.3% | 94.2% | 97.9% |
| 3 | 2,619 | 59.5% | 91.3% | 97.2% |
| 4 | 2,775 | 51.1% | 84.6% | 96.0% |
| 5 | 3,493 | 35.0% | 65.2% | 89.6% |

A cross-check on the method: within each model, the share of Level 5 problems
falls monotonically as measured pass@1 rises (Qwen non-thinking: 67.5% at c=0
down to 17.6% at c=8). The measurement is tracking real difficulty, not noise.

---

## 3. Contamination

### Llama-3.2-3B was trained on the original MATH train split

The 12k pool is 7,498 problems from the original Hendrycks *train* split plus
4,498 the PRM800K re-split moved out of the original *test* split. Splitting the
profile by provenance:

| Pool | Problems | Mean pass@1 | All-wrong rate |
|---|---:|---:|---:|
| Original train split | 7,498 | **61.0%** | 5.4% |
| Original test split, in the 12k | 4,498 | 40.3% | 18.1% |
| MATH-500 (held out, control) | 500 | 38.8% | — |

The test-derived half matches the held-out control. The train-derived half sits
20.7 points above it, and the gap widens monotonically with difficulty:

| Level | orig_train | orig_test | Gap |
|---|---:|---:|---:|
| 1 | 81.5% | 71.0% | +10.5 |
| 2 | 73.4% | 57.2% | +16.2 |
| 3 | 67.1% | 47.7% | +19.4 |
| 4 | 61.9% | 34.4% | +27.5 |
| 5 | 44.0% | 17.8% | +26.2 |

That widening is the signature of memorisation rather than a difficulty artefact:
on easy problems the model can genuinely solve the task so exposure adds little;
on hard ones, where real ability collapses, recall dominates.

**All Llama subsets are drawn only from the 4,498 clean problems.**

### Neither Qwen configuration shows the same asymmetry

| Model | orig_train | orig_test | MATH-500 |
|---|---:|---:|---:|
| Qwen3-4B non-thinking | 82.5% | 84.2% | 83.3% |
| Qwen3-4B thinking | 94.7% | 95.6% | 95.5% |

Within 1.7 points, with the test-derived half marginally *higher* in both cases —
the opposite direction from contamination. Their subsets therefore use the full
pool.

**Read that narrowly.** A train-vs-test comparison is a *differential* test. It
detects contamination that is asymmetric across splits, which is what Llama shows.
It is blind to uniform exposure: if all 12,500 MATH problems were in a model's
training data, every split would lift together and the difference would vanish,
producing a reading identical to a genuinely clean model. The Llama result is
positive evidence and stands on its own; the Qwen result is only the *absence of
differential evidence*, which is weaker — particularly for a model described as
aggressively mid-trained for math that scores 95% on every split at 4B parameters.

Separating the two hypotheses needs either a benchmark released after the training
cutoff, or a verbatim-continuation probe on the problem statements. **Neither has
been run.** The HMMT sets in `data/benchmark/` were added for the first of these.

---

## 4. Curated subsets

`data/further_improve/<model>/`, sliced by measured difficulty. Filenames encode
model, band and count. Sampling is uniform without replacement under seed
20260828, with a separate stream per band, so changing one band's size does not
perturb another's draw; re-running reproduces byte-identical files.

**Llama-3.2-3B** — 2,200 problems from the 4,498 clean pool, profiled at 32
samples (resolution 3.1%):

| Band | Pool | Sampled |
|---|---:|---:|
| p = 0 | 814 | 800 |
| 5% ≤ p < 15% | 487 | 400 |
| 40% ≤ p ≤ 60% | 570 | 500 |
| 85% < p < 95% | 437 | 400 |
| p = 1 | 131 | 100 |

The 5–15% band has a measured mean pass@1 of 8.7%, the closest analogue here to
the paper's ~8% target.

**Both Qwen configurations** — same bands, so the two decoding modes sit on the
same grid and are directly comparable:

| Band | non-thinking pool → taken | thinking pool → taken |
|---|---:|---:|
| p = 0 | 779 → 700 | 295 → 200 |
| 12.5% ≤ p ≤ 25% | 624 → 600 | 136 → 100 |
| 37.5% ≤ p ≤ 62.5% | 1,021 → 1,000 | 235 → 200 |
| 75% ≤ p ≤ 87.5% | 1,373 → 1,000 | 480 → 400 |
| p = 1 | 8,199 → 1,000 | 10,850 → 500 |
| | **4,300 total** | **1,400 total** |

Qwen bands group adjacent `c/8` values rather than taking single points. An
earlier version took single points (c = 0, 2, 4, 6, 8) and left 1,926 of the 3,018
trainable non-thinking problems — 64%, including the hardest usable band at
12.5% — in no subset at all.

Llama keeps its own bands: at 32 samples pass@1 takes 33 values rather than 9,
so the two are not on the same grid to begin with.

Each record carries `id`, `problem`, `answer`, `level`, `subject`, `pass_at_1`,
`n_correct`, `n_samples`, `split_origin`, `band`, `profiled_model`. Every
directory has a `manifest.json` recording the decoding settings the pass@1 was
measured under, the pool policy and why, and each subset's composition.

`pass_at_1` is a property of *a model under a specific decoding configuration*,
not of the problem. Read it with the manifest's `profiled_with` block.

---

## 5. Grading calibration

A score is only worth as much as the parser under it, and a parser that fails
silently is indistinguishable from a weak model. `eval/calibrate.py`
exercises `grade()` on known inputs — no GPU — so any failure is unambiguously a
harness bug:

| Check | What it catches |
|---|---|
| identity | The gold answer, boxed, must grade correct. Below 100% the extractor cannot read this answer format. |
| equivalence | 28 pairs that are the same value written differently. Failures mean correct answers get marked wrong. |
| specificity | A sentinel answer must grade **wrong**. Without it, a grader returning `True` unconditionally passes everything else. |
| thinking | Identity behind a `</think>` tag, since thinking models are graded only on what follows it. |

All four pass at 100% on MATH-500 (500), AIME (150) and HMMT (93).

**One defect found and fixed.** `\frac{-1\pm\sqrt{17}}{2}` — the natural way to
write two roots — graded wrong against a gold of two comma-separated values.
`grading.py` now expands `\pm` and `\mp` before comparing.

**One behaviour documented rather than changed.** `math_verify` compares
numerically within a tolerance, so a truncated decimal (`-0.047619047619`)
matches an exact form (`-\frac{1}{21}`). This makes the grader more permissive,
equally so for every model.

---

## 6. What has not been run

- **A post-cutoff contamination test.** The discriminating experiment described
  in §3. `MathArena/aime_2026` is the cheapest starting point: 30 problems, all
  integer answers, zero integration cost against the existing AIME task.
- **Teacher-side training.** §7 is plain GRPO — the control that a teacher has to
  beat. Nothing has yet used a frontier model's trajectories.

### A statistical limit worth stating before HMMT runs

The standard error is set by the number of problems, and no amount of extra
sampling moves it. At 30 problems it is ±8.9 points around a 50% score, so two
single competitions can only be distinguished if they differ by roughly 30 points
— larger than the 20.7-point contamination effect measured on Llama. Any
date-based comparison has to pool competitions to have the power to see anything.

---

## 7. GRPO across difficulty bands

Nine runs: three model configurations × three bands of measured pass@1 (low but
not zero, ~50%, high but not one). Each run is 20 optimisation steps consuming
512 rollouts per step, so all nine sit on a shared x-axis of 10,240 student
rollouts, with checkpoints every 5 steps evaluated on all three benchmarks by
the unchanged `eval/evaluate.py`.

**Group size follows the band.** A GRPO group whose G rollouts are all wrong (or
all right) has zero advantage and contributes no gradient; the wasted fraction is
`(1-p)^G + p^G`. At the bands' measured means that is 48.3% for G=8 on a p≈8.7%
band but under 1% at p≈50%, so the extreme bands use G=32 and the middle band
G=8. Train batch is set against G to hold 512 rollouts/step (16×32 and 64×8),
with the mini-batch at half of it — two gradient updates per step.

**This is a pilot, not a converged comparison.** 20 steps is where standard GRPO
recipes use hundreds. It answers "does the curve move, and does it move
differently by band", nothing stronger. The thinking middle band is thinner
still: 200 problems over 20 steps is 6.4 epochs, so a rise there is as plausibly
memorisation of 200 items as learning.

### Results: fifteen cells and three controls

Change from the untrained model at 10,240 rollouts, with a two-level bootstrap
over both problems and generations. Bold is an interval clear of zero.

**Llama-3.2-3B** — baseline 39.1 / 12.0 / 1.8

| Band | MATH-500 pass@1 | AIME pass@4 | HMMT pass@4 |
|---|---|---|---|
| hard, 5–15% | 44.0 **+5.0** [2.1, 8.1] | 17.0 **+4.9** [0.3, 8.9] | 2.1 +0.3 [−1.7, 2.4] |
| medium, 40–60% | 44.4 **+5.3** [2.5, 8.3] | 14.3 +2.2 [−2.5, 6.4] | 2.8 +1.0 [−1.1, 3.5] |
| easy, 85–95% | 45.1 **+6.0** [3.1, 9.0] | 13.8 +1.8 [−2.2, 5.7] | 2.3 +0.5 [−1.3, 2.5] |

**Qwen3-4B non-thinking** — baseline 83.4 / 36.0 / 20.5

| Band | MATH-500 pass@1 | AIME pass@4 | HMMT pass@4 |
|---|---|---|---|
| hard, 12–25% | 85.7 **+2.4** [0.4, 4.2] | 40.3 +4.3 [−0.1, 8.8] | 25.3 **+4.8** [0.4, 9.2] |
| medium, 37–62% | 87.3 **+4.0** [1.9, 5.9] | 40.0 +4.0 [−0.9, 8.9] | 23.3 +2.7 [−1.3, 7.2] |
| easy, 75–87% | 86.5 **+3.1** [1.1, 5.1] | 40.9 **+4.8** [0.0, 9.5] | 25.0 **+4.5** [0.2, 9.0] |

**Every cell moves up on the benchmark its training data came from.** Plain GRPO
on MATH problems improves MATH-500 for both configurations and all three bands.
Nothing below disputes that; the interest is in what else moves, and when.

#### The degenerate bands are not degenerate

`p = 0` and `p = 1` were included as the cases plain GRPO provably cannot use:
a group whose G rollouts all fail, or all succeed, has zero advantage and no
gradient. **That reasoning was wrong**, and the error is instructive. The bands
were profiled at n = 8 (Qwen) or n = 32 (Llama); training draws G = 32 fresh
samples every step. "Never solved in 8 samples" is not "never solved".

Measured training reward on the `p = 0` bands, first step to last:

| | first step | last step | 512 rollouts correct, first → last |
|---|---:|---:|---|
| Llama | 1.4% | 6.3% | 7 → 32 |
| non-thinking | 4.7% | 9.6% | 24 → 49 |
| thinking | 0.4% | 1.8% | 2 → 9 |

At a true rate of 4.7%, eight samples miss it 68% of the time. `critic/advantages/max`
reaches 5.48 on nearly every step of every `p = 0` run — the exact value for a
group of 32 with one success, which is the strongest signal a group can carry.
For the thinking `p = 0` band, bounding from the reward, at most 48% of its 320
groups were degenerate and exactly one step out of twenty had no gradient at all.

**This corrects a claim made elsewhere in this file.** "92.9% of the thinking
pool produces no gradient" is derived from an n = 8 profile and is an
overestimate. The direction of the argument — that this configuration has little
for on-policy RL to work with — survives; the number does not. `docs/plan.md` §3
(re-profile the `p = 0` sets at n = 32) moves from optional to necessary.

#### Training on what a model already knows buys consistency, not capability

Llama's `p = 1` band — problems it solved 32 times out of 32 when profiled —
gains **+3.7 [1.6, 5.9] on MATH-500**, an interval clear of zero and larger than
several bands achieve on the transfer benchmarks. Out of domain it gains nothing:
+0.8 on AIME, +0.9 on HMMT, both well inside the noise.

The mechanism is visible in the training reward: it starts at 97.3%, not 100%. At
temperature the model fails these problems a few percent of the time, and GRPO
suppresses that failure. What improves is reliability on the distribution the
problems came from, which is exactly what does not transfer.

#### The in-domain metric saturates where transfer begins

Split each non-thinking run at 5,120 rollouts:

| | first half | second half |
|---|---|---|
| MATH-500, hard band | **+2.2** (p = 0.001) | +0.1 (p = 0.46) |
| AIME, hard band | −0.7 (p = 0.64) | **+5.0** (p = 0.008) |
| AIME, medium band | −2.2 (p = 0.87) | **+6.1** (p = 0.002) |
| AIME, easy band | +0.6 (p = 0.39) | **+4.2** (p = 0.009) |

All three bands put their entire AIME gain in the second half and none of it in
the first — three independent replications of the same reversal. Stopping at
5,120 rollouts because MATH-500 had flattened, which is what a normal early-stop
rule would do, produces the conclusion that nothing transferred.

#### For the student, only the hard band transfers

Llama's three bands are indistinguishable on MATH-500: +5.0, +5.3, +6.0, with
intervals that overlap almost entirely, and the *easy* band nominally highest.
On AIME they separate:

| rollouts | 0 | 2,560 | 5,120 | 7,680 | 10,240 |
|---|---|---|---|---|---|
| hard | 12.0 | 12.8 | 14.5 | 16.0 | **17.0** |
| medium | 12.0 | 14.3 | 14.7 | 14.0 | 14.3 |
| easy | 12.0 | 13.2 | 15.2 | 12.6 | 13.8 |

The hard band rises at every one of four checkpoints and is still rising when the
budget ends. A specific monotone ordering has probability 1/24 under no trend,
which together with the endpoint interval is what makes this more than one
significant cell out of nine. The medium and easy bands are noise.

Difficulty does not change how much the model gains on the distribution it
trained on. It changes whether the gain leaves that distribution — and MATH-500
alone cannot see the difference.

#### HMMT resolves nothing for Llama, as predicted

Baseline 1.8% pass@4 is 10 correct samples out of 1,488, over 4 of 93 problems.
Two more problems solved doubles the metric. All three bands wander between 1.5%
and 4.1% with no trend and no interval clear of zero. §1 said the floor would
make this uninformative for a 3B model; it did.

#### The matched control confirms it rather than explaining it away

The middle bands were confounded: group size ties to the band, so at a fixed 512
rollouts per step the middle band also covered 64 distinct problems per step
against the extremes' 16 — four times the data and, for Llama, 2.56 epochs
against 0.80. Re-running Llama's middle band at `GROUP_SIZE=32` matches it to the
extremes on every axis (verified: `training/epoch` stays 0 for all 20 steps, so
no problem is seen twice). All four runs below are now G=32, 16 problems/step,
320 problem instances, 20 steps:

| Llama band | MATH-500 pass@1 | AIME pass@4 |
|---|---|---|
| hard, 5–15% | 44.0 **+5.0** [2.1, 8.1] | 17.0 **+4.9** [0.3, 8.9] |
| medium, 40–60% (G=8, confounded) | 44.4 **+5.3** [2.5, 8.3] | 14.3 +2.2 [−2.5, 6.4] |
| **medium, 40–60% (G=32, matched)** | 45.4 **+6.3** [3.5, 9.3] | 12.1 +0.1 [−3.8, 3.8] |
| easy, 85–95% | 45.1 **+6.0** [3.1, 9.0] | 13.8 +1.8 [−2.2, 5.7] |

Removing the confound does not rescue the middle band on AIME — it makes it
worse, from +2.2 to +0.1, while MATH-500 improves from +5.3 to +6.3. Its AIME
trajectory rises and comes back down (12.0 → 14.9 → 16.0 → 13.7 → 12.1), ending
where it started, against the hard band's strictly monotone 12.0 → 17.0.

So for Llama, with every setting matched and only difficulty varying: **MATH-500
gains are the same across bands (+5.0 / +6.3 / +6.0) and the AIME gain belongs to
the hard band alone.**

The non-thinking control moves the other way, which is worth stating plainly
rather than picking the reading that suits:

| Qwen3-4B non-thinking, middle band | MATH-500 | AIME pass@4 | HMMT pass@4 |
|---|---|---|---|
| G=8, confounded | 87.3 **+4.0** | 40.0 +4.0 [−0.9, 8.9] | 23.3 +2.7 [−1.3, 7.2] |
| **G=32, matched** | 88.4 **+5.1** [2.9, 7.1] | 41.6 **+5.5** [0.3, 11.2] | 24.7 +4.2 [−0.2, 8.8] |

The thinking control, the third and last, leaves that configuration exactly where
it was: MATH-500 +0.5 [−0.5, 1.6], AIME −0.2 [−3.9, 2.7], HMMT +0.5 [−3.1, 4.1].
Matching the group size does not make a model with no usable gradient trainable.

Matching *helped* the non-thinking middle band on all three benchmarks, *hurt*
Llama's on AIME, and did nothing for thinking. The three controls therefore do
not identify a consistent direction for the confound, and the honest conclusion
is that the G=8 / G=32 differences are themselves inside the noise these
five-point curves can resolve.
What survives is that the controls do not overturn anything: Llama's hard band
still owns the only AIME gain that clears zero, and non-thinking still transfers
from every band.

**The two models disagree, and that is a result rather than a defect.** For
non-thinking, matched, all three bands transfer to AIME and HMMT at similar
magnitude (+4.3 / +5.5 / +4.8 on AIME) and the middle band is the strongest cell
overall. For Llama only the hard band transfers at all. The configurations differ
in almost every way that could matter — Llama sits at 39.1% on MATH-500 against
83.4%, its AIME baseline of 12.0% is close to the floor where non-thinking's 36%
is not, and a band labelled "hard" means 5–15% pass@1 for one and 12–25% for the
other. Nothing here separates those explanations; a difficulty-band prescription
that transfers across models is not supported by this data.

### The rollout length cap changes what the reward measures

The first thinking run was capped at 12,288 response tokens while evaluation
allowed 30,000. That is not a tuning difference. A truncated rollout carries no
`\boxed{}`, so the reward function scores it 0 regardless of whether the model
was on its way to a correct answer — the cap becomes part of the objective.

Measured over that run's 20 steps:

| | cap 12,288 | cap 32,768 (step 1) |
|---|---:|---:|
| rollouts hitting the cap | 51.9% mean, 31.4% → 60.2% across steps | 0.0% |
| mean response length | 8,658 → 10,596 | 11,197 |
| training reward (`critic/score/mean`) | 13.5% mean | 18.0% |
| the band's measured pass@1 | 18.6% | 18.6% |
| seconds per step | 389 | 672 |
| peak memory allocated | 33.5 GB | 43.9 GB |

Two things are visible. The truncation rate *rises* during training — GRPO
lengthens responses, so an initially tolerable cap tightens under its own
optimisation. And at 32,768 the training reward lands on the band's independently
measured pass@1, where at 12,288 it sat 5 points below it; the gap was the cap,
not the model. All three thinking bands were re-run at 32,768. The 12,288 results
are kept under `outputs/cap12k_ablation/` as the paired comparison.

Llama (4,096) and non-thinking (8,192) are unchanged — neither generates near its
cap, and their truncation rates in §1 are 3.2% and 1.5%.

At 32,768 a fixed micro-batch *count* no longer expresses the memory budget: two
33,792-token sequences in one backward pass is ~2.5× what fit at 12k, while the
same count wastes the card on a sequence that comes back at 1,700 tokens.
Thinking therefore batches by token budget, which has the identical worst case —
one maximal sequence — and packs the short ones densely.

### verl resumes silently, which invalidates a re-run

`trainer.resume_mode` defaults to `auto`: verl scans the checkpoint directory and
continues from whatever it finds, logging one line about it. Re-running a band
after an aborted attempt therefore does not re-run it — it resumes the aborted
attempt at an unknown step, and the resulting cell is not comparable with the
other eight. Every run now sets `resume_mode=disable` and wipes any stale
checkpoint directory first. Nine clean runs from the released weights is what the
shared x-axis assumes.

Nine summaries from one band were produced this way before the default was
noticed, from checkpoints whose training history could not be established; they
were discarded rather than plotted.

### What the 32k cap cost, and one cell it broke

Raising the thinking cap fixed the reward but moved the cost elsewhere. Two
failures followed, both worth recording because they shape how §7's thinking row
should be read.

**The low thinking band has no 10,240-rollout endpoint.** It ran out of GPU
memory at step 19 of 20, after five hours, and its step-20 checkpoint does not
exist. At a 34,816-token micro-batch budget the logits tensor alone is 10.6 GB
(tokens x 151,936 vocab x 2 bytes), the temperature division makes a second copy,
and vLLM still held 35.9 GB of the card: 11.63 GB wanted against 11.03 GB free.
The curve therefore has four points (0 / 2,560 / 5,120 / 7,680) where every other
cell has five, and its last point is NOT comparable with the others' endpoints.
Subsequent runs use a tighter token budget and give vLLM 0.45 rather than 0.6.

The failure was unrecoverable rather than a five-step top-up, because the
background job that reclaims redundant FSDP shards had already deleted the only
thing that could have resumed from step 15. Stripping them is correct while a run
is healthy and removes the recovery path when one is not; with `resume_mode`
disabled that trade is deliberate, but it is a trade.

**Six thinking evaluations were killed and recorded as failures** by a one-hour
per-job timeout. One evaluation job runs on one GPU; MATH-500 alone is 2,000
generations at roughly 39 gen/min/GPU for this configuration, and a trained
checkpoint generates longer than the base model it started from, so the cap was
simply below the work. Thinking now gets four hours per job. Llama and
non-thinking never came close to the old limit.

**Length grows with difficulty, so the cap is not neutral across bands.** Mean
response length over training: 15.7k tokens for the low band against 10.9k for
the high band, with truncation at 6% and 1.5% respectively. The harder the band,
the longer the model reasons and the more often it hits the ceiling — so any
response cap bites hardest on exactly the band whose behaviour the experiment is
most trying to measure. At 12,288 that bite was 52%; at 32,768 it is 6%, and the
low band's step-19 OOM is the same phenomenon arriving as a memory failure
instead of a truncation.
