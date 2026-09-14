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
- **A second seed for anything.** Every cell in §7 and §8 is n = 1. The AIME
  variance note in §1 shows the harness alone moves a pass@4 by ±3 points, which
  is the size of several effects reported here.
- **Verification of the teacher's answers.** §8's reward target is whatever the
  teacher wrote, unchecked. The key-error rate is unmeasured.
- **An overlap check between the teacher's problems and the benchmarks.** §8
  argues from three read curricula that the problems are new; it does not measure
  it.

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

---

## 8. The frontier model as teacher

Three runs, one per student configuration, launched 2026-09-02. Claude
(`claude-opus-4-8`, driven through `claude -p`) sits inside the GRPO loop and
writes the training data. Everything §7 held fixed is still fixed: the student's
released weights, G = 32, 16 problems per step, 512 rollouts per step, 20 steps,
the same sampling parameters, the same eight GPUs, the same grader. **A teacher
run and a §7 run differ in the problems and in nothing else**, which is what
makes the two comparable at all.

The teacher's only channel to the student is a file. Each turn it writes a
`decision.json` naming `evaluation` or `train` and a `data.jsonl` of
`{id, problem, answer}`. An evaluation grades the current checkpoint on problems
the teacher just wrote and returns per-problem pass rates and the full
generations; the step stays open and the teacher may look again. A train consumes
exactly 16 problems, runs one GRPO update, and closes the step. No reasoning
trace, no rationale and no logits reach the student — **whatever the student
gains cannot be an imitation of the teacher's reasoning.** The `answer` field
becomes the reward target directly, with no verification.

The teacher also sees three reference sets after every update: 100 of MATH-500,
30 of the 150 AIME problems, 19 of the 93 HMMT. Those are the problems it is
allowed to know about, so every score below is reported on their **complements**.
It turned out to know about more than those (see "What the teacher read off
disk" below), so the complements are further cleaned: any benchmark problem that
**any** arm of a student trained on — a §7 band or the teacher — is removed from
every arm of that student. That leaves 400 / 120 / 74 held-out problems for Llama,
399 / 115 / 74 for non-thinking and 397 / 104 / 74 for thinking.
`tools/holdout_split.py` and `tools/paired_stats.py --teacher --exclude
outputs/analysis/benchmark_contamination.json` apply the same ids to §7's runs,
so the two are read on one basis.

### Results: the teacher cannot be told apart from the fixed bands, except where it is worse

Clean held-out score at 10,240 rollouts, against the same student's untrained
score on the same problems. Rank is among the seven arms measured for that
student — the five §7 bands, the matched-group-size control, and the teacher.
Intervals are 95%, from the two-level bootstrap over problems and generations
(6,000 resamples, `outputs/analysis/paired_stats_teacher_clean.json`).

**Llama-3.2-3B** — held-out baseline 38.9 / 12.7 / 1.0

| | MATH-500 pass@1 | AIME pass@4 | HMMT pass@4 |
|---|---|---|---|
| teacher | 42.4 **+3.5 [0.2, 6.9]** (rank 6/7) | 12.3 −0.4 [−4.9, 4.1] (rank **7/7**) | 1.6 +0.6 [−1.6, 3.1] (rank 2/7) |
| best fixed band | 45.6 **+6.8 [3.6, 10.1]** (medium, G=32) | 17.0 +4.3 [−0.8, 8.7] (hard) | 2.3 +1.3 [−1.4, 4.4] (medium) |
| teacher − best band | **−3.2 [−6.2, −0.3]** | −4.7 [−8.9, 0.0] | −0.6 [−3.3, 1.8] |

**Qwen3-4B non-thinking** — held-out baseline 83.5 / 36.8 / 18.7

| | MATH-500 pass@1 | AIME pass@4 | HMMT pass@4 |
|---|---|---|---|
| teacher | 87.2 **+3.6 [1.6, 5.8]** (rank 2/7) | 40.5 +3.7 [−1.7, 9.1] (rank 1/7) | 22.3 +3.6 [−1.0, 8.3] (rank 2/7) |
| best fixed band | 88.2 **+4.7 [2.3, 7.1]** (medium, G=32) | 40.2 +3.5 [−1.7, 8.9] (hard, 2nd) | 23.5 +4.8 [−0.3, 9.8] (hard) |
| teacher − best band | −1.1 [−3.1, 1.1] | +0.3 [−5.1, 5.4] | −1.2 [−5.4, 3.0] |

**Qwen3-4B thinking** — held-out baseline 96.2 / 80.4 / 56.9

| | MATH-500 pass@1 | AIME pass@4 | HMMT pass@4 |
|---|---|---|---|
| teacher | 96.0 −0.2 [−1.4, 1.0] (rank 2/7, tied with `p = 1`) | 81.3 +0.9 [−2.4, 4.4] (rank 3/7) | 55.7 −1.2 [−5.0, 3.0] (rank 6/7) |
| best fixed band | 96.3 +0.1 [−1.0, 1.3] (medium, G=32) | 82.5 +2.1 [−2.3, 6.0] (medium) | 58.0 +1.1 [−2.7, 5.2] (easy) |
| teacher − best band | −0.3 [−1.6, 0.9] | −1.2 [−4.9, 3.4] | −2.3 [−6.2, 1.7] |

**Of the 54 teacher-minus-band comparisons, one resolves against a real band and
none in the teacher's favour.** The one is Llama MATH-500, where the teacher ends
3.2 points below medium at G = 32. The only intervals that clear zero on the
teacher's side are against the two degenerate bands on non-thinking MATH-500 —
`p = 0` by +2.3 [0.5, 4.3] and `p = 1` by +3.1 [0.9, 5.2] — which is beating the
controls, not the curricula. Llama AIME against the hard band, −4.7 [−8.9, 0.0],
stops exactly at zero. Everything else, including every AIME and HMMT cell on
both Qwen students, is inside its interval. Read as a whole: **at this budget the
teacher is indistinguishable from choosing a fixed band, and on Llama it is below
the good ones.**

The ranks agree with that and say no more. The teacher's single first place is
non-thinking AIME, 0.3 points ahead of the hard band against a ±4.2 standard error.
Before the contaminated problems were removed its first place was thinking AIME
instead; that one came from problems it trained on (below). On Llama it is sixth,
seventh and second, and its AIME result is the worst of the seven — one of three
arms that end below where they started, and the lowest of them.

Llama's teacher curve is the clearest statement of the problem, because it does
not fail so much as peak and give back:

| held-out, Llama | 0 | 2,560 | 5,120 | 7,680 | 10,240 |
|---|---|---|---|---|---|
| MATH-500 pass@1 | 38.9 | 41.4 | 42.9 | 42.4 | 42.4 |
| AIME pass@4 | 12.7 | 14.0 | 14.5 | 13.9 | 12.3 |

Both rise through 5,120 and then flatten or fall. Compare §7's hard band, which
rose at all four checkpoints and was still rising at the budget's end. Only the
non-thinking teacher run is monotone on its in-domain metric (83.5 → 85.2 → 85.3
→ 86.1 → 87.2), and it is also the run whose AIME gain arrives entirely in the
second half (36.8 → 36.1 → 34.9 → 38.5 → 40.5) — the same late-transfer shape §7
found in all three non-thinking bands.

### What the teacher actually chose, and why it probably explains the result

The system prompt describes the mechanism, names `critic/advantages/max` as the
signal for "whether any group carried gradient", and then says: *"No curriculum
strategy is prescribed."* The `p(1-p)` argument that §7 uses to set group size is
never stated. What the teacher did with that latitude is measurable — training
reward is the fraction of the step's 512 rollouts that were correct, so it is a
direct read of how hard the curriculum was **for the student that received it**:

| training reward | first step | last step | mean | steps at `advantages/max` = 5.48 |
|---|---:|---:|---:|---:|
| Llama | 67.6% | 79.1% | **64.3%** | 9 / 20 |
| non-thinking | 41.6% | 72.5% | **59.2%** | 3 / 20 |
| thinking | 42.0% | 67.8% | **63.6%** | 3 / 20 |

Two things follow. **The teacher settled around 60% and drifted upward**, ending
20–30 points above where it started in every run; the curricula got easier
relative to a student that was itself improving. And the strongest signal a group
of 32 can carry — one success, `advantages/max` = 5.48 — appears on 3 of 20 steps
for both Qwen runs, where §7 records it on nearly every step of every `p = 0`
band.

A 60% curriculum is not absurd; it is near §7's medium band, which is the best
Llama cell on MATH-500. But it is chosen without the argument that would justify
it, it is not held there, and on Llama the band that transferred to AIME was the
hard one at 5–15%. **The teacher was not told that difficulty is the lever, and
the evidence here is that it did not infer it.** That is a statement about this
prompt and this teacher over 20 steps, not about what a frontier model could do
if told.

### What the teacher read off disk

Across the three runs the teacher wrote **5,152 problems**: 1,661 for Llama,
1,455 for non-thinking, 2,036 for thinking, of which 320 per run were training
curricula and the rest were its own evaluations. An earlier version of this
section called them "written, not retrieved" on the strength of reading three
files. `tools/teacher_overlap.py` now measures it — normalised exact match, shared
13-grams and exhaustive character similarity against MATH-500, AIME, HMMT and the
12k pool, with every near-match read and judged (`outputs/analysis/teacher_overlap.json`).
**Most of the curricula are written. Two runs also copied problems from files the
teacher could read, and one of those copies reached held-out AIME.**

The cause is the same in both: the prompt says the held-out sets are ones "you
never see", and also that anything which is not the student is the teacher's to
read. The benchmark files sit in `data/benchmark/` of the checkout the teacher
runs from, beside the manifests that say which fifth is the reference. Nothing
but the prompt stood between them.

- **Thinking, step 6 — 14 held-out AIME problems trained on.** In
  `step_6/eval_0/teacher.log` the teacher `cd`s into `data/`, reads
  `aime_ref30.manifest.json`, sets the 30 reference ids aside as the ones it may
  not use, and loads the other 120 from `aime_20*.jsonl` as "clean". Thirty of
  them became an evaluation probe with their benchmark ids intact; fourteen went
  into the step's 16-problem training curriculum, verbatim, with the benchmark
  answer (2020-11, 2020-21, 2021-11, 2021-19, 2022-05, 2022-15, 2022-19, 2022-23,
  2023-04, 2023-13, 2023-21, 2024-17, 2024-21, 2024-26). Every thinking checkpoint
  from step 9 on was trained on them. On those fourteen the teacher's student goes
  83.6 → 91.2 pass@4 — but the easy band, which never saw them, reaches 90.7, so at
  n = 14 memorisation cannot be separated from ordinary improvement. What can be
  measured is the effect on the headline: on the uncleaned 120, thinking AIME read
  82.8, +1.7, rank 1/7; with these problems removed it is the 81.3, +0.9, rank 3/7
  in the table above.
- **Non-thinking, steps 0–3 — the curriculum is the §7 pool.** All 64 training
  problems in the first four steps, and 120 probes, are MATH pool items copied from
  `data/further_improve/qwen3-4b-nothink/`, ids such as `mathtrain-03303` still
  attached. For a fifth of the run the teacher was picking from the band files —
  which also carry the student's measured pass rate on every problem — rather than
  writing. This is not an evaluation leak, but it is not what a "teacher-written
  curriculum" means either.
- **Llama — reference-fifth paraphrases.** From step 12 two training problems
  restate MATH-500 reference items (0215, "x² − x − 1 = 0, find x³ − 2x + 1"; and
  0186, "for how many n > 1 is 2²⁴ a perfect nth power"), and four more reference
  items appear verbatim in its step-12 probes. The held-out numbers are unaffected;
  the seen column below is not.
- **Nothing else reached training.** No HMMT problem appears anywhere. The one
  held-out MATH-500 item in a teacher curriculum is a judged paraphrase on
  non-thinking (0118). No logged tool call touched the network.

The rest reads as written. Step 0 of the Llama run opens `"The sides of a triangle
are 6, 25, and 29. Find its area."` (answer 60); step 19 is a number-theory set —
`"Find the remainder when 2^2024 is divided by 7."`, `"If x + 1/x = 6, find x^3 +
1/x^3."` Beyond the cases above, the near-matches are same-form exercises —
reference items with the numbers changed, classics such as a telescoping log
product in new wording — and none of the borderline ones reached a training step.

**The bands are not entirely clean either.** The 12k pool's `orig_train` split
contains nine AIME 2020 problems, eight verbatim, and the §7 band files for both
Qwen students picked some of them up — non-thinking `p = 0` holds five, thinking's
100-problem hard band three. Llama's pool is `orig_test` only and none of its
bands is affected, including the hard band whose AIME transfer §7 reports; the
non-thinking hard band is clean too. These ids are in the exclusion set above, which is
why a band's clean number can differ from §7's. §7's own figures are on the full
benchmarks and have not been re-scored.

### The answer keys are right

The teacher's `answer` becomes the reward target unverified, so a wrong key would
train the student toward a wrong answer. `tools/teacher_keycheck.py` re-solved all
960 training problems with `claude-sonnet-5` — a different model, no tools, key
not shown — graded the solve against the key with the training grader, sent every
disagreement to a blind adjudication, and gave the harder cases a Python check run
in a network-less sandbox (`outputs/analysis/teacher_keycheck/`, $31.56 recorded).

| | problems | agree | solver wrong | key judged wrong | wrong-key rate |
|---|---:|---:|---:|---:|---|
| Llama | 320 | 320 | 0 | 0 | 0% [0, 1.2] |
| non-thinking | 320 | 313 | 6 | 1 | 0.3% [0.06, 1.8] |
| thinking | 320 | 312 | 8 | 0 | 0% [0, 1.2] |

"Agree" includes two answers the grader missed as equivalent and the five
enumeration problems the solver timed out on, whose keys the Python check
confirmed. "Solver wrong" is the adjudication's verdict. The Python check covered
fourteen disagreements and upheld the key on all but the ambiguous one below,
including one where it overturned an adjudication that had sided with the solver on
a counting problem. The one
key judged wrong is ambiguous rather than wrong: "44 candies, every kid at least 3,
the largest number of kids given an equal number" is 14 if leftovers are allowed
and 11 if not, and the key is the original MATH answer, from the pool rows above.
**Across 960 problems the rate is 0.1% [0.02, 0.6]; wrong keys do not explain
anything in this section.**

The check did surface two grader false negatives, which bear on training reward in
every arm, not only here: `\tan 75° = 2 + \sqrt{3}` fails against `2+\sqrt{3}`
because of the Unicode degree sign (`^\circ` passes), and `x \geq 8` fails against
`[8,\infty)`. How often a student wrote either has not been counted.

### No sign that the visible reference sets were targeted

The teacher can read its 100 / 30 / 19 reference problems, so a curriculum aimed
at them would move the seen scores and not the held-out ones. MATH-500 at 10,240
rollouts:

| | held-out (clean) | seen (100) |
|---|---|---|
| Llama | 38.9 → 42.4 | 39.8 → 41.5 |
| non-thinking | 83.5 → 87.2 | 82.8 → 86.3 |
| thinking | 96.2 → 96.0 | 94.8 → 95.5 |

The two move together, and if anything the held-out gain is the larger one. The
seen column is the full reference hundred, including the two items Llama trained
on paraphrases of; removing them moves Llama's seen figures to 39.7 → 41.8. This is
the absence of a systematic failure, not evidence of good faith, and the previous
section shows that where the teacher did use benchmark files it went for the
problems it believed were **not** the reference.

### Cost

| | teacher wall clock | output tokens | teacher cost | GPU wall clock |
|---|---:|---:|---:|---:|
| Llama | 2.7 h | 702,426 | $50.45 | 1.7 h |
| non-thinking | 2.3 h | 593,723 | $44.31 | 2.2 h |
| thinking | 4.4 h | 1,171,642 | $78.65 | 22.0 h |
| | | | **$173.42** | |

**The input-token column is 834–1,370 per run, and that is not the size of what
the teacher was given.** Each turn is handed a task message built by
`tools.render_context`, and the run directory keeps every one of them as
`task_context.txt`: 1,985 characters at step 0, 74,237 at step 19, of which 79%
is the verl metrics dictionary printed in full for every closed step. So the
figure above is what the CLI reports, not what the model read, and it should not
be used to argue anything about how the teacher gets its context. The runs' cost
is dominated by output tokens either way.

**The comparison this does not license is a compute-matched one.** A §7 band
costs the GPU time and no API spend; the teacher adds $44–79 and, on the thinking
run, 4.4 hours of serial turn latency. Whether $173 buys more than spending it on
rollouts is not answerable from these runs.

### What this section does not establish

**It does not establish that the teacher is worse than, or equal to, a fixed band
in general.** The intervals above resolve one difference out of 54, and an
interval that crosses zero is an absence of evidence at this n, not evidence of
equality — on AIME and HMMT even the best bands' gains over the base model do not
clear zero. The removal of contaminated problems changes no interval's verdict and
two ranks; a stricter exclusion that also drops shared-setup items
(`--exclude-level strict`) changes neither.

Each cell is also n = 1, at 20 steps, with one teacher and one prompt. §7 already
says this about the bands; it is more binding here, because a teacher run has a
second source of variance the bands do not — the teacher's own sampling. Two runs
of this identical setup could differ in what gets taught, and nothing here bounds
that.

### Operational record

Both Llama and non-thinking halted once and were resumed; thinking ran 38 hours
without a failure.

| | step | error | resumed from |
|---|---|---|---|
| Llama | 4.0 (eval) | `evaluate.py produced no summary.json` | step 4, 5.7 min later |
| non-thinking | 2.1 (train) | `train failed rc=1 timed_out=False hf_exists=False` | step 2, 3.6 min later |

Both behaved as the loop is specified to: it halts on the first failure rather
than retrying or best-effort parsing, and `--resume` rebuilds the position from
the artifacts. Neither run fabricated a step, and the resumed runs re-materialised
their `pipeline/` snapshot, leaving the superseded copy under its timestamp. The
underlying cause of either failure was not diagnosed further; both look like
single-job flakes rather than anything about the curriculum.

The thinking run's 22.0 hours of GPU wall clock against 4.4 hours of teacher time
is the same 32,768-token cap §7 describes, now applied to 73 evaluations the
teacher chose to run — 3.7 per step against 2.0 for the other two students. The
teacher's freedom to evaluate is the dominant cost term for that configuration.
