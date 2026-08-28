# Experiment log

What has been run, what it measured, and what the numbers mean. The README covers
the repository and how to set it up; this file is the experimental record.

Everything below was produced on three 8×H100 nodes (`lumen-1`, `lumen-2`,
`lumen-3`) under one shared environment: vLLM 0.27.1, torch 2.13.0+cu130,
transformers 5.10.4, math-verify 0.9.0 with the `antlr4_9_3` extra, verl 0.9.0.
Decoding uses each vendor's recommended preset at seed 1234.

---

## 1. Baselines

Three model configurations on two held-out benchmarks.

| Configuration | MATH-500 pass@1 | AIME 2020–2024 pass@4 | AIME pass@1 |
|---|---:|---:|---:|
| Llama-3.2-3B-Instruct | 38.8% ± 1.7 | 15.3% ± 2.6 | 6.3% |
| Qwen3-4B non-thinking | 83.3% ± 1.4 | 36.0% ± 3.6 | 21.3% |
| Qwen3-4B thinking | 95.5% ± 0.8 | 80.6% ± 3.0 | 67.3% |

MATH-500: 4 samples per problem, pass@1 as the mean.
AIME: 8 samples, pass@4 via the unbiased estimator (Chen et al., 2021).

**Two checks against the reference work.** Llama-3.2-3B lands on 38.8% where the
Pedagogical RL blog states 38%, which says the grading path is calibrated rather
than silently depressing scores. On AIME the blog reports 22.5% pass@4 for its
method and describes it as more than a 40% relative gain; the measured baseline
of 15.3% gives 22.5 / 15.3 = +47%, consistent with that.

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
silently is indistinguishable from a weak model. `src/calibrate_grading.py`
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

- **HMMT.** Configured (`hmmt` over all 93, plus one task per competition) and
  grading-calibrated, but not executed.
- **Teacher-side training.** No GRPO run yet. The environment is in place —
  verl 0.9.0 with flash-attn built against CUDA 13.2 on all three nodes.
- **A post-cutoff contamination test.** The discriminating experiment described
  in §3. `MathArena/aime_2026` is the cheapest starting point: 30 problems, all
  integer answers, zero integration cost against the existing AIME task.
- **Benchmark generations.** Would require re-running the six baseline jobs with
  `--save-generations`.

### A statistical limit worth stating before HMMT runs

The standard error is set by the number of problems, and no amount of extra
sampling moves it. At 30 problems it is ±8.9 points around a 50% score, so two
single competitions can only be distinguished if they differ by roughly 30 points
— larger than the 20.7-point contamination effect measured on Llama. Any
date-based comparison has to pool competitions to have the power to see anything.
