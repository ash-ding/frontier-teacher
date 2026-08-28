# Frontier Teacher

**Can a frontier model act as a teacher to train and improve a smaller model?**

This repository is the experimental infrastructure for that question, run on math
reasoning tasks. The premise is that a large, strong model has knowledge a small
model cannot reach on its own, and that the useful form of that knowledge is not
just "the right answer" but *trajectories the student can actually learn from*.
Whether a frontier teacher can supply those, and whether the student measurably
improves as a result, is what we are here to measure.

Math is the testbed because it gives cheap, exact, automatic verification: an
answer is right or wrong with no judge model in the loop, so any improvement is
attributable to the teaching signal rather than to grader noise.

## Where the work stands

Teacher–student training needs two things established before it can mean anything:
an honest baseline for the student, and a training pool whose difficulty is
actually known. Both are done and shipped here.

- **Baselines** for three model configurations on MATH-500 and AIME 2020–2024.
- **Difficulty profiles** over the 11,996-problem MATH training pool: every
  problem's empirical pass@1 measured by direct sampling, per-sample verdicts kept.
- **Curated training subsets** sliced by measured difficulty — the pool from which
  teacher–student experiments draw.

Teacher-side training is not in this repository yet.

### Baselines

| Configuration | MATH-500 pass@1 | AIME 2020–2024 pass@4 |
|---|---:|---:|
| Llama-3.2-3B-Instruct | 38.8% ± 1.7 | 15.3% ± 2.6 |
| Qwen3-4B (non-thinking) | 83.3% ± 1.4 | 36.0% ± 3.6 |
| Qwen3-4B (thinking) | 95.5% ± 0.8 | 80.6% ± 3.0 |

MATH-500 uses 4 samples per problem and reports mean pass@1; AIME uses 8 samples
and the unbiased pass@4 estimator (Chen et al., 2021).

**Llama-3.2-3B is the student.** Qwen3-4B has no room to move on MATH — at 95.5%
there are 4.5 points of headroom, so no teaching intervention could be resolved
above noise. The two Qwen configurations are kept as reference points and as
candidate teachers, not as students.

### Two findings that shape the experimental design

**Llama-3.2-3B was trained on the original MATH train split.** Profiling the full
12k pool and splitting by provenance:

| Pool | Problems | Mean pass@1 |
|---|---:|---:|
| Original train split | 7,498 | 61.0% |
| Original test split (moved into the 12k by the PRM800K re-split) | 4,498 | 40.3% |
| MATH-500 (held out, control) | 500 | 38.8% |

The test-derived problems match the held-out control; the train-derived ones sit
20.7 points above it, and the gap widens monotonically with difficulty (+10.5
points at Level 1, +27.5 at Level 4). That is the signature of memorisation, not
of a difficulty artefact. **All Llama subsets are drawn only from the 4,498 clean
problems.**

**The same test finds no asymmetry for Qwen3-4B** (82.5% / 84.2% / 83.3%), so its
subsets use the full pool. Read that result narrowly: a train-vs-test comparison
is a *differential* test and is blind to uniform exposure. If every MATH problem
were in a model's training data, all splits would lift together and the difference
would vanish, producing a reading identical to a genuinely clean model. The Llama
result is positive evidence; the Qwen result is only the absence of differential
evidence. Separating the hypotheses needs a benchmark released after the training
cutoff, or a verbatim-continuation probe. Neither has been run.

## Setup

### Requirements

- Linux, NVIDIA GPU with bf16 support. Everything here was run on 8× H100 80GB
  per node; a single 80GB card is enough for one shard.
- NVIDIA driver new enough for CUDA 12.x/13.x (ours: 595.71.05 / CUDA 13.2).
- `conda` (Miniforge). `scripts/setup_env.sh` installs it if absent.
- ~15 GB disk for the environment, plus ~7 GB per model checkpoint.
- Network access to the HuggingFace Hub.

### Install

```bash
git clone <this-repo> frontier-teacher
cd frontier-teacher
./scripts/setup_env.sh          # idempotent: miniforge -> conda env -> vLLM
conda activate frontier-teacher
```

`setup_env.sh` is safe to re-run on a node that is already provisioned; it skips
each step that is already satisfied.

### Dependencies

Installed by `setup_env.sh`, pinned where the pin matters:

| Package | Version | Why |
|---|---|---|
| `python` | 3.12 | |
| `vllm` | **0.27.1** (pinned) | Batch inference engine. Pulls a matching `torch` (2.13.0+cu130). |
| `math-verify[antlr4_13_2]` | 0.9.0 | Symbolic answer equivalence for MATH. The `antlr4_13_2` extra is required — without it LaTeX parsing silently degrades. |
| `datasets` | 5.x | Dataset loading. |
| `transformers` | 5.x | Tokenizer and chat templates. |
| `accelerate`, `pandas`, `tabulate`, `pyyaml` | — | Support. |

Only `vllm` is version-pinned. It selects the CUDA-matched `torch` build, and
mixing an independently chosen `torch` with vLLM is the usual way this environment
breaks.

Verify:

```bash
python -c "import vllm, torch; print(vllm.__version__, torch.__version__)"
# 0.27.1 2.13.0+cu130
```

### Model access

`Qwen/Qwen3-4B` is open. `meta-llama/Llama-3.2-3B-Instruct` is gated and needs an
approved HF token; the configs default to `unsloth/Llama-3.2-3B-Instruct`, an
ungated mirror of the same weights. To use the official repository instead, set a
token and change `model:` in `configs/llama32-3b.yaml`:

```bash
export HF_TOKEN=hf_...          # or: huggingface-cli login
```

## Quick start

```bash
conda activate frontier-teacher

python data/build_datasets.py     # MATH-500 (500) + AIME 2020-2024 (150)
python data/build_math_train.py   # MATH train pool (11,996)

./scripts/run_all.sh              # baselines: 6 jobs, one GPU each
python src/report.py              # results table
```

Difficulty-profile a model over the training pool (8-way sharded, ~26 min for
Llama at 32 samples; Qwen thinking takes hours):

```bash
./scripts/run_profile.sh llama32-3b
python src/merge_shards.py --config configs/llama32-3b.yaml --task mathtrain
python src/profile_report.py --records outputs/llama32-3b__mathtrain.records.jsonl
python src/make_subsets.py --model llama32-3b
```

Both builders assert on their output — 30 AIME problems per year, every AIME
answer an integer 0–999, and zero overlap between the training pool and MATH-500.
A build that would contaminate the eval set fails instead of producing files.

## Layout

```
configs/          One YAML per model configuration: weights, decoding preset,
                  and per-task sample counts and token budgets.
data/
  build_*.py      Dataset builders (assertions included).
  subsets/        Curated training subsets, sliced by measured difficulty.
src/
  evaluate.py     vLLM generation + scoring for one (config, task) pair.
                  Supports --shard/--num-shards for multi-GPU splitting.
  grading.py      Answer extraction and equivalence checking.
  summarize.py    pass@k estimators, shared by evaluate and merge.
  merge_shards.py Recombine sharded runs into one summary + records file.
  report.py       Benchmark results table.
  profile_report.py  pass@1 distribution and band counts over a profiled pool.
  make_subsets.py    Seeded sampling of the difficulty-banded subsets.
scripts/
  setup_env.sh    Environment bootstrap.
  run_all.sh      All baseline jobs, one GPU each.
  run_profile.sh  One model over the training pool, 8-way sharded.
  sync_subsets.sh Push subsets to every node and verify by hash.
results/
  benchmarks/     MATH-500 and AIME summaries + per-problem records.
  profiles/       Training-pool profile summaries.
```

## Datasets

| Set | Size | Source |
|---|---:|---|
| MATH-500 | 500 | `HuggingFaceH4/MATH-500` — the problems Lightman et al. held out of the original MATH *test* split for PRM800K. |
| AIME 2020–2024 | 150 | Assembled: `di-zhang-fdu/AIME_1983_2024` for 2020–21, `AI-MO/aimo-validation-aime` for 2022–24. |
| MATH train pool | 11,996 | `nlile/hendrycks-MATH-benchmark` train split, deduplicated. 7,498 from the original train split + 4,498 from the original test split. |

AIME has to be assembled because no single public set covers all five years.
`di-zhang`'s 2023 and 2024 are incomplete (29 and 14 problems) and are
deliberately unused. The full Hendrycks MATH set is 12,500 problems — 7,500 train
and 5,000 test; the PRM800K re-split moves 4,500 test problems into training and
keeps 500 as MATH-500, which is why the training pool and the eval set are
disjoint by construction.

## Curated subsets

`data/subsets/<model>/`, one JSONL per difficulty band. Filenames encode the
model, the pass@1 band, and the sample count.

**Llama-3.2-3B** — drawn from the 4,498 clean problems, profiled at 32 samples
(pass@1 resolution 3.1%):

| File | Band | Pool | Sampled |
|---|---|---:|---:|
| `llama32-3b__pass1_eq_0__n800.jsonl` | p = 0 | 814 | 800 |
| `llama32-3b__pass1_05-15pct__n400.jsonl` | 5% ≤ p < 15% | 487 | 400 |
| `llama32-3b__pass1_40-60pct__n500.jsonl` | 40% ≤ p ≤ 60% | 570 | 500 |
| `llama32-3b__pass1_85-95pct__n400.jsonl` | 85% < p < 95% | 437 | 400 |
| `llama32-3b__pass1_eq_1__n100.jsonl` | p = 1 | 131 | 100 |

**Qwen3-4B non-thinking** — full pool, 8 samples (resolution 12.5%, so the bands
are exact values):

| File | Band | Pool | Sampled |
|---|---|---:|---:|
| `qwen3-4b-nothink__pass1_eq_0__n700.jsonl` | p = 0 | 779 | 700 |
| `qwen3-4b-nothink__pass1_eq_25pct__n200.jsonl` | p = 25% | 288 | 200 |
| `qwen3-4b-nothink__pass1_eq_50pct__n200.jsonl` | p = 50% | 294 | 200 |
| `qwen3-4b-nothink__pass1_eq_75pct__n500.jsonl` | p = 75% | 510 | 500 |
| `qwen3-4b-nothink__pass1_eq_1__n500.jsonl` | p = 1 | 8,199 | 500 |

Each record carries `id`, `problem`, `answer`, `level`, `subject`, `pass_at_1`,
`n_correct`, `n_samples`, `split_origin`, `band`, `profiled_model`. Every
directory has a `manifest.json` recording the decoding settings the pass@1 was
measured under, the pool policy and why, the sampling seed, and each subset's
level and provenance composition.

`pass_at_1` is a property of *a model under a specific decoding configuration*,
not of the problem. Read it together with the manifest's `profiled_with` block.

**Why these bands.** Group-normalised RL (GRPO and relatives) produces gradient
only when `0 < pass@1 < 1`: an all-correct or all-wrong group has zero advantage
and contributes nothing. The bands cover that axis end to end — the p = 0 and
p = 1 sets are deliberately included as the degenerate cases, useful for
teacher-side experiments precisely because plain RL cannot use them.

The `split_origin` field is preserved on the Qwen subsets so they can be filtered
to test-derived problems without re-sampling, should the contamination question
be settled later.

## Reproducibility

Generation runs at `temperature > 0` under a fixed seed (1234), per each vendor's
recommended decoding preset rather than one shared setting. Subset sampling is
uniform without replacement under seed 20260828, with a separate stream per band
so changing one band's size does not perturb another's draw; re-running
`make_subsets.py` reproduces byte-identical files.

Every run records `truncation_rate` and `no_answer_rate` alongside the score. If
either is high the number is a harness artefact rather than a model result. Two
known ones: Llama emits no `\boxed{}` on about 10% of samples (weak format
adherence at 3B, only ~3% of it truncation), and Qwen3-4B thinking truncates on
13.3% of AIME samples against the 30,000-token cap, so its true AIME score is
somewhat above the reported figure.

## What is not here

Raw generations (~880 MB per profiled model) and the per-problem profiling
records over the 12k pool (12–36 MB each) are not tracked. They live on the
compute nodes under `$HOME/data/frontier-teacher/generations/` and
`outputs/` respectively, and the records are rebuildable from the shards with
`src/merge_shards.py`.

## Related work

The difficulty-band methodology — selecting a training subset by the student's
measured pass@1 rather than by a difficulty label — follows *Pedagogical RL:
Teaching Models to Teach Themselves from Privileged Information* (Chakraborty,
Ziems et al., 2026), which reports that a hard subset at roughly 8% pass@1
concentrates the learning signal. Our 5%–15% Llama band has a measured mean
pass@1 of 8.7% and is the closest analogue in this repository.
