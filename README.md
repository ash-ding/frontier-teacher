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

### Benchmarks

| Benchmark | Problems | Samples / problem | Headline metric | Answer format | Status |
|---|---:|---:|---|---|---|
| MATH-500 | 500 | 4 | **pass@1** | mixed; symbolic compare | measured |
| AIME 2020–2024 | 150 | 8 | **pass@4** | integers 0–999; exact compare | measured |
| HMMT (3 competitions) | 93 | 16 | **pass@4** | 51/93 integers, 42 exact forms; symbolic compare | grading calibrated, **not yet run** |

HMMT is also configured per competition — `hmmt_feb_2025`, `hmmt_nov_2025`,
`hmmt_feb_2026` — so the three dates can be compared against a training cutoff.
Sixteen samples yield the whole pass@1 / @4 / @8 / @16 ladder from one run; the
headline is only which number a comparison should lead with.

**Why the metric differs by benchmark.** The point of choosing *k* is to keep the
score off both the floor and the ceiling, where it stops discriminating.

On MATH-500, pass@1 works because 500 problems give a ±1.7 point standard error
and Llama-3.2-3B scores 38.8% — mid-range, with room to move in both directions.

On AIME it does not. Llama's pass@1 there is 6.3%, about nine problems out of
150, and the standard error on that is ±2.0 — **32% relative noise**. A real
improvement from 6.3% to 8% would be invisible. pass@4 lifts the same runs to
15.3% ± 2.6, cutting relative noise to 17%. pass@k also answers the question this
project actually cares about: whether a correct trajectory exists anywhere in the
model's sampling distribution, since a problem the student never solves offers
reinforcement learning nothing to reinforce.

HMMT is harder than AIME with a third of the problems per competition, so pass@1
would sit further onto the floor still, and pass@4 is the headline for the same
reason.

**A limit worth stating plainly:** the standard error is set by the number of
problems, and no amount of extra sampling moves it. At 30 problems it is ±8.9
points around a 50% score. Two single competitions can therefore only be
distinguished if they differ by roughly 30 points — larger than the 20.7-point
contamination effect measured on Llama. Any date-based comparison has to pool
competitions to have the power to see anything.

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
./scripts/setup_env.sh          # idempotent: miniforge -> conda env -> requirements.txt
conda activate frontier-teacher
```

For the exact pinned set instead of a fresh resolution:

```bash
LOCK=1 ./scripts/setup_env.sh
```

Or without the script, if conda is already present:

```bash
conda env create -f environment.yml && conda activate frontier-teacher
```

`setup_env.sh` is safe to re-run on a node that is already provisioned; it skips
each step that is already satisfied.

### Dependencies

Three files, for three different needs:

| File | Use it when |
|---|---|
| `requirements.txt` | Normal install. Names the 8 direct dependencies at the versions this was verified on, and lets pip resolve the rest against the local CUDA. |
| `requirements.lock.txt` | Exact reproduction. The full 207-package resolved set, captured with `pip freeze` from a working node. |
| `environment.yml` | `conda env create -f environment.yml`. Takes python from conda and everything else from `requirements.txt`. |

One environment both evaluates and trains.

| Package | Version | Why |
|---|---|---|
| `python` | 3.12 | |
| `vllm` | **0.27.1** | Batch inference engine. Selects a CUDA-matched `torch` (2.13.0+cu130 here). |
| `verl` | **0.9.0** | GRPO training. Brings `ray`, `tensordict`, `peft`, `hydra-core`. |
| `math-verify[antlr4_9_3]` | 0.9.0 | Symbolic answer equivalence — MATH and HMMT answers are exact forms, not integers. |
| `transformers` | 5.10.4 | Tokenizers and chat templates. Held here by verl. |
| `datasets` | 5.0.1 | Dataset loading. |
| `accelerate`, `pandas`, `tabulate`, `PyYAML` | — | Support. |
| `flash-attn` | 2.8.3.post1 | Optional; see below. |

Two of these are load-bearing rather than cautious.

**`vllm` must be installed first and alone.** It selects the CUDA-matched `torch`
build; installing `torch` yourself, or letting a later resolution step move it,
is the usual way this environment breaks. `setup_env.sh` installs `vllm` on its
own before the rest for exactly this reason.

**`math-verify`'s extra must match the antlr4 runtime the environment ends up
with.** verl depends on `hydra-core`, which pins `antlr4-python3-runtime==4.9.*`,
so the correct extra here is `antlr4_9_3` — not `antlr4_13_2`, which is right
only for an evaluation-only environment. A mismatch does not raise: the LaTeX
parser degrades *silently*, failing to parse and depressing every score. A
grading path that fails quietly is worse than one that crashes, because the
result still looks like a number. `src/calibrate_grading.py` is what proves the
installed combination actually works.

### Evaluation and training share one environment

They were initially built separately, on the finding that
`math-verify[antlr4_13_2]` and verl's `hydra-core` pin antlr4 to mutually
exclusive versions. That reading was incomplete — math-verify publishes an
`antlr4_9_3` extra whose purpose is matching whatever antlr4 is already present,
so with the right extra there is no conflict.

Three independent checks confirmed the shared environment measures the same
thing the evaluation-only one did:

- **Chat templates are byte-identical.** Rendering the same prompt under
  transformers 5.16.1 and 5.10.4 gives the same md5 for all three model
  configurations, so the models are asked the same question.
- **Grading is unchanged.** `calibrate_grading.py` passes identity, specificity,
  thinking and all 28 equivalence cases at 100% on MATH-500, AIME and HMMT under
  antlr4 4.9.3.
- **A re-run reproduces the number.** Llama-3.2-3B on MATH-500 scored 39.1% in
  the shared environment against 38.8% before — a 0.35-point difference, well
  inside the ±1.7 standard error.

### flash-attn

Optional. verl runs without it, using PyTorch attention, which is slower.

There is no prebuilt wheel for torch 2.13+cu130, so it builds from source, and
the build fails out of the box: `CUDA_HOME` on these nodes points at a 12.9
toolkit while torch was built with 13.0, and PyTorch's extension builder rejects
a **major** version mismatch (it only warns on a minor one). CUDA 13.2 is present
at `/usr/local/cuda`, which shares torch's major version:

```bash
CUDA_HOME=/usr/local/cuda TORCH_CUDA_ARCH_LIST="9.0" MAX_JOBS=8 \
  pip install flash-attn==2.8.3.post1 --no-build-isolation
```

`TORCH_CUDA_ARCH_LIST="9.0"` builds for H100 only; the default builds every
architecture and produces a ~930 MB extension. Keep `MAX_JOBS` modest — each
nvcc job takes 2–4 GB.

Verify an install:

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

The datasets are committed, so a fresh clone can evaluate immediately:

```bash
conda activate frontier-teacher
./scripts/run_all.sh              # baselines: 6 jobs, one GPU each
python src/report.py              # results table
```

The builders are only needed to verify the data or to refresh it from upstream.
Re-running them reproduces the committed files byte for byte:

```bash
python data/build_datasets.py     # MATH-500 (500) + AIME, one file per year (150)
python data/build_hmmt.py         # HMMT, one file per competition (93)
python data/build_math_train.py   # MATH train pool (11,996)
```

Difficulty-profile a model over the training pool (8-way sharded, ~26 min for
Llama at 32 samples; Qwen thinking takes hours):

```bash
./scripts/run_profile.sh llama32-3b
python src/merge_shards.py --config configs/llama32-3b.yaml --task mathtrain
python src/profile_report.py --records outputs/llama32-3b__mathtrain.records.jsonl
python src/make_subsets.py --model llama32-3b
```

Every builder asserts on its output — 30 AIME problems per year with integer
answers 0–999, the HMMT per-competition counts and mutual disjointness, the
7,498 / 4,498 training-pool partition, and zero overlap between the training
pool and MATH-500. A build that would contaminate the eval set fails instead of
producing files.

## Layout

```
docs/experiment.md      The experimental record: what was run, what it
                        measured, and what the numbers mean.
requirements.txt        Direct dependencies.
requirements.lock.txt   Full resolved set, for exact reproduction.
environment.yml         conda env spec.
configs/          One YAML per model configuration: weights, decoding preset,
                  and per-task sample counts and token budgets.
data/
  build_*.py      Dataset builders (assertions included).
  benchmark/      Held-out evaluation sets: MATH-500, AIME, HMMT.
  training_set/   The MATH training pool, split by original provenance.
  further_improve/  Curated subsets sliced by measured difficulty - the pool
                  teacher-student experiments draw from.
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
outputs/          Where evaluate.py writes and where results live - one
                  directory, no second copy to drift.
  *.summary.json    Benchmark results (MATH-500, AIME, HMMT), tracked.
  *.records.jsonl   Per-problem, per-sample verdicts, tracked.
  math_profiling/   Training-pool profiling runs. The mathtrain task routes
                  here via out_subdir, so it is declared once rather than
                  passed on every command line. Summaries tracked; the 12-36 MB
                  records and shard intermediates are not (see .gitignore).
```

## Datasets

`data/` is split by what the data is *for*, since the three roles must never be
confused with one another:

| Directory | Role |
|---|---|
| `benchmark/` | Held-out evaluation. Never trained on. MATH-500, AIME, HMMT. |
| `training_set/` | The pool experiments train on, split by original provenance. |
| `further_improve/` | Curated subsets drawn from that pool by measured difficulty. |

**The data files are tracked in this repository, not fetched at run time.** What a
score means depends on exactly which problems were scored, so the problem set is
pinned rather than left to an upstream that may revise it. The builders in `data/`
remain the reproducibility check: re-running them reproduces every file byte for
byte, verified.

One file per competition, so each set carries its own date — which is what makes
these usable for reasoning about training cutoffs, not just about difficulty.

| File(s) | Set | Problems | Source |
|---|---|---:|---|
| `benchmark/math500.jsonl` | MATH-500 | 500 | `HuggingFaceH4/MATH-500` — the problems Lightman et al. held out of the original MATH *test* split for PRM800K. |
| `benchmark/aime_2020.jsonl` … `aime_2024.jsonl` | AIME, one file per year | 30 each, 150 total | `di-zhang-fdu/AIME_1983_2024` for 2020–21, `AI-MO/aimo-validation-aime` for 2022–24. |
| `benchmark/hmmt_*.jsonl` (3 files) | HMMT | 30 / 30 / 33, 93 total | MathArena. |
| `training_set/math_train_orig_train.jsonl` | MATH train pool, original-train half | 7,498 | `nlile/hendrycks-MATH-benchmark` train split, deduplicated. |
| `training_set/math_train_orig_test.jsonl` | MATH train pool, original-test half | 4,498 | Same upstream; these were moved out of the original MATH *test* split by the PRM800K re-split. |

A task reads either one file (`data_file`) or several concatenated in order
(`data_files`), so the `aime` task spans all five years while the years stay
separate on disk:

```yaml
aime: {n: 8, max_tokens: 4096, max_model_len: 8192,
       data_files: [benchmark/aime_2020.jsonl, benchmark/aime_2021.jsonl,
                    benchmark/aime_2022.jsonl, benchmark/aime_2023.jsonl,
                    benchmark/aime_2024.jsonl]}
```

AIME has to be assembled from two upstreams because no single public set covers
all five years; `di-zhang`'s 2023 and 2024 are incomplete (29 and 14 problems)
and are deliberately unused. Problem ids are `aime-<year>-<nn>`. The full
Hendrycks MATH set is 12,500 problems — 7,500 train and 5,000 test; the PRM800K
re-split moves 4,500 test problems into training and keeps 500 as MATH-500,
which is why the training pool and the eval set are disjoint by construction.

The benchmark records under `outputs/` predate the per-year split and use the
earlier flat `aime-<nnnn>` ids. They carry the full problem text, so they still
join to the current files on that.

### Why the training pool is split in two

The 11,996-problem pool is stored as two files because the halves are not
interchangeable. 7,498 problems come from the original Hendrycks *train* split;
4,498 were moved out of the original *test* split by the PRM800K re-split. A
model may have been exposed to the first and not the second — Llama-3.2-3B
demonstrably was — in which case only the test-derived half measures reasoning
rather than recall. Each record carries `split_origin`, computed at build time.

Problem ids (`mathtrain-00000` … `mathtrain-11995`) are assigned over the pool as
a whole in upstream order, *before* the split, so they run across both files
rather than restarting in each. Existing profiling records and curated subsets
reference these ids; the numbering is verified stable across the partition.

### HMMT

The Harvard-MIT Mathematics Tournament runs twice a year — November at Harvard,
February at MIT — and is generally harder than AIME, since entry is by invitation
rather than open selection. MathArena publishes each competition shortly after it
is held, which is what makes these sets useful beyond raw difficulty: they carry
a known date.

| File | Competition | Problems | Integer answers |
|---|---|---:|---:|
| `benchmark/hmmt_feb_2025.jsonl` | February 2025 | 30 | 14 |
| `benchmark/hmmt_nov_2025.jsonl` | November 2025 | 30 | 21 |
| `benchmark/hmmt_feb_2026.jsonl` | February 2026 | 33 | 16 |
| | **total** | **93** | **51** |

Unlike the other benchmarks here these files are committed rather than rebuilt on
demand, because MathArena's sets are recent enough that upstream revisions are
plausible and the exact problem set should stay pinned. `data/build_hmmt.py`
regenerates them and asserts the per-competition counts, that the three sets are
mutually disjoint, and that none of the 93 problems appears in MATH-500.

Only 51 of 93 HMMT answers are integers; the rest are exact forms —
`\frac{1}{576}`, `\frac{9\sqrt{23}}{23}`, `1-\frac{2}{\pi}`,
`(3+\sqrt{6})^{-1/3}` — so scoring goes through `math_verify` symbolic
equivalence rather than the integer comparison AIME uses (`integer_answer:
false` on every HMMT task).

### Grading calibration

An HMMT score is only worth reading if the parser can actually read HMMT
answers, and this matters more here than usual: the reason to run these sets is
to compare competitions against a training cutoff, and a parser that failed
differently across competitions would look exactly like the effect being tested
for. `src/calibrate_grading.py` exercises `grade()` on known inputs — no GPU
time — so a failure is unambiguously a harness bug rather than a weak model:

| Check | What it catches |
|---|---|
| identity | The gold answer, boxed, must grade correct. Below 100% the extractor cannot read this answer format. |
| equivalence | 28 pairs that are the same value written differently. Failures mean correct answers get marked wrong. |
| specificity | A sentinel answer must grade **wrong**. Without it, a grader that returns `True` unconditionally passes everything else. |
| thinking | Identity again behind a `</think>` tag, since thinking models are graded only on what follows it. |

```bash
python src/calibrate_grading.py --data data/benchmark/hmmt_*.jsonl
```

All four pass at 100% on MATH-500 (500), AIME (150) and HMMT (93).

Calibration found and fixed one real defect: `\frac{-1\pm\sqrt{17}}{2}` — the
natural way to write two roots — was graded wrong against a gold of two
comma-separated values. `grading.py` now expands `\pm` / `\mp` before comparing.

One behaviour is documented rather than fixed: `math_verify` compares
numerically within a tolerance, so a truncated decimal (`-0.047619047619`)
matches an exact form (`-\frac{1}{21}`). That makes the grader more permissive,
equally so for every model.

The data is redistributed from MathArena under CC BY-NC-SA 4.0.

## Curated subsets

`data/further_improve/<model>/`, one JSONL per difficulty band. Filenames encode the
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
compute nodes under `$HOME/data/frontier-teacher/generations/` and `outputs/`
respectively, and the records are rebuildable from the shards with
`src/merge_shards.py`.

Raw generations were only ever written for the training-pool profiling runs.
`run_all.sh` does not pass `--save-generations`, so the reasoning traces behind
the MATH-500 and AIME numbers were never saved anywhere — the per-sample verdicts
and extracted answers in `outputs/` are all that exists of those runs.
Re-run with `--save-generations` if the traces themselves are needed.

## Related work

The difficulty-band methodology — selecting a training subset by the student's
measured pass@1 rather than by a difficulty label — follows *Pedagogical RL:
Teaching Models to Teach Themselves from Privileged Information* (Chakraborty,
Ziems et al., 2026), which reports that a hard subset at roughly 8% pass@1
concentrates the learning signal. Our 5%–15% Llama band has a measured mean
pass@1 of 8.7% and is the closest analogue in this repository.
