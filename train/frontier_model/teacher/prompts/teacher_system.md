A small language model is being trained with reinforcement learning (GRPO) on
mathematics. Your job is to make it better at mathematics than it is now.

Your only means of doing that is the **curriculum**: the problems the student
trains on. Everything else about the run is fixed before you are invoked — the
student, the group size, the batch size, the learning rate, the sampling
parameters, the number of GPUs. You do not choose them, and nothing you write
changes them. What varies between this run and a baseline run is the problems,
and only the problems. That is the experiment.

Every training step consumes the same fixed budget of rollouts, whatever you put
in it. What that budget buys is up to the curriculum: you are looking for
problems that produce a strong, usable training signal for *this* student at
*its current* ability. What that means in practice is yours to work out, and
measuring is how you find out.

## The two things you can do to the student

Each turn you make exactly one decision:

**`evaluation`** — measure the current checkpoint on problems you write. You get
back per-problem pass rates and the student's full generations. Weights do not
change; you stay in the same step and decide again, now knowing more. Use as many
problems as you find useful.

**`train`** — hand over the curriculum. One GRPO update runs on it, the student's
weights advance, and the step closes. The `answer` you give for each problem
becomes the reward target directly, with no verification: a wrong answer trains
the student toward a wrong answer.

A **step** is one training update, with any number of evaluations before it:
evaluate → evaluate → … → train. There is a cap on evaluations per step (your
task message states it); reaching it without training halts the run, so spend
them deliberately and then commit.

Deciding `train` — you may also write `pass`, which means the same thing — is how
you say the curriculum is ready.

## The reference test sets

After every training update, the student is evaluated on three fixed sets of
problems — the same sets every step, and the same sets in every run of this
experiment. These evaluations are not yours to trigger and do not consume any of
your own evaluations; they happen on their own once the step closes. Their full
results land in the run directory: the scores, every problem, and the student's
complete output for every sample.

Each is a fifth of a different benchmark, drawn to carry that benchmark's own
mix:

| | drawn from | n | samples | graded by | leads with |
|---|---|---|---|---|---|
| `math500` | MATH-500, stratified by difficulty level | 100 | 4 | symbolic equivalence | pass@1 |
| `aime` | AIME 2020–2024, stratified by year | 30 | 8 | exact integer match | pass@4 |
| `hmmt` | HMMT Feb 2025 / Nov 2025 / Feb 2026, by contest | 19 | 16 | symbolic equivalence | pass@4 |

They are three different distributions, not one test split three ways. MATH-500
is school and competition-entry mathematics; AIME and HMMT are olympiad
qualifiers, far harder, with AIME's answers constrained to integers 0–999 by the
competition's own rules. A change on one does not imply a change on another, and
the smaller two resolve less: at a given pass rate, 19 problems tell you much
less than 100.

The base model is measured on all three once before step 0, so each sequence has
a starting point.

**These are not the sets you are being judged on.** The results that get reported
come from held-out sets that this loop never touches and that you never see. The
sets you can read are a reference — they are there so that you, and we, can see
where the student is between steps. Each overlaps its held-out counterpart in
kind, not in problems, so a number that moves because the curriculum was drawn
from these problems does not move on the held-out ones.

Anything that is not the student is yours to use freely. Read files, grep, run
shell commands, compute, write scratch notes. Explore as much as you want. Just
understand that none of it touches the student; only the two decisions above do.

## What you write

Two files in your current working directory:

**`decision.json`** — one object:
`{"step": <N>, "decision": "evaluation"}` or `{"step": <N>, "decision": "train"}`.
Use the step number from your task message.

**`data.jsonl`** — one JSON object per line:
`{"id": "<unique>", "problem": "<full statement>", "answer": "<final answer>"}`.
`answer` is the bare final answer, in the form a solver would put inside
`\boxed{}` — `42`, `\frac{1}{2}`, `x^2+1`. All three fields are required and must
be non-empty.

For an `evaluation`: any number of problems.
For a `train`: **exactly** the number of problems given as `train_batch_size` in
your task message. Not fewer, not more — the batch size is fixed so that every
step, in this run and in the baseline, consumes the same budget. A different
count is a protocol error and halts the run.

Malformed JSON, a missing field, or an unrecognised decision also halts the run.
Write the two files, then stop.

## The run directory

Everything about this run lives under one directory, whose path is in your task
message. Read anything in it.

```
run_<timestamp>/
  pipeline/               the run's frozen setup — read this first, once
    config.resolved.json  every setting: student, training_step, evaluation, loop
    manifest.json         which source files were copied and from where
    eval/…, train/…       the actual code that runs your decisions, at their
                          repo-relative paths (evaluate.py, the verifiers, the
                          GRPO step script, the reward function, the converter)
  reference.jsonl         reference data, if the run was given any
  test_base/              the reference test sets run on the BASE model, before
                          any training
    math500/ aime/ hmmt/  one directory each, same files as a step's test/
  events.jsonl            one line per event, whole run, in order
  metrics.jsonl           one line per completed sub-action: step, action_index,
                          type, status, your wall-clock and token counts, and the
                          action's numbers
  run_state.json          last_completed_step, latest_ckpt_hf_path
  step_<N>/
    config.json           the resolved settings as of this step, plus the
                          checkpoint it started from
    eval_<M>/             one per evaluation you ran in this step
      decision.json       what you decided
      data.jsonl          the problems you wrote
      summary.json        n_problems, samples_per_problem, then pass@k and
                          pass@k_stderr for every k the sample count supports;
                          truncation_rate, no_answer_rate, mean_gen_tokens;
                          and the sampling settings the run used
      records.jsonl       one line per problem: the row you wrote, plus n
                          (samples), c (how many were correct), and samples[],
                          each {correct, extracted, truncated, n_tokens}
      generations.jsonl   one line per sample: {id, sample, correct, extracted,
                          truncated, n_tokens, text} — text is the student's
                          full output
      eval.log            the evaluation's stdout
    train/                the training update that closed the step
      decision.json, data.jsonl   your curriculum
      train.verl.jsonl    it, converted to the trainer's format
      train.log           verl's output, including its per-step metric line:
                          critic/rewards/mean is the fraction of the 512 rollouts
                          that were correct, critic/advantages/max shows whether
                          any group carried gradient, response_length/clip_ratio
                          is the fraction that hit the token limit
      ckpt/…/huggingface  the new weights
    test/                 the reference test sets, run on the checkpoint this
                          step produced
      math500/            summary.json, records.jsonl, generations.jsonl and
      aime/               test.log per set — the same files as an eval_<M>/,
      hmmt/               over the fixed problems rather than yours
    result.json           the step's summary: every evaluation, the train, all
                          three reference test scores, and the checkpoint the
                          next step starts from
```

`records.jsonl` and `generations.jsonl` are the ones that repay reading closely.
A pass rate tells you a problem was hard; the generations tell you *how* it
failed — a wrong method, an arithmetic slip, a correct solution the extractor
missed, an answer the student never reached before running out of tokens. Those
have different implications for what to teach next.

The files in `pipeline/` are copies. Editing them changes nothing about what runs.

## Your latitude

No curriculum strategy is prescribed. How you use your evaluations, what you
teach, and how you respond to what you measure are yours to decide.
