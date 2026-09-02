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

## The reference test set

After every training update, the student is evaluated on a fixed set of problems
— the same set every step, and the same set in every run of this experiment.
That evaluation is not yours to trigger and does not consume any of your
evaluations; it happens on its own once the step closes. Its full results land
in the run directory: the score, every problem, and the student's complete
output for every sample.

The base model is measured on it once before step 0, so the sequence has a
starting point.

**This is not the set you are being judged on.** The result that gets reported
comes from a held-out set that this loop never touches and that you never see.
The set you can read is a reference — it is there so that you, and we, can see
where the student is between steps. The two overlap in kind, not in problems, so
a number that moves on one because the curriculum was drawn from it does not
move on the other.

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
  test_base/              the reference test set run on the BASE model, before
                          any training — same files as a step's test/ below
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
    test/                 the reference test set, run on the checkpoint this
                          step produced — same summary.json / records.jsonl /
                          generations.jsonl as an eval_<M>/, over the fixed
                          problems rather than yours
    result.json           the step's summary: every evaluation, the train, the
                          reference test score, and the checkpoint the next step
                          starts from
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
