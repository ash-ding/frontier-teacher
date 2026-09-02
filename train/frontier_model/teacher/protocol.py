"""Strict schemas, atomic file I/O, and the duplicate-step circuit breaker.

The whole loop is file-mediated: the teacher writes `decision.json` + `data.jsonl`,
the orchestrator writes `result.json`, and the next teacher turn reads them back.
A half-written file read across that boundary silently becomes poisoned training
data, so every write here goes through write-tmp -> fsync -> os.replace (atomic on
POSIX). Every read validates against a strict schema and raises on the first
violation rather than best-effort parsing an ambiguous file into a real GRPO run.
"""
import hashlib
import json
import os
import tempfile
from pathlib import Path

VALID_DECISIONS = ("evaluation", "train")

# What the teacher may write for each decision. "pass" reads naturally for
# "the curriculum is ready, hand it over" and costs nothing to accept.
DECISION_ALIASES = {"evaluation": "evaluation", "evaluate": "evaluation",
                    "eval": "evaluation",
                    "train": "train", "pass": "train"}


class ProtocolError(ValueError):
    """A decision/data/result file failed strict validation. Halt, do not retry."""


# --------------------------------------------------------------------------- IO

def atomic_write_text(path, text: str) -> None:
    """Write text durably: tmp in the same dir -> fsync -> atomic rename."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".tmp_", suffix=path.suffix)
    try:
        with os.fdopen(fd, "w") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def atomic_write_json(path, obj) -> None:
    atomic_write_text(path, json.dumps(obj, indent=2, ensure_ascii=False))


def atomic_write_jsonl(path, rows) -> None:
    atomic_write_text(path, "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def read_json_strict(path):
    """Read+parse a JSON file, raising ProtocolError on a MISSING file or a JSON
    syntax error -- not just a schema violation.

    The teacher writes `decision.json` with `Write`; a stray comma or a partial
    file yields a syntactically-broken document, and a missing file means the
    teacher wrote nothing. Both must funnel into the same halt-and-log path as a
    schema violation, never escape as an uncaught FileNotFoundError/JSONDecodeError.
    """
    path = Path(path)
    try:
        return json.loads(path.read_text())
    except FileNotFoundError as e:
        raise ProtocolError(f"{path.name} does not exist") from e
    except json.JSONDecodeError as e:
        raise ProtocolError(f"{path.name} is not valid JSON: {e}") from e


def read_jsonl_strict(path):
    """Like read_jsonl, but a missing file or a malformed line becomes a
    ProtocolError rather than an uncaught FileNotFoundError/JSONDecodeError."""
    path = Path(path)
    try:
        return read_jsonl(path)
    except FileNotFoundError as e:
        raise ProtocolError(f"{path.name} does not exist") from e
    except json.JSONDecodeError as e:
        raise ProtocolError(f"{path.name} is not valid JSON: {e}") from e


# --------------------------------------------------------------------- schemas

def validate_decision(obj, step: int):
    """Return the validated decision dict or raise ProtocolError.

    Strict: `decision` must be exactly one of VALID_DECISIONS and `step` must
    match the step we asked for (a stale file from a prior step is a bug, not a
    thing to guess around).
    """
    if not isinstance(obj, dict):
        raise ProtocolError(f"decision.json is not a JSON object: {type(obj).__name__}")
    if "decision" not in obj:
        raise ProtocolError("decision.json missing required key 'decision'")
    raw = str(obj["decision"]).strip().lower()
    if raw not in DECISION_ALIASES:
        raise ProtocolError(
            f"decision {obj['decision']!r} not in {sorted(DECISION_ALIASES)}")
    if "step" in obj and obj["step"] != step:
        raise ProtocolError(f"decision.json step {obj['step']} != expected {step}")
    return {"step": step, "decision": DECISION_ALIASES[raw]}


def validate_data_rows(rows):
    """Every row must carry non-empty id/problem/answer strings. Raise otherwise.

    `answer` becomes GRPO ground_truth verbatim with no verification, so an empty
    or missing answer would poison a whole rollout group; reject it up front.
    """
    if not isinstance(rows, list) or not rows:
        raise ProtocolError("data.jsonl produced no rows")
    clean = []
    for i, r in enumerate(rows):
        if not isinstance(r, dict):
            raise ProtocolError(f"data.jsonl row {i} is not a JSON object")
        for k in ("id", "problem", "answer"):
            if k not in r or r[k] is None or str(r[k]).strip() == "":
                raise ProtocolError(f"data.jsonl row {i} missing/empty {k!r}")
        clean.append({
            "id": str(r["id"]),
            "problem": str(r["problem"]),
            "answer": str(r["answer"]),
            # optional, passed through if present
            **{k: r[k] for k in ("level", "subject", "band") if k in r},
        })
    return clean


def validate_train_batch(rows, train_batch_size):
    """A training curriculum must be EXACTLY train_batch_size problems.

    The batch size is fixed for the run, so that every teacher step consumes the
    same rollout budget as every other and as the no-teacher baseline. Padding a
    short curriculum would invent problems; truncating a long one would silently
    discard the teacher's design; accepting either would make one step's update
    incomparable to the rest. So neither - say what is wrong and halt.
    """
    if len(rows) != train_batch_size:
        raise ProtocolError(
            f"a train needs exactly {train_batch_size} problems "
            f"(train_batch_size); data.jsonl has {len(rows)}")
    return rows


def validate_result(obj):
    """Light validation used only on the idempotent-restart path.

    A step's result.json is a STEP SUMMARY, written only when the step closes on
    a train sub-action: it carries `n_evals`, the per-eval `evals` list, and the
    terminating `train`. `step`/`status` are required; `train` +
    `latest_checkpoint_hf_path` mark a genuinely completed TRAINING step (the
    orchestrator additionally checks those before treating a step as closed).
    """
    if not isinstance(obj, dict):
        raise ProtocolError("result.json is not a JSON object")
    for k in ("step", "status"):
        if k not in obj:
            raise ProtocolError(f"result.json missing required key {k!r}")
    return obj


# ------------------------------------------------------------ duplicate breaker

def content_hash(rows) -> str:
    """Stable hash of the teacher's authored problems (order-sensitive)."""
    blob = json.dumps(rows, sort_keys=True, ensure_ascii=False).encode()
    return hashlib.sha256(blob).hexdigest()


def is_duplicate_step(prev_decision, prev_data_hash, cur_decision, cur_data_hash) -> bool:
    """Circuit breaker for a stuck teacher.

    The runaway-loop incident this guards against is a teacher that emits the
    *identical* action twice in a row and makes no progress. We deliberately do
    NOT halt merely because the decision label repeats (training on two different
    problem sets back-to-back is legitimate and expected in an observational run)
    -- only when BOTH the decision and the authored data are byte-identical to the
    immediately preceding step.
    """
    if prev_decision is None or prev_data_hash is None:
        return False
    return prev_decision == cur_decision and prev_data_hash == cur_data_hash
