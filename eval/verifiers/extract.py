"""Answer extraction and equivalence checking - the shared core of both verifiers.

Kept as one module because extraction is identical for every benchmark; only the
comparison differs, and that is what eval/verifiers/{exact_integer,symbolic}.py
select between. verifier and reward both import from here, so training and
evaluation cannot disagree about what a correct answer is.

The extraction/grading path is the single most common source of bogus
reproduction numbers: a parser that silently misses \boxed{} depresses pass@1
across the board and looks like "the model is weak". Everything here is
deliberately conservative and reports extraction failures separately.
"""
import re

from math_verify import parse, verify

_BOXED = re.compile(r"\\boxed\s*{")


def extract_boxed(text: str):
    """Return the content of the LAST \boxed{...}, brace-balanced. None if absent."""
    starts = [m.end() for m in _BOXED.finditer(text)]
    if not starts:
        return None
    start = starts[-1]
    depth = 1
    i = start
    while i < len(text):
        c = text[i]
        if c == "\\":           # skip escaped char
            i += 2
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[start:i].strip()
        i += 1
    return None                  # unbalanced -> treat as no answer


def extract_answer(text: str):
    """Best-effort final answer. Prefers \boxed{}, then explicit 'answer is X'."""
    b = extract_boxed(text)
    if b is not None:
        return b
    m = re.findall(r"(?:final answer|answer)\s*(?:is|:)\s*\$?([^\n\.\$]{1,80})", text, re.I)
    if m:
        return m[-1].strip().rstrip("$.,")
    return None


def _int_or_none(s):
    if s is None:
        return None
    t = re.sub(r"[,\s$\\]|\\text\{.*?\}", "", str(s))
    t = t.strip()
    m = re.fullmatch(r"[-+]?\d+", t)
    return int(m.group(0)) if m else None


def _variants(pred: str):
    """The prediction, plus rewrites that mean the same thing to a reader.

    \\pm is how a model naturally writes the two roots of a quadratic, but it is
    one expression where the gold answer is two comma-separated ones, so the
    symbolic comparator sees a mismatch. Expanding it costs one extra parse and
    recovers a real correct answer that would otherwise be scored wrong.
    """
    yield pred
    if r"\pm" in pred:
        yield ", ".join((pred.replace(r"\pm", "+"), pred.replace(r"\pm", "-")))
    if r"\mp" in pred:
        yield ", ".join((pred.replace(r"\mp", "-"), pred.replace(r"\mp", "+")))


def grade(pred_text: str, gold_answer: str, integer_answer: bool = False):
    """-> (is_correct, extracted_or_None).

    integer_answer=True (AIME): compare as integers, which is exact and avoids
    all symbolic-equivalence ambiguity. Falls back to math_verify if the
    extracted string is not a bare integer.
    """
    pred = extract_answer(pred_text)
    if pred is None:
        return False, None

    if integer_answer:
        pi, gi = _int_or_none(pred), _int_or_none(gold_answer)
        if pi is not None and gi is not None:
            return pi == gi, pred

    for candidate in _variants(pred):
        try:
            g = parse(f"${gold_answer}$")
            p = parse(f"${candidate}$")
            if g and p and verify(g, p):
                return True, pred
        except Exception:
            continue

    # last resort: normalized string equality
    norm = lambda s: re.sub(r"[\s${}]|\\left|\\right|\\!|\\,", "", str(s))
    return norm(pred) == norm(gold_answer), pred


def answer_segment(text: str) -> str:
    """For thinking models, grade only what follows </think>.

    A model that reasons out loud writes candidate answers while working; the
    last \boxed{} before </think> is a discarded attempt, not the answer. Lived
    in evaluate.py before, which meant the reward function had to import the
    evaluation harness to grade a rollout.
    """
    tag = "</think>"
    return text.split(tag, 1)[1] if tag in text else text
