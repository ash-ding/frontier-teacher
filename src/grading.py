"""Answer extraction and equivalence checking.

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

    try:
        g = parse(f"${gold_answer}$")
        p = parse(f"${pred}$")
        if g and p and verify(g, p):
            return True, pred
    except Exception:
        pass

    # last resort: normalized string equality
    norm = lambda s: re.sub(r"[\s${}]|\\left|\\right|\\!|\\,", "", str(s))
    return norm(pred) == norm(gold_answer), pred
