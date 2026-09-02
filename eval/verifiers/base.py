"""What every verifier must provide."""
from dataclasses import dataclass


@dataclass
class Verdict:
    """One graded sample.

    `extracted` is kept separately from `correct` because "wrote nothing a
    parser could read" and "wrote a wrong answer" are different failures that
    score the same. Conflating them makes a broken extractor look like a weak
    model, which is the single most common way a reproduction goes wrong.
    """
    correct: bool
    extracted: str | None


class Verifier:
    name = "base"

    def grade(self, text: str, gold: str) -> Verdict:
        raise NotImplementedError
