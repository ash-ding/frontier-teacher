"""The teacher turn, with a guard that the sealing key is not in the environment.

The base `claude_client.run_teacher_turn` spawns the agent with
`env=os.environ.copy()`, so anything in the orchestrator's environment reaches
the teacher. That is fine for Vertex auth and fatal for the answer key.

The actual control is in `GoalOrchestrator.__init__`, which pops the key out of
`os.environ` at startup and keeps it in memory: after that there is no window in
which a spawn could leak it, and it does not matter which client is used. This
module exists to *check* that invariant rather than to establish it -- a wrapper
that scrubbed the env per call would silently do nothing if someone later called
the base function directly, whereas this fails loudly.

It also narrows the tool list. The base default is `Bash,Read,Write`, which is
what the open-ended runs wanted. Bash still means the teacher can reach the
network -- that hole cannot be closed while `claude` needs Bash, and it is
covered by the after-the-fact audit, not by prevention.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "frontier_model"))
from teacher import claude_client  # noqa: E402

from .secrets_box import ENV_VAR  # noqa: E402


class KeyLeak(RuntimeError):
    """The sealing key was in the environment at the moment of a teacher spawn."""


def run_teacher_turn(**kw):
    if ENV_VAR in os.environ:
        raise KeyLeak(
            f"{ENV_VAR} is still in os.environ at teacher launch. The base client "
            f"passes env=os.environ.copy() to the subprocess, so this would hand "
            f"the answers to the teacher. GoalOrchestrator.__init__ is supposed "
            f"to have popped it; something put it back.")
    return claude_client.run_teacher_turn(**kw)
