"""Observational teacher-in-the-loop RL curriculum loop.

An external Python orchestrator (NOT Claude Code) drives a run: each step it asks
a frontier model, invoked headlessly as `claude -p`, to choose either to
*evaluate* the current student checkpoint or to *train* it on up to 16 problems
the teacher authors, then executes that choice for real on `llama32-3b` (GRPO via
verl). This is deliberately observational: the teacher's stated answers become
GRPO ground truth directly, with no verification gate, so we can watch the raw
behaviour of a frontier model acting as a curriculum teacher.

Modules:
  protocol       strict decision/data/result schemas, atomic writes, duplicate detector
  claude_client  thin headless `claude -p` wrapper (Vertex auth, stream-json parse)
  tools          the two decision executors + read-only context assembly
  orchestrator   the driver loop (entry point: `python -m src.teacher.orchestrator`)
"""

__all__ = ["protocol", "claude_client", "tools", "orchestrator"]
