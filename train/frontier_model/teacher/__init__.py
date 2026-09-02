"""Observational teacher-in-the-loop RL curriculum loop.

An external Python orchestrator (NOT Claude Code) drives a run: each step it asks
a frontier model, invoked headlessly as `claude -p`, to either *evaluate* the
current student checkpoint or *train* it on a curriculum the teacher authors,
then executes that choice for real (GRPO via verl). The teacher's only interface
is the data; the student, the group size, the batch size, the sampling and the
GPU count are fixed from the config at run start, so a teacher run and the
no-teacher baseline differ in the problems and in nothing else.

Deliberately observational: the teacher's stated answers become GRPO ground truth
directly, with no verification gate, so we can watch the raw behaviour of a
frontier model acting as a curriculum teacher.

Modules:
  protocol       strict decision/data/result schemas, atomic writes, duplicate detector
  claude_client  thin headless `claude -p` wrapper (Vertex auth, stream-json parse)
  tools          the two decision executors, the pipeline snapshot, per-turn context
  orchestrator   the driver loop (entry point: `python train/frontier_model/teacher/orchestrator.py`)
"""

__all__ = ["protocol", "claude_client", "tools", "orchestrator"]
