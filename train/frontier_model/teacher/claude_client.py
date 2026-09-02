"""Headless `claude -p` wrapper for the teacher turn.

Auth: the child inherits the parent environment, which on this machine carries
CLAUDE_CODE_USE_VERTEX and the Google/Vertex credentials -- the child
authenticates via Vertex; NO ANTHROPIC_API_KEY is set or required. We never log
os.environ or credential contents.

Verified against the installed CLI (claude 2.1.258) rather than the spec:
  * `--max-turns` DOES NOT EXIST in this version. The runaway-loop guard is a
    per-turn wall-clock timeout plus an optional `--max-budget-usd` cost ceiling.
  * the system prompt is passed as a STRING via `--append-system-prompt` (the
    `-file` variant is undocumented here; passing the string is guaranteed).
  * `--permission-mode acceptEdits` is REQUIRED for the Write tool to run
    non-interactively; without it headless writes are denied.
  * `--setting-sources user` is REQUIRED: the step workspace lives under the repo
    tree, so a default invocation would load this repo's project CLAUDE.md -- which
    carries the Factory CEO identity that FORBIDS writing files. Loading only the
    user settings source drops that project memory and lets the teacher write its
    decision.json / data.jsonl. Verified empirically against claude 2.1.258.
  * `--output-format stream-json --verbose` streams one JSON event per line; the
    final `{"type":"result", ...}` line carries result/usage/cost/session/turns.
"""
import json
import os
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class TeacherResult:
    ok: bool
    wallclock_s: float
    returncode: int
    timed_out: bool = False
    result_text: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    num_turns: int = 0
    session_id: str = ""
    is_error: bool = False
    error: str = ""
    raw_result: dict = field(default_factory=dict)


def _parse_final_result(log_path):
    """Reverse-scan the stream-json log for the final {"type":"result", ...} line."""
    try:
        lines = Path(log_path).read_text(errors="replace").splitlines()
    except OSError:
        return None
    for line in reversed(lines):
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if obj.get("type") == "result":
            return obj
    return None


def run_teacher_turn(
    prompt: str,
    system_prompt: str,
    cwd,
    log_path,
    repo_root,
    model: str = "",
    timeout_s: int = 1800,
    allowed_tools: str = "Bash,Read,Write",
    disallowed_tools: str = "Edit",
) -> TeacherResult:
    """Invoke `claude -p` once. Stream its output to log_path; parse the result.

    The teacher writes decision.json + data.jsonl into `cwd` (the step dir). It is
    given read access to the repo via --add-dir so it can `cat`/Read the mounted
    read-only context. Edit is disallowed; the orchestrator additionally
    re-materialises the read-only source from git before it acts on anything.
    """
    argv = [
        "claude", "-p", prompt,
        "--output-format", "stream-json",
        "--verbose",
        "--permission-mode", "acceptEdits",
        "--setting-sources", "user",
        "--allowedTools", allowed_tools,
        "--disallowedTools", disallowed_tools,
        "--add-dir", str(repo_root),
        "--append-system-prompt", system_prompt,
    ]
    if model:
        argv += ["--model", model]

    Path(cwd).mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    timed_out = False
    with open(log_path, "w") as logf:
        # A clean copy of the parent env carries Vertex auth; we do not scrub it,
        # but we never write it anywhere either.
        proc = subprocess.Popen(
            argv, cwd=str(cwd), env=os.environ.copy(),
            stdout=logf, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
        )
        try:
            rc = proc.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            timed_out = True
            proc.kill()
            proc.wait()
            rc = proc.returncode if proc.returncode is not None else -9
    wall = time.time() - t0

    final = _parse_final_result(log_path)
    if timed_out:
        return TeacherResult(ok=False, wallclock_s=wall, returncode=rc,
                             timed_out=True, error=f"teacher turn exceeded {timeout_s}s")
    if final is None:
        return TeacherResult(ok=False, wallclock_s=wall, returncode=rc,
                             error="no final result event in claude stream output")

    usage = final.get("usage", {}) or {}
    is_error = bool(final.get("is_error"))
    return TeacherResult(
        ok=(rc == 0 and not is_error),
        wallclock_s=wall,
        returncode=rc,
        result_text=final.get("result", "") or "",
        input_tokens=int(usage.get("input_tokens", 0) or 0),
        output_tokens=int(usage.get("output_tokens", 0) or 0),
        cost_usd=float(final.get("total_cost_usd", 0.0) or 0.0),
        num_turns=int(final.get("num_turns", 0) or 0),
        session_id=final.get("session_id", "") or "",
        is_error=is_error,
        error="" if not is_error else str(final.get("subtype", "claude reported is_error")),
        raw_result={k: final.get(k) for k in
                    ("subtype", "num_turns", "total_cost_usd", "session_id")},
    )
