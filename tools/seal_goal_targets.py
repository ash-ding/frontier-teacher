#!/usr/bin/env python3
"""Seal .secrets/arxiv3.answers.json -> .secrets/arxiv3.sealed, shred the plaintext.

Prints the key ONCE. It is never written to disk: a key file would sit where the
teacher can read it, which is the failure this whole mechanism exists to prevent.
Store it off this machine. Without it the orchestrator cannot grade the targets;
with it, nothing else about a run changes.

    python tools/seal_goal_targets.py                      # generate a fresh key
    GOAL_TEACHER_KEY=... python tools/seal_goal_targets.py # reuse one
"""
import json
import os
import pathlib
import secrets
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "train"))
from goal_teacher.secrets_box import ENV_VAR, seal, unseal  # noqa: E402

plain = ROOT / ".secrets" / "arxiv3.answers.json"
sealed = ROOT / ".secrets" / "arxiv3.sealed"

if not plain.exists():
    sys.exit(f"{plain} not found -- run tools/fetch_goal_targets.py first")

key = os.environ.get(ENV_VAR) or secrets.token_urlsafe(32)
obj = json.loads(plain.read_text())

sealed.write_bytes(seal(obj, key))
sealed.chmod(0o600)
assert unseal(sealed.read_bytes(), key) == obj, "round-trip failed"

# Overwrite before unlink: unlink alone leaves the bytes recoverable from the
# page cache and the free list.
n = plain.stat().st_size
with open(plain, "r+b") as f:
    for _ in range(3):
        f.seek(0)
        f.write(secrets.token_bytes(n))
        f.flush()
        os.fsync(f.fileno())
plain.unlink()

print(f"sealed {len(obj)} answers -> {sealed}")
print(f"ids: {', '.join(sorted(obj))}")
print("plaintext shredded and removed")
print(f"\n  export {ENV_VAR}={key}\n")
print("Store this key somewhere the teacher agent cannot reach. It is not on")
print("disk anywhere, and it cannot be recovered from the sealed file.")
