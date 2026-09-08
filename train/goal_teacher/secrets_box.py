"""Answer secrecy that holds against the teacher agent.

Ported from curriculum-rl's `src/secrets_box.py`, unchanged in construction. Its
reasoning applies here verbatim: the teacher runs as the same UNIX user that owns
the run, so file permissions buy nothing -- it can `cat` any 0600 file it finds.
Since "the teacher cannot leak an answer it does not have" is the whole claim of
the Q3 arm, the plaintext must not exist on disk while the teacher is running.

So the answers are sealed under a key that lives only in the orchestrator's
environment (`GOAL_TEACHER_KEY`), and the teacher subprocess is launched with
that variable stripped -- see `client.py`, which exists only to do that. The base
`claude_client.py` passes `env=os.environ.copy()`, which would hand the key
straight to the teacher.

Construction: scrypt KDF -> HMAC-SHA256 counter-mode keystream -> XOR, with an
HMAC-SHA256 tag over the ciphertext. Stdlib only. This is not meant to withstand
cryptanalysis; it is meant to make the answers unreadable to a process that does
not hold the key.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import pathlib
import secrets

ENV_VAR = "GOAL_TEACHER_KEY"
MAGIC = b"GTSEAL01"


def _kdf(key: str, salt: bytes) -> bytes:
    return hashlib.scrypt(key.encode(), salt=salt, n=2**14, r=8, p=1, dklen=64)


def _keystream(k: bytes, nonce: bytes, n: int) -> bytes:
    out, i = bytearray(), 0
    while len(out) < n:
        out += hmac.new(k, nonce + i.to_bytes(8, "big"), hashlib.sha256).digest()
        i += 1
    return bytes(out[:n])


def seal(obj, key: str) -> bytes:
    salt, nonce = secrets.token_bytes(16), secrets.token_bytes(16)
    dk = _kdf(key, salt)
    enc_k, mac_k = dk[:32], dk[32:]
    pt = json.dumps(obj, ensure_ascii=False).encode()
    ct = bytes(a ^ b for a, b in zip(pt, _keystream(enc_k, nonce, len(pt))))
    tag = hmac.new(mac_k, salt + nonce + ct, hashlib.sha256).digest()
    return MAGIC + salt + nonce + tag + ct


def unseal(blob: bytes, key: str):
    if blob[:8] != MAGIC:
        raise ValueError("not a sealed answers file")
    salt, nonce, tag, ct = blob[8:24], blob[24:40], blob[40:72], blob[72:]
    dk = _kdf(key, salt)
    enc_k, mac_k = dk[:32], dk[32:]
    if not hmac.compare_digest(
            tag, hmac.new(mac_k, salt + nonce + ct, hashlib.sha256).digest()):
        raise ValueError("wrong key or corrupted answers file")
    pt = bytes(a ^ b for a, b in zip(ct, _keystream(enc_k, nonce, len(ct))))
    return json.loads(pt.decode())


def require_key() -> str:
    k = os.environ.get(ENV_VAR)
    if not k:
        raise RuntimeError(
            f"{ENV_VAR} is not set. The orchestrator needs it to grade the "
            f"targets. Export it in the shell that launches the run; it is "
            f"stripped from the teacher subprocess automatically.")
    return k


def load_answers(path, key: str | None = None) -> dict:
    return unseal(pathlib.Path(path).read_bytes(), key or require_key())
