"""Shared summary computation, used by evaluate.py and merge_shards.py."""
import numpy as np


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased estimator of pass@k given c correct out of n samples."""
    if n - c < k:
        return 1.0
    return 1.0 - float(np.prod(1.0 - k / np.arange(n - c + 1, n + 1)))


def summarize(records, cfg, task_name, task_cfg, extra=None):
    N = len(records)
    n = records[0]["n"]
    n_total = sum(r["n"] for r in records)
    n_trunc = sum(s["truncated"] for r in records for s in r["samples"])
    n_noext = sum(s["extracted"] is None for r in records for s in r["samples"])
    acc = np.array([r["c"] / r["n"] for r in records])

    out = {
        "tag": f"{cfg['name']}__{task_name}",
        "model": cfg["model"],
        "config": cfg["name"],
        "task": task_name,
        "n_problems": N,
        "samples_per_problem": n,
        "pass@1": float(acc.mean()),
        "pass@1_stderr": float(acc.std(ddof=1) / np.sqrt(N)),
        "truncation_rate": n_trunc / n_total,
        "no_answer_rate": n_noext / n_total,
        "sampling": {"temperature": cfg["temperature"], "top_p": cfg["top_p"],
                     "top_k": cfg.get("top_k", -1), "max_tokens": task_cfg["max_tokens"],
                     "enable_thinking": cfg.get("enable_thinking")},
    }
    for k in (1, 4, 8, 16):
        if k <= n:
            vals = [pass_at_k(r["n"], r["c"], k) for r in records]
            out[f"pass@{k}"] = float(np.mean(vals))
            out[f"pass@{k}_stderr"] = float(np.std(vals, ddof=1) / np.sqrt(N))
    out.update(extra or {})
    return out
