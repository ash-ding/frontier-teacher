"""Fail if a baseline's record ids do not match what the checkpoint evaluations use.

Baseline files were copied between nodes by hand, and older copies survived in
places: a stale `outputs/{cfg}__aime.records.jsonl` holds ids `aime-0000` where
the current harness emits `aime-2020-00`. The scores are identical either way, so
nothing looks wrong - the paired analysis simply finds an empty intersection and
reports the cell as missing. That failure mode cost three separate debugging
rounds, twice on a node and once in the committed repository.

  python eval/check_baselines.py [outputs-dir]
"""
import json, glob, os, re, sys, collections

OUT = sys.argv[1] if len(sys.argv) > 1 else "outputs"
TASKS = ("math500", "aime", "hmmt")
CKPT = re.compile(r"^(?P<cfg>.+?)__pass1_.+?(?:__g\d+)?__step\d+__(?P<task>\w+)$")

def first_id(p):
    with open(p) as f:
        return json.loads(f.readline())["id"]

# what the checkpoint evaluations actually emit, per task
seen = collections.defaultdict(set)
for p in glob.glob(os.path.join(OUT, "grpo", "*", "records.jsonl")):
    m = CKPT.match(os.path.basename(os.path.dirname(p)))
    if m and m["task"] in TASKS:
        seen[m["task"]].add(first_id(p))

bad = 0
for p in sorted(glob.glob(os.path.join(OUT, "benchmarks", "*", "records.jsonl"))):
    b = os.path.basename(os.path.dirname(p))
    task = b.rsplit("__", 1)[-1]
    if task not in TASKS or not seen.get(task):
        continue
    fid = first_id(p)
    if fid not in seen[task]:
        bad += 1
        print(f"  STALE  {b}\n         has {fid!r}, checkpoints use {sorted(seen[task])}")
    else:
        print(f"  ok     {b:<44} {fid}")
print(f"\n{bad} stale baseline(s)")
sys.exit(1 if bad else 0)
