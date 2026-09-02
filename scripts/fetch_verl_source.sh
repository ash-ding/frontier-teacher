#!/usr/bin/env bash
# Fetch the exact verl source this project runs on, and check it against the
# installed package.
#
#   scripts/fetch_verl_source.sh            fetch into package/, then verify
#   scripts/fetch_verl_source.sh --verify   verify only, fetching if absent
#
# The source is deliberately not committed: 8.3 MB across 362 files would bury
# this repository's own history in `git log` and `git grep`. The pin in
# package/verl.lock reconstructs it exactly, which is what actually matters.
#
# READ-ONLY SNAPSHOT. Python imports verl from site-packages, not from here.
# Editing this tree changes nothing. To modify verl, patch it and reinstall,
# then update the pin and re-run this script - otherwise the snapshot and what
# runs diverge, which the verify step exists to catch.
set -u
cd "$(dirname "$0")/.."
REPO="$PWD"
# shellcheck disable=SC1091
. "$REPO/package/verl.lock"

DEST="$REPO/package/verl-${VERL_COMMIT:0:7}"
VERIFY_ONLY=0
[ "${1:-}" = "--verify" ] && VERIFY_ONLY=1

if [ ! -d "$DEST/verl" ]; then
  echo "fetching verl $VERL_VERSION @ ${VERL_COMMIT:0:7} -> package/"
  tmp=$(mktemp -d)
  trap 'rm -rf "$tmp"' EXIT
  curl -sSL --fail -m 600 -o "$tmp/src.tar.gz" \
    "$VERL_REPO/archive/${VERL_COMMIT}.tar.gz" || {
      echo "  download failed"; exit 1; }
  tar xzf "$tmp/src.tar.gz" -C "$tmp" || exit 1
  src=$(ls -d "$tmp"/verl-"$VERL_COMMIT" 2>/dev/null | head -1)
  [ -d "$src" ] || { echo "  unexpected archive layout"; exit 1; }
  rm -rf "$DEST"; mkdir -p "$DEST"; cp -R "$src"/. "$DEST"/
  echo "  -> $DEST  ($(find "$DEST/verl" -name '*.py' | wc -l | tr -d ' ') .py files)"
else
  echo "already present: $DEST"
fi

# The check that gives the pin its value: is what we fetched what is running?
python3 - "$DEST/verl" <<'PY'
import hashlib, os, sys
snap = sys.argv[1]
try:
    import verl
except ImportError:
    print("  verl is not importable here - skipping the comparison.")
    print("  Run this on a node with the environment active to verify.")
    raise SystemExit(0)
inst = os.path.dirname(verl.__file__)


def digest(root):
    out = {}
    for d, _, fs in os.walk(root):
        for f in fs:
            if f.endswith(".py"):
                p = os.path.join(d, f)
                out[os.path.relpath(p, root)] = hashlib.sha256(
                    open(p, "rb").read()).hexdigest()
    return out


a, b = digest(snap), digest(inst)
only_snap, only_inst = sorted(set(a) - set(b)), sorted(set(b) - set(a))
differ = sorted(f for f in set(a) & set(b) if a[f] != b[f])
print(f"  installed: {inst}")
print(f"  snapshot {len(a)} files, installed {len(b)} files")
if not (only_snap or only_inst or differ):
    print(f"  IDENTICAL - all {len(a)} .py files match sha256")
    raise SystemExit(0)
print(f"  MISMATCH: {len(differ)} differ, {len(only_snap)} only in snapshot, "
      f"{len(only_inst)} only installed")
for f in (differ + only_snap + only_inst)[:10]:
    print("   ", f)
print("  The installed verl is not the pinned commit. Either the environment was"
      " changed, or package/verl.lock is stale.")
raise SystemExit(1)
PY
