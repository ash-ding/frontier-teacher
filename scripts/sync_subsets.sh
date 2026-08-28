#!/usr/bin/env bash
# Push data/further_improve/ (the curated subsets) from this checkout to every compute node, then verify by
# hash. Subsets are the curated output of a profiling run; they are small and
# every node should hold an identical copy.
set -u
cd "$(dirname "$0")/.."
HOSTS="${HOSTS:-lumen-1 lumen-2 lumen-3}"
SRC=data/further_improve

[ -d "$SRC" ] || { echo "no $SRC here"; exit 1; }
base=$(mktemp); find "$SRC" -name '*.jsonl' | sed "s|$SRC/||" | sort | \
  while read -r f; do printf '%s  %s\n' "$(md5sum "$SRC/$f" | cut -d' ' -f1)" "$f"; done > "$base"
echo "local baseline: $(wc -l < "$base") files"

for h in $HOSTS; do
  echo "--- $h"
  ssh -o ConnectTimeout=30 "$h" 'mkdir -p ~/code/frontier-teacher/data/further_improve ~/data/frontier-teacher'
  scp -q -o ConnectTimeout=60 -r "$SRC"/* "$h:~/code/frontier-teacher/data/further_improve/"
  ssh -o ConnectTimeout=30 "$h" '
    rm -rf ~/data/frontier-teacher/further_improve
    mkdir -p ~/data/frontier-teacher/further_improve
    cp -r ~/code/frontier-teacher/data/further_improve/* ~/data/frontier-teacher/further_improve/'
  remote=$(ssh -o ConnectTimeout=30 "$h" \
    "cd ~/code/frontier-teacher/data/further_improve && find . -name '*.jsonl' | sed 's|\./||' | sort | \
     while read -r f; do printf '%s  %s\n' \"\$(md5sum \"\$f\" | cut -d' ' -f1)\" \"\$f\"; done")
  if [ "$remote" = "$(cat "$base")" ]; then echo "    verified: identical to local"
  else echo "    MISMATCH"; diff <(cat "$base") <(echo "$remote") | head; fi
done
rm -f "$base"
