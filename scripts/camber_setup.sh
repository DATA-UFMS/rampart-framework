#!/usr/bin/env bash
# Put the code and the two panels into Stash, so a Camber job can find them.
#
# The panels are the reason this exists. They live outside the repository, the
# Dockerfile does not carry them, and the copies under outputs/ are eight-byte
# fixtures -- so a job that only had the repo would fail on the first read with a
# message about a parquet footer. Both go up, and the probes find them through
# RAMPART_PANEL_DIR.
#
#   STASH   destination root. Default stash://<username>/rampart/
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PANELS="${RAMPART_PANEL_DIR:-$(cd "$REPO/.." && pwd)/dw-vs-dl-dropout-prediction-latam}"
USER_NAME="${CAMBER_USER:-$(camber me 2>/dev/null | awk '/^Username:/ {print $2}')}"
STASH="${STASH:-stash://${USER_NAME}/rampart}"

[ -n "$USER_NAME" ] || { echo "could not read the username from 'camber me'" >&2; exit 1; }
echo "repo    $REPO"
echo "panels  $PANELS"
echo "stash   $STASH"

for f in "$PANELS/azure_results_v7_wb/collection/raw_data/complete_data.parquet" \
         "$PANELS/azure_results_v7_inep/collection/inep_raw/complete_data.parquet"; do
  # A parquet has PAR1 at both ends; the fixtures under outputs/ are the string
  # "PAR1fake" and would upload happily and fail hours later.
  [ -s "$f" ] && [ "$(stat -c%s "$f")" -gt 1000 ] \
    || { echo "missing or truncated: $f" >&2; exit 1; }
done
echo "panels look real."

# No mkdir: `stash cp` creates the parents, and `stash mkdir` on a path whose
# parent does not exist yet fails loudly with "parent directory does not exist"
# while the upload that follows succeeds anyway. Two lines of alarming output
# for nothing.

echo "uploading code..."
camber stash cp -r "$REPO/src"     "$STASH/src"     --exclude '__pycache__'
camber stash cp -r "$REPO/scripts" "$STASH/scripts" --exclude '__pycache__'
camber stash cp "$REPO/pyproject.toml" "$STASH/pyproject.toml"
camber stash cp "$REPO/requirements-lock.txt" "$STASH/requirements-lock.txt"

echo "uploading panels..."
for pair in \
  "azure_results_v7_wb/collection/raw_data" \
  "azure_results_v7_inep/collection/inep_raw"; do
  camber stash cp "$PANELS/$pair/complete_data.parquet" \
                  "$STASH/panels/$pair/complete_data.parquet"
done

echo
echo "done. Stash now holds:"
camber stash ls "$STASH"
echo
echo "next: scripts/camber_run.sh r1 r2"
