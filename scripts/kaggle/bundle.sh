#!/usr/bin/env bash
# Code and panels in one zip, because Kaggle cannot reach either from here.
#
# The panels are the reason this exists: they live outside the repository, and the
# copies under outputs/ are eight-byte fixtures that would upload happily and fail
# on the first read.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PANELS="${RAMPART_PANEL_DIR:-$(cd "$REPO/.." && pwd)/dw-vs-dl-dropout-prediction-latam}"
OUT="${1:-$(cd "$REPO/.." && pwd)/kaggle-rampart}"

rm -rf "$OUT/rampart"; mkdir -p "$OUT/rampart"
cp -r "$REPO/src" "$REPO/scripts" "$OUT/rampart/"
find "$OUT/rampart" -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true

for pair in "azure_results_v7_wb/collection/raw_data" \
            "azure_results_v7_inep/collection/inep_raw"; do
  src="$PANELS/$pair/complete_data.parquet"
  [ -s "$src" ] && [ "$(stat -c%s "$src")" -gt 1000 ] \
    || { echo "missing or truncated: $src" >&2; exit 1; }
  mkdir -p "$OUT/rampart/panels/$pair"
  cp "$src" "$OUT/rampart/panels/$pair/"
done

# The recollected World Bank panel, the one without the cross-sectional imputation
# tiers. Optional, so a bundle built before the recollection still packs; the probe
# that wants it names it and fails loudly if it is absent.
CLEAN_PAIR="worldbank_clean/collection/raw_data"
CLEAN="$PANELS/$CLEAN_PAIR/complete_data.parquet"
if [ -s "$CLEAN" ] && [ "$(stat -c%s "$CLEAN")" -gt 1000 ]; then
  mkdir -p "$OUT/rampart/panels/$CLEAN_PAIR"
  cp "$CLEAN" "$OUT/rampart/panels/$CLEAN_PAIR/"
  echo "packed the recollected World Bank panel as well"
fi

( cd "$OUT" && rm -f rampart-bundle.zip && zip -qr rampart-bundle.zip rampart \
  && rm -rf rampart )
echo "wrote $OUT/rampart-bundle.zip ($(du -h "$OUT/rampart-bundle.zip" | cut -f1))"
