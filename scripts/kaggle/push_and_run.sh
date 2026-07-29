#!/usr/bin/env bash
# Push a probe to Kaggle, run it on their GPU, and bring the log back here.
#
# Kaggle's free tier gives thirty GPU hours a week, which is the reason this path
# exists: the Camber student quota is monthly and was already spent. Everything
# below happens over the API, so a run is a command rather than a browser session.
#
#   scripts/kaggle/push_and_run.sh              # push, wait, print the log
#   scripts/kaggle/push_and_run.sh --no-wait    # push and return
#
# Needs KAGGLE_API_TOKEN in the environment (Settings -> API -> Create New Token;
# the KGAT_ form goes in that variable, not in kaggle.json) and the dataset built
# by scripts/kaggle/bundle.sh already uploaded.
set -euo pipefail

AQUI="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RAIZ="$(cd "$AQUI/../.." && pwd)"
CELULA="${CELULA:-$AQUI/kaggle_r3c.py}"
METADADOS="${METADADOS:-$AQUI/kernel-metadata.json}"
KAGGLE="${KAGGLE:-$RAIZ/.venv/bin/kaggle}"
ESPERAR=1
[[ "${1:-}" == "--no-wait" ]] && ESPERAR=0

: "${KAGGLE_API_TOKEN:?exporte KAGGLE_API_TOKEN=KGAT_... antes de rodar}"
[[ -x "$KAGGLE" ]] || { echo "kaggle CLI ausente em $KAGGLE"; exit 1; }
[[ -f "$CELULA" ]] || { echo "celula ausente: $CELULA"; exit 1; }
[[ -f "$METADADOS" ]] || { echo "metadados ausentes: $METADADOS"; exit 1; }

SLUG="$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['id'])" "$METADADOS")"
PALCO="$(mktemp -d)"
trap 'rm -rf "$PALCO"' EXIT
cp "$METADADOS" "$PALCO/kernel-metadata.json"

# The cell is kept as a plain .py so it stays greppable and diffable; the notebook
# is generated from it, because the kernel is declared kernel_type notebook and a
# mismatch there is rejected on push.
NOTEBOOK="$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['code_file'])" "$METADADOS")"
python3 - "$CELULA" "$PALCO/$NOTEBOOK" <<'PY'
import json, sys
fonte, destino = sys.argv[1], sys.argv[2]
with open(fonte) as f:
    linhas = f.readlines()
notebook = {
    'cells': [{'cell_type': 'code', 'metadata': {}, 'execution_count': None,
               'outputs': [], 'source': linhas}],
    'metadata': {'kernelspec': {'name': 'python3', 'display_name': 'Python 3',
                                'language': 'python'},
                 'language_info': {'name': 'python'}},
    'nbformat': 4, 'nbformat_minor': 5,
}
with open(destino, 'w') as f:
    json.dump(notebook, f, indent=1)
PY

echo "== empurrando $SLUG =="
"$KAGGLE" kernels push -p "$PALCO"

if (( ! ESPERAR )); then
  echo "== rodando; acompanhe com: kaggle kernels status $SLUG =="
  exit 0
fi

echo "== aguardando; a corrida leva ~30 min no INEP =="
while true; do
  sleep 60
  ESTADO="$("$KAGGLE" kernels status "$SLUG" 2>&1 || true)"
  case "$ESTADO" in
    *complete*|*COMPLETE*) echo "== completo =="; break ;;
    *error*|*ERROR*|*cancel*|*CANCEL*)
      echo "== terminou mal =="; echo "$ESTADO"; break ;;
    *) printf '.' ;;
  esac
done

echo "== log =="
"$KAGGLE" kernels output "$SLUG" -p "$RAIZ/outputs/kaggle/${SLUG##*/}" 2>&1 | tail -2
LOG="$RAIZ/outputs/kaggle/${SLUG##*/}/${SLUG##*/}.log"
[[ -f "$LOG" ]] && python3 -c "
import json, sys
for e in json.load(open(sys.argv[1])):
    print(e.get('data', ''), end='')
" "$LOG" || echo "sem log em $LOG"
