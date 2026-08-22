#!/usr/bin/env bash
# Push a probe to Kaggle, run it on their GPU, and bring the log back here.
#
# Kaggle's free tier gives thirty GPU hours a week, which is the reason this path
# exists: the Camber student quota is monthly and was already spent. Everything
# below happens over the API, so a run is a command rather than a browser session.
#
#   scripts/kaggle/push_and_run.sh              # push, wait, print the log
#   scripts/kaggle/push_and_run.sh --no-wait    # push and return
#   scripts/kaggle/push_and_run.sh --cap-all 0  # the uncapped arm, as its own kernel
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
CAP_ALL=1
PAINEIS=""
SUFIXO=""
SONDAS=""
PROBE=""
FOLDS=""
SEM_GPU=0
while (( $# )); do
  case "$1" in
    --no-wait) ESPERAR=0; shift ;;
    --cap-all) CAP_ALL="${2:?--cap-all precisa de 0 ou 1}"; shift 2 ;;
    --panels) PAINEIS="${2:?--panels precisa da lista separada por virgula}"; shift 2 ;;
    --suffix) SUFIXO="${2:?--suffix precisa de um nome}"; shift 2 ;;
    --probes) SONDAS="${2:?--probes precisa de um inteiro}"; shift 2 ;;
    --probe) PROBE="${2:?--probe precisa de um nome registrado em kaggle_r3c.SCRIPTS}"; shift 2 ;;
    --folds) FOLDS="${2:?--folds precisa da lista de indices separada por virgula}"; shift 2 ;;
    --no-gpu) SEM_GPU=1; shift ;;
    *) echo "argumento desconhecido: $1"; exit 1 ;;
  esac
done
[[ "$CAP_ALL" == "0" || "$CAP_ALL" == "1" ]] || { echo "--cap-all: use 0 ou 1"; exit 1; }

: "${KAGGLE_API_TOKEN:?exporte KAGGLE_API_TOKEN=KGAT_... antes de rodar}"
[[ -x "$KAGGLE" ]] || { echo "kaggle CLI ausente em $KAGGLE"; exit 1; }
[[ -f "$CELULA" ]] || { echo "celula ausente: $CELULA"; exit 1; }
[[ -f "$METADADOS" ]] || { echo "metadados ausentes: $METADADOS"; exit 1; }

PALCO="$(mktemp -d)"
trap 'rm -rf "$PALCO"' EXIT

# Each arm is pushed as its own kernel, so the logs survive side by side and the
# tables can be diffed instead of one overwriting the other.
python3 - "$METADADOS" "$PALCO/kernel-metadata.json" "$CAP_ALL" "$SUFIXO" "$SEM_GPU" <<'PY'
import json, sys
origem, destino, cap, sufixo, sem_gpu = sys.argv[1:6]
meta = json.load(open(origem))
marca = sufixo or ('uncapped' if cap == '0' else '')
if marca:
    meta['id'] += f'-{marca}'
    meta['title'] += f'-{marca}'
# Kaggle allows two concurrent GPU sessions, and the routes probe is classical-only:
# asking for an accelerator it never touches would queue behind the runs that need
# one. Declared here rather than in a second metadata file, so the two arms differ in
# the one field that differs.
if sem_gpu == '1':
    meta['enable_gpu'] = False
    meta.pop('machine_shape', None)
with open(destino, 'w') as f:
    json.dump(meta, f, indent=2)
PY
SLUG="$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['id'])" "$PALCO/kernel-metadata.json")"

# The cell is kept as a plain .py so it stays greppable and diffable; the notebook
# is generated from it, because the kernel is declared kernel_type notebook and a
# mismatch there is rejected on push.
NOTEBOOK="$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['code_file'])" "$PALCO/kernel-metadata.json")"
python3 - "$CELULA" "$PALCO/$NOTEBOOK" "$CAP_ALL" "$PAINEIS" "$SONDAS" "$PROBE" \
        "${TABPFN_TOKEN:-}" "$FOLDS" <<'PY'
import json, sys
fonte, destino, cap, paineis, sondas, probe, tabpfn, folds = sys.argv[1:9]
with open(fonte) as f:
    linhas = f.readlines()

def celula(fonte):
    return {'cell_type': 'code', 'metadata': {}, 'execution_count': None,
            'outputs': [], 'source': fonte}

# A leading cell rather than surgery on the body: the arm is chosen by setting the
# variables the body reads with setdefault/get, so the shipped text is never
# rewritten and what ran is legible in the notebook itself.
prologo = ["import os\n"]
if cap == '0':
    prologo.append("os.environ['RAMPART_CAP_ALL'] = '0'\n")
if paineis:
    prologo.append(f"os.environ['RAMPART_PROBE_PANELS'] = {paineis!r}\n")
if sondas:
    prologo.append(f"os.environ['RAMPART_PROBES'] = {sondas!r}\n")
# The v3 weights are gated and Kaggle's Secrets do not reach a run pushed through
# the API -- measured, the proxy is not provisioned and UserSecretsClient raises
# ConnectionError. So the token travels in the generated prologue instead. It is
# written HERE and not into the shipped cell: the staging directory is temporary,
# while scripts/kaggle/kaggle_r3c.py is committed, and a credential in git history
# outlives the revocation in a way a private notebook does not. The notebook is
# is_private; rotate the token at ux.priorlabs.ai when the runs are done.
if probe:
    prologo.append(f"os.environ['RAMPART_PROBE'] = {probe!r}\n")
if folds:
    prologo.append(f"os.environ['RAMPART_FOLDS'] = {folds!r}\n")
if tabpfn:
    prologo.append(f"os.environ['TABPFN_TOKEN'] = {tabpfn!r}\n")
celulas = [celula(linhas)]
if len(prologo) > 1:
    celulas.insert(0, celula(prologo))

notebook = {
    'cells': celulas,
    'metadata': {'kernelspec': {'name': 'python3', 'display_name': 'Python 3',
                                'language': 'python'},
                 'language_info': {'name': 'python'}},
    'nbformat': 4, 'nbformat_minor': 5,
}
with open(destino, 'w') as f:
    json.dump(notebook, f, indent=1)
PY

echo "== empurrando $SLUG =="
# `kaggle kernels push` exits 0 even when it refuses -- the concurrent-GPU-session
# limit prints "Kernel push error" and returns success -- so the output is what says
# whether anything was submitted.
SAIDA="$("$KAGGLE" kernels push -p "$PALCO" 2>&1)"
echo "$SAIDA"
if ! grep -q "successfully pushed" <<<"$SAIDA"; then
  echo "== NAO submetido =="
  exit 1
fi

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
