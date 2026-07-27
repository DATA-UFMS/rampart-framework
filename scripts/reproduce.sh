#!/usr/bin/env bash
#
# Reproduz os resultados de ponta a ponta, para um dataset.
#
# O que este script existe para garantir: que reproduzir não dependa de conhecer
# uma sequência. Ele faz o que o pipeline faz, mais as verificações que só fazem
# sentido antes de começar -- ambiente instalado a partir do lock, orçamento de
# núcleos que cabe na máquina, e o hash dos dados de entrada quando um snapshot
# é fornecido.
#
# Uso:
#   scripts/reproduce.sh                          # World Bank, coletando da API
#   scripts/reproduce.sh --dataset inep_censo
#   scripts/reproduce.sh --data-snapshot caminho/  # sem rede, com hash verificado
#
# Os dois datasets têm custos muito diferentes: World Bank leva cerca de uma hora
# e meia, INEP cerca de vinte e nove. Não rode os dois em sequência sem saber
# disso.

set -euo pipefail

DATASET="worldbank"
SNAPSHOT=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset)       DATASET="$2"; shift 2 ;;
        --data-snapshot) SNAPSHOT="$2"; shift 2 ;;
        -h|--help)       sed -n '2,20p' "$0"; exit 0 ;;
        *) echo "Argumento desconhecido: $1" >&2; exit 2 ;;
    esac
done

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "== Ambiente"
python3 -c 'import sys; assert sys.version_info >= (3, 12), sys.version'
python3 -m pip install --quiet --requirement requirements-lock.txt
python3 -m pip install --quiet --editable .

echo "== Orçamento de núcleos"
# Falha antes de começar, e não depois de horas: sobrescrever os núcleos faria a
# latência medida refletir contenção de escalonamento.
python3 - <<'PY'
import sys
sys.path.insert(0, 'src')
from core.scientific_config import SCIENTIFIC_CONFIG
import multiprocessing
engine = SCIENTIFIC_CONFIG['engine_threads']
blas = SCIENTIFIC_CONFIG['blas_threads']
available = multiprocessing.cpu_count()
print(f"  engine={engine} blas={blas} disponíveis={available}")
if engine + blas - 1 > available:
    raise SystemExit(
        f"Orçamento não cabe: ajuste engine_threads em scientific_config.py.")
PY

if [[ -n "$SNAPSHOT" ]]; then
    echo "== Snapshot de dados: $SNAPSHOT"
    python3 scripts/verify_data_snapshot.py --snapshot "$SNAPSHOT" \
        --dataset "$DATASET" --install
fi

echo "== Pipeline (dataset: $DATASET)"
python3 pipeline.py --dataset "$DATASET"

echo "== Testes"
python3 -m pytest tests/ -q

OUTPUTS="outputs/${DATASET}"
echo
echo "Concluído. Artefatos em ${OUTPUTS}/"
echo "  configuração e ambiente: ${OUTPUTS}/scientific_config_snapshot.json"
echo "  cobertura do alvo:       ${OUTPUTS}/collection/raw_data/target_coverage.json"
echo "  equivalência bitwise:    ${OUTPUTS}/statistics/prediction_equivalence.json"
echo "  estatística:             ${OUTPUTS}/statistics/"
