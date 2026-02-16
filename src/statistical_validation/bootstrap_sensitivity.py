#!/usr/bin/env python3
"""Análise de sensibilidade bootstrap para SESOI e contagens de iterações.

Este utilitário reutiliza a extração de deltas pareados implementada em
`tost_baseline.py` para recomputar estimativas pontuais, intervalos de confiança
e decisões sob valores alternativos de SESOI e contagens de iterações bootstrap.

Saídas:
- Resumo JSON armazenado em outputs/statistics/bootstrap_sensitivity.json
- Tabela LaTeX opcional (quando --latex é fornecido)
"""

import argparse
import json
import os
import sys
from itertools import product
from typing import Dict, List

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.core.scientific_config import SCIENTIFIC_CONFIG
from src.statistical_validation.tost_baseline import (
    _bootstrap_ci,
    _decision_equivalence,
    _load_baseline_pairs,
    _paired_deltas_for_metric,
)

DEFAULT_OUTPUT_JSON = os.path.join(
    PROJECT_ROOT, "outputs", "statistics", "bootstrap_sensitivity.json"
)
DEFAULT_OUTPUT_LATEX = os.path.join(
    PROJECT_ROOT, "outputs", "statistics", "bootstrap_sensitivity.tex"
)

MetricConfig = Dict[str, Dict[str, float]]


def _collect_deltas() -> Dict[str, np.ndarray]:
    pairs = _load_baseline_pairs()
    metrics = {}
    for metric in ("r2", "nrmse", "mase"):
        metrics[metric] = _paired_deltas_for_metric(pairs, metric)
    return metrics


def _sensitivity_grid(
    metrics: Dict[str, np.ndarray],
    delta_scales: List[float],
    bootstrap_iters: List[int],
    seed: int,
) -> List[Dict[str, object]]:
    """Computa resumo de sensibilidade para todas as combinações."""

    base_deltas: MetricConfig = {
        "r2": {"delta": float(SCIENTIFIC_CONFIG.get("sesoi_r2", 0.01))},
        "nrmse": {"delta": float(SCIENTIFIC_CONFIG.get("sesoi_nrmse", 0.05))},
        "mase": {"delta": float(SCIENTIFIC_CONFIG.get("sesoi_mase", 0.05))},
    }

    rows: List[Dict[str, object]] = []

    for metric, deltas in metrics.items():
        if deltas.size == 0:
            continue
        for scale, iters in product(delta_scales, bootstrap_iters):
            sesoi = base_deltas[metric]["delta"] * scale
            point, (ci_lo, ci_hi) = _bootstrap_ci(deltas, iters=iters, seed=seed)
            decision = _decision_equivalence(ci_lo, ci_hi, sesoi)
            rows.append(
                {
                    "metric": metric,
                    "delta_scale": float(scale),
                    "sesoi": float(sesoi),
                    "bootstrap_iters": int(iters),
                    "n_pairs": int(len(deltas)),
                    "point": float(point),
                    "ci_lo": float(ci_lo),
                    "ci_hi": float(ci_hi),
                    "decision": decision,
                }
            )
    return rows


def _write_outputs(rows: List[Dict[str, object]], json_path: str, latex_path: str) -> None:
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    with open(json_path, "w") as fh:
        json.dump(rows, fh, indent=2)

    df = pd.DataFrame(rows)
    if df.empty:
        return

    df_sorted = df.sort_values(["metric", "delta_scale", "bootstrap_iters"])  # deterministic

    if latex_path:
        os.makedirs(os.path.dirname(latex_path), exist_ok=True)
        latex_table = df_sorted.to_latex(
            index=False,
            float_format=lambda x: f"{x:.3f}",
            longtable=True,
        )
        with open(latex_path, "w") as fh:
            fh.write(latex_table)

    summary_path = os.path.splitext(json_path)[0] + "_summary.csv"
    df_sorted.to_csv(summary_path, index=False)


def main(argv: List[str] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--delta-scales",
        nargs="*",
        type=float,
        default=[0.5, 1.0, 1.5],
        help="Multiplicadores aplicados aos valores de SESOI definidos em scientific_config.py.",
    )
    parser.add_argument(
        "--bootstrap-iters",
        nargs="*",
        type=int,
        default=[1000, 3000, 5000],
        help="Números de iterações de bootstrap a serem avaliados.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=int(SCIENTIFIC_CONFIG.get("random_seed", 42)),
        help="Seed para o gerador pseudoaleatório.",
    )
    parser.add_argument(
        "--json-out",
        type=str,
        default=DEFAULT_OUTPUT_JSON,
        help="Caminho para salvar o resumo em JSON.",
    )
    parser.add_argument(
        "--latex",
        action="store_true",
        help="Gerar tabela LaTeX com os resultados.",
    )
    args = parser.parse_args(argv)

    metrics = _collect_deltas()
    rows = _sensitivity_grid(metrics, args.delta_scales, args.bootstrap_iters, args.seed)
    latex_path = DEFAULT_OUTPUT_LATEX if args.latex else ""
    _write_outputs(rows, args.json_out, latex_path)

    print(f"Resultados gravados em {args.json_out}")
    if args.latex:
        print(f"Tabela LaTeX gravada em {latex_path}")
    return 0



if __name__ == "__main__":
    raise SystemExit(main())
