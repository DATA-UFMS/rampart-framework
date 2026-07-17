#!/usr/bin/env python3
"""Atribui a latência do estágio hierárquico ao engine ou à parte comum.

Entrada:
  - outputs/<dataset>/ml_pipeline/architectures/<paradigma>/models/... (por fold)

Saídas:
  - outputs/<dataset>/statistics/stage_attribution.json
  - outputs/<dataset>/statistics/stage_attribution.tex

Por que existe: o estágio de ML contém o carregamento do fold, que é do engine, e
o ajuste dos modelos, que os três paradigmas fazem igual depois de materializar em
pandas. Reportar só o total atribui ao paradigma uma parcela que ele não controla,
e é sobre a parcela de carregamento que a narrativa do cache de partições fala.

Esta tabela não decide nada por si: ela mostra onde a diferença medida está. Se o
ganho de um paradigma não aparece em fold_load, a explicação por cache de
partições não se sustenta nos números.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_SRC_DIR = os.path.join(_BASE_DIR, "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from core.config import get_absolute_output_path
from core.paradigm_registry import discover_paradigms

OUT_DIR = Path(get_absolute_output_path("outputs/statistics"))


def _results_path(paradigm: str) -> Optional[Path]:
    """Localiza o JSON de resultados hierárquicos de um paradigma."""
    root = Path(get_absolute_output_path(
        f"outputs/ml_pipeline/architectures/{paradigm}/models"))
    if not root.exists():
        return None
    candidates = sorted(root.rglob("hierarchical_analysis*results*.json"))
    return candidates[0] if candidates else None


def _fold_segments(path: Path) -> List[Dict[str, float]]:
    """Pares (fold_load_s, fit_predict_s) por fold, quando registrados."""
    payload = json.loads(path.read_text())
    segments = []
    for fold in payload.get("folds", []):
        load = fold.get("fold_load_s")
        fit = fold.get("fit_predict_s")
        if load is None or fit is None:
            continue
        segments.append({
            "fold_id": fold.get("fold_id"),
            "fold_load_s": float(load),
            "fit_predict_s": float(fit),
        })
    return segments


def attribute() -> Dict:
    per_paradigm: Dict[str, Dict] = {}
    for paradigm in sorted(discover_paradigms()):
        path = _results_path(paradigm)
        if path is None:
            print(f"  [WARN] {paradigm}: resultados hierárquicos ausentes")
            continue
        segments = _fold_segments(path)
        if not segments:
            # Um resultado anterior à decomposição não tem os campos. Reportado,
            # e não preenchido com zero, que entraria nas somas como medição.
            print(f"  [WARN] {paradigm}: {path.name} não registra a decomposição")
            continue
        load_total = sum(s["fold_load_s"] for s in segments)
        fit_total = sum(s["fit_predict_s"] for s in segments)
        total = load_total + fit_total
        per_paradigm[paradigm] = {
            "folds": len(segments),
            "fold_load_s": load_total,
            "fit_predict_s": fit_total,
            "total_s": total,
            # A fração que o engine controla é o que torna a comparação
            # atribuível; o resto é comum aos três por construção.
            "engine_share": load_total / total if total > 0 else None,
            "per_fold": segments,
        }

    report: Dict = {"paradigms": per_paradigm}
    if len(per_paradigm) >= 2:
        # Razões entre paradigmas, por segmento: um ganho no total que não
        # aparece em fold_load não vem do engine.
        baseline = min(per_paradigm, key=lambda p: per_paradigm[p]["total_s"])
        report["fastest_total"] = baseline
        report["ratios_against_fastest"] = {
            paradigm: {
                segment: (values[segment] / per_paradigm[baseline][segment]
                          if per_paradigm[baseline][segment] > 0 else None)
                for segment in ("fold_load_s", "fit_predict_s", "total_s")
            }
            for paradigm, values in per_paradigm.items()
        }
    return report


def _latex(report: Dict) -> str:
    lines = [
        "% Atribuição do estágio hierárquico — gerado automaticamente",
        "\\begin{table}[htb]",
        "\\centering",
        "\\caption{Latência do estágio hierárquico decomposta em carregamento do "
        "fold (engine) e ajuste (comum aos paradigmas).}",
        "\\begin{tabular}{lrrrr}",
        "\\toprule",
        "Paradigma & Folds & Carregamento (s) & Ajuste (s) & Parcela do engine \\\\",
        "\\midrule",
    ]
    for paradigm, values in sorted(report.get("paradigms", {}).items()):
        share = values["engine_share"]
        lines.append(
            f"{paradigm} & {values['folds']} & {values['fold_load_s']:.2f} & "
            f"{values['fit_predict_s']:.2f} & "
            f"{'—' if share is None else f'{share:.1%}'.replace('%', r'\\%')} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(lines)


def main() -> int:
    report = attribute()
    if not report.get("paradigms"):
        print("  Nenhum paradigma registra a decomposição; nada a atribuir.")
        return 0

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "stage_attribution.json").write_text(
        json.dumps(report, indent=2))
    (OUT_DIR / "stage_attribution.tex").write_text(_latex(report))
    print(json.dumps({
        "status": "ok",
        "json": str(OUT_DIR / "stage_attribution.json"),
        "tex": str(OUT_DIR / "stage_attribution.tex"),
        "paradigms": sorted(report["paradigms"]),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
