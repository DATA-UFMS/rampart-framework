#!/usr/bin/env python3
"""Attributes the latency of the ML stages to the engine or to the common part.

Input:
  - outputs/<dataset>/ml_pipeline/architectures/<paradigm>/models/... (per fold)

Outputs:
  - outputs/<dataset>/statistics/stage_attribution.json
  - outputs/<dataset>/statistics/stage_attribution.tex

Why it exists: the ML stage contains the fold loading, which belongs to the engine,
and the model fitting, which the three paradigms do identically after materialising
into pandas. Reporting only the total attributes to the paradigm a share it does not
control, and it is about the loading share that the partition-cache narrative speaks.

This table decides nothing on its own: it shows where the measured difference sits.
If a paradigm's gain does not appear in fold_load, the partition-cache explanation
does not hold up in the numbers.
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


STAGES = {
    # The artifact name pattern of each stage. The two ML stages are
    # decomposed, and Dask wins in both on INEP, so attributing only one would
    # leave half of the claim unmeasured.
    'hierarchical': "hierarchical_analysis*results*.json",
    'baseline': "baseline_analysis*results*.json",
}


def _results_path(paradigm: str, pattern: str) -> Optional[Path]:
    """Locates the results JSON of a stage of a paradigm."""
    root = Path(get_absolute_output_path(
        f"outputs/ml_pipeline/architectures/{paradigm}/models"))
    if not root.exists():
        return None
    candidates = sorted(root.rglob(pattern))
    return candidates[0] if candidates else None


def _folds_of(payload: Dict) -> List[Dict]:
    """The folds, under the key that each stage uses.

    The hierarchical stage writes a list under 'folds'; the baseline one writes a
    dictionary with 'fold_<n>' keys. The difference is of layout, not of content.
    """
    if isinstance(payload.get("folds"), list):
        return payload["folds"]
    container = (payload.get("baseline_model_results")
                 or payload.get("baseline_models") or payload)
    if not isinstance(container, dict):
        return []
    return [value for key, value in sorted(container.items())
            if isinstance(key, str) and key.startswith("fold_")
            and isinstance(value, dict)]


def _fold_segments(path: Path) -> List[Dict[str, float]]:
    """Pairs (fold_load_s, fit_predict_s) per fold, when recorded."""
    payload = json.loads(path.read_text())
    segments = []
    for fold in _folds_of(payload):
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
    report: Dict = {"stages": {}}
    for stage, pattern in sorted(STAGES.items()):
        per_paradigm: Dict[str, Dict] = {}
        for paradigm in sorted(discover_paradigms()):
            path = _results_path(paradigm, pattern)
            if path is None:
                print(f"  [WARN] {paradigm}/{stage}: results missing")
                continue
            segments = _fold_segments(path)
            if not segments:
                # A result predating the decomposition does not have the fields.
                # Reported, and not filled with zero, which would enter the sums
                # as if it were a measurement.
                print(f"  [WARN] {paradigm}/{stage}: {path.name} does not record "
                      f"the decomposition")
                continue
            load_total = sum(s["fold_load_s"] for s in segments)
            fit_total = sum(s["fit_predict_s"] for s in segments)
            total = load_total + fit_total
            per_paradigm[paradigm] = {
                "folds": len(segments),
                "fold_load_s": load_total,
                "fit_predict_s": fit_total,
                "total_s": total,
                # The fraction the engine controls is what makes the comparison
                # attributable; the rest is common to all three by construction.
                "engine_share": load_total / total if total > 0 else None,
                "per_fold": segments,
            }

        entry: Dict = {"paradigms": per_paradigm}

        # The ratios divide totals summed over each paradigm's folds. With
        # different counts -- a partial artifact, a fold that failed -- the
        # ratio compares the work of nine folds against that of eight, and the
        # 12% difference shows up as if it came from the engine. It is the whole
        # attribution that this file exists to perform.
        counts = {paradigm: values["folds"]
                  for paradigm, values in per_paradigm.items()}
        if len(set(counts.values())) > 1:
            raise ValueError(
                f"Stage '{stage}': the paradigms record different fold "
                f"counts {counts}. The ratios between them are not "
                f"attributable to the engine while that remains true."
            )

        if len(per_paradigm) >= 2:
            # Per-segment ratios: a gain in the total that does not appear in
            # fold_load does not come from the engine.
            baseline = min(per_paradigm, key=lambda p: per_paradigm[p]["total_s"])
            entry["fastest_total"] = baseline
            entry["ratios_against_fastest"] = {
                paradigm: {
                    segment: (values[segment] / per_paradigm[baseline][segment]
                              if per_paradigm[baseline][segment] > 0 else None)
                    for segment in ("fold_load_s", "fit_predict_s", "total_s")
                }
                for paradigm, values in per_paradigm.items()
            }
        report["stages"][stage] = entry
    return report


def _escape(text: str) -> str:
    """Paradigm and stage names carry an underscore, which LaTeX does not accept.

    Without this the generated file does not compile, and the error surfaces to
    whoever assembles the paper, not to whoever runs the pipeline.
    """
    return str(text).replace('_', r'\_')


def _latex(report: Dict) -> str:
    lines = [
        "% Hierarchical stage attribution — automatically generated",
        "\\begin{table}[htb]",
        "\\centering",
        "\\caption{Latency of the hierarchical stage decomposed into fold "
        "loading (engine) and fitting (common to the paradigms).}",
        "\\begin{tabular}{llrrrr}",
        "\\toprule",
        "Stage & Paradigm & Folds & Loading (s) & Fitting (s) & "
        "Engine share \\\\",
        "\\midrule",
    ]
    for stage, entry in sorted(report.get("stages", {}).items()):
        for paradigm, values in sorted(entry.get("paradigms", {}).items()):
            share = values["engine_share"]
            pct = '—' if share is None else f"{share * 100:.1f}\\%"
            lines.append(
                f"{_escape(stage)} & {_escape(paradigm)} & "
                f"{values['folds']} & "
                f"{values['fold_load_s']:.2f} & "
                f"{values['fit_predict_s']:.2f} & {pct} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(lines)


def main() -> int:
    report = attribute()
    if not any(e.get("paradigms") for e in report.get("stages", {}).values()):
        print("  No stage records the decomposition; nothing to attribute.")
        return 0

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "stage_attribution.json").write_text(
        json.dumps(report, indent=2))
    (OUT_DIR / "stage_attribution.tex").write_text(_latex(report))
    print(json.dumps({
        "status": "ok",
        "json": str(OUT_DIR / "stage_attribution.json"),
        "tex": str(OUT_DIR / "stage_attribution.tex"),
        "stages": {k: sorted(v.get("paradigms", {}))
                   for k, v in report["stages"].items()},
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
