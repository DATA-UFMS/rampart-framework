#!/usr/bin/env python3
"""
Summary of resource usage per phase (means/peaks) from the JSONL log.

Input:
  - outputs/benchmarks/architectural_benchmark_resource_log.jsonl

Outputs:
  - outputs/statistics/architectural_resource_usage.json
  - outputs/statistics/architectural_resource_usage.tex

Columns in the (LaTeX) table:
  Phase, Arch, CPU(proc) mean, CPU(proc) peak, RSS(MB) mean, RSS(MB) peak,
  CPU(sys) mean, CPU(sys) peak, Mem(sys)% mean, Mem(sys)% peak, IO Read(MB), IO Write(MB), n

If there is no data, generates a minimal table with "No data".
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd


import os
import sys

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_SRC_DIR = os.path.join(_BASE_DIR, "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from core.config import get_absolute_output_path

LOG = Path(get_absolute_output_path(
    "outputs/benchmarks/architectural_benchmark_resource_log.jsonl"))
OUT_DIR = Path(get_absolute_output_path("outputs/statistics"))
OUT_JSON = OUT_DIR / "architectural_resource_usage.json"
OUT_TEX = OUT_DIR / "architectural_resource_usage.tex"


def _garantir_dir():
    OUT_DIR.mkdir(parents=True, exist_ok=True)


def _load_jsonl(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            # Flattens the nested fields
            flat_obj = {
                'run_id': obj.get('run_id'),
                'phase': obj.get('phase'),
                'architecture': obj.get('architecture'),
                'step': obj.get('step'),
                'cpu_proc_mean': obj.get('cpu_proc', {}).get('mean'),
                'cpu_proc_max': obj.get('cpu_proc', {}).get('max'),
                'cpu_sys_mean': obj.get('cpu_sys', {}).get('mean'),
                'cpu_sys_max': obj.get('cpu_sys', {}).get('max'),
                'rss_mb_mean': obj.get('rss_mb', {}).get('mean'),
                'rss_mb_max': obj.get('rss_mb', {}).get('max'),
                'mem_sys_mean': obj.get('mem_sys_percent', {}).get('mean'),
                'mem_sys_max': obj.get('mem_sys_percent', {}).get('max'),
                'io_read_mb': obj.get('io_read_mb', 0),
                'io_write_mb': obj.get('io_write_mb', 0),
                'n': obj.get('cpu_proc', {}).get('n', 0),
                'is_warmup': obj.get('is_warmup'),
            }
            rows.append(flat_obj)
        except Exception as e:
            print(f"Error processing line: {e}")
            continue
    
    if not rows:
        return pd.DataFrame()

    frame = pd.DataFrame(rows)
    # A record without the field predates the distinction, and there is no way
    # to know which side it falls on. Filling it with False would treat it as a
    # measurement.
    unmarked = int(frame['is_warmup'].isna().sum())
    if unmarked:
        raise ValueError(
            f"{unmarked} of {len(frame)} resource records do not say whether "
            f"they are warmup ({path}). They come from a run that predates the "
            f"distinction; the log is truncated at the start of every "
            f"benchmark, so this means the file did not come from this run."
        )
    return frame[~frame['is_warmup'].astype(bool)].drop(columns=['is_warmup'])


def _escape_latex(text: str) -> str:
    if text is None:
        return ""
    return text.replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")


def _fmt(x: float | None, unit: str = "") -> str:
    if x is None or not np.isfinite(x):
        return "—"
    
    if unit == "%":
        return f"{x:.1f}\\%"
    elif unit == "MB":
        return f"{x:.1f}"
    else:
        return f"{x:.1f}{unit}"


def resumir(df: pd.DataFrame) -> Dict:
    if df.empty:
        return {"erro": "sem_dados"}
    
    grouped = df.groupby(['phase', 'architecture'])
    metrics = grouped.agg({
        'cpu_proc_mean': 'mean',
        'cpu_proc_max': 'max',
        'rss_mb_mean': 'mean',
        'rss_mb_max': 'max',
        'cpu_sys_mean': 'mean',
        'cpu_sys_max': 'max',
        'mem_sys_mean': 'mean',
        'mem_sys_max': 'max',
        'io_read_mb': 'sum',
        'io_write_mb': 'sum',
        'n': 'count'
    }).reset_index()
    
    result = {"per_phase": {}}
    
    for _, row in metrics.iterrows():
        phase = row['phase']
        arch = row['architecture']
        
        if phase not in result["per_phase"]:
            result["per_phase"][phase] = {}
            
        result["per_phase"][phase][arch] = {
            'cpu_proc_mean': row.get('cpu_proc_mean'),
            'cpu_proc_max': row.get('cpu_proc_max'),
            'rss_mb_mean': row.get('rss_mb_mean'),
            'rss_mb_max': row.get('rss_mb_max'),
            'cpu_sys_mean': row.get('cpu_sys_mean'),
            'cpu_sys_max': row.get('cpu_sys_max'),
            'mem_sys_mean': row.get('mem_sys_mean'),
            'mem_sys_max': row.get('mem_sys_max'),
            'io_read_mb': row.get('io_read_mb'),
            'io_write_mb': row.get('io_write_mb'),
            'n': row.get('n', 0)
        }
    
    return result


def para_latex(resumo: Dict) -> str:

    wrapper_prefix = [
        r"\begingroup",
        r"\setlength{\tabcolsep}{4pt}",
        r"\scriptsize",
        r"\begin{center}",
        r"\begin{tabular}{@{}l l r r r r r r r r r r r@{}}",  # 13 columns
        r"\hline",
        r"\multicolumn{13}{c}{\textbf{Resource Usage per Phase}} \\",
        r"\hline",
        r"Phase & Arch & \multicolumn{2}{c}{CPU (\%)} & \multicolumn{2}{c}{RSS (MB)} & \multicolumn{2}{c}{CPU (s)} & \multicolumn{2}{c}{Mem (\%)} & \multicolumn{2}{c}{I/O (MB)} & n \\",
        r"\cline{3-4} \cline{5-6} \cline{7-8} \cline{9-10} \cline{11-12}",
        r" & & $\mu$ & max & $\mu$ & max & $\mu$ & max & $\mu$ & max & R & W & \\",
        r"\hline"
    ]
    

    wrapper_suffix = [
        r"\hline",
        r"\end{tabular}",
        r"\end{center}",
        r"\endgroup"
    ]

    if not resumo or "per_phase" not in resumo or not resumo["per_phase"]:
        linhas_erro = wrapper_prefix + [
            r"No data & -- & -- & -- & -- & -- & -- & -- & -- & -- & -- & -- & -- \\",
        ] + wrapper_suffix
        return "\n".join(linhas_erro) + "\n"

    # Builds the data rows
    linhas = [
        "% Automatically generated by derive_resource_usage_table.py",
        "% Resource usage per phase and architecture",
        "",
    ]
    linhas.extend(wrapper_prefix)


    for fase in sorted(resumo.get('per_phase', {}).keys()):
        por_arq = resumo['per_phase'][fase]
        for arq in sorted(por_arq.keys()):
            r = por_arq[arq]
            
            fase_latex = _escape_latex(fase)
            arq_latex = _escape_latex(arq)
            
           
            linha = (
                f"{fase_latex} & {arq_latex}"
                f" & {_fmt(r.get('cpu_proc_mean'), '%')}"
                f" & {_fmt(r.get('cpu_proc_max'), '%')}"
                f" & {_fmt(r.get('rss_mb_mean'))}"
                f" & {_fmt(r.get('rss_mb_max'))}"
                f" & {_fmt(r.get('cpu_sys_mean'), '%')}"
                f" & {_fmt(r.get('cpu_sys_max'), '%')}"
                f" & {_fmt(r.get('mem_sys_mean'), '%')}"
                f" & {_fmt(r.get('mem_sys_max'), '%')}"
                f" & {_fmt(r.get('io_read_mb'))}"
                f" & {_fmt(r.get('io_write_mb'))}"
                f" & {int(r.get('n', 0))}"
                f" \\\\"  
            )
            linhas.append(linha)

    linhas.extend(wrapper_suffix)
    return "\n".join(linhas) + "\n"


def main() -> None:
    _garantir_dir()
    
    if not LOG.exists():
        msg = {
            "erro": f"Log file not found: {str(LOG)}",
            "dica": "Run the architectural benchmark to generate the resource log.",
        }
        OUT_JSON.write_text(json.dumps(msg, indent=2, ensure_ascii=False), encoding="utf-8")
        OUT_TEX.write_text("% No data\n" + para_latex({}), encoding="utf-8")
        print(json.dumps(msg, indent=2, ensure_ascii=False))
        return
    
    try:
        df = _load_jsonl(LOG)
        if df.empty:
            msg = {"erro": "No valid data found in the log file."}
            OUT_JSON.write_text(json.dumps(msg, indent=2, ensure_ascii=False), encoding="utf-8")
            OUT_TEX.write_text("% No data\n" + para_latex({}), encoding="utf-8")
            print(json.dumps(msg, indent=2, ensure_ascii=False))
            return
        

        resumo = resumir(df)
        

        OUT_JSON.write_text(json.dumps(resumo, indent=2, ensure_ascii=False), encoding="utf-8")
        

        latex_content = para_latex(resumo)
        OUT_TEX.write_text(latex_content, encoding="utf-8")
        

        print(json.dumps({
            "status": "ok", 
            "json": str(OUT_JSON), 
            "tex": str(OUT_TEX), 
            "rows": len(df)
        }, indent=2, ensure_ascii=False))
        
    except Exception as e:
        erro_msg = {
            "erro": f"Processing failure: {str(e)}",
            "arquivo": str(LOG)
        }
        OUT_JSON.write_text(json.dumps(erro_msg, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(erro_msg, indent=2, ensure_ascii=False))
        raise


if __name__ == "__main__":
    main()