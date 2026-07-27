#!/usr/bin/env python3
"""Generates the paper's latency table from the artifacts, for both panels.

Why it exists: transcribing cell by cell is the mechanism by which a published
table stops matching the data without anything flagging it. This table is
derived, and carries in its caption the commit, the instant and the core budget
of each run -- every latency is conditional on them.

Runs after both pipelines, because it crosses datasets: the artifacts of each one
live under outputs/<dataset>/, and a table that compares scales needs both.

Two explicit decisions, which transcription hid:

  * The winner per stage is computed, not marked by hand.
  * The p column reports the **largest** value among the pairs, after Bonferroni
    over the entire family. A stage's claim is "the paradigms differ here", and
    it requires that *all* pairs differ -- the largest p is what bounds it. A
    single, unqualified p leaves it ambiguous which pair it describes.

Usage:
    python scripts/derive_paper_tables.py [--datasets worldbank inep_censo]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / 'src'))

from core.paradigm_registry import discover_paradigms  # noqa: E402

# The order of the phases in the table follows the pipeline, not alphabetical order.
STAGE_ORDER = ['processing', 'setup', 'baseline', 'hierarchical']
ALPHA = 0.05


def _dataset_root(dataset: str) -> Path:
    return _ROOT / 'outputs' / dataset


def _require_commit(commit: str, dataset: str) -> str:
    """A latency table without the commit that produced it is not comparable.

    `write_environment_snapshot` writes 'unavailable' when it cannot resolve
    the commit. The caption truncated to ten characters, so the published
    artifact said "at unavailabl" -- a meaningless string, in the exact place
    where the reader looks for provenance.
    """
    if not commit or commit == 'unavailable':
        raise ValueError(
            f"{dataset}: the snapshot does not record the commit "
            f"(git_commit={commit!r}). The latency table is conditional on the "
            f"code that produced it; without that it cannot be published. "
            f"Run the pipeline from a git clone with a clean tree."
        )
    return commit


def _read(dataset: str) -> Optional[Dict]:
    """Benchmark, significance and provenance of a dataset."""
    root = _dataset_root(dataset)
    benchmark = root / 'benchmarks' / 'architectural_benchmark_results.csv'
    significance = root / 'statistics' / 'significance_summary.csv'
    snapshot = root / 'scientific_config_snapshot.json'

    missing = [p.name for p in (benchmark, significance, snapshot)
               if not p.exists()]
    if missing:
        print(f"  [WARN] {dataset}: missing {missing}")
        return None

    provenance = json.loads(snapshot.read_text())
    config = provenance.get('scientific_config', {})
    return {
        'benchmark': pd.read_csv(benchmark),
        'significance': pd.read_csv(significance),
        # Without provenance the table is not published: a latency without the
        # commit and the budget that produced it is comparable to nothing.
        'commit': _require_commit(provenance['git_commit'], dataset),
        'timestamp': provenance['timestamp'],
        'engine_threads': config['engine_threads'],
        'blas_threads': config['blas_threads'],
    }


def _stage_rows(data: Dict, paradigms: List[str]) -> List[Dict]:
    benchmark = data['benchmark']
    significance = data['significance']
    # The family of comparisons is the entire set of reported tests, and not
    # the pairs of one stage: the threshold depends on it.
    family_size = len(significance)
    threshold = ALPHA / family_size if family_size else float('nan')

    rows = []
    for stage in STAGE_ORDER:
        subset = benchmark[benchmark['phase'] == stage]
        if subset.empty:
            continue
        cells = {}
        repetitions = {}
        for paradigm in paradigms:
            values = subset[subset['architecture'] == paradigm]['duration_s']
            if values.empty:
                continue
            # Counted, not assumed. mean() and std() skip missing values, so a
            # paradigm with fewer repetitions produces a cell that looks just
            # like the others -- a mean over a smaller n, presented next to
            # means over a larger n, with nothing in the table saying so.
            repetitions[paradigm] = int(values.notna().sum())
            cells[paradigm] = (float(values.mean()), float(values.std(ddof=1)))
        if not cells:
            continue

        if len(set(repetitions.values())) > 1:
            raise ValueError(
                f"Stage '{stage}': the paradigms have different repetition "
                f"counts {repetitions}. The means are not comparable, and the "
                f"table would present them side by side as if they were."
            )

        stage_tests = significance[significance['phase'] == stage]
        # Floor of the two-sided signed-rank: 2/2^n. The n is that of the
        # non-zero differences, not that of the pairs -- the test discards ties,
        # and the floor computed over the pairs underestimates the smallest
        # attainable p. With three ties out of ten it is off by a factor of eight.
        if not stage_tests.empty:
            if 'n_nonzero_diffs' not in stage_tests.columns:
                raise ValueError(
                    f"Stage '{stage}': the significance summary does not carry "
                    f"n_nonzero_diffs. It is an artifact predating this column, "
                    f"and the floor derived from the number of pairs "
                    f"underestimates the smallest attainable p. Regenerate the "
                    f"summary."
                )
            n_pairs = int(stage_tests['n_nonzero_diffs'].min())
        else:
            n_pairs = 0

        # The two artifacts have to come from the same run. The n of the
        # significance summary is the number of pairs; if it disagrees with the
        # repetitions in the CSV, one of the two fell behind.
        if not stage_tests.empty and 'n' in stage_tests.columns:
            declared = set(int(value) for value in stage_tests['n'].unique())
            measured = set(repetitions.values())
            if declared != measured:
                raise ValueError(
                    f"Stage '{stage}': the significance summary declares "
                    f"n={sorted(declared)} and the benchmark CSV has "
                    f"{sorted(measured)} repetitions per paradigm. The two "
                    f"artifacts do not come from the same run."
                )

        p_column = ('wilcoxon_p' if 'wilcoxon_p' in stage_tests.columns
                    else 't_p')
        # The largest p among the pairs bounds the stage's claim: it is "the
        # paradigms differ here", and that requires that *all* pairs differ.
        #
        # skipna=False on purpose. A pair whose test could not be computed
        # -- all differences zero, n below the minimum -- vanished from the
        # maximum, and the stage came out more significant than the family
        # supports. Without that pair's test the stage's claim is not established.
        worst_p = (float(stage_tests[p_column].max(skipna=False))
                   if not stage_tests.empty else float('nan'))
        untested = (int(stage_tests[p_column].isna().sum())
                    if not stage_tests.empty else 0)

        # Floor and threshold are independent: the floor comes from the
        # repetitions, the threshold comes from the family size, which grows
        # with the number of paradigms. With a fourth paradigm the family goes
        # from 15 to 30 and the threshold falls below the floor -- no stage can
        # be significant, whatever the data. Passing silently in that condition
        # is reporting an absence of difference when what there was was an
        # absence of resolution.
        floor = (2.0 / 2 ** n_pairs) if n_pairs else float('nan')
        if n_pairs and floor > threshold:
            raise ValueError(
                f"Stage '{stage}': the floor of the two-sided Wilcoxon with "
                f"{n_pairs} pairs is {floor:.5f}, above the corrected threshold "
                f"{threshold:.5f} (alpha={ALPHA} over a family of "
                f"{family_size}). No stage can be significant in this "
                f"configuration. Raise the repetitions to "
                f"{math.ceil(math.log2(2.0 / threshold))} or more, or reduce the "
                f"family."
            )
        rows.append({
            'stage': stage,
            'cells': cells,
            'winner': min(cells, key=lambda p: cells[p][0]),
            'pairs_tested': int(len(stage_tests)),
            'worst_pair_p': worst_p,
            'family_size': int(family_size),
            'n_observations': n_pairs,
            'wilcoxon_floor': floor,
            'threshold': threshold,
            # A single formulation: raw p against alpha/m. Reporting the
            # multiplied p as well would invite comparing it with 0.05 out of
            # habit, and the two readings mixed is how a cell comes to say two
            # things.
            'pairs_untested': untested,
            'repetitions': int(next(iter(repetitions.values()))),
            'p_bonferroni_equivalent': min(1.0, worst_p * family_size)
                                       if family_size else worst_p,
            # NaN < x is False, so a stage with an untested pair already came
            # out non-significant -- but by accident of the comparison, and
            # after the maximum had hidden the pair. Explicit now.
            'significant': bool(np.isfinite(worst_p) and worst_p < threshold),
        })
    return rows


def _fmt(mean: float, sd: float, winner: bool) -> str:
    body = f"{mean:.3g}{{\\tiny$\\pm${sd:.3g}}}"
    return f"\\textbf{{{body}}}" if winner else body


def build(datasets: List[str]) -> Dict:
    paradigms = sorted(discover_paradigms())
    report = {'paradigms': paradigms, 'datasets': {}}
    for dataset in datasets:
        data = _read(dataset)
        if data is None:
            continue
        report['datasets'][dataset] = {
            'commit': data['commit'],
            'timestamp': data['timestamp'],
            'engine_threads': data['engine_threads'],
            'blas_threads': data['blas_threads'],
            'stages': _stage_rows(data, paradigms),
        }
    return report


def to_latex(report: Dict) -> str:
    paradigms = report['paradigms']
    present = report['datasets']
    if not present:
        return ''

    budgets = {(d['engine_threads'], d['blas_threads'])
               for d in present.values()}
    if len(budgets) > 1:
        raise ValueError(
            f"The panels were measured with different core budgets "
            f"{budgets}: the latencies are not comparable across them."
        )
    engine, blas = budgets.pop()
    thresholds = {row['threshold'] for d in present.values()
                  for row in d['stages']}
    threshold_text = ('/'.join(f'{t:.5f}' for t in sorted(thresholds))
                      if thresholds else 'n/a')
    provenance = '; '.join(
        f"{name.replace('_', chr(92) + '_')} at {d['commit'][:10]} "
        f"({d['timestamp'][:19]})"
        for name, d in sorted(present.items()))

    lines = [
        '% Generated by scripts/derive_paper_tables.py -- do not edit by hand',
        '\\begin{table}[htb]',
        '\\centering',
        '\\caption{Latency per stage (mean $\\pm$ SD, seconds). '
        '\\textbf{Bold}: lowest mean of the stage, computed. '
        '$p$: largest value among the pairs (Wilcoxon), against the Bonferroni '
        f'threshold {threshold_text}. '
        f'{engine} cores per engine, {blas} BLAS thread. '
        f'Provenance: {provenance}.}}',
        '\\label{tab:latency}',
        '\\begin{tabular}{ll' + 'r' * len(paradigms) + 'l}',
        '\\toprule',
        'Panel & Stage & ' + ' & '.join(
            p.replace('_', r'\_') for p in paradigms) + ' & $p$ \\\\',
        '\\midrule',
    ]
    for dataset, data in sorted(present.items()):
        stages = data['stages']
        for index, row in enumerate(stages):
            # Escaping is mandatory: a dataset name with an underscore breaks
            # compilation, and the table is generated, not reviewed by hand.
            safe = dataset.replace('_', r'\_')
            label = (f"\\multirow{{{len(stages)}}}{{*}}{{{safe}}}"
                     if index == 0 else '')
            cells = [
                _fmt(*row['cells'][p], p == row['winner'])
                if p in row['cells'] else '---'
                for p in paradigms
            ]
            floor = row['wilcoxon_floor']
            at_floor = floor == floor and row['worst_pair_p'] <= floor
            if row['significant']:
                # At the test's floor: report "=" to the floor instead of digits
                # that suggest precision n does not offer.
                marker = (f"{floor:.5f} (floor, $n$={row['n_observations']})"
                          if at_floor else f"{row['worst_pair_p']:.4f}")
            else:
                # Explicit instead of omitted: a p above the corrected threshold
                # does not support the difference, and the cell's bold face must
                # not suggest that it does.
                marker = f"{row['worst_pair_p']:.4f} (n.s.)"
            lines.append(f"{label} & {row['stage']} & "
                         + ' & '.join(cells) + f" & {marker} \\\\")
        lines.append('\\midrule')
    lines[-1] = '\\bottomrule'
    lines += ['\\end{tabular}', '\\end{table}']
    return '\n'.join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--datasets', nargs='+',
                        default=['worldbank', 'inep_censo'])
    parser.add_argument('--out-dir', default=str(_ROOT / 'outputs' / 'tables'))
    args = parser.parse_args(argv)

    report = build(args.datasets)
    if not report['datasets']:
        print("  No panel with complete artifacts; nothing to generate.")
        return 0

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'latency_table.json').write_text(json.dumps(report, indent=2))
    (out / 'latency_table.tex').write_text(to_latex(report))
    print(json.dumps({
        'status': 'ok',
        'datasets': sorted(report['datasets']),
        'tex': str(out / 'latency_table.tex'),
        'json': str(out / 'latency_table.json'),
    }, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
