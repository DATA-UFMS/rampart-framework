#!/usr/bin/env python3
"""
Testes de anti-leak para lags do target e gaps temporais por fold.
Requerem outputs pré-gerados (integration tests).
"""
import json
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

_DL_MASTER = _PROJECT_ROOT / 'outputs/ml_pipeline/architectures/task_graph/prep/master_data_task_graph.parquet'
_DW_FOLDS = _PROJECT_ROOT / 'outputs/ml_pipeline/architectures/sql_engine/prep/temporal_folds_sql_engine.json'


@pytest.mark.skipif(not _DL_MASTER.exists(), reason='Master DL não encontrado; rode setup DL')
def test_dl_lag2_anti_leak():
    import dask.dataframe as dd
    df = dd.read_parquet(str(_DL_MASTER))
    base_df = df[['country_code','year','dropout_rate_task_graph']].rename(columns={'dropout_rate_task_graph':'dropout_rate_t'})
    prev = base_df.assign(year=base_df['year']+2).rename(columns={'dropout_rate_t':'ref'})
    merged = df.merge(prev[['country_code','year','ref']], on=['country_code','year'], how='left')
    mask = ~merged['dropout_rate_lag_2'].isna()
    bad_df = merged[mask & merged['ref'].isna()].compute()
    assert len(bad_df) == 0, f'Encontrados {len(bad_df)} lag2 sem registro correspondente em t-2'


@pytest.mark.skipif(not _DW_FOLDS.exists(), reason='Folds não encontrados; rode pipeline')
def test_fold_gaps_structure():
    for arch in ['task_graph', 'sql_engine']:
        p = _PROJECT_ROOT / f'outputs/ml_pipeline/architectures/{arch}/prep/temporal_folds_{arch}.json'
        if not p.exists():
            pytest.skip(f'Folds {arch} não encontrados')
        conf = json.loads(p.read_text())
        folds = conf['folds']
        for f in folds:
            assert f['val_start'] - f['train_end'] - 1 >= 2, \
                f"Fold {f.get('fold_id','?')}: gap train-val < 2"
            assert f['test_start'] - f['val_end'] - 1 >= 2, \
                f"Fold {f.get('fold_id','?')}: gap val-test < 2"
