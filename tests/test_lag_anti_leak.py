#!/usr/bin/env python3
"""
Testes de anti-leak para lags do target e gaps temporais por fold.
"""
import json
import os
from pathlib import Path


def test_dl_lag2_anti_leak():
    base = Path('outputs/ml_pipeline/architectures/data_lake/prep')
    master = base / 'master_data_data_lake.parquet'
    assert master.exists(), 'Master DL não encontrado; rode setup DL'
    try:
        import dask.dataframe as dd
        df = dd.read_parquet(str(master))
        # Para cada linha com lag2, deve existir a linha (country, year-2)
        base = df[['country_code','year','dropout_rate_data_lake']].rename(columns={'dropout_rate_data_lake':'dropout_rate_t'})
        prev = base.assign(year=base['year']+2).rename(columns={'dropout_rate_t':'ref'})
        merged = df.merge(prev[['country_code','year','ref']], on=['country_code','year'], how='left')
        # Onde dropout_rate_lag_2 não é nulo, ref não deve ser nulo
        mask = ~merged['dropout_rate_lag_2'].isna()
        bad_df = merged[mask & merged['ref'].isna()].compute()
        assert len(bad_df) == 0, 'Encontrados lag2 sem registro correspondente em t-2'
    except Exception as e:
        raise AssertionError(f'Falha no teste DL lag2: {e}')


def test_fold_gaps_structure():
    # Verifica gaps ≥2 anos nos folds DL e DW
    for arch in ['data_lake','data_warehouse']:
        p = Path(f'outputs/ml_pipeline/architectures/{arch}/prep/temporal_folds_{arch}.json')
        assert p.exists(), f'Folds {arch} não encontrados'
        conf = json.loads(p.read_text())
        folds = conf['folds']
        for f in folds:
            assert f['val_start'] - f['train_end'] - 1 >= 2
            assert f['test_start'] - f['val_end'] - 1 >= 2
