#!/usr/bin/env python3
"""
Teste de injeção de leakage: valida que o gate anti-leakage detecta
violações deliberadas de integridade temporal.

Três cenários de injeção:
  S1 – Gap insuficiente (gap=0 entre train e val)
  S2 – Sobreposição temporal (anos de treino aparecem no teste)
  S3 – Ordem temporal invertida (test_start < train_end)

Para cada cenário, o teste verifica que:
  (a) TemporalValidator.enforce_walk_forward() levanta ValueError
  (b) A mensagem de erro inclui diagnóstico específico da violação

Adicionalmente, S4 roda um experimento empírico comparando métricas
preditivas em configuração limpa (walk-forward, gap=2) vs contaminada
(k-fold naive sem respeitar ordem temporal), quantificando a inflação
de métricas causada por leakage temporal.

Uso:
    python tests/test_leakage_injection.py          # roda todos os cenários
    python tests/test_leakage_injection.py --quick   # só cenários S1-S3 (sem dados)
"""
import argparse
import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / 'src'))

from core.validation import TemporalValidator
from core.scientific_config import SCIENTIFIC_CONFIG


# ---------------------------------------------------------------------------
# Folds válidos (baseline do paper: 9 walk-forward folds, gap=2)
# ---------------------------------------------------------------------------
def generate_valid_folds():
    """Gera folds walk-forward válidos usando a mesma lógica do framework."""
    cfg = SCIENTIFIC_CONFIG
    start_year = cfg.get('temporal_range_start', 2000)
    end_year = cfg.get('temporal_range_end', 2023)
    min_train = cfg.get('folds_min_train_years', 8)
    val_len = cfg.get('folds_val_len_years', 2)
    test_len = cfg.get('folds_test_len_years', 2)
    gap = cfg.get('temporal_gap_years', 2)
    step = cfg.get('folds_step_years', 1)

    test_start_min = start_year + min_train + val_len + 2 * gap
    test_start_max = end_year - test_len + 1

    folds = []
    for fold_id, test_start in enumerate(
        range(test_start_min, test_start_max + 1, step)
    ):
        test_end = test_start + test_len - 1
        val_end = test_start - gap - 1
        val_start = val_end - val_len + 1
        train_end = val_start - gap - 1
        train_start = start_year

        folds.append({
            'fold_id': fold_id,
            'train_start': train_start, 'train_end': train_end,
            'val_start': val_start, 'val_end': val_end,
            'test_start': test_start, 'test_end': test_end,
        })
    return folds


# ---------------------------------------------------------------------------
# S1: Gap insuficiente (gap=0 entre train-val)
# ---------------------------------------------------------------------------
def inject_s1_zero_gap(folds):
    """Remove gaps temporais: val_start = train_end + 1 (gap efetivo = 0)."""
    contaminated = []
    for f in folds:
        c = dict(f)
        c['val_start'] = c['train_end'] + 1
        c['val_end'] = c['val_start'] + 1
        c['test_start'] = c['val_end'] + 1
        c['test_end'] = c['test_start'] + 1
        contaminated.append(c)
    return contaminated


# ---------------------------------------------------------------------------
# S2: Sobreposição temporal (anos de treino no teste)
# ---------------------------------------------------------------------------
def inject_s2_temporal_overlap(folds):
    """Faz test_start cair dentro do período de treino."""
    contaminated = []
    for f in folds:
        c = dict(f)
        c['test_start'] = c['train_start'] + 2
        c['test_end'] = c['test_start'] + 1
        contaminated.append(c)
    return contaminated


# ---------------------------------------------------------------------------
# S3: Ordem invertida (test antes de val)
# ---------------------------------------------------------------------------
def inject_s3_reversed_order(folds):
    """Inverte test e val: test_start < val_start."""
    contaminated = []
    for f in folds:
        c = dict(f)
        c['val_start'], c['test_start'] = c['test_start'], c['val_start']
        c['val_end'], c['test_end'] = c['test_end'], c['val_end']
        contaminated.append(c)
    return contaminated


# ---------------------------------------------------------------------------
# Runner dos cenários de injeção
# ---------------------------------------------------------------------------
def run_injection_scenario(name, description, contaminated_folds, validator):
    """Executa um cenário e verifica que o gate bloqueia."""
    print(f"\n--- Cenario {name}: {description} ---")

    f0 = contaminated_folds[0]
    print(f"  Fold 0 contaminado: train=[{f0['train_start']},{f0['train_end']}] "
          f"val=[{f0['val_start']},{f0['val_end']}] "
          f"test=[{f0['test_start']},{f0['test_end']}]")

    try:
        validator.enforce_walk_forward(contaminated_folds)
        print(f"  FALHA: Gate nao detectou a violacao!")
        return False
    except ValueError as e:
        print(f"  Gate bloqueou: {str(e)[:200]}")
        return True


# ---------------------------------------------------------------------------
# S4: Experimento empírico — naive k-fold vs walk-forward
# ---------------------------------------------------------------------------
def run_s4_empirical_comparison():
    """
    Compara métricas preditivas entre:
      - Configuração limpa: walk-forward temporal (gap=2 anos)
      - Configuração contaminada: k-fold naive (sem respeitar tempo)

    Usa dados sintéticos com estrutura temporal realista para
    demonstrar inflação de métricas causada por leakage.
    """
    import numpy as np
    import pandas as pd
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import KFold
    from sklearn.metrics import r2_score, mean_absolute_error

    np.random.seed(SCIENTIFIC_CONFIG.get('random_seed', 42))

    print(f"\n--- Cenario S4: Experimento empirico - leakage vs clean ---")

    n_countries = 32
    years = list(range(2000, 2024))
    n_years = len(years)

    rows = []
    for c in range(n_countries):
        base_level = np.random.uniform(15, 55)
        # Regime shifts: tendência muda a cada ~8 anos
        trends = [
            np.random.uniform(-2.0, -0.5),   # 2000-2007: melhora
            np.random.uniform(-0.5, 1.5),     # 2008-2015: estagnação/piora
            np.random.uniform(-3.0, -1.0),    # 2016-2023: melhora forte
        ]
        for y_idx, y in enumerate(years):
            regime = min(y_idx // 8, 2)
            trend = trends[regime]

            # Features observáveis (ruidosas, correlação parcial)
            enrollment = 70 + trend * (y_idx % 8) + np.random.normal(0, 8)
            expenditure = 3.5 + np.random.normal(0, 1.2)
            completion = 100 - base_level + trend * y_idx + np.random.normal(0, 6)

            # Target: dropout com regime shifts + ruído substancial
            dropout = base_level + trend * y_idx + np.random.normal(0, 5)
            dropout = max(0, min(100, dropout))

            # target + ruído pequeno (proxy quase perfeito)
            future_leak = dropout + np.random.normal(0, 0.3)

            rows.append({
                'country': c, 'year': y,
                'enrollment': enrollment,
                'expenditure': expenditure,
                'completion_rate': completion,
                'dropout_lag1': np.nan,
                'dropout_lag2': np.nan,
                'future_leak': future_leak,
                'dropout_rate': dropout,
            })

    df = pd.DataFrame(rows)

    # Preencher lags corretamente (sem leakage)
    df = df.sort_values(['country', 'year'])
    df['dropout_lag1'] = df.groupby('country')['dropout_rate'].shift(1)
    df['dropout_lag2'] = df.groupby('country')['dropout_rate'].shift(2)
    df = df.dropna().reset_index(drop=True)

    clean_features = ['enrollment', 'expenditure', 'completion_rate', 'dropout_lag1', 'dropout_lag2']
    leaked_features = clean_features + ['future_leak']
    target = 'dropout_rate'

    print("\n  [CLEAN] Walk-forward temporal, gap=2 anos, sem future features")
    clean_r2s = []
    clean_maes = []
    valid_folds = generate_valid_folds()

    for fold in valid_folds:
        train_mask = (df['year'] >= fold['train_start']) & (df['year'] <= fold['train_end'])
        test_mask = (df['year'] >= fold['test_start']) & (df['year'] <= fold['test_end'])

        train_df = df[train_mask]
        test_df = df[test_mask]

        if len(train_df) < 10 or len(test_df) < 5:
            continue

        X_train = train_df[clean_features].values
        y_train = train_df[target].values
        X_test = test_df[clean_features].values
        y_test = test_df[target].values

        model = Ridge(alpha=1.0, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        clean_r2s.append(r2_score(y_test, y_pred))
        clean_maes.append(mean_absolute_error(y_test, y_pred))

    print("  [LEAKED] K-fold naive (ignora tempo) + future-derived features")
    leaked_r2s = []
    leaked_maes = []

    kf = KFold(n_splits=9, shuffle=True, random_state=42)
    X_all = df[leaked_features].values
    y_all = df[target].values

    for train_idx, test_idx in kf.split(X_all):
        X_train = X_all[train_idx]
        y_train = y_all[train_idx]
        X_test = X_all[test_idx]
        y_test = y_all[test_idx]

        model = Ridge(alpha=1.0, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        leaked_r2s.append(r2_score(y_test, y_pred))
        leaked_maes.append(mean_absolute_error(y_test, y_pred))

    clean_r2_mean = np.mean(clean_r2s)
    clean_mae_mean = np.mean(clean_maes)
    leaked_r2_mean = np.mean(leaked_r2s)
    leaked_mae_mean = np.mean(leaked_maes)

    r2_inflation = leaked_r2_mean - clean_r2_mean
    mae_deflation = clean_mae_mean - leaked_mae_mean

    print(f"\n  Resultados ({len(clean_r2s)} folds clean, {len(leaked_r2s)} folds leaked):")
    print(f"  {'Métrica':<12} {'Clean (WF)':>12} {'Leaked (KF)':>12} {'Diferença':>12}")
    print(f"  {'-'*48}")
    print(f"  {'R²':<12} {clean_r2_mean:>12.4f} {leaked_r2_mean:>12.4f} {r2_inflation:>+12.4f}")
    print(f"  {'MAE':<12} {clean_mae_mean:>12.4f} {leaked_mae_mean:>12.4f} {-mae_deflation:>+12.4f}")

    print(f"\n  Conclusão: Leakage temporal inflou R² em {r2_inflation:+.4f} pontos")
    print(f"  ({r2_inflation/max(abs(clean_r2_mean), 1e-9)*100:+.1f}% relativo ao baseline limpo)")

    results = {
        'clean_walk_forward': {
            'r2_mean': round(clean_r2_mean, 6),
            'r2_std': round(np.std(clean_r2s), 6),
            'mae_mean': round(clean_mae_mean, 6),
            'mae_std': round(np.std(clean_maes), 6),
            'n_folds': len(clean_r2s),
        },
        'leaked_naive_kfold': {
            'r2_mean': round(leaked_r2_mean, 6),
            'r2_std': round(np.std(leaked_r2s), 6),
            'mae_mean': round(leaked_mae_mean, 6),
            'mae_std': round(np.std(leaked_maes), 6),
            'n_folds': len(leaked_r2s),
        },
        'inflation': {
            'r2_absolute': round(r2_inflation, 6),
            'r2_relative_pct': round(r2_inflation / max(abs(clean_r2_mean), 1e-9) * 100, 2),
            'mae_reduction': round(mae_deflation, 6),
        }
    }

    out_path = _PROJECT_ROOT / 'outputs'
    out_path.mkdir(parents=True, exist_ok=True)
    results_file = out_path / 'leakage_injection_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Resultados salvos em: {results_file}")

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Teste de injeção de leakage")
    parser.add_argument('--quick', action='store_true',
                        help='Roda apenas S1-S3 (sem experimento empírico)')
    args = parser.parse_args()

    print("Teste de injecao de leakage")
    print("Validacao negativa do gate anti-leakage")

    valid_folds = generate_valid_folds()
    validator = TemporalValidator(min_gap_years=2)

    print(f"\n  Baseline: {len(valid_folds)} folds walk-forward válidos")
    try:
        validator.enforce_walk_forward(valid_folds)
        print("  Gate PASSOU folds válidos (esperado)")
        baseline_ok = True
    except ValueError:
        print("  FALHA: Gate rejeitou folds válidos!")
        baseline_ok = False

    scenarios = [
        ("S1", "Gap insuficiente (gap=0)", inject_s1_zero_gap),
        ("S2", "Sobreposição temporal (test dentro do train)", inject_s2_temporal_overlap),
        ("S3", "Ordem invertida (test antes de val)", inject_s3_reversed_order),
    ]

    results = {'baseline_valid': baseline_ok, 'scenarios': {}}

    for name, desc, injector in scenarios:
        contaminated = injector(valid_folds)
        detected = run_injection_scenario(name, desc, contaminated, validator)
        results['scenarios'][name] = {
            'description': desc,
            'leakage_detected': detected,
        }

    # Sumário S1-S3
    all_detected = all(s['leakage_detected'] for s in results['scenarios'].values())
    print(f"\nSumario: {'Todos os cenarios detectados' if all_detected else 'Falha em algum cenario'}")
    print(f"  Baseline valido: {baseline_ok}")
    for name, s in results['scenarios'].items():
        status = "DETECTADO" if s['leakage_detected'] else "FALHOU"
        print(f"  {name}: {status} — {s['description']}")

    if not args.quick:
        s4_results = run_s4_empirical_comparison()
        results['s4_empirical'] = s4_results

    out_path = _PROJECT_ROOT / 'outputs'
    out_path.mkdir(parents=True, exist_ok=True)
    report_file = out_path / 'leakage_injection_report.json'
    with open(report_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Relatório salvo em: {report_file}")

    if baseline_ok and all_detected:
        print("\nResultado: OK - Gate anti-leakage funcional")
        sys.exit(0)
    else:
        print("\nResultado: FALHA")
        sys.exit(1)


if __name__ == '__main__':
    main()
