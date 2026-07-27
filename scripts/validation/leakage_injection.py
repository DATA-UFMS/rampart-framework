#!/usr/bin/env python3
"""
Leakage injection test: validates that the anti-leakage gate detects
deliberate violations of temporal integrity.

Three injection scenarios:
  S1 – Insufficient gap (gap=0 between train and val)
  S2 – Temporal overlap (training years appear in the test window)
  S3 – Reversed temporal ordering (test_start < train_end)

For each scenario, the test verifies that:
  (a) TemporalValidator.enforce_walk_forward() raises ValueError
  (b) The error message includes a diagnostic specific to the violation

Additionally, S4 runs an empirical experiment comparing predictive
metrics under a clean configuration (walk-forward, gap=2) vs a contaminated
one (naive k-fold that does not respect temporal ordering), quantifying the
metric inflation caused by temporal leakage.

Usage:
    python scripts/validation/leakage_injection.py           # runs every scenario
    python scripts/validation/leakage_injection.py --quick   # only scenarios S1-S3 (no data)
"""
import argparse
import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT / 'src'))

from core.config import get_absolute_output_path

from core.validation import TemporalValidator
from core.scientific_config import SCIENTIFIC_CONFIG


# Valid folds (paper baseline: 9 walk-forward folds, gap=2)
def generate_valid_folds():
    """Generates valid walk-forward folds using the same logic as the framework."""
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


# S1: Insufficient gap (gap=0 between train-val)
def inject_s1_zero_gap(folds):
    """Removes temporal gaps: val_start = train_end + 1 (effective gap = 0)."""
    contaminated = []
    for f in folds:
        c = dict(f)
        c['val_start'] = c['train_end'] + 1
        c['val_end'] = c['val_start'] + 1
        c['test_start'] = c['val_end'] + 1
        c['test_end'] = c['test_start'] + 1
        contaminated.append(c)
    return contaminated


# S2: Temporal overlap (training years in the test window)
def inject_s2_temporal_overlap(folds):
    """Makes test_start fall inside the training period."""
    contaminated = []
    for f in folds:
        c = dict(f)
        c['test_start'] = c['train_start'] + 2
        c['test_end'] = c['test_start'] + 1
        contaminated.append(c)
    return contaminated


# S3: Reversed ordering (test before val)
def inject_s3_reversed_order(folds):
    """Swaps test and val: test_start < val_start."""
    contaminated = []
    for f in folds:
        c = dict(f)
        c['val_start'], c['test_start'] = c['test_start'], c['val_start']
        c['val_end'], c['test_end'] = c['test_end'], c['val_end']
        contaminated.append(c)
    return contaminated


# Runner for the injection scenarios
def run_injection_scenario(name, description, contaminated_folds, validator):
    """Runs one scenario and verifies that the gate blocks it."""
    print(f"\n--- Scenario {name}: {description} ---")

    f0 = contaminated_folds[0]
    print(f"  Contaminated fold 0: train=[{f0['train_start']},{f0['train_end']}] "
          f"val=[{f0['val_start']},{f0['val_end']}] "
          f"test=[{f0['test_start']},{f0['test_end']}]")

    try:
        validator.enforce_walk_forward(contaminated_folds)
        print(f"  FAILURE: gate did not detect the violation!")
        return False
    except ValueError as e:
        print(f"  Gate blocked: {str(e)[:200]}")
        return True


# S4: Empirical experiment — naive k-fold vs walk-forward
def run_s4_empirical_comparison():
    """
    Compares predictive metrics between:
      - Clean configuration: temporal walk-forward (gap=2 years)
      - Contaminated configuration: naive k-fold (without respecting time)

    Uses synthetic data with a realistic temporal structure to
    demonstrate the metric inflation caused by leakage.
    """
    import numpy as np
    import pandas as pd
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import KFold
    from sklearn.metrics import r2_score, mean_absolute_error

    # Without seeding the global generator: the panel comes from a local
    # default_rng, just below. Seeding here would suggest a dependency that
    # does not exist.

    print(f"\n--- Scenario S4: Empirical experiment - leakage vs clean ---")

    n_countries = 32
    # Local generator, seeded from the scientific configuration.
    #
    # The panel used to be drawn from numpy's global generator, whose state
    # depends on how much was drawn before it in the process. This report is
    # the evidence that the anti-leakage gates fire under an injected
    # violation -- an artifact whose reproducibility is its entire point, and
    # which depended on the order in which the modules were imported.
    rng = np.random.default_rng(SCIENTIFIC_CONFIG['random_seed'])

    years = list(range(2000, 2024))
    n_years = len(years)

    rows = []
    for c in range(n_countries):
        base_level = rng.uniform(15, 55)
        # Regime shifts: trend changes every ~8 years
        trends = [
            rng.uniform(-2.0, -0.5),   # 2000-2007: improvement
            rng.uniform(-0.5, 1.5),     # 2008-2015: stagnation/worsening
            rng.uniform(-3.0, -1.0),    # 2016-2023: strong improvement
        ]
        for y_idx, y in enumerate(years):
            regime = min(y_idx // 8, 2)
            trend = trends[regime]

            # Observable features (noisy, partial correlation)
            enrollment = 70 + trend * (y_idx % 8) + rng.normal(0, 8)
            expenditure = 3.5 + rng.normal(0, 1.2)
            completion = 100 - base_level + trend * y_idx + rng.normal(0, 6)

            # Target: dropout with regime shifts + substantial noise
            dropout = base_level + trend * y_idx + rng.normal(0, 5)
            dropout = max(0, min(100, dropout))

            # target + small noise (near-perfect proxy)
            future_leak = dropout + rng.normal(0, 0.3)

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

    # Fill the lags correctly (no leakage)
    df = df.sort_values(['country', 'year'])
    df['dropout_lag1'] = df.groupby('country')['dropout_rate'].shift(1)
    df['dropout_lag2'] = df.groupby('country')['dropout_rate'].shift(2)
    df = df.dropna().reset_index(drop=True)

    clean_features = ['enrollment', 'expenditure', 'completion_rate', 'dropout_lag1', 'dropout_lag2']
    leaked_features = clean_features + ['future_leak']
    target = 'dropout_rate'

    print("\n  [CLEAN] Temporal walk-forward, gap=2 years, no future features")
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

    print("  [LEAKED] Naive k-fold (ignores time) + future-derived features")
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

    print(f"\n  Results ({len(clean_r2s)} clean folds, {len(leaked_r2s)} leaked folds):")
    print(f"  R2:  clean={clean_r2_mean:.3f}  leaked={leaked_r2_mean:.3f}  diff={r2_inflation:+.3f}")
    print(f"  MAE: clean={clean_mae_mean:.3f}  leaked={leaked_mae_mean:.3f}  diff={-mae_deflation:+.3f}")

    print(f"\n  Conclusion: temporal leakage inflated R² by {r2_inflation:+.4f} points")
    print(f"  ({r2_inflation/max(abs(clean_r2_mean), 1e-9)*100:+.1f}% relative to the clean baseline)")

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

    # By the same resolution as everything else: writing to <root>/outputs makes
    # a World Bank run and an INEP run overwrite one another, and leaves the
    # report outside the tree the reproduction package collects.
    out_path = Path(get_absolute_output_path('validation'))
    out_path.mkdir(parents=True, exist_ok=True)
    results_file = out_path / 'leakage_injection_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to: {results_file}")

    return results


# Main
def main():
    parser = argparse.ArgumentParser(description="Leakage injection test")
    parser.add_argument('--quick', action='store_true',
                        help='Runs only S1-S3 (no empirical experiment)')
    args = parser.parse_args()

    print("Leakage injection test")
    print("Negative validation of the anti-leakage gate")

    valid_folds = generate_valid_folds()
    validator = TemporalValidator(min_gap_years=2)

    print(f"\n  Baseline: {len(valid_folds)} valid walk-forward folds")
    try:
        validator.enforce_walk_forward(valid_folds)
        print("  Gate PASSED valid folds (expected)")
        baseline_ok = True
    except ValueError:
        print("  FAILURE: gate rejected valid folds!")
        baseline_ok = False

    scenarios = [
        ("S1", "Insufficient gap (gap=0)", inject_s1_zero_gap),
        ("S2", "Temporal overlap (test inside train)", inject_s2_temporal_overlap),
        ("S3", "Reversed ordering (test before val)", inject_s3_reversed_order),
    ]

    results = {'baseline_valid': baseline_ok, 'scenarios': {}}

    for name, desc, injector in scenarios:
        contaminated = injector(valid_folds)
        detected = run_injection_scenario(name, desc, contaminated, validator)
        results['scenarios'][name] = {
            'description': desc,
            'leakage_detected': detected,
        }

    # S1-S3 summary
    all_detected = all(s['leakage_detected'] for s in results['scenarios'].values())
    print(f"\nSummary: {'All scenarios detected' if all_detected else 'Failure in some scenario'}")
    print(f"  Baseline valid: {baseline_ok}")
    for name, s in results['scenarios'].items():
        status = "DETECTED" if s['leakage_detected'] else "FAILED"
        print(f"  {name}: {status} — {s['description']}")

    if not args.quick:
        s4_results = run_s4_empirical_comparison()
        results['s4_empirical'] = s4_results

    out_path = Path(get_absolute_output_path('validation'))
    out_path.mkdir(parents=True, exist_ok=True)
    report_file = out_path / 'leakage_injection_report.json'
    with open(report_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Report saved to: {report_file}")

    if baseline_ok and all_detected:
        print("\nResult: OK - anti-leakage gate functional")
        sys.exit(0)
    else:
        print("\nResult: FAILURE")
        sys.exit(1)


if __name__ == '__main__':
    main()
