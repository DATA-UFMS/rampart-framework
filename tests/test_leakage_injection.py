#!/usr/bin/env python3
"""
Validação negativa do gate anti-leakage, executada pela suíte de testes.

O script `scripts/validation/leakage_injection.py` demonstra empiricamente que o
gate detecta violações deliberadas de integridade temporal, mas só roda sob
demanda. Estes testes reaproveitam os mesmos injetores para que a demonstração
seja exercitada em cada execução da suíte, sem depender de dados externos.

Cenários (idênticos aos do script):
  S1 - gap insuficiente entre train e val
  S2 - sobreposição temporal (anos de treino aparecem no teste)
  S3 - ordem temporal invertida (test_start < train_end)
"""

import importlib.util
from pathlib import Path

import pytest

_INJECTOR = (
    Path(__file__).resolve().parents[1]
    / 'scripts' / 'validation' / 'leakage_injection.py'
)


def _load_injector():
    """Carrega o script injetor como módulo, sem exigir que scripts/ seja pacote."""
    spec = importlib.util.spec_from_file_location('leakage_injection', _INJECTOR)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope='module')
def injector():
    if not _INJECTOR.exists():
        pytest.skip(f"injetor não encontrado: {_INJECTOR}")
    return _load_injector()


@pytest.fixture(scope='module')
def validator(injector):
    return injector.TemporalValidator(min_gap_years=2)


def test_baseline_folds_pass_the_gate(injector, validator):
    """Folds walk-forward válidos não devem ser rejeitados (evita falso positivo)."""
    folds = injector.generate_valid_folds()
    assert folds, "gerador de folds retornou lista vazia"
    validator.enforce_walk_forward(folds)


@pytest.mark.parametrize('scenario', ['s1_zero_gap', 's2_temporal_overlap', 's3_reversed_order'])
def test_injected_violation_is_rejected(injector, validator, scenario):
    """Cada violação injetada deve fazer o gate levantar ValueError."""
    inject = getattr(injector, f'inject_{scenario}')
    contaminated = inject(injector.generate_valid_folds())

    with pytest.raises(ValueError) as exc:
        validator.enforce_walk_forward(contaminated)

    assert 'Anti-leakage violation' in str(exc.value), \
        f"mensagem de erro sem diagnóstico esperado: {exc.value}"


def test_injection_does_not_mutate_the_valid_folds(injector):
    """Os injetores devem operar sobre cópias, não sobre os folds de referência."""
    original = injector.generate_valid_folds()
    reference = [dict(f) for f in original]

    for name in ('s1_zero_gap', 's2_temporal_overlap', 's3_reversed_order'):
        getattr(injector, f'inject_{name}')(original)

    assert original == reference, "um injetor mutou os folds válidos in-place"
