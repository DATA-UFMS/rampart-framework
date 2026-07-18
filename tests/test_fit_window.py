#!/usr/bin/env python3
"""Em que janela o modelo avaliado no teste é ajustado.

Decisão registrada, e não default: o modelo avaliado no teste é ajustado apenas na
janela de treino, e a validação serve exclusivamente para selecionar
hiperparâmetros.

A alternativa -- reajustar em treino+validação com os hiperparâmetros escolhidos --
é prática padrão e foi verificada como compatível com P2: o gap de val_end ao
teste é exatamente os 2 anos exigidos. Usaria 25% mais anos por entidade e moveria
a origem 4 anos para mais perto do teste.

Não foi adotada por uma assimetria. O que compraria é eficiência estatística num
dispositivo cuja acurácia preditiva não é o objeto de estudo -- o paper afirma
equivalência entre paradigmas e latência. O que custaria é margem na garantia
anti-leakage, que É o objeto: a separação efetiva entre o último dado de ajuste e
o primeiro de avaliação cairia de 6 anos para o mínimo declarado de 2. E exigiria
um segundo ajuste de imputação e scaler dentro dos três run_fold_analysis, que têm
implementações distintas por engine -- a configuração que produz divergência entre
paradigmas, quando equivalência bitwise é a afirmação central.

Estes testes existem para que uma mudança dessa escolha seja deliberada.
"""

import ast
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / 'src'
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from core.scientific_config import SCIENTIFIC_CONFIG

MODELS = sorted((_SRC / 'architectures_ml').glob('*/models/hierarchical_model.py'))
FINAL_FITS = ('simple_hierarchical_model', 'random_forest_hierarchical')


def _fold_analysis(path):
    tree = ast.parse(path.read_text())
    return next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
                and n.name == 'run_fold_analysis')


def _calls_evaluating_test(path):
    """Chamadas de modelo cujo conjunto de avaliação é a janela de teste."""
    found = []
    for call in ast.walk(_fold_analysis(path)):
        if not (isinstance(call, ast.Call)
                and getattr(call.func, 'attr', None) in FINAL_FITS):
            continue
        names = [getattr(a, 'id', None) for a in call.args]
        if any(n and 'test' in n for n in names):
            found.append((call, names))
    return found


class TestTheFinalModelFitsOnTrainOnly:

    def test_all_three_models_were_found(self):
        assert len(MODELS) == 3

    @pytest.mark.parametrize('path', MODELS, ids=lambda p: p.parts[-3])
    def test_both_models_are_evaluated_on_the_test_window(self, path):
        assert len(_calls_evaluating_test(path)) == len(FINAL_FITS), (
            f'{path.parts[-3]}: esperado um ajuste final por modelo'
        )

    @pytest.mark.parametrize('path', MODELS, ids=lambda p: p.parts[-3])
    def test_the_fit_arguments_are_the_training_window(self, path):
        """Os dois primeiros argumentos são X e y de ajuste."""
        for call, names in _calls_evaluating_test(path):
            assert names[0] == 'X_train_scaled', (
                f'{path.parts[-3]}:{call.lineno} ajusta em {names[0]}'
            )
            assert names[1] == 'y_train', (
                f'{path.parts[-3]}:{call.lineno} ajusta em {names[1]}'
            )

    @pytest.mark.parametrize('path', MODELS, ids=lambda p: p.parts[-3])
    def test_the_validation_window_is_not_concatenated_into_the_fit(self, path):
        """Um concat de treino com validação é a mudança que isto guarda."""
        for call, _ in _calls_evaluating_test(path):
            for argument in call.args[:2]:
                assert not isinstance(argument, ast.Call), (
                    f'{path.parts[-3]}:{call.lineno} passa uma expressão como '
                    f'dado de ajuste, e não a janela de treino'
                )

    @pytest.mark.parametrize('path', MODELS, ids=lambda p: p.parts[-3])
    def test_the_validation_window_is_used_for_selection(self, path):
        """A validação tem de servir para algo, senão é desperdício puro."""
        calls = []
        for call in ast.walk(_fold_analysis(path)):
            if isinstance(call, ast.Call) and \
                    getattr(call.func, 'attr', None) in FINAL_FITS:
                names = [getattr(a, 'id', None) for a in call.args]
                if any(n and 'val' in n for n in names):
                    calls.append(call)
        assert len(calls) == len(FINAL_FITS), (
            f'{path.parts[-3]}: a validação não é avaliada na seleção'
        )


class TestTheEffectiveSeparationIsRecorded:

    @staticmethod
    def _fold(gap, min_train=8, val_len=2, test_len=2):
        start = SCIENTIFIC_CONFIG['temporal_range_start']
        train_end = start + min_train - 1
        val_start = train_end + gap + 1
        val_end = val_start + val_len - 1
        test_start = val_end + gap + 1
        return {'train_end': train_end, 'val_end': val_end,
                'test_start': test_start,
                'fit_to_test_gap': test_start - train_end - 1}

    def test_the_fold_record_carries_it(self):
        source = (_SRC / 'core' / 'base_architecture.py').read_text()
        assert "'fit_to_test_gap'" in source
        assert "'fit_window': 'train_only'" in source

    def test_the_separation_exceeds_the_declared_minimum(self):
        """É por isso que a escolha compra algo: 6 anos contra 2."""
        gap = SCIENTIFIC_CONFIG['temporal_gap_years']
        fold = self._fold(gap)
        assert fold['fit_to_test_gap'] > gap, (
            'a separação efetiva não excede o mínimo declarado, então a escolha '
            'de não reajustar deixaria de comprar margem'
        )

    def test_refitting_would_reduce_it_to_the_minimum(self):
        """A verificação que sustenta a decisão, não uma afirmação solta."""
        gap = SCIENTIFIC_CONFIG['temporal_gap_years']
        fold = self._fold(gap)
        would_be = fold['test_start'] - fold['val_end'] - 1
        assert would_be == gap, (
            f'reajustar em treino+validação daria separação {would_be}, e a '
            f'decisão foi tomada supondo que cairia ao mínimo {gap}'
        )
        assert would_be < fold['fit_to_test_gap']

    def test_p2_would_still_hold_under_the_alternative(self):
        """A alternativa foi recusada por margem, não por violar P2."""
        gap = SCIENTIFIC_CONFIG['temporal_gap_years']
        fold = self._fold(gap)
        assert fold['test_start'] - fold['val_end'] - 1 >= gap
