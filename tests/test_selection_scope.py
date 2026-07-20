#!/usr/bin/env python3
"""P4: a seleção de features enxerga só a janela de treino do primeiro fold.

Escolher features pela concordância delas com valores futuros do alvo é
look-ahead bias (Kapoor & Narayanan, 2023): a feature entra no modelo porque
funciona no período em que o modelo será avaliado, e o desempenho reportado
mede a escolha, não a capacidade preditiva.

Nada testava isso. Quatro mutações em `run_feature_selection` -- passar o painel
inteiro para as correlações, empurrar `_first_fold_train_end` cem anos à frente,
tornar `_filter_by_year` inerte, e atribuir `data_train_only = data` --
sobreviviam todas com a suíte verde. As quatro têm o mesmo efeito observável, e
é esse efeito que os testes aqui detectam: uma feature que só correlaciona com o
alvo *depois* da janela de treino não pode ser selecionada.

O painel é construído para discriminar, e o próprio teste verifica isso antes de
concluir qualquer coisa: se a correlação dentro da janela não fosse desprezível
e a correlação sobre o painel inteiro não passasse do piso de seleção, passar
não significaria nada.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_SRC = str(Path(__file__).resolve().parents[1] / 'src')
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from core.base_architecture import BaseArchitectureML

YEARS = list(range(2000, 2016))
ENTITIES = ['BRA', 'ARG', 'CHL', 'URY']
TARGET = 'dropout_rate_sql_engine'

# Derivado da config em _first_fold_train_end: start=2000, min_train=8,
# val_len=2, gap=2 dão test_start=2014, val=[2010,2011], train_end=2007.
# O teste abaixo confere contra a fórmula em vez de confiar neste número.
TRAIN_END = 2007


class _Config:
    year_column = 'year'
    entity_column = 'country_code'
    entity_name_column = 'country_name'
    stratification_column = None
    target_source_column = 'source_rate'
    feature_columns = ['honest', 'future_only']
    excluded_columns = ['year', 'country_code', 'source_rate']
    temporal_range = (2000, 2015)
    walk_forward_config = {'min_train': 8, 'val_len': 2}


def _panel():
    """Painel onde `future_only` só se acopla ao alvo depois de TRAIN_END.

    Dentro da janela a correlação é zerada por construção, não por sorte da
    semente: com 32 linhas o ruído independente ainda dá |r| perto de 0.27, o
    bastante para a feature entrar pelo critério relaxado e o teste medir a
    semente em vez de P4. Fora da janela o acoplamento é forte, mas fica abaixo
    do teto de proxy sobre o painel inteiro -- senão a auditoria P3 abortaria
    antes e mascararia o que P4 faz.
    """
    rng = np.random.default_rng(20260726)
    rows = []
    for entity in ENTITIES:
        target = rng.normal(size=len(YEARS))
        honest = 0.6 * target + 0.8 * rng.normal(size=len(YEARS))
        future = np.where(np.array(YEARS) > TRAIN_END,
                          0.95 * target + 0.3 * rng.normal(size=len(YEARS)),
                          rng.normal(size=len(YEARS)))
        for index, year in enumerate(YEARS):
            rows.append({'year': year, 'country_code': entity,
                         'country_name': entity, 'source_rate': 0.0,
                         TARGET: target[index], 'honest': honest[index],
                         'future_only': future[index]})

    panel = pd.DataFrame(rows)
    window = panel['year'] <= TRAIN_END
    x = panel.loc[window, TARGET].to_numpy()
    y = panel.loc[window, 'future_only'].to_numpy()
    beta = np.cov(x, y, bias=True)[0, 1] / x.var()
    panel.loc[window, 'future_only'] = y - beta * x
    return panel


def _probe():
    class Probe(BaseArchitectureML):
        """Correlações reais: um probe que devolve {} atravessa vacuamente."""

        def setup_environment(self): pass
        def load_data(self): pass
        def validate_data(self, data): pass
        def create_target_implementation(self, data): return data
        def _compute_target_statistics(self, data): pass
        def _validate_temporal_folds(self, data, folds): pass
        def save_folds(self, data, folds): pass
        def prepare_features(self, data, features): return data

        def discover_numeric_columns(self, data):
            return ['honest', 'future_only']

        def compute_feature_correlations(self, data, features):
            frame = self._materialise_pandas(data, list(features) + [TARGET])
            return {feat: float(frame[feat].corr(frame[TARGET]))
                    for feat in features}

        def apply_collinearity_filter(self, data, features, threshold=0.8):
            return list(features)

    return Probe


@pytest.fixture
def architecture(tmp_path):
    return _probe()('sql_engine', str(tmp_path), dataset_config=_Config())


@pytest.fixture
def panel():
    return _panel()


class TestThePanelDiscriminates:
    """Sem isto, passar não distingue P4 de um painel sem sinal nenhum."""

    def test_the_late_feature_is_inert_inside_the_window(self, panel):
        window = panel[panel['year'] <= TRAIN_END]
        corr = abs(window['future_only'].corr(window[TARGET]))
        assert corr < 0.10, (
            f'|corr| = {corr:.3f} dentro da janela: o piso de seleção '
            f'relaxado é 0.1005, então a feature entraria mesmo sob P4 e o '
            f'teste não estaria medindo P4'
        )

    def test_the_late_feature_clears_the_floor_on_the_full_panel(self, panel):
        corr = panel['future_only'].corr(panel[TARGET])
        assert corr >= 0.15, (
            f'corr = {corr:.3f} sobre o painel inteiro: abaixo do piso de '
            f'seleção, então nem sem P4 a feature seria escolhida'
        )

    def test_the_late_feature_stays_below_the_proxy_ceiling(self, panel):
        """Senão a auditoria P3 abortaria e mascararia o que P4 faz."""
        corr = abs(panel['future_only'].corr(panel[TARGET]))
        assert corr < 0.80, corr

    def test_the_honest_feature_is_visible_inside_the_window(self, panel):
        window = panel[panel['year'] <= TRAIN_END]
        assert window['honest'].corr(window[TARGET]) >= 0.15


class TestTheWindowIsTheFirstFoldTrainWindow:

    def test_it_matches_the_configured_folds(self, architecture):
        """Fim do treino = início da validação menos o gap, menos um."""
        cfg = architecture.config
        wf = _Config.walk_forward_config
        gap = int(cfg['temporal_gap_years'])
        test_start = (_Config.temporal_range[0] + wf['min_train']
                      + wf['val_len'] + 2 * gap)
        val_end = test_start - gap - 1
        val_start = val_end - wf['val_len'] + 1
        assert architecture._first_fold_train_end() == val_start - gap - 1

    def test_the_gap_separates_it_from_validation(self, architecture):
        """P2: entre treino e validação há anos que ninguém lê."""
        cfg = architecture.config
        gap = int(cfg['temporal_gap_years'])
        train_end = architecture._first_fold_train_end()
        wf = _Config.walk_forward_config
        test_start = (_Config.temporal_range[0] + wf['min_train']
                      + wf['val_len'] + 2 * gap)
        val_start = test_start - gap - 1 - wf['val_len'] + 1
        assert val_start - train_end - 1 == gap

    def test_it_leaves_evaluation_years_outside(self, architecture, panel):
        train_end = architecture._first_fold_train_end()
        assert train_end < max(YEARS), (
            'a janela cobre o painel inteiro, então filtrar por ela não '
            'restringe nada'
        )
        kept = architecture._filter_by_year(panel, max_year=train_end)
        assert len(kept) < len(panel)
        assert kept['year'].max() <= train_end


class TestSelectionIsRestrictedToTheWindow:
    """O invariante. Cada uma das quatro mutações reprovadas cai aqui."""

    def test_a_feature_that_only_works_later_is_not_selected(
            self, architecture, panel):
        stats = architecture.run_feature_selection(panel)
        assert 'future_only' not in stats['selected_features'], (
            'feature escolhida pela concordância com o alvo em anos que o '
            'modelo ainda vai prever -- é a seleção que P4 existe para barrar'
        )

    def test_a_feature_that_works_inside_the_window_is_selected(
            self, architecture, panel):
        """Senão passar seria só a seleção não escolher nada."""
        stats = architecture.run_feature_selection(panel)
        assert 'honest' in stats['selected_features']

    def test_the_correlations_recorded_come_from_the_window(
            self, architecture, panel):
        stats = architecture.run_feature_selection(panel)
        window = panel[panel['year'] <= TRAIN_END]
        for feat, recorded in stats['target_correlations'].items():
            expected = float(window[feat].corr(window[TARGET]))
            assert recorded == pytest.approx(expected, abs=1e-9), (
                f'{feat}: gravado {recorded:.4f}, janela {expected:.4f}, '
                f'painel {panel[feat].corr(panel[TARGET]):.4f}'
            )

    def test_the_recorded_scope_names_the_window(self, architecture, panel):
        stats = architecture.run_feature_selection(panel)
        assert str(architecture._first_fold_train_end()) in \
            stats['temporal_scope']


class TestSelectionReadsOnlyWindowRows:
    """Observa as chamadas: cobre a mutação que passa o painel e reordena."""

    @staticmethod
    def _record(architecture):
        seen = []
        original = architecture.compute_feature_correlations

        def spy(data, features):
            seen.append(architecture._materialise_pandas(data, ['year']))
            return original(data, features)

        architecture.compute_feature_correlations = spy
        return seen

    def test_the_selection_call_sees_no_evaluation_year(
            self, architecture, panel):
        seen = self._record(architecture)
        architecture.run_feature_selection(panel)
        assert seen, 'a seleção não chegou a computar correlação nenhuma'
        train_end = architecture._first_fold_train_end()
        assert seen[0]['year'].max() <= train_end, (
            f"a seleção leu até {seen[0]['year'].max()}, além de {train_end}"
        )

    def test_the_collinearity_filter_sees_the_same_rows(
            self, architecture, panel):
        seen = []
        original = architecture.apply_collinearity_filter

        def spy(data, features, threshold=0.8):
            seen.append(architecture._materialise_pandas(data, ['year']))
            return original(data, features, threshold)

        architecture.apply_collinearity_filter = spy
        architecture.run_feature_selection(panel)
        assert seen
        assert seen[0]['year'].max() <= architecture._first_fold_train_end()

    def test_the_proxy_audit_reads_the_full_panel(self, architecture, panel):
        """A auditoria é o oposto de P4, e por um motivo declarado no código.

        Uma feature cuja correlação só passa do teto fora da janela de treino
        continua sendo proxy; auditar dentro da janela foi o que deixou uma
        passar.
        """
        seen = self._record(architecture)
        architecture.run_feature_selection(panel)
        assert len(seen) >= 2, 'não houve auditoria depois da seleção'
        assert seen[-1]['year'].max() == max(YEARS), (
            'a auditoria de proxy também ficou restrita à janela'
        )
