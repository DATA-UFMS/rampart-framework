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

A segunda metade do arquivo cobre os gates P3 que rodam depois da seleção --
proxy sobre o painel inteiro, reconstrução conjunta ajustada na janela, coluna
excluída na seleção final. Eles eram inalcançáveis nos testes existentes, cujo
probe devolve correlações vazias: sem feature selecionada não há o que auditar.
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
from core.validation import AntiLeakageViolation

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


class TestTheLeakageGatesFire:
    """Gates P3 que rodam depois da seleção, e nunca dispararam num teste.

    `TestPoolGate` cobre o gate do pool, que roda antes de qualquer correlação.
    Os três seguintes -- coluna excluída na seleção final, proxy sobre o painel
    inteiro, e reconstrução conjunta do alvo -- ficavam inalcançáveis porque
    aquele probe devolve correlações vazias: sem feature selecionada, não há o
    que auditar. Aqui as correlações são reais.
    """

    @staticmethod
    def _run(tmp_path, build, config=None, **overrides):
        panel = _panel()
        columns = build(panel)
        Probe = _probe()

        class Gated(Probe):
            def discover_numeric_columns(self, data):
                return sorted(columns)

        for name, function in overrides.items():
            setattr(Gated, name, function)

        class Config(_Config):
            feature_columns = sorted(columns)

        architecture = Gated('sql_engine', str(tmp_path),
                             dataset_config=Config())
        architecture.config = {**architecture.config, **(config or {})}
        return architecture.run_feature_selection(panel)

    def test_a_proxy_over_the_full_panel_halts(self, tmp_path):
        """|corr| acima do teto significa que a feature é o alvo com outro nome."""
        def build(panel):
            panel['proxy'] = 0.98 * panel[TARGET] + 0.02 * panel['honest']
            return ['honest', 'proxy']

        with pytest.raises(AntiLeakageViolation, match='P3 proxy detection'):
            self._run(tmp_path, build)

    def test_a_proxy_only_outside_the_window_halts(self, tmp_path):
        """Auditar dentro da janela de P4 foi o que deixou uma passar.

        Dentro da janela a correlação é moderada, então a feature é escolhida;
        sobre o painel inteiro ela passa do teto. Auditoria restrita à janela
        não veria nada.
        """
        def build(panel):
            late = panel['year'] > TRAIN_END
            panel['proxy'] = np.where(late, panel[TARGET],
                                      0.35 * panel[TARGET]
                                      + 0.94 * panel['honest'])
            return ['honest', 'proxy']

        panel = _panel()
        build(panel)
        window = panel[panel['year'] <= TRAIN_END]
        assert abs(window['proxy'].corr(window[TARGET])) < 0.80, (
            'dentro da janela já passa do teto, então o teste não distingue '
            'auditoria de painel inteiro de auditoria de janela'
        )
        assert abs(panel['proxy'].corr(panel[TARGET])) > 0.80

        with pytest.raises(AntiLeakageViolation, match='P3 proxy detection'):
            self._run(tmp_path, build)

    def test_a_negative_proxy_halts(self, tmp_path):
        """A seleção só admite correlação positiva; ela é medida na janela.

        Uma feature moderadamente positiva na janela e fortemente negativa
        depois dela é escolhida e só então auditada. Sem o valor absoluto no
        gate, ela passa.

        O teto é baixado para 0.50 nesta configuração porque a correlação sobre
        o painel inteiro satura perto de -0.66: as linhas da janela têm relação
        positiva com o alvo e puxam o coeficiente conjunto, por mais forte que
        seja o acoplamento tardio. O gate lê este parâmetro da config, então
        baixá-lo exercita o mesmo caminho.
        """
        def build(panel):
            rng = np.random.default_rng(5)
            late = (panel['year'] > TRAIN_END).to_numpy()
            panel['proxy'] = np.where(
                late, -5.0 * panel[TARGET],
                0.35 * panel[TARGET] + 0.9 * rng.normal(size=len(panel)))
            # Só a proxy no pool: com o teto em 0.50, `honest` (corr 0.6) o
            # dispararia sozinha e a exceção não seria atribuível.
            return ['proxy']

        threshold = 0.50
        panel = _panel()
        build(panel)
        window = panel[panel['year'] <= TRAIN_END]
        assert window['proxy'].corr(window[TARGET]) >= 0.15, (
            'não seria escolhida, então o gate nunca a veria'
        )
        full = panel['proxy'].corr(panel[TARGET])
        assert full < -threshold, (
            f'corr = {full:.3f}: sem correlação negativa além do teto, o valor '
            f'absoluto no gate é indiferente e o teste não o exercita'
        )
        assert full <= threshold, 'com o sinal, o gate não dispararia'

        with pytest.raises(AntiLeakageViolation,
                           match='P3 proxy detection') as exc:
            self._run(tmp_path, build,
                      config={'proxy_correlation_threshold': threshold})
        assert 'proxy' in str(exc.value)

    def test_features_that_reconstruct_the_target_halt(self, tmp_path):
        """Identidade aditiva: cada parcela correlaciona fraco, juntas fecham.

        Correlação par a par não enxerga isso -- é exatamente por isso que o
        gate de reconstrução existe.
        """
        def build(panel):
            rng = np.random.default_rng(7)
            target = panel[TARGET].to_numpy()
            noise = 1.5 * rng.normal(size=len(panel))
            # Ortogonalizado: com covariância amostral não-nula o ruído entra
            # numa parcela e sai da outra, e as duas deixam de ter a mesma
            # correlação -- uma delas cai abaixo do piso de seleção.
            noise -= (np.cov(target, noise, bias=True)[0, 1]
                      / target.var()) * target
            panel['half_a'] = 0.5 * target + noise
            panel['half_b'] = 0.5 * target - noise
            return ['half_a', 'half_b']

        panel = _panel()
        build(panel)
        for part in ('half_a', 'half_b'):
            corr = abs(panel[part].corr(panel[TARGET]))
            assert 0.15 <= corr < 0.80, (
                f'{part}: |corr| = {corr:.3f}. Fora dessa faixa o gate de '
                f'proxy dispara antes e o teste não alcança a reconstrução'
            )

        with pytest.raises(AntiLeakageViolation,
                           match='P3 joint reconstruction'):
            self._run(tmp_path, build)

    def test_the_reconstruction_is_fitted_on_the_window(self, tmp_path):
        """Uma identidade exata é detectada sem consultar os anos de avaliação.

        As parcelas somam o alvo exatamente dentro da janela e deixam de somar
        depois dela. Ajustar sobre o painel inteiro dilui o R2 abaixo do teto e
        a identidade passa -- que é o oposto do que o gate promete.
        """
        def build(panel):
            rng = np.random.default_rng(7)
            target = panel[TARGET].to_numpy()
            noise = 1.5 * rng.normal(size=len(panel))
            noise -= (np.cov(target, noise, bias=True)[0, 1]
                      / target.var()) * target
            late = (panel['year'] > TRAIN_END).to_numpy()
            panel['half_a'] = 0.5 * target + noise
            panel['half_b'] = np.where(
                late, rng.normal(size=len(panel)), 0.5 * target - noise)
            return ['half_a', 'half_b']

        panel = _panel()
        build(panel)
        window = panel[panel['year'] <= TRAIN_END]
        assert abs(window['half_a'] + window['half_b']
                   - window[TARGET]).max() < 1e-9
        assert abs(panel['half_a'] + panel['half_b']
                   - panel[TARGET]).max() > 1.0, (
            'a identidade também vale fora da janela, então o teste não '
            'distingue onde o ajuste é feito'
        )
        for part in ('half_a', 'half_b'):
            corr = window[part].corr(window[TARGET])
            assert corr >= 0.1005, (
                f'{part}: corr {corr:.3f} na janela, não seria selecionada'
            )

        with pytest.raises(AntiLeakageViolation,
                           match='P3 joint reconstruction'):
            self._run(tmp_path, build)

    def test_an_excluded_column_in_the_final_selection_halts(self, tmp_path):
        """O filtro de colinearidade devolve a lista; nada garante que ela seja
        um subconjunto do que entrou."""
        def build(panel):
            return ['honest']

        def smuggle(self, data, features, threshold=0.8):
            return list(features) + [self.target_column]

        with pytest.raises(AntiLeakageViolation, match='P3 data separation'):
            self._run(tmp_path, build, apply_collinearity_filter=smuggle)

    def test_a_clean_panel_reaches_the_end(self, tmp_path):
        """Base: sem isto, cada teste acima poderia estar falhando por outra razão."""
        stats = self._run(tmp_path, lambda panel: ['honest'])
        assert stats['selected_features'] == ['honest']
