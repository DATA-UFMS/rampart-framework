#!/usr/bin/env python3
"""
Módulo centralizado de validação para arquiteturas ML.

Centraliza toda lógica de validação temporal, integridade de dados e
métricas científicas, eliminando duplicação entre arquiteturas.

Protocolo anti-leakage (P1-P5):
    P1 — Ordenação temporal: train_end < val_start < val_end < test_start.
    P2 — Gap mínimo: N anos entre splits (default 2), configurável via
         temporal_gap_years. Embargo opcional para dados sub-anuais.
    P3 — Separação de features: lista de exclusão (derivadas do target,
         metadados) + detecção de proxy (|r| com target acima de
         proxy_correlation_threshold, medido no painel completo) + rejeição
         de reconstrução conjunta (R2 de mínimos quadrados do target sobre as
         features selecionadas acima de identity_r2_threshold, medido na
         janela de treino).
    P4 — Escopo de seleção: feature selection restrita ao período de
         treino do primeiro fold (Kapoor & Narayanan, 2023).
    P5 — Escopo de preprocessing: scaling e imputação ajustados
         exclusivamente nos dados de treino (Kaufman et al. 2012).

HPO: grid search no conjunto de validação; modelo final retreinado
no treino completo. Previne leakage por otimização no teste (Kapoor & Narayanan, 2023).

Enforcement: violações de P1/P2 geram ValueError via enforce_walk_forward().

Mapeamento à taxonomia de Kapoor & Narayanan (2023), oito tipos:
    L1.2 pré-processamento sobre treino+teste ....... P5
    L1.3 seleção de features sobre treino+teste ..... P4
    L1.4 duplicatas no conjunto de dados ............ canonical_fold
    L2   feature ilegítima .......................... P3 (rastreio, não quitação)
    L3.1 vazamento temporal ......................... P1
    L3.2 dependência entre treino e teste ........... P2 mitiga em parte

O gap de P2 não vem de K&N: a taxonomia deles não menciona gaps. Ele segue a
literatura de validação cruzada em blocos com buffer (Roberts et al., 2017),
que é a referência que os próprios K&N citam ao tratar de L3.2, com a variante
de embargo de López de Prado (2018).

L2 fica como rastreio e não como quitação: K&N deliberadamente não subdividem
essa categoria porque "o julgamento de se o uso de uma dada feature é legítimo
exige conhecimento de domínio". Um limiar de correlação detecta o subconjunto
detectável -- o proxy fortemente associado -- e não alcança uma feature que é
ilegítima por ser consequência do desfecho em vez de causa dele.

Dois tipos NÃO são cobertos por P1-P5, e isso é declarado aqui em vez de ficar
implícito na ausência:

    L3.2 dependência entre treino e teste. O mesmo país está em treino e em
         teste; o split é temporal, não por entidade. K&N dizem que isso é
         vazamento "a menos que a afirmação científica seja sobre uma
         distribuição com a mesma estrutura de dependência". Para previsão em
         painel a estrutura bate -- é o mesmo país, anos à frente -- mas o
         argumento é do autor e não do código. Note a assimetria deliberada:
         a CV interna agrupa por país (GroupKFold) porque ali o vazamento de
         entidade infla a seleção de hiperparâmetro; o split externo não
         agrupa porque agrupar mudaria a afirmação de previsão para outra.

    L3.3 viés de amostragem no teste. Coberto pela metade: a cobertura
         geográfica mínima por fold trata o viés espacial, que é o exemplo dos
         próprios K&N. A outra metade é criada por este pipeline -- linhas sem
         alvo observado são removidas, e ausência de alvo não é aleatória. A
         info sheet deles pergunta exatamente isso (Q18-19: "descreva como as
         linhas incluídas na análise foram selecionadas"). A evidência está em
         target_coverage.json; o argumento é do autor.
Violações de P3/P4 geram ValueError em run_feature_selection().
P5 é enforced por contrato (docstring + testes unitários).
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime


class AntiLeakageViolation(ValueError):
    """A property of the anti-leakage protocol was violated.

    Distinguished from operational failures because it is not recoverable: the
    experiment is invalid, and continuing would produce measurements of a
    pipeline that does not hold the guarantees the results are reported under.
    Subclasses ValueError so existing handlers and tests continue to match.
    """


try:
    import polars as pl
    _HAS_POLARS = True
except ImportError:
    _HAS_POLARS = False


def materialise_pandas(data: Any, columns: List[str]) -> pd.DataFrame:
    """Materialise the given columns of any supported frame as pandas."""
    if _HAS_POLARS and isinstance(data, pl.LazyFrame):
        return data.select(columns).collect().to_pandas()
    if _HAS_POLARS and isinstance(data, pl.DataFrame):
        return data.select(columns).to_pandas()
    if hasattr(data, 'compute'):  # Dask
        return data[columns].compute()
    if isinstance(data, pd.DataFrame):
        return data[columns].copy()
    raise TypeError(f"Unsupported data type for materialisation: {type(data)}")


def linear_reconstruction_r2(
    data: Any, features: List[str], target_column: str
) -> Optional[float]:
    """R2 of an ordinary least squares fit of the target on `features`.

    None when the fit is not determined: no features, too few complete rows for
    the number of predictors, or a constant target.
    """
    if not features:
        return None

    frame = materialise_pandas(data, list(features) + [target_column]).dropna()
    if len(frame) <= len(features) + 1:
        return None

    X = frame[list(features)].to_numpy(dtype=float)
    y = frame[target_column].to_numpy(dtype=float)
    design = np.column_stack([X, np.ones(len(X))])
    coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
    residual = y - design @ coefficients
    total = ((y - y.mean()) ** 2).sum()
    if total <= 0:
        return None
    return float(1.0 - (residual ** 2).sum() / total)



def canonical_fold(X, y, entities, years, *, paradigm: str):
    """Check the fold each paradigm materialised, and index it positionally.

    Every paradigm applies the same policy -- drop rows with no target, order by
    entity then year -- in its own idiom: an ORDER BY in the SQL view, a Polars
    sort, a pandas sort after compute. Three implementations of one policy are
    three chances to disagree, and a disagreement here is not a small one: the
    models would be fitted on the same rows in different orders, which
    falsifies the bitwise claim for a reason that has nothing to do with the
    paradigms.

    The policy stays inside each engine, because performing it is part of what
    the benchmark measures. What moves here is the verification.

    Three things are checked, each of which has a distinct failure behind it:

      * lengths agree, and no target is missing. A filter applied in one
        paradigm and not another changes n, and n reaches the reported degrees
        of freedom.
      * (entity, year) pairs are unique. A join that multiplies rows produces
        exactly this, and nothing downstream would notice: the fit succeeds and
        the latency simply grows. This is L1.4 in Kapoor & Narayanan (2023) --
        duplicates in the dataset -- whose info sheet asks whether duplicates
        exist and how they are handled. Here the answer is derived rather than
        asserted: the run halts if any survive.
      * the order is non-decreasing by (entity, year) under Python comparison.
        The engines order under their own rules -- a database collation, a Rust
        string comparison -- and only agreement between them makes the
        comparison meaningful.

    The returned objects carry a positional index. Downstream alignment is
    positional, and a label index that survives this far is a hazard rather
    than information.
    """
    frames = {'X': X, 'y': y, 'entities': entities, 'years': years}
    lengths = {name: len(value) for name, value in frames.items()}
    if len(set(lengths.values())) != 1:
        raise ValueError(
            f"{paradigm}: the materialised fold has inconsistent lengths "
            f"{lengths}."
        )

    if len(X) == 0:
        raise ValueError(
            f"{paradigm}: the materialised fold is empty. There is nothing to "
            f"fit, and an empty fold reaches the reported n as a zero."
        )

    target = pd.Series(y).reset_index(drop=True)
    if bool(target.isna().any()):
        missing = int(target.isna().sum())
        raise ValueError(
            f"{paradigm}: {missing} of {len(target)} rows carry no target. "
            f"Rows without a target are dropped upstream in every paradigm; "
            f"their presence here means one of them stopped doing it."
        )

    entity_values = pd.Series(entities).reset_index(drop=True).to_numpy()
    year_values = pd.Series(years).reset_index(drop=True).to_numpy()

    duplicated = pd.MultiIndex.from_arrays(
        [entity_values, year_values]).duplicated()
    if duplicated.any():
        raise ValueError(
            f"{paradigm}: {int(duplicated.sum())} duplicated (entity, year) "
            f"pairs in the materialised fold. One row per entity and year is "
            f"the panel's shape; duplicates come from a join that multiplied "
            f"rows, and nothing downstream would notice."
        )

    order = np.lexsort((year_values, entity_values))
    if not np.array_equal(order, np.arange(len(order))):
        first = int(np.flatnonzero(order != np.arange(len(order)))[0])
        raise ValueError(
            f"{paradigm}: the materialised fold is not ordered by (entity, "
            f"year); the first row out of place is at position {first} "
            f"({entity_values[first]!r}, {year_values[first]!r}). The "
            f"paradigms must present the same rows in the same order, or the "
            f"models are fitted on different matrices."
        )

    return (pd.DataFrame(X).reset_index(drop=True),
            target,
            pd.Series(entities).reset_index(drop=True))



def assert_lag_columns(present, paradigm: str, lag_orders) -> None:
    """The autoregressive columns exist, in every paradigm.

    Two of the three built them inside a try/except that printed a warning and
    returned the frame without them -- one of those catching bare Exception.
    A paradigm missing its lags trains on a different feature set from the
    other two, so the bitwise claim fails for a reason that has nothing to do
    with the paradigms, and the only trace is a line of stdout in a run that
    takes hours.

    Lags are not optional. Where the entity's past target was never observed
    the join yields NULL, which is the honest value and is handled downstream;
    a missing *column* is a different thing entirely.
    """
    expected = {f'dropout_rate_lag_{order}' for order in lag_orders}
    missing = sorted(expected - set(present))
    if missing:
        raise ValueError(
            f"{paradigm}: as colunas de defasagem do alvo não foram criadas "
            f"{missing}. Sem elas este paradigma treina sobre um conjunto de "
            f"features diferente dos outros dois, e a comparação deixa de ser "
            f"entre paradigmas."
        )


def audit_feature_set(
    data: Any, features: List[str], target_column: str, config: Dict
) -> Dict:
    """Apply the P3 checks to the feature set a model actually trains on.

    Feature selection audits what it produced. Features appended afterwards --
    autoregressive lags of the target, added by the models -- never passed
    through it, so the set the models consume was never the set the gate saw.

    Autoregressive features are exempt from the pairwise proxy check: predicting
    a series from its own past is the task rather than a leak, and a lag
    correlates with the target by construction. The exemption is recorded with
    the measured correlation.

    The joint reconstruction check asks two separate questions, because one
    threshold cannot answer both:

      * Do the *non-autoregressive* features jointly determine the target?
        That is the leakage question -- an additive identity that pairwise
        correlation cannot see, such as rates that sum to a constant. Judged at
        `identity_r2_threshold`.

      * Does the *whole set*, lags included, reproduce the target exactly?
        A genuine lag never does: y_t is not an exact linear function of
        y_{t-2} and y_{t-3}. An R2 at machine precision means a column labelled
        as lagged is not lagged -- an off-by-one join, or a lag of zero. Judged
        against `target_reproduction_tolerance`, which is not a modelling
        choice but a numerical one.

    Applying the 0.95 ceiling to the whole set conflated the two. On an annual
    panel pooled across entities, a lag carries the entity's level and the
    pooled R2 is high by construction, so the check would abort a valid run for
    exhibiting the autocorrelation the task exists to exploit -- and it had
    never been evaluated on real data, only on fixtures.

    Correlations are computed here rather than through each paradigm's own
    implementation, so the gate behaves identically whichever paradigm invokes
    it.
    """
    marker = str(config.get('autoregressive_feature_marker', '_lag_'))
    proxy_threshold = float(config.get('proxy_correlation_threshold', 0.80))
    identity_threshold = float(config.get('identity_r2_threshold', 0.95))
    reproduction_tolerance = float(
        config.get('target_reproduction_tolerance', 1e-9))

    features = list(features)
    frame = materialise_pandas(data, features + [target_column])
    correlations = {}
    for feature in features:
        pair = frame[[feature, target_column]].dropna()
        if len(pair) < 3 or pair[feature].nunique() < 2:
            continue
        correlations[feature] = float(abs(pair.corr().iloc[0, 1]))

    autoregressive = [f for f in features if marker in f]
    proxies = {
        feature: value for feature, value in correlations.items()
        if marker not in feature and value > proxy_threshold
    }
    if proxies:
        raise AntiLeakageViolation(
            f"Anti-leakage violation (P3 proxy detection) in the final feature "
            f"set: |correlation| > {proxy_threshold} with target "
            f"(Kapoor & Narayanan, 2023): {proxies}"
        )

    exogenous = [f for f in features if marker not in f]
    identity_r2 = linear_reconstruction_r2(data, exogenous, target_column)
    if identity_r2 is not None and identity_r2 > identity_threshold:
        raise AntiLeakageViolation(
            f"Anti-leakage violation (P3 joint reconstruction) in the final "
            f"feature set: the non-autoregressive features explain the target "
            f"with R2 = {identity_r2:.4f} > {identity_threshold}, indicating "
            f"the target is an algebraic function of them: {sorted(exogenous)}"
        )

    reproduction_r2 = linear_reconstruction_r2(data, features, target_column)
    if (reproduction_r2 is not None
            and reproduction_r2 > 1.0 - reproduction_tolerance):
        raise AntiLeakageViolation(
            f"Anti-leakage violation (P3 target reproduction) in the final "
            f"feature set: R2 = {reproduction_r2:.12f} reproduces the target "
            f"to numerical precision. No genuine lag does this; a column "
            f"labelled as lagged is carrying the contemporaneous value: "
            f"{sorted(features)}"
        )

    return {
        'features_audited': sorted(features),
        'proxy_correlation_threshold': proxy_threshold,
        'identity_r2_threshold': identity_threshold,
        # Over the non-autoregressive features: the leakage question.
        'joint_reconstruction_r2': identity_r2,
        # Over the whole set: only exact reproduction is a defect here.
        'full_set_reconstruction_r2': reproduction_r2,
        'target_reproduction_tolerance': reproduction_tolerance,
        'autoregressive_exemptions': {
            f: correlations[f] for f in autoregressive if f in correlations
        },
    }


class TemporalValidator:
    """
    Validador temporal para prevenção de vazamento em séries temporais.

    Implementa validação de splits temporais com gaps obrigatórios
    e embargo configurável para garantir validade científica em previsão
    de dropout educacional.

    O protocolo combina dois mecanismos complementares:
      - **Gap temporal**: período mínimo entre splits consecutivos,
        impedindo que informação futura influencie o treino.
      - **Embargo**: um acréscimo exigido ao gap, não uma exclusão de
        observações. López de Prado (2018) descreve o embargo como a remoção
        das observações de treino adjacentes ao limite de cada split; aqui o
        validador apenas verifica que o gap declarado cobre o embargo
        declarado, e reprova o fold quando não cobre. Nada é removido -- o
        que remove observações é o gap, na geração dos folds.

        A distinção importa porque as duas formulações diferem quando o gap
        não é uniforme. Com gap constante de dois anos e um ponto por
        entidade/ano, exigir gap >= embargo e excluir `embargo` observações
        adjacentes selecionam o mesmo conjunto de treino, e é por isso que a
        verificação basta neste painel.

    Nota sobre purging (López de Prado 2018):
        Purging remove observações de treino cujos labels sobrepõem
        temporalmente o período de teste. Em dados com granularidade
        anual (um ponto por país/ano), não há sobreposição de labels
        entre splits — cada observação é um ponto discreto. Portanto,
        purging é desnecessário neste contexto. O gap temporal de N
        anos já subsume o efeito do embargo para dados anuais, pois
        não existem observações sub-anuais intermediárias a excluir.
        O parâmetro embargo_years existe para uso em adaptações do
        framework a dados de maior frequência (mensal, diário).
    """

    def __init__(self, min_gap_years: int = 2, embargo_years: int = 0):
        """
        Inicializa validador temporal.

        Args:
            min_gap_years: Gap mínimo em anos entre splits (default: 2).
                Controla a separação temporal obrigatória entre períodos.
            embargo_years: Período adicional de embargo em anos (default: 0).
                Quando > 0, observações no intervalo [train_end+1,
                train_end+embargo] são excluídas do treino, mesmo que
                já estejam fora do split de treino. Previne leakage
                por autocorrelação residual em dados com dependência
                temporal (lagged features, médias móveis).
        """
        self.min_gap_years = min_gap_years
        self.embargo_years = embargo_years
    
    def validate_fold_integrity(self, fold: Dict) -> Tuple[bool, List[str]]:
        """
        Valida integridade completa de um fold temporal.
        
        Args:
            fold: Dicionário com configuração do fold
            
        Returns:
            Tupla (is_valid, lista_de_erros)
        """
        errors = []
        
        # Verificar campos obrigatórios
        required_fields = [
            'train_start', 'train_end', 'val_start', 'val_end',
            'test_start', 'test_end'
        ]
        
        for field in required_fields:
            if field not in fold:
                errors.append(f"Campo obrigatório ausente: {field}")
        
        if errors:
            return False, errors
        
        # Verificar ordem cronológica
        if fold['train_start'] > fold['train_end']:
            errors.append(f"Train: início ({fold['train_start']}) > fim ({fold['train_end']})")
        
        if fold['val_start'] > fold['val_end']:
            errors.append(f"Val: início ({fold['val_start']}) > fim ({fold['val_end']})")
        
        if fold['test_start'] > fold['test_end']:
            errors.append(f"Test: início ({fold['test_start']}) > fim ({fold['test_end']})")
        
        # Verificar sequência temporal
        if fold['train_end'] >= fold['val_start']:
            errors.append(f"Sobreposição train-val: train_end={fold['train_end']}, val_start={fold['val_start']}")
        
        if fold['val_end'] >= fold['test_start']:
            errors.append(f"Sobreposição val-test: val_end={fold['val_end']}, test_start={fold['test_start']}")
        
        # Verificar gaps mínimos
        train_val_gap = fold['val_start'] - fold['train_end'] - 1
        val_test_gap = fold['test_start'] - fold['val_end'] - 1

        if train_val_gap < self.min_gap_years:
            errors.append(f"Gap train-val insuficiente: {train_val_gap} < {self.min_gap_years}")

        if val_test_gap < self.min_gap_years:
            errors.append(f"Gap val-test insuficiente: {val_test_gap} < {self.min_gap_years}")

        # Verificar embargo: o gap efetivo deve cobrir também o embargo
        if self.embargo_years > 0:
            effective_gap_tv = train_val_gap - self.embargo_years
            effective_gap_vt = val_test_gap - self.embargo_years
            if effective_gap_tv < 0:
                errors.append(
                    f"Embargo train-val violado: gap={train_val_gap} < "
                    f"embargo={self.embargo_years}"
                )
            if effective_gap_vt < 0:
                errors.append(
                    f"Embargo val-test violado: gap={val_test_gap} < "
                    f"embargo={self.embargo_years}"
                )

        is_valid = len(errors) == 0
        return is_valid, errors
    
    def validate_walk_forward(self, folds: List[Dict]) -> Tuple[bool, Dict]:
        """
        Valida estrutura walk-forward de múltiplos folds.
        
        Args:
            folds: Lista de folds para validação
            
        Returns:
            Tupla (is_valid, relatório_detalhado)
        """
        report = {
            'total_folds': len(folds),
            'valid_folds': 0,
            'invalid_folds': 0,
            'fold_errors': {},
            'walk_forward_valid': True,
            'expanding_window': True
        }
        
        for i, fold in enumerate(folds):
            is_valid, errors = self.validate_fold_integrity(fold)
            
            if is_valid:
                report['valid_folds'] += 1
            else:
                report['invalid_folds'] += 1
                report['fold_errors'][f'fold_{i}'] = errors
        
        # Verificar se é walk-forward expansivo
        if len(folds) > 1:
            for i in range(1, len(folds)):
                # Train deve expandir ou manter
                if folds[i]['train_end'] < folds[i-1]['train_end']:
                    report['expanding_window'] = False
                    report['walk_forward_valid'] = False
                    break
        
        # A structural walk-forward violation is recorded above and must
        # reach the verdict; counting invalid folds alone would let an
        # invalid fold sequence pass with every fold individually valid.
        report['all_valid'] = (
            report['invalid_folds'] == 0 and report['walk_forward_valid']
        )

        return report['all_valid'], report

    def enforce_walk_forward(self, folds: List[Dict]) -> None:
        """
        Valida estrutura walk-forward e interrompe execução em caso de violação.

        Raises:
            ValueError: Se qualquer fold violar integridade temporal
        """
        # Um conjunto vazio satisfaz "nenhum fold inválido" vacuamente, e o
        # pipeline registrava "0 folds -- integridade temporal verificada".
        # Zero folds significa que os modelos não tiveram nada em que treinar,
        # ou que o artefato está quebrado; em nenhum dos dois casos há
        # integridade a atestar.
        if not folds:
            raise AntiLeakageViolation(
                "Anti-leakage violation: the fold configuration is empty. "
                "There is no temporal integrity to attest to, and the models "
                "had nothing to train on."
            )

        all_valid, report = self.validate_walk_forward(folds)
        if not all_valid:
            errors = report.get('fold_errors', {})
            raise AntiLeakageViolation(
                f"Anti-leakage violation: {report['invalid_folds']} of "
                f"{report['total_folds']} folds failed temporal integrity. "
                f"Errors: {errors}"
            )
    


class DataIntegrityValidator:
    """
    Validador de integridade de dados para ML.
    
    Verifica qualidade, completude e consistência dos dados
    antes do treinamento de modelos.
    """
    
    def validate_target_distribution(self, target_values: np.ndarray,
                                    expected_range: Tuple[float, float] = (0, 100),
                                    name: str = "target") -> Dict:
        """
        Valida distribuição da variável target.
        
        Args:
            target_values: Valores do target
            expected_range: Range esperado (min, max)
            name: Nome da variável para relatório
            
        Returns:
            Dicionário com análise da distribuição
        """
        # Remover NaN para análise
        clean_values = target_values[~np.isnan(target_values)]
        
        validation = {
            'variable': name,
            'total_observations': len(target_values),
            'valid_observations': len(clean_values),
            'missing_count': len(target_values) - len(clean_values),
            'missing_rate': (len(target_values) - len(clean_values)) / len(target_values) * 100
        }
        
        if len(clean_values) > 0:
            validation.update({
                'mean': float(np.mean(clean_values)),
                'std': float(np.std(clean_values)),
                'min': float(np.min(clean_values)),
                'max': float(np.max(clean_values)),
                'median': float(np.median(clean_values)),
                'q25': float(np.percentile(clean_values, 25)),
                'q75': float(np.percentile(clean_values, 75))
            })
            
            # Verificar range
            out_of_range = np.sum((clean_values < expected_range[0]) | 
                                 (clean_values > expected_range[1]))
            validation['out_of_range_count'] = int(out_of_range)
            validation['out_of_range_rate'] = float(out_of_range / len(clean_values) * 100)
            
            negative_count = np.sum(clean_values < 0)
            validation['negative_values'] = int(negative_count)
            
            # Alertas
            validation['warnings'] = []
            
            if validation['missing_rate'] > 20:
                validation['warnings'].append(f"Alta taxa de missing: {validation['missing_rate']:.1f}%")
            
            if validation['out_of_range_rate'] > 5:
                validation['warnings'].append(f"Valores fora do range: {validation['out_of_range_rate']:.1f}%")
            
            if negative_count > 0:
                validation['warnings'].append(f"Valores negativos detectados: {negative_count}")
            
            if validation['std'] < 1:
                validation['warnings'].append(f"Baixa variabilidade: std={validation['std']:.2f}")
        else:
            validation['warnings'] = ["Sem dados válidos para análise"]
        
        validation['is_valid'] = len(validation.get('warnings', [])) == 0
        
        return validation
    
    def validate_dataframe(self, df: pd.DataFrame,
                          target_col: str = None,
                          check_completeness: bool = True) -> Tuple[bool, Dict]:
        """
        Valida integridade completa de um DataFrame.
        
        Args:
            df: DataFrame para validar
            target_col: Nome da coluna target (opcional)
            check_completeness: Se deve verificar completude
            
        Returns:
            Tupla (is_valid, validation_report)
        """
        validation_report = {
            'is_valid': True,
            'shape': df.shape,
            'columns': list(df.columns),
            'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()},
            'missing_data': {},
            'warnings': [],
            'errors': []
        }
        
        if df.empty:
            validation_report['is_valid'] = False
            validation_report['errors'].append("DataFrame está vazio")
            return False, validation_report
        
        missing_counts = df.isnull().sum()
        missing_rates = (missing_counts / len(df)) * 100
        
        for col in df.columns:
            if missing_counts[col] > 0:
                validation_report['missing_data'][col] = {
                    'count': int(missing_counts[col]),
                    'rate': float(missing_rates[col])
                }
                
                # Se completude é necessária
                if check_completeness and missing_rates[col] > 50:
                    validation_report['warnings'].append(
                        f"Coluna '{col}' tem {missing_rates[col]:.1f}% de dados faltantes"
                    )
        
        # Validar target se especificado
        if target_col and target_col in df.columns:
            target_validation = self.validate_target_distribution(
                df[target_col].values,
                name=target_col
            )
            validation_report['target_validation'] = target_validation
            
            if not target_validation['is_valid']:
                validation_report['warnings'].extend(target_validation.get('warnings', []))
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].var() == 0:
                validation_report['warnings'].append(f"Coluna '{col}' tem variância zero")
        
        inf_counts = np.isinf(df.select_dtypes(include=[np.number])).sum()
        for col, count in inf_counts.items():
            if count > 0:
                validation_report['warnings'].append(f"Coluna '{col}' tem {count} valores infinitos")
                validation_report['is_valid'] = False
        
        duplicates = df.duplicated().sum()
        if duplicates > 0:
            validation_report['warnings'].append(f"DataFrame tem {duplicates} linhas duplicadas")
        
        # Determinar validade final
        if validation_report['errors']:
            validation_report['is_valid'] = False
        
        # Heurística: mais de MAX_TOLERABLE_WARNINGS indica dataset degradado
        MAX_TOLERABLE_WARNINGS = 5
        if len(validation_report['warnings']) > MAX_TOLERABLE_WARNINGS:
            validation_report['is_valid'] = False
            validation_report['errors'].append(
                f"Número de warnings ({len(validation_report['warnings'])}) "
                f"excede o limite tolerável ({MAX_TOLERABLE_WARNINGS})"
            )
        
        return validation_report['is_valid'], validation_report
    


def impute_from_training_window(train: pd.DataFrame, *apply_to: pd.DataFrame,
                               strategy: str = 'median'
                               ) -> Tuple[List[pd.DataFrame], Dict]:
    """Fill missing feature values with statistics fitted on the training window.

    P5 (preprocessing scope; Kaufman et al., 2012) requires that any fitted
    statistic come from training data alone. The collection stage previously
    imputed with the mean of stratum peers in the same year and with the mean of
    the whole panel across all years -- statistics computed over validation and
    test periods and written into training cells, before folds existed, where the
    P1-P5 gates could not reach.

    Forward fill within an entity needs no fitted statistic, so it stays in
    collection and is P5-safe by construction. Everything that needs a statistic
    happens here, once, for every paradigm: three implementations of this would be
    three chances for the paradigms to preprocess differently, and the equivalence
    claim assumes they do not.

    A column with no observed value in the training window raises. It cannot occur
    while the invariants hold: the training window is expansive (train_start is
    fixed at the start year), and feature selection runs on the first fold's
    training window under P4, so a feature that was selected has observations in
    that window and therefore in every later one. If it occurs, an invariant broke.

    The three alternatives are all worse. Filling a constant fabricates a value the
    training window never observed, and makes the feature constant in training and
    variable in test -- a distribution shift introduced by preprocessing itself.
    Dropping the column changes the feature set between folds and possibly between
    paradigms, which breaks both cross-fold comparability and the equivalence
    claim. Leaving the value missing defers the failure to RidgeCV, since
    StandardScaler propagates NaN silently rather than rejecting it.

    Args:
        train: the fold's training frame; the only source of statistics
        apply_to: further frames (validation, test) receiving the same values
        strategy: 'median' (default, resistant to outliers) or 'mean'

    Returns:
        ([train, *apply_to] filled, report) with the fitted value per column and
        the columns left untouched.
    """
    if strategy not in ('median', 'mean'):
        raise ValueError(f"unsupported strategy: {strategy}")

    fitted: Dict[str, float] = {}
    unobserved: List[str] = []
    for column in train.columns:
        observed = train[column].dropna()
        if observed.empty:
            unobserved.append(column)
            continue
        fitted[column] = float(observed.median() if strategy == 'median'
                               else observed.mean())

    if unobserved:
        raise ValueError(
            f"Features com nenhuma observação na janela de treino: "
            f"{sorted(unobserved)}. Com janela expansiva e seleção sob P4 na "
            f"janela do primeiro fold, isso não pode ocorrer: uma feature "
            f"selecionada tem dados ali e portanto em toda janela posterior. "
            f"Preencher com constante fabricaria um valor que o treino nunca "
            f"observou."
        )

    # Counted per split, because how much of each window is fabricated is what
    # a reader needs and the report carried only the fitted values. The extent
    # of fold-level imputation appeared in no artifact at all.
    split_names = ['train'] + [f'apply_{index}'
                               for index in range(len(apply_to))]
    filled = []
    filled_cells = {}
    for name, frame in zip(split_names, (train, *apply_to)):
        out = frame.copy()
        per_column = {}
        for column, value in fitted.items():
            if column not in out.columns:
                continue
            missing = int(out[column].isna().sum())
            if missing:
                per_column[column] = missing
            out[column] = out[column].fillna(value)
        filled_cells[name] = {
            'rows': int(len(out)),
            'by_column': per_column,
            'total': int(sum(per_column.values())),
        }
        filled.append(out)

    report = {
        'strategy': strategy,
        'fitted_on_rows': int(len(train)),
        'filled_cells': filled_cells,
        'values': fitted,
        # Mantido, sempre vazio: a condição levanta acima. A chave permanece
        # para que um artefato antigo e um novo sejam comparáveis.
        'columns_without_training_observation': [],
    }
    return filled, report
