#!/usr/bin/env python3
"""
Classe base abstrata para arquiteturas ML.

Este módulo define a estrutura comum para todas as arquiteturas de ML,
garantindo consistência metodológica e facilitando manutenção sem duplicação.
Preserva a lógica de cada arquitetura.
"""

from abc import ABC, abstractmethod
import os
import json
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from core.scientific_config import SCIENTIFIC_CONFIG, setup_reproducibility
from core.validation import AntiLeakageViolation

try:
    import polars as pl
    _HAS_POLARS = True
except ImportError:
    _HAS_POLARS = False


class BaseArchitectureML(ABC):
    """
    Classe base abstrata para arquiteturas de Machine Learning.

    Define a estrutura comum e métodos compartilhados entre diferentes
    arquiteturas (Data Lake, Data Warehouse), garantindo consistência
    metodológica e eliminando duplicação de código.

    Protocolo anti-leakage (P1-P5):
        P1 — Ordenação temporal: train < val < test estritamente.
        P2 — Gap mínimo: N anos entre splits consecutivos (default 2).
        P3 — Separação de features: exclusão de derivadas do target
             e detecção de proxy (|correlação| > threshold).
        P4 — Escopo temporal da seleção: feature selection restrita
             ao período de treino do primeiro fold (Kapoor & Narayanan, 2023).
        P5 — Escopo de preprocessing: transformações estatísticas
             (scaling, imputação) ajustadas exclusivamente no treino
             (Kaufman et al. 2012).

    Estratégia de HPO:
        Hiperparâmetros são selecionados via grid search no conjunto
        de validação, nunca no teste. O modelo final é retreinado no
        treino completo com os hiperparâmetros selecionados. Isso
        previne leakage por otimização no conjunto de teste (Kapoor & Narayanan, 2023).

    Attributes:
        architecture_name: Nome da arquitetura (task_graph, sql_engine)
        output_base: Diretório base para outputs
        prep_dir: Diretório de preparação
        target_column: Nome da coluna target criada
        source_column: Coluna fonte para criar target
    """

    _registry: Dict[str, type] = {}

    # Radical dos nomes derivados do target: o target de cada paradigma
    # (TARGET_STEM_<paradigma>) e seus lags (TARGET_STEM_lag_<k>).
    TARGET_STEM = 'dropout_rate'

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # Registrar apenas classes completamente concretas (sem métodos abstratos restantes).
        # Detalhe de implementação CPython: ABCMeta.__new__ dispara __init_subclass__
        # (via type.__new__) antes de computar __abstractmethods__. Por isso,
        # recomputamos os abstratos pendentes percorrendo o MRO manualmente.
        # Premissa: todas as sobreposições de métodos abstratos aparecem em __dict__
        # de alguma classe do MRO. Protocolos de atributos dinâmicos (__getattr__)
        # não são suportados.
        pending: set = set()
        for klass in reversed(cls.__mro__):
            for name, val in klass.__dict__.items():
                if getattr(val, '__isabstractmethod__', False):
                    pending.add(name)
                elif name in pending:
                    pending.discard(name)
        if pending:
            return
        # Registro opt-in: apenas classes que definem explicitamente PARADIGM_META
        # em seu próprio __dict__ são tratadas como paradigmas. Subclasses concretas
        # sem PARADIGM_META (helpers, stubs de teste) são ignoradas silenciosamente.
        if 'PARADIGM_META' not in cls.__dict__:
            return
        meta = cls.PARADIGM_META
        if not meta.get('name'):
            raise TypeError(
                f"{cls.__name__} define PARADIGM_META mas está faltando "
                f"a chave obrigatória 'name'."
            )
        existing = BaseArchitectureML._registry.get(meta['name'])
        if existing is not None:
            same_source = (
                existing.__qualname__ == cls.__qualname__
                and existing.__name__ == cls.__name__
            )
            if not same_source:
                raise TypeError(
                    f"O nome de paradigma '{meta['name']}' já está registrado por "
                    f"{existing.__name__}. {cls.__name__} não pode reutilizá-lo."
                )
        BaseArchitectureML._registry[meta['name']] = cls

    @classmethod
    def get_registered_paradigms(cls) -> Dict[str, type]:
        """Retorna todas as classes de paradigmas concretos registrados."""
        return dict(cls._registry)

    def __init__(self, architecture_name: str, output_base_path: str,
                 dataset_config=None):
        """
        Inicializa a arquitetura base.

        Args:
            architecture_name: Identificador da arquitetura
            output_base_path: Caminho base para outputs
            dataset_config: DatasetConfig (default: worldbank)
        """
        self.architecture_name = architecture_name
        self.output_base = output_base_path
        self.prep_dir = f"{self.output_base}/prep"

        # Configuração científica centralizada
        self.config = SCIENTIFIC_CONFIG
        setup_reproducibility()

        # Dataset config (lazy import, detecta via env var para subprocessos)
        if dataset_config is None:
            dataset_name = os.environ.get('DATASET_NAME', 'worldbank')
            if dataset_name == 'inep_censo':
                from datasets.inep_censo import InepCensoDatasetConfig
                dataset_config = InepCensoDatasetConfig()
            else:
                from datasets.worldbank import WorldBankDatasetConfig
                dataset_config = WorldBankDatasetConfig()
        self.dataset_config = dataset_config

        # Configuração de target (derivada do dataset)
        self.target_column = f"{self.TARGET_STEM}_{architecture_name}"
        self.source_column = dataset_config.target_source_column
        
        self._create_directory_structure()
        
    def _create_directory_structure(self):
        """Cria estrutura de diretórios necessária."""
        os.makedirs(self.prep_dir, exist_ok=True)
        os.makedirs(f"{self.prep_dir}/folds", exist_ok=True)
        
    @abstractmethod
    def setup_environment(self) -> None:
        """
        Configura ambiente específico da arquitetura.
        
        Método abstrato que deve ser implementado por cada arquitetura
        para configurar seu ambiente específico (Dask, SQL, etc.).
        """
        pass
    
    @abstractmethod
    def load_data(self) -> Any:
        """
        Carrega dados específicos da arquitetura.
        
        Returns:
            Dados carregados no formato específico da arquitetura
            (dd.DataFrame para Dask, conexão SQL para DW, etc.)
        """
        pass
    
    @abstractmethod
    def validate_data(self, data: Any) -> None:
        """
        Valida integridade dos dados.
        
        Args:
            data: Dados para validação no formato da arquitetura
            
        Raises:
            ValueError: Quando validação falha
        """
        pass
    
    @abstractmethod
    def create_target_implementation(self, data: Any) -> Any:
        """
        Implementação específica para criar variável target.
        
        Args:
            data: Dados de entrada
            
        Returns:
            Dados com variável target criada
        """
        pass
    
    def create_target(self, data: Any) -> Any:
        """
        Cria variável target com validação científica comum.
        
        Este método implementa a lógica comum de criação de target
        (dropout_rate = 100 - completion_rate) com validações científicas
        idênticas para todas as arquiteturas.
        
        Args:
            data: Dados de entrada no formato da arquitetura
            
        Returns:
            Dados com variável target criada e validada
            
        Inversão simples --- raw_data_collector garante range [0,100].
        """
        print(f"\nCriando target {self.architecture_name}: {self.source_column} -> {self.target_column}")
        
        data_with_target = self.create_target_implementation(data)
        
        self._save_target_statistics(data_with_target)
        
        return data_with_target
    
    @abstractmethod
    def _compute_target_statistics(self, data: Any) -> Dict[str, float]:
        """
        Computa estatísticas do target específicas da arquitetura.
        
        Args:
            data: Dados com target criado
            
        Returns:
            Dicionário com estatísticas (mean, std, min, max, etc.)
        """
        pass
    
    def _save_target_statistics(self, data: Any) -> None:
        """
        Salva estatísticas do target de forma padronizada.
        
        Args:
            data: Dados com target para computar estatísticas
        """
        stats = self._compute_target_statistics(data)
        
        stats.update({
            'architecture': self.architecture_name,
            'target_variable': self.target_column,
            'source_column': self.source_column,
            'creation_timestamp': datetime.now().isoformat()
        })
        
        expected_range = list(self.dataset_config.target_expected_range)
        if stats['mean'] < expected_range[0] or stats['mean'] > expected_range[1]:
            print(f"   Aviso: Média de dropout ({stats['mean']:.2f}%) "
                  f"fora do range esperado {expected_range}")

        if stats['valid_count'] < self.dataset_config.min_valid_count:
            print(f"   Aviso: Poucos dados válidos ({stats['valid_count']}) "
                  f"para ML")
        
        stats_path = f"{self.prep_dir}/target_statistics.json"
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
            
        print(f"   Estatísticas salvas: {stats_path}")
    
    def create_temporal_folds(self, data: Any = None) -> List[Dict]:
        """
        Cria folds temporais científicos com metodologia walk-forward.
        
        Implementa estrutura de validação temporal idêntica para todas
        as arquiteturas, garantindo comparabilidade científica.
        
        Args:
            data: Dados opcionais para validação de folds
            
        Returns:
            Lista de configurações de folds com metadados completos
            
        Aplica gaps temporais de 2 anos entre train/val e val/test
        para prevenção de vazamento temporal.
        """
        print("\nCriando folds temporais...")
        gap = int(self.config.get('temporal_gap_years', 2))
        embargo = int(self.config.get('embargo_years', 0))
        print(f"Metodologia: Walk-forward automático com gaps de {gap} anos"
              + (f" e embargo de {embargo} anos" if embargo > 0 else ""))
        folds = self._generate_walkforward_folds_auto()

        # Impor anti-leakage: interromper em caso de violação
        from core.validation import TemporalValidator
        validator = TemporalValidator(min_gap_years=gap, embargo_years=embargo)
        validator.enforce_walk_forward(folds)

        if data is not None:
            if hasattr(data, 'reset_index'):
                data = data.reset_index(drop=True)
            self._validate_temporal_folds(data, folds)

        return folds

    def _generate_walkforward_folds_auto(self) -> List[Dict]:
        """
        Gera folds walk-forward expansivos automaticamente, respeitando gaps e janelas.

        Parâmetros são lidos de SCIENTIFIC_CONFIG (com defaults seguros):
          - temporal_range_start / end
          - folds_min_train_years
          - folds_val_len_years
          - folds_test_len_years
          - temporal_gap_years
          - folds_step_years
          - folds_max (opcional)
        """
        cfg = self.config
        # Override temporal range e walk-forward do dataset_config se disponível
        ds = self.dataset_config
        start_year = int(ds.temporal_range[0]) if ds else int(cfg.get('temporal_range_start', 2000))
        end_year = int(ds.temporal_range[1]) if ds else int(cfg.get('temporal_range_end', 2023))
        wf = ds.walk_forward_config if ds else {}
        min_train = int(wf.get('min_train', cfg.get('folds_min_train_years', 8)))
        val_len = int(wf.get('val_len', cfg.get('folds_val_len_years', 2)))
        test_len = int(wf.get('test_len', cfg.get('folds_test_len_years', 2)))
        gap = int(cfg.get('temporal_gap_years', 2))
        step = int(wf.get('step', cfg.get('folds_step_years', 1)))
        max_folds = cfg.get('folds_max', None)
        try:
            max_folds = int(max_folds) if max_folds is not None else None
        except Exception:
            max_folds = None

        # Calcular limites de início para a janela de teste
        # Derivação: val_start >= start_year + min_train + gap
        # test_start = val_start + val_len + gap
        # => test_start_min = start_year + min_train + val_len + 2*gap
        test_start_min = start_year + min_train + val_len + 2 * gap
        test_start_max = end_year - test_len + 1

        folds: List[Dict] = []
        fold_id = 0
        for test_start in range(test_start_min, test_start_max + 1, step):
            test_end = test_start + test_len - 1
            # Derivar val_end e val_start a partir do gap
            val_end = test_start - gap - 1
            val_start = val_end - val_len + 1
            # Derivar train_end e train_start
            train_end = val_start - gap - 1
            train_start = start_year

            # Verificações de validade
            if train_end < train_start:
                continue
            train_len = train_end - train_start + 1
            if train_len < min_train:
                continue
            if not (train_start <= train_end < val_start <= val_end < test_start <= test_end <= end_year):
                continue

            train_val_gap = val_start - train_end - 1
            val_test_gap = test_start - val_end - 1
            if train_val_gap < gap or val_test_gap < gap:
                continue

            fold = {
                'fold_id': fold_id,
                'architecture': self.architecture_name,
                'methodology': 'walk_forward_with_gaps_auto',
                'train_start': int(train_start), 'train_end': int(train_end),
                'train_gap_start': int(train_end + 1), 'train_gap_end': int(val_start - 1),
                'val_start': int(val_start), 'val_end': int(val_end),
                'val_gap_start': int(val_end + 1), 'val_gap_end': int(test_start - 1),
                'test_start': int(test_start), 'test_end': int(test_end),
                'total_train_years': int(train_len),
                'total_val_years': int(val_len),
                'total_test_years': int(test_len),
                'train_val_gap': int(train_val_gap),
                'val_test_gap': int(val_test_gap),
                # Separação efetiva entre a última observação usada para ajustar
                # o modelo e a primeira observação avaliada.
                #
                # Registrada porque é maior que o gap declarado, e por decisão: o
                # modelo avaliado no teste é ajustado apenas na janela de treino,
                # e a validação serve exclusivamente para selecionar
                # hiperparâmetros. Reajustar em treino+validação usaria 25% mais
                # anos e aproximaria a origem, mas reduziria esta separação ao
                # mínimo declarado em P2 -- trocaria margem de segurança na
                # garantia anti-leakage por eficiência estatística num
                # dispositivo cuja acurácia não é o objeto de estudo.
                'fit_to_test_gap': int(test_start - train_end - 1),
                'fit_window': 'train_only',
                'description': f'Walk-forward auto (gap={gap}y, val={val_len}y, test={test_len}y)',
                'forecast_horizon': '1-2 anos à frente'
            }
            folds.append(fold)
            fold_id += 1
            if max_folds is not None and len(folds) >= max_folds:
                break

        if not folds:
            raise ValueError(
                f"Nenhum fold pôde ser gerado com os parâmetros atuais. "
                f"Ajuste temporal_range_start/end, folds_min_train_years ou gaps. "
                f"Parâmetros: start={start_year}, end={end_year}, min_train={min_train}, "
                f"val_len={val_len}, test_len={test_len}, gap={gap}"
            )

        print(f"   Folds auto-gerados: {len(folds)} (gap={gap}, val={val_len}, test={test_len})")
        return folds
    
    @abstractmethod
    def _validate_temporal_folds(self, data: Any, folds: List[Dict]) -> None:
        """
        Valida estrutura científica dos folds.
        
        Args:
            data: Dados para validação
            folds: Lista de folds para validar
        """
        pass
    
    @abstractmethod
    def save_folds(self, data: Any, folds: List[Dict]) -> None:
        """
        Salva folds no formato específico da arquitetura.
        
        Args:
            data: Dados processados
            folds: Lista de configurações de folds
        """
        pass
    
    def _filter_by_year(self, data: Any, max_year: int) -> Any:
        """Filtra dados para year <= max_year. Suporta pandas, Dask e Polars."""
        if _HAS_POLARS and isinstance(data, pl.DataFrame):
            return data.filter(pl.col('year') <= max_year)
        elif hasattr(data, 'compute'):  # Dask DataFrame
            return data[data['year'] <= max_year]
        elif isinstance(data, pd.DataFrame):
            return data[data['year'] <= max_year]
        else:
            raise TypeError(f"Tipo de dados não suportado para filtro temporal: {type(data)}")

    @staticmethod
    def _count_rows(data: Any) -> int:
        """Conta linhas de um DataFrame (pandas ou Dask)."""
        if hasattr(data, 'compute'):  # Dask
            return len(data)
        return len(data)

    def _materialise_pandas(self, data: Any, columns: List[str]) -> pd.DataFrame:
        """Materialise the given columns as a pandas frame.

        Mirrors the dispatch of _filter_by_year so that checks needing a dense
        matrix stay in this class instead of becoming a per-paradigm obligation.
        """
        if _HAS_POLARS and isinstance(data, pl.DataFrame):
            return data.select(columns).to_pandas()
        elif hasattr(data, 'compute'):  # Dask DataFrame
            return data[columns].compute()
        elif isinstance(data, pd.DataFrame):
            return data[columns].copy()
        else:
            raise TypeError(f"Unsupported data type for materialisation: {type(data)}")

    def _linear_reconstruction_r2(self, data: Any, features: List[str]) -> Optional[float]:
        """R2 of an ordinary least squares fit of the target on `features`.

        Returns None when the fit is not determined -- too few complete rows for
        the number of predictors, or a constant target.
        """
        if not features:
            return None

        frame = self._materialise_pandas(
            data, list(features) + [self.target_column]
        ).dropna()
        if len(frame) <= len(features) + 1:
            return None

        X = frame[list(features)].to_numpy(dtype=float)
        y = frame[self.target_column].to_numpy(dtype=float)
        design = np.column_stack([X, np.ones(len(X))])
        coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
        residual = y - design @ coefficients
        total = ((y - y.mean()) ** 2).sum()
        if total <= 0:
            return None
        return float(1.0 - (residual ** 2).sum() / total)

    def get_excluded_features(self) -> List[str]:
        """
        Retorna lista de features a excluir (vazamento/metadados).

        Lista harmonizada entre todas as arquiteturas para garantir
        comparação científica justa.

        Returns:
            Lista de nomes de colunas a excluir
        """
        base_excluded = list(self.dataset_config.excluded_columns)
        if self.target_column not in base_excluded:
            base_excluded.append(self.target_column)
        return base_excluded
    
    @abstractmethod
    def compute_feature_correlations(self, data: Any, features: List[str]) -> Dict[str, float]:
        """
        Computa correlações entre features e target.
        
        Args:
            data: Dados com features
            features: Lista de features para análise
            
        Returns:
            Dicionário com correlações absolutas
        """
        pass
    
    def select_features_by_correlation(self, 
                                      correlations: Dict[str, float],
                                      min_corr: float = 0.15,
                                      max_corr: float = 0.8) -> List[str]:
        """
        Seleciona features por correlação moderada com target.
        
        Args:
            correlations: Dicionário de correlações
            min_corr: Correlação mínima (evita irrelevância)
            max_corr: Correlação máxima (evita vazamento)
            
        Returns:
            Lista de features selecionadas
        """
        selected = sorted([
            feat for feat, corr in correlations.items()
            if min_corr <= corr <= max_corr
        ])

        print(f"   Features com correlação moderada ({min_corr}-{max_corr}): "
              f"{len(selected)}")

        # Relaxar critério se muito poucas features
        if len(selected) < 5:
            selected = sorted([
                feat for feat, corr in correlations.items()
                if corr >= min_corr * 0.67
            ])
            print(f"   Critério relaxado: {len(selected)} features")
        
        return selected
    
    @abstractmethod
    def apply_collinearity_filter(self, data: Any, features: List[str],
                                  threshold: float = 0.8) -> List[str]:
        """
        Remove multicolinearidade via filtragem greedy de correlação pairwise.

        Para cada feature candidata, calcula a correlação absoluta máxima com
        as features já selecionadas. Rejeita se max |r| >= threshold.

        Args:
            data: Dados para análise
            features: Features candidatas
            threshold: Threshold de correlação pairwise para remoção

        Returns:
            Lista de features após remoção de multicolinearidade
        """
        pass
    
    def _first_fold_train_end(self) -> int:
        """
        Calcula train_end do primeiro fold a partir da config científica.

        Usado para restringir feature selection ao período de treino,
        prevenindo leakage temporal (Kapoor & Narayanan, 2023): seleção
        de features usando dados que pertencem a validação/teste.
        """
        cfg = self.config
        ds = self.dataset_config
        start_year = int(ds.temporal_range[0]) if ds else int(cfg.get('temporal_range_start', 2000))
        wf = ds.walk_forward_config if ds else {}
        min_train = int(wf.get('min_train', cfg.get('folds_min_train_years', 8)))
        val_len = int(wf.get('val_len', cfg.get('folds_val_len_years', 2)))
        gap = int(cfg.get('temporal_gap_years', 2))
        test_start_min = start_year + min_train + val_len + 2 * gap
        val_end = test_start_min - gap - 1
        val_start = val_end - val_len + 1
        train_end = val_start - gap - 1
        return train_end

    def run_feature_selection(self, data: Any) -> Dict:
        """
        Executa pipeline completo de seleção de features.

        Pipeline padronizado com enforcement anti-leakage:
        1. Remove features com vazamento/metadados (P3)
        2. Restringe dados ao período de treino do primeiro fold (P4)
        3. Seleciona por correlação moderada com target
        4. Remove multicolinearidade via filtragem pairwise
        5. Detecta features proxy do target (P3 estendido)

        Preprocessing (scaling, imputação) ocorre em prepare_features()
        e nos modelos, com enforcement de P5 (escopo de preprocessing).

        P4 (Kapoor & Narayanan, 2023; Kaufman et al., 2012):
        Correlações são computadas usando apenas dados até train_end do
        primeiro fold, impedindo que informação de períodos de validação
        ou teste influencie a seleção de features.

        Args:
            data: Dados para seleção

        Returns:
            Dicionário com estatísticas e features selecionadas
        """
        print(f"\nFeature selection {self.architecture_name}...")

        exclude_cols = self.get_excluded_features()
        feature_cols = self.get_numeric_features(data)

        # P3: o target e a coluna de que ele deriva não podem ser candidatos.
        # Verificado sobre o pool, e não apenas sobre a seleção final: a
        # auditoria de proxy roda depois da seleção, então uma candidata que o
        # teto de correlação descarte nunca chega a ser auditada. Foi assim que
        # a coluna-fonte do target, com correlação -1.0, atravessou o gate.
        # Redundante com a política de candidatas por decisão: uma regressão
        # naquela política aparece aqui como parada, não como contaminação.
        forbidden = {self.target_column, self.source_column} & set(feature_cols)
        if forbidden:
            raise AntiLeakageViolation(
                f"Anti-leakage violation (P3 data separation): candidate pool "
                f"contains the target or the column it is derived from: "
                f"{sorted(forbidden)}"
            )

        print(f"   {len(feature_cols)} candidatas ({len(exclude_cols)} excluidas)")

        # P4: Restringir ao período de treino para evitar leakage na seleção
        # (Kapoor & Narayanan, 2023; Kaufman et al., 2012)
        train_end = self._first_fold_train_end()
        data_train_only = self._filter_by_year(data, max_year=train_end)
        n_total = self._count_rows(data)
        n_train = self._count_rows(data_train_only)
        print(f"   P4: Correlações restritas ao período de treino "
              f"(≤{train_end}): {n_train}/{n_total} observações")

        # Correlação com target (usando apenas dados de treino)
        correlations = self.compute_feature_correlations(data_train_only, feature_cols)
        selected_by_corr = self.select_features_by_correlation(correlations)

        # Filtragem de colinearidade pairwise (usando dados de treino)
        final_features = self.apply_collinearity_filter(
            data_train_only, selected_by_corr,
            float(self.config['collinearity_threshold']))

        # P3: Impor que nenhuma feature excluída/derivada do target esteja na seleção
        leaked = set(final_features) & set(exclude_cols)
        if leaked:
            raise AntiLeakageViolation(
                f"Anti-leakage violation (P3 data separation): "
                f"excluded features found in final selection: {leaked}"
            )

        # P3 extended: proxy detection (Kapoor & Narayanan, 2023; Kaufman et
        # al., 2012).
        #
        # Selection and auditing serve different purposes and use different
        # data. Selection reads only the P4 window, because choosing features by
        # their agreement with future target values is look-ahead bias. The
        # audit below reads the full panel: a feature whose correlation clears
        # the threshold only outside the first training window is still a proxy,
        # and restricting the audit to that window is what let one through here.
        #
        # The audit may only abort, never filter. Aborting reports that the
        # design is invalid without letting full-panel information reach the
        # model; silently dropping the feature would.
        PROXY_THRESHOLD = float(self.config.get('proxy_correlation_threshold', 0.80))
        audit_correlations = self.compute_feature_correlations(data, final_features)
        proxies = {
            feat: corr for feat, corr in audit_correlations.items()
            if feat in final_features and abs(corr) > PROXY_THRESHOLD
        }
        if proxies:
            raise AntiLeakageViolation(
                f"Anti-leakage violation (P3 proxy detection): "
                f"features with |correlation| > {PROXY_THRESHOLD} with target "
                f"over the full panel suggest proxy leakage "
                f"(Kapoor & Narayanan, 2023): {proxies}"
            )

        # P3 extended: joint reconstruction of the target.
        #
        # Pairwise correlation cannot see an additive identity. Where the target
        # partitions into several features -- rates that sum to a constant, for
        # instance -- each one correlates weakly while together they determine
        # the target exactly. Fitted on the training window, so an exact
        # identity is detected without consulting the evaluation periods.
        IDENTITY_THRESHOLD = float(self.config.get('identity_r2_threshold', 0.95))
        identity_r2 = self._linear_reconstruction_r2(data_train_only, final_features)
        if identity_r2 is not None and identity_r2 > IDENTITY_THRESHOLD:
            raise AntiLeakageViolation(
                f"Anti-leakage violation (P3 joint reconstruction): selected "
                f"features explain the target with R2 = {identity_r2:.4f} > "
                f"{IDENTITY_THRESHOLD} on the training window, indicating the "
                f"target is an algebraic function of the feature set: "
                f"{sorted(final_features)}"
            )

        # Estatísticas de seleção
        selection_stats = {
            'architecture': self.architecture_name,
            'total_features_analyzed': len(feature_cols),
            'features_selected': len(final_features),
            'selection_method': 'correlation_pairwise_filter',
            'temporal_scope': f'train_only (≤{train_end})',
            'proxy_threshold': PROXY_THRESHOLD,
            'selected_features': final_features,
            'target_correlations': {
                feat: float(correlations.get(feat, 0))
                for feat in final_features
            },
            'selection_timestamp': datetime.now().isoformat()
        }
        
        selection_path = f"{self.prep_dir}/feature_selection_{self.architecture_name}.json"
        with open(selection_path, 'w') as f:
            json.dump(selection_stats, f, indent=2)
        
        print(f"   Features selecionadas: {len(final_features)} -> {selection_path}")
        
        return selection_stats
    
    @abstractmethod
    def discover_numeric_columns(self, data: Any) -> List[str]:
        """
        Lista as colunas de tipo numérico presentes nos dados.

        Descoberta apenas: cada paradigma inspeciona o schema com seus próprios
        meios (metadados do catálogo, inferência de dtype). A política de quais
        colunas são candidatas legítimas não pertence aqui — vive em
        candidate_exclusions(), única para todos os paradigmas.

        Args:
            data: Dados de entrada

        Returns:
            Lista de nomes de colunas numéricas, em qualquer ordem
        """
        pass

    def candidate_exclusions(self) -> Tuple[set, str]:
        """
        Nomes e prefixo que desqualificam uma coluna como candidata.

        Derivada da configuração, não enumerada. Uma lista enumerada envelhece
        em silêncio: foi assim que a coluna-fonte do target (correlação -1.0 com
        o target) entrou no pool de um paradigma e sobreviveu ao gate P3, sendo
        descartada apenas pelo teto de correlação da seleção.

        O prefixo derivado do target cobre, de uma vez, o target deste
        paradigma, os targets dos demais paradigmas e os lags do target —
        nenhum deles é candidato à seleção.

        Returns:
            (nomes a excluir, prefixo a excluir)
        """
        excluded = set(self.get_excluded_features())
        excluded.add(self.source_column)
        for attr in ('entity_column', 'entity_name_column',
                     'year_column', 'stratification_column'):
            name = getattr(self.dataset_config, attr, None)
            if name:
                excluded.add(name)
        return excluded, f'{self.TARGET_STEM}_'

    def get_numeric_features(self, data: Any) -> List[str]:
        """
        Pool de candidatas à seleção de features.

        Idêntico entre paradigmas por construção: um pool divergente faria a
        comparação cross-paradigma partir de espaços de busca diferentes.

        Args:
            data: Dados de entrada

        Returns:
            Lista ordenada de candidatas
        """
        excluded, derived_prefix = self.candidate_exclusions()

        # Uma feature declarada que casasse com o prefixo derivado seria
        # descartada em silêncio, alterando o resultado sem aviso.
        declared = set(getattr(self.dataset_config, 'feature_columns', None) or ())
        shadowed = {c for c in declared if c.startswith(derived_prefix)} - excluded
        if shadowed:
            raise ValueError(
                f"{self.architecture_name}: declared features collide with the "
                f"prefix reserved for target-derived columns "
                f"('{derived_prefix}'): {sorted(shadowed)}. Rename them or "
                f"change TARGET_STEM; leaving them would drop them silently."
            )

        candidates = sorted(
            col for col in self.discover_numeric_columns(data)
            if col not in excluded and not col.startswith(derived_prefix)
        )

        if len(candidates) < 5:
            print(f"  [WARN] Poucas candidatas ({len(candidates)}) podem "
                  f"limitar capacidade preditiva")
        if len(candidates) > 100:
            print(f"  [WARN] Muitas candidatas ({len(candidates)}) requerem "
                  f"selecao cuidadosa (curse of dimensionality)")

        return candidates


    @abstractmethod
    def prepare_features(self, data: Any, selected_features: List[str]) -> Any:
        """
        Prepara features finais para ML.

        P5 (escopo de preprocessing — Kaufman et al. 2012):
        Implementações devem garantir que qualquer transformação
        estatística (scaling, imputação, encoding) seja ajustada
        exclusivamente nos dados de treino. Estatísticas derivadas
        do conjunto completo (incluindo validação/teste) configuram
        leakage de preprocessing, mesmo quando a separação temporal
        dos folds está correta.

        Padrão exigido nas subclasses:
          - scaler.fit(X_train) → scaler.transform(X_val), scaler.transform(X_test)
          - fillna(reference_data.median()) onde reference_data = train_data
          - NUNCA usar data.median() ou scaler.fit(X_completo)

        Args:
            data: Dados completos
            selected_features: Lista de features selecionadas

        Returns:
            Dados preparados com features finais
        """
        pass
    
    @staticmethod
    def _convert_numpy_types(obj):
        """Converte tipos numpy para tipos Python nativos para serialização JSON."""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: BaseArchitectureML._convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [BaseArchitectureML._convert_numpy_types(v) for v in obj]
        else:
            return obj

    def save_fold_metadata(self, fold: Dict, fold_dir: str) -> None:
        """
        Salva metadados de um fold de forma padronizada.

        Args:
            fold: Configuração do fold
            fold_dir: Diretório do fold
        """
        fold_metadata = {
            **self._convert_numpy_types(fold),
            'data_source': self.architecture_name,
            'target_variable': self.target_column,
            'temporal_boundaries_preserved': True,
            'gaps_applied_effectively': True,
            'saved_timestamp': datetime.now().isoformat()
        }
        
        with open(f'{fold_dir}/metadata.json', 'w') as f:
            json.dump(fold_metadata, f, indent=2)
    
    def save_master_config(self, folds: List[Dict], total_observations: int,
                          total_entities: int, year_range: Tuple[int, int]) -> str:
        """
        Salva configuração master dos folds.

        Args:
            folds: Lista de folds
            total_observations: Total de observações
            total_entities: Total de entidades geográficas (países, municípios)
            year_range: Tupla (ano_min, ano_max)

        Returns:
            Caminho do arquivo de configuração salvo
        """
        folds_config = {
            'architecture': self.architecture_name,
            'creation_timestamp': datetime.now().isoformat(),
            'total_observations': int(total_observations),
            'total_entities': int(total_entities),
            'year_range': [int(year_range[0]), int(year_range[1])],
            'target_variable': self.target_column,
            'folds': self._convert_numpy_types(folds)
        }
        
        folds_path = f"{self.prep_dir}/temporal_folds_{self.architecture_name}.json"
        with open(folds_path, 'w') as f:
            json.dump(folds_config, f, indent=2)
        
        return folds_path
    
    def audit_final_features(self, data: Any, features: List[str]) -> Dict:
        """Apply the P3 checks to the feature set the models train on.

        Delegates to core.validation.audit_feature_set, which the model classes
        call directly since they do not inherit from this class.
        """
        from core.validation import audit_feature_set

        return audit_feature_set(data, features, self.target_column, self.config)

    def run_setup(self) -> Dict:
        """
        Executa pipeline completo de setup da arquitetura.
        
        Pipeline padronizado:
        1. Setup do ambiente
        2. Carregamento de dados
        3. Validação de dados
        4. Criação de target
        5. Seleção de features
        6. Preparação de features
        7. Criação de folds temporais
        8. Salvamento de folds e configurações
        
        Returns:
            Dicionário com resultados do setup
        """
        print(f"Executando setup {self.architecture_name}")
        
        try:
            # Setup do ambiente específico
            self.setup_environment()
            
            # Carregar dados
            data = self.load_data()
            
            # Validar dados
            self.validate_data(data)
            
            # Criar target
            data_with_target = self.create_target(data)
            
            selection_stats = self.run_feature_selection(data_with_target)
            
            data_processed = self.prepare_features(
                data_with_target, 
                selection_stats['selected_features']
            )
            
            # Criar folds temporais
            folds = self.create_temporal_folds(data_processed)
            
            # Salvar folds
            self.save_folds(data_processed, folds)
            
            print(f"\nSetup {self.architecture_name} concluido")
            
            return {
                'architecture': self.architecture_name,
                'status': 'success',
                'setup_timestamp': datetime.now().isoformat(),
                'features_selected': len(selection_stats['selected_features']),
                'folds_created': len(folds)
            }
            
        except AntiLeakageViolation:
            # Never reported as a recoverable failure. A violation means the
            # experiment does not hold the guarantees its results would be
            # reported under, so it must reach the caller and stop the run.
            print(f"\nAnti-leakage violation in {self.architecture_name}")
            raise

        except Exception as e:
            print(f"\nErro no setup {self.architecture_name}: {e}")

            return {
                'architecture': self.architecture_name,
                'status': 'failed',
                'error': str(e),
                'setup_timestamp': datetime.now().isoformat()
            }
    
    def validate_temporal_integrity_years(self, train_years: Tuple[int, int],
                                   val_years: Tuple[int, int],
                                   test_years: Tuple[int, int]) -> bool:
        """
        Valida integridade temporal dos splits para prevenir vazamento.
        
        Verifica se:
        1. Os períodos estão em ordem cronológica correta
        2. Existem gaps temporais adequados entre splits
        3. Não há sobreposição entre períodos
        
        Args:
            train_years: Tupla (ano_inicial, ano_final) do treino
            val_years: Tupla (ano_inicial, ano_final) da validação
            test_years: Tupla (ano_inicial, ano_final) do teste
            
        Returns:
            True se a integridade temporal está preservada, False caso contrário
            
        Requer gap mínimo de 2 anos entre splits para prevenir
        vazamento temporal em dados educacionais.
        """
        # P1: Verificar ordem temporal
        # val/test podem ter 1 ano (start == end), portanto <=
        if not (train_years[1] < val_years[0] <= val_years[1] < test_years[0]):
            raise AntiLeakageViolation(
                f"Anti-leakage violation (P1 temporal ordering): "
                f"Train: {train_years}, Val: {val_years}, Test: {test_years}"
            )

        # P2: Verificar gaps
        train_val_gap = val_years[0] - train_years[1] - 1
        val_test_gap = test_years[0] - val_years[1] - 1

        MIN_GAP = int(self.config.get('temporal_gap_years', 2))

        if train_val_gap < MIN_GAP:
            raise AntiLeakageViolation(
                f"Anti-leakage violation (P2 gap sufficiency): "
                f"train-val gap={train_val_gap} < {MIN_GAP}"
            )

        if val_test_gap < MIN_GAP:
            raise AntiLeakageViolation(
                f"Anti-leakage violation (P2 gap sufficiency): "
                f"val-test gap={val_test_gap} < {MIN_GAP}"
            )

        print(f"   Integridade temporal OK (gaps: train-val={train_val_gap}yr, val-test={val_test_gap}yr)")

        return True
