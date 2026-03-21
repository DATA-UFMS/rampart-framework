#!/usr/bin/env python3
"""
Módulo unificado para modelos baseline.

Implementa interface unificada para RandomForest, XGBoost e LightGBM
através do Strategy Pattern.
"""

import os
import json
import sys
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime
from abc import ABC, abstractmethod

_baseline_dir = os.path.dirname(os.path.abspath(__file__))
_core_dir = os.path.dirname(_baseline_dir)
_project_root = os.path.dirname(os.path.dirname(_core_dir))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
from src.core.scientific_config import RANDOM_SEED

# Importações condicionais para modelos
try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False


class BaselineModelStrategy(ABC):
    """
    Estratégia abstrata para modelos baseline.

    Define interface comum para diferentes implementações de modelos,
    permitindo alternância entre backends sem modificar código cliente.
    """

    def __init__(self, model_name: str, params: Optional[Dict] = None):
        """
        Inicializa estratégia de modelo.

        Args:
            model_name: Nome do modelo
            params: Hiperparâmetros customizados
        """
        self.model_name = model_name
        self.params = params or {}
        self.model = None
        self.training_history = []

    @abstractmethod
    def train(self, X_train: Any, y_train: Any,
              X_val: Optional[Any] = None, y_val: Optional[Any] = None) -> None:
        """Treina o modelo."""
        pass

    @abstractmethod
    def predict(self, X: Any) -> np.ndarray:
        """Realiza predições."""
        pass

    @abstractmethod
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Retorna importância das features."""
        pass

    def evaluate(self, X: Any, y_true: Any) -> Dict[str, float]:
        """
        Avalia modelo com métricas padrão.

        Args:
            X: Features para predição
            y_true: Valores verdadeiros

        Returns:
            Dicionário com métricas de avaliação
        """
        y_pred = self.predict(X)

        # Converter para numpy se necessário
        if hasattr(y_true, 'values'):
            y_true = y_true.values
        if hasattr(y_pred, 'values'):
            y_pred = y_pred.values

        metrics = {
            'mae': float(mean_absolute_error(y_true, y_pred)),
            'mse': float(mean_squared_error(y_true, y_pred)),
            'rmse': float(np.sqrt(mean_squared_error(y_true, y_pred))),
            'r2': float(r2_score(y_true, y_pred)),
            'mape': float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100) if np.all(y_true != 0) else np.nan
        }

        return metrics


class RandomForestStrategy(BaselineModelStrategy):
    """
    Estratégia para Random Forest.
    """

    def __init__(self, params: Optional[Dict] = None):
        """
        Inicializa Random Forest.

        Args:
            params: Hiperparâmetros customizados
        """
        super().__init__("RandomForest", params)

        # Parâmetros padrão otimizados para educação
        default_params = {
            'n_estimators': 100,
            'max_depth': 15,
            'min_samples_split': 20,
            'min_samples_leaf': 10,
            'max_features': 'sqrt',
            'random_state': RANDOM_SEED,
            'n_jobs': 1
        }
        default_params.update(self.params)
        self.params = default_params

    def train(self, X_train: Any, y_train: Any,
              X_val: Optional[Any] = None, y_val: Optional[Any] = None) -> None:
        """
        Treina Random Forest.

        Args:
            X_train: Features de treino
            y_train: Target de treino
            X_val: Features de validação (opcional)
            y_val: Target de validação (opcional)
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn não está instalado")

        self.model = RandomForestRegressor(**self.params)
        self.model.fit(X_train, y_train)

        # Registrar histórico
        training_info = {
            'timestamp': datetime.now().isoformat(),
            'train_samples': len(X_train),
            'features': X_train.shape[1] if hasattr(X_train, 'shape') else len(X_train.columns),
            'params': self.params
        }

        if X_val is not None and y_val is not None:
            val_metrics = self.evaluate(X_val, y_val)
            training_info['validation_metrics'] = val_metrics

        self.training_history.append(training_info)

    def predict(self, X: Any) -> np.ndarray:
        """Realiza predições com Random Forest."""
        if self.model is None:
            raise ValueError("Modelo não treinado")
        return self.model.predict(X)

    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Retorna importância das features do Random Forest.

        Nota: nomes genéricos (feature_0..N) porque o modelo
        recebe numpy arrays sem metadata de colunas.
        """
        if self.model is None or not hasattr(self.model, 'feature_importances_'):
            return None
        importances = self.model.feature_importances_
        return {f"feature_{i}": float(imp) for i, imp in enumerate(importances)}


class XGBoostStrategy(BaselineModelStrategy):
    """
    Estratégia para XGBoost.
    """

    def __init__(self, params: Optional[Dict] = None):
        """
        Inicializa XGBoost.

        Args:
            params: Hiperparâmetros customizados
        """
        super().__init__("XGBoost", params)

        # Parâmetros padrão otimizados
        default_params = {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'objective': 'reg:squarederror',
            'random_state': RANDOM_SEED,
            'n_jobs': 1
        }
        default_params.update(self.params)
        self.params = default_params

    def train(self, X_train: Any, y_train: Any,
              X_val: Optional[Any] = None, y_val: Optional[Any] = None) -> None:
        """
        Treina XGBoost.

        Args:
            X_train: Features de treino
            y_train: Target de treino
            X_val: Features de validação (opcional)
            y_val: Target de validação (opcional)
        """
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost não está instalado")

        self.model = xgb.XGBRegressor(**self.params)

        eval_set = None
        if X_val is not None and y_val is not None:
            eval_set = [(X_val, y_val)]

        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            verbose=False
        )

        # Registrar histórico
        training_info = {
            'timestamp': datetime.now().isoformat(),
            'train_samples': len(X_train) if hasattr(X_train, '__len__') else X_train.shape[0],
            'features': X_train.shape[1] if hasattr(X_train, 'shape') else len(X_train.columns),
            'params': self.params
        }

        if X_val is not None and y_val is not None:
            val_metrics = self.evaluate(X_val, y_val)
            training_info['validation_metrics'] = val_metrics

        self.training_history.append(training_info)

    def predict(self, X: Any) -> np.ndarray:
        """Realiza predições com XGBoost."""
        if self.model is None:
            raise ValueError("Modelo não treinado")
        return self.model.predict(X)

    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Retorna importância das features do XGBoost."""
        if self.model is None:
            return None

        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            return {f"feature_{i}": float(imp) for i, imp in enumerate(importances)}
        elif hasattr(self.model, 'get_score'):
            return self.model.get_score(importance_type='gain')

        return None


class LightGBMStrategy(BaselineModelStrategy):
    """
    Estratégia para LightGBM.
    """

    def __init__(self, params: Optional[Dict] = None):
        """
        Inicializa LightGBM.

        Args:
            params: Hiperparâmetros customizados
        """
        super().__init__("LightGBM", params)

        # Parâmetros padrão otimizados
        default_params = {
            'n_estimators': 100,
            'max_depth': -1,
            'num_leaves': 31,
            'learning_rate': 0.1,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'objective': 'regression',
            'metric': 'rmse',
            'random_state': RANDOM_SEED,
            'n_jobs': 1,
            'verbose': -1
        }
        default_params.update(self.params)
        self.params = default_params

    def train(self, X_train: Any, y_train: Any,
              X_val: Optional[Any] = None, y_val: Optional[Any] = None) -> None:
        """
        Treina LightGBM.

        Args:
            X_train: Features de treino
            y_train: Target de treino
            X_val: Features de validação (opcional)
            y_val: Target de validação (opcional)
        """
        if not LIGHTGBM_AVAILABLE:
            raise ImportError("LightGBM não está instalado")

        self.model = lgb.LGBMRegressor(**self.params)

        eval_set = None
        if X_val is not None and y_val is not None:
            eval_set = [(X_val, y_val)]

        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            eval_metric='rmse',
            callbacks=[lgb.log_evaluation(0)]
        )

        # Registrar histórico
        training_info = {
            'timestamp': datetime.now().isoformat(),
            'train_samples': len(X_train) if hasattr(X_train, '__len__') else X_train.shape[0],
            'features': X_train.shape[1] if hasattr(X_train, 'shape') else len(X_train.columns),
            'params': self.params
        }

        if X_val is not None and y_val is not None:
            val_metrics = self.evaluate(X_val, y_val)
            training_info['validation_metrics'] = val_metrics

        self.training_history.append(training_info)

    def predict(self, X: Any) -> np.ndarray:
        """Realiza predições com LightGBM."""
        if self.model is None:
            raise ValueError("Modelo não treinado")
        return self.model.predict(X)

    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Retorna importância das features do LightGBM."""
        if self.model is None:
            return None

        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            return {f"feature_{i}": float(imp) for i, imp in enumerate(importances)}
        elif hasattr(self.model, 'feature_importance'):
            importances = self.model.feature_importance(importance_type='gain')
            return {f"feature_{i}": float(imp) for i, imp in enumerate(importances)}

        return None


class BaselineModelFactory:
    """
    Factory para criação de modelos baseline.

    Centraliza criação de modelos e seleção de estratégias,
    facilitando extensão e manutenção.
    """

    @staticmethod
    def create_model(model_type: str,
                    params: Optional[Dict] = None,
                    use_dask: bool = False) -> BaselineModelStrategy:
        """
        Cria modelo baseline com estratégia apropriada.

        Args:
            model_type: Tipo do modelo ('rf', 'xgboost', 'lightgbm')
            params: Hiperparâmetros customizados
            use_dask: Deprecated, mantido por compatibilidade (ignorado)

        Returns:
            Instância da estratégia de modelo apropriada

        Raises:
            ValueError: Se tipo de modelo não é suportado
        """
        model_type = model_type.lower()

        if model_type in ['rf', 'randomforest', 'random_forest']:
            return RandomForestStrategy(params)
        elif model_type in ['xgb', 'xgboost']:
            return XGBoostStrategy(params)
        elif model_type in ['lgb', 'lightgbm', 'lgbm']:
            return LightGBMStrategy(params)
        else:
            raise ValueError(f"Tipo de modelo não suportado: {model_type}")

    @staticmethod
    def get_available_models() -> List[str]:
        """
        Retorna lista de modelos disponíveis.

        Returns:
            Lista com nomes dos modelos disponíveis
        """
        models = []

        if SKLEARN_AVAILABLE:
            models.append('RandomForest')
        if XGBOOST_AVAILABLE:
            models.append('XGBoost')
        if LIGHTGBM_AVAILABLE:
            models.append('LightGBM')

        return models

    @staticmethod
    def get_default_params(model_type: str) -> Dict:
        """
        Retorna parâmetros padrão para um tipo de modelo.

        Args:
            model_type: Tipo do modelo

        Returns:
            Dicionário com parâmetros padrão
        """
        model_type = model_type.lower()

        if model_type in ['rf', 'randomforest', 'random_forest']:
            return {
                'n_estimators': 100,
                'max_depth': 15,
                'min_samples_split': 20,
                'min_samples_leaf': 10,
                'max_features': 'sqrt',
                'random_state': RANDOM_SEED
            }
        elif model_type in ['xgb', 'xgboost']:
            return {
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'objective': 'reg:squarederror',
                'random_state': RANDOM_SEED
            }
        elif model_type in ['lgb', 'lightgbm', 'lgbm']:
            return {
                'n_estimators': 100,
                'num_leaves': 31,
                'learning_rate': 0.1,
                'feature_fraction': 0.8,
                'bagging_fraction': 0.8,
                'objective': 'regression',
                'metric': 'rmse',
                'random_state': RANDOM_SEED
            }
        else:
            return {}


class BaselineEnsemble:
    """
    Ensemble de modelos baseline para melhor performance.

    Combina predições de múltiplos modelos usando média ponderada
    ou voting para robustez aumentada.
    """

    def __init__(self, models: List[Tuple[str, Optional[Dict], float]],
                 use_dask: bool = False):
        """
        Inicializa ensemble.

        Args:
            models: Lista de tuplas (tipo_modelo, params, peso)
            use_dask: Deprecated, mantido por compatibilidade (ignorado)
        """
        self.models = []
        self.weights = []

        for model_type, params, weight in models:
            model = BaselineModelFactory.create_model(model_type, params)
            self.models.append(model)
            self.weights.append(weight)

        # Normalizar pesos
        total_weight = sum(self.weights)
        self.weights = [w / total_weight for w in self.weights]

    def train(self, X_train: Any, y_train: Any,
              X_val: Optional[Any] = None, y_val: Optional[Any] = None) -> None:
        """
        Treina todos os modelos do ensemble.

        Args:
            X_train: Features de treino
            y_train: Target de treino
            X_val: Features de validação (opcional)
            y_val: Target de validação (opcional)
        """
        for model in self.models:
            print(f"Treinando {model.model_name}...")
            model.train(X_train, y_train, X_val, y_val)

    def predict(self, X: Any) -> np.ndarray:
        """
        Realiza predições com ensemble (média ponderada).

        Args:
            X: Features para predição

        Returns:
            Array com predições do ensemble
        """
        predictions = []

        for model, weight in zip(self.models, self.weights):
            pred = model.predict(X)
            predictions.append(pred * weight)

        return np.sum(predictions, axis=0)

    def evaluate(self, X: Any, y_true: Any) -> Dict[str, Any]:
        """
        Avalia ensemble e modelos individuais.

        Args:
            X: Features para predição
            y_true: Valores verdadeiros

        Returns:
            Dicionário com métricas do ensemble e modelos individuais
        """
        # Métricas do ensemble
        y_pred = self.predict(X)

        if hasattr(y_true, 'values'):
            y_true = y_true.values

        ensemble_metrics = {
            'mae': float(mean_absolute_error(y_true, y_pred)),
            'mse': float(mean_squared_error(y_true, y_pred)),
            'rmse': float(np.sqrt(mean_squared_error(y_true, y_pred))),
            'r2': float(r2_score(y_true, y_pred))
        }

        # Métricas individuais
        individual_metrics = {}
        for model in self.models:
            individual_metrics[model.model_name] = model.evaluate(X, y_true)

        return {
            'ensemble': ensemble_metrics,
            'individual': individual_metrics,
            'weights': self.weights
        }
