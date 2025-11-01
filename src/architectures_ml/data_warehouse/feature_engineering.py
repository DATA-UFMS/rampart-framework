#!/usr/bin/env python3
"""
Feature Engineering refatorado - DataWarehouse.

Usa módulo centralizado FeatureEngineer, preservando lógica científica
específica da arquitetura.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from core.features import FeatureEngineer
from architectures_ml.data_warehouse.setup import DataWarehouseArchitectureML


class DataWarehouseFeatureEngineering:
    """
    Feature Engineering para DataWarehouse.
    
    Wrapper que usa FeatureEngineer centralizado mas preserva
    customizações específicas da arquitetura.
    """
    
    def __init__(self):
        """Inicializa com setup da arquitetura."""
        self.setup = DataWarehouseArchitectureML()
        self.engineer = FeatureEngineer()
        self.architecture = 'data_warehouse'
    
    def create_features(self, df):
        """
        Cria features usando módulo centralizado.
        
        Preserva 100% da lógica científica original através
        de parâmetros específicos da arquitetura.
        """
        # Criar lags (preservando configuração original)
        df = self.engineer.create_lag_features(
            df, 
            columns=['gdp_per_capita', 'population_total'],
            lags=[1, 2, 3],
            group_by=['country_code']
        )
        
        # Criar rolling statistics
        df = self.engineer.create_rolling_features(
            df,
            columns=['inflation_rate', 'unemployment_rate'],
            windows=[3, 5],
            functions=['mean', 'std'],
            group_by=['country_code']
        )
        
        # Features de tendência temporal
        df = self.engineer.create_temporal_trend_features(
            df,
            columns=['gdp_growth', 'education_expenditure'],
            year_col='year',
            group_by=['country_code']
        )
        
        # Interações (específicas da arquitetura)
        if self.architecture == 'data_lake':
            # Data Lake: mais interações por capacidade distribuída
            df = self.engineer.create_interaction_features(
                df,
                feature_pairs=[
                    ('gdp_per_capita', 'education_expenditure'),
                    ('population_0_14', 'school_enrollment_primary'),
                    ('literacy_rate', 'internet_users')
                ],
                operations=['ratio', 'multiply']
            )
        else:
            # Data Warehouse: interações seletivas via SQL
            df = self.engineer.create_interaction_features(
                df,
                feature_pairs=[
                    ('gdp_per_capita', 'education_expenditure')
                ],
                operations=['ratio']
            )
        
        return df
    
    def validate_features(self, df):
        """Valida qualidade das features criadas."""
        return self.engineer.validate_feature_quality(
            df,
            columns=df.columns.tolist(),
            min_non_null_ratio=0.5,
            max_constant_ratio=0.95
        )


def main():
    """Função principal para teste."""
    fe = DataWarehouseFeatureEngineering()
    print(f"Feature Engineering DataWarehouse refatorado inicializado")


if __name__ == "__main__":
    main()
