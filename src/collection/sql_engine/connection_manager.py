                                                                                                    #!/usr/bin/env python3
"""
Gerenciador de Conexões DuckDB para Arquitetura Data Warehouse.

Implementa gerenciamento de conexões persistentes com DuckDB,
incluindo suporte transacional, retry automático e tratamento de erros.
"""

import duckdb

from core.scientific_config import SCIENTIFIC_CONFIG
import time
import logging
from typing import Optional, List, Any
import pandas as pd


class SQLProcessingError(Exception):
    """Exceção personalizada para erros de processamento SQL."""
    pass


class DuckDBConnectionManager:
    """
    Gerenciador de conexões DuckDB com suporte transacional e recuperação automática.

    Características:
    - Conexões persistentes
    - Retry automático com backoff exponencial
    - Suporte completo a transações ACID
    - Logging detalhado para auditoria
    - Context managers para gerenciamento seguro de recursos
    """
    
    def __init__(self, db_path: str, max_retries: int = 3, retry_delay: float = 1.0):
        """
        Inicializa o gerenciador de conexões DuckDB.
        
        Args:
            db_path: Caminho para o arquivo de banco DuckDB
            max_retries: Número máximo de tentativas para operações falhadas
            retry_delay: Delay base entre tentativas (backoff exponencial)
        """
        self.db_path = db_path
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self._connection: Optional[duckdb.DuckDBPyConnection] = None
        self._transaction_active = False
        
        self.logger = logging.getLogger(__name__)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
    
    def get_connection(self) -> duckdb.DuckDBPyConnection:
        """
        Obtém conexão persistente DuckDB, criando se necessário.
        
        Returns:
            Objeto de conexão DuckDB ativo
            
        Raises:
            SQLProcessingError: Se falha na criação da conexão
        """
        if self._connection is None:
            try:
                self.logger.info(f"Creating new DuckDB connection to: {self.db_path}")
                self._connection = duckdb.connect(self.db_path)
                # Orçamento explícito de núcleos: sem isto o engine dimensiona
                # seu pool pela máquina, e a latência medida deixa de ser
                # comparável a outra execução ou a outro paradigma.
                threads = int(SCIENTIFIC_CONFIG['engine_threads'])
                self._connection.execute(f"SET threads = {threads}")
                self.logger.info(
                    f"DuckDB connection established successfully "
                    f"({threads} threads)")
            except Exception as e:
                self.logger.error(f"Failed to create DuckDB connection: {e}")
                raise SQLProcessingError(f"Failed to connect to DuckDB: {e}")
        
        return self._connection
    
    def close_connection(self):
        """Fecha a conexão atual, fazendo rollback de transações ativas."""
        if self._connection is not None:
            try:
                if self._transaction_active:
                    self.logger.warning("Closing connection with active transaction - rolling back")
                    self._connection.rollback()
                    self._transaction_active = False
                
                self._connection.close()
                self._connection = None
                self.logger.info("DuckDB connection closed")
            except Exception as e:
                self.logger.error(f"Error closing connection: {e}")
    
    def _execute_with_retry(self, operation_name: str, query: str, params: Optional[List[Any]], operation_func):
        """Executa operação SQL com retry automático e tratamento de erros."""
        for attempt in range(self.max_retries):
            try:
                conn = self.get_connection()
                return operation_func(conn, query, params)
                    
            except duckdb.Error as e:
                self.logger.warning(f"Tentativa {operation_name} {attempt + 1} falhou: {e}")
                
                if attempt == self.max_retries - 1:
                    self.logger.error(f"{operation_name} falhou após {self.max_retries} tentativas: {query[:100]}...")
                    raise SQLProcessingError(f"{operation_name} falhou após {self.max_retries} tentativas: {e}")
                
                delay = self.retry_delay * (2 ** attempt)
                self.logger.info(f"Retry em {delay} segundos...")
                time.sleep(delay)
                self.close_connection()
            
            except Exception as e:
                self.logger.error(f"Erro inesperado na execução {operation_name}: {e}")
                raise SQLProcessingError(f"Erro {operation_name} inesperado: {e}")
    
    def execute_sql(self, query: str, params: Optional[List[Any]] = None) -> pd.DataFrame:
        """
        Executa consulta SQL com retry automático e retorna DataFrame.
        
        Args:
            query: Comando SQL para execução
            params: Parâmetros opcionais para a consulta
            
        Returns:
            Resultados como pandas DataFrame (vazio se sem resultados)
            
        Raises:
            SQLProcessingError: Se consulta falha após todas as tentativas
        """
        def _operation(conn, query, params):
            result = conn.execute(query, params) if params else conn.execute(query)
            try:
                df = result.df()
                self.logger.debug(f"Query executada: {len(df)} registros retornados")
                return df
            except Exception:
                self.logger.debug("Query executada com sucesso (sem resultados)")
                return pd.DataFrame()
        
        return self._execute_with_retry("Query SQL", query, params, _operation)
    
    def execute_sql_no_return(self, query: str, params: Optional[List[Any]] = None) -> bool:
        """
        Executa comando SQL sem retorno de dados (DDL, DML).
        
        Args:
            query: Comando SQL para execução
            params: Parâmetros opcionais para o comando
            
        Returns:
            True se executado com sucesso
            
        Raises:
            SQLProcessingError: Se comando falha após todas as tentativas
        """
        def _operation(conn, query, params):
            conn.execute(query, params) if params else conn.execute(query)
            self.logger.debug("Comando SQL executado com sucesso")
            return True
        
        return self._execute_with_retry("Comando SQL", query, params, _operation)
    
    def execute_scalar(self, query: str, params: Optional[List[Any]] = None) -> Any:
        """
        Executa consulta SQL e retorna valor escalar único (primeira linha, primeira coluna).
        
        Args:
            query: Consulta SQL que retorna um único valor
            params: Parâmetros opcionais para a consulta
            
        Returns:
            Valor escalar único do resultado da consulta
            
        Raises:
            SQLProcessingError: Se consulta falha ou não retorna resultados
        """
        def _operation(conn, query, params):
            result = conn.execute(query, params).fetchone() if params else conn.execute(query).fetchone()
            if result is None:
                raise SQLProcessingError(f"Query não retornou resultados: {query}")
            scalar_value = result[0] if isinstance(result, (list, tuple)) else result
            self.logger.debug(f"Query escalar retornou: {scalar_value}")
            return scalar_value
        
        return self._execute_with_retry("Query escalar", query, params, _operation)
    
    def execute_transaction(self, queries: List[str], params_list: Optional[List[List[Any]]] = None) -> bool:
            """
            Executa múltiplas consultas SQL em uma única transação.

            Args:
                queries: Lista de strings contendo comandos SQL.
                params_list: (Opcional) Lista de listas de parâmetros correspondentes a cada query.

            Returns:
                True se todas as queries forem executadas com sucesso, False caso contrário.

            Raises:
                SQLProcessingError: Se a transação falhar.
            """
            if self._transaction_active:
                raise SQLProcessingError("Transação já está ativa")
    
            conn = self.get_connection()
    
            try:
                conn.begin()
                self._transaction_active = True
                self.logger.info(f"Começando transação com {len(queries)} queries")

                for i, query in enumerate(queries):
                    params = params_list[i] if params_list and i < len(params_list) else None
    
                    if params:
                        conn.execute(query, params)
                    else:
                        conn.execute(query)

                    self.logger.debug(f"Transação {i + 1}/{len(queries)} executada")
                conn.commit()
                self._transaction_active = False
                self.logger.info("Transação comitada com sucesso")
                return True
    
            except Exception as e:
                try:
                    conn.rollback()
                    self._transaction_active = False
                    self.logger.warning("Transação falhou - rollback executado")
                except Exception as rollback_error:
                    self.logger.error(f"Error during rollback: {rollback_error}")

                self.logger.error(f"Transação falhou: {e}")
                raise SQLProcessingError(f"Transação falhou: {e}")
    
    def create_view(self, view_name: str, query: str, replace: bool = True) -> bool:
        """
        Cria ou substitui uma view SQL.
        
        Args:
            view_name: Nome da view a ser criada
            query: Consulta SQL que define a view
            replace: Se deve usar CREATE OR REPLACE (padrão: True)
            
        Returns:
            True se criação foi bem-sucedida
            
        Raises:
            SQLProcessingError: Se criação da view falha
        """
        create_statement = "CREATE OR REPLACE VIEW" if replace else "CREATE VIEW"
        full_query = f"{create_statement} {view_name} AS {query}"
        
        self.logger.info(f"Criando view: {view_name}")
        success = self.execute_sql_no_return(full_query)
        
        if success:
            self.logger.info(f"View {view_name} criada com sucesso")
        
        return success
    
    def __enter__(self):
        """Entrada do context manager."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Saída do context manager - fecha conexão."""
        self.close_connection()