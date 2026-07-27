                                                                                                    #!/usr/bin/env python3
"""
DuckDB Connection Manager for the Data Warehouse architecture.

Implements the management of persistent DuckDB connections,
including transactional support, automatic retry and error handling.
"""

import duckdb

from core.scientific_config import SCIENTIFIC_CONFIG
import time
import logging
from typing import Optional, List, Any
import pandas as pd


class SQLProcessingError(Exception):
    """Custom exception for SQL processing errors."""
    pass


class DuckDBConnectionManager:
    """
    DuckDB connection manager with transactional support and automatic recovery.

    Features:
    - Persistent connections
    - Automatic retry with exponential backoff
    - Full support for ACID transactions
    - Detailed logging for auditing
    - Context managers for safe resource management
    """
    
    def __init__(self, db_path: str, max_retries: int = 3, retry_delay: float = 1.0):
        """
        Initialise the DuckDB connection manager.
        
        Args:
            db_path: Path to the DuckDB database file
            max_retries: Maximum number of attempts for failed operations
            retry_delay: Base delay between attempts (exponential backoff)
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
        Obtain a persistent DuckDB connection, creating one if needed.

        Returns:
            An active DuckDB connection object

        Raises:
            SQLProcessingError: If the connection could not be created
        """
        if self._connection is None:
            try:
                self.logger.info(f"Creating new DuckDB connection to: {self.db_path}")
                self._connection = duckdb.connect(self.db_path)
                # Explicit core budget: without this the engine sizes its pool
                # by the machine, and the measured latency stops being
                # comparable to another run or to another paradigm.
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
        """Close the current connection, rolling back active transactions."""
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
        """Execute an SQL operation with automatic retry and error handling."""
        for attempt in range(self.max_retries):
            try:
                conn = self.get_connection()
                return operation_func(conn, query, params)
                    
            except duckdb.Error as e:
                self.logger.warning(f"{operation_name} attempt {attempt + 1} failed: {e}")
                
                if attempt == self.max_retries - 1:
                    self.logger.error(f"{operation_name} failed after {self.max_retries} attempts: {query[:100]}...")
                    raise SQLProcessingError(f"{operation_name} failed after {self.max_retries} attempts: {e}")
                
                delay = self.retry_delay * (2 ** attempt)
                self.logger.info(f"Retrying in {delay} seconds...")
                time.sleep(delay)
                self.close_connection()
            
            except Exception as e:
                self.logger.error(f"Unexpected error while executing {operation_name}: {e}")
                raise SQLProcessingError(f"Unexpected {operation_name} error: {e}")
    
    def execute_sql(self, query: str, params: Optional[List[Any]] = None) -> pd.DataFrame:
        """
        Execute an SQL query with automatic retry and return a DataFrame.

        Args:
            query: SQL command to execute
            params: Optional parameters for the query

        Returns:
            Results as a pandas DataFrame (empty if there are no results)

        Raises:
            SQLProcessingError: If the query fails after all attempts
        """
        def _operation(conn, query, params):
            result = conn.execute(query, params) if params else conn.execute(query)
            try:
                df = result.df()
                self.logger.debug(f"Query executed: {len(df)} records returned")
                return df
            except Exception:
                self.logger.debug("Query executed successfully (no results)")
                return pd.DataFrame()

        return self._execute_with_retry("SQL query", query, params, _operation)
    
    def execute_sql_no_return(self, query: str, params: Optional[List[Any]] = None) -> bool:
        """
        Execute an SQL command that returns no data (DDL, DML).

        Args:
            query: SQL command to execute
            params: Optional parameters for the command

        Returns:
            True if executed successfully

        Raises:
            SQLProcessingError: If the command fails after all attempts
        """
        def _operation(conn, query, params):
            conn.execute(query, params) if params else conn.execute(query)
            self.logger.debug("SQL command executed successfully")
            return True

        return self._execute_with_retry("SQL command", query, params, _operation)
    
    def execute_scalar(self, query: str, params: Optional[List[Any]] = None) -> Any:
        """
        Execute an SQL query and return a single scalar value (first row, first column).

        Args:
            query: SQL query returning a single value
            params: Optional parameters for the query

        Returns:
            The single scalar value from the query result

        Raises:
            SQLProcessingError: If the query fails or returns no results
        """
        def _operation(conn, query, params):
            result = conn.execute(query, params).fetchone() if params else conn.execute(query).fetchone()
            if result is None:
                raise SQLProcessingError(f"Query returned no results: {query}")
            scalar_value = result[0] if isinstance(result, (list, tuple)) else result
            self.logger.debug(f"Scalar query returned: {scalar_value}")
            return scalar_value

        return self._execute_with_retry("Scalar query", query, params, _operation)
    
    def execute_transaction(self, queries: List[str], params_list: Optional[List[List[Any]]] = None) -> bool:
            """
            Execute multiple SQL queries in a single transaction.

            Args:
                queries: List of strings containing SQL commands.
                params_list: (Optional) List of parameter lists corresponding to each query.

            Returns:
                True if all queries are executed successfully, False otherwise.

            Raises:
                SQLProcessingError: If the transaction fails.
            """
            if self._transaction_active:
                raise SQLProcessingError("A transaction is already active")
    
            conn = self.get_connection()
    
            try:
                conn.begin()
                self._transaction_active = True
                self.logger.info(f"Beginning transaction with {len(queries)} queries")

                for i, query in enumerate(queries):
                    params = params_list[i] if params_list and i < len(params_list) else None
    
                    if params:
                        conn.execute(query, params)
                    else:
                        conn.execute(query)

                    self.logger.debug(f"Transaction statement {i + 1}/{len(queries)} executed")
                conn.commit()
                self._transaction_active = False
                self.logger.info("Transaction committed successfully")
                return True
    
            except Exception as e:
                try:
                    conn.rollback()
                    self._transaction_active = False
                    self.logger.warning("Transaction failed - rollback executed")
                except Exception as rollback_error:
                    self.logger.error(f"Error during rollback: {rollback_error}")

                self.logger.error(f"Transaction failed: {e}")
                raise SQLProcessingError(f"Transaction failed: {e}")
    
    def create_view(self, view_name: str, query: str, replace: bool = True) -> bool:
        """
        Create or replace an SQL view.

        Args:
            view_name: Name of the view to create
            query: SQL query defining the view
            replace: Whether to use CREATE OR REPLACE (default: True)

        Returns:
            True if creation succeeded

        Raises:
            SQLProcessingError: If view creation fails
        """
        create_statement = "CREATE OR REPLACE VIEW" if replace else "CREATE VIEW"
        full_query = f"{create_statement} {view_name} AS {query}"
        
        self.logger.info(f"Creating view: {view_name}")
        success = self.execute_sql_no_return(full_query)
        
        if success:
            self.logger.info(f"View {view_name} created successfully")
        
        return success
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - closes the connection."""
        self.close_connection()