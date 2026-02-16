#!/usr/bin/env python3
"""
Configuração de logging estruturado para o projeto ML.

Implementa logging centralizado com:
- Múltiplos handlers (console, arquivo, JSON)
- Níveis configuráveis por módulo
- Contexto estruturado
- Rotação de logs
- Performance tracking
"""

import os
import sys
import json
import logging
import logging.handlers
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, Union
from functools import wraps
import time
import traceback
from contextlib import contextmanager


class StructuredFormatter(logging.Formatter):
    """
    Formatter para logs estruturados em JSON.
    
    Adiciona contexto relevante como timestamp, módulo, função,
    e informações de erro quando aplicável.
    """
    
    def format(self, record: logging.LogRecord) -> str:
        """
        Formata o log record como JSON estruturado.
        
        Args:
            record: LogRecord a formatar
            
        Returns:
            String JSON formatada
        """
        log_obj = {
            'timestamp': datetime.now().astimezone().isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
            'message': record.getMessage(),
            'process_id': os.getpid(),
        }
        
        if hasattr(record, 'context'):
            log_obj['context'] = record.context
        
        if record.exc_info:
            log_obj['exception'] = {
                'type': record.exc_info[0].__name__,
                'message': str(record.exc_info[1]),
                'traceback': traceback.format_exception(*record.exc_info)
            }
        
        if hasattr(record, 'duration'):
            log_obj['performance'] = {
                'duration_seconds': record.duration
            }
        
        if hasattr(record, 'ml_context'):
            log_obj['ml'] = record.ml_context
        
        return json.dumps(log_obj, default=str)


class ColoredConsoleFormatter(logging.Formatter):
    """
    Formatter colorido para console.
    
    Usa cores ANSI para melhor visualização no terminal.
    """
    
    # Cores ANSI
    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m',  # Magenta
    }
    RESET = '\033[0m'
    BOLD = '\033[1m'
    
    def format(self, record: logging.LogRecord) -> str:
        """
        Formata o log record com cores.
        
        Args:
            record: LogRecord a formatar
            
        Returns:
            String formatada com cores ANSI
        """
        levelname = record.levelname
        if levelname in self.COLORS:
            record.levelname = f"{self.COLORS[levelname]}{self.BOLD}{levelname}{self.RESET}"
        
        timestamp = datetime.fromtimestamp(record.created).strftime('%H:%M:%S')
        
        if hasattr(record, 'context'):
            context_str = ' | '.join(f"{k}={v}" for k, v in record.context.items())
            message = f"[{timestamp}] {record.levelname} [{record.name}] {record.getMessage()} | {context_str}"
        else:
            message = f"[{timestamp}] {record.levelname} [{record.name}] {record.getMessage()}"
        
        if record.exc_info:
            message += f"\n{self.COLORS.get('ERROR', '')}{self.formatException(record.exc_info)}{self.RESET}"
        
        return message


class MLContextLogger:
    """
    Logger com contexto específico para ML.
    
    Adiciona automaticamente contexto relevante para
    operações de machine learning.
    """
    
    def __init__(self, logger: logging.Logger):
        """
        Inicializa o logger com contexto ML.
        
        Args:
            logger: Logger base
        """
        self.logger = logger
        self.context = {}
        self.ml_context = {}
    
    def set_context(self, **kwargs):
        """Define contexto permanente para todos os logs."""
        self.context.update(kwargs)
    
    def set_ml_context(self, **kwargs):
        """Define contexto ML específico."""
        self.ml_context.update(kwargs)
    
    def clear_context(self):
        """Limpa o contexto."""
        self.context.clear()
        self.ml_context.clear()
    
    def _log_with_context(self, level: int, msg: str, **kwargs):
        """Log com contexto adicionado."""
        extra = {'context': {**self.context, **kwargs}}
        if self.ml_context:
            extra['ml_context'] = self.ml_context
        self.logger.log(level, msg, extra=extra)
    
    def debug(self, msg: str, **kwargs):
        """Log de debug com contexto."""
        self._log_with_context(logging.DEBUG, msg, **kwargs)
    
    def info(self, msg: str, **kwargs):
        """Log de info com contexto."""
        self._log_with_context(logging.INFO, msg, **kwargs)
    
    def warning(self, msg: str, **kwargs):
        """Log de warning com contexto."""
        self._log_with_context(logging.WARNING, msg, **kwargs)
    
    def error(self, msg: str, **kwargs):
        """Log de erro com contexto."""
        self._log_with_context(logging.ERROR, msg, **kwargs)
    
    def critical(self, msg: str, **kwargs):
        """Log crítico com contexto."""
        self._log_with_context(logging.CRITICAL, msg, **kwargs)
    
    @contextmanager
    def timer(self, operation: str):
        """
        Context manager para medir tempo de operações.
        
        Args:
            operation: Nome da operação
            
        Example:
            with logger.timer('model_training'):
                model.fit(X, y)
        """
        start_time = time.time()
        self.info(f"Iniciando: {operation}")
        
        try:
            yield
        finally:
            duration = time.time() - start_time
            extra = {'context': self.context, 'duration': duration}
            if self.ml_context:
                extra['ml_context'] = self.ml_context
            
            self.logger.info(
                f"Concluído: {operation} ({duration:.2f}s)",
                extra=extra
            )
    
    def log_model_metrics(self, model_name: str, metrics: Dict[str, float], 
                         phase: str = 'test'):
        """
        Log de métricas de modelo.
        
        Args:
            model_name: Nome do modelo
            metrics: Dicionário de métricas
            phase: Fase (train/val/test)
        """
        self.set_ml_context(
            model=model_name,
            phase=phase,
            metrics=metrics
        )
        
        metrics_str = ', '.join(f"{k}={v:.4f}" for k, v in metrics.items())
        self.info(f"Métricas {phase} para {model_name}: {metrics_str}")
    
    def log_data_info(self, dataset_name: str, shape: tuple, 
                      missing_pct: float = None):
        """
        Log de informações sobre dados.
        
        Args:
            dataset_name: Nome do dataset
            shape: Shape dos dados
            missing_pct: Percentual de dados faltantes
        """
        info = {
            'dataset': dataset_name,
            'rows': shape[0],
            'columns': shape[1] if len(shape) > 1 else 1
        }
        
        if missing_pct is not None:
            info['missing_pct'] = missing_pct
        
        self.set_context(**info)
        self.info(f"Dataset {dataset_name} carregado: {shape}")


def setup_logging(
    log_dir: str = "logs",
    log_level: str = "INFO",
    log_to_file: bool = True,
    log_to_console: bool = True,
    structured: bool = False,
    max_bytes: int = 10485760,  # 10MB
    backup_count: int = 5
) -> logging.Logger:
    """
    Configura o sistema de logging.
    
    Args:
        log_dir: Diretório para arquivos de log
        log_level: Nível de logging (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_to_file: Se deve logar para arquivo
        log_to_console: Se deve logar para console
        structured: Se deve usar formato JSON estruturado
        max_bytes: Tamanho máximo do arquivo de log antes de rotacionar
        backup_count: Número de backups a manter
        
    Returns:
        Logger configurado
    """
    if log_to_file:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))
    
    root_logger.handlers.clear()
    
    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, log_level.upper()))
        
        if structured:
            console_handler.setFormatter(StructuredFormatter())
        else:
            console_handler.setFormatter(ColoredConsoleFormatter())
        
        root_logger.addHandler(console_handler)
    
    # Adicionar handler de arquivo
    if log_to_file:
        # Log geral
        general_log = Path(log_dir) / f"ml_pipeline_{datetime.now().strftime('%Y%m%d')}.log"
        file_handler = logging.handlers.RotatingFileHandler(
            general_log,
            maxBytes=max_bytes,
            backupCount=backup_count
        )
        file_handler.setLevel(getattr(logging, log_level.upper()))
        
        if structured:
            file_handler.setFormatter(StructuredFormatter())
        else:
            file_handler.setFormatter(
                logging.Formatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                )
            )
        
        root_logger.addHandler(file_handler)
        
        # Log de erros separado
        error_log = Path(log_dir) / f"errors_{datetime.now().strftime('%Y%m%d')}.log"
        error_handler = logging.handlers.RotatingFileHandler(
            error_log,
            maxBytes=max_bytes,
            backupCount=backup_count
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(StructuredFormatter())
        root_logger.addHandler(error_handler)
    
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('sklearn').setLevel(logging.WARNING)
    
    return root_logger


def get_logger(name: str, with_ml_context: bool = False) -> Union[logging.Logger, MLContextLogger]:
    """
    Obtém um logger para um módulo específico.
    
    Args:
        name: Nome do módulo (geralmente __name__)
        with_ml_context: Se deve retornar MLContextLogger
        
    Returns:
        Logger configurado
    """
    logger = logging.getLogger(name)
    
    if with_ml_context:
        return MLContextLogger(logger)
    
    return logger


def log_execution(func):
    """
    Decorator para logar execução de funções.
    
    Loga início, fim, duração e erros de execução.
    
    Example:
        @log_execution
        def process_data(df):
            return df.dropna()
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        logger = get_logger(func.__module__)
        func_name = func.__name__
        
        # Log início
        logger.info(f"Iniciando execução: {func_name}")
        start_time = time.time()
        
        try:
            result = func(*args, **kwargs)
            
            # Log sucesso
            duration = time.time() - start_time
            logger.info(
                f"Execução concluída: {func_name}",
                extra={'duration': duration}
            )
            
            return result
            
        except Exception as e:
            # Log erro
            duration = time.time() - start_time
            logger.error(
                f"Erro na execução: {func_name}",
                exc_info=True,
                extra={'duration': duration}
            )
            raise
    
    return wrapper


def log_ml_pipeline(phase: str):
    """
    Decorator para logar fases do pipeline ML.
    
    Args:
        phase: Nome da fase (e.g., 'preprocessing', 'training', 'evaluation')
        
    Example:
        @log_ml_pipeline('training')
        def train_model(X, y):
            return model.fit(X, y)
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            logger = get_logger(func.__module__, with_ml_context=True)
            
            logger.set_ml_context(
                pipeline_phase=phase,
                function=func.__name__
            )
            
            with logger.timer(f"Pipeline ML - {phase}"):
                try:
                    result = func(*args, **kwargs)
                    logger.info(f"Fase {phase} concluída com sucesso")
                    return result
                except Exception as e:
                    logger.error(f"Erro na fase {phase}: {str(e)}", exc_info=True)
                    raise
                finally:
                    logger.clear_context()
        
        return wrapper
    return decorator


# Configuração global padrão
def init_default_logging():
    """Inicializa logging com configuração padrão."""
    setup_logging(
        log_dir="logs",
        log_level="INFO",
        log_to_file=True,
        log_to_console=True,
        structured=False
    )


# Exemplo de uso
if __name__ == "__main__":
    setup_logging(
        log_dir="test_logs",
        log_level="DEBUG",
        structured=True
    )
    
    logger = get_logger(__name__, with_ml_context=True)
    
    logger.info("Sistema de logging inicializado")
    
    logger.set_context(
        experiment_id="exp_001",
        dataset="worldbank",
        architecture="data_lake"
    )
    
    logger.info("Iniciando processamento de dados")
    
    # Log de métricas
    logger.log_model_metrics(
        "RandomForest",
        {"r2": 0.85, "rmse": 0.12, "mae": 0.08},
        phase="validation"
    )
    
    # Log com timer
    with logger.timer("feature_engineering"):
        time.sleep(1)  # Simular processamento
    
    # Log de informações de dados
    logger.log_data_info("train_data", shape=(10000, 50), missing_pct=5.2)
    
    # Exemplo com decorator
    @log_execution
    def example_function(x: int) -> int:
        return x * 2
    
    result = example_function(5)
    print(f"Resultado: {result}")
    
    logger.info("Teste de logging concluído")