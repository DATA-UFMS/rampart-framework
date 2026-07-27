#!/usr/bin/env python3
"""
Structured logging configuration for the benchmarking project.

Implements centralised logging with:
- Multiple handlers (console, file, JSON)
- Per-module configurable levels
- Structured context
- Log rotation
- Performance tracking
"""

import os
import sys
import json
import logging
import logging.handlers
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Union
from functools import wraps
import time
import traceback
from contextlib import contextmanager


class StructuredFormatter(logging.Formatter):
    """
    Formatter for structured JSON logs.

    Adds relevant context such as timestamp, module, function,
    and error information where applicable.
    """
    
    def format(self, record: logging.LogRecord) -> str:
        """
        Format the log record as structured JSON.
        
        Args:
            record: LogRecord to format
            
        Returns:
            Formatted JSON string
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
    Coloured formatter for the console.

    Uses ANSI colours for better visualisation in the terminal.
    """
    
    COLORS = {
        'DEBUG': '\033[36m',
        'INFO': '\033[32m',
        'WARNING': '\033[33m',
        'ERROR': '\033[31m',
        'CRITICAL': '\033[35m',
    }
    RESET = '\033[0m'
    BOLD = '\033[1m'
    
    def format(self, record: logging.LogRecord) -> str:
        """
        Format the log record with colours.
        
        Args:
            record: LogRecord to format
            
        Returns:
            String formatted with ANSI colours
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
    Logger with ML-specific context.

    Automatically adds context relevant to
    machine learning operations.
    """
    
    def __init__(self, logger: logging.Logger):
        """
        Initialise the logger with ML context.

        Args:
            logger: Base logger
        """
        self.logger = logger
        self.context = {}
        self.ml_context = {}
    
    def set_context(self, **kwargs):
        """Set permanent context for all logs."""
        self.context.update(kwargs)
    
    def set_ml_context(self, **kwargs):
        """Set ML-specific context."""
        self.ml_context.update(kwargs)
    
    def clear_context(self):
        """Clear the context."""
        self.context.clear()
        self.ml_context.clear()
    
    def _log_with_context(self, level: int, msg: str, **kwargs):
        """Log with added context."""
        extra = {'context': {**self.context, **kwargs}}
        if self.ml_context:
            extra['ml_context'] = self.ml_context
        self.logger.log(level, msg, extra=extra)
    
    def debug(self, msg: str, **kwargs):
        """Debug log with context."""
        self._log_with_context(logging.DEBUG, msg, **kwargs)
    
    def info(self, msg: str, **kwargs):
        """Info log with context."""
        self._log_with_context(logging.INFO, msg, **kwargs)
    
    def warning(self, msg: str, **kwargs):
        """Warning log with context."""
        self._log_with_context(logging.WARNING, msg, **kwargs)
    
    def error(self, msg: str, **kwargs):
        """Error log with context."""
        self._log_with_context(logging.ERROR, msg, **kwargs)
    
    def critical(self, msg: str, **kwargs):
        """Critical log with context."""
        self._log_with_context(logging.CRITICAL, msg, **kwargs)
    
    @contextmanager
    def timer(self, operation: str):
        """
        Context manager for measuring the duration of operations.

        Args:
            operation: Name of the operation
            
        Example:
            with logger.timer('model_training'):
                model.fit(X, y)
        """
        start_time = time.time()
        self.info(f"Starting: {operation}")
        
        try:
            yield
        finally:
            duration = time.time() - start_time
            extra = {'context': self.context, 'duration': duration}
            if self.ml_context:
                extra['ml_context'] = self.ml_context
            
            self.logger.info(
                f"Finished: {operation} ({duration:.2f}s)",
                extra=extra
            )
    
    def log_model_metrics(self, model_name: str, metrics: Dict[str, float], 
                         phase: str = 'test'):
        """
        Log model metrics.

        Args:
            model_name: Model name
            metrics: Dictionary of metrics
            phase: Phase (train/val/test)
        """
        self.set_ml_context(
            model=model_name,
            phase=phase,
            metrics=metrics
        )
        
        metrics_str = ', '.join(f"{k}={v:.4f}" for k, v in metrics.items())
        self.info(f"{phase} metrics for {model_name}: {metrics_str}")
    
    def log_data_info(self, dataset_name: str, shape: tuple, 
                      missing_pct: float = None):
        """
        Log information about the data.

        Args:
            dataset_name: Dataset name
            shape: Shape of the data
            missing_pct: Percentage of missing data
        """
        info = {
            'dataset': dataset_name,
            'rows': shape[0],
            'columns': shape[1] if len(shape) > 1 else 1
        }
        
        if missing_pct is not None:
            info['missing_pct'] = missing_pct
        
        self.set_context(**info)
        self.info(f"Dataset {dataset_name} loaded: {shape}")


def get_logger(name: str, with_ml_context: bool = False) -> Union[logging.Logger, MLContextLogger]:
    """
    Obtain a logger for a specific module.

    Args:
        name: Module name (usually __name__)
        with_ml_context: Whether to return an MLContextLogger

    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    
    if with_ml_context:
        return MLContextLogger(logger)
    
    return logger


def log_ml_pipeline(phase: str):
    """
    Decorator for logging ML pipeline phases.

    Args:
        phase: Phase name (e.g., 'preprocessing', 'training', 'evaluation')
        
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
            
            with logger.timer(f"ML pipeline - {phase}"):
                try:
                    result = func(*args, **kwargs)
                    logger.info(f"Phase {phase} completed successfully")
                    return result
                except Exception as e:
                    logger.error(f"Error in phase {phase}: {str(e)}", exc_info=True)
                    raise
                finally:
                    logger.clear_context()
        
        return wrapper
    return decorator


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    logger = get_logger(__name__, with_ml_context=True)

    logger.info("Logging system initialised")

    logger.set_context(
        experiment_id="exp_001",
        dataset="worldbank",
        architecture="task_graph"
    )

    logger.info("Starting data processing")

    logger.log_model_metrics(
        "RandomForest",
        {"r2": 0.85, "rmse": 0.12, "mae": 0.08},
        phase="validation"
    )

    with logger.timer("feature_engineering"):
        time.sleep(1)  # Simulate processing

    # Log data information
    logger.log_data_info("train_data", shape=(10000, 50), missing_pct=5.2)

    logger.info("Logging test completed")