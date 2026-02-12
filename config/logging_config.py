"""Logging configuration for AI Trading Agent"""

import logging
import logging.handlers
import os
from pathlib import Path


def setup_logging(log_level: str = "INFO", log_dir: str = "logs") -> None:
    """
    Configure logging for the application.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir: Directory to store log files
    """
    # Create logs directory if it doesn't exist
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    # Convert string log level to logging constant
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    simple_formatter = logging.Formatter(
        fmt='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    
    # Remove existing handlers
    root_logger.handlers.clear()
    
    # Console handler (simple format)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(simple_formatter)
    root_logger.addHandler(console_handler)
    
    # File handler for all logs (detailed format, rotating)
    all_logs_file = log_path / "trading_agent.log"
    file_handler = logging.handlers.RotatingFileHandler(
        all_logs_file,
        maxBytes=10 * 1024 * 1024,  # 10 MB
        backupCount=5
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(detailed_formatter)
    root_logger.addHandler(file_handler)
    
    # File handler for errors only (detailed format, rotating)
    error_logs_file = log_path / "errors.log"
    error_handler = logging.handlers.RotatingFileHandler(
        error_logs_file,
        maxBytes=10 * 1024 * 1024,  # 10 MB
        backupCount=5
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(detailed_formatter)
    root_logger.addHandler(error_handler)
    
    # File handler for trades (separate log for audit trail)
    trades_logs_file = log_path / "trades.log"
    trades_handler = logging.handlers.RotatingFileHandler(
        trades_logs_file,
        maxBytes=10 * 1024 * 1024,  # 10 MB
        backupCount=10
    )
    trades_handler.setLevel(logging.INFO)
    trades_handler.setFormatter(detailed_formatter)
    
    # Create a separate logger for trades
    trades_logger = logging.getLogger('trades')
    trades_logger.addHandler(trades_handler)
    trades_logger.setLevel(logging.INFO)
    trades_logger.propagate = False  # Don't propagate to root logger
    
    # Suppress noisy third-party loggers
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('requests').setLevel(logging.WARNING)
    logging.getLogger('alpaca').setLevel(logging.INFO)
    logging.getLogger('tensorflow').setLevel(logging.WARNING)
    logging.getLogger('transformers').setLevel(logging.WARNING)
    
    logging.info(f"Logging configured with level: {log_level}")
    logging.info(f"Log files location: {log_path.absolute()}")


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a specific module.
    
    Args:
        name: Name of the logger (typically __name__)
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)


def get_trades_logger() -> logging.Logger:
    """
    Get the dedicated trades logger for audit trail.
    
    Returns:
        Trades logger instance
    """
    return logging.getLogger('trades')
