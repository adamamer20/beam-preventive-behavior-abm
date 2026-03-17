"""Centralized logging configuration using loguru."""

import os
import sys
from pathlib import Path
from typing import Any

from loguru import logger

# Default log configuration
DEFAULT_LOG_CONFIG = {
    "level": "INFO",
    "format": "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
    "rotation": "10 MB",
    "retention": "7 days",
    "compression": "zip",
    "colorize": True,
    "backtrace": True,
    "diagnose": True,
}

_LOGGING_STATE: dict[str, bool] = {"configured": False}


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _running_in_quarto() -> bool:
    """Detect Quarto execution contexts."""
    return any(
        os.getenv(var)
        for var in (
            "QUARTO_PROJECT_ROOT",
            "QUARTO_PROFILE",
            "QUARTO_RENDER",
            "QUARTO_PYTHON",
        )
    )


def setup_logging(
    log_level: str | None = None,
    log_file: str | None = None,
    log_dir: str | None = None,
    config_override: dict[str, Any] | None = None,
    enable_file_logging: bool = True,
    component: str = "general",
    run_id: str | None = None,
    *,
    force: bool = False,
    configure_third_party: bool = True,
) -> None:
    """
    Set up centralized logging configuration using loguru.

    Parameters
    ----------
    log_level : str, optional
        Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL). Defaults to INFO.
    log_file : str, optional
        Name of the log file. Defaults to run-based 'beam_run_{run_id}.log'.
    log_dir : str, optional
        Directory to store log files. Defaults to 'logs/{component}' in project root.
    config_override : dict, optional
        Dictionary to override default configuration settings.
    enable_file_logging : bool, optional
        Whether to enable file logging. Defaults to True.
    component : str, optional
        Component name for organizing logs (e.g., 'nn', 'llm_behaviour'). Defaults to 'general'.
    run_id : str, optional
        Run identifier for consistent logging across a single evaluation run.
        If not provided, defaults to current timestamp.
    """
    if _LOGGING_STATE["configured"] and not force:
        return

    # Remove default handler
    logger.remove()

    # Merge configurations
    config = DEFAULT_LOG_CONFIG.copy()
    if config_override:
        config.update(config_override)

    # Get log level from environment variable or parameter
    log_level = log_level or os.getenv("LOG_LEVEL", config["level"])
    log_format = os.getenv("LOG_FORMAT", config["format"])
    colorize = _env_bool("LOG_COLORIZE", bool(config["colorize"]))
    enable_console_logging = _env_bool(
        "LOG_TO_CONSOLE",
        default=not _running_in_quarto(),
    )

    if enable_console_logging:
        # Set up console handler
        logger.add(
            sys.stderr,
            format=log_format,
            level=log_level,
            colorize=colorize,
            backtrace=config["backtrace"],
            diagnose=config["diagnose"],
            enqueue=True,
        )

    # Set up file handler (enabled by default)
    # Allow explicit environment override via LOG_TO_FILE (true/false)
    if os.getenv("LOG_TO_FILE") is not None:
        enable_file_logging = _env_bool("LOG_TO_FILE", enable_file_logging)
    else:
        # Keep explicit function argument (default True) unless env overrides it
        enable_file_logging = bool(enable_file_logging)

    if enable_file_logging:
        # Generate run identifier for log file
        if run_id is None:
            import datetime

            run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        log_dir = log_dir or os.getenv("LOG_DIR", f"logs/{component}")
        log_file = log_file or os.getenv("LOG_FILE", f"beam_run_{run_id}.log")

        # Create log directory if it doesn't exist
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)

        # Full path to log file
        full_log_path = log_path / log_file

        logger.add(
            str(full_log_path),
            format=log_format,
            level=log_level,
            rotation=config["rotation"],
            retention=config["retention"],
            compression=config["compression"],
            backtrace=config["backtrace"],
            diagnose=config["diagnose"],
            enqueue=True,
        )

        logger.info(f"Logging to file: {full_log_path}")

    if configure_third_party:
        configure_third_party_loggers()

    _LOGGING_STATE["configured"] = True


def get_logger(name: str) -> Any:
    """
    Get a logger instance with the specified name.

    Parameters
    ----------
    name : str
        Name for the logger, typically __name__.

    Returns
    -------
    logger
        Configured loguru logger instance.
    """
    if not _LOGGING_STATE["configured"]:
        setup_logging()
    return logger.bind(name=name)


def configure_third_party_loggers(level: str = "WARNING") -> None:
    """
    Configure third-party library loggers to reduce noise.

    Parameters
    ----------
    level : str, optional
        Log level for third-party loggers. Defaults to WARNING.
    """
    import logging

    # List of noisy third-party loggers
    noisy_loggers = [
        "urllib3",
        "requests",
        "matplotlib",
        "PIL",
        "transformers",
        "torch",
        "tensorboard",
        "vllm",
        "asyncio",
    ]

    for logger_name in noisy_loggers:
        logging.getLogger(logger_name).setLevel(getattr(logging, level))
