"""Central logging infrastructure for ECG Arrhythmia Classification."""

import logging
import os
import sys
from typing import Optional


def get_logger(name: str = "ecg_arrhythmia", log_level: Optional[str] = None) -> logging.Logger:
    """Return a configured logger with standard formatting.

    Args:
        name: Name of the logger, typically __name__ or package identifier.
        log_level: Optional log level string (DEBUG, INFO, WARNING, ERROR, CRITICAL).
            Defaults to the LOG_LEVEL environment variable or INFO.

    Returns:
        logging.Logger instance configured with console stream handler.
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        level_name = log_level or os.getenv("LOG_LEVEL", "INFO").upper()
        level = getattr(logging, level_name, logging.INFO)
        logger.setLevel(level)

        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)

        formatter = logging.Formatter(
            fmt="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False

    return logger
