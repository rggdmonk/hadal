"""This module contains the `fuction default custom logger` that can be used to create a logger with a specified name, level, and format."""
from __future__ import annotations

import logging


def default_custom_logger(
    name: str,
    level: int | None = logging.DEBUG,
    log_format: str | None = None,
) -> logging.Logger:
    """Create a logger with a specified name, level, and format.

    Args:
        name (str): The name of the logger.
        level (int | None, optional): Logging level.
        log_format (str | None, optional): The format of the log messages. If `None`, the default format is `%(asctime)s | %(name)s | %(module)s | %(levelname)s | %(message)s`.

    Returns:
        logger (logging.Logger): A logger object with the specified name, logging level, and format.
    """
    if log_format is None:
        log_format = "%(asctime)s | %(name)s | %(module)s | %(levelname)s | %(message)s"

    datefmt = "%Y-%m-%d %H:%M:%S"

    logger = logging.basicConfig(level=level, format=log_format, datefmt=datefmt, encoding="utf-8")
    logger = logging.getLogger(name=name)

    return logger
