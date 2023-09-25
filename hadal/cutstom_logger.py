"""Creates a module containing a function for creating a custom logger with a specified name, log level, and format."""

import logging


def default_custom_logger(
    name: str,
    level: int | None = logging.DEBUG,
    log_format: str | None = None,
) -> logging.Logger:
    """Returns a logger object with the specified name, logging level, and format.

    Args:
        name (str): The name of the logger.
        level (int | None, optional): The logging level. Defaults to `logging.DEBUG`.
        log_format (str | None, optional): The format of the log messages. If not specified, the default format is `%(asctime)s | %(name)s | %(module)s | %(levelname)s | %(message)s`.

    Returns:
        logging.Logger: A logger object with the specified name, logging level, and format.
    """
    if log_format is None:
        log_format = "%(asctime)s | %(name)s | %(module)s | %(levelname)s | %(message)s"

    datefmt = "%Y-%m-%d %H:%M:%S"

    logger = logging.basicConfig(level=level, format=log_format, datefmt=datefmt, encoding="utf-8")
    logger = logging.getLogger(name=name)

    return logger
