import logging
import sys
from typing import Optional


def setup_logging(
    level: str = "INFO",
    log_format: Optional[str] = None,
    log_file: Optional[str] = None,
) -> None:
    if log_format is None:
        log_format = "%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d - %(message)s"

    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]

    if log_file:
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format=log_format,
        handlers=handlers,
        force=True,
    )


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
