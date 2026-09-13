"""Logging configuration helpers."""

import logging


def configure_logging(level: int = logging.INFO) -> None:
    """Configure a simple project-wide logging format."""
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(message)s")
