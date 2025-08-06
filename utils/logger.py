import logging
import sys
from datetime import datetime
from typing import Optional
import os


class ColoredFormatter(logging.Formatter):
    """Colored log formatter"""

    COLORS = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[35m",  # Purple
        "RESET": "\033[0m",  # Reset
    }

    def format(self, record):
        # Add color
        if record.levelname in self.COLORS:
            record.levelname = f"{self.COLORS[record.levelname]}{record.levelname}{self.COLORS['RESET']}"

        # Add timestamp
        record.asctime = datetime.fromtimestamp(record.created).strftime(
            "%Y-%m-%d %H:%M:%S"
        )

        return super().format(record)


class TaskLogger:
    """Task-specific logger"""

    def __init__(self, task_id: str, log_dir: str = "./logs"):
        self.task_id = task_id
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

        # Create task-specific log file
        self.log_file = os.path.join(log_dir, f"task_{task_id}.log")

        # Create task-specific logger
        self.logger = logging.getLogger(f"task_{task_id}")
        self.logger.setLevel(logging.DEBUG)

        # File handler
        file_handler = logging.FileHandler(self.log_file, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        file_handler.setFormatter(file_formatter)

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_formatter = ColoredFormatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        console_handler.setFormatter(console_formatter)

        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

    def info(self, message: str):
        self.logger.info(f"[{self.task_id}] {message}")

    def error(self, message: str):
        self.logger.error(f"[{self.task_id}] {message}")

    def warning(self, message: str):
        self.logger.warning(f"[{self.task_id}] {message}")

    def debug(self, message: str):
        self.logger.debug(f"[{self.task_id}] {message}")

    def critical(self, message: str):
        self.logger.critical(f"[{self.task_id}] {message}")


def setup_logger(
    name: str = "rknn_converter", level: int = logging.INFO
) -> logging.Logger:
    """Setup global logger"""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid adding duplicate handlers
    if not logger.handlers:
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_formatter = ColoredFormatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        console_handler.setFormatter(console_formatter)

        # File handler
        os.makedirs("./logs", exist_ok=True)
        file_handler = logging.FileHandler("./logs/server.log", encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        file_handler.setFormatter(file_formatter)

        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

    return logger


# Global logger
logger = setup_logger()
