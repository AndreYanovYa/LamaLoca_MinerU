# src/utils/logger.py
import sys
from loguru import logger
from pathlib import Path

def setup_logger(logs_dir: Path, level: str = "INFO") -> None:
    logger.remove()
    
    # Консольный вывод с цветами
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level=level
    )
    
    # Файловый вывод с ротацией (10 МБ, хранение 5 файлов)
    log_file = logs_dir / "app.log"
    logger.add(
        log_file,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{line} - {message}",
        level=level,
        rotation="10 MB",
        retention=5,
        encoding="utf-8"
    )
    logger.info("Логирование инициализировано.")