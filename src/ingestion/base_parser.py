# src/ingestion/base_parser.py
from abc import ABC, abstractmethod
from pathlib import Path
from .models import ParsedDocument

class BaseParser(ABC):
    """Абстрактный базовый класс для всех парсеров документов."""
    
    @abstractmethod
    def parse(self, file_path: Path, doc_id: str, progress_callback=None) -> ParsedDocument:
        """
        Извлекает текст и метаданные из файла.
        
        Args:
            file_path: Путь к исходному файлу.
            doc_id: SHA-256 хеш файла (уже вычислен роутером).
        
        Returns:
            ParsedDocument с блоками и метаданными.
        
        Raises:
            DocumentParseError: Если файл повреждён или не может быть прочитан.
        """
        pass
    
    @abstractmethod
    def supported_extensions(self) -> list[str]:
        """Список расширений, которые поддерживает этот парсер (с точкой)."""
        pass