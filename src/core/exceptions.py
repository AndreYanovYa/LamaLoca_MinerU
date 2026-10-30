# src/core/exceptions.py

class LamaLocaError(Exception):
    """Базовое исключение приложения."""
    pass

class DocumentParseError(LamaLocaError):
    """Ошибка парсинга (повреждённый файл, неподдерживаемый формат)."""
    pass

class IndexSyncError(LamaLocaError):
    """Рассогласование: чанк есть в SQLite, но отсутствует в ChromaDB (или наоборот)."""
    pass

class ExternalToolNotFoundError(LamaLocaError):
    """Внешний инструмент (Tesseract, Calibre, DjVuLibre) не найден в PATH или по указанному пути."""
    pass

class OfflineModeViolationError(LamaLocaError):
    """Попытка сетевого запроса к внешнему ресурсу в офлайн-режиме."""
    pass