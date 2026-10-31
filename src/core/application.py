# src/core/application.py
"""
Composition root lama-loca.

Единая точка создания разделяемых компонентов (SQLiteStore, ChromaStore,
DocumentRouter, TextChunker, OllamaEmbedder, RAGPipeline).

Ранее каждый модуль UI (LibraryView, ImportWorker, ChatView/RAGPipeline)
создавал собственные экземпляры SQLiteStore/ChromaStore с отдельными
соединениями к одним и тем же файлам — это приводило к рассогласованию
кэшей, блокировкам в WAL и «призрачным» данным (например, список библиотеки
не обновлялся после импорта без перезапуска). Теперь все представления
получают одни и те же инстансы через Application.
"""
from __future__ import annotations

from pathlib import Path

from loguru import logger

from .paths import Paths
from .settings import Settings
from ..embeddings.ollama_embedder import OllamaEmbedder
from ..ingestion.document_router import DocumentRouter
from ..llm.rag_pipeline import RAGPipeline
from ..processing.chunker import TextChunker
from ..storage.chroma_store import ChromaStore
from ..storage.sqlite_store import SQLiteStore


class Application:
    """Хранилище singleton-компонентов приложения."""

    def __init__(self, settings: Settings, paths: Paths):
        self.settings = settings
        self.paths = paths

        # Гарантируем наличие всех директорий хранилища.
        settings.ensure_dirs()

        self.sqlite_store = SQLiteStore(settings.sqlite_db_path)
        self.chroma_store = ChromaStore(settings.chroma_db_path)
        self.router = DocumentRouter(settings)
        self.chunker = TextChunker(settings.chunking)

        # Эмбеддер создаём один раз: Ollama может быть недоступна на старте,
        # тогда поднимаем ошибку сразу и явно (UI покажет её пользователю).
        self.embedder = OllamaEmbedder(settings.ollama)

        self.rag_pipeline = RAGPipeline(
            settings=settings,
            sqlite_store=self.sqlite_store,
            chroma_store=self.chroma_store,
            embedder=self.embedder,
        )

        logger.info("Application: все компоненты инициализированы")

    # ------------------------------------------------------------------
    # Общие пути библиотеки (единое соглашение для всего приложения)
    # ------------------------------------------------------------------

    @property
    def library_pdf_dir(self) -> Path:
        """Каталог для сохранённых исходников PDF (data/library/pdf)."""
        return self.paths.library_dir / "pdf"

    @property
    def library_epub_dir(self) -> Path:
        """Каталог для сохранённых исходников EPUB (data/library/epub)."""
        return self.paths.library_dir / "epub"

    def saved_file_path(self, doc_id: str, source_path: Path) -> Path:
        """Возвращает путь, под которым файл хранится в библиотеке."""
        ext = source_path.suffix.lower()
        base = f"{source_path.stem}_{doc_id[:8]}{ext}"
        if ext == ".epub":
            return self.library_epub_dir / base
        return self.library_pdf_dir / base

    def close(self):
        """Закрывает соединения при завершении приложения."""
        try:
            self.sqlite_store.close()
        except Exception as e:  # pragma: no cover
            logger.warning(f"Ошибка закрытия SQLite: {e}")
