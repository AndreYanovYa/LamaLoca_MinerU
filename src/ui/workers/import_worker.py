# src/ui/workers/import_worker.py
"""
Фоновый воркер для импорта документов в библиотеку.

АУДИТ (исправлено):
1. Зависимости (router, chunker, sqlite_store, chroma_store, embedder) теперь
   инжектируются из Application — ранее воркер создавал СОБСТВЕННЫЕ соединения
   к SQLite/ChromaDB, и UI-список библиотеки не видел импортированные книги
   без перезапуска приложения.
2. Сохранение исходного файла: каталог data/library/pdf (для PDF) и
   data/library/epub (для EPUB) задаётся единообразно через Application.
3. Атомарность: запись блоков + FTS5 выполняется одним методом
   SQLiteStore.save_blocks() в одной транзакции; при ошибке векторизации
   статус документа помечается ERROR, а частично записанные данные очищаются
   (SQLite + FTS + Chroma), чтобы не оставлять «висячие» блоки без эмбеддингов.
4. Убраны ручные SQL-дубликаты INSERT в обход save_blocks — block_type
   сохранялся в верхнем регистре ("TEXT"), из-за чего запросы
   block_type='heading'/'text' ничего не находили.
"""
import shutil
import threading
from pathlib import Path
from typing import Callable, Optional

from loguru import logger

from src.core.settings import Settings
from src.core.paths import Paths
from src.ingestion.document_router import DocumentRouter
from src.processing.chunker import TextChunker
from src.storage.sqlite_store import SQLiteStore
from src.storage.chroma_store import ChromaStore
from src.embeddings.ollama_embedder import OllamaEmbedder


class _ProgressBridge:
    """
    Мост между внутренним прогрессом парсеров/OCR и UI-прогрессом импорта.

    Парсеры вызывают update(done, total, message) из рабочих потоков; мост
    конвертирует это в долю 0..1 и частоту вызовов (не чаще одного обновления
    на 2% изменения), чтобы не заваливать page.update().
    """

    def __init__(self, emit):
        self._emit = emit
        self._lock = threading.Lock()
        self._last_pct = -1

    def update(self, done: int, total: int, message: str = ""):
        if total <= 0:
            return
        frac = max(0.0, min(1.0, done / total))
        pct = int(frac * 100)
        with self._lock:
            # обновляем не чаще чем на каждые 2% (или на финальной точке)
            if pct - self._last_pct < 2 and pct != 100:
                return
            self._last_pct = pct
        self._emit(frac, message or f"Обработка... {pct}%")


class ImportWorker:
    """Класс для асинхронного импорта и индексации документов."""

    def __init__(
        self,
        settings: Settings,
        paths: Paths = None,
        router: DocumentRouter = None,
        chunker: TextChunker = None,
        sqlite_store: SQLiteStore = None,
        chroma_store: ChromaStore = None,
        embedder: OllamaEmbedder = None,
    ):
        """
        Инициализация воркера.

        Все компоненты можно передать из Application (общие экземпляры).
        Если что-то не передано — создаётся локально (например, в тестах).
        """
        self.settings = settings
        # АУДИТ: paths больше не обязателен — каталоги выводятся из
        # settings.storage.root (раньше UI мог случайно создать второй
        # ImportWorker без paths и упасть с AttributeError на None).
        self.paths = paths

        self.router = router or DocumentRouter(settings)
        self.chunker = chunker or TextChunker(settings.chunking)
        self.sqlite_store = sqlite_store or SQLiteStore(settings.sqlite_db_path)
        self.chroma_store = chroma_store or ChromaStore(settings.chroma_db_path)
        self.embedder = embedder or OllamaEmbedder(settings.ollama)

    async def import_file(self, file_path: Path, progress_callback: Optional[Callable] = None):
        """
        Асинхронно импортирует файл, обрабатывает его и сохраняет в хранилища.

        :param file_path: Путь к файлу (PDF или EPUB)
        :param progress_callback: Функция (message, percent) для обновления UI
        :return: doc_id импортированного документа
        """
        logger.info(f"Импорт файла: {file_path.name}")

        def report(message: str, percent: int):
            if progress_callback:
                try:
                    progress_callback(message, percent)
                except Exception:
                    pass  # UI-прогресс не должен ронять импорт

        # Этап 1: Парсинг (внутри — SHA-256 дедупликация по doc_id).
        # Парсер/OCR отчитываются о внутреннем прогрессе через
        # _ProgressBridge — он транслируется в шкалу 5–45%.
        report("Чтение и парсинг документа...", 5)
        bridge = _ProgressBridge(
            emit=lambda frac, msg: report(msg, 5 + int(frac * 40))
        )
        doc = self.router.parse(file_path, progress_callback=bridge)

        try:
            # Этап 2: Сохранение исходного файла в библиотеку
            report("Сохранение исходного файла...", 48)
            final_dest_path = self._save_source_file(file_path, doc.doc_id)
            # В базе храним путь к копии в библиотеке — он стабилен,
            # даже если пользователь переместит/удалит исходник.
            doc.source_path = str(final_dest_path.resolve())

            # Этап 3: Чанкинг
            report("Разбиение на чанки...", 50)
            raw_chunks = self.chunker.chunk_blocks(doc.blocks, doc_id=doc.doc_id)
            n_chunks = len(raw_chunks)
            report(f"Готово: {len(doc.blocks)} блоков → {n_chunks} чанков", 52)

            # Этап 4: Метаданные документа — фиксируем В БАЗЕ ДО долгой
            # векторизации. Фикс «карточка книги утеряна после закрытия»:
            # раньше очистка+запись happening только здесь, и если процесс
            # завершался во время эмбеддинга (закрыли окно / Ctrl+C),
            # ChromaDB уже содержала векторы, а строка documents() была
            # удалена/не записана → библиотека пустая при живых чанках.
            # Теперь карточка существует на протяжении всего импорта
            # (статус INDEXING), а UI показывает её как «индексация».
            doc.status = "INDEXING"
            self.sqlite_store.save_document(doc)

            # Этап 4.5: Очистка старых блоков/векторов этого же файла
            # (реимпорт). Карточку НЕ удаляем — только содержимое.
            report("Очистка старых данных...", 54)
            try:
                self.sqlite_store.delete_blocks_of(doc.doc_id)
            except AttributeError:
                # Совместимость со старыми сборками без нового метода.
                self._cleanup_doc(doc.doc_id)
                self.sqlite_store.save_document(doc)
            self.chroma_store.delete_by_doc_id(doc.doc_id)

            # Этап 5: (метаобновление выполнено на этапе 4)
            report("Сохранение в базу данных...", 56)

            # Этап 6: Блоки + FTS5 одной транзакцией (SQLiteStore.save_blocks)
            blocks_data = []
            for chunk in raw_chunks:
                meta = chunk["metadata"]
                blocks_data.append({
                    "block_id": chunk["block_id"],
                    "doc_id": doc.doc_id,
                    "block_type": str(meta.get("block_type", "text")).lower(),
                    "text": chunk["text"],
                    "page_number": meta.get("page_number"),
                    "chapter": meta.get("chapter") or "",
                    "section": meta.get("section") or "",
                    "coordinates": meta.get("coordinates"),
                    "extraction_method": meta.get("extraction_method", "unknown"),
                })
            self.sqlite_store.save_blocks(blocks_data)

            # Этап 7: Векторизация (60–90%) — прогресс по батчам эмбеддера
            report(f"Векторизация {n_chunks} чанков... (0/{n_chunks})", 60)
            texts = [c["text"] for c in raw_chunks]

            def on_embed_batch(done, total):
                pct = 60 + int(done / max(total, 1) * 30)
                report(f"Векторизация чанков... ({done}/{total})", min(pct, 90))

            embeddings = self.embedder.embed_texts(texts, progress_callback=on_embed_batch)

            # Этап 8: Сохранение векторов в ChromaDB (90–99%)
            report(f"Сохранение {n_chunks} векторов в ChromaDB...", 90)

            def on_chroma_batch(done, total):
                pct = 90 + int(done / max(total, 1) * 9)
                report(f"Индексация в ChromaDB... ({done}/{total})", min(pct, 99))

            metadatas = [dict(c["metadata"]) for c in raw_chunks]
            for m in metadatas:
                m.setdefault("doc_id", doc.doc_id)
                m.setdefault("title", doc.title)
                m.setdefault("file_format", doc.file_format)
            self.chroma_store.add_chunks(
                ids=[c["block_id"] for c in raw_chunks],  # те же ID, что в SQLite
                texts=texts,
                embeddings=embeddings,
                metadatas=metadatas,
                progress_callback=on_chroma_batch,
            )

            # Этап 8.5: GRAPH-RAG (Этап 1) — строим граф онтологических связей
            # математических сущностей (DEFINITION/THEOREM/PROOF/...). Ошибки
            # здесь НЕ должны ронять импорт: book уже проиндексирована, граф —
            # обогащение; при падении фиксируем warning и продолжаем.
            try:
                report("Построение графа знаний...", 99)
                from src.processing.entity_extractor import build_relations
                relations, _types = build_relations(raw_chunks)
                self.sqlite_store.save_entity_relations(doc.doc_id, relations)
            except Exception as ge:
                logger.warning(
                    f"Graph-RAG: не удалось построить связи для "
                    f"{file_path.name}: {ge} (импорт продолжен)"
                )

            # Этап 9: Финализация
            self.sqlite_store.conn.execute(
                "UPDATE documents SET status = ? WHERE doc_id = ?",
                ("READY", doc.doc_id),
            )
            self.sqlite_store.conn.commit()

            report("Готово!", 100)
            logger.info(f"✅ Файл успешно импортирован: {file_path.name} ({doc.doc_id[:8]}...)")
            return doc.doc_id

        except Exception as e:
            logger.error(f"❌ Ошибка импорта {file_path.name}: {e}")
            # Откат: удаляем частичные данные, чтобы документ не остался
            # «полуготовым» (блоки без эмбеддингов / рассинхрон SQLite-Chroma).
            try:
                self._cleanup_doc(doc.doc_id)
            except Exception as cleanup_error:
                logger.error(f"Ошибка отката импорта {file_path.name}: {cleanup_error}")
            raise

    # ------------------------------------------------------------------
    # Вспомогательные методы
    # ------------------------------------------------------------------

    def _save_source_file(self, file_path: Path, doc_id: str) -> Path:
        """Копирует исходник в data/library/{pdf,epub}. Возвращает путь копии."""
        ext = file_path.suffix.lower()
        subdir = "epub" if ext == ".epub" else "pdf"
        dest_dir = Path(self.settings.storage.root) / "library" / subdir
        dest_dir.mkdir(parents=True, exist_ok=True)
        final_dest_path = dest_dir / f"{file_path.stem}_{doc_id[:8]}{ext}"
        # Разные файлы с одинаковым SHA-256 — это один и тот же файл,
        # повторное копирование пропускаем (дедупликация).
        if not final_dest_path.exists():
            shutil.copy2(file_path, final_dest_path)
        return final_dest_path

    def _cleanup_doc(self, doc_id: str):
        """Полностью удаляет данные документа из обоих хранилищ."""
        self.sqlite_store.delete_document(doc_id)
        self.chroma_store.delete_by_doc_id(doc_id)
