# src/storage/sqlite_store.py
"""
Модуль для работы с SQLite базой данных.
Управляет метаданными документов, текстовыми блоками и полнотекстовым поиском (FTS5).
"""
import re
import sqlite3
import json
from pathlib import Path
from typing import List, Dict, Any, Optional

from loguru import logger


class SQLiteStore:
    """Класс для управления SQLite хранилищем метаданных."""
    
    def __init__(self, db_path: str):
        """
        Инициализирует подключение к базе данных.
        
        Args:
            db_path: Путь к файлу базы данных SQLite.
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Подключаемся к БД
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)

        # АУДИТ (баг «после перезапуска источники не отображаются»): при
        # рассинхроне WAL-файла и основной БД (например, процесс был убит во
        # время импорта) sqlite3 открывает повреждённую копию и все SELECT
        # падают с "database disk image is malformed" — UI молча показывал
        # пустую библиотеку. Ранняя проверка целостности даёт понятную ошибку
        # вместо тихого «пусто».
        try:
            row = self.conn.execute("PRAGMA integrity_check").fetchone()
        except sqlite3.DatabaseError as e:
            self.conn.close()
            raise RuntimeError(
                f"База данных повреждена ({self.db_path}): {e}. "
                "Восстановите файл из резервной копии или удалите его и "
                "реимпортируйте книги."
            ) from e
        if not row or row[0] != "ok":
            self.conn.close()
            raise RuntimeError(
                f"База данных повреждена ({self.db_path}): {row}. "
                "Восстановите файл из резервной копии или удалите его и "
                "реимпортируйте книги."
            )
        
        # Включаем поддержку внешних ключей и WAL режим (для лучшей производительности при чтении/записи)
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.execute("PRAGMA journal_mode = WAL")
        
        # Создаем таблицы при инициализации
        self._create_tables()
        logger.info(f"SQLite хранилище инициализировано: {self.db_path}")
    
    def _create_tables(self):
        """Создает необходимые таблицы и индексы, если они не существуют."""
        cursor = self.conn.cursor()
        
        # Таблица документов
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                doc_id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                author TEXT,
                file_format TEXT NOT NULL,
                source_path TEXT NOT NULL,
                language TEXT DEFAULT 'ru',
                total_pages INTEGER DEFAULT 0,
                status TEXT DEFAULT 'READY',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Таблица текстовых блоков (чанков)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS blocks (
                block_id TEXT PRIMARY KEY,
                doc_id TEXT NOT NULL,
                block_type TEXT NOT NULL,
                text TEXT NOT NULL,
                page_number INTEGER,
                chapter TEXT,
                section TEXT,
                coordinates TEXT,
                extraction_method TEXT,
                FOREIGN KEY (doc_id) REFERENCES documents(doc_id) ON DELETE CASCADE
            )
        """)
        
        # Индекс для быстрого поиска блоков по документу
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_blocks_doc_id ON blocks(doc_id)
        """)
        
        # Виртуальная таблица FTS5 для полнотекстового поиска.
        # ВАЖНО: это standalone-индекс (без content='blocks'). Ранее использовалась
        # связь content='blocks', content_rowid='rowid', которая несовместима с
        # PRIMARY KEY(block_id TEXT): при удалении блоков rowid переиспользовались,
        # индекс рассогласовывался и SQLite выдавал "database disk image is malformed".
        cursor.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS blocks_fts USING fts5(
                block_id UNINDEXED,
                text
            )
        """)

        # GRAPH-RAG (Этап 1): таблица онтологических связей математических
        # сущностей. Создана отдельно от blocks, чтобы НЕ менять существующую
        # схему (ограничение ТЗ). entity_type хранится в JSON metadata вершины.
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS entity_relations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                doc_id TEXT NOT NULL,
                source_chunk_id TEXT NOT NULL,
                target_chunk_id TEXT NOT NULL,
                relation_type TEXT NOT NULL,
                metadata TEXT DEFAULT '{}',
                FOREIGN KEY (doc_id) REFERENCES documents(doc_id) ON DELETE CASCADE
            )
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_er_doc ON entity_relations(doc_id)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_er_source
            ON entity_relations(source_chunk_id)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_er_target
            ON entity_relations(target_chunk_id)
        """)

        # Примечание: отдельный триггер для updated_at не нужен —
        # save_document() сам выставляет updated_at = CURRENT_TIMESTAMP.
        # Старый AFTER UPDATE-триггер делал UPDATE той же таблицы изнутри,
        # что приводило к рекурсивному срабатыванию триггера.
        
        self.conn.commit()
        logger.debug("Таблицы SQLite созданы или уже существуют")
    
    def save_document(self, doc: Any):
        """
        Сохраняет или обновляет метаданные документа.
        Использует INSERT OR REPLACE для безопасного перезаписывания при реиндексации.
        
        Args:
            doc: Объект документа (например, ParsedDocument из ingestion.models)
        """
        try:
            # Извлекаем атрибуты с безопасными значениями по умолчанию
            doc_id = getattr(doc, 'doc_id', '')
            title = getattr(doc, 'title', 'Без названия')
            author = getattr(doc, 'author', None)
            file_format = getattr(doc, 'file_format', 'pdf')
            source_path = getattr(doc, 'source_path', '')
            language = getattr(doc, 'language', 'ru')
            total_pages = getattr(doc, 'total_pages', 0)
            status = getattr(doc, 'status', 'READY')
            
            query = """
                INSERT OR REPLACE INTO documents 
                (doc_id, title, author, file_format, source_path, language, total_pages, status, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """
            
            self.conn.execute(query, (
                doc_id, title, author, file_format, source_path, 
                language, total_pages, status
            ))
            self.conn.commit()
            logger.info(f"Документ сохранён/обновлён: {title} ({doc_id[:8]}...)")
            
        except Exception as e:
            logger.error(f"Ошибка сохранения документа: {e}")
            self.conn.rollback()
            raise
    
    def get_all_documents(self) -> List[Dict[str, Any]]:
        """
        Получает список всех документов из базы данных.
        
        Returns:
            Список словарей с информацией о документах, отсортированный по дате создания (новые сверху).
        """
        try:
            cursor = self.conn.execute("""
                SELECT doc_id, title, author, file_format, source_path, 
                       language, total_pages, status, created_at, updated_at
                FROM documents
                ORDER BY created_at DESC
            """)
            
            documents = []
            for row in cursor.fetchall():
                documents.append({
                    "doc_id": row[0],
                    "title": row[1],
                    "author": row[2] or "Неизвестный автор",
                    "file_format": row[3],
                    "source_path": row[4],
                    "language": row[5],
                    "total_pages": row[6],
                    "status": row[7],
                    "created_at": row[8],
                    "updated_at": row[9],
                })
            
            return documents
            
        except Exception as e:
            logger.error(f"Ошибка получения списка документов: {e}")
            return []
    
    def get_document_by_id(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Получает метаданные конкретного документа по его ID."""
        cursor = self.conn.execute("""
            SELECT doc_id, title, author, file_format, source_path, 
                   language, total_pages, status, created_at, updated_at
            FROM documents
            WHERE doc_id = ?
        """, (doc_id,))
        
        row = cursor.fetchone()
        if row:
            return {
                "doc_id": row[0],
                "title": row[1],
                "author": row[2],
                "file_format": row[3],
                "source_path": row[4],
                "language": row[5],
                "total_pages": row[6],
                "status": row[7],
                "created_at": row[8],
                "updated_at": row[9],
            }
        return None
    
    def save_blocks(self, blocks_data: List[Dict[str, Any]]):
        """
        Сохраняет блоки и записи FTS5-индекса в одной транзакции.

        Args:
            blocks_data: Список словарей с ключами block_id, doc_id, block_type,
                         text, page_number, chapter, section, coordinates,
                         extraction_method.
        """
        try:
            for b in blocks_data:
                coords = b.get("coordinates")
                if coords is not None and not isinstance(coords, str):
                    coords = json.dumps(coords, ensure_ascii=False)

                self.conn.execute(
                    """INSERT OR REPLACE INTO blocks
                       (block_id, doc_id, block_type, text, page_number, chapter,
                        section, coordinates, extraction_method)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        b["block_id"], b["doc_id"], b.get("block_type", "text"),
                        b["text"], b.get("page_number"), b.get("chapter") or "",
                        b.get("section") or "", coords,
                        b.get("extraction_method", "unknown"),
                    ),
                )
                # Синхронизируем standalone-индекс FTS5 (без триггеров).
                self.conn.execute("DELETE FROM blocks_fts WHERE block_id = ?", (b["block_id"],))
                self.conn.execute(
                    "INSERT INTO blocks_fts (block_id, text) VALUES (?, ?)",
                    (b["block_id"], b["text"]),
                )
            self.conn.commit()
        except Exception as e:
            logger.error(f"Ошибка сохранения блоков: {e}")
            self.conn.rollback()
            raise

    def save_entity_relations(self, doc_id: str, relations: List[Dict[str, Any]]):
        """GRAPH-RAG (Этап 1): сохраняет рёбра графа знаний документа.

        Args:
            doc_id: идентификатор документа.
            relations: список словарей {source_chunk_id, target_chunk_id,
                     relation_type, metadata(dict, опционально)}.
        """
        try:
            self.conn.execute(
                "DELETE FROM entity_relations WHERE doc_id = ?", (doc_id,)
            )
            for r in relations:
                meta = r.get("metadata") or {}
                if not isinstance(meta, str):
                    meta = json.dumps(meta, ensure_ascii=False)
                self.conn.execute(
                    """INSERT OR REPLACE INTO entity_relations
                       (doc_id, source_chunk_id, target_chunk_id,
                        relation_type, metadata)
                       VALUES (?, ?, ?, ?, ?)""",
                    (
                        doc_id,
                        r["source_chunk_id"],
                        r["target_chunk_id"],
                        r["relation_type"],
                        meta,
                    ),
                )
            self.conn.commit()
            logger.debug(f"Graph-RAG: сохранено {len(relations)} рёбер для {doc_id[:8]}...")
        except Exception as e:
            logger.error(f"Ошибка сохранения рёбер графа: {e}")
            self.conn.rollback()
            raise

    def expand_graph_context(
        self,
        seed_ids: List[str],
        max_hops: int = 2,
        limit: int = 60,
    ) -> List[Dict[str, Any]]:
        """GRAPH-RAG (Этап 2): транзитивное раскрытие подграфа от точек входа.

        Обход в обе стороны: от найденной теоремы добираем её доказательства
        (reverse PROVES) и определения, на которые она ссылается (REFERENCES),
        а также ссылки этих определений (hop 2). Возвращает полные тексты
        блоков с метаданными в порядке BFS от seeds.

        Args:
            seed_ids: block_id точек входа из гибридного поиска.
            max_hops: глубина обхода (1-3).
            limit: максимум возвращаемых блоков.

        Returns:
            Список словарей {block_id, text, page_number, chapter, section,
            entity_type, hop}.
        """
        if not seed_ids:
            return []
        visited: Dict[str, int] = {sid: 0 for sid in seed_ids}
        frontier = list(seed_ids)

        for hop in range(1, max_hops + 1):
            if not frontier or len(visited) >= limit:
                break
            fp = ",".join("?" * len(frontier))
            rows = self.conn.execute(
                f"""
                SELECT source_chunk_id, target_chunk_id FROM entity_relations
                WHERE source_chunk_id IN ({fp})
                   OR target_chunk_id IN ({fp})
                """,
                frontier + frontier,
            ).fetchall()
            nxt = []
            for src, tgt in rows:
                for node in (src, tgt):
                    if node not in visited and node not in nxt:
                        nxt.append(node)
                        visited[node] = hop
            frontier = nxt

        all_ids = list(visited.keys())[:limit]
        ph = ",".join("?" * len(all_ids))
        rows = self.conn.execute(
            f"""SELECT block_id, text, page_number, chapter, section
                FROM blocks WHERE block_id IN ({ph})""",
            all_ids,
        ).fetchall()
        by_id = {r[0]: r for r in rows}

        out: List[Dict[str, Any]] = []
        for bid in all_ids:
            r = by_id.get(bid)
            if not r:
                continue
            try:
                from src.processing.entity_extractor import classify_entity
            except ImportError:  # pragma: no cover
                from ..processing.entity_extractor import classify_entity
            out.append({
                "block_id": r[0],
                "text": r[1],
                "page_number": r[2],
                "chapter": r[3] or "",
                "section": r[4] or "",
                "entity_type": classify_entity(r[1] or ""),
                "hop": visited.get(bid, 0),
            })
        return out

    def delete_blocks_of(self, doc_id: str):
        """Удаляет блоки/FTS/рёбра графа документа, СОХРАНЯЯ карточку в
        таблице documents. Используется при реимпорте: карточка должна
        оставаться в библиотеке на всё время переиндексации (иначе при
        закрытии приложения во время векторизации книга «исчезает», хотя
        векторы в ChromaDB уже записаны)."""
        try:
            self.conn.execute(
                "DELETE FROM entity_relations WHERE doc_id = ?", (doc_id,)
            )
            self.conn.execute(
                "DELETE FROM blocks_fts WHERE block_id IN "
                "(SELECT block_id FROM blocks WHERE doc_id = ?)",
                (doc_id,),
            )
            self.conn.execute("DELETE FROM blocks WHERE doc_id = ?", (doc_id,))
            self.conn.commit()
            logger.debug(f"Блоки документа {doc_id[:8]}... удалены (карточка сохранена)")
        except Exception as e:
            logger.error(f"Ошибка удаления блоков {doc_id}: {e}")
            self.conn.rollback()
            raise

    def delete_document(self, doc_id: str):
        """
        Полностью удаляет документ и все связанные с ним блоки из базы данных
        (включая записи FTS5-индекса) в одной транзакции.
        """
        try:
            # GRAPH-RAG: чистим связи (FK CASCADE тоже сработал бы, но
            # явный DELETE гарантирует корректность при реимпорте).
            self.conn.execute(
                "DELETE FROM entity_relations WHERE doc_id = ?", (doc_id,)
            )
            # Сначала чистим FTS5 — standalone-индекс синхронизируется вручную.
            self.conn.execute(
                "DELETE FROM blocks_fts WHERE block_id IN (SELECT block_id FROM blocks WHERE doc_id = ?)",
                (doc_id,),
            )
            # Затем блоки и сам документ.
            self.conn.execute("DELETE FROM blocks WHERE doc_id = ?", (doc_id,))
            self.conn.execute("DELETE FROM documents WHERE doc_id = ?", (doc_id,))

            self.conn.commit()
            logger.info(f"Документ {doc_id[:8]}... и все его блоки успешно удалены из SQLite")

        except Exception as e:
            logger.error(f"Ошибка удаления документа {doc_id}: {e}")
            self.conn.rollback()
            raise

    def search_fts(self, phrase: str, limit: int) -> List[Dict[str, Any]]:
        """
        Полнотекстовый поиск по FTS5-индексу.

        ✅ ИСПРАВЛЕНИЕ: раньше весь запрос оборачивался во фразу '"..."' —
        точное совпадение всей последовательности слов встречался редко, и
        FTS-канал гибридного поиска почти всегда возвращал 0 результатов.
        Теперь запрос разбивается на слова, каждое экранируется как строковый
        литерал (безопасно для MATCH), а повтор даётся через OR: bm25 выше
        ранжирует документы, содержащие больше разных слов запроса.

        Args:
            phrase: Текстовый запрос.
            limit: Максимальное число результатов.

        Returns:
            Список словарей с текстом блока и метаданными документа.
        """
        or_query = self._build_match_query(phrase, joiner=" OR ")
        if not or_query:
            return []

        # Бэкап-запросы через AND: если OR-набор слишком большой/шумный,
        # AND даёт более точные совпадения. Объединяем результаты.
        queries = [or_query]
        and_query = self._build_match_query(phrase, joiner=" AND ")
        if and_query and and_query != or_query:
            queries.append(and_query)

        rows: Dict[str, sqlite3.Row] = {}
        for query in queries:
            cursor = self.conn.execute(
                """SELECT f.block_id, f.text, b.doc_id, b.page_number, b.chapter,
                          d.title, d.file_format
                   FROM blocks_fts f
                   JOIN blocks b ON b.block_id = f.block_id
                   JOIN documents d ON d.doc_id = b.doc_id
                   WHERE blocks_fts MATCH ?
                   ORDER BY bm25(blocks_fts)
                   LIMIT ?""",
                (query, limit),
            )
            for r in cursor.fetchall():
                rows.setdefault(r[0], r)

        # FIX (Тест 2 / А1): короткие идентификаторы заданий («А1», «Б2») и
        # склеенные токены («тест2а1») FTS5/bm25 не находит — токенезатор
        # unicode61 выбрасывает односимвольные слова и склеивает «Тест 2 А1»
        # с соседями при плотной вёрстке. Добавляем LIKE-канал по редким
        # коротким тегам запроса: он ДОПОЛНЯЕТ полнотекстовые результаты,
        # чтобы страница с тестом гарантированно попала в контекст RAG.
        try:
            for tok in self._rare_short_tokens(phrase):
                # LIKE всегда ищет токен как подстроку («%а1%»), поэтому
                # находит и «ТЕСТ 2 А1», и слитый OCR-«ВАРИАНТТЕСТ2А1...».
                for r in self._like_search(tok, limit):
                    rows.setdefault(r[0], r)
        except Exception as e:  # LIKE-канал — косметика, не роняем поиск
            logger.debug(f"LIKE-канал поиска недоступен: {e}")

        results = [
            {
                "block_id": r[0],
                "text": r[1],
                "doc_id": r[2],
                "page_number": r[3],
                "chapter": r[4],
                "title": r[5],
                "file_format": r[6],
            }
            for r in list(rows.values())[: max(limit * 2, limit)]
        ]
        return results

    @staticmethod
    def _build_match_query(phrase: str, joiner: str) -> str:
        """Строит безопасный FTS5 MATCH-запрос из естественного текста."""
        words = re.findall(r"[\wёЁ-]{2,}", phrase, flags=re.UNICODE)
        if not words:
            # Если слов нет — пробуем экранировать всю фразу целиком.
            cleaned = phrase.replace('"', '""').strip()
            return f'"{cleaned}"' if cleaned else ""
        quoted = [f'"{w}"' for w in dict.fromkeys(words)]  # уникальные, порядок сохранён
        return joiner.join(quoted)

    def _rare_short_tokens(self, phrase: str) -> List[str]:
        """Короткие «теги» запроса (А1, Б2, В3), которые не берёт FTS5.

        Токенизатор unicode61 отбрасывает односимвольные слова и склеивает
        плотную нумерацию тестов с соседями, поэтому bm25 по таким меткам
        ничего не находит. Берём токены вида <1 буква><1-2 цифры>, а также
        склеенные варианты («тест2а1»). Возвращаем не более 3 меток.
        """
        text = phrase.lower()
        cands: List[str] = []
        # Раздельные короткие метки: «а1», «б2», «в12».
        for tok in re.findall(r"[а-яёa-z]\d{1,2}", text):
            if tok not in cands:
                cands.append(tok)
        # Склеенные токены с цифрой внутри: «тест2а1», «вариант3б2».
        for tok in re.findall(r"[а-яёa-z]*\d+[а-яёa-z\d]*", text):
            if len(tok) >= 4 and any(ch.isalpha() for ch in tok) \
                    and tok not in cands:
                cands.append(tok)
        return cands[:3]

    def _like_search(self, token: str, limit: int) -> List[sqlite3.Row]:
        """LIKE-поиск подстроки по блокам (для тегов, недоступных FTS5).

        ВАЖНО: встроенный LIKE в SQLite регистрозависим для кириллицы и не
        умеет нормализовать текст, поэтому сравнение делаем на стороне
        Python: нижний регистр + схлопывание пробелов. Это ловит и
        «ТЕСТ 2 А1», и слитый OCR-вариант «ВАРИАНТТЕСТ2А1...». Блоки
        сортируются от коротких к длинным, чтобы первыми шли плотные
        страницы с заданиями, а не многостраничные главы.
        """
        token = token.strip("%").lower()
        if not token:
            return []
        compact = re.sub(r"\s+", "", token)
        cursor = self.conn.execute(
            """SELECT b.block_id, b.text, b.doc_id, b.page_number, b.chapter,
                      d.title, d.file_format
               FROM blocks b
               JOIN documents d ON d.doc_id = b.doc_id""",
        )
        matched: list = []
        for r in cursor.fetchall():
            text_norm = (r[1] or "").lower()
            if token in text_norm or compact in re.sub(r"\s+", "", text_norm):
                matched.append((len(text_norm), r))
        matched.sort(key=lambda t: t[0])
        return [r for _, r in matched[: max(limit, 5)]]

    def get_chapter_blocks(self, doc_id: str, chapter: str,
                           start_page: Optional[int] = None,
                           end_page: Optional[int] = None) -> List[Dict[str, Any]]:
        """Получает ВСЕ блоки главы по её названию (Full-Chapter Context).

        REFACTORING (ТЗ п.1 «цифровой двойник»): вместо наивного RAG с
        ограниченным top_k модель получает весь параграф целиком — все блоки
        главы в исходном порядке чтения (page_number → rowid). Совпадение
        названия нестрогое (подстрока, регистронезависимо), т.к. TOC-записи
        и блоки могут различаться суффиксами/двоеточиями.
        """
        if not chapter:
            return []
        # ВАЖНО: lower() в SQL не работает с кириллицей (SQLite ASCII-only),
        # поэтому нормализуем регистр на стороне Python для обеих строк.
        needle = chapter.lower().strip()
        query = ("SELECT block_id, block_type, text, page_number, chapter, "
                 "section, coordinates, extraction_method FROM blocks "
                 "WHERE doc_id = ? AND chapter IS NOT NULL")
        params: List[Any] = [doc_id]
        if start_page is not None:
            query += " AND page_number >= ?"
            params.append(int(start_page))
        if end_page is not None:
            query += " AND page_number <= ?"
            params.append(int(end_page))
        query += " ORDER BY page_number ASC, rowid ASC"
        try:
            cursor = self.conn.execute(query, params)
            rows = [r for r in cursor.fetchall()
                    if needle in (r[4] or "").lower()]
        except Exception as e:  # битая БД — не роняем чат, деградируем к поиску
            logger.warning(f"get_chapter_blocks failed: {e}")
            return []
        blocks = []
        for row in rows:
            blocks.append({
                "block_id": row[0],
                "block_type": row[1],
                "text": row[2],
                "page_number": row[3],
                "chapter": row[4],
                "section": row[5],
                "coordinates": json.loads(row[6]) if row[6] else None,
                "extraction_method": row[7],
            })
        return blocks

    def get_blocks_by_doc_id(self, doc_id: str) -> List[Dict[str, Any]]:
        """Получает все текстовые блоки для конкретного документа."""
        cursor = self.conn.execute("""
            SELECT block_id, block_type, text, page_number, chapter, section, coordinates, extraction_method
            FROM blocks
            WHERE doc_id = ?
            ORDER BY page_number ASC, rowid ASC
        """, (doc_id,))
        
        blocks = []
        for row in cursor.fetchall():
            blocks.append({
                "block_id": row[0],
                "block_type": row[1],
                "text": row[2],
                "page_number": row[3],
                "chapter": row[4],
                "section": row[5],
                "coordinates": json.loads(row[6]) if row[6] else None,
                "extraction_method": row[7]
            })
        return blocks

    def flush(self):
        """Мгновенно закрепляет все незаписанные изменения на диске:
        commit + PRAGMA wal_checkpoint(TRUNCATE). Вызывается после каждого
        импорта, чтобы карточка книги не зависела от корректного закрытия
        приложения (фикс «карточка утеряна после перезапуска»)."""
        if not self.conn:
            return
        try:
            self.conn.commit()
            self.conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            logger.debug("SQLite: WAL закреплён на диске (flush)")
        except sqlite3.Error as e:
            logger.warning(f"wal_checkpoint(TRUNCATE) не выполнен: {e}")

    def close(self):
        """Закрывает соединение с базой данных.

        АУДИТ (баг «после перезапуска источники не отображаются»): перед
        закрытием принудительно вливаем WAL в основной файл БД
        (PRAGMA wal_checkpoint(TRUNCATE)). Иначе при аварийном завершении
        процесса часть последних коммитов остаётся в library.db-wal, и
        следующая операция записи может выполнить rollback всего журнала —
        импортированные книги «исчезают» после рестарта приложения.
        """
        if self.conn:
            try:
                self.conn.commit()
                self.conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            except sqlite3.Error as e:
                logger.warning(f"wal_checkpoint не выполнен: {e}")
            self.conn.close()
            logger.info("Соединение с SQLite закрыто")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()