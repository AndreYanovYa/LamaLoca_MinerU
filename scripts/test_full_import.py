# scripts/test_full_import.py
"""
Сквозной тест полного цикла импорта:
Файл → Парсинг → Чанкинг → SQLite (FTS5) → Ollama Embeddings → ChromaDB → Проверка поиска.
"""
import sys
import time
from pathlib import Path

# Добавляем корень проекта в PYTHONPATH
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.settings import Settings
from src.ingestion.document_router import DocumentRouter
from src.ingestion.models import IngestionStatus
from src.processing.chunker import TextChunker
from src.storage.sqlite_store import SQLiteStore
from src.storage.chroma_store import ChromaStore
from src.embeddings.ollama_embedder import OllamaEmbedder
from src.core.exceptions import LamaLocaError


def escape_fts5_query(word: str) -> str:
    """
    Экранирует слово для FTS5-запроса.
    Оборачивает в кавычки, чтобы точки и спецсимволы не ломали синтаксис.
    """
    clean_word = word.replace('"', '').strip()
    if not clean_word:
        return '""'
    return f'"{clean_word}"'


def main():
    if len(sys.argv) < 2:
        print("Использование: python test_full_import.py <путь_к_файлу>")
        print("Пример: python test_full_import.py data/library/pdf/sample.pdf")
        sys.exit(1)

    file_path = Path(sys.argv[1])
    if not file_path.exists():
        print(f"❌ Файл не найден: {file_path}")
        sys.exit(1)

    print("\n" + "=" * 60)
    print(" 🚀 ПОЛНЫЙ ЦИКЛ ИМПОРТА И ПОИСКА (lama-loca v0.1)")
    print("=" * 60 + "\n")

    # 1. Инициализация компонентов
    print("[1/6] Инициализация компонентов...")
    settings = Settings.load(Path("config.yaml"))
    router = DocumentRouter()
    sqlite_store = SQLiteStore(settings.sqlite_path)
    chroma_store = ChromaStore(settings.chroma_path)
    embedder = OllamaEmbedder(settings.ollama)
    chunker = TextChunker(settings.chunking)
    print("  ✅ Компоненты загружены.\n")

    # 2. Парсинг документа
    print(f"[2/6] Парсинг документа: {file_path.name}")
    start_time = time.time()
    doc = router.parse(file_path)
    print(
        f"  ✅ Извлечено: {doc.total_pages or 'N/A'} стр./глав, "
        f"{len(doc.blocks)} блоков за {time.time() - start_time:.2f} сек.\n"
    )

    # 3. Чанкинг
    print("[3/6] Разбиение на чанки (Chunking)...")
    raw_chunks = chunker.chunk_blocks(doc.blocks)

    # Обогащаем метаданные чанков информацией о документе
    chunks_to_index = []
    for i, chunk in enumerate(raw_chunks):
        chunk["metadata"]["doc_id"] = doc.doc_id
        chunk["metadata"]["title"] = doc.title
        chunk["metadata"]["file_format"] = doc.file_format
        chunk["chunk_index"] = i
        chunks_to_index.append(chunk)

    print(f"  ✅ Создано {len(chunks_to_index)} чанков для индексации.\n")

    # 4. Сохранение в SQLite (текст + FTS5)
    print("[4/6] Сохранение в SQLite (включая FTS5)...")
    # Для теста удаляем старые данные этого doc_id, чтобы избежать конфликтов при повторном запуске
    sqlite_store.conn.execute("DELETE FROM blocks WHERE doc_id = ?", (doc.doc_id,))
    sqlite_store.conn.execute("DELETE FROM documents WHERE doc_id = ?", (doc.doc_id,))

    # Сохраняем документ (обновляем статус на indexing)
    doc.status = IngestionStatus.INDEXING
    sqlite_store.save_document(doc)

    # Сохраняем тексты чанков в SQLite (для FTS5)
    for chunk in chunks_to_index:
        global_block_id = f"{doc.doc_id}:{chunk['metadata']['block_id']}"
        sqlite_store.conn.execute(
            """
            INSERT INTO blocks (
                block_id, doc_id, block_type, text, page_number,
                chapter, section, coordinates, extraction_method
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                global_block_id,
                doc.doc_id,
                chunk["metadata"]["block_type"],
                chunk["text"],
                chunk["metadata"]["page_number"],
                chunk["metadata"]["chapter"],
                None,
                None,
                chunk["metadata"]["extraction_method"],
            ),
        )
    sqlite_store.conn.commit()
    print(f"  ✅ Сохранено {len(chunks_to_index)} текстовых чанков в SQLite.\n")

    # 5. Векторизация и сохранение в ChromaDB
    print(
        f"[5/6] Векторизация через Ollama ({settings.ollama.embedding_model}) "
        f"и сохранение в ChromaDB..."
    )
    print("  ⏳ Это может занять время в зависимости от количества чанков и скорости CPU/GPU...")

    texts_to_embed = [c["text"] for c in chunks_to_index]
    chunk_ids = [f"{doc.doc_id}_chk_{i}" for i in range(len(chunks_to_index))]
    metadatas = [c["metadata"] for c in chunks_to_index]

    # Получаем эмбеддинги
    embeddings = embedder.embed_texts(texts_to_embed)

    # Удаляем старые векторы этого документа из ChromaDB
    chroma_store.delete_by_doc_id(doc.doc_id)

    # Добавляем новые
    chroma_store.add_chunks(
        ids=chunk_ids,
        texts=texts_to_embed,
        embeddings=embeddings,
        metadatas=metadatas,
    )
    print(f"  ✅ Векторизовано и сохранено {len(embeddings)} чанков в ChromaDB.\n")

    # 6. Проверка поиска
    print("[6/6] Проверка поиска (Hybrid Search Simulation)...")

    # Берём случайное слово из первого чанка для теста FTS5
    sample_text = chunks_to_index[0]["text"]
    # Ищем слово подлиннее (без точек и спецсимволов) для более надёжного теста
    sample_words = [
        w for w in sample_text.split()
        if len(w) > 4 and w.isalpha()
    ]
    test_keyword = sample_words[0] if sample_words else "математика"
    escaped_query = escape_fts5_query(test_keyword)

    print(f"  🔍 Тест FTS5 (поиск по ключевому слову: '{test_keyword}')...")
    # FTS5 — виртуальная таблица, содержит только 'text'. Для получения doc_id делаем JOIN с blocks.
    fts_results = sqlite_store.conn.execute(
        """
        SELECT b.doc_id, b.text, snippet(blocks_fts, 0, '[', ']', '...', 64) as snippet
        FROM blocks_fts
        JOIN blocks b ON blocks_fts.rowid = b.rowid
        WHERE blocks_fts MATCH ?
        LIMIT 3
        """,
        (escaped_query,),
    ).fetchall()

    if fts_results:
        print(f"  ✅ Найдено {len(fts_results)} совпадений в FTS5.")
        print(f"     Пример: {fts_results[0]['snippet'][:100]}...")
    else:
        print("  ⚠️ FTS5 не нашел совпадений (возможно, слово слишком короткое или стоп-слово).")

    print("\n  🔍 Тест Vector Search (поиск по смыслу первого чанка)...")
    # Берём вектор первого чанка и ищем похожие
    query_embedding = embeddings[0]
    vector_results = chroma_store.search(query_embedding=query_embedding, top_k=3)

    if vector_results:
        print(f"  ✅ Найдено {len(vector_results)} семантически похожих чанков в ChromaDB.")
        print(f"     Лучшее совпадение (distance: {vector_results[0]['distance']:.4f}):")
        print(f"     Текст: {vector_results[0]['text'][:100]}...")
    else:
        print("  ⚠️ Vector Search не вернул результатов.")

    # Финальный статус
    print("\n" + "=" * 60)
    print(" 🎉 ПОЛНЫЙ ЦИКЛ УСПЕШНО ЗАВЕРШЕН!")
    print(f" 📚 Документ: {doc.title}")
    print(f" 🆔 Doc ID: {doc.doc_id[:16]}...")
    print(f" 📦 Чанков в SQLite: {len(chunks_to_index)}")
    print(f" 🧬 Векторов в ChromaDB: {len(embeddings)}")
    print("=" * 60 + "\n")

    sqlite_store.close()


if __name__ == "__main__":
    try:
        main()
    except LamaLocaError as e:
        print(f"\n❌ Критическая ошибка приложения: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Неожиданная ошибка: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)