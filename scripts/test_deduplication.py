# scripts/test_deduplication.py
"""Проверка дедупликации: повторный импорт не создаёт дубликат."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ingestion.document_router import DocumentRouter
from src.storage.sqlite_store import SQLiteStore
from src.core.settings import Settings


def main():
    if len(sys.argv) < 2:
        print("Использование: python test_deduplication.py <путь_к_pdf>")
        sys.exit(1)
    
    file_path = Path(sys.argv[1])
    settings = Settings.load(Path("config.yaml"))
    store = SQLiteStore(settings.sqlite_path)
    router = DocumentRouter()
    
    print(f"\n=== Тест дедупликации ===")
    print(f"Файл: {file_path}")
    
    # Первый импорт
    print("\n[1] Первый импорт...")
    doc1 = router.parse(file_path)
    saved1 = store.save_document(doc1)
    print(f"  Сохранён: {saved1}")
    
    # Второй импорт (тот же файл)
    print("\n[2] Повторный импорт (тот же файл)...")
    doc2 = router.parse(file_path)
    saved2 = store.save_document(doc2)
    print(f"  Сохранён: {saved2}")
    print(f"  Ожидалось: False (дубликат)")
    
    # Проверка
    if saved1 and not saved2:
        print("\n✅ Дедупликация работает корректно!")
    else:
        print(f"\n❌ Ошибка: saved1={saved1}, saved2={saved2}")
    
    # Извлечение из БД
    print(f"\n[3] Извлечение из БД...")
    loaded = store.get_document(doc1.doc_id)
    if loaded:
        print(f"  Название: {loaded.title}")
        print(f"  Блоков: {len(loaded.blocks)}")
        print(f"  Статус: {loaded.status.value}")
    else:
        print("  ❌ Документ не найден в БД!")
    
    store.close()


if __name__ == "__main__":
    main()