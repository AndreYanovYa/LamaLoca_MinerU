# scripts/test_pdf_import.py
"""Быстрый тест парсинга PDF без БД и эмбеддингов."""
import sys
from pathlib import Path

# Добавляем корень проекта в PYTHONPATH
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ingestion.document_router import DocumentRouter
from src.core.exceptions import DocumentParseError


def main():
    if len(sys.argv) < 2:
        print("Использование: python test_pdf_import.py <путь_к_pdf>")
        print("Пример: python test_pdf_import.py data/library/pdf/sample.pdf")
        sys.exit(1)
    
    file_path = Path(sys.argv[1])
    router = DocumentRouter()
    
    print("\n=== Тест парсинга PDF ===")
    print(f"Файл: {file_path}")
    print(f"Поддерживается: {router.is_supported(file_path)}")
    print(f"Доступные форматы: {router.supported_extensions()}")
    print()
    
    try:
        doc = router.parse(file_path)
    except DocumentParseError as e:
        print(f"❌ Ошибка парсинга: {e}")
        sys.exit(1)
    except FileNotFoundError as e:
        print(f"❌ Файл не найден: {e}")
        sys.exit(1)
    
    print("✅ Успешно!")
    print(f"  doc_id:        {doc.doc_id[:32]}...")
    print(f"  Название:      {doc.title}")
    print(f"  Автор:         {doc.author or '—'}")
    print(f"  Язык:          {doc.language}")
    print(f"  Страниц:       {doc.total_pages}")
    print(f"  Блоков:        {len(doc.blocks)}")
    print(f"  Статус:        {doc.status.value}")
    
    # Показать первые 3 блока
    print("\n--- Первые 3 блока ---")
    for block in doc.blocks[:3]:
        preview = block.text[:100].replace("\n", " ")
        print(f"  [{block.block_type.value}] стр.{block.page_number} | {preview}...")


if __name__ == "__main__":
    main()