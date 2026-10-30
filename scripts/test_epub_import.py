# scripts/test_epub_import.py
"""Тест парсинга EPUB."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ingestion.document_router import DocumentRouter
from src.core.exceptions import DocumentParseError


def main():
    if len(sys.argv) < 2:
        print("Использование: python test_epub_import.py <путь_к_epub>")
        sys.exit(1)
    
    file_path = Path(sys.argv[1])
    router = DocumentRouter()
    
    print(f"\n=== Тест парсинга EPUB ===")
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
    
    print(f"✅ Успешно!")
    print(f"  doc_id:        {doc.doc_id[:32]}...")
    print(f"  Название:      {doc.title}")
    print(f"  Автор:         {doc.author or '—'}")
    print(f"  Язык:          {doc.language}")
    print(f"  Страниц:       {doc.total_pages or '—'}")
    print(f"  Блоков:        {len(doc.blocks)}")
    print(f"  Статус:        {doc.status.value}")
    
    # Показать первые 3 блока
    print(f"\n--- Первые 3 блока ---")
    for block in doc.blocks[:3]:
        preview = block.text[:100].replace("\n", " ")
        chapter_info = block.chapter or "—"
        print(f"  [{block.block_type.value}] {chapter_info} | {preview}...")


if __name__ == "__main__":
    main()