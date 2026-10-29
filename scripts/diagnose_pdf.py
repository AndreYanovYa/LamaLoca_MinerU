# scripts/diagnose_pdf.py
"""Диагностика PDF: TOC, ссылки, текстовый слой."""
import sys
from pathlib import Path
import pymupdf

pdf_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/library/pdf/sample.pdf")

if not pdf_path.exists():
    print(f"❌ Файл не найден: {pdf_path}")
    sys.exit(1)

doc = pymupdf.open(str(pdf_path))

print(f"📄 Файл: {pdf_path.name}")
print(f" Страниц: {doc.page_count}")
print(f"📝 Метаданные: {doc.metadata}")
print()

# 1. Встроенное оглавление (TOC)
toc = doc.get_toc()
print(f"📑 Встроенное оглавление (TOC): {len(toc)} записей")
if toc:
    print("\nПервые 30 записей TOC:")
    for i, (level, title, page) in enumerate(toc[:30]):
        indent = "  " * (level - 1)
        print(f"  {indent}[ур.{level}] стр.{page:4d} — {title[:80]}")
    if len(toc) > 30:
        print(f"  ... и ещё {len(toc) - 30} записей")
else:
    print("⚠️  Встроенного TOC нет!")

print()

# 2. Ссылки на первых 10 страницах
print(" Ссылки на первых 10 страницах:")
for page_num in range(min(10, doc.page_count)):
    page = doc[page_num]
    links = page.get_links()
    if links:
        print(f"\n  Страница {page_num + 1}: {len(links)} ссылок")
        for link in links[:3]:
            if link.get("kind") == pymupdf.LINK_GOTO:
                print(f"    → стр. {link.get('page', '?')}: {link.get('uri', '')[:50]}")
            elif link.get("kind") == pymupdf.LINK_URI:
                print(f"    → URI: {link.get('uri', '')[:60]}")

print()

# 3. Качество текстового слоя (первые 5 страниц)
print("🔤 Качество текстового слоя (первые 5 страниц):")
for page_num in range(min(5, doc.page_count)):
    page = doc[page_num]
    text = page.get_text("text")
    has_cyrillic = any('\u0400' <= c <= '\u04FF' for c in text)
    print(f"  Стр. {page_num + 1}: {len(text)} символов, кириллица: {'✅' if has_cyrillic else '❌'}")
    if len(text) > 0:
        print(f"    Пример: {text[:100].replace(chr(10), ' ')}...")

doc.close()