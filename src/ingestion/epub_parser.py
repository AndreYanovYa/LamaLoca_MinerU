# src/ingestion/epub_parser.py
from pathlib import Path
from loguru import logger
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
import warnings

# Подавляем предупреждение: EPUB использует XHTML, но HTML-парсер работает нормально
from bs4 import XMLParsedAsHTMLWarning
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

from .base_parser import BaseParser
from .models import ParsedDocument, DocumentBlock, BlockType
from ..core.exceptions import DocumentParseError


class EpubParser(BaseParser):
    """Улучшенный парсер EPUB через EbookLib + BeautifulSoup.
    
    Гарантирует правильный порядок чтения через book.spine.
    Устойчив к нестандартной вёрстке (отсутствие тегов <p>).
    """
    
    def supported_extensions(self) -> list[str]:
        return [".epub"]
    
    def parse(self, file_path: Path, doc_id: str, progress_callback=None) -> ParsedDocument:
        logger.info(f"Парсинг EPUB: {file_path.name}")
        
        try:
            # ignore_ncx=False заставляет использовать современный nav document, если он есть
            book = epub.read_epub(str(file_path), options={"ignore_ncx": False})
        except Exception as e:
            raise DocumentParseError(f"Не удалось открыть EPUB {file_path.name}: {e}") from e
        
        try:
            metadata = self._extract_metadata(book)
            title = metadata.get("title") or file_path.stem
            author = metadata.get("creator") or None
            language = metadata.get("language") or "unknown"
            
            blocks: list[DocumentBlock] = []
            block_counter = 0
            
            # Используем spine для гарантированного порядка чтения
            spine_items = [book.get_item_with_id(item_id) for item_id, _ in book.spine if book.get_item_with_id(item_id)]
            
            # Если spine пуст (редкий случай битого EPUB), берём все документы
            if not spine_items:
                spine_items = list(book.get_items_of_type(ebooklib.ITEM_DOCUMENT))
            
            for chapter_idx, item in enumerate(spine_items):
                chapter_title = self._extract_chapter_title(item)
                chapter_label = f"Глава {chapter_idx + 1}: {chapter_title}"
                
                chapter_blocks = self._extract_chapter_blocks(
                    item, chapter_label, block_counter
                )
                blocks.extend(chapter_blocks)
                block_counter += len(chapter_blocks)
            
            logger.info(
                f"EPUB извлечён: {len(spine_items)} частей, {len(blocks)} блоков, язык: {language}"
            )
            
            return ParsedDocument(
                doc_id=doc_id,
                source_path=str(file_path.resolve()),
                original_filename=file_path.name,
                file_format="epub",
                title=title,
                author=author,
                language=language,
                total_pages=None,
                blocks=blocks,
            )
        finally:
            pass
    
    def _extract_metadata(self, book: epub.EpubBook) -> dict:
        metadata = {}
        for prop in ["title", "creator", "language"]:
            items = book.get_metadata("DC", prop)
            if items:
                metadata[prop] = items[0][0]
        return metadata
    
    def _extract_chapter_title(self, item: epub.EpubHtml) -> str:
        html_content = item.get_content().decode("utf-8", errors="ignore")
        soup = BeautifulSoup(html_content, "lxml")
        
        for tag in ["h1", "h2", "h3"]:
            heading = soup.find(tag)
            if heading and len(heading.get_text(strip=True)) > 3:
                return heading.get_text(strip=True)
        
        # Если заголовка нет, очищаем имя файла от расширения
        name = item.get_name().replace(".xhtml", "").replace(".html", "")
        return name if name else "Без названия"
    
    def _extract_chapter_blocks(
        self,
        item: epub.EpubHtml,
        chapter_label: str,
        start_block_id: int,
    ) -> list[DocumentBlock]:
        """Извлекает блоки текста, устойчиво к разной вёрстке."""
        html_content = item.get_content().decode("utf-8", errors="ignore")
        soup = BeautifulSoup(html_content, "lxml")
        
        blocks: list[DocumentBlock] = []
        block_counter = 0
        
        # Ищем смысловые теги. Порядок важен: заголовки сначала, потом абзацы и списки
        elements = soup.find_all(["h1", "h2", "h3", "h4", "p", "li", "div"])
        
        for element in elements:
            text = element.get_text(separator=" ", strip=True)
            
            # Фильтр мусора: игнорируем блоки короче 20 символов (навигация, номера страниц)
            if len(text) < 20:
                continue
            
            # Определяем тип блока
            if element.name in ["h1", "h2", "h3", "h4"]:
                block_type = BlockType.HEADING
            else:
                block_type = BlockType.TEXT
            
            # Защита от дублирования: если этот текст уже добавлен в предыдущий блок, пропускаем
            if blocks and blocks[-1].text == text:
                continue
                
            blocks.append(DocumentBlock(
                block_id=f"b{start_block_id + block_counter}",
                block_type=block_type,
                text=text,
                page_number=None,
                chapter=chapter_label,
                section=None,
                coordinates=None,
                extraction_method="ebooklib",
            ))
            block_counter += 1
        
        # Fallback: если вообще ничего не нашли, берём весь текст и бьём по абзацам
        if not blocks:
            full_text = soup.get_text(separator="\n", strip=True)
            paragraphs = [p.strip() for p in full_text.split("\n") if len(p.strip()) > 50]
            
            for para in paragraphs:
                blocks.append(DocumentBlock(
                    block_id=f"b{start_block_id + block_counter}",
                    block_type=BlockType.TEXT,
                    text=para,
                    page_number=None,
                    chapter=chapter_label,
                    section=None,
                    coordinates=None,
                    extraction_method="ebooklib",
                ))
                block_counter += 1
        
        return blocks