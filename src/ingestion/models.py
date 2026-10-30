# src/ingestion/models.py
from dataclasses import dataclass, field
from typing import List, Optional
from enum import Enum
from datetime import datetime

class BlockType(str, Enum):
    TEXT = "text"
    HEADING = "heading"
    FORMULA = "formula"
    CODE = "code"
    TABLE = "table"
    FIGURE = "figure"  # Level 1 ТЗ: ленивый PNG-рендер зоны со сложной графикой

class IngestionStatus(str, Enum):
    PENDING = "pending"
    PARSING = "parsing"
    CHUNKING = "chunking"
    INDEXING = "indexing"
    READY = "ready"
    FAILED = "failed"

@dataclass
class DocumentBlock:
    block_id: str
    block_type: BlockType
    text: str
    # Для PDF/DjVu:
    page_number: Optional[int] = None
    # Для EPUB/FB2 (приоритетнее page_number, если он None):
    chapter: Optional[str] = None
    section: Optional[str] = None
    # Для точных цитат и рендеринга:
    coordinates: Optional[dict] = None
    extraction_method: str = "unknown" # "pymupdf", "tesseract", "ebooklib"

@dataclass
class ParsedDocument:
    # SHA-256 хеш содержимого файла как стабильный ID
    doc_id: str
    source_path: str
    original_filename: str
    file_format: str # "pdf", "epub", "fb2", "djvu", "txt"
    title: str
    author: Optional[str]
    language: str
    total_pages: Optional[int] # None для EPUB/FB2 без фиксированной пагинации
    blocks: List[DocumentBlock]
    status: IngestionStatus = IngestionStatus.PENDING
    error_message: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)