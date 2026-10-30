# src/ingestion/document_router.py
from pathlib import Path
from loguru import logger

from .base_parser import BaseParser
from .pdf_parser import PDFParser
from .epub_parser import EpubParser
from .models import ParsedDocument
from ..core.exceptions import DocumentParseError
from ..utils.hashing import compute_file_sha256
from ..processing.ocr_processor import OCRProcessor
from ..core.settings import Settings


class DocumentRouter:
    """Маршрутизатор документов.
    
    Определяет формат файла по расширению, вычисляет SHA-256 для doc_id,
    делегирует работу нужному парсеру (с поддержкой OCR-fallback для PDF).
    """
    
    def __init__(self, settings: Settings = None):
        # Инициализируем OCR, если указан Tesseract и файл существует
        ocr_processor = None
        if settings:
            tesseract_cmd = getattr(settings, 'tesseract_cmd', None)
            if tesseract_cmd and Path(tesseract_cmd).exists():
                try:
                    ocr_processor = OCRProcessor(tesseract_cmd, language="rus+eng")
                    logger.info("OCR (Tesseract) доступен и инициализирован")
                except Exception as e:
                    logger.warning(f"OCR недоступен: {e}")
            else:
                logger.info("Tesseract не найден, OCR отключен (будет использоваться только PyMuPDF)")

            # Image-to-LaTeX распознаватель формул: текстовый слой PDF даёт
            # по формулам нечитаемую кашу — без LaTeX-распознавания ИИ на
            # математике не учится. Настройки: config.yaml -> formula_ocr.
            try:
                from ..processing.formula_recognizer import FormulaRecognizer
                focr = getattr(settings, "formula_ocr", None)
                FormulaRecognizer.configure(
                    prefer=getattr(focr, "engine", "auto") or "auto",
                    allow_network=bool(getattr(focr, "allow_network_api", False)),
                )
            except Exception as e:  # noqa: BLE001 — модуль опционален
                logger.debug(f"FormulaRecognizer не настроен: {e}")
        
        # Регистрируем парсеры. PDF: основной движок — MinerU (Markdown +
        # LaTeX-формулы, layout-модель), PyMuPDF+pix2tex остаётся fallback'ом
        # внутри PDFParser.parse() (ТЗ-2026: отказ от сырого текстового слоя).
        pdf_engine = "auto"
        self._pdf_languages = "ru,en"
        _vram_override = 0
        _fast_batch = True
        if settings:
            pe = getattr(settings, "pdf_engine", None)
            pdf_engine = (getattr(pe, "engine", "auto") or "auto").lower()
            self._pdf_languages = getattr(pe, "languages", "ru,en") or "ru,en"
            # FIX (лог 22:10): ускорение MinerU (см. PdfEngineConfig).
            _vram_override = int(getattr(pe, "vram_gb_override", 0) or 0)
            _fast_batch = bool(getattr(pe, "fast_batch", True))
        self._parsers: list[BaseParser] = [
            PDFParser(ocr_processor=ocr_processor, pdf_engine=pdf_engine,
                      languages=self._pdf_languages,
                      vram_gb_override=_vram_override, fast_batch=_fast_batch),
            EpubParser(),
        ]
        if pdf_engine in ("auto", "mineru"):
            from .mineru_parser import MinerUParser
            if MinerUParser.is_available():
                logger.info("PDF-движок: MinerU доступна и будет использована как основная")
            else:
                logger.warning(
                    "PDF-движок: MinerU НЕ установлена — используется PyMuPDF+pix2tex. "
                    "Для точной математики: pip install 'mineru[core]' && mineru-models-download"
                )
        
        # Карта расширения → парсер для быстрого поиска
        self._ext_map: dict[str, BaseParser] = {}
        for parser in self._parsers:
            for ext in parser.supported_extensions():
                self._ext_map[ext.lower()] = parser
    
    def parse(self, file_path: str | Path, progress_callback=None) -> ParsedDocument:
        """
        Парсит файл и возвращает унифицированный ParsedDocument.
        
        Args:
            file_path: Путь к файлу (поддерживает кириллицу и пробелы).
        
        Returns:
            ParsedDocument с блоками, метаданными и doc_id.
        
        Raises:
            FileNotFoundError: Если файл не существует.
            DocumentParseError: Если формат не поддерживается или парсинг не удался.
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Файл не найден: {file_path}")
        
        if not file_path.is_file():
            raise DocumentParseError(f"Путь не является файлом: {file_path}")
        
        # 1. Вычисляем SHA-256 — это наш стабильный doc_id
        logger.debug(f"Вычисление SHA-256 для {file_path.name}")
        doc_id = compute_file_sha256(file_path)
        logger.debug(f"doc_id: {doc_id[:16]}...")
        
        # 2. Находим подходящий парсер
        ext = file_path.suffix.lower()
        parser = self._ext_map.get(ext)
        
        if parser is None:
            supported = ", ".join(sorted(self._ext_map.keys()))
            raise DocumentParseError(
                f"Формат '{ext}' не поддерживается. "
                f"Поддерживаемые форматы: {supported}"
            )
        
        # 3. Делегируем парсинг
        logger.info(f"Маршрутизация: {file_path.name} → {parser.__class__.__name__}")
        return parser.parse(file_path, doc_id, progress_callback=progress_callback)
    
    def is_supported(self, file_path: str | Path) -> bool:
        """Проверяет, поддерживается ли формат файла."""
        ext = Path(file_path).suffix.lower()
        return ext in self._ext_map
    
    def supported_extensions(self) -> list[str]:
        """Возвращает список всех поддерживаемых расширений."""
        return sorted(self._ext_map.keys())