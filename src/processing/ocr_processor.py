# src/processing/ocr_processor.py
"""OCR-обработка PDF через Tesseract с улучшенным preprocessing."""
from pathlib import Path
from loguru import logger
from typing import List, Optional

import pymupdf
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter
import io

from ..core.exceptions import ExternalToolNotFoundError
from .ocr_postprocessor import OCRPostProcessor


class OCRProcessor:
    """Распознаёт текст PDF через Tesseract OCR."""
    
    def __init__(self, tesseract_cmd: str, language: str = "rus+eng"):
        self.tesseract_cmd = Path(tesseract_cmd)
        self.language = language
        
        if not self.tesseract_cmd.exists():
            raise ExternalToolNotFoundError(
                f"Tesseract не найден по пути: {tesseract_cmd}. "
                "Установите Tesseract OCR или исправьте путь в config.yaml"
            )
        
        pytesseract.pytesseract.tesseract_cmd = str(self.tesseract_cmd)
        logger.info(f"pytesseract инициализирован: {self.tesseract_cmd}")
    
    def extract_text_from_pdf(self, pdf_path: Path, pages: Optional[List[int]] = None,
                                progress_callback=None) -> List[str]:
        """Извлекает текст из PDF через OCR."""
        logger.info(f"OCR: начинаем распознавание {pdf_path.name}")
        
        doc = pymupdf.open(str(pdf_path))
        page_texts = []
        
        try:
            total_pages = doc.page_count
            pages_to_process = pages if pages is not None else list(range(total_pages))
            
            logger.info(f"OCR: обработка {len(pages_to_process)} из {total_pages} страниц")
            
            processed = 0
            for page_num in pages_to_process:
                page = doc[page_num]
                
                # DPI 400 для лучшего качества
                mat = pymupdf.Matrix(450 / 72, 450 / 72)
                pix = page.get_pixmap(matrix=mat)
                
                img_data = pix.tobytes("png")
                img = Image.open(io.BytesIO(img_data))
                
                pix = None
                
                # PREPROCESSING
                img = self._preprocess_image(img)
                
                # OCR: PSM 6 (uniform block of text), БЕЗ whitelist
                custom_config = r'--oem 3 --psm 6'
                text = pytesseract.image_to_string(img, lang=self.language, config=custom_config)
                
                # POSTPROCESSING
                text = OCRPostProcessor.clean_toc_text(text)
                
                page_texts.append(text)
                img.close()
                
                processed += 1
                if progress_callback:
                    progress_callback.update(
                        processed, len(pages_to_process),
                        f"OCR: страница {page_num + 1} из {total_pages}",
                    )
                if (page_num + 1) % 10 == 0:
                    logger.info(f"OCR: обработано {processed}/{len(pages_to_process)} страниц")
        
        finally:
            doc.close()
        
        logger.info(f"OCR: завершено, извлечено {len(page_texts)} страниц")
        return page_texts
    
    def _preprocess_image(self, img: Image) -> Image:
        """
        Улучшает изображение для лучшего OCR.
        Мягкий preprocessing без агрессивной бинаризации.
        """
        # 1. Grayscale
        img = img.convert('L')
        
        # 2. Увеличение контраста (умеренное)
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(1.5)
        
        # 3. Увеличение резкости
        img = img.filter(ImageFilter.SHARPEN)
        
        return img