# src/processing/ocr_postprocessor.py
"""Постобработка текста после OCR с использованием универсального корректора."""
from .text_corrector import TextCorrector


class OCRPostProcessor:
    """Очищает текст после OCR."""
    
    _corrector = None
    
    @classmethod
    def _get_corrector(cls) -> TextCorrector:
        if cls._corrector is None:
            cls._corrector = TextCorrector(use_llm=False)  # LLM можно включить позже
        return cls._corrector
    
    @staticmethod
    def clean_toc_text(text: str) -> str:
        """Очищает текст оглавления после OCR."""
        corrector = OCRPostProcessor._get_corrector()
        return corrector.correct_toc(text)