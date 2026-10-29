# src/processing/text_quality.py
"""Эвристики для оценки качества извлечённого текста."""
import re
from typing import List


def assess_text_quality(text: str) -> float:
    """
    Оценивает качество текста от 0.0 (мусор) до 1.0 (чистый текст).
    
    Критерии:
    - Доля кириллицы/латиницы
    - Отсутствие "мусорных" символов
    - Наличие слов разумной длины
    """
    if not text or len(text.strip()) < 10:
        return 0.0
    
    total_chars = len(text)
    
    # Полезные символы: буквы (кириллица + латиница), цифры, базовая пунктуация, пробелы
    useful_pattern = re.compile(r'[а-яА-ЯёЁa-zA-Z0-9\s.,;:!?—–\-()\[\]«»"\'/\\]')
    useful_chars = len(useful_pattern.findall(text))
    
    # Доля полезных символов
    useful_ratio = useful_chars / total_chars
    
    # "Мусорные" паттерны: длинные последовательности заглавных латинских букв без смысла
    garbage_patterns = [
        re.compile(r'[A-Z]{5,}'),  # 5+ заглавных латинских подряд
        re.compile(r'[^\x00-\x7Fа-яА-ЯёЁ\s]{3,}'),  # 3+ странных символов
    ]
    
    garbage_count = 0
    for pattern in garbage_patterns:
        garbage_count += len(pattern.findall(text))
    
    garbage_ratio = garbage_count / max(total_chars / 10, 1)
    
    # Итоговая оценка
    quality = useful_ratio - garbage_ratio
    return max(0.0, min(1.0, quality))


def assess_document_quality(page_texts: List[str], sample_pages: int = 5) -> float:
    """Оценивает качество текста по первым N страницам."""
    if not page_texts:
        return 0.0
    
    sample = page_texts[:sample_pages]
    scores = [assess_text_quality(t) for t in sample]
    return sum(scores) / len(scores)


# Порог: если качество ниже — нужен OCR
QUALITY_THRESHOLD = 0.6