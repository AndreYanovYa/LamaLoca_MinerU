# src/processing/text_corrector.py
"""Универсальный корректор текста после OCR.

Работает без внешних зависимостей (кроме LLM опционально).
Использует эвристики и регулярные выражения для очистки любого текста.

ТЗ раздел 1: перед эвристиками оглавления применяется препроцессор
``src.processing.text_cleaner`` — автоматическая коррекция шума Tesseract
на стыке распознавания окрестностей (ложная ``X \\setminus O(x_0)``
заменяется на проколотую окрестность ``x \\in \\dot{O}(x_0)``), чтобы
LLM не переворачивала смысл определения предела по Коши.
"""
import re

from .text_cleaner import clean_text as _clean_math_noise


class TextCorrector:
    """Универсальный корректор текста после OCR."""
    
    def __init__(self, use_llm: bool = False, llm_engine=None):
        self.use_llm = use_llm
        self.llm = llm_engine
    
    def correct_toc(self, text: str) -> str:
        """
        Корректирует текст оглавления.
        
        Этапы:
        0. Пре-процессор математики (text_cleaner): коррекция OCR-шума
           проколотых окрестностей и изоляция плоских math-глифов
        1. Нормализация символов
        2. Исправление знаков параграфов и страниц
        3. Удаление мусора между названием и номером
        4. Восстановление разорванных слов
        5. (Опционально) LLM-коррекция
        """
        # Этап 0: математический препроцессор ТЗ п.1 (ложная разность
        # множеств -> проколотая окрестность \dot{O}(x_0)). Работает до
        # _normalize_chars, который целенаправленно меняет регистр/латиницу.
        text = _clean_math_noise(text)

        lines = text.split('\n')
        corrected = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Этап 1: Нормализация символов
            line = self._normalize_chars(line)
            
            # Этап 2: Исправление знаков параграфов и страниц
            line = self._fix_section_numbers(line)
            
            # Этап 3: Удаление мусора между названием и номером
            line = self._clean_before_page_number(line)
            
            # Этап 4: Восстановление разорванных слов
            line = self._fix_broken_words(line)
            
            # Финальная очистка пробелов
            line = re.sub(r'\s+', ' ', line).strip()
            
            if line and len(line) > 2:
                corrected.append(line)
        
        result = '\n'.join(corrected)
        
        # Этап 5: LLM-коррекция (опционально)
        if self.use_llm and self.llm:
            result = self._llm_correct(result)
        
        return result
    
    def _normalize_chars(self, text: str) -> str:
        """Заменяет похожие символы и удаляет артефакты."""
        # Латиница → Кириллица (похожие по начертанию)
        replacements = {
            'c': 'с', 'C': 'С',
            'o': 'о', 'O': 'О',
            'a': 'а', 'A': 'А',
            'e': 'е', 'E': 'Е',
            'p': 'р', 'P': 'Р',
            'x': 'х', 'X': 'Х',
            'y': 'у', 'Y': 'У',
            'H': 'Н',
            'K': 'К',
            'M': 'М',
            'T': 'Т',
            'B': 'В',
        }
        
        result = text
        for wrong, correct in replacements.items():
            result = result.replace(wrong, correct)
        
        # Удаляем одиночные символы-артефакты
        result = re.sub(r'[°`„"~@#|]', '', result)
        
        return result
    
    def _fix_section_numbers(self, text: str) -> str:
        """Исправляет знаки параграфов и номера страниц."""
        # $ → §
        text = re.sub(r'^\$\s*(\d+)', r'§ \1', text)
        text = re.sub(r'^\$\.?\s*(\d+)', r'§ \1', text)
        
        # % → * (звёздочка у параграфа)
        text = re.sub(r'(\d+\.\d+)%', r'\1*', text)
        text = re.sub(r'(\d+\.\d+\*)\s*%', r'\1', text)
        
        # Исправляем номера страниц > 750 (для Кудрявцева)
        page_match = re.search(r'(\d{3,4})\s*$', text)
        if page_match:
            page_num = int(page_match.group(1))
            if page_num > 750:
                corrected_page = page_num // 10
                text = text[:page_match.start()] + str(corrected_page) + text[page_match.end():]
        
        return text
    
    def _clean_before_page_number(self, text: str) -> str:
        """Удаляет мусор между названием и номером страницы."""
        # Ищем номер страницы в конце
        page_match = re.search(r'(\d{1,4})\s*$', text)
        
        if not page_match:
            return text
        
        page_number = page_match.group(1)
        before_page = text[:page_match.start()].strip()
        
        if not before_page:
            return text
        
        # Разбиваем на токены
        tokens = before_page.split()
        
        if not tokens:
            return text
        
        # Фильтруем: оставляем только "нормальные" токены
        clean_tokens = []
        for token in tokens:
            clean = re.sub(r'[^\wа-яА-ЯёЁ]', '', token)
            
            # Критерии:
            # 1. Длина > 2 символов
            # 2. Содержит хотя бы одну букву
            # 3. Или это номер параграфа (1, 1.3, 1.3*)
            if len(clean) > 2 and re.search(r'[а-яА-ЯёЁa-zA-Z]', clean):
                clean_tokens.append(token)
            elif re.match(r'^\d+[\.\*]?$', clean):
                clean_tokens.append(token)
        
        if not clean_tokens:
            return text
        
        cleaned_text = ' '.join(clean_tokens)
        return f"{cleaned_text} . . . . . . {page_number}"
    
    def _fix_broken_words(self, text: str) -> str:
        """Восстанавливает слова, разорванные переносом."""
        # "Последователь- ности" → "Последовательности"
        text = re.sub(r'(\w+)-\s+(\w+)', r'\1\2', text)
        text = re.sub(r'(\w+)-\s*([А-Яа-я]{3,})', r'\1\2', text)
        
        return text
    
    def _llm_correct(self, text: str) -> str:
        """LLM-коррекция для сложных случаев (опционально)."""
        if not self.llm:
            return text
        
        prompt = f"""Ты — корректор текста после OCR. Исправь ошибки распознавания в оглавлении книги.

ПРАВИЛА:
1. Сохрани структуру оглавления (номера параграфов, точки, номера страниц)
2. Исправь опечатки в словах
3. Восстанови разорванные слова
4. Замени латинские буквы на похожие кириллические
5. НЕ добавляй и НЕ удаляй строки
6. Верни только исправленный текст, без комментариев

Текст для исправления:
{text}

Исправленный текст:"""
        
        try:
            messages = [
                {"role": "system", "content": "Ты — корректор текста после OCR."},
                {"role": "user", "content": prompt}
            ]
            corrected = self.llm.chat(messages)
            return corrected.strip()
        except Exception as e:
            print(f"LLM-коррекция не удалась: {e}")
            return text