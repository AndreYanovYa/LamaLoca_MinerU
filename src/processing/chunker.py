# src/processing/chunker.py
"""
Модуль для разбиения текстовых блоков на чанки для векторизации.

Чанкинг выполняется по границам блоков с перекрытием (overlap) по границам
слов. Параметры берутся из ChunkingConfig (target_tokens / overlap_tokens /
max_tokens), где 1 токен оценивается как ~4 символа.
"""
import re
from typing import List, Dict, Any, Optional

from loguru import logger

try:
    from src.ingestion.models import DocumentBlock, BlockType
except ImportError:
    from ..ingestion.models import DocumentBlock, BlockType

from src.processing.text_cleaner import (
    balance_math_delimiters,
    clean_text as math_preprocess,
)

# Приблизительный пересчёт токенов в символы (для русского текста).
CHARS_PER_TOKEN = 4


class TextChunker:
    def __init__(self, config):
        """
        Инициализация чанкера.

        :param config: Объект ChunkingConfig с атрибутами
                       target_tokens, overlap_tokens, max_tokens.
        """
        self.config = config
        self.target_len = getattr(config, "target_tokens", 500) * CHARS_PER_TOKEN
        self.max_len = getattr(config, "max_tokens", 800) * CHARS_PER_TOKEN
        self.overlap_len = getattr(config, "overlap_tokens", 100) * CHARS_PER_TOKEN

    # FIX (AttributeError: 'TextChunker' object has no attribute 'math_preprocess'):
    # math_preprocess — это импортированная на уровне модуля функция
    # text_cleaner.clean_text (канон ограничителей KaTeX, OCR-шум, чётность $).
    # Обёртка-метод нужна, чтобы вызовы self.math_preprocess(...) работали как
    # задумано;staticmethod не связывает функцию с экземпляром, поэтому
    # AttributeError возникал на каждом блоке при импорте больших PDF.
    @staticmethod
    def math_preprocess(text: str) -> str:
        """Канонизация математики блока перед чанкингом (text_cleaner.clean_text)."""
        return math_preprocess(text)

    def chunk_blocks(self, blocks: List[DocumentBlock], doc_id: str = "") -> List[Dict[str, Any]]:
        """
        Разбивает блоки на чанки с перекрытием по границам слов.

        FIX (рассогласование форматов): перед чанкингом каждый блок проходит
        через math_preprocess() — канон ограничителей KaTeX ($...$/$$...$$),
        коррекция OCR-шума и гарантия чётности маркеров. Иначе в ChromaDB
        уходил сырой UnicodeMath/\\[..\\], а разорванное $$ ... $$ («открытый
        тег в одном чанке, закрывающий в другом») ронял KaTeX на фронтенде
        (throwOnError) — весь ответ отображался нечитаемым текстом.

        Генерирует глобально уникальные block_id вида "{doc_id}_blk_{N}",
        чтобы избежать конфликтов PRIMARY KEY в SQLite и ChromaDB.

        :param blocks: Список DocumentBlock (порядок сохраняется).
        :param doc_id: SHA-256 идентификатор документа.
        :return: Список словарей {"block_id", "text", "metadata"}.
        """
        if not doc_id:
            raise ValueError("chunk_blocks требует ненулевой doc_id для генерации уникальных block_id")

        chunks: List[Dict[str, Any]] = []

        current_parts: List[str] = []
        current_meta_block: Optional[DocumentBlock] = None
        current_len = 0

        def flush():
            nonlocal current_parts, current_meta_block, current_len
            if not current_parts:
                return
            # FIX (дубли абзацев в базе): PyMuPDF на двухколоночной вёрстке и
            # при доклейке блоков-продолжений может отдать один и тот же
            # абзац дважды подряд. Дедупликация уровня файлов (SHA-256)
            # такое не ловит — убираем соседние полностью совпадающие куски.
            deduped: List[str] = []
            for p in current_parts:
                norm = re.sub(r"\s+", " ", p).strip()
                if deduped and re.sub(r"\s+", " ", deduped[-1]).strip() == norm:
                    continue
                deduped.append(p)
            text = " ".join(deduped)
            # Финальная страховка: склейка частей могла породить нечётный
            # маркер $ — нейтрализуем перед отправкой в хранилище.
            text = balance_math_delimiters(text)
            if text.strip() and current_meta_block is not None:
                chunks.append(self._create_chunk_dict(
                    text=text,
                    source_block=current_meta_block,
                    doc_id=doc_id,
                    chunk_idx=len(chunks),
                ))
            current_parts = []
            current_meta_block = None
            current_len = 0

        for block in blocks:
            block_text = self.math_preprocess((block.text or "").strip())
            if not block_text:
                continue

            # Заголовки не склеиваем с предыдущим содержимым — начинаем новый чанк.
            if block.block_type == BlockType.HEADING and current_parts:
                flush()

            # Очень длинный блок режем на куски не больше target_len.
            pieces = self._split_long_text(block_text) if len(block_text) > self.target_len else [block_text]

            for piece in pieces:
                piece_len = len(piece)

                if current_parts and current_len + piece_len > self.target_len:
                    overlap_text = self._take_overlap(" ".join(current_parts))
                    flush()
                    if overlap_text:
                        current_parts = [overlap_text]
                        current_len = len(overlap_text) + 1

                # Метаданные чанка берём из первого НЕперекрывающего блока.
                if current_meta_block is None:
                    current_meta_block = block

                current_parts.append(piece)
                current_len += piece_len + 1

        flush()

        logger.info(f"Чанкинг завершён: {len(blocks)} блоков → {len(chunks)} чанков")
        return chunks

    # ------------------------------------------------------------------
    # Вспомогательные методы
    # ------------------------------------------------------------------

    def _split_long_text(self, text: str) -> List[str]:
        """Режет слишком длинный текст на куски <= target_len по границам слов.

        FIX (разрыв разметки при чанкинге): $$...$$-блоки считаются АТОМИРОВАНЫМИ
        кусками — резать их посередине запрещено. Иначе в ChromaDB уходил
        открытый «$$» в одном чанке и закрывающий в другом: контекст ломался,
        LLM получала огрызок математики, а KaTeX на фронтенде падал с
        throwOnError. Если одиночный блок длиннее target_len — он остаётся
        целым (до max_len включительно; сверх max_len допускаем вынужденный
        перерез, но маркеры пар всё равно не разрываются).
        """
        pieces: List[str] = []
        buf: List[str] = []
        buf_len = 0

        def flush_buf():
            nonlocal buf, buf_len
            if buf:
                pieces.append(" ".join(buf))
                buf, buf_len = [], 0

        # Токенизируем: обычные слова + неразрывные $$-блоки.
        tokens = re.split(r"(\$\$.+?\$\$)", text, flags=re.DOTALL)
        for tok in tokens:
            if not tok:
                continue
            is_block = tok.startswith("$$") and tok.endswith("$$") and len(tok) > 4
            if is_block or len(tok) <= self.target_len:
                # Атомарный блок или короткое слово — обычная упаковка.
                piece_limit = self.max_len if is_block else self.target_len
                if buf and buf_len + len(tok) + 1 > piece_limit:
                    flush_buf()
                buf.append(tok)
                buf_len += len(tok) + 1
            else:
                # Длинный не-формульный текст: режем по словам, не трогая блоки.
                flush_buf()
                words = tok.split()
                sub: List[str] = []
                sub_len = 0
                for w in words:
                    if sub and sub_len + len(w) + 1 > self.target_len:
                        pieces.append(" ".join(sub))
                        sub, sub_len = [], 0
                    sub.append(w)
                    sub_len += len(w) + 1
                if sub:
                    buf, buf_len = sub, sub_len
        flush_buf()
        return pieces

    def _take_overlap(self, text: str) -> str:
        """Возвращает последние overlap_len символов, обрезая по границе слова.

        FIX: перекрытие НИКОГДА не начинается/не заканчивается внутри $$...$$
        или $...$ — хвост переносится целиком только если все маркеры в нём
        парные; иначе перекрытие усекается до безопасной границы.
        """
        if self.overlap_len <= 0 or len(text) <= self.overlap_len:
            tail = text
        else:
            cut = len(text) - self.overlap_len
            space = text.find(" ", cut)
            if space != -1:
                cut = space + 1
            tail = text[cut:]
        # Усекаем до зоны с чётными маркерами (иначе перенесём «огрызок» $$).
        while tail and not self._balanced_math(tail):
            nxt = tail.find("$")
            if nxt == -1:
                break
            tail = tail[nxt + 1:]
        return tail

    @staticmethod
    def _balanced_math(text: str) -> bool:
        """Чётность маркеров $$ и одиночных $ в фрагменте."""
        n_blocks = len(re.findall(r"\$\$", text))
        if n_blocks % 2:
            return False
        n_inline = len(re.findall(r"(?<!\\)\$", text.replace("$$", "")))
        return n_inline % 2 == 0

    def _create_chunk_dict(
        self,
        text: str,
        source_block: DocumentBlock,
        doc_id: str,
        chunk_idx: int,
    ) -> Dict[str, Any]:
        """Создаёт словарь чанка с корректными метаданными."""
        block_type = source_block.block_type
        if hasattr(block_type, "value"):  # Enum -> str ("text", "heading", ...)
            block_type = block_type.value

        return {
            "block_id": f"{doc_id}_blk_{chunk_idx}",
            "text": text,
            "metadata": {
                "block_type": block_type,
                "page_number": source_block.page_number,
                "chapter": source_block.chapter or "",
                "section": source_block.section or "",
                "coordinates": source_block.coordinates,
                "extraction_method": source_block.extraction_method or "unknown",
                "doc_id": doc_id,
            },
        }
