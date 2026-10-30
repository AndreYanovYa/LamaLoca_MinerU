# src/ingestion/pix2text_parser.py
"""Pix2Text-парсер PDF — альтернативный движок инжеста математики.

Назначение (запрос 2026-10-30, «сырая математика в прозе»): MinerU отлично
даёт структуру и блочные формулы ($$..$$), но инлайн-математика в прозе
иногда проходит как голый текст ("P n^(k)(x 0) = f^(k)(x 0)"). Pix2Text
(P2T) имеет собственный конвейер Layout + OCR + MFD/MFR и умеет оборачивать
инлайн-формулы прямо в потоке текста.

Архитектурное решение: НЕ менять BaseParser/ParsedDocument (стабильные
контракты всего пайплайна — чанкинг, ChromaDB, UI). Вместо этого класс
наследуется от MinerUParser и подменяет только метод _run_mineru: вместо
MinerU он вызывает P2T и возвращает markdown в том же формате, после чего
весь остальной код (_markdown_to_blocks, fallback на PyMuPDF+pix2tex,
page_number'ация, метаданные) работает без изменений.

Включение: config.yaml -> pdf_engine.engine: "pix2text"
(или env LAVALOCA_PDF_ENGINE=pix2text — см. settings/document_router).

Требования: pip install "pix2text[all]" (torch уже стоит с CUDA — P2T
подхватит GPU автоматически; модели скачиваются при первом запуске).
"""
from __future__ import annotations

import logging
import shutil
import tempfile
from pathlib import Path

from .mineru_parser import MinerUParser

logger = logging.getLogger(__name__)


class Pix2TextParser(MinerUParser):
    """PDF-движок на базе Pix2Text, встроенный в MinerU-контракт.

    Языки OCR: ru -> cnocr-код "ch_sim" (Кириллица в P2T поддерживается только
    им; "en" покрывает латиницу формул). Остальные коды проходят как есть.

    Возвращает ParsedDocument ровно той же структуры, что и MinerU: блоки
    text/heading/formula/table с LaTeX в $...$ / $$...$$, извлечённые через
    _markdown_to_blocks(). При недоступности пакета/ошибке P2T поведение
    совпадает с engine=auto: ошибка логируется, роутер уходит в fallback
    PyMuPDF+pix2tex (если pdf_engine != "mineru").
    """

    # ------------------------------------------------------------------ API
    @staticmethod
    def is_available() -> bool:
        """Тихая проверка: пакет pix2text импортируется?"""
        try:
            import pix2text  # noqa: F401
            return True
        except Exception as e:  # noqa: BLE001
            logger.debug(f"Pix2Text недоступна: {e}")
            return False

    def __init__(self, ocr_processor=None, languages: str = "ru,en",
                 device: str | None = None, **kwargs):
        super().__init__(ocr_processor=ocr_processor, languages=languages,
                         **kwargs)
        self._device = device or self._detect_device()
        self._p2t = None  # ленивая тяжёлая инициализация (модели ~ГБ)

    @staticmethod
    def _detect_device() -> str:
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
        except Exception:  # noqa: BLE001
            pass
        return "cpu"

    # ------------------------------------------------------------- основной путь
    def _run_mineru(self, pdf_path: Path, work_dir: Path):
        """Переопределение «ядра» MinerUParser: вместо MinerU — Pix2Text.

        Contract MinerUParser (строка ~234 в mineru_parser.py):
        вернуть (markdown_text, images_dir | None). Всё остальное
        (_markdown_to_blocks, метаданные, fallback PyMuPDF+pix2tex при
        pdf_engine=auto) родитель обрабатывает сам — без изменений.

        Проверено на API pix2text 1.1.7:
          Pix2Text.from_config(total_configs={...}, device=...) — единственный
          публичный конструктор; языки задаются ТОЛЬКО через total_configs
          ['text_formula']['languages'] (ключ 'langs' в пакете отсутствует);
          recognize_pdf() возвращает Document, markdown — через
          doc.to_markdown(out_dir) (сохраняет output.md и возвращает строку).
        """
        from pix2text import Pix2Text

        if self._p2t is None:
            _P2T_LANG_MAP = {"ru": "ch_sim", "rus": "ch_sim"}  # cnocr: кириллица = ch_sim
            raw = [s.strip().lower()
                   for s in str(self.languages or "ru,en").replace(";", ",").split(",")
                   if s.strip()]
            langs = tuple(dict.fromkeys(_P2T_LANG_MAP.get(x, x) for x in raw)) or ("en",)
            logger.info(
                f"Pix2Text: инициализация движка device={self._device}, "
                f"languages={list(langs)} (первый запуск скачивает модели P2T)")
            try:
                self._p2t = Pix2Text.from_config(
                    total_configs={"text_formula": {"languages": langs}},
                    enable_formula=True,   # MFD + MFR: инлайн-формулы → $..$
                    enable_table=True,     # таблицы → markdown-таблицы (как MinerU)
                    device=self._device,
                )
            except TypeError:
                # Очень старые/форк-сборки: пробуем минимум аргументов
                self._p2t = Pix2Text.from_config(device=self._device)

        doc = self._p2t.recognize_pdf(str(pdf_path))
        out_dir = Path(tempfile.mkdtemp(prefix="lamaloca_p2t_", dir=str(work_dir)))
        text = ""
        try:
            md_val = doc.to_markdown(str(out_dir / "output"))
            if isinstance(md_val, str):
                text = md_val
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Pix2Text: to_markdown не отработал ({e}) — читаю output.md с диска")
        if not text.strip():
            cand = sorted(out_dir.rglob("*.md"), key=lambda p: p.stat().st_size, reverse=True)
            if cand:
                try:
                    text = cand[0].read_text(encoding="utf-8", errors="replace")
                except Exception:  # noqa: BLE001
                    pass
        if not text.strip():
            raise RuntimeError("Pix2Text вернула пустой markdown")

        # Изображения P2T сохраняет рядом с output.md; передаём каталог родителю,
        # если он непустой (иначе None — родитель это терпит).
        imgs_dir = out_dir / "output"
        has_imgs = imgs_dir.is_dir() and any(imgs_dir.rglob("*.*"))
        logger.info(f"Pix2Text: получено {len(text)} симв. markdown")
        return text, (imgs_dir if has_imgs else None)

    # helpers родителя (_count_pages, _markdown_to_blocks, fallback) — наследуются
