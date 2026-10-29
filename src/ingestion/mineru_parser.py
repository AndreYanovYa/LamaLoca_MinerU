# src/ingestion/mineru_parser.py
"""Парсер PDF на базе MinerU (OpenDataLab) — основной движок извлечения.

ПОЧЕМУ ЭТОТ ПАРСЕР ЗАМЕНИЛ «сырой» PyMuPDF по текстовому слою:
PyMuPDF (как и любой стандартный текстовый экстрактор PDF) извлекает сложную
математику плохо или некорректно. Инлайн-формулы в вёрстке учебников живут в
математических шрифтах (Symbol/CMMI/CMSY...), чей текстовый слой даёт кашу
вида ``quad x ->x 0`` вместо ``\\sum_{k=0}^n a_k(x-x_0)^k``; многоколоночная
вёрстка и выноски дублируют/рвут блоки. Никакая постобработка этой каши
спасает — нужен анализ СТРАНИЦЫ КАК КАРТИНКИ.

MinerU решает это нативно: модель разметки страницы (layout-детектор +
Formula Detection/Recognition, Unimernet) находит формулы прямо на отрендеренных
страницах и возвращает готовый Markdown с валидным LaTeX: инлайн-формулы в
``$ ... $``, отображённые — в блоках ``$$ ... $$``. Дробная/индексная структура
при этом сохраняется точно.

Интеграция:
* Вызов через Python API ``mineru.cli.common.do_parse`` (backend="pipeline",
  formula_enable=True). Если пакет установлен, но API другой версии —
  автоматически используется CLI ``mineru -p ... -o ...``.
* Все страницы обрабатываются всегда («математика — точная наука»): никаких
  выборок страниц, диапазон = весь документ.
* Изображения/зоны, которые MinerU сохранил как PNG-файлы, дополнительно
  прогоняются через штатный Image-to-LaTeX стек (page_renderer не нужен —
  картинка уже вырезана; formula_recognizer → pix2tex даёт LaTeX для них).
* Markdown разбирается на DocumentBlock'и (структурные уровни ТЗ) и уходит в
  общий пайплайн (чанки → ChromaDB → RAG).

Деградация: если MinerU недоступен (не установлен / нет моделей / ошибка),
исключение поднимается наружу — DocumentRouter переключится на PyMuPDF-парсер
(он остаётся в проекте как fallback и источник координат зон).
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from loguru import logger

from .base_parser import BaseParser
from .models import ParsedDocument, DocumentBlock, BlockType
from ..core.exceptions import DocumentParseError


class MinerUParser(BaseParser):
    """PDF → Markdown (с LaTeX-формулами) через MinerU, затем → DocumentBlock'и."""

    def __init__(self, ocr_processor=None, pdf_engine: str = "auto", languages: str = "ru,en"):
        self.ocr_processor = ocr_processor
        self.pdf_engine = (pdf_engine or "auto").lower()
        self.languages = languages

    @staticmethod
    def is_available() -> bool:
        """MinerU установлена (или хотя бы её CLI в PATH)?"""
        try:
            import mineru  # noqa: F401
            return True
        except ImportError:
            return shutil.which("mineru") is not None or shutil.which("mineru.exe") is not None

    def parse(self, file_path: Path, doc_id: str, progress_callback=None) -> ParsedDocument:
        """Точка входа DocumentRouter для PDF.

        engine=auto    → MinerU как основной; при недоступности/ошибке — fallback
                         на PyMuPDF+pix2tex (импортируется лениво, циклов нет).
        engine=mineru  → только MinerU (ошибка поднимается наружу).
        engine=pymupdf → только старый пайплайн.
        """
        if self.pdf_engine == "pymupdf":
            return self._fallback_parse(file_path, doc_id, progress_callback)
        try:
            return self._parse_mineru(Path(file_path), doc_id, progress_callback)
        except Exception as e:  # noqa: BLE001
            if self.pdf_engine == "mineru":
                raise
            logger.warning(
                f"MinerU недоступна/ошибка ({type(e).__name__}: {str(e)[:200]}); "
                f"fallback на PyMuPDF+pix2tex пайплайн")
            return self._fallback_parse(file_path, doc_id, progress_callback)

    def _fallback_parse(self, file_path: Path, doc_id: str, progress_callback=None) -> ParsedDocument:
        from .pdf_parser import PDFParser
        return PDFParser(ocr_processor=self.ocr_processor)._parse_pymupdf_fallback(
            Path(file_path), doc_id, progress_callback=progress_callback)

    # ------------------------------------------------------------------ API
    def supported_extensions(self) -> list[str]:
        return [".pdf"]

    def _parse_mineru(self, file_path: Path, doc_id: str, progress_callback=None) -> ParsedDocument:
        file_path = Path(file_path)
        logger.info(f"MinerU-парсинг PDF: {file_path.name}")

        work_dir = Path(tempfile.mkdtemp(prefix="lamaloca_mineru_"))
        try:
            if progress_callback:
                progress_callback.update(0, 1, "MinerU: разметка страниц и распознавание формул...")
            md_text, images_dir = self._run_mineru(file_path, work_dir)
            if not md_text.strip():
                raise DocumentParseError("MinerU вернула пустой Markdown")

            if progress_callback:
                progress_callback.update(1, 1, "MinerU: сборка блоков документа...")
            blocks = self._markdown_to_blocks(md_text, doc_id, images_dir)
            if not blocks:
                raise DocumentParseError("В Markdown MinerU не найдено ни одного текстового блока")

            # FIX (кликабельные источники → страница оригинала): привязываем
            # блоки к номерам страниц из middle.json / PyMuPDF.
            try:
                self._attach_page_numbers(blocks, file_path, work_dir)
            except Exception as pe:  # noqa: BLE001 — метаданные не валят импорт
                logger.warning(f"MinerU: привязка страниц не удалась ({pe})")

            total_pages = self._count_pages(file_path)
            metadata = self._read_metadata(file_path)
            title = metadata.get("title") or file_path.stem
            author = metadata.get("author")

            formulas = sum(1 for b in blocks if b.block_type == BlockType.FORMULA)
            logger.info(
                f"MinerU: {len(blocks)} блоков, {formulas} формульных, "
                f"{total_pages} стр., изображения: {images_dir}"
            )
            return ParsedDocument(
                doc_id=doc_id,
                source_path=str(file_path.resolve()),
                original_filename=file_path.name,
                file_format="pdf",
                title=title,
                author=author,
                language="ru",
                total_pages=total_pages,
                blocks=blocks,
            )
        finally:
            shutil.rmtree(work_dir, ignore_errors=True)

    # ------------------------------------------------------------ запуск MinerU
    @staticmethod
    def _count_pages(pdf_path: Path) -> int:
        try:
            import pymupdf
            with pymupdf.open(str(pdf_path)) as doc:
                return doc.page_count
        except Exception:  # noqa: BLE001
            return 0

    @staticmethod
    def _read_metadata(pdf_path: Path) -> dict:
        try:
            import pymupdf
            with pymupdf.open(str(pdf_path)) as doc:
                return doc.metadata or {}
        except Exception:  # noqa: BLE001
            return {}

    def _run_mineru(self, pdf_path: Path, work_dir: Path) -> tuple[str, Path | None]:
        """Возвращает (markdown, каталог извлечённых изображений)."""
        out_dir = work_dir / "out"
        out_dir.mkdir(parents=True, exist_ok=True)
        try:
            try:
                self._mineru_python_api(pdf_path, out_dir)
            except ImportError:
                # Версии MinerU без привычного Python-API (или другая раскладка
                # модулей) — идём через стабильный CLI.
                logger.debug("MinerU: Python-API недоступен, использую CLI")
                self._mineru_cli(pdf_path, out_dir)
        except DocumentParseError:
            raise
        except Exception as e:  # noqa: BLE001
            raise DocumentParseError(f"MinerU не смогла обработать PDF: {e}") from e

        md_path = self._find_markdown(out_dir)
        if md_path is None:
            raise DocumentParseError(f"MinerU не создала Markdown в {out_dir}")
        md_text = md_path.read_text(encoding="utf-8", errors="replace")
        images_dir = self._find_images_dir(out_dir)
        return md_text, images_dir

    # Сопоставление языков конфига Lama Loca с языковым кодом OCR-ветки MinerU.
    # MinerU принимает ОДИН язык (click.Choice): ch/en/korean/.../cyrillic/east_slavic.
    # Для русско-английских учебников cyrillic покрывает основной текст лучше всех.
    _LANG_MAP = (
        (("ru", "uk", "be", "bg", "sr", "mk", "mn"), "cyrillic"),
        (("el",), "el"),
        (("ja",), "japan"),
        (("ko",), "korean"),
        (("th",), "th"),
        (("ar",), "arabic"),
        (("hi",), "devanagari"),
        (("zh", "cn", "ch"), "ch"),
        (("en",), "en"),
    )

    def _pick_mineru_lang(self) -> str:
        raw = (self.languages or "ru,en").lower()
        codes = [c.strip() for c in re.split(r"[,;|\s]+", raw) if c.strip()]
        for prefixes, mineru_lang in self._LANG_MAP:
            if any(c.startswith(p) for c in codes for p in prefixes):
                return mineru_lang
        return "en"

    def _mineru_python_api(self, pdf_path: Path, out_dir: Path) -> None:
        """Синхронный вызов mineru.cli.common.do_parse (backend=pipeline)."""
        from mineru.cli.common import do_parse, read_fn  # ImportError → наружу

        try:
            from mineru.cli.common import MakeMode
            md_mode = MakeMode.MM_MD
        except Exception:  # noqa: BLE001 — в старых версиях MakeMode нет
            md_mode = "mm"

        pdf_bytes = read_fn(pdf_path)
        kwargs = {}
        try:  # v3.x: вывод только markdown (без лишних dump'ов), если поддержано
            import inspect
            params = set(inspect.signature(do_parse).parameters)
            if "f_dump_md_only" in params:
                # FIX (переход к странице из чата): f_dump_md_only вырезал бы
                # middle.json — единственный источник номеров страниц блоков.
                # Жёстко не ставим True; ниже явно просим f_dump_middle_json.
                pass
        except Exception:  # noqa: BLE001
            pass
        # ВАЖНО: обходим ВСЕ страницы (start/end по умолчанию = весь файл) —
        # «математика точная наука», частичный разбор запрещён.
        do_parse(
            output_dir=str(out_dir),
            pdf_file_names=[pdf_path.stem],
            pdf_bytes_list=[pdf_bytes],
            p_lang_list=[self._pick_mineru_lang()],  # один язык, валидный для MinerU
            backend="pipeline",     # локальный layout+MFD+MFR стек (Unimernet)
            parse_method="auto",    # auto end2end: колонки/выноски/таблицы
            formula_enable=True,    # распознавание формул → LaTeX в markdown
            table_enable=True,
            f_draw_layout_bbox=False,
            f_draw_span_bbox=False,
            f_dump_md=True,
            # FIX (переход к странице из чата): middle.json содержит page_idx
            # каждого блока — единственный точный источник номеров страниц в
            # MinerU-разборе. Раньше он не выгружался → все блоки уходили в
            # SQLite/ChromaDB с page_number=None → кнопки источников в чате
            # были декоративными.
            f_dump_middle_json=True,
            f_dump_model_output=False,
            f_dump_orig_pdf=False,
            f_dump_content_list=False,
            f_make_md_mode=md_mode,  # MM_MD: markdown с картинками и $$..$$
            **kwargs,
        )

    def _mineru_cli(self, pdf_path: Path, out_dir: Path) -> None:
        exe = shutil.which("mineru") or shutil.which("mineru.exe")
        if exe is None:
            # Python-API уже поднял ImportError — CLI тоже нет: MinerU нет вовсе.
            raise DocumentParseError(
                "MinerU не установлена (pip install 'mineru[core]' + mineru-models-download)")
        # Синтаксис CLI MinerU 2.x/3.x: обязательные -p/--path и -o/--output,
        # далее -b backend, -m method, -l lang (один код из click.Choice).
        # Явный "-b pipeline" гарантирует разбор через локальный layout+MFR стек
        # (в 3.x дефолт — hybrid-auto-engine, требующий доп. моделей/сервера).
        cmd = [
            exe,
            "-p", str(pdf_path),
            "-o", str(out_dir),
            "-b", "pipeline",
            "-m", "auto",
            "-l", self._pick_mineru_lang(),
        ]
        env = dict(os.environ)
        env.setdefault("MINERU_DEVICE_MODE", "cuda" if self._cuda_available() else "cpu")
        proc = subprocess.run(
            cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", env=env
        )
        if proc.returncode != 0:
            tail = (proc.stderr or proc.stdout or "").strip()[-500:]
            raise DocumentParseError(f"CLI mineru завершился с кодом {proc.returncode}: {tail}")

    @staticmethod
    def _cuda_available() -> bool:
        try:
            import torch
            return bool(torch.cuda.is_available())
        except Exception:  # noqa: BLE001
            return False

    @staticmethod
    def _find_markdown(out_dir: Path) -> Path | None:
        candidates = sorted(
            (p for p in out_dir.rglob("*.md") if p.is_file()),
            key=lambda p: p.stat().st_size, reverse=True,
        )
        return candidates[0] if candidates else None

    @staticmethod
    def _find_images_dir(out_dir: Path) -> Path | None:
        for p in out_dir.rglob("images"):
            if p.is_dir() and any(p.iterdir()):
                return p
        return None

    # ------------------------------------------------- Привязка блоков к страницам
    @staticmethod
    def _norm_for_match(s: str) -> str:
        """Нормализация текста для нестрогого сопоставления: только буквы/цифры,
        нижний регистр. LaTeX-команды ($\\frac{}{}$) при этом схлопываются в
        свои имена — достаточно для якорения прозы."""
        return re.sub(r"[^0-9a-zа-яё]+", "", (s or "").lower())

    def _attach_page_numbers(self, blocks: list, pdf_path: Path,
                             work_dir: Path) -> None:
        """Проставляет DocumentBlock.page_number (1-based) для всех блоков.

        Источники (по убыванию точности):
        1. middle.json MinerU (<out>/**/<stem>*_middle.json): page_idx + bbox
           каждого блока — сопоставление по нормализованному префиксу текста.
        2. PyMuPDF search_for(): поиск текстовых «якорей» первых слов каждого
           блока прямо в PDF (работает и для сканов с текстовым слоем).
        3. Пропорциональная интерполяция между уже привязанными соседями.

        FIX: раньше все MinerU-блоки уходили в базу с page_number=None — кнопки
        источников в чате не могли открыть нужную страницу просмотрщика.
        """
        assigned = 0

        # --- 1. middle.json --------------------------------------------------
        page_map: dict[str, int] = {}   # norm-prefix -> page (1-based)
        middle = None
        try:
            cands = sorted(work_dir.rglob("*_middle.json"),
                           key=lambda p: p.stat().st_size, reverse=True)
            if cands:
                import json
                middle = json.loads(cands[0].read_text(encoding="utf-8"))
        except Exception as e:  # noqa: BLE001
            logger.debug(f"MinerU: middle.json не прочитан ({e})")

        if isinstance(middle, dict):
            for pdf_info in middle.get("pdf_info") or []:
                page_no = int(pdf_info.get("page_idx", -1)) + 1
                if page_no < 1:
                    continue
                buckets = []
                for key in ("preproc_blocks", "discarded_blocks"):
                    buckets.extend(pdf_info.get(key) or [])
                for b in buckets:
                    txt = self._block_plain_text(b)
                    norm = self._norm_for_match(txt)
                    if len(norm) >= 24:
                        page_map.setdefault(norm[:64], page_no)
                        page_map.setdefault(norm[:24], page_no)

        if page_map:
            for blk in blocks:
                if blk.page_number:
                    continue
                norm = self._norm_for_match(blk.text)
                pg = page_map.get(norm[:64]) or page_map.get(norm[:24])
                if pg:
                    blk.page_number = pg
                    assigned += 1
            logger.debug(f"MinerU: страницы по middle.json проставлены {assigned} блокам")

        # --- 2. PyMuPDF search_for по якорям ----------------------------------
        missing = [b for b in blocks if not b.page_number]
        if missing:
            try:
                import pymupdf
                doc = pymupdf.open(str(pdf_path))
            except Exception:
                doc = None
            if doc is not None:
                try:
                    n_pages = doc.page_count
                    last_page = 0
                    for blk in blocks:
                        if blk.page_number:
                            last_page = blk.page_number - 1
                            continue
                        anchor = self._anchor_phrase(blk.text)
                        if not anchor:
                            continue
                        found = None
                        scan_from = max(0, min(last_page, n_pages - 1))
                        for lo, hi in ((scan_from, min(n_pages, scan_from + 6)),
                                       (0, n_pages)):
                            for pno in range(lo, hi):
                                try:
                                    rects = doc[pno].search_for(anchor)
                                except Exception:
                                    rects = []
                                if rects:
                                    found = pno
                                    break
                            if found is not None:
                                break
                        if found is not None:
                            blk.page_number = found + 1
                            last_page = found
                            assigned += 1
                    doc.close()
                except Exception as e:  # noqa: BLE001
                    logger.debug(f"MinerU: якорный поиск страниц не завершен ({e})")
                    try:
                        doc.close()
                    except Exception:
                        pass

        # --- 3. Интерполяция оставшихся ---------------------------------------
        if any(b.page_number for b in blocks):
            prev_val = None
            prev_i = -1
            for i, blk in enumerate(blocks):
                if blk.page_number:
                    if prev_val is not None and i - prev_i > 1:
                        step = (blk.page_number - prev_val) / (i - prev_i)
                        for j in range(prev_i + 1, i):
                            blocks[j].page_number = max(1, int(round(prev_val + step * (j - prev_i))))
                    prev_val = blk.page_number
                    prev_i = i
            last_known = next((b.page_number for b in reversed(blocks)
                               if b.page_number), None)
            if last_known:
                for b in blocks:
                    if not b.page_number:
                        b.page_number = last_known

        still = sum(1 for b in blocks if not b.page_number)
        logger.info(f"MinerU: привязка страниц — {assigned}/{len(blocks)} блоков, "
                    f"{still} без страницы")

    @staticmethod
    def _block_plain_text(mineru_block: dict) -> str:
        """Собирает plain text из блока middle.json (lines→spans→content)."""
        parts: list[str] = []
        for line in mineru_block.get("lines") or []:
            for span in line.get("spans") or []:
                content = span.get("content")
                if isinstance(content, list):
                    content = "".join(str(x) for x in content)
                if content:
                    parts.append(str(content))
        return " ".join(parts)

    @staticmethod
    def _anchor_phrase(text: str) -> str:
        """Первая «живая» фраза блока (5–7 слов), пригодная для search_for."""
        # убираем markdown/LaTeX обёртки
        t = re.sub(r"\$\$?.{0,400}?\$\$?|[#*_`>\[\](){}\\]", " ", text or "")
        words = [w for w in re.split(r"\s+", t) if len(w) >= 2][:7]
        phrase = " ".join(words).strip()
        return phrase if len(phrase) >= 12 else ""


    # ------------------------------------------------- Markdown → DocumentBlock
    _HEAD_RE = re.compile(r"^(#{1,6})\s+(.*)$")
    _MD_IMAGE_RE = re.compile(r"!\[[^\]]*\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
    _HTML_IMG_RE = re.compile(r"<img[^>]+src=[\"']([^\"']+)[\"'][^>]*>", re.IGNORECASE)

    def _markdown_to_blocks(self, md_text: str, doc_id: str, images_dir: Path | None) -> list:
        """Разбор markdown MinerU на блоки. Формулы ($$..$$, $..$) сохраняются
        как есть — дальше их штатно рисует ft.Markdown/KaTeX в UI и exporter."""
        blocks: list[DocumentBlock] = []
        counter = 0
        chapter = ""

        def add(btype: BlockType, text: str, extra_method: str = "") -> None:
            nonlocal counter
            text = text.strip()
            if not text:
                return
            blocks.append(DocumentBlock(
                block_id=f"b{counter}",
                block_type=btype,
                text=text,
                page_number=None,
                chapter=chapter or None,
                coordinates=None,
                extraction_method=("mineru_markdown" + (":" + extra_method if extra_method else "")),
            ))
            counter += 1

        # 1. Блочный LaTeX $$...$$ (отдельные строки) вырезаем маркерами.
        parts = re.split(r"(?m)^\s*\$\$(.+?)\$\$\s*$", md_text, flags=re.S)
        # re.split с одной группой: [текст, формула, текст, формула, ...]
        for i, chunk in enumerate(parts):
            if i % 2 == 1:  # сама формула
                latex = self._normalize_display_math(chunk)
                if latex:
                    add(BlockType.FORMULA, f"$$ {latex} $$", "block_math")
                continue
            for line_block in self._split_paragraphs(chunk):
                # Заголовки
                m = self._HEAD_RE.match(line_block)
                if m:
                    level, title = len(m.group(1)), m.group(2).strip()
                    if level <= 2:
                        chapter = re.sub(r"[#*_`]", "", title)
                    btype = BlockType.HEADING if level <= 3 else BlockType.TEXT
                    add(btype, title, "heading" if level <= 3 else "")
                    continue
                # Чистые формульные строки вида $$...$$ внутри абзаца
                only_math = re.fullmatch(r"\s*\$\$(.+?)\$\$\s*", line_block, flags=re.S)
                if only_math:
                    latex = self._normalize_display_math(only_math.group(1))
                    if latex:
                        add(BlockType.FORMULA, f"$$ {latex} $$", "inline_block_math")
                    continue
                # Изображения: ссылки ![](images/x.png) → FIGURE-блок + pix2tex
                line_block = self._rewrite_image_paths(line_block, images_dir)
                img_links = self._MD_IMAGE_RE.findall(line_block) + self._HTML_IMG_RE.findall(line_block)
                prose = self._MD_IMAGE_RE.sub("", line_block)
                prose = self._HTML_IMG_RE.sub("", prose).strip()
                if img_links:
                    for link in img_links:
                        png = self._resolve_image(link, images_dir)
                        figure_text = ""
                        method = "figure_png"
                        if png is not None:
                            latex = self._pix2tex_for(png)
                            if latex:
                                figure_text = f"$$ {self._normalize_display_math(latex)} $$"
                                method = "figure_png+formula_latex"
                        add(BlockType.FIGURE, figure_text or f"![изображение]({link})", method)
                if prose:
                    self._add_prose(add, prose)
        return blocks

    def _add_prose(self, add, prose: str) -> None:
        """Проза: маркированные списки режем по пунктам, остальное — TEXT.

        Таблицы (строки с |...|) сохраняем как TABLE целиком — чанкер и RAG
        получают их единым куском.
        """
        lines = [l.rstrip() for l in prose.splitlines() if l.strip()]
        if not lines:
            return
        # Markdown-таблица: consecutive строки, начинающиеся с «|»
        if all(re.match(r"^\s*\|.*\|\s*$", l) for l in lines) and len(lines) >= 2:
            add(BlockType.TABLE, "\n".join(lines), "table")
            return
        item_re = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+")
        buf: list[str] = []
        for line in lines:
            if item_re.match(line):
                if buf:
                    add(BlockType.TEXT, " ".join(buf))
                    buf = []
                add(BlockType.TEXT, item_re.sub("", line), "list_item")
            else:
                buf.append(line.strip())
        if buf:
            add(BlockType.TEXT, " ".join(buf))

    @staticmethod
    def _split_paragraphs(chunk: str) -> list[str]:
        parts = [p.strip() for p in re.split(r"\n\s*\n", chunk) if p.strip()]
        return parts

    @staticmethod
    def _normalize_display_math(latex: str) -> str:
        s = (latex or "").strip()
        s = re.sub(r"^\\\[|\\\]$|^\\\[|\\\]$", "", s)
        s = s.replace("\\[", "").replace("\\]", "") if s.count("\\[") == s.count("\\]") else s
        s = re.sub(r"\s+", " ", s).strip(" $")
        return s

    def _resolve_image(self, link: str, images_dir: Path | None) -> Path | None:
        if images_dir is None:
            return None
        name = Path(link.replace("\\", "/")).name
        cand = images_dir / name
        return cand if cand.exists() else None

    def _rewrite_image_paths(self, text: str, images_dir: Path | None) -> str:
        return text

    @staticmethod
    def _pix2tex_for(png_path: Path) -> str | None:
        try:
            from ..processing.formula_recognizer import get_formula_recognizer
            rec = get_formula_recognizer()
            if rec is None:
                return None
            return rec.recognize(png_path)
        except Exception as e:  # noqa: BLE001
            logger.debug(f"pix2tex для {png_path.name} недоступна: {e}")
            return None
