# src/ingestion/pdf_parser.py
"""Адаптивный парсер PDF с полным циклом «текст → PNG → pix2tex → LaTeX».

Пайплайн (требование ТЗ: PyMuPDF не просто забирает текстовый слой, а
взаимодействует с page_renderer.py и formula_recognizer.py):

1. Для КАЖДОЙ страницы извлекаются блоки через page.get_text(\"blocks\")
   РАЗДЕЛЬНО по span'ам. Если span набран математическим шрифтом
   (Symbol / Math / CMMI / CMSY / MSAM / MSXM / Euclid / MT ...) или его
   текст содержит некорректные символы (каша из замещённых глифов) —
   зона этого блока помечается как формульная.
2. Помеченная зона рендерится в PNG 300 dpi через page_renderer.render_zone_png.
3. PNG прогоняется через formula_recognizer (pix2tex / texify), валидный
   LaTeX вставляется в текст блока как $$ ... $$ — и попадает в чанки,
   эмбеддинги и RAG-контекст.
4. Экономия без потери точности: если Unicode-текстовый слой блока
   полностью конвертируется правилами в LaTeX (formula_unicode, Фаза 0),
   инференс сети для этой зоны не нужен; при низком качестве текста всей
   страницы («математика — точная наука») страница рендерится целиком и
   каждый её блок уходит в pix2tex.
"""
import re
from pathlib import Path
from loguru import logger
import pymupdf

from .base_parser import BaseParser
from .models import ParsedDocument, DocumentBlock, BlockType
from ..core.exceptions import DocumentParseError
from ..processing.ocr_processor import OCRProcessor
from ..processing.text_quality import (
    assess_text_quality,
    QUALITY_THRESHOLD,
)


class PDFParser(BaseParser):
    """Адаптивный парсер PDF: использует TOC, текст или OCR в зависимости от качества.

    ВНИМАНИЕ (смена архитектуры): основным движком извлечения PDF теперь является
    MinerUParser (src/ingestion/mineru_parser.py) — PyMuPDF как текстовый слой не
    справляется со сложной математикой (инлайн-формулы в Symbol/Math шрифтах
    превращаются в кашу «quad x ->x 0», многоколоночная вёрстка рвёт блоки).
    MinerU (OpenDataLab) анализирует страницы как картинки (layout + формульный
    детектор/распознаватель) и отдаёт Markdown с валидным LaTeX.
    Этот класс остаётся ПОЛНОЦЕННЫМ FALLBACK'ом: DocumentRouter вызывает его,
    если MinerU недоступна или упала. Пайплайн zone→PNG→pix2tex здесь сохранён.
    """

    # Математические шрифты в PDF-вёрстке учебников (Symbol/Math/CMS...).
    # Span с таким font — почти всегда формула: текстовый слой у этих
    # шрифтов кодует глифы нестандартно (quad x ->x 0 и т.п.), поэтому
    # зона обязана идти через рендер + pix2tex.
    _MATH_FONT_RE = re.compile(
        r"(math|symbol|cmmi|cmsy|cmex|cmr|msam|msxm|euclid|"
        r"mt[^a-z]|wasy|stmary|mathematical)",
        re.IGNORECASE,
    )

    def __init__(self, ocr_processor: OCRProcessor = None, pdf_engine: str = "auto",
                 languages: str = "ru,en", vram_gb_override: int = 0,
                 fast_batch: bool = True):
        self.ocr_processor = ocr_processor
        self.pdf_engine = (pdf_engine or "auto").lower()
        self.languages = languages or "ru,en"
        # FIX (лог 22:10): прокидываем настройки ускорения MinerU дальше, в
        # MinerUParser (см. PdfEngineConfig.vram_gb_override / fast_batch).
        self.vram_gb_override = int(vram_gb_override or 0)
        self.fast_batch = bool(fast_batch)

    def parse(self, file_path: Path, doc_id: str, progress_callback=None) -> ParsedDocument:
        """Точка входа для DocumentRouter.

        pdf_engine=auto (дефолт ТЗ-2026): основным движком является MinerU —
        PyMuPDF-текстовый слой не справляется со сложной/инлайн-математикой.
        При недоступности или сбое MinerU прозрачно деградируем на собственный
        пайплайн zone→PNG→pix2tex. pdf_engine=pymupdf — только старый пайплайн.
        """
        if self.pdf_engine != "pymupdf":
            try:
                from .mineru_parser import MinerUParser
                return MinerUParser(
                    ocr_processor=self.ocr_processor,
                    pdf_engine="mineru",  # fallback здесь делаем сами, ниже
                    languages=self.languages,
                    vram_gb_override=getattr(self, "vram_gb_override", 0),
                    fast_batch=getattr(self, "fast_batch", True),
                ).parse(file_path, doc_id, progress_callback=progress_callback)
            except Exception as e:  # noqa: BLE001 — fallback обязан ловить всё
                logger.warning(
                    f"MinerU недоступна/ошибка ({type(e).__name__}: {str(e)[:200]}); "
                    f"fallback на PyMuPDF+pix2tex пайплайн")
        return self._parse_pymupdf_fallback(file_path, doc_id, progress_callback)
    
    def supported_extensions(self) -> list[str]:
        return [".pdf"]
    
    def _parse_pymupdf_fallback(self, file_path: Path, doc_id: str, progress_callback=None) -> ParsedDocument:
        logger.info(f"Fallback-парсинг PDF (PyMuPDF+pix2tex): {file_path.name}")
        
        try:
            doc = pymupdf.open(str(file_path))
        except Exception as e:
            raise DocumentParseError(f"Не удалось открыть PDF: {e}") from e
        
        try:
            metadata = doc.metadata or {}
            title = metadata.get("title") or file_path.stem
            author = metadata.get("author")
            total_pages = doc.page_count
            
            # 1. Пытаемся получить встроенное оглавление (TOC)
            toc = doc.get_toc(simple=False) # simple=False дает больше деталей
            
            blocks = []
            block_counter = 0
            
            if toc:
                logger.info(f"Найдено встроенное TOC: {len(toc)} записей. Используем структурный парсинг.")
                blocks = self._parse_with_toc(doc, toc, doc_id, block_counter, progress_callback)
                extraction_method = "structural_toc"
            else:
                logger.info("TOC не найден. Используем постраничный парсинг.")
                blocks, _ = self._extract_via_pymupdf(doc, doc_id, 0)
                # Прогресс постраничного извлечения (текстовый слой быстрый).
                if progress_callback:
                    progress_callback.update(total_pages, total_pages, "Извлечение текста...")
                extraction_method = "pymupdf_fallback"
            
            logger.info(f"PDF извлечён: {total_pages} стр., {len(blocks)} блоков, метод: {extraction_method}")
            
            return ParsedDocument(
                doc_id=doc_id,
                source_path=str(file_path.resolve()),
                original_filename=file_path.name,
                file_format="pdf",
                title=title,
                author=author,
                language="ru", # Упрощено для примера
                total_pages=total_pages,
                blocks=blocks,
            )
        finally:
            doc.close()

    def _parse_with_toc(self, doc: pymupdf.Document, toc: list, doc_id: str, start_block_id: int,
                          progress_callback=None) -> list:
        """Парсинг с использованием структуры TOC и адаптивным выбором Text/OCR."""
        blocks = []
        block_counter = start_block_id
        
        # Группируем TOC по страницам для эффективности
        # Формат TOC: [level, title, page, ...]
        
        for i, item in enumerate(toc):
            # Сквозной прогресс импорта: текущий раздел TOC (OCR — самая
            # медленная фаза, поэтому шкала идёт по всем заголовкам).
            if progress_callback:
                progress_callback.update(
                    i, len(toc), f"Раздел {i + 1} из {len(toc)}: {item[1].strip()[:60]}"
                )
            level = item[0]
            title = item[1].strip()
            start_page = item[2] - 1 # PyMuPDF использует 0-based индексацию
            
            # Определяем конец раздела (начало следующего раздела того же или более высокого уровня)
            end_page = doc.page_count
            for j in range(i + 1, len(toc)):
                if toc[j][0] <= level:
                    end_page = toc[j][2] - 1
                    break
            
            # 1. Создаем идеальный блок-заголовок из TOC (без артефактов!)
            heading_text = f"{'  ' * (level - 1)}{title}"
            blocks.append(DocumentBlock(
                block_id=f"b{block_counter}",
                block_type=BlockType.HEADING,
                text=heading_text,
                page_number=start_page + 1,
                chapter=title,
                coordinates=None,
                extraction_method="toc_metadata",
            ))
            block_counter += 1
            
            # 2. Извлекаем текст для этого раздела
            # Берем сэмпл с первой страницы раздела для проверки качества
            sample_page = doc[start_page]
            sample_text = sample_page.get_text("text")
            quality = self._assess_single_page_quality(sample_text)
            
            if quality < QUALITY_THRESHOLD and self.ocr_processor is not None:
                # Мусорный текст -> используем OCR для этого раздела
                logger.debug(f"Низкое качество текста в '{title}' (стр. {start_page+1}). Применяем OCR.")
                ocr_texts = self.ocr_processor.extract_text_from_pdf(
                    Path(doc.name), 
                    pages=list(range(start_page, min(end_page, start_page + 5))) # Ограничим OCR первыми 5 стр. раздела для скорости, или уберите min для полного OCR
                )
                
                for page_idx, text in enumerate(ocr_texts):
                    if text.strip():
                        # Разбиваем на абзацы
                        for para in text.split('\n\n'):
                            para = para.strip()
                            if len(para) > 20: # Игнорируем мусорные короткие строки
                                blocks.append(DocumentBlock(
                                    block_id=f"b{block_counter}",
                                    block_type=BlockType.TEXT,
                                    text=para,
                                    page_number=start_page + 1 + page_idx,
                                    chapter=title,
                                    coordinates=None,
                                    extraction_method="tesseract_ocr",
                                ))
                                block_counter += 1
            else:
                # Хороший текст -> используем PyMuPDF
                section_blocks, _ = self._extract_via_pymupdf(doc, doc_id, block_counter, start_page, end_page)
                # Присваиваем chapter всем блокам раздела
                for b in section_blocks:
                    b.chapter = title
                blocks.extend(section_blocks)
                block_counter += len(section_blocks)
                
        if progress_callback:
            progress_callback.update(len(toc), len(toc), "Парсинг PDF завершён")
        return blocks

    @staticmethod
    def _garbage_fraction(text: str) -> float:
        """Доля «мусорных» строк в текстовом слое страницы.

        Мусор: строки без слов длиной >= 4 букв (сплющенные формулы, замещённые
        глифы Symbol/Math) либо с признаками битой кодировки. Используется для
        решения «рендерить ли страницу ЦЕЛИКОМ через pix2tex».
        """
        lines = [l for l in (text or "").splitlines() if l.strip()]
        if not lines:
            return 1.0
        bad = 0
        for l in lines:
            stripped = re.sub(r"[^A-Za-zА-Яа-яЁё]", "", l)
            # Слова короче 4 букв = сплющенная формула/битые глифы; слова из
            # повтора одной буквы = каша Symbol/Math (AAAAAAAAAA BBBBBBBB...).
            words = re.findall(r"[A-Za-zА-Яа-яЁё]{4,}", l)
            junk_words = [w for w in words if len(set(w.lower())) <= 2]
            if junk_words and len(junk_words) >= max(2, len(words) // 2):
                bad += 1
            elif not stripped and len(l.strip()) >= 8:
                bad += 1
            elif any(ord(c) == 0xFFFD or 0xE000 <= ord(c) <= 0xF8FF for c in l):
                bad += 1
        return bad / len(lines)

    def _assess_single_page_quality(self, text: str) -> float:
        """Быстрая оценка качества одной страницы."""
        if not text or len(text.strip()) < 10:
            return 0.0
        
        # Считаем долю "мусорных" паттернов (как в нашем sample.pdf)
        garbage_patterns = [r'[A-Z]{5,}', r'ее\s+ее', r'ини\s+ии', r'\(ии']
        garbage_count = sum(len(re.findall(p, text, re.IGNORECASE)) for p in garbage_patterns)
        
        # Если много мусора, качество низкое
        if garbage_count > 3:
            return 0.3
        
        # Проверка на наличие кириллицы
        cyrillic_ratio = len(re.findall(r'[а-яА-ЯёЁ]', text)) / max(len(text), 1)
        return cyrillic_ratio

    def _extract_via_pymupdf(self, doc: pymupdf.Document, doc_id: str, start_block_id: int, start_page: int = 0, end_page: int = None) -> tuple:
        """Извлекает блоки через PyMuPDF для указанного диапазона страниц.

        Обходит ВСЕ страницы диапазона («математика — точная наука»: ни одна
        страница не пропускается). Для каждой страницы оценивается качество
        текстового слоя; на проблемных страницах каждый текстовый блок
        дополнительно уходит в полный цикл page_renderer → formula_recognizer
        (pix2tex) → LaTeX.
        """
        blocks = []
        page_texts = []
        block_counter = start_block_id

        if end_page is None:
            end_page = doc.page_count

        for page_num in range(start_page, end_page):
            page = doc[page_num]
            page_text = page.get_text("text")
            # «Математика — точная наука»: если ТЕКСТОВЫЙ СЛОЙ СТРАНИЦЫ в
            # целом испорчен (низкое качество + высокая доля мусора),
            # рендерим и распознаём сетью КАЖДЫЙ блок страницы, а не только
            # зоны, пойманные локальными эвристиками. Чистая проза при этом
            # остаётся на текстовом слое — GPU не тратится впустую.
            force_all_zones = (
                assess_text_quality(page_text) < QUALITY_THRESHOLD
                or self._garbage_fraction(page_text) > 0.5
            )
            page_blocks = self._extract_page_blocks(
                page, page_num, block_counter, force_all_zones=force_all_zones)
            blocks.extend(page_blocks)
            block_counter += len(page_blocks)
            page_texts.append(page_text)

        return blocks, page_texts

    def _extract_page_blocks(self, page: pymupdf.Page, page_num: int, start_block_id: int,
                             force_all_zones: bool = False) -> list:
        """Извлекает блоки со страницы в РЕЖИМЕ БЛОЧНОЙ СБОРКИ (Level 1 ТЗ).

        Пайплайн извлечения (универсальный, для Математики/Физики/Кода):
        1. page.get_text("dict") — блоки РАЗДЕЛЬНО по span'ам: видно имя
           шрифта каждого фрагмента. Span с математическим шрифтом
           (Symbol/Math/CMMI/CMSY...) или некорректными символами помечает
           весь блок формульной зоной (требование ТЗ п.1).
        2. _merge_paragraphs() — доклейка блоков-продолжений: если PyMuPDF
           разбил один абзац на несколько геометрических блоков (висячая
           последняя строка, обрыв перед встроенной формулой), они
           объединяются обратно в единый логический абзац.
        3. _stitch_block() — нормализация: принудительные \n внутри блока
           (возникают из-за встроенных формул, ∀/∃/∈, индексов) заменяются
           пробелами; дефисные переносы слов склеиваются; математические
           глифы получают защитные пробели и не слипаются со словами.
        4. Формульная зона (math-шрифт / каша / картинка / плохая страница) →
           page_renderer.render_zone_png (PNG 300 dpi) →
           formula_recognizer.recognize (pix2tex) → валидный $$ LaTeX $$
           вставляется в текст блока и уходит в чанки/эмбеддинги/RAG.
        5. Экономия без потери качества: если Unicode-слой блока полностью
           конвертируется правилами (Фаза 0, formula_unicode), сеть для этой
           зоны не вызывается. При force_all_zones в сеть идёт каждый блок.
        """
        raw_blocks = self._blocks_with_fonts(page)
        # Отбрасываем чисто графические блоки (type==1) и пустые.
        parsed = [rb for rb in raw_blocks
                  if rb["btype"] != 1 and rb["text"].strip()]
        merged = self._merge_paragraphs(parsed)
        img_zones = self._image_zones(page)

        blocks = []
        for idx, mb in enumerate(merged):
            if isinstance(mb, dict):
                x0, y0, x1, y1 = mb["rect"]
                text = mb["text"]
                has_math_font = mb.get("has_math_font", False)
            else:
                x0, y0, x1, y1, text = mb[0], mb[1], mb[2], mb[3], mb[4]
                has_math_font = False
            block_text = self._stitch_block(text or "")
            if not block_text:
                continue

            coordinates = {"x0": round(x0, 2), "y0": round(y0, 2),
                           "x1": round(x1, 2), "y1": round(y1, 2)}

            # Заголовок — короткий блок с большой высотой строки
            # (эвристика bbox вместо span.size).
            block_type = self._classify_bbox_block(block_text, y1 - y0)

            method = "pymupdf_blocks"
            zone_png = None
            # ТРИГГЕРЫ зоны формул (ТЗ п.1): math-шрифт в span'ах блока,
            # некорректные символы/каша, пересечение с растровой картинкой,
            # либо вся страница помечена низкокачественной (force_all_zones).
            garbage = self._block_has_math_garbage(block_text)
            needs_render = (
                has_math_font
                or garbage
                or self._zone_intersects(img_zones, (x0, y0, x1, y1))
                or force_all_zones
            )
            if needs_render:
                # ФАЗА 0 (rule-based): текстовый слой многих учебников
                # (Кудрявцев и т.п.) содержит формулы корректными
                # Unicode-глифами (∀ ∃ ∈ → ⩽ ′ ² ₙ). Если из него собираются
                # ВСЕ математические фрагменты блока — нейросетевой инференс
                # этой зоны не нужен вовсе: это микросекунды вместо секунд на
                # зону, и нет галлюцинаций OCR («мусора» вида a\leqslant|a|...).
                rule_frags = self._rules_only_formulas(block_text)
                if rule_frags is not None and rule_frags:
                    block_text = "\n".join(rule_frags)
                    method = "pymupdf_blocks+formula_unicode"
                else:
                    # ФАЗА 0b: чистая проза, которая лишь пересекается с
                    # иллюстрацией, не гонится в GPU — но ТОЛЬКО если блок не
                    # набран math-шрифтом и страница не помечена проблемной.
                    plain_prose_skip = (
                        not force_all_zones
                        and not has_math_font
                        and not garbage
                    )
                    if not plain_prose_skip:
                        # ТЗ п.1-3: зона рендерится в PNG (page_renderer) и
                        # уходит в pix2tex (formula_recognizer); валидный
                        # LaTeX вставляется в текст блока как $$ ... $$ —
                        # попадает в document.md, чанки, эмбеддинги и RAG.
                        zone_png = self._render_zone_lazy(page, (x0, y0, x1, y1), page_num, idx)
                        if zone_png is not None:
                            method = "pymupdf_blocks+zone_png"
                            recognized = self._recognize_formula(zone_png)
                            if recognized:
                                if garbage or has_math_font:
                                    # Кашу/math-шрифт заменяем ЦЕЛИКОМ: текст
                                    # таких зон непригоден ни людям, ни
                                    # эмбеддеру — в базу уходит только LaTeX.
                                    block_text = f"$$ {recognized} $$"
                                else:
                                    block_text = (f"{block_text}\n$$ {recognized} $$").strip()
                                method = "pymupdf_blocks+zone_png+formula_latex"
                                if "$$" in block_text and not re.search(
                                        r"[A-Za-zА-Яа-яЁё]{4}", block_text.replace(recognized, "")):
                                    block_type = BlockType.FORMULA

            blocks.append(DocumentBlock(
                block_id=f"b{start_block_id + idx}",
                block_type=block_type,
                text=block_text,
                page_number=page_num + 1,
                coordinates=coordinates,
                extraction_method=method,
            ))
            if zone_png is not None:
                # PNG-копия зоны идёт отдельным блоком: векторный поиск
                # по тексту остаётся, а зрительный слой доступен viewer'у.
                blocks.append(DocumentBlock(
                    block_id=f"b{start_block_id + idx}z",
                    block_type=BlockType.FIGURE,
                    text="",
                    page_number=page_num + 1,
                    coordinates=coordinates,
                    extraction_method="lazy_png:" + zone_png,
                ))
        return blocks

    # ------------------------------------------------------------------
    # Span-уровневый сбор блоков (детект math-шрифтов и некорректных символов)
    # ------------------------------------------------------------------
    @staticmethod
    def _font_is_math(font_name: str) -> bool:
        return bool(PDFParser._MATH_FONT_RE.search(font_name or ""))

    @staticmethod
    def _span_has_bad_chars(span_text: str) -> bool:
        """Некорректные символы: замещённые глифы math-шрифтов, PUA, U+FFFD."""
        if not span_text:
            return False
        for ch in span_text:
            o = ord(ch)
            if o == 0xFFFD or 0xE000 <= o <= 0xF8FF:  # replacement / PUA
                return True
        return False

    def _blocks_with_fonts(self, page: pymupdf.Page) -> list:
        """Блоки страницы с учётом шрифтов span'ов (ТЗ п.1).

        Возвращает список dict: rect=(x0,y0,x1,y1), text, number, btype —
        плюс has_math_font (bool): True, если ХОТЬ ОДИН span блока набран
        математическим шрифтом (Symbol/Math/CMMI/CMSY/Euclid...) или содержит
        некорректные символы (битые глифы, PUA). Такую зону парсер обязан
        отрендерить в PNG и отправить в pix2tex.
        """
        out = []
        try:
            d = page.get_text("dict", flags=pymupdf.TEXT_PRESERVE_WHITESPACE)
        except Exception as e:  # noqa: BLE001 — повреждённый PDF: откат к blocks
            logger.debug(f"get_text('dict') не удался ({e}), фолбэк на 'blocks'")
            raw = page.get_text("blocks", flags=pymupdf.TEXT_PRESERVE_WHITESPACE)
            return [{"rect": tuple(rb[:4]), "text": rb[4] or "",
                     "number": rb[5], "btype": rb[6], "has_math_font": False}
                    for rb in raw]
        for blk in d.get("blocks", []):
            if blk.get("type", 0) == 1:
                out.append({"rect": tuple(blk["bbox"]), "text": "",
                            "number": blk.get("number", 0), "btype": 1,
                            "has_math_font": False})
                continue
            pieces, has_math = [], False
            for line in blk.get("lines", []):
                for span in line.get("spans", []):
                    stext = span.get("text", "")
                    if self._font_is_math(span.get("font", "")) or \
                            self._span_has_bad_chars(stext):
                        has_math = True
                    pieces.append(stext)
                pieces.append("\n")
            out.append({"rect": tuple(blk["bbox"]),
                        "text": "".join(pieces),
                        "number": blk.get("number", 0),
                        "btype": blk.get("type", 0),
                        "has_math_font": has_math})
        return out


    def _ends_like_paragraph(text: str) -> bool:
        """Текст блока выглядит завершённым (абзац можно не доклеивать)."""
        t = (text or "").rstrip()
        if not t:
            return True
        return t[-1] in ".!?:;…\"'»)]}$"

    @classmethod
    def _merge_paragraphs(cls, items: list) -> list:
        """Доклейка геометрических блоков в логические абзацы.

        Правила объединения соседних текстовых блоков:
        * вертикальный зазор < ~1.5 максимальных высот строк (тот же абзац);
        * предыдущий блок НЕ заканчивается знаком препинания — он обван на
          середине предложения (типично, когда встроенная формула или индекс
          заставили PyMuPDF открыть новый геометрический блок);
        * горизонтальные границы сопоставимы (тот же столбец) либо следую-
          щий блок — «висячая» короткая строка абзаца.
        """
        if len(items) < 2:
            return items

        def g(r, i):
            """Доступ к полю блока независимо от формата (tuple/dict)."""
            if isinstance(r, dict):
                rect = r["rect"]
                return (rect[0], rect[1], rect[2], rect[3], r["text"],
                        r.get("number", 0), r.get("btype", 0))[i]
            return r[i]

        def join(a, b):
            """Склейка двух блоков в один с сохранением формата."""
            x0 = min(g(a, 0), g(b, 0)); y0 = min(g(a, 1), g(b, 1))
            x1 = max(g(a, 2), g(b, 2)); y1 = max(g(a, 3), g(b, 3))
            text = g(a, 4).rstrip() + "\n" + g(b, 4).lstrip()
            if isinstance(a, dict):
                m = dict(a); m["rect"] = (x0, y0, x1, y1); m["text"] = text
                # math-флаг наследуется: если ХОТЬ ОДИН span блока был из
                # Symbol/Math шрифта — зона остаётся формульной.
                m["has_math_font"] = bool(a.get("has_math_font") or b.get("has_math_font"))
                return m
            return [x0, y0, x1, y1, text, g(a, 5), g(a, 6)]

        # Сортировка по порядку чтения: сверху вниз, слева направо.
        items = sorted(items, key=lambda r: (round(g(r, 1), 1), round(g(r, 0), 1)))
        merged = [items[0]]
        for cur in items[1:]:
            prev = merged[-1]
            prev_h = max(g(prev, 3) - g(prev, 1), 1.0)
            cur_h = max(g(cur, 3) - g(cur, 1), 1.0)
            gap = g(cur, 1) - g(prev, 3)
            same_column = (min(g(prev, 2), g(cur, 2)) - max(g(prev, 0), g(cur, 0))) > 0.5 * min(
                g(prev, 2) - g(prev, 0), g(cur, 2) - g(cur, 0))
            dangling = (g(cur, 2) - g(cur, 0)) < 0.6 * max(g(prev, 2) - g(prev, 0), 1.0)
            breakable = not cls._ends_like_paragraph(g(prev, 4))
            if gap < 1.5 * max(prev_h, cur_h) and breakable and (same_column or dangling):
                merged[-1] = join(prev, cur)
            else:
                merged.append(cur)
        return merged

    # ------------------------------------------------------------------
    # Level 1 helpers: склейка разрывов, инварианты формул, геометрия
    # ------------------------------------------------------------------
    # Математические инварианты: символы, которые не должны слипаться с
    # соседними словами после склейки строк (голые Unicode-глифы).
    _MATH_INVARIANTS = "∀∃∈∉⊂⊆⊇∪∩∅→⇒⇔ℝℕℤℚℂ×±≤≥≠≈∑∏∫√∞∧∨¬∪̸"

    @classmethod
    def _stitch_block(cls, raw_text: str) -> str:
        """Склеивает принудительные переносы внутри текстового блока PDF.

        Правила:
        1. Дефисный перенос слова ("функ-\nция") -> склейка без пробела.
        2. Все остальные \\n -> одиночный пробел (формулы и индексы больше
           не рвут предложение на куски).
        3. Избыточные пробелы схлопываются (" ".join(split())).
        4. Математические инварианты получают защитные пробели с обеих
           сторон, если парсер извлёк их вплотную к словам/скобкам.
        """
        text = raw_text.replace("\r\n", "\n").replace("\r", "\n")
        # 1. Мягкий дефисный перенос: буква + '-' + перевод строки + буква.
        text = re.sub(r"(?<=[\w])(-\n)(?=[\w])", "", text)
        # 2. Принудительные переносы строк -> пробелы.
        text = text.replace("\n", " ")
        # 3. Схлопывание пробельного шума.
        text = " ".join(text.split())
        # 3b. FIX (номера сносок [3], [1], |4| рвут контекст эмбеддингов):
        # в книгах Кудрявцева ссылки на сноски прилипают к концу предложений
        # («...ограничена [3].»). Удаляем только короткие цифровые маркеры,
        # стоящие у конца предложения/слова; математические [0, 1], [a, b]
        # (есть пробел/буква внутри) не трогаем.
        text = re.sub(r"(?<=[а-яёА-ЯЁ\w).,;:!?…])\s*\[\d{1,3}\](?=[.,;:!?\s]|$)",
                      "", text)
        text = re.sub(r"(?<=[а-яёА-ЯЁ\w).\s])\|\s*\d{1,3}\s*\|(?=[.,;:!?\s]|$)",
                      "", text)
        # 4. Защитные пробели вокруг математических глифов-инвариантов.
        for sym in cls._MATH_INVARIANTS:
            esc = re.escape(sym)
            # прижимаем к символам, которые не являются пробелом/матем. знаком
            text = re.sub(rf"(?<=[^\s{esc}])(?={esc})", " ", text)
            text = re.sub(rf"(?<={esc})(?=[^\s{esc}])", " ", text)
        return " ".join(text.split()).strip()

    @staticmethod
    def _image_zones(page: pymupdf.Page) -> list:
        """Прямоугольники встроенных изображений на странице (для пересечений)."""
        zones = []
        try:
            for info in page.get_image_info():
                bb = info.get("bbox")
                if bb:
                    zones.append(tuple(bb))
        except Exception:  # pragma: no cover - повреждённый PDF
            pass
        return zones

    @staticmethod
    def _zone_intersects(zones: list, rect: tuple, eps: float = 2.0) -> bool:
        x0, y0, x1, y1 = rect
        for zx0, zy0, zx1, zy1 in zones:
            if zx0 < x1 - eps and zx1 > x0 + eps and zy0 < y1 - eps and zy1 > y0 + eps:
                return True
        return False

    @staticmethod
    def _block_has_math_garbage(text: str) -> bool:
        """Блок похож на склейку формул/графики в кашу текстовым слоем.

        Признаки: плотная смесь одиночных латинских букв, операторов и цифр
        без слов (сплющенные формулы), либо обилие замусоренных скобок.
        """
        if not text:
            return False
        letters = sum(ch.isalpha() and ch.isascii() for ch in text)
        words = len(re.findall(r"[A-Za-zА-Яа-яЁё]{4,}", text))
        ops = len(re.findall(r"[∑∫∏√⊂⊆∈∀∃≤≥≠±×÷^_{}]", text))
        digits = sum(ch.isdigit() for ch in text)
        dense = letters + digits + ops
        if dense == 0:
            return False
        # «каша»: много изолированных символов при почти полном отсутствии слов
        return words == 0 and dense >= 12 and ops >= 2

    _STRONG_MATH = set("∀∃∈∉⊂⊆∪∩∑∏∫√≤≥≠≈⇒⇔↦∂∇×÷±"
                       "αβγδεζηθικλμνξπρστυφχψω"
                       "ΑΒΓΔΕΖΗΘΙΚΛΜΝΞΠΡΣΤΥΦΧΨΩ")

    _HINT_CACHE: "dict[str, str]" = {}

    @staticmethod
    def _convert_hint_tail(hint: str) -> str:
        """Пропускает хвост-подсказку через тот же конвертер (для точного
        вырезания из результата). Результат кэшируется — набор сильных
        символов блока стабилен в пределах страницы/документа."""
        cached = PDFParser._HINT_CACHE.get(hint)
        if cached is not None:
            return cached
        from ..processing.formula_unicode import convert_unicode_math as _c
        # Пустая «проза» + хвост: looks_mathy сработает на самих глифах.
        res = _c("z" + hint) or ""
        tail = res[1:].lstrip() if res.startswith("z") else ""
        if len(PDFParser._HINT_CACHE) > 4096:
            PDFParser._HINT_CACHE.clear()
        PDFParser._HINT_CACHE[hint] = tail
        return tail

    @staticmethod
    def _rules_only_formulas(block_text: str) -> "list[str] | None":
        """Фаза 0 (rule-based): собрать LaTeX из Unicode-слоя ПРАВИЛАМИ.

        Разбивает склеенный блок на фрагменты и для каждого математического
        fragment'а пробует convert_unicode_math(). Экономия критична: на
        714 страницах Кудрявцева подавляющее большинство формул — печатный
        Unicode-слой, инференс сети для них не нужен (микросекунды вместо
        секунд, и нет OCR-галлюцинаций).

        Возвращает список готовых строк ``$$ ... $$`` либо None, если хотя бы
        один mathy-фрагмент блокам не поддаётся правилам — тогда зону берёт
        нейросеть ровно как раньше (консервативно, качество не понижаем).
        Если математических фрагментов нет вовсе — тоже None (зона не наша).
        """
        try:
            from ..processing.formula_unicode import (
                convert_unicode_math,
                looks_mathy,
            )
        except Exception as exc:  # pragma: no cover - модуль обязателен
            logger.debug(f"formula_unicode недоступен: {exc}")
            return None

        if not block_text or not block_text.strip():
            return None

        parts = PDFParser._split_math_fragments(block_text)
        # «Сильная» математика блока (кванторы, операторы, греки) — это и есть
        # настоящий признак зоны формул. Встроенные формулы рвутся PyMuPDF-
        # склейкой на клаузы, и одиночная «f(x+T)=f(x)» или «⇒ ...» сама по
        # себе не mathy; при сильных символах в блоке такие клаузы тоже
        # считаются формулами и конвертируются правилами.
        strong_chars = {ch for ch in block_text if ch in PDFParser._STRONG_MATH}
        frags: list[str] = []
        ok_any = False
        for part in parts:
            p = part.strip()
            if not p:
                continue
            is_formula_clause = looks_mathy(p) or bool(strong_chars)
            if not is_formula_clause:
                # Прозаический фрагмент внутри mathy-блока. Правила его не
                # трогают, но и отдавать всю зону в сеть из-за одного «Следо-
                # вательно,» — расточительство на 714 страницах. Оставляем как
                # есть (текстовый слой корректен), в LaTeX конвертируем только
                # математические клаузы.
                frags.append(p)
                continue
            tex = convert_unicode_math(p)
            if tex is None and strong_chars:
                # Клауза-формула без «сильных» глифов внутри («f ( x + T ) =
                # f ( x )») не проходит looks_mathy. Хвост из сильных символов
                # блока — подсказка конвертеру; после конвертации хвост
                # вырезается из результата. Вырезается по исходному хвосту,
                # пропущенному через те же правила замены (порядок вызова
                # гарантирует это: сначала пробный convert(p+hint), который
                # инициализирует кэш, затем _convert_hint_tail).
                hint = " " + "".join(sorted(strong_chars))
                tex = convert_unicode_math(p + hint)
                if tex is not None:
                    tail_tex = PDFParser._convert_hint_tail(hint)
                    if tail_tex and tex.endswith(tail_tex):
                        tex = tex[: len(tex) - len(tail_tex)].rstrip()
                    elif tex.endswith(hint.strip()):
                        tex = tex[: -len(hint.strip())].rstrip()
            if tex is None or not tex:
                return None  # правила бессильны → пусть работает сеть
            frags.append(f"$${tex}$$")
            ok_any = True
        if not ok_any:
            return None
        # Блокировка ложных срабатываний: если ни один $$ -фрагмент не содержит
        # «сильный» математический символ (квантор/оператор/set-теория) или
        # LaTeX-команду, а лишь бытовые «→», «·» или «=» в прозе — это не зона
        # формул, решение за сетью/текстом как раньше.
        if not any(any(ch in PDFParser._STRONG_MATH or ch == "\\" for ch in fr)
                   for fr in frags if fr.startswith("$$")):
            return None
        return frags

    @staticmethod
    def _split_math_fragments(text: str) -> "list[str]":
        """Разбивает склеенный блок на клаузы-кандидаты в формулы.

        Границы: препинание + пробел; слипшиеся границы вида «$квантор» или
        «$функция с аргументом» после закрывающей скобки/цифры/буквы без
        пробела (типично для PyMuPDF-склейки встроенных формул)."""
        parts = re.split(r"(?<=[.;:,])\s+", text)
        out: list[str] = []
        for part in parts:
            piece = ""
            for ch in part:
                if (ch in "∀∃λμσπ" and piece
                        and piece[-1] in ")0123456789xyzwt"):
                    out.append(piece)
                    piece = ch
                elif ch == "f" and piece.endswith(") "):
                    out.append(piece.rstrip())
                    piece = "f"
                else:
                    piece += ch
            out.append(piece)
        return out

    def _render_zone_lazy(self, page: pymupdf.Page, rect: tuple,
                          page_num: int, block_idx: int):
        """Ленивый рендер зоны блока в PNG (300 dpi) через page_renderer.

        Возвращает путь к PNG или None, если модуль недоступен/ошибка —
        импорт никогда не роняется из-за визуального fallback-слоя.
        """
        try:
            from .page_renderer import render_zone_png  # ленивый импорт
            out_dir = Path(page.parent.name).parent / (
                Path(page.parent.name).stem + "_zones")
            return render_zone_png(page, rect, out_dir,
                                   f"p{page_num + 1}_b{block_idx}.png", dpi=300)
        except Exception as e:  # noqa: BLE001 — деградация допустима
            logger.debug(f"Ленивый PNG зоны (стр. {page_num + 1}) пропущен: {e}")
            return None

    def _recognize_formula(self, zone_png) -> str | None:
        """Image-to-LaTeX распознавание PNG-зоны формулы (см. formula_recognizer).

        Деградация без исключений: если pix2tex/texify не установлены или
        инференс упал — возвращаем None, импорт продолжается как раньше
        (PNG-слой остаётся, LaTeX-текст просто не появляется).
        """
        if not zone_png:
            return None
        try:
            from ..processing.formula_recognizer import get_formula_recognizer
            return get_formula_recognizer().recognize(zone_png)
        except Exception as e:  # noqa: BLE001
            logger.debug(f"Распознавание формулы пропущено: {e}")
            return None

    @staticmethod
    def _classify_bbox_block(text: str, height: float) -> BlockType:
        """Тип блока по геометрии bbox (замена span-size эвристики)."""
        lines = [l for l in text.split("\n") if l.strip()]
        if height > 60 and len(text) < 150 and len(lines) <= 2:
            return BlockType.HEADING
        return BlockType.TEXT

    def _classify_text_block(self, block: dict) -> tuple:
        """Определяет тип блока."""
        full_text_parts = []
        max_font_size = 0.0
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                full_text_parts.append(span.get("text", ""))
                if span.get("size", 0) > max_font_size:
                    max_font_size = span.get("size", 0)
        
        full_text = "".join(full_text_parts).strip()
        if max_font_size > 14 and 0 < len(full_text) < 150:
            return full_text, BlockType.HEADING
        return full_text, BlockType.TEXT