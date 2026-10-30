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

import importlib
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

    def __init__(self, ocr_processor=None, pdf_engine: str = "auto", languages: str = "ru,en",
                 vram_gb_override: int = 0, fast_batch: bool = True):
        self.ocr_processor = ocr_processor
        self.pdf_engine = (pdf_engine or "auto").lower()
        self.languages = languages
        # FIX (лог 22:10): см. PdfEngineConfig.vram_gb_override — объявляемый
        # MinerU «виртуальный» объём VRAM для выбора Batch Ratio (0 = авто).
        self.vram_gb_override = int(vram_gb_override or 0)
        self.fast_batch = bool(fast_batch)
        # Результат последнего _should_force_gpu() (для _apply_vram_override).
        self._last_gpu_decision = False

    @classmethod
    def from_settings(cls, settings, ocr_processor=None) -> "MinerUParser":
        pe = getattr(settings, "pdf_engine", None)
        return cls(
            ocr_processor=ocr_processor,
            pdf_engine=getattr(pe, "engine", "auto"),
            languages=getattr(pe, "languages", "ru,en"),
            vram_gb_override=getattr(pe, "vram_gb_override", 0),
            fast_batch=getattr(pe, "fast_batch", True),
        )

    # Кэш результата is_available(): None = ещё не проверяли, bool = проверено.
    _AVAIL_CACHE: "bool | None" = None
    # Причина последней недоступности (для честного сообщения об ошибке).
    _LAST_REASON: str = ""

    @classmethod
    def is_available(cls) -> bool:
        """MinerU установлена (или хотя бы её CLI в PATH)?

        FIX (лог 16:09): раньше проверка была «один big-import» `import mineru`.
        В MinerU 3.x корневой пакет тянет на старте тяжёлые зависимости
        (torch/paddle и т.п.), и при любом их сбое `import mineru` бросает
        ModuleNotFoundError с ЧУЖИМ именем модуля — мы же поднимали
        «MinerU не установлена», хотя сам парсер цел и CLI работает. Теперь
        проверяем три независимых признака по отдельности, каждый в своём
        try/except, и пишем в лог, что именно нашлось/упало.

        FIX (лог 16:09, вторая часть): признаки 2 и 3 были недостижимы —
        `from mineru.backend...` гарантированно падает тем же сбоем, что и
        корень (он импортирует mineru первым), а shutil.which() ищет CLI только
        в PATH, которого у пользователя без активированного venv нет. Добавлен
        признак 2b: поиск скрипта `mineru(.exe)` рядом с текущим интерпретатором
        (venv\\Scripts) — тот же путь, которым приложение уже запускается.
        Результат кэшируется: дорогая проверка выполняется один раз за процесс.
        """
        if cls._AVAIL_CACHE is not None:
            return cls._AVAIL_CACHE
        reasons = []
        avail = False
        # 1) Корневой пакет импортируется целиком.
        try:
            importlib.import_module("mineru")  # сам факт импорта и есть проверка
            logger.debug("MinerU: импорт пакета 'mineru' успешен")
            avail = True
        except Exception as e:  # noqa: BLE001 — ловим и ImportError, и прочее
            reasons.append(f"import mineru: {type(e).__name__}: {e}")
        # 2) Рабочий Python-API даже при битом корне (частый случай в 3.x).
        if not avail:
            try:
                importlib.import_module(
                    "mineru.backend.pipeline.pipeline_analyze")
                reasons.append("API-подмодуль доступен несмотря на сбой корня")
                avail = True
            except Exception:  # noqa: BLE001
                pass
        # 3a) CLI в PATH — тоже полноценный способ запуска.
        if not avail and (shutil.which("mineru") or shutil.which("mineru.exe")):
            logger.debug("MinerU: Python-API недоступна (%s), но CLI найден в PATH", reasons)
            avail = True
        # 3b) CLI рядом с нашим интерпретатором (venv\\Scripts\\mineru.exe или
        #     bin/mineru на POSIX) — PATH может не содержать Scripts каталога.
        if not avail:
            for cand in cls._cli_candidates():
                if cand.is_file():
                    logger.debug(
                        "MinerU: API недоступна (%s), но CLI найден рядом с python: %s",
                        "; ".join(reasons), cand)
                    avail = True
                    break
        cls._AVAIL_CACHE = avail
        cls._LAST_REASON = "; ".join(reasons) if not avail else ""
        if not avail:
            logger.warning(
                "MinerU недоступна ({}) — будет использован fallback PyMuPDF+pix2tex",
                cls._LAST_REASON or "пакет и CLI не найдены")
        return avail

    @staticmethod
    def _cli_candidates():
        """Пути к консольному скрипту mineru без зависимости от PATH."""
        exe_dir = Path(sys.executable).parent
        names = ("mineru.exe", "mineru") if os.name == "nt" else ("mineru",)
        return (exe_dir / n for n in names)

    @classmethod
    def _cli_executable(cls) -> "str | None":
        """Находит CLI mineru: сначала PATH, затем рядом с sys.executable."""
        found = shutil.which("mineru") or shutil.which("mineru.exe")
        if found:
            return found
        for cand in cls._cli_candidates():
            if cand.is_file():
                return str(cand)
        return None

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
            # FIX (лог 16:39, конфликт VRAM): Ollama держит qwen+bge в видео-
            # памяти (~4.8 ГБ), из-за чего ONNX/torch внутри MinerU получают
            # "GPU Memory: 1 GB" и уходят в CPU-режим (медленно + RAM 91%).
            # На время парсинга выгружаем модели Ollama (keep_alive=0s); после
            # импорта они перезагрузятся сами при первой векторизации.
            try:
                from src.llm.ollama_vram import release_vram
                vram_ctx = release_vram(enabled=self.fast_batch)
            except Exception:  # noqa: BLE001 — модуль опционален
                import contextlib
                vram_ctx = contextlib.nullcontext()
            with vram_ctx:
                # FIX (лог 21:39): выгрузка Ollama из VRAM сама по себе НЕ
                # помогает инференсу MinerU, если torch видит GPU, но CUDA-процессы
                # ещё не держат в нём памяти. В этом состоянии MinerU выбирает
                # консервативный лимит "GPU Memory: 1 GB" и работает на CPU.
                # Явно указываем режим устройства через переменные окружения
                # MinerU (MINERU_DEVICE_MODE / CUDA_VISIBLE_DEVICES), чтобы он
                # гарантированно попал на GPU после освобождения VRAM.
                if self._should_force_gpu():
                    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
                    os.environ["MINERU_DEVICE_MODE"] = "cuda"
                else:
                    # FIX (лог 22:10): раньше здесь оставалась «cuda» из прошлых
                    # прогонов — при занятой VRAM MinerU получала OOM/тот же
                    # ratio=1. В CPU-режиме явно фиксируем cpu и снимаем override.
                    os.environ["MINERU_DEVICE_MODE"] = "cpu"
                # FIX (лог 22:10, «ты сломал»): Batch Ratio MinerU считается от
                # TOTAL объёма VRAM карты (get_vram): RTX 5060 = 8 ГБ → ratio=1,
                # MFR идёт ~2 стр/с — медленнее прежнего CPU-прохода. Штатный
                # рычаг MinerU — MINERU_VIRTUAL_VRAM_SIZE: объявляем карте больший
                # объём, получаем ratio 2–4. Применяется только при GPU-режиме и
                # выгруженных моделях Ollama (fast_batch) либо явном override.
                self._apply_vram_override(gpu_active=self._last_gpu_decision)
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
        api_err = None
        try:
            self._mineru_python_api(pdf_path, out_dir)
        except ImportError as e:
            # Модули другой раскладки / ранняя MinerU — переходим на CLI.
            api_err = f"ImportError: {e}"
        except Exception as e:  # noqa: BLE001
            # FIX (лог 16:09): при битых транситивных зависимостях MinerU 3.x
            # (нехватка torch/paddle после смены индекса PyTorch и т.п.)
            # do_parse падает НЕ с ImportError, и раньше это мгновенно уводило
            # в PyMuPDF-fallback. Пробуем CLI — он запускает MinerU в своём
            # процессе и часто работает даже когда импорт из нашего процесса
            # сломан; ошибку API сохраняем в лог для диагностики.
            api_err = f"{type(e).__name__}: {e}"

        if api_err is not None:
            logger.warning(
                f"MinerU: Python-API не отработал ({api_err[:200]}) — пробую CLI")
            try:
                self._mineru_cli(pdf_path, out_dir)
            except DocumentParseError:
                raise
            except Exception as e:  # noqa: BLE001
                raise DocumentParseError(
                    f"MinerU не смогла обработать PDF (API: {api_err[:150]}; "
                    f"CLI: {e})") from e

        md_path = self._find_markdown(out_dir)
        if md_path is None:
            # FIX (лог 16:09): раньше здесь молча поднималось «MinerU не
            # создала Markdown», и parse() глушил это в PyMuPDF-fallback.
            # Теперь читаем реальный лог процесса CLI — пользователь видит,
            # почему именно упала MinerU (битый torch/paddle/CUDA и т.п.).
            detail = ""
            log_cands = sorted(out_dir.rglob("mineru.log"),
                               key=lambda p: p.stat().st_size, reverse=True)
            if log_cands:
                try:
                    txt = log_cands[0].read_text(encoding="utf-8", errors="replace")
                    err_lines = [l for l in txt.splitlines()
                                 if any(k in l.upper() for k in
                                        ("ERROR", "TRACEBACK", "EXCEPTION"))]
                    detail = "\n".join(err_lines[-8:]) or txt.strip()[-400:]
                except Exception:  # noqa: BLE001
                    pass
            raise DocumentParseError(
                "MinerU завершилась без Markdown."
                + (f" Лог процесса:\n{detail[:600]}" if detail else
                   " (лог не найден; проверьте 'mineru -V' и установку зависимостей)"))
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

        # AUDIT (чистка стека, 2026-10-30): сигнатура do_parse меняется в каждом
        # майноре MinerU (2.5 → 3.x: появились return_layout/return_middle_json,
        # исчезли/переименовались f_* флаги). Раньше любой неизвестный kwarg
        # ронял Python-API с TypeError → тихий переход на CLI. Теперь фильтруем
        # аргументы по реальной сигнатуре функции: передаём только поддерживаемые,
        # лениво подменяя эквивалентами (middle.json — по любому имени флага).
        try:
            import inspect as _inspect
            params = set(_inspect.signature(do_parse).parameters)
        except Exception:  # noqa: BLE001
            params = None  # не смогли интроспектировать — шлём как есть
        if params is not None and not any(
                p.kind == _inspect.Parameter.VAR_KEYWORD for p in
                _inspect.signature(do_parse).parameters.values()):
            params = None  # функция принимает только явные аргументы —
                           # фильтровать не по чему, идём «как есть» (TypeError
                           # перехватит _run_mineru и переключит на CLI)

        def _supported(**kw):
            """Оставляет только kwarg'и, присутствующие в сигнатуре do_parse."""
            if params is None:
                return kw
            return {k: v for k, v in kw.items() if k in params}

        middle_flags = _supported(f_dump_middle_json=True,
                                  return_middle_json=True,
                                  return_layout=True)
        common = _supported(
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
            f_dump_model_output=False,
            f_dump_orig_pdf=False,
            f_dump_content_list=False,
            f_make_md_mode=md_mode,  # MM_MD: markdown с картинками и $$..$$
        )
        if params is not None and not common:
            raise ImportError("do_parse: неизвестная сигнатура — переключаемся на CLI")
        # ВАЖНО: обходим ВСЕ страницы (start/end по умолчанию = весь файл) —
        # «математика точная наука», частичный разбор запрещён.
        # FIX (переход к странице из чата): middle.json содержит page_idx
        # каждого блока — единственный точный источник номеров страниц в
        # MinerU-разборе; поэтому dump-only режим НЕ включаем, а флаг дампа
        # среднего JSON запрашиваем под любым именем, который понимает данная
        # версия пакета (f_dump_middle_json / return_middle_json / return_layout).
        do_parse(output_dir=str(out_dir), **common, **middle_flags, **kwargs)

    def _mineru_cli(self, pdf_path: Path, out_dir: Path) -> None:
        # FIX (лог 16:09): shutil.which видел только PATH; если приложение
        # запускалось из venv без его активации, mineru.EXE лежал в Scripts
        # рядом с python и НЕ находился → ложвое «MinerU не установлена».
        exe = self._cli_executable()
        if exe is None:
            # Python-API уже поднял ImportError — CLI тоже нет: MinerU нет вовсе.
            reason = getattr(self, "_LAST_REASON", "") or MinerUParser._LAST_REASON
            hint = ("проверьте наличие mineru(.exe) в Scripts/bin вашего venv"
                    if not reason else f"причина недоступности: {reason[:200]}")
            raise DocumentParseError(
                "MinerU не установлена (pip install 'mineru[core]' + "
                f"mineru-models-download); {hint}")
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
        # FIX (чистка стека, 2026-10-30): в CLI MinerU 2.x формулы включены по
        # умолчанию, но флаг "--formula-only" существует; в 3.x появились
        # явные --enable_formula/--disable_formula. Определяем поддержку флага
        # по --help и передаём его, чтобы режим MFD/MFR (LaTeX в markdown)
        # гарантированно совпадал с Python-API (formula_enable=True). Иначе при
        # дефолтах новой версии CLI формулы могли молча выключаться.
        try:
            help_out = subprocess.run([exe, "--help"], capture_output=True,
                                      text=True, encoding="utf-8",
                                      errors="replace", timeout=30)
            help_text = (help_out.stdout or "") + (help_out.stderr or "")
            if "--enable_formula" in help_text:
                cmd.append("--enable_formula")
        except Exception:  # noqa: BLE001 — help не валяет основной запуск
            pass
        env = dict(os.environ)
        # FIX (лог 21:39): setdefault + _cuda_available() игнорировали реальную
        # свободную VRAM — при занятой памяти CLI тоже уходил в CPU-квоту 1 ГБ.
        if self._should_force_gpu():
            env["MINERU_DEVICE_MODE"] = "cuda"
        else:
            env["MINERU_DEVICE_MODE"] = "cpu"
        # FIX (лог 22:10): синхронизируем виртуальный объём VRAM с решением по
        # GPU — иначе CLI наследует устаревший MINERU_VIRTUAL_VRAM_SIZE из API-
        # прогона и батч-рацио не соответствует реальному устройству.
        if env.get("MINERU_DEVICE_MODE") == "cuda":
            gb = self.vram_gb_override or (
                self._FAST_BATCH_VRAM_GB if self.fast_batch else 0)
            if gb > 0:
                env["MINERU_VIRTUAL_VRAM_SIZE"] = str(int(gb))
            else:
                env.pop("MINERU_VIRTUAL_VRAM_SIZE", None)
        else:
            env.pop("MINERU_VIRTUAL_VRAM_SIZE", None)
        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", env=env
            )
        except FileNotFoundError:
            # FIX (лог 16:09): в venv на Windows консольный скрипт лежит как
            # mineru.EXE; иногда shutil его «находит», но CreateProcess спотыкается
            # (битый wrapper). Резервный путь — запуск модуля напрямую тем же
            # интерпретатором, что и приложение.
            logger.debug("MinerU: прямой запуск exe не удался — пробую python -m mineru.cli.client")
            proc = subprocess.run(
                [sys.executable, "-m", "mineru.cli.client", *cmd[1:]],
                capture_output=True, text=True, encoding="utf-8", errors="replace", env=env
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
    def _free_vram_mb() -> "float | None":
        """Свободная VRAM первой CUDA-карты, МБ (None — определить нельзя)."""
        try:
            import pynvml
            pynvml.nvmlInit()
            h = pynvml.nvmlDeviceGetHandleByIndex(0)
            free = pynvml.nvmlDeviceGetMemoryInfo(h).free / 1048576.0
            try:
                pynvml.nvmlShutdown()
            except Exception:  # noqa: BLE001
                pass
            return float(free)
        except Exception:  # noqa: BLE001 — нет nvidia-smi/pynvml и т.п.
            pass
        try:
            out = subprocess.run(["nvidia-smi", "--query-gpu=memory.free",
                                  "--format=csv,noheader,nounits"],
                                 capture_output=True, text=True, timeout=5)
            first = (out.stdout or "").strip().splitlines()
            if first:
                return float(first[0].strip())
        except Exception:  # noqa: BLE001
            pass
        return None

    #: Минимум свободной VRAM, чтобы гонять MinerU на GPU (Unimernet + layout
    #  + OCR стеки просят ~2.5–3 ГБ; ниже этого порога ONNX уходит в 1GB-квоту).
    _GPU_MIN_FREE_MB = 2560.0

    def _should_force_gpu(self) -> bool:
        """Ставить ли MINERU_DEVICE_MODE=cuda принудительно.

        FIX (лог 21:39): раньше режим выбирался по одному лишь факту наличия
        CUDA — при занятой Ollama VRAM MinerU всё равно падала в CPU с
        'GPU Memory: 1 GB'. Теперь решение принимается ПОСЛЕ выгрузки моделей
        Ollama и учитывает реальное количество свободной видеопамяти.
        Решение запоминается в self._last_gpu_decision для _apply_vram_override.
        """
        if not self._cuda_available():
            self._last_gpu_decision = False
            return False
        free = self._free_vram_mb()
        if free is None:
            # Определять нечем — доверяем CUDA (лучше рискнуть GPU, чем
            # гарантированный медленный CPU на машине с дискретной картой).
            self._last_gpu_decision = True
            return True
        if free < self._GPU_MIN_FREE_MB:
            logger.warning(
                f"MinerU: после выгрузки Ollama свободно лишь {free:.0f} МБ VRAM "
                f"(нужно ≥{self._GPU_MIN_FREE_MB:.0f}) — парсинг пойдёт на CPU")
            self._last_gpu_decision = False
            return False
        logger.info(f"MinerU: GPU-режим подтверждён (свободно {free:.0f} МБ VRAM)")
        self._last_gpu_decision = True
        return True

    #: Сколько «виртуальных» ГБ объявлять карте при активном fast_batch, если
    #: пользователь не задал vram_gb_override сам. 16 → Batch Ratio 8 (пороги
    #: MinerU: 6→2, 8→4, 16→8): Unimernet small + layout весят немного, на
    #: 8-ГБ карте ratio=8 реально умещается при полностью выгруженной Ollama.
    _FAST_BATCH_VRAM_GB = 16

    def _apply_vram_override(self, gpu_active: bool) -> None:
        """Выставить/снять MINERU_VIRTUAL_VRAM_SIZE перед прогоном MinerU.

        FIX (лог 22:10): без этого RTX 5060 (8 ГБ total) всегда получает
        Batch Ratio 1 — MFR ползёт ~2 стр/с даже на GPU. Override применяется
        только когда MinerU реально стартует на GPU; в CPU-режиме переменная
        снимается, чтобы не провоцировать ложные ожидания.
        """
        var = "MINERU_VIRTUAL_VRAM_SIZE"
        if not gpu_active:
            os.environ.pop(var, None)
            return
        gb = self.vram_gb_override or (
            self._FAST_BATCH_VRAM_GB if self.fast_batch else 0)
        if gb > 0:
            os.environ[var] = str(int(gb))
            logger.info(
                f"MinerU: объявляем виртуальный объём VRAM {gb} ГБ "
                f"(Batch Ratio ускорение; отключение: pdf_engine.fast_batch: false "
                f"или vram_gb_override: 0)")
        else:
            os.environ.pop(var, None)

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
