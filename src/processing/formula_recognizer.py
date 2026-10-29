# src/processing/formula_recognizer.py
"""Распознавание формул из изображений (Image-to-LaTeX).

ПРОБЛЕМА, которую решает этот модуль: обычные текстовые экстракторы PDF
(PyMuPDF/pdfplumber/PyPDF) превращают математическую вёрстку в «кашу» —
сплющенные глифы, потерянные индексы и дроби. Такой текст непригоден ни для
обучения ИИ, ни для RAG-выдачи. Стек lama-loca уже НЕ содержит PyPDF/pdfplumber
(используется PyMuPDF), но каша от текстового слоя остаётся: pdf_parser
помечает такие зоны эвристикой `_block_has_math_garbage()` и кладёт рядом
PNG-блок с `extraction_method="lazy_png:<путь>"`, однако текст формулы там
остаётся пустым — в чанкинг и эмбеддинги формула не попадает.

РЕШЕНИЕ: перед подачей текста в LLM/эмбеддер PNG-зона прогоняется через
Image-to-LaTeX модель, результат вставляется в текст как LaTeX ($...$),
который дальше штатно конвертируется стеком: latex2mathml → UI (ft.Markdown),
latex_unicode/UnicodeMath → mathtext → PDF (exporter).

Движки (по приоритету, первый доступный побеждает):
  1. pix2tex (LaTeX-OCR)         — локальная лёгкая модель (~20 МБ весов),
                                   pip install pix2tex[gui]; кэширует веса.
  2. texify                       — локальная тяжёлая модель (точнее),
                                   pip install texify; требует torch+transformers.
  3. pix2tex API                  — сетевой фолбэк (https://pix2tex.vercel.app/api),
                                   используется только если allow_network=true.
Если движки не установлены — recognize() возвращает None и пайплайн работает
как раньше (чистая деградация, ImportError не роняется).

Потокобезопасность: ленивая инициализация под замком; инференс модели — тоже
под глобальным замком (модели не реентерабельны).
"""
from __future__ import annotations

import hashlib
import os
import re
import threading
from pathlib import Path
from typing import Optional

from loguru import logger

# Одиночная формула, которую модель вернула «неуверенно»: пустая, мусорная
# или состоящая из одних операторов — в текст её вставлять нельзя.
_MIN_USEFUL_LEN = 2
_LATEX_CMD_RE = re.compile(r"\\[A-Za-z]+|[{}^_=/\[\]\dA-Za-z]")


class FormulaRecognizer:
    """Ленивый синглтон распознавателя формул Image→LaTeX."""

    _instance: Optional["FormulaRecognizer"] = None
    _instance_lock = threading.Lock()

    PIX2TEX_API_URL = "https://pix2tex.vercel.app/api"

    def __init__(self, prefer: str = "auto", allow_network: bool = False,
                 cache_dir: str = "data/temp/formula_cache"):
        self.prefer = prefer  # auto | pix2tex | texify | api | off
        self.allow_network = allow_network
        self.cache_dir = Path(cache_dir)
        # _lock: защита конфигурации/кэша и ленивой инициализации (быстрая).
        # _infer_lock: сериализация инференса модели (долгая) — НЕ удерживается
        # во время загрузки движка, иначе попытка recognize() из другого потока
        # блокировалась бы на минуты скачивания весов.
        self._lock = threading.Lock()
        self._infer_lock = threading.RLock()
        self._engine: Optional[str] = None      # имя выбранного движка
        self._model = None                      # экземпляр движка
        self._device = "cpu"                    # устройство инференса-якорь
        self._tried_init = False
        self._png_cache: dict = {}              # sha1(png) -> latex

    @classmethod
    def get_instance(cls) -> "FormulaRecognizer":
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = cls()
        return cls._instance

    @classmethod
    def configure(cls, prefer: str = "auto", allow_network: bool = False,
                  cache_dir: Optional[str] = None):
        """Вызывается из настроек приложения/document_router (config.yaml)."""
        inst = cls.get_instance()
        with inst._lock:
            inst.prefer = prefer or "auto"
            inst.allow_network = bool(allow_network)
            if cache_dir:
                inst.cache_dir = Path(cache_dir)
            # сброс выбранного движка при смене предпочтений
            inst._tried_init = False
            inst._engine = None
            inst._model = None
            inst._device = "cpu"

    # ------------------------------------------------------------------
    # Выбор и ленивая загрузка движка
    # ------------------------------------------------------------------
    def _ensure_engine(self):
        if self._tried_init:
            return self._engine
        self._tried_init = True
        if self.prefer == "off":
            logger.info("FormulaRecognizer: выключен настройкой prefer=off")
            return None

        order = ["pix2tex", "texify", "api"]
        if self.prefer in ("pix2tex", "texify", "api"):
            order = [self.prefer] + [o for o in order if o != self.prefer]

        for name in order:
            try:
                if name == "pix2tex":
                    # Пакет ставится как `pip install pix2tex` (движок LaTeX-OCR).
                    # PERF: обычный `pip install torch` тянет CPU-сборку —
                    # распознавание ползёт на ядрах. Ставим CUDA-torch:
                    #   pip install torch --index-url https://download.pytorch.org/whl/cu128
                    # Здесь выбираем устройство и включаем inference_mode/no_grad.
                    try:
                        from pix2tex.cli import LatexOCR
                        from munch import Munch  # зависит от pix2tex[gui]
                    except ImportError:
                        raise  # дальше по order: texify/api
                    try:
                        import torch
                        self._torch = torch
                        device = "cuda" if torch.cuda.is_available() else "cpu"
                        # FIX(4): корень всех прошлых Device Mismatch найден
                        # в исходниках pix2tex 0.1.4 (laup/pike-as/LaTeX-OCR):
                        #   __init__: arguments=None => Munch(no_cuda=True),
                        #   args.device = 'cuda' if cuda.is_available() and
                        #                 not no_cuda else 'cpu'  <-- ВСЕГДА
                        #                 'cpu' при default-аргументах!
                        # Прошлые правки (74c1aa1/927248d) патчили несущест-
                        # вующие атрибуты transformsToTensor/Normalize (это API
                        # старых форков) и делали model.to(cuda) только весам,
                        # тогда как __call__ строит входной тензор albumentations-
                        # трансформом на CPU и относит его по self.args.device
                        # (=cpu) -> "Input type (FloatTensor) and weight type
                        # (cuda.FloatTensor)" на каждой зоне. Лечится ЕДИН-
                        # СТВЕННЫМ правильным способом: конструируем LatexOCR
                        # с явным Munch(no_cuda=False) — модель загружается
                        # map_location=cuda СРАЗУ целиком (и генератор, и
                        # image_resizer), args.device становится 'cuda', и
                        # __call__ переносит входы туда же. Никаких ручных
                        # .to() больше не нужно.
                        try:
                            self._model = LatexOCR(Munch(
                                {"config": "settings/config.yaml",
                                 "checkpoint": "checkpoints/weights.pth",
                                 "no_cuda": device != "cpu",
                                 "no_resize": False}))
                        except TypeError:
                            # форк с другой сигнатурой — деградация на old API
                            self._model = LatexOCR()
                        # Синхронизируем аргументы, если конструктор всё же
                        # выбрал не то устройство (защита от форков).
                        margs = getattr(self._model, "args", None)
                        actual = str(getattr(margs, "device", "") or "")
                        if actual.split(":")[0] != device:
                            model = getattr(self._model, "model", None)
                            if model is not None:
                                model.to(device).eval()
                            resizer = getattr(self._model, "image_resizer",
                                              None)
                            if resizer is not None:
                                resizer.to(device).eval()
                            if margs is not None:
                                try:
                                    margs.device = device
                                except Exception:  # noqa: BLE001
                                    pass
                        # Совместимость со старыми форками (если атрибуты
                        # есть — приводим их к фактическому устройству весов;
                        # в 0.1.4 их нет, цикл просто ничего не трогает).
                        wdev = self._weights_device(torch) or device
                        for attr in ("transformsToTensor",
                                     "transformsNormalize"):
                            tr = getattr(self._model, attr, None)
                            td = getattr(tr, "device", None)
                            if td is not None and str(td) != str(wdev):
                                try:
                                    tr.device = wdev
                                except Exception:  # noqa: BLE001
                                    pass
                        self._device = device
                        if device == "cpu":
                            torch.set_num_threads(
                                max(1, (os.cpu_count() or 4) // 2))
                            logger.info(
                                "FormulaRecognizer: torch в CPU-режиме "
                                "(медленно). Для RTX 5060 переустановите torch "
                                "с CUDA: pip install torch --index-url "
                                "https://download.pytorch.org/whl/cu128")
                        else:
                            logger.info(
                                "FormulaRecognizer: pix2tex на GPU (CUDA), "
                                f"args.device={getattr(margs, 'device', '?')}")
                    except ImportError:
                        self._torch = None
                    self._engine = "pix2tex"
                    # HEADLESS: pix2tex 0.1.4 в конце LatexOCR.__call__ делает
                    # clipboard.copy(pred) (pandas.io.clipboard) — на Windows
                    # это OpenClipboard/SetClipboardData, т.е. после КАЖДОЙ
                    # распознанной формулы буфер обмена пользователя трогается
                    # и засоряется («J\vert\mathrm{e}\tilde...»). Нейтрализуем
                    # именно эту запись (результат — в наш RECENT_TEX), саму
                    # модель/инференс не трогаем.
                    try:
                        from ..core.win_headless import neutralize_clipboard_write
                        if neutralize_clipboard_write(self._model):
                            logger.debug(
                                "FormulaRecognizer: запись в clipboard "
                                "нейтрализована (headless)")
                    except Exception as _cb_e:  # noqa: BLE001 — не валит движок
                        logger.debug(f"win_headless: clipboard-патч пропущен: {_cb_e}")
                    logger.info("FormulaRecognizer: движок pix2tex (LaTeX-OCR) готов")
                    return self._engine
                if name == "texify":
                    import torch
                    from texify.model.model import DocumentModel
                    from texify.processor import TexifyProcessor
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                    kwargs = {}
                    if device == "cuda":
                        # PERF: bf16-веса вдвое меньше и быстрее на RTX;
                        # fp32 на 8 ГБ VRAM 5060 может не влезть (CPU-offload).
                        kwargs = dict(torch_dtype=torch.bfloat16)
                    self._model = DocumentModel.from_pretrained(
                        "winvivo/Texify-full", **kwargs).to(device).eval()
                    self._model.processor = TexifyProcessor.from_pretrained(
                        "winvivo/Texify-full")
                    self._model.device = device
                    self._engine = "texify"
                    logger.info("FormulaRecognizer: движок texify готов")
                    return self._engine
                if name == "api":
                    if not self.allow_network:
                        continue
                    import requests  # уже в стеке
                    self._model = requests
                    self._engine = "api"
                    logger.info("FormulaRecognizer: движок pix2tex-API (сеть)")
                    return self._engine
            except ImportError:
                logger.debug(f"FormulaRecognizer: движок '{name}' не установлен")
            except Exception as e:  # noqa: BLE001 — пробуем следующий движок
                logger.warning(f"FormulaRecognizer: движок '{name}' не поднялся: {e}")
        logger.info("FormulaRecognizer: движки недоступны — формулы останутся "
                    "как lazy-PNG без текста (установите `pix2tex` или `texify`)")
        return None

    # ------------------------------------------------------------------
    # Публичный API
    # ------------------------------------------------------------------
    def recognize(self, image_path: str | Path, dpi_hint: int = 300) -> Optional[str]:
        """PNG зоны формулы -> LaTeX-строка (без ограничителей $).

        Возвращает None, если движок недоступен или результат недостоверен.
        Результат кэшируется по содержимому файла (переинпорт того же PDF
        не запускает инференс заново).
        """
        path = Path(image_path)
        if not path.exists():
            return None
        try:
            data = path.read_bytes()
        except OSError:
            return None
        key = hashlib.sha1(data).hexdigest()
        engine = None
        with self._lock:
            if key in self._png_cache:
                return self._png_cache[key] or None
            engine = self._ensure_engine()
            if engine is None:
                return None
        # Инференс — ВНЕ _lock (загрузка весов и вывод модели длятся долго;
        # сериализуем его отдельным _infer_lock).
        latex = None
        try:
            if engine == "api":
                latex = self._recognize_api(data)
            else:
                latex = self._recognize_local(data)
        except Exception as e:  # noqa: BLE001 — одна битая зона не валит импорт
            logger.debug(f"FormulaRecognizer ({engine}): сбой на {path.name}: {e}")
        with self._lock:
            if latex and self._looks_useful(latex):
                latex = latex.strip()
                self._png_cache[key] = latex
                return latex
            self._png_cache[key] = ""
            return None

    def _recognize_local(self, data: bytes) -> Optional[str]:
        import io
        from PIL import Image  # уже в стеке (ImageGrab нами заглушён — Image не трогаем)
        img = Image.open(io.BytesIO(data)).convert("RGB")
        # УМНЫЙ РЕНДЕР (рубеж 2): пустые/вырожденные PNG нулевого содержимого
        # («height and width must be > 0» на p167_b1.png и т.п.) — это зоны с
        # некорректными координатами. Инференс по ним = пустая работа GPU/CPU;
        # отбраковываем до загрузки модели.
        w, h = img.size
        if w <= 8 or h <= 8:
            return None
        try:
            extrema = img.getextrema()  # ((min,max),)*3 для RGB
            if all(mn == mx for mn, mx in extrema):
                return None  # однотонное изображение — формулы нет
        except Exception:  # noqa: BLE001
            pass
        with self._infer_lock:
            if self._engine == "pix2tex":
                # HEADLESS-страховка: если движок поднялся до патча main.py
                # (например, воркер импорта стартовал раньше UI), clipboard.copy
                # всё равно не должен трогать буфер Windows.
                try:
                    import pix2tex.cli as _p2t
                    if getattr(_p2t.clipboard, "copy", None) is not None \
                            and type(_p2t.clipboard).__name__ != "ClipboardStub":
                        from ..core.win_headless import neutralize_clipboard_write
                        neutralize_clipboard_write(None)
                except ImportError:
                    pass
                torch = getattr(self, "_torch", None)
                if torch is not None:
                    # FIX(3b): последний рубеж против рассинхрона устройств.
                    # В pix2tex 0.1.4 входной тензор относится по
                    # self.args.device (см. cli.LatexOCR.__call__:
                    # `im = t.to(self.args.device)`), а НЕ по атрибутам
                    # transforms* (их там нет — это API старых форков).
                    # Поэтому сверяем args.device с фактическим расположением
                    # весов и мол чиним его, пока инференс не упал со
                    # "FloatTensor vs cuda.FloatTensor".
                    wdev = self._weights_device(torch)
                    margs = getattr(self._model, "args", None)
                    if wdev is not None and margs is not None:
                        try:
                            ad = str(getattr(margs, "device", "") or "")
                            if ad.split(":")[0] != str(wdev).split(":")[0]:
                                margs.device = str(wdev)
                        except Exception:  # noqa: BLE001
                            pass
                    if wdev and str(wdev).startswith("cuda"):
                        for attr in ("transformsToTensor",
                                     "transformsNormalize"):
                            try:
                                tr = getattr(self._model, attr, None)
                                if tr is not None and str(
                                        getattr(tr, "device", "cpu")) != str(wdev):
                                    tr.device = wdev
                            except Exception:  # noqa: BLE001
                                pass
                    # PERF: инференс без градиентов ~в 2 раза быстрее и
                    # экономит память активаций.
                    with torch.inference_mode():
                        out = self._model(img)
                    if isinstance(out, torch.Tensor) and wdev \
                            and out.device != torch.device(str(wdev)):
                        out = out.to(wdev)
                else:
                    out = self._model(img)
            else:  # texify
                inputs = self._model.processor(images=img, return_tensors="pt")
                inputs = {k: v.to(self._model.device) for k, v in inputs.items()}
                prediction = self._model.generate(inputs, decode=True,
                                                  post_process=True)[0]
                out = re.sub(r"<\|.*?\|>", "", str(prediction))
                out = out.replace("\n", " ")
        return out.strip() if isinstance(out, str) else None

    # ------------------------------------------------------------------
    # Вспомогательные
    # ------------------------------------------------------------------
    def _weights_device(self, torch_mod):
        """Фактическое устройство весов модели (первый параметр/буфер)."""
        model = getattr(self._model, "model", None)
        if model is None or not hasattr(model, "parameters"):
            return None
        try:
            for p in model.parameters():
                return p.device
        except StopIteration:
            pass
        except Exception:  # noqa: BLE001
            pass
        try:
            for b in model.buffers():
                return b.device
        except Exception:  # noqa: BLE001
            pass
        return None

    def _recognize_api(self, data: bytes) -> Optional[str]:
        """Сетевой фолбэк: тот же pix2tex, но без локальных весов."""
        import base64, json
        b64 = base64.b64encode(data).decode()
        resp = self._model.post(self.PIX2TEX_API_URL, timeout=30,
                                json={"image": b64, "strict": False})
        resp.raise_for_status()
        return json.loads(resp.text).get("latex")

    @staticmethod
    def _looks_useful(latex: str) -> bool:
        """Отсев галлюцинаций/пустых распознований до вставки в корпус."""
        t = (latex or "").strip()
        if len(t) < _MIN_USEFUL_LEN:
            return False
        # должен содержать хотя бы один «математический» токен
        return bool(_LATEX_CMD_RE.search(t))


def get_formula_recognizer() -> FormulaRecognizer:
    """Точка доступа для pdf_parser / ingestion-пайплайна."""
    return FormulaRecognizer.get_instance()
