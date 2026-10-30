# src/core/win_headless.py
"""Изоляция фоновых процессов (page_renderer / pix2tex / LaTeX) от оконной
системы Windows.

ПРОБЛЕМА (подтверждена исходниками pix2tex 0.1.4, скачанными для аудита):
после КАЖДОГО инференса формулы `pix2tex.cli.LatexOCR.__call__` выполняет
`clipboard.copy(pred)` (модуль `pandas.io.clipboard`), а при импорте модуля
`cli` — `from PIL import ImageGrab`. На Windows это означает, что рабочий
процесс импорта 714 страниц постоянно трогает оконную подсистему:

  • pandas `init_windows_clipboard()` делает OpenClipboard/EmptyClipboard/
    SetClipboardData — т.е. после каждой распознанной формулы пользовательский
    буфер обмена ПЕРЕТИРАЕТСЯ («J\\vert\\mathrm{e}\\tilde...» — ровно этот
    эффект) и перехватывается Windows Clipboard History (Win+V);
  • PyQt6-бэкенд pix2tex[gui] и QApplication могут создавать скрытые окна;
  • ImageGrab.grabclipboard() читает системный буфер обратно.

РЕШЕНИЕ (ничего ранее созданного не ломает; деградация безопасна):
  1. `install_gui_block()` — вызывается в main.py ДО импорта torch/pix2tex:
     QT_QPA_PLATFORM=offscreen (Qt не рисует окон), заглушки PIL.ImageGrab,
     принудительный выбор clip.exe-пути для pandas/pyperclip.
  2. `neutralize_clipboard_write()` — точечно патчит `clipboard.copy` у уже
     импортированного LatexOCR (пишем результат в наш внутренний ring-буфер
     вместо системного). Вызывается из formula_recognizer после загрузки движка.
  3. `guard_thread()` / `is_guarded()` — контекст фонового потока: всё, что
     случайно полезет в Tk (диалоги), получает понятное исключение вместо окна.

Все хуки идемпотентны и no-op на не-Windows.
"""
from __future__ import annotations

import contextlib
import os
import sys
import threading
from collections import deque
from typing import Deque, Iterator, Optional

IS_WINDOWS = sys.platform == "win32"

# Поток, которому разрешено создавать окна (UI-поток Flet).
_ui_thread_id: Optional[int] = None
_installed = False
_clipboard_neutralized = False

# Наш внутренний «буфер распознаваний»: последние N результатов pix2tex.
# Заменяет системный clipboard как источник для отладки/UI-экспорта.
RECENT_TEX: Deque[str] = deque(maxlen=50)


def set_ui_thread() -> None:
    """Текущий поток объявляется единственным, кому можно трогать окна."""
    global _ui_thread_id
    _ui_thread_id = threading.get_ident()


def current_thread_is_ui() -> bool:
    return _ui_thread_id is None or threading.get_ident() == _ui_thread_id


# ---------------------------------------------------------------------------
# Глобальные флаги и заглушки (до импорта torch/pix2tex!)
# ---------------------------------------------------------------------------

def install_gui_block() -> None:
    """Настраивает окружение так, чтобы библиотечные цепочки не трогали окна.

    Идемпотентно. Безопасно вызывать на не-Windows (просто выставит env-флаги).
    """
    global _installed
    if _installed:
        return
    _installed = True
    set_ui_thread()

    # 1) Qt (pix2tex[gui] => PyQt6.QtWebEngineWidgets и пр.): offscreen-платформа
    #    не создаёт ни одного окна и не требует дисплея.
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    # 2) Сигнал нашему коду: диалоги Tk из фоновых потоков запрещены.
    os.environ.setdefault("LAMA_HEADLESS_WORKERS", "1")
    # 3) pandas.io.clipboard / pyperclip: на Windows они сначала пробуют тяжёлые
    #    GUI-бэкенды (PyQt) и лишь потом clip.exe. PYPERCLIP_* позволяет выбрать
    #    CLIPBOARD_BACKEND прямо (pyperclip>=1.8.2), остальное — страховка.
    os.environ.setdefault("PYPERCLIP_CLIPBOARD_BACKEND", "CLIP.EXE")
    os.environ.setdefault("PANDAS_IO_CLIPBOARD_BACKEND", "CLIP.EXE")

    _stub_imagegrab()

    try:
        from loguru import logger
        logger.debug("win_headless: изоляция окон включена "
                     f"(QT_QPA_PLATFORM={os.environ['QT_QPA_PLATFORM']}, "
                     f"ImageGrab={'заглушен' if _STUBBED_OK else 'нет'})")
    except Exception:  # noqa: BLE001 — логгер опционален на раннем старте
        pass


_STUBBED_OK = False


def _stub_imagegrab() -> None:
    """Гасим PIL.ImageGrab до того, как его импортирует pix2tex.cli.

    `from PIL import ImageGrab` в cli.py выполняется на импорте модуля; сам
    grabclipboard() вызывается только в интерактивном режиме, но сам факт
    импорта на Windows тянет win32-цепочки. Подсовываем безобидную заглушку в
    sys.modules — реальный PIL не повреждается (мы не трогаем Image/ImageDraw).
    """
    global _STUBBED_OK
    try:
        import types
        stub = types.ModuleType("PIL.ImageGrab")

        def grabclipboard(*a, **kw):   # noqa: D401
            return None

        def grab(*a, **kw):
            raise RuntimeError("PIL.ImageGrab отключён (headless-режим lama-loca)")

        stub.grabclipboard = grabclipboard
        stub.grab = grab
        stub.__doc__ = "lama-loca win_headless stub"
        sys.modules["PIL.ImageGrab"] = stub
        try:
            import PIL  # если уже импортирован — тоже перекрываем атрибут
            setattr(PIL, "ImageGrab", stub)
        except Exception:  # noqa: BLE001
            pass
        _STUBBED_OK = True
    except Exception:  # noqa: BLE001 — не критично
        _STUBBED_OK = False


# ---------------------------------------------------------------------------
# Точечная нейтрализация записи в clipboard у pix2tex
# ---------------------------------------------------------------------------

def neutralize_clipboard_write(model) -> bool:
    """Отключаем `clipboard.copy(pred)` внутри LatexOCR.__call__.

    Метод __call__ pix2tex 0.1.4 в конце каждого инференса вызывает
    `clipboard.copy(pred)` (именованный импорт модуля cli). Патчим именно
    ссылку в словаре глобальных уже загруженного модуля — минимально
    инвазивно, сеть-движок и формат ответа не меняются. Результат складываем
    в RECENT_TEX (наш внутренний буфер). Возвращает True, если патч применён.
    """
    global _clipboard_neutralized
    if _clipboard_neutralized:
        return True
    try:
        import pix2tex.cli as p2t_cli  # уже импортирован (движок готов)
    except Exception:  # noqa: BLE001 — движок другой (texify/api), нечего патчить
        return False
    orig_copy = getattr(p2t_cli.clipboard, "copy", None)
    if orig_copy is None:
        return False

    def _silent_copy(text, *_a, **_kw):
        try:
            RECENT_TEX.append(str(text))
        except Exception:  # noqa: BLE001
            pass
        return None  # системный clipboard НЕ трогаем

    try:
        p2t_cli.clipboard = type("ClipboardStub", (), {
            "copy": staticmethod(_silent_copy),
            "paste": staticmethod(lambda *_a, **_kw: ""),
        })()
        _clipboard_neutralized = True
        try:
            from loguru import logger
            logger.debug("win_headless: запись pix2tex в clipboard отключена "
                         "(нейтрализован pandas.io.clipboard.copy)")
        except Exception:  # noqa: BLE001
            pass
        return True
    except Exception:  # noqa: BLE001
        # Фолбэк: хотя бы восстановим буфер пользователя после записи?
        # Нет — оставляем как есть; повторная попытка при следующем configure().
        return False


def restore_user_clipboard_state() -> None:  # pragma: no cover - не используется
    """Хук на будущее: если когда-нибудь понадобится вежливый save/restore."""
    return None


# ---------------------------------------------------------------------------
# Guard-контекст фоновых потоков (Tk/окна запрещены)
# ---------------------------------------------------------------------------

_GUARD_DEPTH = threading.local()


def _in_guard() -> bool:
    return getattr(_GUARD_DEPTH, "depth", 0) > 0


@contextlib.contextmanager
def guard_thread() -> Iterator[None]:
    """Фоновый поток импорта: любые попытки открыть Tk-окно дают исключение."""
    _GUARD_DEPTH.depth = getattr(_GUARD_DEPTH, "depth", 0) + 1
    try:
        yield
    finally:
        _GUARD_DEPTH.depth -= 1


def is_guarded() -> bool:
    return _in_guard()


class WindowCreationBlocked(RuntimeError):
    """Попытка создать окно из защищённого фонового потока."""


# ---------------------------------------------------------------------------
# Tkinter-диалоги: только из UI-потока, root скрыт, без topmost
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def tk_root_hidden():
    """Tk root без видимого окна (для filedialog из UI-потока)."""
    if _in_guard():
        raise WindowCreationBlocked(
            "tkinter root запрещён в фоновом потоке импорта")
    root = None
    try:
        import tkinter
        root = tkinter.Tk()
        try:
            root.withdraw()  # пустое окно не показываем
        except Exception:  # noqa: BLE001
            pass
        yield root
    finally:
        if root is not None:
            try:
                root.destroy()
            except Exception:  # noqa: BLE001
                pass


def run_tk_dialog(dialog_fn):
    """Выполнить tkinter-диалог, не трогая окна из фоновых потоков.

    dialog_fn: callable(tkinter_module) -> результат. Возвращает None, если
    диалог невозможен (фоновый поток / нет tkinter / отмена).
    """
    if _in_guard() or not current_thread_is_ui():
        return None
    try:
        import tkinter
    except Exception:  # noqa: BLE001
        return None
    with tk_root_hidden():
        try:
            return dialog_fn(tkinter)
        except Exception:  # noqa: BLE001
            return None
