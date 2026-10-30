# src/ui/pdf_viewer.py
"""
Встроенный PDF-просмотрщик (PyMuPDF → PNG).

РЕФАКТОРИНГ (адаптивный layout по ТЗ):
1. Панель просмотра больше не видима всегда — её показом/скрытием управляет
   MainLayout (src/ui/layout.py) через флаги visible + page.update().
2. АУДИТ: ft.Image со src="" / src=None создавался на старте приложения и
   давал «A valid src value must be specified» в пустом состоянии. Теперь
   image_control НЕ существует до первого set_document(): пока документ не
   открыт, в панели лежит текстовый статус (placeholder). Компонент с путём
   инициализируется только когда реальный PNG уже отрендерен.
3. Убран мёртвый импорт SQLiteStore: путь передаётся через
   set_document(doc_id, path).
4. Навигация использует navigate_to_page() (пересчёт + update).
5. Переход по ссылке |page:N| из чата/оглавления — navigate_to_page(N).
6. LRU-кэш снимков страниц ограничен MAX_CACHED_PAGES.
7. EPUB: сообщение + кнопка «Открыть во внешней программе».
8. close_document() сбрасывает весь UI в исходное состояние (заглушка),
   чтобы после крестика «Закрыть просмотр» панель не показывала старый кадр.
"""
import subprocess
import sys
import webbrowser
from pathlib import Path
from typing import Dict, Optional

import flet as ft
from loguru import logger

try:
    import pymupdf
except ImportError:
    pymupdf = None
    logger.warning("PyMuPDF не установлен — PDF-просмотрщик будет недоступен")


MAX_CACHED_PAGES = 60


class PdfViewerView:
    def __init__(self, page: ft.Page):
        self.page = page
        self.current_doc: Optional["pymupdf.Document"] = None
        self.current_page_number: int = 1
        self.total_pages: int = 0
        self.zoom: float = 1.5
        self.pdf_path: Optional[Path] = None
        self.doc_id: Optional[str] = None
        self._image_cache: Dict[int, str] = {}

        # --- Панель инструментов ---
        self.page_input = ft.TextField(
            value="1", width=70, text_align="center", dense=True,
            on_submit=self._on_page_input_submit,
        )
        self.page_label = ft.Text("/ 0", size=12, color="grey")
        self.file_label = ft.Text("Документ не выбран", size=12,
                                  no_wrap=True, overflow=ft.TextOverflow.ELLIPSIS,
                                  expand=True)

        self.toolbar = ft.Row(
            [
                ft.IconButton(ft.Icons.ARROW_BACK, tooltip="Назад",
                              on_click=lambda e: self.navigate_to_page(self.current_page_number - 1)),
                self.page_input,
                self.page_label,
                ft.IconButton(ft.Icons.ARROW_FORWARD, tooltip="Вперёд",
                              on_click=lambda e: self.navigate_to_page(self.current_page_number + 1)),
                ft.IconButton(ft.Icons.ZOOM_OUT, tooltip="Уменьшить",
                              on_click=lambda e: self._change_zoom(-0.25)),
                ft.IconButton(ft.Icons.ZOOM_IN, tooltip="Увеличить",
                              on_click=lambda e: self._change_zoom(0.25)),
                ft.IconButton(ft.Icons.OPEN_IN_NEW, tooltip="Открыть внешне",
                              on_click=lambda e: self._open_externally()),
            ],
            alignment=ft.MainAxisAlignment.CENTER,
            spacing=4,
        )

        # Статус/заглушка для пустого состояния и сообщений об ошибках.
        self.status_text = ft.Text("", size=12, color="grey")

        # АУДИТ: image_control_late — ленивая инициализация ft.Image.
        # Контрол появляется ТОЛЬКО после того, как pixmap реально сохранён
        # на диск; пустой src никогда не попадает в рендер-дерево.
        self.image_control: Optional[ft.Image] = None

        self.stage = ft.Container(
            content=self._make_placeholder("Документ не выбран. Откройте книгу из библиотеки."),
            expand=True,
            alignment=ft.Alignment.TOP_CENTER,
            bgcolor=ft.Colors.GREY_300,
            border_radius=8,
            padding=10,
        )

        self.viewer_column = ft.Column(
            [
                ft.Row([self.file_label], spacing=5),
                self.toolbar,
                self.stage,
                ft.Row([self.status_text], alignment=ft.MainAxisAlignment.CENTER),
            ],
            expand=True,
            spacing=8,
        )

    def build(self) -> ft.Control:
        # Контейнер без padding — внешний отступ задаёт MainLayout.viewer_container,
        # иначе двойные отступы «съедают» 60% ширины.
        return ft.Container(content=self.viewer_column, expand=True)

    # ------------------------------------------------------------------
    # Управление документом
    # ------------------------------------------------------------------

    def set_document(self, doc_id: Optional[str], file_path: str, title: Optional[str] = None):
        """Загружает документ (PDF) для просмотра."""
        self.close_document()
        path = Path(file_path)
        self.file_label.value = title or path.name
        if not path.exists():
            self._show_message(f"Файл не найден: {path.name}")
            return

        if path.suffix.lower() == ".epub":
            self.pdf_path = path
            self.doc_id = doc_id
            self._show_message(
                "EPUB во встроенном просмотрщике не открывается.\n"
                "Используйте кнопку «Открыть внешне»."
            )
            try:
                webbrowser.open(path.as_uri())
            except Exception:
                pass
            return

        if pymupdf is None:
            self._show_message("PyMuPDF не установлен — просмотр недоступен.")
            return

        try:
            self.current_doc = pymupdf.open(str(path))
            self.pdf_path = path
            self.doc_id = doc_id
            self.total_pages = self.current_doc.page_count
            self.page_label.value = f"/ {self.total_pages}"
            self._render_current_page()
        except Exception as e:
            logger.error(f"Ошибка открытия PDF: {e}")
            self._show_message(f"Не удалось открыть файл: {e}")

    def close_document(self):
        """Полный сброс просмотрщика в исходное (пустое) состояние."""
        if self.current_doc:
            try:
                self.current_doc.close()
            except Exception:
                pass
        self.current_doc = None
        self._discard_image()
        self._image_cache.clear()
        self.pdf_path = None
        self.doc_id = None
        self.current_page_number = 1
        self.total_pages = 0
        self.page_input.value = "1"
        self.page_label.value = "/ 0"
        self.file_label.value = "Документ не выбран"
        self.status_text.value = ""
        self.stage.content = self._make_placeholder(
            "Документ не выбран. Откройте книгу из библиотеки.")

    # ------------------------------------------------------------------
    # Навигация
    # ------------------------------------------------------------------

    def navigate_to_page(self, page_number: int):
        """Публичный переход к странице (используется чатом и оглавлением)."""
        if not self.current_doc:
            return
        page_number = max(1, min(int(page_number), self.total_pages))
        self.current_page_number = page_number
        self.page_input.value = str(page_number)
        self._render_current_page()
        try:
            self.page.update()
        except Exception:
            pass

    def _on_page_input_submit(self, e):
        try:
            self.navigate_to_page(int((self.page_input.value or "1").strip()))
        except ValueError:
            self.page_input.value = str(self.current_page_number)
            try:
                self.page.update()
            except Exception:
                pass

    def _change_zoom(self, delta: float):
        self.zoom = max(0.5, min(4.0, round(self.zoom + delta, 2)))
        self._image_cache.clear()  # кэш снимков зависит от масштаба
        self._render_current_page()
        try:
            self.page.update()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Рендеринг
    # ------------------------------------------------------------------

    def _render_current_page(self):
        if not self.current_doc:
            self._show_message("Документ не выбран. Откройте книгу из библиотеки.")
            return

        page_number = self.current_page_number
        cached = self._image_cache.get(page_number)
        if cached and Path(cached).exists():
            self._set_image_src(cached)
            self._set_status()
            return

        try:
            pixmap = self.current_doc[page_number - 1].get_pixmap(
                matrix=pymupdf.Matrix(self.zoom, self.zoom))
            image_path = str(self.pdf_path.parent /
                             f".page_{page_number}_{int(self.zoom * 100)}.png")
            pixmap.save(image_path)

            # Ограниченный LRU-кэш.
            if len(self._image_cache) >= MAX_CACHED_PAGES:
                for old_key in list(self._image_cache.keys())[:10]:
                    old_path = self._image_cache.pop(old_key)
                    try:
                        Path(old_path).unlink(missing_ok=True)
                    except OSError:
                        pass
            self._image_cache[page_number] = image_path

            self._set_image_src(image_path)
            self._set_status()
        except Exception as e:
            logger.error(f"Ошибка рендеринга страницы {page_number}: {e}")
            self._show_message(f"Ошибка отображения страницы: {e}")

    # --- Ленивая инициализация Image (см. п.2 шапки модуля) ---------------

    def _set_image_src(self, path: str):
        if self.image_control is None:
            self.image_control = ft.Image(
                src=path, fit=ft.BoxFit.CONTAIN, expand=True)
            self.stage.content = self.image_control
        else:
            self.image_control.src = path
            # Если раньше в stage был текст-сообщение — возвращаем картинку.
            if self.stage.content is not self.image_control:
                self.stage.content = self.image_control

    def _discard_image(self):
        self.image_control = None

    def _set_status(self):
        self.status_text.value = (f"Стр. {self.current_page_number} из {self.total_pages} | "
                                  f"масштаб {int(self.zoom * 100)}%")

    def _show_message(self, message: str):
        """Сообщение без создания Image — безопасно для пустого состояния."""
        self._discard_image()
        self.stage.content = self._make_placeholder(message)
        self.status_text.value = message

    @staticmethod
    def _make_placeholder(text: str) -> ft.Control:
        return ft.Column(
            [
                ft.Icon(ft.Icons.DESCRIPTION_OUTLINED, size=48, color="grey"),
                ft.Text(text, size=14, color="grey", selectable=True),
            ],
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            alignment=ft.MainAxisAlignment.CENTER,
            spacing=8,
            scroll=ft.ScrollMode.AUTO,
            expand=True,
        )

    def _open_externally(self):
        if not self.pdf_path:
            return
        try:
            if sys.platform.startswith("win"):
                import os
                os.startfile(str(self.pdf_path))  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(self.pdf_path)])
            else:
                subprocess.Popen(["xdg-open", str(self.pdf_path)])
        except Exception as e:
            logger.error(f"Не удалось открыть внешний просмотрщик: {e}")
            self._show_message(f"Не удалось открыть файл: {e}")
