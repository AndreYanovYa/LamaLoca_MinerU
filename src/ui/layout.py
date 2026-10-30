# src/ui/layout.py
"""
Каркас главного окна Lama Loca (Flet 1.x) — рефакторинг по ТЗ «адаптивный layout».

Структура:
    Row
    ├── NavigationRail (левая панель, фикс. ширина SIDEBAR_WIDTH) — глобальный
    │     переключатель контекста: «Чат» / «Библиотека».
    └── main_area (Column)
          ├── chat_container   — вкладка чата; при активном чате занимает ВСЁ
          │                      оставшееся пространство (expand=True).
          └── library_row      — вкладка библиотеки; изначально visible=False.
                 ├── library_container (40%) — список книг и поиск;
                 └── viewer_container  (60%) — панель просмотра документа,
                        ИЗНАЧАЛЬНО СКРЫТА (visible=False), открывается только
                        по целевому действию пользователя (клик по книге,
                        переход к главе из оглавления, клик по источнику в
                        чате) и закрывается кнопкой-крестиком.

Принципы (технические требования ТЗ):
* Переключение вида/панелей — только флагами `visible` + `page.update()`.
* Компонент просмотра не инициализируется с пустым путём: ft.Image создаётся
  лениво внутри PdfViewerView.set_document(); до открытия документа в панели
  лежит заглушка-текст, поэтому ошибка «A valid src value must be specified»
  невозможна ни в одном состоянии.
* Цветовая палитра и стили компонентов сохранены (#121212 фон, #2196F3 акцент,
  светлая #FAFAFA тема библиотеки, тёмная тема чата).
"""
from __future__ import annotations

from typing import Callable, Optional

import flet as ft
from loguru import logger

SIDEBAR_WIDTH = 84          # узкая фиксированная панель навигации
LIBRARY_SPLIT = 40          # % на список книг
VIEWER_SPLIT = 60           # % на панель документа


class MainLayout:
    """Собирает sidebar + переключаемые области и управляет их видимостью."""

    def __init__(self, page: ft.Page):
        self.page = page

        # Содержимое внедряется из app.py (LamaLocaApp._build).
        self.chat_content: Optional[ft.Control] = None
        self.library_content: Optional[ft.Control] = None
        self.viewer_content: Optional[ft.Control] = None
        self.settings_content: Optional[ft.Control] = None

        # Навигационные колбэки (опциональны).
        self.on_viewer_closed: Optional[Callable[[], None]] = None
        self.on_view_changed: Optional[Callable[[int], None]] = None

        # --- Контейнеры (создаются один раз, дальше только флаги visible) ---
        self.chat_container = ft.Container(
            content=self._placeholder("Чат загружается…"),
            expand=True,
            visible=False,
            padding=10,
        )

        # АУДИТ: раньше Image со src="" создавался на старте приложения — это
        # и давало «A valid src value must be specified» при пустом состоянии.
        # Теперь до set_document() показываем только текст-заглушку.
        self.viewer_body = ft.Container(
            content=self._placeholder("Документ не выбран."),
            expand=True,
        )

        self.viewer_header = ft.Row(
            [
                ft.Text("📄 Просмотр документа", size=14,
                        weight=ft.FontWeight.BOLD, expand=True,
                        no_wrap=True, overflow=ft.TextOverflow.ELLIPSIS),
                ft.IconButton(
                    icon=ft.Icons.CLOSE,
                    tooltip="Закрыть просмотр",
                    icon_color=ft.Colors.RED_400,
                    on_click=self._on_close_viewer,
                ),
            ],
            spacing=4,
        )

        self.viewer_panel = ft.Column(
            [self.viewer_header, self.viewer_body],
            expand=True,
            spacing=8,
        )

        self.viewer_container = ft.Container(
            content=self.viewer_panel,
            col=VIEWER_SPLIT,
            visible=False,          # ИЗНАЧАЛЬНО СКРЫТА по ТЗ
            bgcolor="#1E1E1E",
            # АУДИТ Flet 1.0.5: модуль ft.border не экспортирует статические методы
            # Border.only/all/symmetric — вызываем класс ft.Border напрямую.
            border=ft.Border.only(left=ft.BorderSide(1, "#333333")),
            padding=10,
        )

        self.library_container = ft.Container(
            content=self._placeholder("Библиотека загружается…"),
            col=LIBRARY_SPLIT,
            expand=True,
            bgcolor="#FAFAFA",      # светлая тема для библиотеки — по ТЗ
            padding=10,
        )

        self.library_row = ft.Row(
            [self.library_container, self.viewer_container],
            expand=True,
            spacing=0,
            visible=False,          # библиотека скрыта, пока не выбрана вкладка
        )

        # ТЗ п.4: контейнер вкладки «Настройки» (Ф.И.О. студента). Светлая тема —
        # как у библиотеки; видимость только по флагу (принципы layout сохранены).
        self.settings_container = ft.Container(
            content=self._placeholder("Настройки загружаются…"),
            expand=True,
            visible=False,
            bgcolor="#FAFAFA",
            padding=10,
        )

        self.main_area = ft.Column(
            [self.chat_container, self.library_row, self.settings_container],
            expand=True,
            spacing=0,
        )

        self.rail = ft.NavigationRail(
            selected_index=0,
            label_type=ft.NavigationRailLabelType.ALL,
            min_width=SIDEBAR_WIDTH,
            group_alignment=-0.95,
            leading=ft.Container(ft.Icon(ft.Icons.SCHOOL, size=36,
                                          color="#2196F3"), padding=10),
            indicator_color="#2196F3",
            bgcolor="#1A1A1A",
            destinations=[
                ft.NavigationRailDestination(
                    icon=ft.Icons.CHAT_BUBBLE_OUTLINE,
                    selected_icon=ft.Icons.CHAT_BUBBLE,
                    label="Чат",
                ),
                ft.NavigationRailDestination(
                    icon=ft.Icons.MENU_BOOK,
                    selected_icon=ft.Icons.LIBRARY_BOOKS,
                    label="Библиотека",
                ),
                # ТЗ п.4: доступ к форме настроек (Ф.И.О. студента).
                ft.NavigationRailDestination(
                    icon=ft.Icons.SETTINGS_OUTLINED,
                    selected_icon=ft.Icons.SETTINGS,
                    label="Настройки",
                ),
            ],
            on_change=self._on_nav_change,
        )

        self.root = ft.Row(
            [
                self.rail,
                ft.VerticalDivider(width=1),
                self.main_area,
            ],
            expand=True,
            spacing=0,
        )

    # ------------------------------------------------------------------
    # Внедрение контента (вызывается один раз из app.py)
    # ------------------------------------------------------------------

    def attach(self, chat: ft.Control, library: ft.Control, viewer: ft.Control,
               settings: Optional[ft.Control] = None):
        """Кладёт готовые вьюхи в контейнеры каркаса."""
        self.chat_content = chat
        self.library_content = library
        self.viewer_content = viewer
        self.settings_content = settings
        self.chat_container.content = chat
        self.library_container.content = library
        self.viewer_body.content = viewer
        if settings is not None:
            self.settings_container.content = settings

    def mount(self):
        """Добавляет каркас на страницу и применяет начальное состояние."""
        self.page.add(self.root)
        self._apply_view(0)

    # ------------------------------------------------------------------
    # Управление видимостью (только флаги + update — по ТЗ)
    # ------------------------------------------------------------------

    def _on_nav_change(self, e):
        idx = int(getattr(e.control, "selected_index", 0) or 0)
        self._apply_view(idx)
        if self.on_view_changed:
            try:
                self.on_view_changed(idx)
            except Exception as ex:  # pragma: no cover
                logger.warning(f"on_view_changed: {ex}")

    def _apply_view(self, index: int):
        """index 0 → чат на всё пространство; 1 → библиотека; 2 → настройки."""
        self.chat_container.visible = (index == 0)
        self.library_row.visible = (index == 1)
        # ТЗ п.4: третья вкладка — форма настроек (Ф.И.О. студента).
        self.settings_container.visible = (index == 2)
        self._safe_update()

    def show_chat(self):
        self.rail.selected_index = 0
        self._apply_view(0)

    def show_library(self):
        self.rail.selected_index = 1
        self._apply_view(1)

    def show_settings(self):
        self.rail.selected_index = 2
        self._apply_view(2)

    # --- Панель документа -------------------------------------------------

    @property
    def viewer_open(self) -> bool:
        return bool(self.viewer_container.visible)

    def open_viewer(self, switch_to_library: bool = True):
        """Динамически открывает правую панель просмотра (split 40/60).

        Вызывается ТОЛЬКО по целевым действиям: клик по книге, переход к
        главе, запрос контекста документа из чата.
        """
        was_closed = not self.viewer_container.visible
        # Если пользователь открыл просмотр из вкладки «Настройки» —
        # корректно переключаемся на библиотеку (rail.selected_index=1),
        # иначе settings-контейнер остался бы видимым под библиотекой.
        if self.settings_container.visible:
            self.rail.selected_index = 1
        if switch_to_library:
            self.show_library()
        self.library_row.visible = True
        self.viewer_container.visible = True
        if was_closed:
            logger.info("Панель просмотра открыта")
        self._safe_update()

    def close_viewer(self):
        """Крестик на панели: возвращает интерфейс в исходное состояние."""
        self.viewer_container.visible = False
        if self.on_viewer_closed:
            try:
                self.on_viewer_closed()
            except Exception as ex:  # pragma: no cover
                logger.warning(f"on_viewer_closed: {ex}")
        self._safe_update()

    def _on_close_viewer(self, e):
        self.close_viewer()

    def set_viewer_title(self, title: Optional[str]):
        header_text = self.viewer_header.controls[0]
        header_text.value = f"📄 {title}" if title else "📄 Просмотр документа"
        self._safe_update()

    # ------------------------------------------------------------------

    @staticmethod
    def _placeholder(text: str) -> ft.Control:
        return ft.Container(
            content=ft.Column(
                [
                    ft.Icon(ft.Icons.DESCRIPTION_OUTLINED, size=48, color="grey"),
                    ft.Text(text, size=14, color="grey"),
                ],
                horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                alignment=ft.MainAxisAlignment.CENTER,
                spacing=8,
            ),
            alignment=ft.Alignment.CENTER,
            expand=True,
        )

    def _safe_update(self):
        try:
            self.page.update()
        except Exception:
            pass  # страница могла быть закрыта
