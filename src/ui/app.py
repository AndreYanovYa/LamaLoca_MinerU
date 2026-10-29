# src/ui/app.py
"""
Главное приложение lama-loca (Flet 1.x) — рефакторинг структуры главного окна.

НОВАЯ СТРУКТУРА (ТЗ «адаптивный layout»):
* Левая панель навигации (NavigationRail, фикс. ширина) — глобальный
  переключатель контекста «Чат» / «Библиотека».
* Режим чата: окно чата занимает ВСЁ оставшееся пространство (без
  принудительного разделения экрана, как было в двухпанельной версии).
* Режим библиотеки: по умолчанию только список книг; правая панель просмотра
  ИЗНАЧАЛЬНО СКРЫТА и открывается динамически (split 40/60) по целевым
  действиям: клик по книге, переход к главе из оглавления, клик по источнику
  в чате. Крестик на панели возвращает исходное состояние.
Весь механизм видимости — src/ui/layout.py (MainLayout): флаги visible +
page.update(); никакого повторного построения дерева.

Публичный API модуля: LamaLocaApp (класс-обёртка) и main(page) — тонкий вход
для `flet run` / ft.run / main.py.
"""
from pathlib import Path
from typing import Optional

import flet as ft
from loguru import logger

from src.core.application import Application
from src.core.settings import Settings
from src.core.paths import Paths
from src.ui.chat_view import ChatView
from src.ui.layout import MainLayout
from src.ui.library_view import LibraryView
from src.ui.pdf_viewer import PdfViewerView
from src.ui.settings_view import SettingsView
from src.ui.workers.import_worker import ImportWorker


class LamaLocaApp:
    """Классическая точка сборки UI: держит Application и строит страницу."""

    def __init__(self, page: ft.Page, settings: Settings = None, paths: Paths = None,
                 config_path: Path = None):
        self.page = page

        if settings is None:
            # АУДИТ: конфиг искался относительно CWD; теперь — рядом с корнем
            # проекта (src/ui/app.py -> project_root/config.yaml), затем CWD.
            candidates = [
                config_path,
                Path(__file__).resolve().parent.parent.parent / "config.yaml",
                Path("config.yaml"),
            ]
            resolved = next((c for c in candidates if c and c.exists()), None)
            if resolved is None:
                raise FileNotFoundError(
                    "config.yaml не найден — скопируйте config.example.yaml в config.yaml"
                )
            settings = Settings.load(resolved)

        # ТЗ п.4: запоминаем, откуда загружен конфиг — настройки сохраняются
        # именно в этот файл (а не в CWD).
        self.config_path = config_path if config_path is not None else resolved

        if paths is None:
            # storage_root возвращает str — приводим к Path явно.
            paths = Paths(Path(settings.storage_root))

        self.settings = settings
        self.paths = paths
        self.app = Application(settings, paths)

        self._build()

    # ------------------------------------------------------------------

    def _build(self):
        page = self.page
        app = self.app
        settings = self.settings

        # --- Параметры окна ---
        # ТЗ п.2: динамический заголовок с Ф.И.О. студента БГУ ФПМиИ
        # (берётся из settings.student_name; при пустом значении — «не указано»).
        page.title = settings.window_title()
        page.window.width = 1400
        page.window.height = 900
        page.window.min_width = 1000
        page.window.min_height = 700
        page.padding = 0
        page.theme_mode = ft.ThemeMode.DARK
        page.bgcolor = "#121212"

        # --- Компоненты ---
        self.layout = MainLayout(page)
        viewer = PdfViewerView(page)
        self.viewer = viewer

        def doc_title(doc_id) -> Optional[str]:
            row = app.sqlite_store.conn.execute(
                "SELECT title FROM documents WHERE doc_id = ?", (doc_id,)
            ).fetchone()
            return row[0] if row else None

        def open_in_viewer(doc_id, path):
            """Целевое действие пользователя → открываем панель просмотра."""
            title = doc_title(doc_id)
            viewer.set_document(doc_id, path, title=title)
            app.rag_pipeline.set_active_document(doc_id)
            self.layout.set_viewer_title(title or Path(path).name)
            chat_view.set_active_document_label(title)
            # Динамическое раскрытие правой панели (split 40/60) + update.
            self.layout.open_viewer(switch_to_library=True)

        def select_for_chat(doc_id):
            """Делает книгу активной для чата без открытия просмотрщика."""
            app.rag_pipeline.set_active_document(doc_id)
            chat_view.set_active_document_label(doc_title(doc_id))

        def open_source_page(page_number, doc_id: Optional[str] = None):
            """Клик по источнику в чате → контекст документа в просмотрщике.

            Если документ уже открыт — просто переходим на страницу; иначе
            открываем файл источника (по doc_id из source) и показываем
            нужную страницу. Панель при этом раскрывается динамически.
            """
            try:
                page_number = int(page_number)
            except (TypeError, ValueError):
                return
            target_doc_id = doc_id or viewer.doc_id
            if target_doc_id and (viewer.doc_id != target_doc_id or not viewer.current_doc):
                row = app.sqlite_store.conn.execute(
                    "SELECT source_path, title FROM documents WHERE doc_id = ?",
                    (target_doc_id,),
                ).fetchone()
                if row:
                    viewer.set_document(target_doc_id, row[0], title=row[1])
                    self.layout.set_viewer_title(row[1])
            if viewer.current_doc:
                viewer.navigate_to_page(page_number)
            self.layout.open_viewer(switch_to_library=True)

        library_view = LibraryView(
            page,
            settings,
            self.paths,
            import_worker=ImportWorker(
                settings,
                self.paths,
                router=app.router,
                chunker=app.chunker,
                sqlite_store=app.sqlite_store,
                chroma_store=app.chroma_store,
                embedder=app.embedder,
            ),
            sqlite_store=app.sqlite_store,
            on_open_pdf=open_in_viewer,
            on_select_doc=select_for_chat,
        )
        self.library_view = library_view

        chat_view = ChatView(
            page, settings, app.rag_pipeline,
            on_open_source=open_source_page,
            on_show_toc=lambda: self.layout.open_viewer(switch_to_library=True),
        )
        self.chat_view = chat_view

        # ТЗ п.4: вкладка «Настройки» (Ф.И.О. студента) + обновление заголовка
        # окна налету после сохранения (page.title обновляет и сам SettingsView).
        settings_view = SettingsView(
            page, settings,
            config_path=getattr(self, "config_path", None),
            on_settings_saved=lambda new_settings: setattr(self, "settings", new_settings),
        )
        self.settings_view = settings_view

        # Крестик «Закрыть просмотр» → освобожаем ресурс документа и
        # возвращаем библиотеку в исходное состояние (только список книг).
        self.layout.on_viewer_closed = lambda: viewer.close_document()

        # Сборка каркаса: sidebar + переключаемые области.
        self.layout.attach(
            chat=chat_view.build(),
            library=library_view.build(),
            viewer=viewer.build(),
            settings=settings_view.build(),
        )
        self.layout.mount()

        # Если в библиотеке уже есть книги — делаем последнюю активной для RAG,
        # чтобы первый вопрос не «искал по всему» непредсказуемо. Просмотрщик
        # при этом ОСТАЁТСЯ скрытым (ТЗ: панель открывается только по действию).
        last = app.sqlite_store.conn.execute(
            "SELECT doc_id, source_path, title FROM documents "
            "WHERE status = 'READY' ORDER BY created_at DESC LIMIT 1"
        ).fetchone()
        if last:
            app.rag_pipeline.set_active_document(last[0])
            chat_view.set_active_document_label(last[2])
            logger.info(f"Активная книга по умолчанию: {last[2]}")

        def on_disconnect(e):
            self.close()

        def _flush_db():
            """FIX («карточка утеряна»): мгновенно вливает WAL в основной
            файл БД. Вызывается после каждого импорта — карточка и блоки
            закреплены на диске, даже если пользователь закроет окно до
            штатного on_disconnect."""
            try:
                self.app.sqlite_store.flush()
            except Exception as fe:  # pragma: no cover
                logger.warning(f"Не удалось выполнить wal_checkpoint: {fe}")

        library_view.on_import_finished = _flush_db

        page.on_disconnect = lambda e: self.close()
        logger.info("✅ Lama Loca запущена (адаптивный layout)")

    # ------------------------------------------------------------------

    def build(self) -> None:
        """Совместимость со старым вызовом `page.add(app.build())`.

        UI добавляется на страницу внутри _build(); метод существует, чтобы
        внешний код, ожидавший класс, не падал. Возвращает None.
        """
        return None

    def close(self):
        """Освобождает ресурсы приложения при закрытии."""
        try:
            self.viewer.close_document()
        except Exception as e:  # pragma: no cover
            logger.warning(f"Ошибка закрытия просмотрщика: {e}")
        self.app.close()


def main(page: ft.Page, config_path: Path = None):
    """Вход для `flet run src/ui/app.py` и для main.py."""
    LamaLocaApp(page, config_path=config_path)
