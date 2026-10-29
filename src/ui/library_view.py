# src/ui/library_view.py
"""
Представление библиотеки документов (светлая тема).

АУДИТ (исправлено):
1. SQLiteStore/ChromaStore/ImportWorker берутся из Application (общие
   экземпляры) — список теперь видит результаты импорта немедленно.
2. Диалог выбора файлов принимает и PDF, и EPUB (раньше фильтр был *.pdf,
   хотя загрузка EPUB заявлена в ТЗ).
3. Удаление книги инкапсулировано в SQLiteStore.delete_document (FTS-триггеры
   срабатывают корректно), Chroma — через delete_by_doc_id.
4. Добавлены действия для карточки книги: «Открыть в просмотрщике» и
   «Сделать активной» (RAG отвечает по выбранной книге, а не «LIMIT 1»).
5. Импорт выполняется в отдельном потоке (asyncio.to_thread внутри task'а UI
   уже асинхронный; парсинг/OCR/эмбеддинги идут вне event loop) — прогресс-бар
   не блокирует интерфейс.
"""
import asyncio
import webbrowser
from pathlib import Path
from typing import List

# АУДИТ: раньше `from tkinter import Tk, filedialog` был импортом верхнего
# уровня — приложение падало с ImportError на старте в любом окружении без Tk
# (Linux без python3-tk, venv без Tk-библиотек). Tk теперь импортируется
# лениво внутри _pick_files() и работает только там, где доступен (Windows).
try:  # pragma: no cover - окружение без Tk просто получает None
    import tkinter as _tkinter  # noqa: F401
    TK_AVAILABLE = True
except Exception:
    _tkinter = None
    TK_AVAILABLE = False

import flet as ft
from loguru import logger

from src.core.settings import Settings
from src.core.paths import Paths
from src.storage.sqlite_store import SQLiteStore
from src.ui.workers.import_worker import ImportWorker


class LibraryView:
    def __init__(self, page: ft.Page, settings: Settings, paths: Paths,
                 import_worker: ImportWorker = None,
                 sqlite_store: SQLiteStore = None,
                 on_open_pdf=None, on_select_doc=None):
        self.page = page
        self.settings = settings
        self.paths = paths

        # ✅ DI: переиспользуем общие экземпляры из Application.
        self.sqlite_store = sqlite_store or SQLiteStore(settings.sqlite_db_path)
        self.import_worker = import_worker or ImportWorker(settings, paths)

        # Колбэки двухпанельного интерфейса (чат + просмотрщик).
        self._on_open_pdf = on_open_pdf      # (doc_id, source_path) -> None
        self._on_select_doc = on_select_doc  # (doc_id) -> None

        self.search_field = ft.TextField(
            hint_text="🔍 Поиск по названию или автору...",
            expand=True,
            on_change=self._on_search_change,
            prefix_icon=ft.Icons.SEARCH,
            border_radius=8,
        )
        self.add_books_btn = ft.FilledButton(
            content=ft.Text("➕ Добавить книги", color="white"),
            icon=ft.Icons.UPLOAD_FILE,
            icon_color="white",
            on_click=self._on_add_books_click,
            style=ft.ButtonStyle(bgcolor="#2196F3"),
        )
        self.refresh_btn = ft.IconButton(
            icon=ft.Icons.REFRESH,
            tooltip="Обновить список",
            on_click=self._on_refresh_click,
        )

        self.library_list = ft.ListView(expand=True, spacing=10, padding=10)

        # Прогресс импорта: стабильные контролы (обновляем .value/.value,
        # а не пересоздаём всё дерево на каждый тик).
        self._progress_bar = ft.ProgressBar(value=0, color="#2196F3")
        self._progress_text = ft.Text("Ожидание...", size=14, weight=ft.FontWeight.BOLD)
        self._progress_file = ft.Text("", size=12, color="grey", text_align="center")
        self._progress_pct = ft.Text("", size=12, color="#2196F3")
        self.progress_container = ft.Container(
            content=ft.Column(
                [
                    self._progress_bar,
                    self._progress_text,
                    self._progress_file,
                    self._progress_pct,
                ],
                horizontal_alignment="center",
                spacing=8,
            ),
            alignment="center",
            padding=20,
            visible=False,
        )

        self._load_library()

    def build(self) -> ft.Control:
        return ft.Column(
            [
                ft.Row(
                    [
                        ft.Text("📚 Библиотека", size=32, weight=ft.FontWeight.BOLD),
                        ft.Container(expand=True),
                        self.refresh_btn,
                        self.add_books_btn,
                    ],
                    alignment="end",
                ),
                ft.Row([self.search_field]),
                self.progress_container,
                self.library_list,
            ],
            expand=True,
        )

    # ------------------------------------------------------------------
    # Список документов
    # ------------------------------------------------------------------

    def _load_library(self, query: str = ""):
        try:
            documents = self.sqlite_store.get_all_documents()
            self.library_list.controls.clear()

            if query:
                query_lower = query.lower()
                documents = [
                    doc for doc in documents
                    if query_lower in (doc.get("title") or "").lower()
                    or query_lower in (doc.get("author") or "").lower()
                ]

            if not documents:
                self.library_list.controls.append(
                    ft.Container(
                        content=ft.Column(
                            [
                                ft.Icon(ft.Icons.LIBRARY_BOOKS, size=64, color="grey"),
                                ft.Text(
                                    "Библиотека пуста" if not query else f"Ничего не найдено по '{query}'",
                                    size=18,
                                    color="grey",
                                ),
                            ],
                            horizontal_alignment="center",
                            spacing=10,
                        ),
                        alignment=ft.Alignment.CENTER,
                        expand=True,
                    )
                )

            # FIX («карточка утеряна после закрытия», часть 2): если импорт
            # был прерван (закрыли окно во время векторизации), карточка
            # осталась в статусе INDEXING, а блоки/векторы — нет. При
            # загрузке библиотеки такие «полуготовые» записи помечаем как
            # FAILED; READY-книги не трогаем.
            stale_ids = [
                d.get("doc_id") for d in documents if d.get("status") == "INDEXING"
            ]
            if stale_ids:
                try:
                    placeholders = ",".join("?" * len(stale_ids))
                    self.sqlite_store.conn.execute(
                        f"UPDATE documents SET status = 'FAILED' WHERE doc_id IN ({placeholders})",
                        stale_ids,
                    )
                    self.sqlite_store.conn.commit()
                    logger.warning(
                        f"Библиотека: {len(stale_ids)} документов в незавершённой "
                        f"индексации помечены как FAILED (прерванный импорт)"
                    )
                    for d in documents:
                        if d.get("doc_id") in stale_ids:
                            d["status"] = "FAILED"
                except Exception as se:
                    logger.warning(f"Не удалось пометить прерванные импорты: {se}")

            for doc in documents:
                self.library_list.controls.append(self._build_document_card(doc))
        except Exception as e:
            logger.error(f"Ошибка загрузки библиотеки: {e}")

    @staticmethod
    def _status_label(status: str) -> str:
        """Человекочитаемый статус карточки книги."""
        return {
            "READY": "✅ готова",
            "INDEXING": "⏳ индексация…",
            "FAILED": "⚠️ импорт прерван — реимпортируйте",
        }.get((status or "").upper(), status or "?")

    def _build_document_card(self, doc: dict) -> ft.Control:
        fmt = (doc.get("file_format") or "pdf").lower()
        icon = ft.Icons.PICTURE_AS_PDF if fmt == "pdf" else ft.Icons.MENU_BOOK
        pages = doc.get("total_pages")
        pages_label = f"Страниц: {pages}" if pages else "Пагинация отсутствует (EPUB)"

        # REFACTORING (адаптивный layout): клик по самой карточке книги —
        # целевое действие, открывающее панель просмотра. Раньше для этого
        # нужен был отдельный «📖 Открыть»; теперь карточка целиком кликабельна
        # (кнопки внутри остаются самостоятельными контролами).
        # FIX (Flet 1.x): у ft.Card нет параметров ink/on_click/hover_elevation —
        # они есть только у ft.Container. В 1.0.5 это роняло _load_library с
        # "Card.__init__() got an unexpected keyword argument 'ink'". Кликабель-
        # ность и ripple перенесены на внутренний Container.
        card = ft.Card(
            content=ft.Container(
                ink=True,
                on_click=lambda e, d=doc: self._on_open_document(d),
                border_radius=8,
                content=ft.Column(
                    [
                        ft.Row(
                            [
                                ft.Icon(icon, color="red", size=40),
                                ft.Column(
                                    [
                                        ft.Text(doc.get("title", "Без названия"),
                                                weight=ft.FontWeight.BOLD, size=16),
                                        ft.Text(doc.get("author") or "Неизвестный автор",
                                                size=12, color="grey"),
                                        ft.Text(
                                            f"Формат: {fmt.upper()} | {pages_label} | "
                                            f"Статус: {self._status_label(doc.get('status'))}",
                                            size=11, color="grey",
                                        ),
                                    ],
                                    spacing=4,
                                    expand=True,
                                ),
                            ],
                            spacing=15,
                        ),
                        ft.Row(
                            [
                                ft.TextButton(
                                    "📖 Открыть",
                                    icon=ft.Icons.VISIBILITY,
                                    on_click=lambda e, d=doc: self._on_open_document(d),
                                ),
                                ft.TextButton(
                                    "💬 Спросить по книге",
                                    icon=ft.Icons.CHAT,
                                    on_click=lambda e, d=doc: self._on_make_active(d),
                                ),
                                ft.Container(expand=True),
                                ft.IconButton(
                                    icon=ft.Icons.DELETE,
                                    tooltip="Удалить книгу",
                                    icon_color="red",
                                    on_click=lambda e, d_id=doc.get("doc_id", ""),
                                        t=doc.get("title", ""): self._confirm_delete(d_id, t),
                                ),
                            ],
                        ),
                    ],
                    spacing=5,
                ),
                padding=15,
            )
        )
        card.data = doc.get("doc_id")
        return card

    # ------------------------------------------------------------------
    # Действия пользователя
    # ------------------------------------------------------------------

    def _on_search_change(self, e):
        self._load_library(self.search_field.value.strip())
        self.page.update()

    def _on_refresh_click(self, e):
        self.search_field.value = ""
        self._load_library()
        self.page.update()

    def _on_open_document(self, doc: dict):
        """Открывает книгу во встроенном просмотрщике (для PDF)."""
        path = doc.get("source_path") or ""
        if not path or not Path(path).exists():
            self._show_snackbar("❌ Файл не найден на диске", "red")
            return
        if self._on_open_pdf:
            self._on_open_pdf(doc.get("doc_id"), path)
        else:  # fallback: внешний просмотрщик
            webbrowser.open(Path(path).as_uri())

    def _on_make_active(self, doc: dict):
        """Делает книгу активной для чата (RAG отвечает по ней)."""
        if self._on_select_doc:
            self._on_select_doc(doc.get("doc_id"))
            self._show_snackbar(f"✅ Активная книга: {doc.get('title', '')}", "green")

    def _confirm_delete(self, doc_id: str, title: str):
        """Подтверждение удаления перед удалением.

        ВАЖНО (Flet 1.x, проверено на 1.0.5 через introspection):
        * ``page.close(dialog)`` не существует -> был AttributeError в
          обработчике (первая версия бага: зависание кнопок диалога);
        * ``page.dialog = ...`` тоже НЕ существует (у Page нет такого поля —
          dataclass молча создаёт неиспользуемый атрибут, диалог НИКОГДА не
          рендерится). Это была вторая версия бага: «на кнопку Удалить нет
          никакой реакции» — исключение глоталось Flet, и ничего не
          происходило;
        * штатный API Flet 1.x: ``page.show_dialog(dialog)`` (ставит
          open=True, добавляет в стек page._dialogs и обновляет оверлей) и
          закрытие ``dialog.open = False; dialog.update()``.
        """
        def close_dialog():
            self._close_dialog(dialog)

        def do_delete(e):
            close_dialog()
            self._on_delete_book(doc_id)

        def cancel(e):
            close_dialog()

        dialog = ft.AlertDialog(
            modal=True,
            title=ft.Text("Удалить книгу?"),
            content=ft.Text(f"«{title}» будет удалена из библиотеки вместе с индексом."),
            actions=[
                ft.TextButton("Отмена", on_click=cancel),
                ft.TextButton("Удалить", on_click=do_delete,
                             style=ft.ButtonStyle(color=ft.Colors.RED)),
            ],
        )
        try:
            if not hasattr(self.page, "show_dialog"):
                raise RuntimeError("В этом Flet нет page.show_dialog")
            self.page.show_dialog(dialog)
        except Exception as ex:
            logger.error(f"Не удалось показать диалог удаления: {ex}")
            self._show_snackbar(f"❌ Не открылся диалог удаления: {ex}", "red")

    def _close_dialog(self, dialog):
        """Кросс-версионное закрытие AlertDialog (Flet 1.x).

        Порядок: сначала штатный путь ``open=False + dialog.update()``; если
        контрол ещё не «привязан» к странице (RuntimeError в раннем Flet или
        гонкаdismiss'а) — закрываем через стек страницы: ``page.pop_dialog()``
        либо прямой патч ``page._dialogs``. Любая неудача логируется и не
        роняет обработчик on_click (именно так раньше «зависало» окно).
        """
        try:
            dialog.open = False
            dialog.update()
            return
        except Exception as ex_first:
            logger.debug(f"dialog.update() не прошёл ({ex_first}); пробуем pop_dialog")
        try:
            if hasattr(self.page, "pop_dialog"):
                self.page.pop_dialog()
                return
        except Exception as ex2:
            logger.debug(f"pop_dialog не прошёл: {ex2}")
        try:
            # самый нижний уровень: состояние уже изменено (open=False),
            # обновляем контейнер стека диалогов напрямую
            stack = getattr(self.page, "_dialogs", None)
            if stack is not None:
                stack.update()
        except Exception as ex3:
            logger.debug(f"обновить стек диалогов не удалось: {ex3}")

    def _on_delete_book(self, doc_id: str):
        try:
            # Удаляем файл из ChromaDB и все данные из SQLite (FTS чистится
            # триггерами при удалении блоков).
            self.import_worker.chroma_store.delete_by_doc_id(doc_id)
            self.sqlite_store.delete_document(doc_id)
            self._load_library(self.search_field.value.strip())
            self.page.update()
            self._show_snackbar("Книга успешно удалена", "green")
        except Exception as ex:
            logger.error(f"Ошибка удаления: {ex}")
            self._show_snackbar(f"❌ Ошибка удаления: {ex}", "red")

    # ------------------------------------------------------------------
    # Импорт
    # ------------------------------------------------------------------

    def _pick_files(self) -> List[str]:
        """Windows-native диалог выбора PDF/EPUB (отдельный поток не нужен —
        вызываем из run_thread, чтобы не блокировать event loop Flet)."""
        # АУДИТ: ленивый импорт tkinter + понятная деградация вместо падения.
        if not TK_AVAILABLE:
            raise RuntimeError(
                "Tkinter недоступен в этом окружении — установите python3-tk "
                "(Linux) или используйте Windows-дистрибутив Python с Tk."
            )
        from tkinter import Tk, filedialog

        root = Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        try:
            file_paths = filedialog.askopenfilenames(
                title="Выберите учебники (PDF или EPUB)",
                filetypes=[
                    ("PDF и EPUB файлы", "*.pdf *.epub"),
                    ("PDF files", "*.pdf"),
                    ("EPUB files", "*.epub"),
                    ("All files", "*.*"),
                ],
            )
        finally:
            root.destroy()
        return list(file_paths)

    def _on_add_books_click(self, e):
        # Диалог запускаем в фоновом потоке: Tk mainloop не должен
        # блокировать event loop Flet.
        self.page.run_thread(self._files_picked_thread)

    def _files_picked_thread(self):
        try:
            file_paths = self._pick_files()
        except Exception as ex:
            logger.error(f"Ошибка диалога выбора файлов: {ex}")
            return

        if not file_paths:
            return

        self.progress_container.visible = True
        self.library_list.visible = False
        try:
            self.page.update()
        except Exception:
            pass

        self.page.run_task(self._import_files_task, [Path(fp) for fp in file_paths])

    async def _import_files_task(self, file_paths: List[Path]):
        total = len(file_paths)
        success_count = 0
        errors: List[str] = []

        for i, file_path in enumerate(file_paths):
            try:
                self._update_progress("Начало импорта...", file_path.name, 0)

                # Тяжёлый CPU/IO код (парсинг, OCR, эмбеддинги) уходит
                # в отдельный поток — UI остаётся отзывчивым.
                await asyncio.to_thread(
                    self._import_sync,
                    file_path,
                    lambda msg, pct, idx=i: self._update_progress(
                        f"Книга {idx + 1} из {total}: {msg}", file_path.name, pct
                    ),
                )
                success_count += 1
            except Exception as ex:
                logger.error(f"❌ Ошибка импорта {file_path.name}: {ex}")
                errors.append(f"{file_path.name}: {ex}")

        self.progress_container.visible = False
        self.library_list.visible = True
        self._load_library()
        try:
            self.page.update()
        except Exception:
            pass

        if success_count > 0:
            self._show_snackbar(
                f"✅ Успешно импортировано {success_count} из {total} книг", "green"
            )
        if errors:
            self._show_snackbar(f"⚠️ Ошибок: {len(errors)} — см. журнал", "orange")

    def _import_sync(self, file_path: Path, progress_callback):
        """Синхронная обёртка корутины импорта для запуска в потоке."""
        loop = asyncio.new_event_loop()
        try:
            result = loop.run_until_complete(
                self.import_worker.import_file(file_path, progress_callback)
            )
            # FIX («карточка утеряна»): сразу после успешного импорта
            # принудительно вливаем WAL-журнал в основной файл БД. Если
            # пользователь закроет окно раньше штатного on_disconnect,
            # карточка книги всё равно останется на диске.
            try:
                cb = getattr(self, "on_import_finished", None)
                if callable(cb):
                    cb()
            except Exception as fe:
                logger.warning(f"wal_checkpoint после импорта не выполнен: {fe}")
            return result
        finally:
            loop.close()

    def _update_progress(self, message: str, filename: str, percent: int):
        """Обновляет прогресс-бар (вызывается из рабочего потока).

        АУДИТ: контейнер пересоздавался целиком на каждый тик — Flet терял
        ссылку на ProgressBar и анимация «дёргалась». Теперь обновляются
        свойства существующих контролов + page.update().
        """
        percent = max(0, min(100, int(percent or 0)))
        self._progress_bar.value = percent / 100 if percent > 0 else None  # None -> индетерминированный
        self._progress_text.value = message
        self._progress_file.value = filename
        self._progress_pct.value = f"{percent}%"
        try:
            self.page.update()
        except Exception:
            pass  # страница могла закрыться во время тяжелого импорта

    # ------------------------------------------------------------------
    # Уведомления
    # ------------------------------------------------------------------

    def _show_snackbar(self, message: str, color: str = "blue"):
        snack = ft.SnackBar(content=ft.Text(message), bgcolor=color, behavior=ft.SnackBarBehavior.FLOATING)
        self.page.overlay.append(snack)
        snack.open = True
        try:
            self.page.update()
        except Exception:
            pass
