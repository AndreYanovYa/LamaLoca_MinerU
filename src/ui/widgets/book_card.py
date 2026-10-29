# src/ui/widgets/book_card.py
import flet as ft
from typing import Callable


class BookCard:
    """Карточка книги в списке библиотеки."""
    
    def __init__(self, doc: dict, on_delete: Callable[[str], None]):
        self.doc = doc
        self.on_delete = on_delete
    
    def build(self) -> ft.Control:
        """Строит карточку."""
        return ft.Card(
            content=ft.Container(
                content=ft.Row(
                    [
                        # Иконка формата
                        ft.Icon(
                            self._get_format_icon(self.doc["file_format"]),
                            size=40,
                            color=ft.Colors.BLUE,
                        ),
                        
                        # Информация о книге
                        ft.Column(
                            [
                                ft.Text(
                                    self.doc["title"],
                                    size=18,
                                    weight=ft.FontWeight.BOLD,
                                ),
                                ft.Text(
                                    f"Автор: {self.doc['author'] or 'Неизвестен'}",
                                    size=14,
                                    color=ft.Colors.GREY,
                                ),
                                ft.Text(
                                    f"Формат: {self.doc['file_format'].upper()} | "
                                    f"Страниц: {self.doc['total_pages'] or 'N/A'}",
                                    size=12,
                                    color=ft.Colors.GREY,
                                ),
                            ],
                            expand=True,
                        ),
                        
                        # Кнопка удаления
                        ft.IconButton(
                            icon=ft.Icons.DELETE,
                            icon_color=ft.Colors.RED,
                            on_click=lambda _: self._confirm_delete(),
                        ),
                    ],
                    alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                    vertical_alignment=ft.CrossAxisAlignment.CENTER,
                ),
                padding=20,
            ),
            margin=10,
        )
    
    def _get_format_icon(self, format: str) -> str:
        """Возвращает иконку для формата."""
        icons = {
            "pdf": ft.Icons.PICTURE_AS_PDF,
            "epub": ft.Icons.MENU_BOOK,
            "fb2": ft.Icons.BOOK,
            "djvu": ft.Icons.DESCRIPTION,
            "txt": ft.Icons.TEXT_SNIPPET,
        }
        return icons.get(format, ft.Icons.INSERT_DRIVE_FILE)
    
    def _confirm_delete(self):
        """Подтверждение удаления."""
        def close_dialog(e):
            dlg.open = False
            self.doc["page"].update()
        
        def delete(e):
            self.on_delete(self.doc["doc_id"])
            close_dialog(e)
        
        dlg = ft.AlertDialog(
            title=ft.Text("Удалить книгу?"),
            content=ft.Text(f"Вы уверены, что хотите удалить '{self.doc['title']}'?"),
            actions=[
                ft.TextButton("Отмена", on_click=close_dialog),
                ft.TextButton("Удалить", on_click=delete, style=ft.ButtonStyle(color=ft.Colors.RED)),
            ],
        )
        
        # Добавляем диалог в страницу (упрощённо)
        # В реальной реализации нужно передавать page в конструктор