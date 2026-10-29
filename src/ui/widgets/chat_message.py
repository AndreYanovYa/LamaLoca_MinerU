# src/ui/widgets/chat_message.py
import flet as ft
from typing import List, Dict, Any


class ChatMessage:
    """Сообщение в чате (пользователь или ассистент)."""
    
    def __init__(self, role: str, content: str, sources: List[Dict[str, Any]] = None):
        self.role = role
        self.content = content
        self.sources = sources or []
    
    def build(self) -> ft.Control:
        """Строит сообщение."""
        if self.role == "user":
            return self._build_user_message()
        else:
            return self._build_assistant_message()
    
    def _build_user_message(self) -> ft.Control:
        """Сообщение пользователя."""
        return ft.Container(
            content=ft.Column(
                [
                    ft.Text("Вы", size=12, weight=ft.FontWeight.BOLD, color=ft.Colors.BLUE),
                    ft.Text(self.content, size=14),
                ],
                spacing=5,
            ),
            bgcolor=ft.Colors.BLUE_GREY_900,
            border_radius=10,
            padding=15,
            margin=ft.Margin.only(left=100),
            alignment=ft.Alignment.CENTER_RIGHT,
        )
    
    def _build_assistant_message(self) -> ft.Control:
        """Сообщение ассистента с цитатами."""
        controls = [
            ft.Text("ИИ", size=12, weight=ft.FontWeight.BOLD, color=ft.Colors.GREEN),
            ft.Text(self.content, size=14),
        ]
        
        # Добавляем источники, если есть
        if self.sources:
            controls.append(ft.Text("📚 Источники:", size=12, weight=ft.FontWeight.BOLD))
            
            chips = []
            for i, source in enumerate(self.sources, start=1):
                title = source.get("title", "Без названия")
                page = source.get("page_number")
                chapter = source.get("chapter")
                
                if page:
                    label = f"[{i}] {title}, стр. {page}"
                elif chapter:
                    label = f"[{i}] {title}, {chapter}"
                else:
                    label = f"[{i}] {title}"
                
                chips.append(
                    ft.Chip(
                        label=ft.Text(label, size=12),
                        on_click=lambda e, s=source: self._show_source(s),
                    )
                )
            
            controls.append(ft.Row(chips, wrap=True))
        
        return ft.Container(
            content=ft.Column(controls, spacing=5),
            bgcolor=ft.Colors.GREEN_900,
            border_radius=10,
            padding=15,
            margin=ft.Margin.only(right=100),
            alignment=ft.Alignment.CENTER_LEFT,
        )
    
    def _show_source(self, source: Dict[str, Any]):
        """Показывает информацию об источнике (упрощённо)."""
        # В реальной реализации можно открыть предпросмотр страницы
        pass