# src/ui/settings_view.py
"""
Вкладка «Настройки» Lama Loca (Flet 1.x).

ТЗ «Кастомизация интерфейса под студента БГУ ФПМиИ», п.4:
* Поле ft.TextField(label="Ф.И.О. студента", value=settings.student_name).
* Сохранение по кнопке «Сохранить»: запись в config.yaml через
  Settings.save_student_name() (frozen dataclass — возвращается новый
  экземпляр, мутаций нет), мгновенное обновление page.title главного окна
  без перезапуска приложения.

Архитектура сохранена: модуль ничего не знает о RAG/LLM, только UI + Settings.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import flet as ft
from loguru import logger

from src.core.settings import Settings


def window_title_for(settings: Settings) -> str:
    """Единая точка форматирования заголовка окна (используется app.py и здесь)."""
    return settings.window_title()


class SettingsView:
    """Форма настроек: Ф.И.О. студента + сохранение конфигурации."""

    def __init__(
        self,
        page: ft.Page,
        settings: Settings,
        config_path: Optional[Path] = None,
        on_settings_saved: Optional[Callable[[Settings], None]] = None,
    ):
        self.page = page
        self.settings = settings
        self.config_path = config_path
        self.on_settings_saved = on_settings_saved

        self.fio_field = ft.TextField(
            label="Ф.И.О. студента",
            hint_text="Например: Иванов И. И.",
            value=settings.student_name or settings.app.student_name or "",
            width=420,
            border_radius=8,
        )

        self.status_text = ft.Text("", size=12, color=ft.Colors.GREY)

        self.save_btn = ft.FilledButton(
            icon=ft.Icons.SAVE,
            content=ft.Text("Сохранить"),
            bgcolor="#2196F3",
            color="white",
            on_click=self._on_save,
        )

    # ------------------------------------------------------------------

    def build(self) -> ft.Control:
        return ft.Container(
            content=ft.Column(
                [
                    ft.Text("⚙️ Настройки", size=20, weight=ft.FontWeight.BOLD),
                    ft.Divider(color="#DDDDDD"),
                    ft.Text(
                        "Персонализация интерфейса",
                        size=14,
                        weight=ft.FontWeight.W_500,
                    ),
                    ft.Text(
                        "Ф.И.О. студента отображается в заголовке окна "
                        "приложения (БГУ, ФПМиИ).",
                        size=12,
                        color=ft.Colors.GREY_700,
                    ),
                    self.fio_field,
                    ft.Row([self.save_btn, self.status_text], spacing=12),
                ],
                spacing=14,
                expand=True,
            ),
            padding=20,
            expand=True,
        )

    # ------------------------------------------------------------------

    def _on_save(self, e):
        new_name = (self.fio_field.value or "").strip()
        try:
            # frozen-конфиг: save_student_name пишет app.student_name в
            # config.yaml и возвращает НОВЫЙ Settings (без мутаций).
            updated = self.settings.save_student_name(new_name, self.config_path)
        except Exception as ex:  # pragma: no cover - disk errors
            logger.error(f"Не удалось сохранить настройки: {ex}")
            self._set_status(f"Ошибка сохранения: {ex}", ok=False)
            return

        self.settings = updated

        # Мгновенно обновляем заголовок окна «налету», без перезапуска.
        try:
            self.page.title = updated.window_title()
            self.page.update()
        except Exception as ex:  # pragma: no cover
            logger.warning(f"page.title update failed: {ex}")

        if self.on_settings_saved:
            try:
                self.on_settings_saved(updated)
            except Exception as ex:  # pragma: no cover
                logger.warning(f"on_settings_saved: {ex}")

        self._set_status(
            "Сохранено ✔ заголовок окна обновлён" if new_name
            else "Сохранено ✔ (Ф.И.О. пустое — в заголовке «не указано»)",
            ok=True,
        )
        logger.info(f"Настройки сохранены: student_name={new_name!r}")

    def _set_status(self, text: str, ok: bool):
        self.status_text.value = text
        self.status_text.color = ft.Colors.GREEN_600 if ok else ft.Colors.RED_400
        try:
            self.status_text.update()
        except Exception:
            try:
                self.page.update()
            except Exception:
                pass
