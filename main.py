# main.py
"""
Точка входа в приложение lama-loca.
"""
import sys
from pathlib import Path

# Добавляем корень проекта в PYTHONPATH для запуска как из папки, так и через ft.app
sys.path.insert(0, str(Path(__file__).resolve().parent))

# HEADLESS: изоляция фоновых процессов (page_renderer/pix2tex/LaTeX) от оконной
# системы Windows ДО импорта torch/pix2tex/PIL/tkinter. Ставит QT_QPA_PLATFORM=
# offscreen, глушит PIL.ImageGrab и выбор PyQt-бэкенда clipboard (pix2tex 0.1.4
# после КАЖДОГО инференса формулы пишет LaTeX в системный буфер обмена через
# pandas.io.clipboard — отсюда «мусор» J\vert\mathrm{e}\tilde... у пользователя).
from src.core.win_headless import install_gui_block
install_gui_block()

import flet as ft
from loguru import logger

# АУДИТ: здесь был импорт несуществующего класса `LamaLocaApp` (в src/ui/app.py
# определён только `main(page)`) — приложение падало с ImportError на старте.
# Единственная точка сборки UI — src.ui.app.main; main.py отвечает за проверку
# конфигурации, логирование и запуск event loop.
from src.ui.app import main as build_ui


def main(page: ft.Page):
    """Главная функция приложения."""
    # Настраиваем окно
    page.title = "lama-loca — персональная библиотека ПМИ"
    page.window.width = 1200
    page.window.height = 800
    page.window.min_width = 900
    page.window.min_height = 600

    page.theme_mode = ft.ThemeMode.DARK
    page.theme = ft.Theme(color_scheme_seed="blue", use_material3=True)

    # АУДИТ: config.yaml искался относительно текущей директории — запуск из
    # другой папки (двойной клик, планировщик, `flet run ..`) давал «файл не
    # найден». Ищем рядом с main.py, затем в cwd.
    candidates = [Path(__file__).resolve().parent / "config.yaml", Path("config.yaml")]
    config_path = next((c for c in candidates if c.exists()), None)
    if config_path is None:
        page.add(
            ft.Container(
                content=ft.Text(
                    "❌ Файл config.yaml не найден!\n"
                    "Скопируйте config.example.yaml в config.yaml",
                    color=ft.Colors.RED,
                    size=20,
                ),
                alignment=ft.Alignment.CENTER,
                expand=True,
            )
        )
        page.update()
        return

    try:
        logger.info("Запуск lama-loca UI")
        build_ui(page, config_path=config_path)
        logger.info("✅ Приложение успешно инициализировано")

    except Exception as e:
        logger.error(f"Ошибка инициализации: {e}")
        import traceback
        traceback.print_exc()
        page.add(
            ft.Container(
                content=ft.Text(f"❌ Ошибка: {e}", color=ft.Colors.RED, size=20),
                alignment=ft.Alignment.CENTER,
                expand=True,
            )
        )
        page.update()


if __name__ == "__main__":
    ft.run(main)