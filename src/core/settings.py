# src/core/settings.py
"""
Модуль конфигурации lama-loca.

Загружает настройки из config.yaml и предоставляет типизированный доступ
через dataclasses. Поддерживает все секции: app, storage, ollama, search,
chunking, external_tools, offline.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Optional

import yaml
from loguru import logger


# ---------------------------------------------------------------------------
# Вложенные конфигурации
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AppConfig:
    """Настройки приложения."""
    name: str = "lama-loca"
    language: str = "ru"
    # ТЗ «Кастомизация под студента БГУ ФПМиИ»: Ф.И.О. студента подставляется
    # в динамический заголовок окна (page.title) и редактируется в настройках.
    student_name: str = ""


@dataclass(frozen=True)
class StorageConfig:
    """Настройки хранилища."""
    root: str = "data"
    sqlite_path: str = "data/storage/library.db"
    chroma_path: str = "data/storage/chroma"


@dataclass(frozen=True)
class OllamaConfig:
    """Настройки подключения к Ollama."""
    base_url: str = "http://127.0.0.1:11434"
    chat_model: str = "qwen3:4b"  # дефолт приведён к существующему на Hub имени (qwen3.5:4b не существует)
    embedding_model: str = "bge-m3"
    timeout_seconds: int = 180
    # AUDIT (критический сбой «пустого ответа»): аппаратный лимит контекста
    # модели. Если у вас qwen3.5:4b (нативные 16384 токенов), укажите здесь
    # num_ctx_cap: 16384 — движок никогда не попросит Ollama больше и не
    # словит молчаливый done_reason='length' с пустым content. 0 = авто-детект
    # по /api/show с консервативным дефолтом 16384.
    num_ctx_cap: int = 0
    num_predict_cap: int = 0
    # PERF (критическое замедление на RTX 5060): Ollama по умолчанию выгружает
    # модель из VRAM через 5 минут простоя — каждый следующий запрос тратит
    # десятки секунд на повторную загрузку весов. keep_alive_seconds держит
    # модель в памяти. ВНИМАНИЕ (FIX HTTP 400 'missing unit in duration "-1"'):
    # значение -1 («бессрочно») многими сборками Ollama не принимается — код
    # автоматически подменит его на безопасные "1h". 0 = немедленная выгрузка,
    # положительное число = секунды ("<N>s"), -5 = не передавать параметр вовсе
    # (дефолт сервера).
    keep_alive_seconds: int = -5
    # Авто-детект GPU/CPU и предупреждение о CPU-режиме при старте.
    gpu_check: bool = True
    # AUDIT («немой» пустой ответ при num_ctx 19-23k): эффективный потолок
    # токенов ВХОДЯЩЕГО промпта — сколько модель reliably переживает без
    # done_reason='length' с пустым content. 0 = дефолт движка (12000).
    effective_prompt_tokens: int = 0
    # FIX (лог 16:39, конфликт VRAM Ollama vs MinerU): перед запуском MinerU
    # выгружать загруженные модели Ollama из VRAM (keep_alive=0s), чтобы
    # ONNX/torch парсера получили всю видеопамять и не сваливались в CPU-
    # режим ("GPU Memory: 1 GB"). После импорта модели перезагрузятся сами
    # при первой векторизации/вопросе. false — отключить выгрузку.
    unload_for_parse: bool = True
    # FIX (логи 14:32/14:55, «Ollama вернула ПУСТОЙ ответ; done_reason=
    # 'length', eval_count=None»): модели Qwen3/QwQ по умолчанию генерируют
    # скрытые reasoning-токены (поле thinking), которые не попадают в content
    # и съевают весь бюджет num_predict -> пустой ответ. Приложение само
    # передаёт think=False для таких моделей. Значения поля:
    #   null (нет поля) — автодетект по имени модели (рекомендуется);
    #   false — принудительно отключить reasoning даже для неизвестных моделей;
    #   true  — оставить thinking включённым (для chat-шаблонов, где reasoning
    #         виден в content; на Ollama <0.9 может давать пустые ответы).
    # Аналог: env OLLAMA_THINKING=0/1.
    thinking_enabled: Optional[bool] = None


@dataclass(frozen=True)
class SearchConfig:
    """Параметры гибридного поиска."""
    fts_top_k: int = 20
    vector_top_k: int = 20
    final_top_k: int = 8


@dataclass(frozen=True)
class ChunkingConfig:
    """Параметры разбиения текста на чанки."""
    target_tokens: int = 500
    overlap_tokens: int = 100
    min_tokens: int = 100
    max_tokens: int = 800
    preserve_headings: bool = True
    preserve_formulas: bool = True
    preserve_code_blocks: bool = True


@dataclass(frozen=True)
class ExternalToolsConfig:
    """Пути к внешним инструментам (OCR, конвертеры)."""
    tesseract_cmd: str = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    djvutxt_cmd: str = "djvutxt"
    ddjvu_cmd: str = "ddjvu"
    ebook_convert_cmd: str = "ebook-convert"


@dataclass(frozen=True)
class OfflineConfig:
    """Настройки офлайн-режима."""
    enabled: bool = True


@dataclass(frozen=True)
class FormulaOCRConfig:
    """Распознавание формул из PDF-изображений (Image-to-LaTeX).

    Текстовый слой PDF превращает математику в нечитаемую кашу — такой текст
    непригоден для обучения/RAG. formula_recognizer прогоняет PNG-зоны формул
    через локальную модель и возвращает LaTeX.
      engine: auto | pix2tex | texify | api | off
      allow_network_api: использовать сетевой фолбэк pix2tex-API (по умолчанию
          выключено — приложение офлайн-first).
    """
    engine: str = "auto"
    allow_network_api: bool = False


@dataclass(frozen=True)
class PdfEngineConfig:
    """Движок извлечения PDF (см. src/ingestion/mineru_parser.py).

    PyMuPDF — обычный текстовый экстрактор: сложная/инлайн-математика в нём
    превращается в кашу. MinerU анализирует страницы как картинки (layout +
    формульный детектор/распознаватель) и отдаёт Markdown с валидным LaTeX.
      engine: auto | mineru | pymupdf
        auto   — MinerU как основной, PyMuPDF+pix2tex как fallback (рекомендуется);
        mineru — только MinerU (ошибка импорта = ошибка импорта документа);
        pymupdf — старый пайплайн без MinerU.
      languages: языки для OCR-ветки MinerU (ru,en по умолчанию).
    """
    engine: str = "auto"
    languages: str = "ru,en"
    # FIX (лог 22:10, «ты сломал — раньше таких проблем не было»): MinerU сама
    # решает по TOTAL-объёму VRAM, сколько страниц гонять одним батчем
    # (get_vram() → Batch Ratio 1/2/4/8/16). На RTX 5060 (8 ГБ) это ratio=1 —
    # медленнее прежнего CPU-прохода. MINERU_VIRTUAL_VRAM_SIZE — штатный рычаг
    # MinerU для объявления «виртуального» размера VRAM; больше ratio — быстрее,
    # но выше риск CUDA OOM при занятой памяти. 0 = не трогать автоопределение.
    vram_gb_override: int = 0
    # true (дефолт) = выгружать модели Ollama из VRAM на время парсинга и
    # разрешать ускоренный батч; false = парсить без выгрузки и без override.
    fast_batch: bool = True


# ---------------------------------------------------------------------------
# Главный класс Settings
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Settings:
    """
    Корневой объект настроек lama-loca.

    Загружается из YAML-файла через Settings.load(). Все поля неизменяемы
    (frozen), что предотвращает случайное изменение конфигурации во время работы.
    """
    app: AppConfig = field(default_factory=AppConfig)
    # ТЗ п.1: Ф.И.О. студента продублировано на корневом уровне — Settings.load()
    # читает его из app.student_name, а UI-код обращается как settings.student_name.
    student_name: str = ""
    storage: StorageConfig = field(default_factory=StorageConfig)
    ollama: OllamaConfig = field(default_factory=OllamaConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    external_tools: ExternalToolsConfig = field(default_factory=ExternalToolsConfig)
    offline: OfflineConfig = field(default_factory=OfflineConfig)
    formula_ocr: FormulaOCRConfig = field(default_factory=FormulaOCRConfig)
    pdf_engine: PdfEngineConfig = field(default_factory=PdfEngineConfig)

    # -----------------------------------------------------------------------
    # Удобные свойства для прямого доступа к часто используемым путям
    # -----------------------------------------------------------------------

    @property
    def sqlite_path(self) -> str:
        """Путь к SQLite-базе."""
        return self.storage.sqlite_path

    @property
    def chroma_path(self) -> str:
        """Путь к хранилищу ChromaDB."""
        return self.storage.chroma_path

    @property
    def storage_root(self) -> str:
        """Корневая директория хранилища."""
        return self.storage.root

    # АУДИТ: PropertiesStorage (settings.sqlite_path / settings.chroma_path)
    # перекрывали одноимённые dataclass-поля StorageConfig — значения «строкой
    # ниже» фактически были недоступны. Добавлены явные альтернативы, чтобы
    # вызывающий код не зависел от порядка определения свойств.

    @property
    def sqlite_db_path(self) -> Path:
        """Путь к файлу SQLite (Path)."""
        return Path(self.storage.sqlite_path)

    @property
    def chroma_db_path(self) -> Path:
        """Путь к каталогу ChromaDB (Path)."""
        return Path(self.storage.chroma_path)

    @property
    def storage_root_path(self) -> Path:
        """Корень хранилища в виде Path (для операторов /)."""
        return Path(self.storage.root)

    @property
    def tesseract_cmd(self) -> str:
        """Путь к исполняемому файлу Tesseract."""
        return self.external_tools.tesseract_cmd

    # ------------------------------------------------------------------
    # ТЗ п.2: динамический заголовок окна
    # ------------------------------------------------------------------

    def window_title(self) -> str:
        """Заголовок окна с Ф.И.О. студента БГУ ФПМиИ.

        Используется при старте (src/ui/app.py) и при сохранении настроек
        «налету» (src/ui/settings_view.py) без перезапуска приложения.
        """
        fio = (self.student_name or self.app.student_name or "").strip()
        if not fio:
            fio = "(Ф.И.О. не указано)"
        return f"Lama Loca — личная библиотека студента БГУ ФПМиИ {fio}"

    # -----------------------------------------------------------------------
    # Загрузка из YAML
    # -----------------------------------------------------------------------

    @classmethod
    def load(cls, config_path: Path) -> "Settings":
        """
        Загружает настройки из YAML-файла.

        Args:
            config_path: Путь к файлу config.yaml.

        Returns:
            Объект Settings с заполненными полями.

        Raises:
            FileNotFoundError: Если config_path не существует.
            yaml.YAMLError: Если файл некорректен.
        """
        config_path = Path(config_path)

        if not config_path.exists():
            raise FileNotFoundError(
                f"Файл конфигурации не найден: {config_path.resolve()}\n"
                f"Скопируйте config.example.yaml в config.yaml и настройте пути."
            )

        logger.info(f"Загрузка конфигурации из {config_path}")

        with open(config_path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}

        # АУДИТ (баг «после перезапуска библиотека пуста»): пути в config.yaml
        # относительные ("data/storage/library.db"), и раньше они разрешались
        # относительно ТЕКУЩЕЙ ДИРЕКТОРИИ. Запуск из другой папки создавал
        # новую пустую БД рядом с cwd — импортированные книги «исчезали».
        # Теперь относительные пути привязаны к директории config.yaml
        # (он лежит в корне проекта рядом с main.py), абсолютные не трогаем.
        base_dir = config_path.resolve().parent
        storage_raw = raw.get("storage") or {}
        for key in ("root", "sqlite_path", "chroma_path"):
            val = storage_raw.get(key)
            if isinstance(val, str) and val and not Path(val).is_absolute():
                storage_raw[key] = str((base_dir / val).resolve())
        raw["storage"] = storage_raw

        return cls._from_dict(raw)

    @classmethod
    def _from_dict(cls, raw: dict) -> "Settings":
        """Создаёт Settings из словаря (внутренний метод)."""
        app_raw = raw.get("app", {})
        storage_raw = raw.get("storage", {})
        ollama_raw = raw.get("ollama", {})
        search_raw = raw.get("search", {})
        chunking_raw = raw.get("chunking", {})
        tools_raw = raw.get("external_tools", {})
        offline_raw = raw.get("offline", {})
        formula_ocr_raw = raw.get("formula_ocr", {})
        pdf_engine_raw = raw.get("pdf_engine", {})

        return cls(
            app=AppConfig(
                name=app_raw.get("name", "lama-loca"),
                language=app_raw.get("language", "ru"),
                student_name=str(app_raw.get("student_name", "") or ""),
            ),
            student_name=str(app_raw.get("student_name", "") or ""),
            storage=StorageConfig(
                root=storage_raw.get("root", "data"),
                sqlite_path=storage_raw.get("sqlite_path", "data/storage/library.db"),
                chroma_path=storage_raw.get("chroma_path", "data/storage/chroma"),
            ),
            ollama=OllamaConfig(
                base_url=ollama_raw.get("base_url", "http://127.0.0.1:11434"),
                chat_model=ollama_raw.get("chat_model", "qwen3:4b"),
                embedding_model=ollama_raw.get("embedding_model", "bge-m3"),
                timeout_seconds=int(ollama_raw.get("timeout_seconds", 180)),
                num_ctx_cap=int(ollama_raw.get("num_ctx_cap", 0)),
                num_predict_cap=int(ollama_raw.get("num_predict_cap", 0)),
                keep_alive_seconds=int(ollama_raw.get("keep_alive_seconds", -5)),
                gpu_check=bool(ollama_raw.get("gpu_check", True)),
                effective_prompt_tokens=int(
                    ollama_raw.get("effective_prompt_tokens", 0)),
                unload_for_parse=bool(ollama_raw.get("unload_for_parse", True)),
                # None = автодетект thinking по имени модели; true/false —
                # ручной override (см. OllamaConfig.thinking_enabled).
                thinking_enabled=(None if "thinking_enabled" not in ollama_raw
                                  else bool(ollama_raw.get(
                                      "thinking_enabled"))),
            ),
            search=SearchConfig(
                fts_top_k=int(search_raw.get("fts_top_k", 20)),
                vector_top_k=int(search_raw.get("vector_top_k", 20)),
                final_top_k=int(search_raw.get("final_top_k", 8)),
            ),
            chunking=ChunkingConfig(
                target_tokens=int(chunking_raw.get("target_tokens", 500)),
                overlap_tokens=int(chunking_raw.get("overlap_tokens", 100)),
                min_tokens=int(chunking_raw.get("min_tokens", 100)),
                max_tokens=int(chunking_raw.get("max_tokens", 800)),
                preserve_headings=bool(chunking_raw.get("preserve_headings", True)),
                preserve_formulas=bool(chunking_raw.get("preserve_formulas", True)),
                preserve_code_blocks=bool(chunking_raw.get("preserve_code_blocks", True)),
            ),
            external_tools=ExternalToolsConfig(
                tesseract_cmd=tools_raw.get(
                    "tesseract_cmd",
                    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
                ),
                djvutxt_cmd=tools_raw.get("djvutxt_cmd", "djvutxt"),
                ddjvu_cmd=tools_raw.get("ddjvu_cmd", "ddjvu"),
                ebook_convert_cmd=tools_raw.get("ebook_convert_cmd", "ebook-convert"),
            ),
            offline=OfflineConfig(
                enabled=bool(offline_raw.get("enabled", True)),
            ),
            formula_ocr=FormulaOCRConfig(
                engine=str(formula_ocr_raw.get("engine", "auto")),
                allow_network_api=bool(formula_ocr_raw.get("allow_network_api", False)),
            ),
            pdf_engine=PdfEngineConfig(
                engine=str(pdf_engine_raw.get("engine", "auto")).lower(),
                languages=str(pdf_engine_raw.get("languages", "ru,en")),
                vram_gb_override=int(pdf_engine_raw.get("vram_gb_override", 0) or 0),
                fast_batch=bool(pdf_engine_raw.get("fast_batch", True)),
            ),
        )

    # -----------------------------------------------------------------------
    # Утилиты
    # -----------------------------------------------------------------------

    def ensure_dirs(self) -> None:
        """Создаёт все необходимые директории, если их нет."""
        root = Path(self.storage.root)
        root.mkdir(parents=True, exist_ok=True)

        Path(self.sqlite_path).parent.mkdir(parents=True, exist_ok=True)
        Path(self.chroma_path).mkdir(parents=True, exist_ok=True)

        logger.debug(f"Директории хранилища готовы: {root.resolve()}")

    # ------------------------------------------------------------------
    # ТЗ п.4: сохранение Ф.И.О. студента в config.yaml («налету»)
    # ------------------------------------------------------------------

    def save_student_name(self, name: str, config_path: Optional[Path] = None) -> "Settings":
        """Записывает student_name в app секцию YAML и возвращает новый Settings.

        Config frozen dataclass — мутация запрещена архитектурой, поэтому
        обновление конфигурации выполняется перезаписью файла + перезагрузкой
        (dataclasses.replace для in-memory копии заголовка делает UI-слой).

        Args:
            name: Новое Ф.И.О. студента.
            config_path: Путь к config.yaml; по умолчанию — storage_root/../config.yaml.

        Returns:
            Новый объект Settings с обновлённым student_name.
        """
        if config_path is None:
            config_path = Path(self.storage.root).resolve().parent.parent / "config.yaml"
        config_path = Path(config_path)
        name = str(name or "").strip()

        raw: dict = {}
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                raw = yaml.safe_load(f) or {}
        app_raw = raw.get("app") or {}
        app_raw["student_name"] = name
        raw["app"] = app_raw

        try:
            with open(config_path, "w", encoding="utf-8") as f:
                yaml.safe_dump(raw, f, allow_unicode=True, sort_keys=False)
            logger.info(f"Сохранён student_name в {config_path}")
        except OSError as ex:
            logger.warning(f"Не удалось записать config.yaml ({ex}) — применяем только в памяти")

        new_app = AppConfig(
            name=self.app.name, language=self.app.language, student_name=name,
        )
        return replace(self, app=new_app, student_name=name)

    def dump(self) -> dict:
        """Сериализует настройки обратно в словарь (для отладки)."""
        return {
            "app": {
                "name": self.app.name,
                "language": self.app.language,
                "student_name": self.student_name or self.app.student_name,
            },
            "storage": {
                "root": self.storage.root,
                "sqlite_path": self.storage.sqlite_path,
                "chroma_path": self.storage.chroma_path,
            },
            "ollama": {
                "base_url": self.ollama.base_url,
                "chat_model": self.ollama.chat_model,
                "embedding_model": self.ollama.embedding_model,
                "timeout_seconds": self.ollama.timeout_seconds,
                "num_ctx_cap": self.ollama.num_ctx_cap,
                "num_predict_cap": self.ollama.num_predict_cap,
            },
            "search": {
                "fts_top_k": self.search.fts_top_k,
                "vector_top_k": self.search.vector_top_k,
                "final_top_k": self.search.final_top_k,
            },
            "chunking": {
                "target_tokens": self.chunking.target_tokens,
                "overlap_tokens": self.chunking.overlap_tokens,
                "min_tokens": self.chunking.min_tokens,
                "max_tokens": self.chunking.max_tokens,
                "preserve_headings": self.chunking.preserve_headings,
                "preserve_formulas": self.chunking.preserve_formulas,
                "preserve_code_blocks": self.chunking.preserve_code_blocks,
            },
            "external_tools": {
                "tesseract_cmd": self.external_tools.tesseract_cmd,
                "djvutxt_cmd": self.external_tools.djvutxt_cmd,
                "ddjvu_cmd": self.external_tools.ddjvu_cmd,
                "ebook_convert_cmd": self.external_tools.ebook_convert_cmd,
            },
            "offline": {
                "enabled": self.offline.enabled,
            },
        }


# ---------------------------------------------------------------------------
# Быстрый тест при запуске модуля
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cfg = Settings.load(Path("config.yaml"))
    cfg.ensure_dirs()

    print("\n=== Загруженная конфигурация ===")
    print(f"Приложение: {cfg.app.name} (язык: {cfg.app.language})")
    print(f"SQLite: {cfg.sqlite_path}")
    print(f"ChromaDB: {cfg.chroma_path}")
    print(f"Ollama: {cfg.ollama.base_url}")
    print(f"  чат-модель: {cfg.ollama.chat_model}")
    print(f"  эмбеддинги: {cfg.ollama.embedding_model}")
    print(f"Поиск: fts_top_k={cfg.search.fts_top_k}, "
          f"vector_top_k={cfg.search.vector_top_k}, "
          f"final_top_k={cfg.search.final_top_k}")
    print(f"Чанкинг: target={cfg.chunking.target_tokens}, "
          f"overlap={cfg.chunking.overlap_tokens}, "
          f"max={cfg.chunking.max_tokens}")
    print(f"Tesseract: {cfg.tesseract_cmd}")
    print(f"Офлайн-режим: {cfg.offline.enabled}")