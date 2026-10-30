"""
Управление VRAM Ollama: выгрузка моделей на время тяжёлого парсинга (MinerU).

Проблема, которую решает модуль (наблюдение пользователя, лог 16:39):
Ollama держит qwen3.5:4b + bge-m3 в VRAM (~4.8 ГБ), а ONNX Runtime / torch
внутри MinerU при старте не получает нужного объёма видеопамяти и молча
уходит в CPU-режим ("GPU Memory: 1 GB, Batch Ratio: 1") — импорт книги
идёт в разы медленнее и забивает RAM.

Решение: перед запуском MinerU выгрузить ВСЕ загруженные модели Ollama из
VRAM стандартным API-приёмом — POST /api/chat {"model": M, "messages": [],
"keep_alive": "0s"} (пустой запрос с нулевым keep_alive llama.cpp трактует
как команду немедленной выгрузки). После завершения парсинга ничего
«возвращать» не нужно: первая же векторизация/вопрос загрузкият модели
обратно (однократная загрузка весов ~секунды против минут CPU-инференса).

Модуль намеренно не имеет тяжёлых зависимостей и никогда не бросает
исключений наружу: вызывающий код (импорт PDF) обязан работать даже при
полной недоступности Ollama.
"""
from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request

from loguru import logger

#: Кэш base_url по умолчанию — берём из конфига приложения, если он уже
#: загружен; иначе localhost-дефолт Ollama.
_DEFAULT_URL = "http://127.0.0.1:11434"

_lock = threading.Lock()

#: Глубина вложенности release_vram (для отложенной паузы после выгрузки).
_depth = 0


def _wait_for_release(base: str, timeout_s: float = 8.0) -> None:
    """Ждём фактического освобождения VRAM llama.cpp после keep_alive=0s.

    FIX (лог 21:39): POST /api/chat c keep_alive="0s" асинхронен — Ollama
    отвечает мгновенно, но слоты и память llama.cpp освобождаются через
    секунды. MinerU стартовала сразу после unload_all и всё ещё видела
    занятую карту ('GPU Memory: 1 GB'). Опрашиваем /api/ps до пустого списка
    (или таймаута), чтобы парсер стартовал уже на свободной видеопамяти.
    """
    import time as _t
    deadline = _t.monotonic() + timeout_s
    while _t.monotonic() < deadline:
        if not loaded_models(base):
            return
        _t.sleep(0.4)
    logger.debug("ollama_vram: модели всё ещё заняты слоты по истечении паузы")

_CFG_UNSET = object()
_CFG_CACHE = _CFG_UNSET


def _ollama_cfg():
    """Оллама-секция конфига приложения (best effort, без повторных чтений диска).

    FIX (лог 21:27): раньше здесь каждый раз дёргался Settings.load() с диска
    (в логе импорта это давало лишние строки 'Загрузка конфигурации из
    config.yaml' и зависело от CWD). Результат кэшируется в модуле.
    """
    global _CFG_CACHE
    if _CFG_CACHE is not _CFG_UNSET:
        return _CFG_CACHE
    cfg = None
    try:
        from src.core.settings import Settings
        cfg = getattr(Settings.load(__import__("pathlib").Path("config.yaml")),
                      "ollama", None)
    except Exception:  # noqa: BLE001 — конфиг опционален
        cfg = None
    _CFG_CACHE = cfg
    return cfg


def _default_base_url() -> str:
    cfg = _ollama_cfg()
    url = getattr(cfg, "base_url", "") or ""
    return url.rstrip("/") if url else _DEFAULT_URL


def _request(url: str, payload: dict | None = None, timeout: float = 10.0):
    # FIX (лог 21:27, тихий no-op выгрузки): у Request нет метода add_header —
    # был вызов getattr(req, "add_header", req.__class__) и на GET-запросе
    # (payload=None) в качестве заголовка передавался САМ КЛАСС Request.
    # urllib бросал TypeError ещё до urlopen, /api/ps всегда возвращал [],
    # unload_all не выгружал ничего и молча завершался — MinerU по-прежнему
    # получала "GPU Memory: 1 GB". Заголовки теперь задаются конструктором.
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(
        url, data=data,
        headers={"Content-Type": "application/json"} if data else {},
        method="POST" if data else "GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode("utf-8", errors="replace")
    return json.loads(body) if body.strip() else {}


def loaded_models(base_url: str | None = None) -> list[str]:
    """Имена моделей, сейчас держащих слот/VRAM в Ollama (/api/ps)."""
    base = (base_url or _default_base_url()).rstrip("/")
    try:
        ps = _request(f"{base}/api/ps", timeout=5)
        names = []
        for m in ps.get("models") or []:
            n = m.get("name")
            if n:
                names.append(n)
        return names
    except Exception as e:  # noqa: BLE001
        logger.debug(f"ollama_vram: /api/ps недоступен ({e})")
        return []


def unload_all(base_url: str | None = None) -> list[str]:
    """Выгружает все загруженные модели Ollama из VRAM.

    Возвращает список имён реально выгруженных моделей (пустой — если в VRAM
    ничего не было или Ollama недоступна). Исключений не бросает.
    """
    base = (base_url or _default_base_url()).rstrip("/")
    unloaded: list[str] = []
    for name in loaded_models(base):
        try:
            # Пустой messages + keep_alive=0s — канонический способ выгрузки
            # (ollama CLI `ollama stop <model>` делает ровно этот запрос).
            _request(f"{base}/api/chat",
                     {"model": name, "messages": [], "keep_alive": "0s"},
                     timeout=15)
            unloaded.append(name)
        except urllib.error.HTTPError as e:
            # Старые сборки Ollama могут не принимать пустой chat — пробуем
            # /api/generate тем же приёмом.
            try:
                _request(f"{base}/api/generate",
                         {"model": name, "prompt": "", "keep_alive": "0s"},
                         timeout=15)
                unloaded.append(name)
            except Exception as e2:  # noqa: BLE001
                logger.debug(f"ollama_vram: выгрузка '{name}' не удалась ({e}; {e2})")
        except Exception as e:  # noqa: BLE001
            logger.debug(f"ollama_vram: выгрузка '{name}' не удалась ({e})")
    if unloaded:
        logger.info(
            "Ollama: модели выгружены из VRAM на время работы MinerU: "
            + ", ".join(unloaded))
    return unloaded


class release_vram:
    """Контекст-менеджер: на время блока VRAM Ollama освобождена.

    Использование::

        with release_vram():          # no-op при ollama.unload_for_parse=false
            mineru_run(...)          # получает всю VRAM себе

    Идемпотентен и потокобезопасен (вложенные вызовы внутри одного потока
    выгружают только на внешнем входе). Никаких исключений наружу.
    """

    def __init__(self, enabled: bool | None = None, base_url: str | None = None):
        self._base = base_url
        if enabled is None:
            enabled = True
            cfg = _ollama_cfg()
            ka = getattr(cfg, "unload_for_parse", None)
            if ka is not None:
                enabled = bool(ka)
        self._enabled = enabled
        self._unloaded: list[str] = []

    def __enter__(self):
        global _depth
        if not self._enabled:
            return self
        with _lock:
            self._unloaded = unload_all(self._base)
            _depth += 1
            first = (_depth == 1)
        # Пауза до фактического освобождения VRAM — только на внешнем входе,
        # чтобы вложенные блоки не ждали повторно.
        if first and self._unloaded:
            try:
                _wait_for_release((self._base or _default_base_url()).rstrip("/"))
            except Exception:  # noqa: BLE001 — пауза не валяет импорт
                pass
        return self

    def __exit__(self, exc_type, exc, tb):
        global _depth
        if self._enabled and self._unloaded:
            with _lock:
                _depth = max(0, _depth - 1)
        # Модели перезагрузятся сами при первом обращении (embed/chat).
        # Ничего делать не нужно; возвращаем False — не глотаем чужие ошибки.
        return False
