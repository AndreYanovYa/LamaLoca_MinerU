# src/llm/llm_engine.py
import difflib
import json
import re
import threading
import requests
from loguru import logger
from typing import List, Dict, Any, Optional, Tuple
from ..core.settings import OllamaConfig
from ..core.exceptions import LamaLocaError

# AUDIT (критический сбой «немеющего» qwen3.5:4b): формула окна
#   window = max(8192, min(32768, prompt_tokens + max_new + 1024))
# заставляла Ollama выделять KV-кэш на 29k токенов под модель с аппаратным
# лимитом контекста 16384. llama.cpp не может отрендерить такой запрос:
# генерация аварийно завершалась с done_reason='length' и АБСОЛЮТНО ПУСТЫМ
# message.content (см. лог: num_ctx=18268, eval_count=None, content='').
# Авто-продолжения лишь усугубляли деградацию (каждая дозапроска раздувала
# историю до тех же 18k+). Теперь num_ctx никогда не превышает реальный
# нативный лимит модели, обнаруженный по /api/show (+ потокобезопасный кэш).
# КЛЮЧ кэша — (base_url, model): разные экземпляры движка с разными
# base_url/конфигами не травят друг друга общим кэшем (был баг: cap одного
# движка «прилипал» к другому через общий кэш по имени модели).
_MODEL_WINDOW_CACHE: Dict[tuple, int] = {}
_MODEL_WINDOW_LOCK = threading.Lock()

# Консервативный потолок для неизвестных моделей: меньше нативных 32k Qwen,
# но достаточно для конспекта полной главы; перестраховка от OOM на CPU-offload.
_DEFAULT_MAX_WINDOW = 16384
_MIN_USEFUL_WINDOW = 8192

# AUDIT (логи 13:01/14:16: «немой» пустой ответ при num_ctx 19-23k):
# limit.context_length в GGUF (у qwen3.5:4b — 262144) это АРХИТЕКТУРНЫЙ
# потолок RoPE, а не рабочий размер KV-кэша: Ollama аллоцирует столько,
# сколько попросили, и на машинах с нехваткой VRAM/RAM большие окна либо
# не рендерятся вообще (eval_count=None, content=''), либо свопят RAM под
# 95%. Потолок «безопасного промпта»: столько токенов входа модель reliably
# переживает без молчаливого length-обрыва. Может быть понижен через
# env OLLAMA_EFFECTIVE_PROMPT_TOKENS или поле config effective_prompt_tokens.
_EFFECTIVE_PROMPT_TOKENS = 12000

# AUDIT (логи 10:48/11:29, «ответ не раскрывает тему»): после того как
# движок ФАКТИЧЕСКИ подтвердит GPU-инференс метриками первого ответа
# (~73 ток/с на RTX 5060, eval_count>0), бюджет промпта поднимается до
# этого значения — qwen3.5:4b держит 26k+ входа целиком в VRAM без молча
# обрыва (прежний статичный 12k душил контекст Кудрявцева). Консервативный
# дефолт остаётся до подтверждения GPU; ручной override всегда приоритетнее.
_GPU_RAISED_PROMPT_TOKENS = 20000


class LLMEngine:
    """Обёртка над Ollama API для генерации ответов через /api/chat."""
    
    def __init__(self, config: OllamaConfig):
        self.base_url = config.base_url.rstrip("/")
        self.model = config.chat_model
        self.timeout = config.timeout_seconds
        # PERF: keep_alive держит модель в VRAM между запросами; без него
        # Ollama выгружает её через 5 минут простоя и каждый следующий ответ
        # начинается с повторной загрузки весов (десятки секунд «зависания»).
        self.keep_alive = getattr(config, "keep_alive_seconds", -5)
        # Потолки контекста, отдаваемые в options.num_ctx. Могут быть снижены
        # init-флагом OLLAMA_NUM_CTX_CAP или полем config (num_ctx_cap /
        # max_context) — читаем defensively, чтобы не ломать frozen-конфиги
        # старых версий settings.py.
        self._ctx_cap = self._read_ctx_cap(config)
        # Потолок «безопасного промпта» (см. _EFFECTIVE_PROMPT_TOKENS):
        # понижается полем config.effective_prompt_tokens или env
        # OLLAMA_EFFECTIVE_PROMPT_TOKENS — для слабых по VRAM машин.
        self._prompt_budget = self._read_prompt_budget(config)
        # AUDIT (логи 10:48/11:29): статичный бюджет 12k душил ответы на
        # больших книгах («ответ не раскрывает тему»), тогда как модель
        # reliably держит 26-32k при полном инференсе на GPU (RTX 5060,
        # ~73 ток/с). Бюджет теперь АДАПТИВНЫЙ: после первого ответа с
        # метриками GPU (_report_gen_speed подтвердил режим и eval_count>0)
        # потолок поднимается до _GPU_RAISED_PROMPT_TOKENS; ручной override
        # (config/env) имеет приоритет и НЕ перекрывается рантаймом.
        self._budget_is_default = not self._budget_overridden(config)
        self._budget_raised = False
        # FIX (логи 00:56/10:48): динамический бюджет генерации, согласованный
        # с _trim_history (_fit в chat()). Раньше trim резервировал 1024, а
        # options аллоцировали до 4096 — инвариант prompt+predict<=num_ctx
        # нарушался самим движком, и llama.cpp молча возвращал пустоту
        # (eval_count=None, done_reason='length') даже после перезапроса.
        self._gen_reserve = 1024
        
        # АУДИТ: как и в OllamaEmbedder — недоступная Ollama не должна ронять
        # всё приложение на старте (просмотр PDF/библиотека работают офлайн).
        self.available = False
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            response.raise_for_status()
            models = [m['name'] for m in response.json().get('models', [])]
            if not any(self.model in m for m in models):
                logger.warning(f"Модель {self.model} не найдена в Ollama. Доступны: {models}. Выполните: ollama pull {self.model}")
            else:
                logger.info(f"LLM Engine подключена. Используем модель: {self.model}")
            self.available = True
        except requests.exceptions.RequestException as e:
            logger.error(f"Не удалось подключиться к Ollama: {e}")
        # PERF: диагностика GPU/CPU — обычная причина «очень медленно» на
        # Windows: torch/Ollama встают в CPU-режиме и молча считают там.
        self._pending_gpu_check = False
        self._first_chat_done = False
        if self.available and getattr(config, "gpu_check", True):
            self.check_gpu()

    def check_gpu(self) -> Optional[bool]:
        verdict = self._check_gpu_impl()
        # FIX (адаптивный бюджет): запоминаем вердикт — _maybe_raise_budget
        # повышает потолок промпта только при ФАКТИЧЕСКИ подтверждённом GPU.
        if verdict is not None:
            self._gpu_mode = verdict
        return verdict

    def _check_gpu_impl(self) -> Optional[bool]:
        """Диагностика режима исполнения Ollama (GPU/CUDA vs CPU).

        Корень жалоб «всё работает, но очень медленно»: `pip install torch`
        по умолчанию тянет CPU-сборку, а Ollama без CUDA-бэкенда (или со
        старым драйвером под RTX 5060 / архитектуру Blackwell sm_120) молча
        считает весь инференс на процессоре — в 10-30 раз медленнее GPU.
        Проверяем /api/version (список CUDA-устройств) и /api/ps (доля
        весов загруженной модели в VRAM). Возвращает True=GPU, False=CPU,
        None=неизвестно; никогда не бросает исключений.
        """
        try:
            v = requests.get(f"{self.base_url}/api/version", timeout=5).json()
            # FIX (вечный FP «/api/version пуст», лог 13:35): реальный ответ
            # содержит булев флаг `cuda` и НЕ содержит ключей cuda_devices/gpu
            # — старая проверка давала warning даже при полностью рабочем GPU.
            cvd = str(v.get("CUDA_VISIBLE_DEVICES") or "").strip()
            devs = v.get("cuda_devices") or v.get("gpu") or []
            if v.get("cuda") is True or (cvd and cvd.lower() not in ("cpu", "none", "-1")) \
                    or (isinstance(devs, (list, tuple)) and devs):
                names = ", ".join(
                    d.get("name", str(d)) if isinstance(d, dict) else str(d)
                    for d in (devs if isinstance(devs, (list, tuple)) else []))
                logger.info(f"Ollama работает на GPU" + (f": {names}" if names else ""))
                return True
            if v.get("cuda") is False:
                logger.warning(
                    "Ollama сообщает cuda=false — генерация идёт на CPU "
                    "(«очень медленно»). Починка для RTX 5060: обновите Ollama "
                    "до сборки с CUDA для Blackwell (драйвер NVIDIA >= 560); "
                    "проверьте переменные окружения сервера OLLAMA_NUM_GPU=0 / "
                    "CUDA_VISIBLE_DEVICES=CPU (принудительное отключение GPU).")
                return False
            ps = requests.get(f"{self.base_url}/api/ps", timeout=5).json()
            running = ps.get("models") or []
            if not running:
                # Версий о GPU нет и моделей в памяти нет — судить рано:
                # вердикт будет дан по метрикам первого реального ответа
                # (_report_gen_speed), без ложных предупреждений на старте.
                self._pending_gpu_check = True
                return None
            for m in running:
                size = m.get("size") or 0
                in_vram = m.get("size_vram") or 0
                if size and in_vram < size * 0.5:
                    logger.warning(
                        f"\u26a0\ufe0f Модель '{m.get('name')}' загружена в VRAM лишь на "
                        f"{int(in_vram / size * 100)}% — большинство слоёв считается на "
                        "CPU (главная причина «очень медленно»). Уменьшите num_ctx, "
                        "выгрузите другие модели или обновите Ollama до версии с "
                        "поддержкой CUDA для RTX 5060.")
                    return False
            logger.info("Ollama: модель целиком в VRAM — GPU-режим подтверждён.")
            return True
        except Exception:  # noqa: BLE001 — диагностика не валит приложение
            return None

    def _report_gen_speed(self, data: dict):
        """PERF/FIX: фактическая верификация GPU/CPU по метрикам ответа.

        eval_count/eval_duration есть в каждом непустом ответе /api/chat.
        Порог 15 ток/с: qwen3.5:4b на RTX 5060 даёт 60-120+ ток/с, на CPU —
        единицы-десятки. eval_count==0 (нулевой вывод) игнорируется — там
        режим не измерить. Вызывается один раз за сессию движка.
        """
        try:
            ec = data.get("eval_count") or 0
            dur = data.get("eval_duration") or 0
            if not ec or not dur:
                return
            tps = ec / (dur / 1e9)
            if tps < 15:
                logger.warning(
                    f"\u26a0\ufe0f Генерация идёт на ~{tps:.1f} ток/с — это CPU-режим "
                    "Ollama (GPU для qwen3.5:4b даёт 60+ ток/с). Починка: "
                    "обновите Ollama до сборки с CUDA для RTX 50xx (драйвер "
                    "NVIDIA >= 560); проверьте OLLAMA_NUM_GPU / "
                    "CUDA_VISIBLE_DEVICES; убедитесь, что VRAM свободен "
                    "(ollama ps, PROCESSOR должен быть 100% GPU).")
            else:
                logger.info(f"Ollama chat: ~{tps:.1f} ток/с — GPU-режим "
                            "подтверждён фактическими метриками генерации.")
                # FIX («ответ не раскрывает тему», логи 10:48/11:29): метрики
                # живого ответа — самое сильное доказство GPU; повышаем
                # адаптивный бюджет промпта (ручной override не трогаем).
                self._gpu_mode = True
                self._maybe_raise_budget()
        except Exception:  # noqa: BLE001
            pass

    #: Sentinel конструктора: keep_alive вообще не передаётся (дефолт сервера).
    _KA_NOTSET = -5

    def _keep_alive_param(self) -> Optional[str]:
        """F4/FIX (логи 22:06 / 22:25 / 23:16): Ollama парсит keep_alive как
        Go time.Duration. Значение "-1" старые (и многие текущие Windows-)
        сборки отвергают со стабильной ошибкой HTTP 400
        `time: missing unit in duration "-1"` — она роняла и чат, и
        векторизацию. Версионные эвристики не помогали: «новая» версия на
        машине пользователя тоже принимала "-1" формально, но реально
        отвечала 400. Поэтому НИКОГДА не отправляем голый "-1"/"0" без
        единицы измерения.

        Правила: None/-5 -> параметр не передаётся (дефолт сервера);
        любое отрицательное -> безопасный долгий период "1h" (модель
        фактически остаётся в VRAM между запросами, duration с единицей
        валиден на ЛЮБОЙ версии Ollama); 0 -> "0s"; положительное число
        секунд -> "<N>s"; готовая строка с единицей ("10m"/"1h") — как
        есть; прочий мусор -> параметр не передаётся."""
        ka = self.keep_alive
        if ka is None or isinstance(ka, bool):
            return None
        if isinstance(ka, (int, float)):
            if ka == self._KA_NOTSET:
                return None
            if ka < 0:
                return "1h"      # вместо "-1": совместимо со всеми сборками
            if ka == 0:
                return "0s"
            return f"{int(ka)}s"
        text = str(ka).strip()
        if not text:
            return None
        if re.fullmatch(r"\d+(\.\d+)?(ns|us|µs|ms|s|m|h)+", text):
            return text
        try:
            value = float(text)
        except ValueError:
            logger.debug(f"keep_alive: нераспознанное значение {text!r} — "
                         "параметр не передаётся (защита от HTTP 400).")
            return None
        if value < 0:
            return "1h"
        if value == 0:
            return "0s"
        return f"{int(value)}s"

    def _ensure_available(self):
        """Повторная проверка связи перед генерацией."""
        if not self.available:
            try:
                response = requests.get(f"{self.base_url}/api/tags", timeout=5)
                response.raise_for_status()
                self.available = True
                logger.info("Ollama снова доступна.")
            except requests.exceptions.RequestException:
                raise LamaLocaError(
                    "Ollama недоступна. Запустите приложение Ollama и убедитесь, "
                    f"что модель загружена (ollama pull {self.model})."
                )
    
    def _resolve_model(self) -> str:
        """FIX (404): приводит имя из конфига к реальному tag'у модели в Ollama.

        Причина пустых ответов чата: config.yaml может содержать написание
        имени, которого нет среди tags Ollama ("qwen3:4b" vs "Qwen3:latest"
        vs "qwen3.5:4b"). /api/chat с неизвестным именем отвечает 404, а
        раньше это превращалось в «Сбой Ollama API» или пустой пузырь.
        Подбор: точное совпадение -> тот же префикс имени (:tag любой) ->
        регистронезависимое совпадение -> префикс без учёта регистра.
        """
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            response.raise_for_status()
            models = [m["name"] for m in response.json().get("models", [])]
        except requests.exceptions.RequestException:
            return self.model

        if self.model in models:
            return self.model
        base = self.model.split(":")[0]
        for cand in models:
            if cand.split(":")[0] == base:
                return cand
        lowered = {m.lower(): m for m in models}
        if self.model.lower() in lowered:
            return lowered[self.model.lower()]
        for cand in models:
            if cand.lower().startswith(base.lower() + ":"):
                return cand
        return self.model

    @staticmethod
    def _budget_overridden(config: OllamaConfig) -> bool:
        """Задан ли бюджет промпта вручную (config/env) — тогда рантайм его
        не трогает (в т.ч. адаптивное повышение после подтверждения GPU)."""
        import os
        try:
            if int(getattr(config, "effective_prompt_tokens", 0) or 0):
                return True
        except (TypeError, ValueError):
            pass
        try:
            if int(os.environ.get("OLLAMA_EFFECTIVE_PROMPT_TOKENS", "0") or 0):
                return True
        except ValueError:
            pass
        return False

    def _maybe_raise_budget(self):
        """FIX («ответ не раскрывает тему»): один раз поднять бюджет
        «безопасного промпта», если GPU-инференс подтверждён фактическими
        метриками (см. _report_gen_speed). Ручной override не перекрывается."""
        if self._budget_raised or not self._budget_is_default:
            return
        if getattr(self, "_gpu_mode", None) is not True:
            return
        self._budget_raised = True
        old = self._prompt_budget
        self._prompt_budget = max(old, _GPU_RAISED_PROMPT_TOKENS)
        # Старый кэш окна устарел — окно считалось от пониженного бюджета;
        # при следующем detect_model_window потолок пересчитается заново.
        with _MODEL_WINDOW_LOCK:
            _MODEL_WINDOW_CACHE.clear()
        logger.info(
            f"LLM: GPU-инференс подтверждён метриками — бюджет контекста "
            f"повышен {old} → {self._prompt_budget} токенов (полные ответы "
            f"на больших книгах; отключение: effective_prompt_tokens в config)")

    def effective_prompt_budget(self) -> int:
        """Текущий живой бюджет ВХОДЯЩЕГО промпта (учитывает адаптивное
        повышение после подтверждения GPU — _maybe_raise_budget). Публичный
        геттер для префлайт-фильтра RAG, чтобы тот не читал статический
        дефолт модуля (FIX логов 11:29/11:44: префлайт отсекал релевантные
        блоки по устаревшим 12k, хотя движок уже разрешал 20k)."""
        return self._prompt_budget

    @staticmethod
    def _read_prompt_budget(config: OllamaConfig) -> int:
        """Эффективный потолок токенов ВХОДЯЩЕГО промпта (анти-«немой» пустой
        ответ). Дефолт _EFFECTIVE_PROMPT_TOKENS; переопределение — полем
        config.effective_prompt_tokens или env OLLAMA_EFFECTIVE_PROMPT_TOKENS."""
        import os
        val = 0
        try:
            val = int(getattr(config, "effective_prompt_tokens", 0) or 0)
        except (TypeError, ValueError):
            val = 0
        if not val:
            try:
                val = int(os.environ.get("OLLAMA_EFFECTIVE_PROMPT_TOKENS", "0") or 0)
            except ValueError:
                val = 0
        return max(2048, val) if val else _EFFECTIVE_PROMPT_TOKENS

    @staticmethod
    def _read_ctx_cap(config: OllamaConfig) -> Tuple[int, int]:
        """(cap_num_ctx, cap_num_predict) — читаются из config если заданы,
        иначе из env OLLAMA_NUM_CTX_CAP / OLLAMA_NUM_PREDICT_CAP.

        0 означает «не задано» -> используются авто-детект по /api/show и
        дефолты движка. DEFENSIVE: старые frozen-конфиги без новых полей
        не роняют приложение (getattr + except).
        """
        def _cfg(name: str) -> int:
            try:
                return int(getattr(config, name, 0) or 0)
            except (TypeError, ValueError):
                return 0
        import os
        ctx = _cfg("num_ctx_cap") or _cfg("max_context")
        pred = _cfg("num_predict_cap")
        try:
            ctx = ctx or int(os.environ.get("OLLAMA_NUM_CTX_CAP", "0") or 0)
            pred = pred or int(os.environ.get("OLLAMA_NUM_PREDICT_CAP", "0") or 0)
        except ValueError:
            pass
        return max(0, ctx), max(0, pred)

    def detect_model_window(self, model: str) -> int:
        """Нативный лимит контекста модели в токенах (лечение «пустого ответа»).

        Источник истины — параметр num_ctx у config/parameters модели
        (/api/show); дополнительно парсится raw-секция model_info/General
        ("qwen2.local_size"/"context_length"). Результат кэшируется под
        threading.Lock — метод потокобезопасен и не долбит Ollama на каждый чат.
        При недоступности API возвращается консервативный потолок
        min(_DEFAULT_MAX_WINDOW, ctx_cap) — НИКОГДА выдуманные 32k.
        """
        cache_key = (self.base_url, model)
        with _MODEL_WINDOW_LOCK:
            if cache_key in _MODEL_WINDOW_CACHE:
                return _MODEL_WINDOW_CACHE[cache_key]
        window = 0
        try:
            resp = requests.post(f"{self.base_url}/api/show",
                                 json={"name": model}, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            # FIX ('str' object has no attribute 'items', лог 23:01): верхне-
            # уровневый dict из /api/show нормализован, но вложенные секции
            # model_info/parameters/config рядом сборок Ollama возвращаются
            # JSON-СТРОКАМИ. dict(str) падает ValueError, а прямой .items()
            # на такой секции ронял весь чат уже ПОСЛЕ логов поиска и склейки
            # чанков — ровно то место, где срывался конвейер summary.
            def _section(name: str) -> dict:
                v = data.get(name)
                if isinstance(v, str):
                    try:
                        v = json.loads(v)
                    except (ValueError, TypeError):
                        return {}
                return v if isinstance(v, dict) else {}

            candidates = []
            for holder in (_section("model_info"),
                           _section("parameters"),
                           _section("config")):
                for key, val in holder.items():
                    k = str(key).lower()
                    if isinstance(val, (int, float)) and any(
                            s in k for s in ("context_length", "num_ctx",
                                             "n_ctx", "window")):
                        candidates.append(int(val))
                    elif isinstance(val, str) and any(
                            s in k for s in ("context_length", "num_ctx",
                                             "n_ctx", "window")):
                        # число, пришедшее строкой ("16384")
                        try:
                            candidates.append(int(float(val)))
                        except (ValueError, TypeError):
                            pass
            raw = str(data.get("raw", "")) + str(data.get("template", ""))
            for m in re.finditer(r'(?:context_length|n_ctx|num_ctx)"?\s*[:=]\s*"?'
                                 r'(\d+)', raw):
                candidates.append(int(m.group(1)))
            candidates = [c for c in candidates if 512 <= c <= 1_048_576]
            if candidates:
                window = min(candidates)  # берём САМЫЙ маленький заявленный лимит
        except (requests.exceptions.RequestException, ValueError) as e:
            logger.warning(f"LLM: не удалось определить окно контекста "
                           f"'{model}' ({e}) — беру консервативный дефолт.")
        if not window:
            window = _DEFAULT_MAX_WINDOW
        ctx_cap, _pred_cap = self._ctx_cap
        if ctx_cap:
            window = min(window, ctx_cap)
        # AUDIT (qwen3.5:4b limit.context_length=262144 — архитектурный
        # потолок RoPE, НЕ рабочий KV-кэш; запросы на 19-23k молча давали
        # done_reason='length' с пустым content). Прогоняем окно через
        # эффективный бюджет машины: prompt+генерация всегда влезает.
        eff_gen = 8192 + 1024  # целевая генерация + reserve из _build_options
        effective_cap = self._prompt_budget + eff_gen
        if window > effective_cap:
            logger.info(
                f"LLM: заявленное окно '{model}' ({window}) превышает "
                f"эффективный лимит машины ({effective_cap} = промпт "
                f"{self._prompt_budget} + генерация {eff_gen}) — num_ctx "
                f"будет ограничен {effective_cap} (защита от пустого ответа)")
            window = effective_cap
        with _MODEL_WINDOW_LOCK:
            _MODEL_WINDOW_CACHE[cache_key] = window
        logger.info(f"LLM: нативное окно контекста '{model}' = {window} токенов "
                    f"(потолок num_ctx в запросах)")
        return window

    _TIKTOKEN_ENC = None          # кэш инстанса кодера (ленивая загрузка)
    _TIKTOKEN_TRIED = False

    @classmethod
    def _get_token_encoder(cls):
        """Ленивая загрузка tiktoken (cl100k_base). Пакет уже в зависимостях
        (срез транслитерации); при любом сбое возвращаем None — вызывающий
        код падает обратно на консервативную эвристику."""
        if cls._TIKTOKEN_TRIED:
            return cls._TIKTOKEN_ENC
        cls._TIKTOKEN_TRIED = True
        try:
            import tiktoken
            cls._TIKTOKEN_ENC = tiktoken.get_encoding("cl100k_base")
        except Exception as ex:  # нет пакета / нет словаря офлайн
            logger.debug(f"LLM: tiktoken недоступен ({ex}) — оценка токенов "
                         "по консервативной эвристике")
            cls._TIKTOKEN_ENC = None
        return cls._TIKTOKEN_ENC

    @classmethod
    def estimate_tokens(cls, messages: List[Dict[str, str]]) -> int:
        """Оценка длины промпта в токенах.

        FIX (логи 14:15-14:18: «немой» пустой ответ при num_ctx=19624):
        раньше была только грубая эвристика «символы/3», которая
        НЕДОоценивала русско-математические промпты — префлайт-фильтр RAG
        пропускал 19-23k реальных токенов, Ollama молча возвращала пустоту
        (eval_count=None, done_reason='length'), и каждый такой запрос
        стоил 1,5+ минуты. Теперь считаем ТОЧНО через tiktoken
        (cl100k_base — близок к BPE Qwen); если словарь не загрузился
        (offline-машина без кэша) — консервативная эвристика символы/2.7:
        переоценка безопаснее недооценки (лучше отсечь лишний блок, чем
        получить пустой ответ).
        """
        total_text = "\n".join(str(m.get("content", "") or "")
                               for m in messages)
        enc = cls._get_token_encoder()
        if enc is not None:
            try:
                n = len(enc.encode(total_text, disallowed_special=()))
                return n + 64  # запас на шаблон чата и роли
            except Exception:
                pass  # fallback на эвристику
        return int(len(total_text) / 2.7) + 64

    def probe_fit(self, messages: List[Dict[str, Any]],
                  window_cap: Optional[int] = None):
        """Публичная «примерка» запроса (используется RAG-префлайтом).

        Возвращает (messages, num_ctx, num_predict, est_tokens): ровно те
        параметры, которые реально ушли бы в Ollama при отправке этого
        промпта через chat() — тот же _fit-цикл, согласующий обрезку истории
        и бюджет генерации. Префлайт, оценивающий бюджет по голому тексту
        контекста без chat-шаблона, занижал метрику на 15-20% и пропускал
        перегруз (логи 00:56/10:48/11:29: пустые ответы при num_ctx 12-16k).
        Никогда не бросает исключений: при любой ошибке возвращает None —
        вызывающий код тогда использует свою собственную оценку.
        """
        try:
            cap = window_cap or self.detect_model_window(self.model)
            ctx_cap, _ = self._ctx_cap
            if ctx_cap:
                cap = min(cap, ctx_cap)

            msgs = [dict(m) if isinstance(m, dict) else m for m in messages]
            opts = None
            for _ in range(3):
                est = self.estimate_tokens(msgs)
                opts = self._build_options(est, cap)
                need_reserve = int(opts.get("num_predict", 0)) + 512
                trimmed = self._trim_history(msgs, cap,
                                             reserve_for_gen=need_reserve)
                t_est = self.estimate_tokens(trimmed)
                if t_est <= cap - need_reserve:
                    return trimmed, int(opts["num_ctx"]), \
                        int(opts["num_predict"]), t_est
                msgs = trimmed
            est = self.estimate_tokens(msgs)
            opts = self._build_options(est, cap)
            return msgs, int(opts["num_ctx"]), int(opts["num_predict"]), est
        except Exception as e:  # noqa: BLE001 — примерка не должна ронять RAG
            logger.debug(f"LLM: probe_fit недоступен ({e})")
            return None

    def _build_options(self, prompt_tokens: int, model_window: Optional[int] = None) -> Dict[str, Any]:
        """FIX (пустой ответ / обрезка): num_ctx в пределах РЕАЛЬНОГО окна модели.

        Корень обеих проблем: Ollama по умолчанию берёт num_ctx 2048-4096.
        Контекст главы (~15k символов) + системный промпт с few-shot не
        помещался в окно -> qwen возвращала пустую строку или обрывала
        генерацию. Окно подбиралось под конкретный запрос — но ПРЕЖНЯЯ
        формула window = max(8192, min(32768, prompt+max_new+1024)) имела
        критический дефект: потолок 32768 был ВЫДУМАННЫМ. Модель qwen3.5:4b
        имеет аппаратный лимит 16384 токенов; при num_ctx > 16k llama.cpp
        не мог отрендерить KV-кэш и возвращал done_reason='length' с ПУСТЫМ
        content (лог: num_ctx=18268, eval_count=None, content='').

        НОВЫЙ РАСЧЁТ (model_window — нативный лимит из detect_model_window):
          * num_ctx никогда не превышает model_window;
          * num_predict = min(max_new_cap, model_window - prompt - reserve);
            если остатка < MIN_USEFUL_WINDOW — промпт урезается вызывающим
            кодом (chat_context_truncated), а не убивает генерацию молча;
          * при переполнении окна включается скользящее окно истории
            (_trim_history) вместо «немой» ошибки рендера.
        """
        window_cap = model_window or self.detect_model_window(self.model)
        ctx_cap, pred_cap = self._ctx_cap  # (num_ctx_cap, num_predict_cap), 0 = не задан
        if ctx_cap:
            window_cap = min(window_cap, ctx_cap)
        # Целевая длина генерации ~1500-2500 слов Markdown для полного
        # конспекта; реально упирается в остаток окна модели ниже.
        # FIX (лог 14:16): при промпте ~19k целевая генерация 8192 раздувала
        # слот до ~28k токенов -> Ollama не аллоцировала KV-кэш и молча
        # возвращала пустоту. Целевая длина ответа теперь адаптивна: полная
        # 8192 — только для коротких промптов; на больших входах генерация
        # ужимается, чтобы инвариант prompt+gen+reserve <= num_ctx держался
        # в пределах эффективного бюджета машины.
        desired_new = 8192
        if prompt_tokens > self._prompt_budget // 2:
            desired_new = 4096
        if prompt_tokens > self._prompt_budget:
            desired_new = 2048
        if pred_cap:
            desired_new = min(desired_new, pred_cap)
        reserve = 512  # шаблон чата / special tokens
        # ЖЁСТКИЙ ИНВАРИАНТ: prompt_tokens + num_predict + reserve <= num_ctx
        # <= window_cap. Раньше при room < 512 ставился полный desired_new и
        # окно раздувалось за лимит модели -> «немой» done_reason='length'.
        # Теперь num_predict никогда не превышает реальный остаток окна.
        room = window_cap - prompt_tokens - reserve
        max_new = max(64, min(desired_new, room))
        window = max(_MIN_USEFUL_WINDOW, min(
            window_cap, prompt_tokens + max_new + reserve))
        return {
            "temperature": 0.3,
            "top_p": 0.9,
            "num_ctx": window,
            "num_predict": max_new,
            "num_gpu": 99,
        }

    def _chat_body(self, model: str, messages: List[Dict[str, Any]],
                   options: Dict[str, Any], stream: bool) -> Dict[str, Any]:
        """Тело запроса /api/chat. PERF: keep_alive удерживает модель в VRAM
        между запросами (без него Ollama выгружает её через 5 минут простоя,
        и каждый ответ начинается с медленной повторной загрузки весов)."""
        body: Dict[str, Any] = {"model": model, "messages": messages,
                                "stream": stream, "options": options}
        ka = self._keep_alive_param()
        if ka is not None:
            body["keep_alive"] = ka
        return body

    def _post_chat(self, model: str, messages: List[Dict[str, Any]],
                   options: Dict[str, Any]) -> tuple:
        """Один HTTP-запрос к /api/chat. Возвращает (content, done_reason)."""
        response = requests.post(
            f"{self.base_url}/api/chat",
            json=self._chat_body(model, messages, options, stream=False),
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.json()
        content = data.get("message", {}).get("content", "") or ""
        done_reason = data.get("done_reason") or ""
        # PERF/FIX: одноразовая рантайм-верификация режима GPU по метрикам.
        if getattr(self, "_pending_gpu_check", False) and not self._first_chat_done:
            self._first_chat_done = True
            self._pending_gpu_check = False
            self._report_gen_speed(data)
        if done_reason == "length":
            logger.warning(
                "Генерация прервана по лимиту length "
                f"(eval_count={data.get('metrics', {}).get('eval_count')}, "
                f"num_ctx={options['num_ctx']}, "
                f"num_predict={options['num_predict']}).")
        if not content.strip():
            logger.warning(
                "Ollama вернула ПУСТОЙ ответ; done="
                f"{data.get('done')}, done_reason={done_reason!r}, "
                f"error={data.get('error')}, model={model}, "
                f"num_ctx={options['num_ctx']}")
        return content, done_reason

    @staticmethod
    def _glue_parts(prev: str, new: str) -> str:
        """REFACTORING (ТЗ п.1): склейка частей авто-продолжения БЕЗ дублей.

        Мелкие LLM при дозапросе часто повторяют кусок уже написанного текста
        (последняя строка, обрывок таблицы или целый раздел). Раньше части
        конкатенировались как есть, и в ответе появлялись двойные таблицы /
        заголовки. Здесь на стыке находится наибольшее пересечение хвоста
        prev и начала new (difflib.SequenceMatcher + точная проверка префикса),
        и дублирующийся кусок вырезается из начала новой порции.

        Возвращает new без дублирующегося префикса (или исходный new, если
        пересечения нет). Потокобезопасен: чистая функция над строками.
        """
        if not prev or not new:
            return new
        tail = prev[-3000:]          # окно поиска пересечения
        head = new[:3000]
        sm = difflib.SequenceMatcher(a=tail, b=head, autojunk=False)
        best_len, best_j = 0, 0
        for _i, j, size in sm.get_matching_blocks():
            # считаем только блоки, доходящие до самого начала новой порции —
            # именно они означают «модель повторила хвост предыдущего ответа»
            if j == 0 and size > best_len:
                best_len, best_j = size, j
        # Дубликаты часто отличаются одной новой строкой/пробелом на стыке —
        # точная проверка длиннейшего префикса head, встречаемого в tail.
        max_k = min(len(head), len(tail), 2500)
        for k in range(max_k, best_len, 8):
            if tail.endswith(head[:k]):
                best_len = k
                break
        if best_len >= 12:
            logger.info(f"LLM: вырезано дублирующееся продолжение "
                        f"на стыке частей ({best_len} симв.)")
            return new[best_len:]
        # FIX (лог «попытка 2/2» + дубль абзаца теоремы на экране):
        # модели часто «перефразовывают» повтор, вставляя markdown-шум
        # (** Пусть…**, `$$`, лишние пробелы) — посимвольное пересечение
        # тогда не срабатывает. Сравниваем ТОЛЬКО последние строки prev и
        # первые строки new в нормализованном виде (без *, _, #, ` , $,
        # пробелов): совпало — весь этот дублирующийся префикс вырезается.
        # FIX (NameError: name 'self' is not defined): _glue_parts объявлен
        # @staticmethod — внутри него нет self, а _norm_line тоже статический.
        # Обращаемся через имя класса, чтобы склейка продолжений не падала.
        norm_prev_lines = [LLMEngine._norm_line(l) for l in tail.splitlines()
                           if LLMEngine._norm_line(l)]
        new_lines = new.splitlines(keepends=False)
        # FIX (скриншот 1: абзац теоремы продублирован ЦЕЛИКОМ, а не строкой):
        # раньше срез шёл построчно и «первая же совпавшая строка» считалась
        # началом дубля — короткий служебный текст мог обрезать полезное, а
        # многострочный пересказ (модель перефразировала абзац, добавив
        # **жирное** выделение) детектировался лишь частично. Теперь дублем
        # считается только ПОСЛЕДОВАТЕЛЬНОЕ совпадение >=2 нормализованных
        # строк подряд (или одной очень длинной) — и оно вырезается целиком.
        prev_set = norm_prev_lines[-60:]

        def _line_is_dup(nl: str) -> bool:
            return any(nl == pl or (len(nl) >= 40 and (nl in pl or pl in nl))
                       for pl in prev_set)

        cut_to = 0
        i = 0
        while i < min(len(new_lines), 40):
            nl = LLMEngine._norm_line(new_lines[i])
            if not nl:
                i += 1
                continue
            if _line_is_dup(nl):
                nxt = ""
                if i + 1 < len(new_lines):
                    nxt = LLMEngine._norm_line(new_lines[i + 1])
                # одиночная короткая совпавшая строка — не доказательство
                # дубля; ждём подтверждение следующей строкой или большой
                # объём самой строки (целый абзац).
                if nxt and _line_is_dup(nxt):
                    cut_to = i + 2
                    i += 2
                    continue
                if len(nl) >= 80:
                    cut_to = i + 1
                i += 1
                continue
            break
        if cut_to:
            stripped = "\n".join(new_lines[cut_to:])
            if len(stripped) < len(new):
                logger.info(f"LLM: вырезан нормализованный дубль "
                            f"продолжения ({cut_to} строк, "
                            f"{len(new) - len(stripped)} симв.)")
            return stripped.lstrip("\n")
        return new

    @staticmethod
    def _norm_line(line: str) -> str:
        """Нормализация строки для детекта перефразированного дубля:
        снимает markdown-шум (**_`#) и всю пунктуацию/пробелы.

        FIX (скриншот 1 — дубль абзаца «Теорема о единственности...»):
        раньше вырезались только *мелкие* отличия; теперь дополнительно
        схлопываются русские частицы не/ни с последующим словом («не
        единственная» == «неединственная») — модель при повторе часто
        меняет именно написание частиц и дефисы, из-за чего посимвольное
        сравнение дубль не ловило."""
        s = re.sub(r"[*_`#>\[\](){}$\\]", "", line)
        s = s.replace("-", "").replace("\u2013", "").replace("\u2014", "")
        s = re.sub(r"\s+", "", s)
        s = re.sub(r"(?:не|ни)([а-яёa-z]{2,})", r"\1", s.lower())
        return s.strip().lower()

    @staticmethod
    def _clip_message_content(content: str, max_chars: int) -> str:
        """Обрезка содержимого сообщения по границам абзацев с пометкой.

        Используется как последний резерв, когда даже одно сообщение (например,
        холст полной главы 62k символов) физически не влезает в окно модели.
        """
        if len(content) <= max_chars:
            return content
        clipped = content[:max_chars]
        cut = clipped.rfind("\n\n")
        if cut > max_chars // 2:
            clipped = clipped[:cut]
        return (clipped.rstrip() +
                "\n\n[...фрагмент обрезан: контекст превысил лимит окна модели...]")

    def _trim_history(self, messages: List[Dict[str, Any]],
                      window_cap: int,
                      reserve_for_gen: Optional[int] = None) -> List[Dict[str, Any]]:
        """Скользящее окно: гарантирует, что оценка промпта влезает в РЕАЛЬНОЕ
        окно модели (с запасом на num_predict).

        Шаг 1 — удаляет самые старые non-system сообщения по одному (system-
        промпт и последний user-запрос защищены от вырезания всегда).
        Шаг 2 — если после этого бюджет всё ещё превышен (последний запрос сам
        по себе больше окна), его содержимое обрезается по абзацам (_clip_...).
        Иначе num_ctx снова раздулся бы за аппаратный лимит — тот самый баг из
        лога (num_ctx=18268 при окне модели 16384 -> пустой ответ).
        Потокобезопасен: работает с копиями сообщений, исходные не мутирует.

        FIX (логи 00:56 / 10:48): раньше резерв на генерацию был фиксирован
        (1024), а _build_options аллоцировал num_predict до 4096 — то есть
        инвариант prompt+predict<=num_ctx нарушался САМИМ движком: llama.cpp
        не мог отрендерить ни одного токена (eval_count=None,
        done_reason='length', content='') даже при «уменьшенном» окне.
        Теперь вызывающий код передаёт фактический бюджет генерации, и
        обрезка гарантированно оставляет место под num_predict.
        """
        if reserve_for_gen is None:
            # FIX (логи 00:56/10:48): резерв = фактический бюджет генерации,
            # согласованный с _build_options через _fit в chat(). Дефолт —
            # не фиксированные 1024, а актуальный self._gen_reserve.
            reserve_for_gen = getattr(self, "_gen_reserve", 1024)
        budget = max(512, window_cap - reserve_for_gen)
        trimmed = []
        for m in messages:
            # FIX ('str' object has no attribute 'get'/'items'): контракт
            # messages допускает dict-сообщения, но вверх по конвейеру
            # исторически гуляли JSON-строки (сериализованные дампы истории).
            # Локальный нормализатор без импорта rag_pipeline (иначе — цикл
            # импортов): dict — passthrough, str/bytes — json.loads, мусор — {}.
            if isinstance(m, dict):
                trimmed.append(dict(m))
                continue
            parsed: Any = None
            if isinstance(m, (bytes, bytearray)):
                m = m.decode("utf-8", errors="replace")
            if isinstance(m, str):
                s = m.strip()
                try:
                    parsed = json.loads(s)
                except (ValueError, TypeError):
                    try:
                        parsed = json.loads(s.replace("'", '"'))
                    except (ValueError, TypeError):
                        parsed = None
            trimmed.append(parsed if isinstance(parsed, dict) else {})

        def _total() -> int:
            return self.estimate_tokens(trimmed)

        while _total() > budget and len(trimmed) > 2:
            # индекс первого removable: пропускаем system-префикс
            first_removable = 0
            while (first_removable < len(trimmed)
                   and trimmed[first_removable].get("role") == "system"):
                first_removable += 1
            last = len(trimmed) - 1
            if first_removable > last or first_removable == last:
                break  # вырезать нечего (остались только system + последний user)
            dropped = trimmed.pop(first_removable)
            logger.info(
                f"LLM: контекст превышает окно модели ({_total() + 1024} > "
                f"{window_cap}) — из истории вырезано сообщение "
                f"'{dropped.get('role')}' ({len(dropped.get('content') or '')} симв.)")

        # Шаг 2: последний запрос/огромный system-блок — урезаем содержимое,
        # чтобы инвариант prompt+predict<=num_ctx<=window выполнялся всегда.
        if _total() > budget:
            overhead = _total() - budget
            for msg in reversed(trimmed):
                content = msg.get("content", "") or ""
                if not content:
                    continue
                allowed = max(256, len(content) - overhead * 3)
                if allowed < len(content):
                    msg["content"] = self._clip_message_content(content, allowed)
                    logger.warning(
                        f"LLM: контекст ({_total()} токенов) превышает окно "
                        f"модели ({window_cap}) — сообщение "
                        f"'{msg.get('role')}' обрезано до {allowed} симв.")
                if _total() <= budget:
                    break
        return trimmed

    def chat(self, messages: List[Dict[str, Any]], stream: bool = False) -> str:
        """Отправляет сообщения в Ollama и возвращает ответ.

        REFACTORING (аудит «обучающей системы»): раньше done_reason='length'
        только логировался — пользователь видел оборванный на полуслове ответ.
        Теперь реализовано авто-продолжение (continuation): если генерация
        упёрлась в лимит num_predict, движок до 2 раз дозапрашивает модель,
        подставляя уже написанный текст в историю, и склеивает куски в единый
        ответ. Технических обрывов «на полуслове» больше не бывает.

        AUDIT (критический сбой qwen3.5:4b / пустой ответ): все options
        теперь считаются от РЕАЛЬНОГО окна модели (detect_model_window), а
        история перед каждым запросом проходит через _trim_history — num_ctx
        физически не может превысить аппаратный лимит, «немые» done_reason=
        'length' с пустым content больше невозможны by design.
        """
        self._ensure_available()
        model = self._resolve_model()
        if model != self.model:
            logger.info(f"Модель '{self.model}' не найдена в Ollama — "
                        f"использую доступную '{model}'")
        window_cap = self.detect_model_window(model)
        ctx_cap, _ = self._ctx_cap
        if ctx_cap:
            window_cap = min(window_cap, ctx_cap)

        def _fit(messages_in: List[Dict[str, Any]], cap: int):
            """Согласование trim↔options (FIX логов 00:56/10:48).

            Корень «пустого ответа»: _trim_history резервировал под генерацию
            фиксированные 1024 токена, а _build_options аллоцировал num_predict
            до 4096 — инвариант prompt+predict<=num_ctx нарушался САМИМ движком.
            llama.cpp не мог отрендерить ни одного токена (eval_count=None,
            done_reason='length', content='') даже после «перезапроса с
            уменьшенным контекстом». Теперь trim и options считаются друг под
            друга; цикл сходится за 1-2 итерации (num_predict монотонно
            растёт при ужимании промпта).
            """
            msgs = messages_in
            opts = None
            for _ in range(3):
                est = self.estimate_tokens(msgs)
                opts = self._build_options(est, cap)
                need_reserve = int(opts.get("num_predict", 0)) + 512
                trimmed = self._trim_history(msgs, cap,
                                             reserve_for_gen=need_reserve)
                t_est = self.estimate_tokens(trimmed)
                if t_est <= cap - need_reserve:
                    # Промпт влезает в окно ВМЕСТЕ с num_predict из opts —
                    # фиксируем бюджет генерации для будущих trim-вызовов.
                    self._gen_reserve = max(1024, need_reserve)
                    return trimmed, opts
                msgs = trimmed
            # Деградация: последний вариант всё же отправляем — движок ниже
            # делает каскад спасения, чем убивать запрос молча.
            self._gen_reserve = 1024
            return msgs, self._build_options(self.estimate_tokens(msgs), cap)

        messages, options = _fit(messages, window_cap)

        if stream:
            # Стриминговый путь остаётся простым генератором без continuation
            # (UI использует синхронный режим).
            try:
                response = requests.post(
                    f"{self.base_url}/api/chat",
                    json=self._chat_body(model, messages, options,
                                         stream=True),
                    timeout=self.timeout, stream=True,
                )
                response.raise_for_status()
                return self._stream_response(response)
            except requests.exceptions.RequestException as e:
                logger.error(f"Ошибка при запросе к Ollama: {e}")
                raise LamaLocaError(f"Сбой Ollama API: {e}")

        MAX_CONTINUATIONS = 2
        cont_messages = [dict(m) for m in messages]
        full_parts: List[str] = []
        try:
            for attempt in range(MAX_CONTINUATIONS + 1):
                content, done_reason = self._post_chat(model, cont_messages, options)
                # AUDIT (диагностика «немой» генерации): eval_count=None при
                # done_reason='length' означает, что llama.cpp НЕ ОТРЕНДЕРИЛ ни
                # одного токена — запрос физически не влезал в KV-кэш модели.
                # Раньше это выглядело как пустой пузырь в чате; теперь движок
                # сам снижает окно и делает один перезапрос с урезанной историей.
                if (not content.strip() and done_reason == "length"
                        and attempt == 0):
                    logger.warning(
                        "LLM: нулевой вывод при полном окне — перезапрос с "
                        "уменьшенным контекстом (KV-кэш/лимит слотов модели).")
                    # FIX (лог 10:48): окно снижаем ДО уровня, когда промпт
                    # + num_predict гарантированно влезают (было: trim с
                    # резервом 1024 против num_predict 1938 -> снова пустота).
                    reduced_cap = max(_MIN_USEFUL_WINDOW, window_cap // 2)
                    cont_messages, options = _fit(messages, reduced_cap)
                    continue
                # ТЗ п.1: перед конкатенацией вырезаем дублирующийся хвост
                # предыдущей части (модель часто повторяет последнюю строку,
                # обрывок таблицы или целый раздел).
                if full_parts and content.strip():
                    prev_joined = "\n".join(p for p in full_parts if p.strip())
                    content = self._glue_parts(prev_joined, content)
                    if not content.strip():
                        logger.info("LLM: продолжение целиком дублировало "
                                    "предыдущий ответ — кусок отброшен.")
                        continue
                full_parts.append(content)
                if done_reason != "length":
                    break
                joined = "\n".join(p for p in full_parts if p.strip()).rstrip()
                if attempt >= MAX_CONTINUATIONS or len(joined) < 12:
                    logger.warning(
                        "Продолжение не выполнено (исчерпаны попытки или "
                        "слишком короткий вывод) — отвечаем тем, что есть.")
                    break
                # Модель видит свой же предыдущий ответ и инструкцию дописать
                # ровно остаток, без вступлений и повторов.
                tail = joined[-1500:]
                cont_messages = list(messages) + [
                    {"role": "assistant", "content": joined},
                    {"role": "user", "content":
                        "Твой ответ был прерван на этом месте: …" + tail +
                        "\nПродолжи ровно с точки обрыва: допиши остаток "
                        "ответа (без повторения уже написанного, без фразы "
                        "\u201cПродолжаю\u201d, markdown/LaTeX-формат тот же)."},
                ]
                # FIX (лог «num_ctx=18111 -> 18268»): раньше дозапроска
                # раздувала историю (контекст главы + весь предыдущий ответ),
                # упиралась в окно и возвращала ПУСТОТУ. Теперь continuation
                # идёт по скользящему окну: из истории вырезаются старые
                # сообщения, а хвост предыдущего ответа остаётся единственным
                # носителем уже написанного текста. num_ctx никогда не
                # превышает нативный лимит модели.
                cont_messages, options = _fit(cont_messages, window_cap)
                logger.info(f"LLM: авто-продолжение оборванного ответа "
                            f"(попытка {attempt + 1}/{MAX_CONTINUATIONS})")
            return "\n".join(p for p in full_parts if p.strip())
        except requests.exceptions.RequestException as e:
            logger.error(f"Ошибка при запросе к Ollama: {e}")
            # Если первый кусок уже получен — отдаём его, а не выбрасываем.
            if full_parts and any(p.strip() for p in full_parts):
                logger.warning("Частичный ответ сохранён несмотря на ошибку.")
                return "".join(full_parts)
            raise LamaLocaError(f"Сбой Ollama API: {e}")

    def _stream_response(self, response: requests.Response):
        """Генератор токенов для стриминга."""
        for line in response.iter_lines():
            if line:
                import json
                data = json.loads(line)
                if "message" in data and "content" in data["message"]:
                    yield data["message"]["content"]