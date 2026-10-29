# src/llm/llm_engine.py
import difflib
import json
import os
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

# FIX (400 Bad Request, лог 27.10 16:01): практический RAM-безопасный потолок
# num_ctx. config.json модели может заявлять context_length 262144 — это
# архитектурный максимум, а НЕ то, что llama.cpp реально аллоцирует: KV-кэш
# под 262k токенов не влезает в память типовой машины и Ollama отвечает
# 400 Bad Request ещё до генерации. Потолок можно поднять переменной
# окружения OLLAMA_NUM_CTX_CAP / полем num_ctx_cap в config.yaml.
_SAFE_CTX_CEILING = int(os.environ.get("OLLAMA_SAFE_CTX_CEILING", "32768"))


class LLMEngine:
    """Обёртка над Ollama API для генерации ответов через /api/chat."""
    
    def __init__(self, config: OllamaConfig):
        self.base_url = config.base_url.rstrip("/")
        self.model = config.chat_model
        self.timeout = config.timeout_seconds
        # Потолки контекста, отдаваемые в options.num_ctx. Могут быть снижены
        # init-флагом OLLAMA_NUM_CTX_CAP или полем config (num_ctx_cap /
        # max_context) — читаем defensively, чтобы не ломать frozen-конфиги
        # старых версий settings.py.
        self._ctx_cap = self._read_ctx_cap(config)
        
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
        with _MODEL_WINDOW_LOCK:
            _MODEL_WINDOW_CACHE[cache_key] = window
        logger.info(f"LLM: нативное окно контекста '{model}' = {window} токенов "
                    f"(потолок num_ctx в запросах)")
        return window

    @staticmethod
    def estimate_tokens(messages: List[Dict[str, str]]) -> int:
        """\u0413\u0440\u0443\u0431\u0430\u044f \u043e\u0446\u0435\u043d\u043a\u0430 \u0434\u043b\u0438\u043d\u044b \u043f\u0440\u043e\u043c\u043f\u0442\u0430 \u0432 \u0442\u043e\u043a\u0435\u043d\u0430\u0445.

        \u0414\u043b\u044f \u043a\u0438\u0440\u0438\u043b\u043b\u0438\u0447\u0435\u0441\u043a\u0438\u0445 \u0442\u0435\u043a\u0441\u0442\u043e\u0432 \u0443 Qwen-\u0442\u043e\u043a\u0435\u043d\u0438\u0437\u0430\u0442\u043e\u0440\u0430 \u044d\u0442\u043e ~3.5 \u0441\u0438\u043c\u0432\u043e\u043b\u0430 \u043d\u0430
        \u0442\u043e\u043a\u0435\u043d; \u0431\u0435\u0440\u0451\u043c \u043a\u043e\u043d\u0441\u0435\u0440\u0432\u0430\u0442\u0438\u0432\u043d\u044b\u0439 \u0434\u0435\u043b\u0438\u0442\u0435\u043b\u044c 3, \u0447\u0442\u043e\u0431\u044b \u043d\u0435\u0434\u043e\u043e\u0446\u0435\u043d\u043a\u0430 \u043d\u0435
        \u043f\u0440\u0438\u0432\u043e\u0434\u0438\u043b\u0430 \u043a \u043c\u043e\u043b\u0447\u0430\u043b\u0438\u0432\u043e\u043c\u0443 \u0443\u0441\u0435\u0447\u0435\u043d\u0438\u044e \u043a\u043e\u043d\u0442\u0435\u043a\u0441\u0442\u0430 (done_reason='length').
        """
        total_chars = sum(len(m.get("content", "") or "") for m in messages)
        return total_chars // 3 + 64  # \u0437\u0430\u043f\u0430\u0441 \u043d\u0430 \u0448\u0430\u0431\u043b\u043e\u043d \u0447\u0430\u0442\u0430 \u0438 \u0440\u043e\u043b\u0438

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
        desired_new = 8192
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
        # FIX (400 Bad Request, лог 16:01): detect_model_window вернул
        # 262144 — это ПАРАМЕТР ARXITEКТУРЫ из config.json модели (напр.,
        # Qwen3 context_length), а НЕ лимит установленного GGUF-чекапойнта.
        # llama.cpp обязан аллоцировать KV-кэш под весь num_ctx; при 262k на
        # машине пользователя аллокация не помещается и Ollama ОТКЛОНЯЕТ
        # запрос ещё до инференса: 400 Bad Request на /api/chat. Раньше
        # работало потому, что движок никогда не просил окно больше 16-32k.
        # Поэтому вводим практический потолок RAM-безопасного окна и
        # DEGRADE-MECHANISM ниже по чату (см. _chat_with_degradation).
        window = min(window, _SAFE_CTX_CEILING)
        return {
            "temperature": 0.3,
            "top_p": 0.9,
            "num_ctx": window,
            "num_predict": max_new,
            "num_gpu": 99,
        }

    def _post_chat(self, model: str, messages: List[Dict[str, Any]],
                   options: Dict[str, Any],
                   json_schema: Optional[Dict[str, Any]] = None) -> tuple:
        """Один HTTP-запрос к /api/chat. Возвращает (content, done_reason).

        AUDIT (съеденные бэкслэши \\frac/\\mathbf/\\to): qwen-семейство на уровне
        декодирования выбрасывает токен обратного слэша вне JSON-строковых
        значений — никакие regex-препроцессоры это не лечат надёжно. Нативное
        инженерное решение llama.cpp/Ollama: structured output (format=
        json_schema). При заданном json_schema запрос уходит с
        "format": {"type": "json_schema", "schema": ...}, и grammar-constrained
        декодирование ГАРАНТИРУЕТ сохранение всех символов, включая \\, внутри
        поля answer. Путь без схемы остаётся прежним (fallback для старых
        версий Ollama и стриминга).
        """
        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": options,
        }
        if json_schema is not None:
            payload["format"] = {"type": "json_schema", "schema": json_schema}
        response = requests.post(
            f"{self.base_url}/api/chat",
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.json()
        content = data.get("message", {}).get("content", "") or ""
        done_reason = data.get("done_reason") or ""
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
        return new

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
                      protect_last: bool = True) -> List[Dict[str, Any]]:
        """Скользящее окно: гарантирует, что оценка промпта влезает в РЕАЛЬНОЕ
        окно модели (с запасом на num_predict).

        Шаг 1 — удаляет самые старые non-system сообщения по одному (system-
        промпт и последний user-запрос защищены от вырезания всегда).
        Шаг 2 — если после этого бюджет всё ещё превышен (последний запрос сам
        по себе больше окна), его содержимое обрезается по абзацам (_clip_...).
        Иначе num_ctx снова раздулся бы за аппаратный лимит — тот самый баг из
        лога (num_ctx=18268 при окне модели 16384 -> пустой ответ).
        Потокобезопасен: работает с копиями сообщений, исходные не мутирует.

        FIX («текст оборван на полуслове»): раньше continuation-дозапроска
        резала историю тем же способом, что и обычный чат, — вырезанное
        assistant-сообщение (уже написанный текст ответа) уничтожало смысл
        продолжения, модель обрывалась заново, а лимит попыток кончался. Для
        пути авто-продолжения вызывается с protect_last=False: сначала
        вырезаются ВСЕ старые сообщения, кроме system-префикса и последних
        двух (носители уже написанного текста), и лишь затем урезается
        огромный холст главы.
        """
        reserve_for_gen = 1024
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

        while _total() > budget:
            # индекс первого removable: пропускаем system-префикс
            first_removable = 0
            while (first_removable < len(trimmed)
                   and trimmed[first_removable].get("role") == "system"):
                first_removable += 1
            last = len(trimmed) - 1
            # защищённый хвост: обычный чат — последний запрос; continuation
            # (protect_last=False) — последние ДВА сообщения (assistant-текст
            # ответа + инструкция продолжения), иначе обрезается уже написан-
            # ное и ответ «оборван на полуслове» остаётся навсегда.
            floor = last if protect_last else max(first_removable, last - 1)
            if first_removable >= floor:
                break  # вырезать нечего (остались только защищённые сообщения)
            dropped = trimmed.pop(first_removable)
            logger.info(
                f"LLM: контекст превышает окно модели ({_total() + 1024} > "
                f"{window_cap}) — из истории вырезано сообщение "
                f"'{dropped.get('role')}' ({len(dropped.get('content') or '')} симв.)")

        # Шаг 2: последний запрос/огромный system-блок — урезаем содержимое,
        # чтобы инвариант prompt+predict<=num_ctx<=window выполнялся всегда.
        if _total() > budget:
            # FIX («текст оборван на полуслове»): хвост истории — носители
            # уже написанного ответа и инструкции продолжения. Раньше шаг 2
            # шёл по reversed и первым делом урезал именно их до 256 символов
            # с маркером «[...фрагмент обрезан...]»: модель теряла точку
            # обрыва, continuation не склеивался, пользователь видел обрезан-
            # ный ответ. Теперь очередь урезания строится от НАИБОЛЕЕ ценного
            # к наименее: protected tail (не урезается никогда), затем
            # последний user-вопрос, и только потом — старый контекст
            # (system/холст главы). Обход в обратном порядке => system
            # жертвуется первым.
            ordered = list(trimmed)
            if not protect_last and len(ordered) >= 2:
                # хвост (последние 2 сообщения) переносим в конец обхода,
                # чтобы reversed() брал их на урезание последними
                ordered = trimmed[:-2] + [trimmed[-1], trimmed[-2]]
            for msg in reversed(ordered):
                content = msg.get("content", "") or ""
                if not content:
                    continue
                # FIX (NameError 'overhead'): переменная никогда не была
                # определена — шаг 2 падал при переполнении окна. Считаем
                # превышение бюджета в символах (~3 симв/токен для кириллицы,
                # консервативно 2) и урезаем сообщение с запасом x3, чтобы
                # цикл сходился за 1-2 прохода.
                excess_chars = (_total() - budget) * 2
                allowed = max(256, len(content) - excess_chars * 3)
                if allowed < len(content):
                    msg["content"] = self._clip_message_content(content, allowed)
                    logger.warning(
                        f"LLM: контекст ({_total()} токенов) превышает окно "
                        f"модели ({window_cap}) — сообщение "
                        f"'{msg.get('role')}' обрезано до {allowed} симв.")
                if _total() <= budget:
                    break
        return trimmed

    # ── Structured output (нативное решение вместо regex-костылей) ──────────
    # Схема ответа: markdown с LaTeX — внутри JSON-строки grammar-декодер
    # Ollama обязан emit-ить все символы, включая обратный слэш. Это единствен-
    # ный надёжный способ сохранить \frac/\mathbf/\to у qwen на уровне токенов.
    ANSWER_JSON_SCHEMA: Dict[str, Any] = {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
        "required": ["answer"],
        "additionalProperties": False,
    }

    @staticmethod
    def _extract_json_answer(content: str) -> Optional[str]:
        """Достаёт поле answer из structured-output ответа модели.

        Возвращает markdown-текст либо None, если content не является валид-
        ным JSON-объектом со строковым полем answer (тогда вызывающий код
        использует content как есть — полная обратная совместимость).
        Допускаются обрамляющие пробелы и ```json-ограждения (мелкие модели
        иногда оборачивают вывод). Потокобезопасен: чистая функция.
        """
        if not content:
            return None
        s = content.strip()
        if s.startswith("```"):
            # снимаем markdown-ограждение кода, если модель его добавила
            first_nl = s.find("\n")
            if first_nl != -1 and s.rstrip().endswith("```"):
                s = s[first_nl + 1:s.rstrip().rfind("```")].strip()
        if not (s.startswith("{") and s.endswith("}")):
            return None
        try:
            obj = json.loads(s)
        except (ValueError, TypeError):
            return None
        if isinstance(obj, dict) and isinstance(obj.get("answer"), str):
            return obj["answer"]
        return None

    def chat(self, messages: List[Dict[str, Any]], stream: bool = False,
             use_json_schema: bool = True) -> str:
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
        messages = self._trim_history(messages, window_cap)
        options = self._build_options(self.estimate_tokens(messages), window_cap)

        if stream:
            # Стриминговый путь остаётся простым генератором без continuation
            # (UI использует синхронный режим).
            try:
                response = requests.post(
                    f"{self.base_url}/api/chat",
                    json={"model": model, "messages": messages,
                          "stream": True, "options": options},
                    timeout=self.timeout, stream=True,
                )
                response.raise_for_status()
                return self._stream_response(response)
            except requests.exceptions.RequestException as e:
                logger.error(f"Ошибка при запросе к Ollama: {e}")
                raise LamaLocaError(f"Сбой Ollama API: {e}")

        MAX_CONTINUATIONS = 2
        # флаг одноразового ретрая пустой дозапроски continuation
        self._cont_retry_used = False
        cont_messages = [dict(m) for m in messages]
        full_parts: List[str] = []
        # FIX (400): если сборка Ollama отвергла structured output — запоминаем
        # это на весь движок и больше не шлём format=json_schema.
        if getattr(self, "_schema_broken", False):
            use_json_schema = False
        schema = self.ANSWER_JSON_SCHEMA if use_json_schema else None

        def _unwrap(raw: str) -> str:
            """structured-output content -> чистый markdown (с фолбэком)."""
            if schema is None:
                return raw
            ans = self._extract_json_answer(raw)
            return raw if ans is None else ans

        try:
            for attempt in range(MAX_CONTINUATIONS + 1):
                content, done_reason = self._post_chat(
                    model, cont_messages, options, json_schema=schema)
                # FIX (старые Ollama <0.5 молча игнорируют/ошибаются на format):
                # если со схемой пришёл пустой ответ, а без неё модель жива —
                # деградируем в обычный режим один раз, не ломая UX.
                if (schema is not None and not content.strip()
                        and not full_parts):
                    logger.warning(
                        "LLM: structured-output вернул пустоту — "
                        "перезапрос без json_schema (fallback-режим).")
                    schema = None
                    continue
                content = _unwrap(content)
                # AUDIT (диагностика «немой» генерации): eval_count=None при
                # done_reason='length' означает, что llama.cpp НЕ ОТРЕНДЕРИЛ ни
                # одного токена — запрос физически не влезал в KV-кэш модели.
                # Раньше это выглядело как пустой пузырь в чате; теперь движок
                # сам снижает окно и делает один перезапрос с урезанной историей.
                if (not content.strip() and full_parts
                        and not getattr(self, "_cont_retry_used", False)):
                    # FIX («текст оборван на полуслове»): пустая дозапроска
                    # продолжения. Раньше такой ответ просто записывался как
                    # часть ответа и цикл завершался — пользователь видел
                    # обрезанный текст. Делаем ещё один заход с уменьшенным
                    # окном (KV-кэш), не расходуя попытку continuation.
                    logger.warning(
                        "LLM: пустой ответ при продолжении — перезапрос с "
                        "уменьшенным контекстом (KV-кэш/лимит слотов модели).")
                    reduced_cap = max(_MIN_USEFUL_WINDOW, window_cap // 2)
                    self._cont_retry_used = True
                    cont_messages = self._trim_history(
                        cont_messages, reduced_cap, protect_last=False)
                    options = self._build_options(
                        self.estimate_tokens(cont_messages), reduced_cap)
                    continue
                if (not content.strip() and done_reason == "length"
                        and attempt == 0):
                    logger.warning(
                        "LLM: нулевой вывод при полном окне — перезапрос с "
                        "уменьшенным контекстом (KV-кэш/лимит слотов модели).")
                    reduced_cap = max(_MIN_USEFUL_WINDOW, window_cap // 2)
                    cont_messages = self._trim_history(messages, reduced_cap)
                    options = self._build_options(
                        self.estimate_tokens(cont_messages), reduced_cap)
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
                # FIX («текст оборван на полуслове»): раньше в assistant-сооб-
                # щение клался ВЕСЬ уже написанный текст (joined), который
                # вместе с холстом главы превышал окно модели — _trim_history
                # урезал его по абзацам, точка обрыва терялась, модель писала
                # заново/неизвестно что, лимит попыток кончался, и пользователь
                # видел оборванный ответ. Теперь носитель контекста — только
                # хвост ответа (tail); весь накопленный текст живёт в
                # full_parts и склеивается движком, а не моделью.
                tail = joined[-1500:]
                cont_messages = list(messages) + [
                    {"role": "assistant", "content": tail},
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
                # FIX («оборван на полуслове», protect_last=False): раньше эти
                # же два последних сообщения (assistant-текст + инструкция)
                # тоже попадали под обрезку/вырезание — модель теряла точку
                # обрыва, продолжение стартовало с нуля и снова упиралось в
                # лимит попыток. Теперь они защищены; режется только контекст.
                cont_messages = self._trim_history(
                    cont_messages, window_cap, protect_last=False)
                options = self._build_options(
                    self.estimate_tokens(cont_messages), window_cap)
                logger.info(f"LLM: авто-продолжение оборванного ответа "
                            f"(попытка {attempt + 1}/{MAX_CONTINUATIONS})")
            return "\n".join(p for p in full_parts if p.strip())
        except requests.exceptions.RequestException as e:
            # FIX (400 Bad Request, лог 16:01): Ollama отклонила запрос ЕЩЁ ДО
            # генерации. Две известные причины 400 на /api/chat:
            #   1) сборка Ollama/модель не принимает format=json_schema
            #      (structured output появился только в Ollama >= 0.5);
            #   2) KV-кэш под запрошенный num_ctx не аллоцируется (память).
            # Раньше оба случая роняли чат с «Сбой Ollama API». Теперь —
            # контролируемая деградация: сначала перезапрос без json_schema,
            # затем с уменьшенным вдвое окном; при успехе схема запоминается
            # отключённой для движка (не долбим сервер заведомо битым
            # параметром). Исходная ошибка пробрасывается только если все
            # ступени деградации исчерпаны.
            status = getattr(getattr(e, "response", None), "status_code", None)
            if isinstance(e, requests.exceptions.HTTPError) and status == 400 \
                    and full_parts is not None and not any(
                        p.strip() for p in full_parts):
                if schema is not None:
                    logger.warning(
                        "LLM: Ollama отвергла structured-output (400) — "
                        "повторяю запрос БЕЗ json_schema (режим совместимости "
                        "со сборками < 0.5).")
                    self._schema_broken = True
                    return self.chat(messages, stream=False,
                                     use_json_schema=False)
                cur_ctx = options.get("num_ctx", 0)
                if cur_ctx > _MIN_USEFUL_WINDOW:
                    logger.warning(
                        f"LLM: 400 Bad Request при num_ctx={cur_ctx} — "
                        f"KV-кэш не аллоцируется; перезапрос с окном "
                        f"{max(_MIN_USEFUL_WINDOW, cur_ctx // 2)}.")
                    reduced_cap = max(_MIN_USEFUL_WINDOW, cur_ctx // 2)
                    trimmed = self._trim_history(messages, reduced_cap)
                    retry_opts = self._build_options(
                        self.estimate_tokens(trimmed), reduced_cap)
                    try:
                        content, _dr = self._post_chat(
                            model, trimmed, retry_opts, json_schema=None)
                        if content.strip():
                            return content
                    except requests.exceptions.RequestException as e2:
                        logger.error(f"LLM: деградация не помогла: {e2}")
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