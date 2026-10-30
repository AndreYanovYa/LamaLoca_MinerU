# src/embeddings/ollama_embedder.py
import json
import re
import requests
from loguru import logger
from typing import List, Optional
from tqdm import tqdm
from ..core.settings import OllamaConfig
from ..core.exceptions import LamaLocaError


# FIX (400 Bad Request на /api/embed при импорте Кудрявцева, лог 17:38):
# батч падал ПЕРВЫМ ЖЕ запросом — значит причина не в железе, а в содержимом.
# Типовые источники 400 у Ollama /api/embed:
#   1) пустые строки "" (валидация API их отвергает);
#   2) управляющие символы текста, извлечённого PyMuPDF из PDF-слоя (\x00–\x1f
#      без \t\n\r) — ломают сериализацию/токенизацию на сервере;
#   3) surrogate-символы (U+D800–U+DFFF), которые парсинг может привнести из
#      битых шрифтовых cmap — requests c jsonify НЕ умеет их кодировать и сам
#      падает, а сторонние клиенты отдают 400;
#   4) гигантские чанки, превышающие контекст модели эмбеддинга.
# Нормализуем тексты перед отправкой и умеем читать реальный текст ошибки из
# тела 4xx ответа (раньше был только «400 Client Error» без причины).

_CTRL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_SURR_RE = re.compile(r"[\ud800-\udfff]")


def sanitize_for_embed(text: str, max_chars: int = 6000) -> str:
    """Готовит чанк к отправке в /api/embed: вычищает мусорные символы слоя
    PDF и ограничивает длину (bge-m3 работает с контекстом 8k токенов;
    6000 символов кириллицы ≈ до 4–5k токенов — безопасно)."""
    if not isinstance(text, str):
        text = str(text or "")
    t = _CTRL_RE.sub(" ", text)
    t = _SURR_RE.sub("", t)          # одиночные surrogate → удаляем
    t = re.sub(r"[ \t]{2,}", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    t = t.strip()
    if len(t) > max_chars:
        # Режем по границам $$-блоков, чтобы не оставить открытый маркер
        # KaTeX в хвосте (важно для последующего ранжирования/ответов).
        cut = t[:max_chars]
        # Обломок парного маркера на самой границе среза ("...$" | "$...") —
        # убираем всегда: одиночный '$' на конце ломает и блочные, и
        # инлайновые формулы.
        while cut.endswith("$") and not cut.endswith("$$"):
            cut = cut[:-1]
        if cut.count("$$") % 2 == 1:
            open_at = cut.find("$$")                 # единственный маркер = открывающий
            if open_at >= 0:
                cut = (cut[:open_at] + cut[open_at + 2:]).strip()
        t = cut
    return t


class OllamaEmbedder:
    """Получает эмбеддинги через локальный API Ollama с батчингом."""
    
    def __init__(self, config: OllamaConfig, batch_size: int = 16):
        self.base_url = config.base_url.rstrip("/")
        self.model = config.embedding_model
        self.timeout = config.timeout_seconds
        self.batch_size = batch_size  # Ollama хорошо принимает батчи по 16-32
        # PERF: keep_alive держит модель эмбеддингов (bge-m3) в VRAM на время
        # всего импорта — иначе Ollama выгружает её между запросами и каждый
        # батч векторизации начинается с медленной повторной загрузки весов.
        self.keep_alive = getattr(config, "keep_alive_seconds", -5)
        
        # Проверка доступности.
        # АУДИТ: раньше при недоступной Ollama конструктор бросал LamaLocaError —
        # всё приложение падало на старте, хотя просмотр PDF/библиотека работают
        # без эмбеддингов. Теперь это деградирующее состояние: флаг available,
        # явная ошибка только при реальной попытке векторизации.
        self.available = False
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            response.raise_for_status()
            models = [m['name'] for m in response.json().get('models', [])]
            if not any(self.model in m for m in models):
                logger.warning(
                    f"Модель {self.model} не найдена в Ollama. Доступны: {models}. "
                    f"Выполните: ollama pull {self.model}"
                )
            else:
                logger.info(f"Ollama подключена. Используем модель: {self.model}")
            self.available = True
        except requests.exceptions.RequestException as e:
            logger.error(f"Не удалось подключиться к Ollama: {e}")
        # PERF: предупреждение о CPU-режиме Ollama (главная причина «очень
        # медленно» на Windows при установке зависимостей обычным pip).
        self._pending_gpu_check = False   # верифицируем режим по 1-му батчу
        self._first_batch_done = False
        if self.available and getattr(config, "gpu_check", True):
            self._check_gpu()

    #: Sentinel конструктора: keep_alive вообще не передаётся (дефолт сервера).
    _KA_NOTSET = -5

    def _format_keep_alive(self, ka) -> Optional[str]:
        """F4/FIX (логи 22:06 / 22:25 / 23:16): Ollama парсит keep_alive как
        Go time.Duration. Значение "-1" (канонический sentinel бессрочного
        удержания в новых сборках) старые серверы отвергают со стабильной
        ошибкой HTTP 400 `time: missing unit in duration "-1"` — она роняла
        ВЕСЬ импорт и каждый вопрос в чате. Версионные эвристики ситуацию не
        спасали: реальные Windows-сборки Ollama (0.9.x / 0.12.x) до сих пор
        падают на "-1", а логика «отправим -1, если версия новая» продолжала
        слать именно его и воспроизводить ошибку.

        Новая стратегия — НИКОГДА не отправлять голый "-1"/"0" без единицы:
        * None / -5 (sentinel) — параметр не передаётся вовсе (дефолт сервера);
        * любое отрицательное («бессрочно») — безопасный долгий период "1h":
          модель реально остаётся в VRAM между батчами импорта, запрос
          проходит на ЛЮБОЙ версии Ollama (Go duration «1h» валиден всегда);
        * положительное число секунд — "<N>s";
        * готовая строка с единицей ("10m", "1h") — передаём как есть;
        * прочий мусор / пустое — параметр не передаётся (не ломать запрос).
        """
        if ka is None:
            return None
        if isinstance(ka, bool):
            return None
        if isinstance(ka, (int, float)):
            if ka == self._KA_NOTSET:
                return None
            if ka < 0:
                return "1h"      # вместо "-1": совместимо со всеми сборками
            if ka == 0:
                return "0s"      # немедленная выгрузка (единица обязательна)
            return f"{int(ka)}s"
        text = str(ka).strip()
        if not text:
            return None
        # Готовая duration-строка? ("10m", "1h", "300ms", "1h30m")
        if re.fullmatch(r"\d+(\.\d+)?(ns|us|µs|ms|s|m|h)+", text):
            return text
        # Числовая строка из YAML ("-1", "3600") — применяем те же правила.
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

    @staticmethod
    def _version_reports_gpu(v: dict) -> Optional[bool]:
        """Признаки CUDA-бэкенда в ответе /api/version.

        FIX (вечный FP «Ollama не сообщает о CUDA-устройствах»): раньше
        проверялось наличие ключей cuda_devices/gpu, которых в реальном
        ответе /api/version НЕТ вовсе — там лишь {version, ollama_version}.
        Предупреждение срабатывало ВСЕГДА, даже при полностью рабочем GPU
        (лог 13:35). Теперь читаем поле `cuda` — фактический булев флаг
        сборки с CUDA-поддержкой. Возвращает True/False только при явных
        данных; None = unknown → решение отдаётся рантайм-метрикам
        /api/embed (см. _runtime_gpu_report), ложных предупреждений нет.
        """
        if not isinstance(v, dict):
            return None
        cvd = str(v.get("CUDA_VISIBLE_DEVICES") or "").strip()
        if cvd and cvd.lower() not in ("cpu", "none", "-1"):
            return True
        for key in ("cuda_devices", "gpu", "gpus", "devices"):
            val = v.get(key) or []
            if isinstance(val, str):
                if val.strip().lower() not in ("", "cpu", "none", "null"):
                    return True
            elif isinstance(val, (list, tuple)) and len(val) > 0:
                return True
        c = v.get("cuda")
        if isinstance(c, bool):
            return c
        return None  # поля нет / неизвестный формат — не гадаем

    def check_gpu(self) -> Optional[bool]:
        """Публичная диагностика режима Ollama (True=GPU, False=CPU, None=?)."""
        try:
            v = requests.get(f"{self.base_url}/api/version", timeout=5).json()
            verdict = self._version_reports_gpu(v)
            if verdict is not None:
                return verdict
            ps = requests.get(f"{self.base_url}/api/ps", timeout=5).json()
            running = ps.get("models") or []
            if running and all((m.get("size_vram") or 0) >= (m.get("size") or 0) > 0
                               for m in running):
                return True
            return None
        except Exception:  # noqa: BLE001
            return None

    def _check_gpu(self):
        """Диагностика режима исполнения Ollama (GPU/CUDA vs CPU).

        Источники: /api/version (сигнатура CUDA-бэкенда) + /api/ps (доля
        весов загруженных моделей в VRAM). Никогда не бросает исключений;
        если данных нет — молчит и откладывает вердикт до метрик первого
        реального батча эмбеддингов.
        """
        try:
            v = requests.get(f"{self.base_url}/api/version", timeout=5).json()
            verdict = self._version_reports_gpu(v)
            if verdict is True:
                logger.info("Ollama: CUDA-бэкенд подтверждён — эмбеддинги на GPU.")
                return
            if verdict is False:
                logger.warning(
                    "Ollama сообщает об отсутствии CUDA (cuda=false) — "
                    "векторизация идёт на CPU («очень медленно»). Починка для "
                    "RTX 5060: обновите Ollama до свежей сборки с CUDA для "
                    "Blackwell (драйвер NVIDIA >= 560); проверьте переменные "
                    "окружения сервера (OLLAMA_NUM_GPU=0 / CUDA_VISIBLE_DEVICES="
                    "CPU принудительно отключают GPU).")
                return
            # verdict is None — версия ничего не сообщает: смотрим /api/ps.
            running = []
            try:
                ps = requests.get(f"{self.base_url}/api/ps", timeout=5).json()
                running = ps.get("models") or []
            except Exception:  # noqa: BLE001
                pass
            if not running:
                # Ни CUDA, ни загруженных моделей — судить рано. Не пугаем
                # пользователя: проверим фактический режим после первого батча.
                self._pending_gpu_check = True
                return
            offloaded = [m for m in running
                         if (m.get("size") or 0) > (m.get("size_vram") or 0)]
            if offloaded:
                names = ", ".join(m.get("name", "?") for m in offloaded)
                logger.warning(
                    f"⚠️ Ollama считает частично на CPU: модели [{names}] "
                    "загружены в VRAM не целиком (или вообще на CPU). Для "
                    "RTX 5060: обновите Ollama до сборки с CUDA для Blackwell "
                    "(драйвер NVIDIA >= 560); проверьте переменные окружения "
                    "сервера — CUDA_VISIBLE_DEVICES=CPU или OLLAMA_NUM_GPU=0 "
                    "принудительно отключают GPU; убедитесь, что VRAM не занят "
                    "(ollama ps / диспетчер задач).")
            else:
                logger.info("Ollama: все загруженные модели целиком в VRAM — "
                            "эмбеддинги на GPU.")
        except Exception:  # noqa: BLE001 — диагностика не валит приложение
            pass

    def _runtime_gpu_report(self, data: dict, batch_len: int):
        """PERF/FIX: верификация режима по ПОСЛЕДОВАТЕЛЬНЫМ метрикам /api/embed.

        eval_count == 0 при ненулевом input — Ollama НЕ посчитала батч заново,
        а отдала кэш инференса (повтор той же строки). Это ловушка тестов:
        такие замеры показывают ложные «десятки ток/с». Реальный режим виден
        только на УНИКАЛЬНЫХ текстах. Скорость < 30 ток/с на уникальном батче
        — почти наверняка CPU. Вызывается один раз за прогон (_first_batch_done).
        """
        try:
            ec = data.get("eval_count")
            dur = data.get("eval_duration")
            if not ec or not dur:
                return
            if ec == 0:
                logger.debug(
                    "Ollama embed: eval_count=0 — сработал кэш инференса "
                    "(идентичные входы); режим GPU/CPU по этому запросу не "
                    "определяется.")
                return
            tps = ec / (dur / 1e9)
            if tps < 30:
                logger.warning(
                    f"⚠️ Векторизация идёт на ~{tps:.0f} ток/с — похоже на "
                    "CPU-режим Ollama (GPU bge-m3 даёт сотни–тысячи ток/с). "
                    "Обновите Ollama (сборка с CUDA для RTX 50xx, драйвер "
                    ">= 560) и проверьте CUDA_VISIBLE_DEVICES / OLLAMA_NUM_GPU.")
            else:
                logger.info(f"Ollama embed: ~{tps:.0f} ток/с — GPU-режим "
                            "подтверждён фактическими метриками.")
        except Exception:  # noqa: BLE001
            pass

    def _ensure_available(self):
        """Повторная проверка связи с Ollama перед запросом эмбеддингов."""
        if not self.available:
            try:
                response = requests.get(f"{self.base_url}/api/tags", timeout=5)
                response.raise_for_status()
                self.available = True
                logger.info("Ollama снова доступна.")
            except requests.exceptions.RequestException:
                raise LamaLocaError(
                    "Ollama недоступна. Запустите приложение Ollama и убедитесь, "
                    "что модель эмбеддингов загружена (ollama pull bge-m3)."
                )

    def _embed_body(self, batch: List[str]):
        # FIX(400): нормализуем содержимое чанков (управляющие/surrogate-символы
        # из текстового слоя PDF) и длину перед отправкой. Пустые после
        # зачистки строки заменяем заглушкой — Ollama отвергает "" в input.
        cleaned = []
        for t in batch:
            s = sanitize_for_embed(t)
            if not s:
                s = "(пустой блок)"
            cleaned.append(s)
        body = {"model": self.model, "input": cleaned}
        ka = self._format_keep_alive(self.keep_alive)
        if ka is not None:
            body["keep_alive"] = ka
        return body

    @staticmethod
    def _api_error_detail(resp) -> str:
        """Читает реальный текст ошибки из тела 4xx ответа Ollama
        (раньше в лог попадал только «400 Client Error» без причины)."""
        try:
            payload = resp.json()
            msg = payload.get("error") if isinstance(payload, dict) else None
            if msg:
                return str(msg)[:500]
        except (ValueError, json.JSONDecodeError):
            pass
        try:
            return (resp.text or "")[:500]
        except Exception:  # noqa: BLE001
            return ""

    def embed_texts(self, texts: List[str], progress_callback=None) -> List[List[float]]:
        """
        Получает векторы для списка текстов с батчингом.
        
        Args:
            texts: Список текстов для векторизации.
            progress_callback: необязательная функция (done, total) —
                вызывается после каждого успешного батча для UI-прогресса.
            
        Returns:
            Список векторов (каждый — список float).
        """
        if not texts:
            return []

        self._ensure_available()

        all_embeddings: List[List[float]] = []
        
        # Разбиваем на батчи
        batches = [texts[i:i + self.batch_size] for i in range(0, len(texts), self.batch_size)]
        
        logger.info(f"Векторизация {len(texts)} текстов в {len(batches)} батчах (batch_size={self.batch_size})")
        
        for batch_idx, batch in enumerate(tqdm(batches, desc="Векторизация"), start=1):
            try:
                # Ollama /api/embed поддерживает batch: input = [str, str, ...]
                response = requests.post(
                    f"{self.base_url}/api/embed",
                    json=self._embed_body(batch),
                    timeout=self.timeout
                )
                if response.status_code >= 400:
                    detail = self._api_error_detail(response)
                    raise requests.exceptions.HTTPError(
                        f"{response.status_code} Client Error: Bad Request for "
                        f"url: {self.base_url}/api/embed | ответ сервера: {detail}",
                        response=response,
                    )
                data = response.json()
                
                # Ollama возвращает {"embeddings": [[...], [...], ...]}
                if "embeddings" in data:
                    batch_embeddings = data["embeddings"]
                    if len(batch_embeddings) != len(batch):
                        raise LamaLocaError(
                            f"Несоответствие размеров: запросили {len(batch)}, получили {len(batch_embeddings)}"
                        )
                    all_embeddings.extend(batch_embeddings)
                    # PERF/FIX: фактическая верификация GPU/CPU по метрикам
                    # первого реального батча (см. _runtime_gpu_report).
                    if self._pending_gpu_check and not self._first_batch_done:
                        self._first_batch_done = True
                        self._pending_gpu_check = False
                        self._runtime_gpu_report(data, len(batch))
                else:
                    raise LamaLocaError(f"Неожиданный формат ответа Ollama: {data.keys()}")
                    
            except requests.exceptions.RequestException as e:
                logger.error(f"Ошибка при векторизации батча {batch_idx}: {e}")
                raise LamaLocaError(f"Сбой Ollama API на батче {batch_idx}: {e}")

            if progress_callback:
                try:
                    progress_callback(len(all_embeddings), len(texts))
                except Exception:
                    pass  # прогресс не должен ронять векторизацию
        
        logger.info(f"✅ Векторизация завершена: {len(all_embeddings)} векторов")
        return all_embeddings