# src/llm/rag_pipeline.py
"""
RAG-пайплайн с навигацией по структуре документа, гибридным поиском
(FTS5 + векторы + RRF-ранжирование) и памятью диалога.

АУДИТ (переписано):
1. Зависимости (SQLiteStore, ChromaStore, OllamaEmbedder) теперь инжектируются
   извне (composition root - src/core/application.py). Ранее пайплайн создавал
   собственные экземпляры хранилищ: в приложении одновременно открывалось
   три независимых соединения к одной SQLite и два клиента ChromaDB.
2. Гибридный поиск: добавлено RRF-ранжирование (Reciprocal Rank Fusion, k=60)
   вместо конкатенации списков - как требует ТЗ.
3. FTS-запрос: раньше весь вопрос оборачивался во фразу "...", что давало почти
   всегда 0 попаданий на естественных вопросах. Теперь запрос разбивается на
   слова, каждое экранируется, склейка - OR (bm25 ранжирует).
4. Добавлена контекстная память диалога (последние N обменов) - требование ТЗ;
   ранее каждый вопрос обрабатывался изолированно.
5. TOC для не-PDF документов (EPUB) берётся из HEADING-блоков в SQLite, а не
   только через pymupdf (раньше для EPUB оглавление было недоступно).
"""
import json
import re
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pymupdf
from loguru import logger

from src.processing.text_cleaner import normalize_metadata


def _to_dicts(seq: Any) -> List[Dict[str, Any]]:
    """Потокобезопасный нормализатор последовательности dict-подобных элементов.

    FIX ('str' object has no attribute 'items'/'get'): исторически контракты
    модулей (SQLiteStore.expand_graph_context, возвраты ChromaDB, history)
    допускали ЭЛЕМЕНТЫ-строки — JSON-сериализованные словари (например,
    metadata рёбер графа, сохранённая через json.dumps). Любой потребитель,
    вызывающий .get()/.items() на таком элементе, падал с AttributeError уже
    ПОСЛЕ успешных логов пайплайна (поэтому в логе ошибка appearала позже
    строк Graph-RAG и склейки чанков). Здесь: dict — passthrough, str/bytes —
    json.loads (толерантно к одинарным кавычкам), мусор — {}. Ничего не
    бросаем: тихая деградация лучше падения всего ответа.
    """
    out: List[Dict[str, Any]] = []
    for it in (seq or []):
        if isinstance(it, dict):
            out.append(it)
            continue
        if isinstance(it, bytes):
            try:
                it = it.decode("utf-8", "replace")
            except Exception:
                out.append({})
                continue
        if isinstance(it, str):
            s = it.strip()
            parsed = None
            try:
                parsed = json.loads(s)
            except (ValueError, TypeError):
                try:
                    parsed = json.loads(s.replace("'", '"'))
                except (ValueError, TypeError):
                    parsed = None
            if isinstance(parsed, dict):
                out.append(parsed)
            elif isinstance(parsed, list):
                out.extend(d for d in parsed if isinstance(d, dict))
            else:
                out.append({})
            continue
        out.append({})
    return out

from .llm_engine import LLMEngine
from .prompt_builder import PromptBuilder
from ..storage.sqlite_store import SQLiteStore
from ..storage.chroma_store import ChromaStore
from ..embeddings.ollama_embedder import OllamaEmbedder
from ..core.settings import Settings
from ..processing.ocr_processor import OCRProcessor

# Параметр сглаживания RRF (стандартное значение из оригинальной статьи).
RRF_K = 60
# Сколько последних обменов держать в памяти диалога.
MAX_HISTORY_TURNS = 4


class RAGPipeline:
    """RAG-пайплайн с навигацией по структуре документа."""

    def __init__(
        self,
        settings: Settings,
        sqlite_store: SQLiteStore,
        chroma_store: ChromaStore,
        embedder: OllamaEmbedder,
    ):
        self.settings = settings
        # DI: переиспользуем общие экземпляры хранилищ (создаёт Application).
        self.llm = LLMEngine(settings.ollama)
        self.embedder = embedder
        self.sqlite_store = sqlite_store
        self.chroma_store = chroma_store
        self.prompt_builder = PromptBuilder()

        tesseract_cmd = getattr(settings, "tesseract_cmd", None)
        self.ocr_processor = None
        if tesseract_cmd and Path(tesseract_cmd).exists():
            try:
                self.ocr_processor = OCRProcessor(tesseract_cmd, language="rus+eng")
                logger.info("OCR (Tesseract) доступен как fallback")
            except Exception as e:
                logger.warning(f"OCR недоступен: {e}")

        # Кэш и контекст активного документа
        self._toc_cache: Optional[List[Tuple[int, str, int]]] = None
        self._active_doc_id: Optional[str] = None
        self._current_pdf_path: Optional[Path] = None

        # Память диалога: список {"question", "answer"} последних обменов.
        self.history: List[Dict[str, str]] = []

    # ------------------------------------------------------------------
    # Управление состоянием
    # ------------------------------------------------------------------

    def set_active_document(self, doc_id: Optional[str]):
        """Явно устанавливает активный документ для RAG (устраняет LIMIT 1)."""
        self._active_doc_id = doc_id
        self._toc_cache = None
        self._current_pdf_path = None
        if doc_id:
            logger.info(f"Активный документ установлен: {doc_id[:8]}...")

    def reset_history(self):
        """Очищает память диалога."""
        self.history.clear()

    def answer(self, question: str) -> Dict[str, Any]:
        """Отвечает на вопрос с использованием RAG и навигации по документу."""
        logger.info(f"RAG: вопрос - '{question}'")

        intent = self._classify_intent(question)
        logger.info(f"Намерение: {intent}")

        if intent == "toc":
            result = self._handle_toc_request()
        elif intent == "summary":
            # FIX: конспект строится по РЕЛЕВАНТНЫМ чанкам гибридного поиска,
            # а не по сырому тексту страниц подряд.
            result = self._handle_summary_request(question)
        elif intent == "chapter":
            result = self._handle_chapter_request(question)
        else:
            result = self._handle_semantic_request(question)

        # REFACTORING (адаптивный layout): прокидываем намерение в результат —
        # UI по нему решает, раскрывать ли панель документа (toc/chapter).
        result.setdefault("intent", intent)

        # FIX ('str' object has no attribute 'items'): ЖЁСТКАЯ ГРАНУЛЯРНАЯ
        # ГАРАНТИЯ контрактов ответа. Ошибка всплывала в UI-потоке уже ПОСЛЕ
        # логов поиска/склейки, т.к. нарушались типовые контракты хранилищ
        # (SQLite/Chroma отдавали JSON-строки вместо dict; Ollama /api/show
        # может вернуть parameters/config строкой). Нормализуем все три поля
        # результата прямо здесь — что бы ни вернула внутренняя ветка, наружу
        # уходят только списки словарей. Потокобезопасно: чистые функции.
        result["sources"] = _to_dicts(result.get("sources") or [])
        result["context_chunks"] = _to_dicts(result.get("context_chunks") or [])
        result["answer"] = str(result.get("answer") or "")

        # Запоминаем обмен для контекста диалога.
        self.history.append({"question": question, "answer": result.get("answer", "")})
        if len(self.history) > MAX_HISTORY_TURNS:
            del self.history[:-MAX_HISTORY_TURNS]

        return result

    # ------------------------------------------------------------------
    # Классификация намерений
    # ------------------------------------------------------------------

    # Ключевые слова запросов на конспект/саммари (намерение "summary").
    # Проверяются ДО chapter: «подготовь конспект по множествам» содержит
    # слово «по», но не является запросом конкретной главы.
    SUMMARY_PHRASES = [
        "конспект", "кратко", "краткий", "краткое", "резюме", "саммари",
        "суммариз", "перескажи", "пересказ", "объясни прост", "объясни доступн",
        "изложи", "выжимк", "tl;dr", "summary", "сделай обзор", "обзор темы",
    ]

    def _classify_intent(self, question: str) -> str:
        """Классифицирует намерение пользователя (toc / summary / chapter / semantic)."""
        question_lower = question.lower().strip()

        # FIX (69f1537-ре): запросы вида «подготовь конспект по ...» раньше
        # попадали в ветку chapter/semantic и возвращали сырой текст книги.
        if any(phrase in question_lower for phrase in self.SUMMARY_PHRASES):
            return "summary"

        toc_phrases = ["оглавление", "содержание", "содержимое", "структура книги",
                       "список глав", "что в книге"]
        for phrase in toc_phrases:
            if phrase in question_lower:
                return "toc"
            # Fuzzy-сравнение всего вопроса осмысленно только для коротких
            # запросов; для длинных предложений ratio() давала ложные toc.
            if len(question_lower) <= 30 and \
               SequenceMatcher(None, question_lower, phrase).ratio() >= 0.7:
                return "toc"
            for word in question_lower.split():
                # Ловим опечатки в отдельном слове ("оглавдение" -> "оглавление").
                if len(word) >= 5 and SequenceMatcher(None, word, phrase).ratio() >= 0.8:
                    return "toc"

        if re.search(r'глав[ауы]\s+["\']?([^"\']+?)["\']?', question_lower) or \
           re.search(r'параграф[ауы]?\s+(\d+[\.\d]*)', question_lower) or \
           re.search(r'§\s*(\d+[\.\d]*)', question_lower):
            return "chapter"

        return "semantic"

    # ------------------------------------------------------------------
    # Обработчики намерений
    # ------------------------------------------------------------------

    def _handle_toc_request(self) -> Dict[str, Any]:
        return self._get_toc_from_metadata()

    def _handle_summary_request(self, question: str) -> Dict[str, Any]:
        """Конспект по теме: релевантные чанки (гибридный поиск с RRF) +
        специальный summary-промпт. В отличие от chapter-ветки никогда не
        возвращает сырой текст страниц — ответ всегда генерирует LLM."""
        query_embedding = self.embedder.embed_texts([question])[0]
        # FIX (короткий конспект): 12 релевантных чанков давали модели материал
        # лишь на 2-3 абзаца. 16 чанков (~12-16k символов после RRF-отбора)
        # покрывают всю тему и позволяют выдать полные 500-800 слов.
        # FIX (проблема 3 ТЗ): конспект строится по широкой выборке —
        # 20 релевантных чанков вместо 16, чтобы модели хватило материала
        # на полный разбор темы (700-1000 слов).
        # AUDIT ('str' object has no attribute 'items'): каждый этап конвейера
        # нормализует тип ЭЛЕМЕНТОВ списка (_to_dicts) — исторически SQLite/
        # Chroma возвращали JSON-строки вместо dict; .get()/.items() на таком
        # элементе роняли весь ответ уже ПОСЛЕ логов поиска (как в логе 22:49).
        context_chunks = _to_dicts(
            self._hybrid_search(question, query_embedding, k=20))

        if not context_chunks:
            return {"answer": "Не нашёл в библиотеке материалов для конспекта "
                              "по этой теме.",
                    "sources": [], "context_chunks": []}

        # GRAPH-RAG (Этап 2): раскрываем подграф знаний от точек входа.
        context_chunks = _to_dicts(self._expand_with_graph(context_chunks))

        # AUDIT (порядок конвейера): склейка соседних чанков выполняется ПОСЛЕ
        # графового расширения — иначе добавленные графом вершины (доказательство
        # к найденной теореме) оставались бы отдельными кусками и модель получала
        # «нарванный» контекст. Теперь холст склеивается по всей раскрытой
        # окрестности главы.
        context_chunks = _to_dicts(self._merge_adjacent_chunks(context_chunks))

        messages = self.prompt_builder.build_summary_prompt(
            question, context_chunks, history=_to_dicts(self.history)
        )
        # FIX ('str' object has no attribute 'items', лог 23:01): последний
        # рубеж перед движком. Выше по конвейеру нормализованы metadata и
        # history, но сам список messages может содержать JSON-строки вместо
        # dict (историческая сериализация промптов). _to_dicts гарантирует:
        # в llm_engine/prompt_builder уходят только словари — .get()/.items()
        # на элементах больше не способны уронить UI-поток. Пустые элементы
        # отбрасываются, чтобы trim/оценка токенов не считали мусор.
        messages = [m for m in _to_dicts(messages) if m.get("role")]
        answer = self._chat_with_retry(messages)
        answer = self._strip_code_fences(answer)
        # FIX («ИИ-коллаж»): анти-фрагментация — до извлечения источников,
        # чтобы инлайн-метки [N] остались валидными ключами для _extract_sources.
        answer = self._anti_collage(answer)
        sources = self._extract_sources(answer, context_chunks)
        return {"answer": answer, "sources": sources,
                "context_chunks": context_chunks}

    @staticmethod
    def _strip_code_fences(text: str) -> str:
        """FIX (проблема «сырого» markdown): мелкие модели иногда оборачивают
        весь ответ в ```markdown ... ``` — тогда ft.Markdown показывает
        решётки и трубы буквально. Снимаем внешние обёртки.

        FIX (рассогласование форматов): снятые обёртки могут обнажить сырой
        UnicodeMath/\\[..\\] или нечётные маркеры, пришедшие из контекста через
        копирование моделью. Ответ проходит math_guard — канон KaTeX-маркеров
        ($...$/$$...$$) и чётность ограничителей, иначе KaTeX фронтенда падает
        с throwOnError и пользователь видит сырой текст вместо формул.
        """
        t = (text or "").strip()
        if t.startswith("```") and t.endswith("```"):
            body = t[3:]
            nl = body.find("\n")
            if nl != -1:
                lang = body[:nl].strip().lower()
                if lang in ("", "md", "markdown"):
                    t = body[nl + 1:].strip()
        from src.llm.prompt_builder import PromptBuilder
        return PromptBuilder.math_guard(t)

    @staticmethod
    def _fit_context(text: str, budget: int = 14000) -> str:
        """Упаковывает длинный текст в бюджет символов РАВНОМЕРНО.

        Вместо отсечения начала («голова» главы) разбиваем текст на страницы
        и берём пропорциональные фрагменты от каждой, сохраняя связность
        по всей главе. Если текст влезает целиком — возвращаем как есть.
        """
        if len(text) <= budget:
            return text
        parts = [p for p in text.split("\n\n") if p.strip()]
        if not parts:
            return text[:budget]
        per = max(300, budget // len(parts))
        out, used = [], 0
        for i, part in enumerate(parts):
            if len(part) > per:
                out.append(part[:per] + f" …[стр.{i + 1} сжата]")
            else:
                out.append(part)
            used += len(out[-1])
            if used >= budget:
                out.append("\n…[контекст главы равномерно сжат до лимита модели]")
                break
        return "\n\n".join(out)

    # Бюджет «полного холста» главы в символах. ~24k токенов при 3 симв/токен
    # (оценка estimate_tokens в LLMEngine); RTX 5060 + qwen держит num_ctx 24-32k.
    CHAPTER_HOLST_CHARS = 62000

    def _build_chapter_holst_from_db(self, chapter_title: str,
                                     start_page: int,
                                     end_page: Optional[int]) -> Optional[Tuple[str, List[Dict[str, Any]]]]:
        """REFACTORING (ТЗ п.1 «Full-Chapter Context»): собирает ВЕСЬ параграф
        из SQLite (направленный запрос по chapter_name), а не из PDF и не из
        top_k нарезки поиска. Возвращает (текст_холста, context_chunks) или
        None, если глав в базе нет — тогда вызывающий деградирует к старому
        PDF-пути / гибридному поиску (обратная совместимость)."""
        if not self._active_doc_id:
            return None
        try:
            blocks = self.sqlite_store.get_chapter_blocks(
                self._active_doc_id, chapter_title,
                start_page=start_page if start_page else None,
                end_page=end_page if end_page else None,
            )
        except Exception as e:
            logger.warning(f"Сборка главы из SQLite не удалась: {e}")
            return None
        texts = [b["text"].strip() for b in blocks
                 if (b.get("text") or "").strip() and b.get("block_type") != "heading"]
        if not texts:
            return None
        holst = "\n\n".join(texts)
        pages = sorted({int(b["page_number"]) for b in blocks
                        if isinstance(b.get("page_number"), (int, float))})
        chunks_meta = {
            "title": chapter_title,
            "chapter": chapter_title,
            "page_number": f"{pages[0]}-{pages[-1]}" if pages else str(start_page),
        }
        if len(holst) > self.CHAPTER_HOLST_CHARS:
            holst = self._fit_context(holst, budget=self.CHAPTER_HOLST_CHARS)
        logger.info(
            f"Full-Chapter Context: '{chapter_title}' — {len(texts)} блоков, "
            f"{len(holst)} символов из SQLite (стр. {chunks_meta['page_number']})")
        return holst, [{"text": holst, "metadata": chunks_meta}]

    def _handle_chapter_request(self, question: str) -> Dict[str, Any]:
        chapter_name = self._extract_chapter_name(question)
        if not chapter_name:
            return {"answer": "Не понял, какую главу вы имеете в виду.",
                    "sources": [], "context_chunks": []}

        chapter_info = self._find_chapter_in_toc(chapter_name)
        if chapter_info:
            level, title, start_page = chapter_info
            # Шаг 1 ТЗ: определяем конец главы по следующей записи TOC того же
            # уровня — чтобы выбрать из SQLite именно параграф, а не всю книгу.
            end_page = None
            try:
                toc = self._get_toc()
                for i, (lvl, t, p) in enumerate(toc):
                    if t == title and p == start_page:
                        for j in range(i + 1, len(toc)):
                            if toc[j][0] <= level:
                                end_page = max(start_page, toc[j][2] - 1)
                                break
                        break
            except Exception:
                end_page = None
            built = self._build_chapter_holst_from_db(title, start_page, end_page)
            if built:
                chapter_text, context_chunks = built
                result = self._summarize_chapter(title, start_page,
                                                 end_page or start_page,
                                                 chapter_text, question)
                result["context_chunks"] = context_chunks
                return result
            # Фолбэк: старый путь через PDF (EPUB/битые блоки) — не ломаем.
            return self._extract_chapter_text(title, start_page, level,
                                              question=question)

        # Для EPUB/сканов без числовых страниц: обычный гибридный поиск
        # по формулировке вопроса (название главы учтётся в FTS/векторе).
        return self._handle_semantic_request(question)

    def _handle_semantic_request(self, question: str) -> Dict[str, Any]:
        query_embedding = self.embedder.embed_texts([question])[0]
        # AUDIT ('str' object has no attribute 'items'): нормализация типа
        # элементов на входе каждого канала — та же гарантия, что в summary-
        # ветке (см. комментарий к _to_dicts там).
        # FIX («ИИ-коллаж», перегрузка контекста): для обычного вопроса берём
        # final_top_k из конфига (по умолчанию 8), а не раздутые 16-20: 22k+
        # токенов на входе сбивали qwen3.5 в режим «цитатник» — ответ
        # распадался на изолированные куски с [N] перед каждым. Конспект
        # (summary-ветка) по-прежнему получает широкие k=20.
        context_chunks = _to_dicts(self._hybrid_search(question, query_embedding))

        if not context_chunks:
            return {"answer": "Не нашёл информации по этому вопросу.",
                    "sources": [], "context_chunks": []}

        # GRAPH-RAG (Этап 2): раскрываем подграф от точек входа — найденная
        # теорема тянет за собой доказательство и определения (REFERENCES).
        context_chunks = _to_dicts(self._expand_with_graph(context_chunks))
        # Склеиваем соседние куски одного раздела в связные блоки (тот же
        # принцип, что и в summary-ветке): модель читает цельный текст.
        context_chunks = _to_dicts(self._merge_adjacent_chunks(context_chunks))

        # FIX (логи 13:01 и 14:16, «немой» пустой ответ при перегрузе):
        # контекст должен влезать в ЭФФЕКТИВНЫЙ бюджет модели ещё ДО отправки.
        # estimate_tokens() движка — та же метрика, по которой считает бюджет
        # _trim_history/_build_options. ВАЖНО: поле limit.context_length из
        # GGUF (qwen3.5:4b = 262144) — это АРХИТЕКТУРНЫЙ потолок RoPE, а не
        # реальный рабочий размер KV-кэша: Ollama по умолчанию аллоцирует
        # num_ctx всего 2048-4096, и запросы на 19-23k токенов упираются в
        # лимит слотов -> eval_count=None, done_reason='length', content=''
        # (см. лог: num_ctx=19624). Поэтому потолок бюджета жёстко ограничен
        # снизу: не более 12k токенов входа, независимо от «нативного» окна.
        try:
            # FIX (логи 11:29/11:44 «ответ не раскрывает тему»): префлайт
            # читал СТАТИЧЕСКИЙ дефолт _EFFECTIVE_PROMPT_TOKENS (12000), из-за
            # чего адаптивное повышение бюджета после подтверждения GPU-инференса
            # (_maybe_raise_budget → _prompt_budget=20000) игнорировалось, и
            # релевантные блоки («§ 13. ФОРМУЛА ТЕЙЛОРА…») отсекались навсегда.
            # Теперь метрика берётся из живого метода движка effective_prompt_budget().
            window_cap = self.llm.detect_model_window(self.llm.model)
            # FIX (лог 14:25): num_predict тоже ЕСТЬ бюджет окна. При
            # num_ctx=19624 и num_predict=8192 суммарный слот ~28k — Ollama
            # не аллоцирует KV-кэш такого размера на машине с 95% занятой
            # RAM и молча возвращает пустоту (done_reason='length',
            # eval_count=None). Поэтому потолок окна = budget + генерация +
            # reserve, и сам budget уменьшаем, если окно модели меньше.
            pred_cap = getattr(self.llm, "_ctx_cap", (0, 0))[1] or 8192
            gen_reserve = min(pred_cap, 4096) + 1024
            prompt_budget = None
            try:
                prompt_budget = self.llm.effective_prompt_budget()
            except Exception:  # noqa: BLE001  # старые движки без метода
                pass
            if not prompt_budget:
                from .llm_engine import _EFFECTIVE_PROMPT_TOKENS as prompt_budget  # type: ignore[no-redef]
            window_cap = min(window_cap, prompt_budget + gen_reserve)
            ctx_cap, _ = getattr(self.llm, "_ctx_cap", (0, 0))
            if ctx_cap:
                window_cap = min(window_cap, ctx_cap)
            # Эффективный бюджет: сколько промпт реально переживает без
            # «немого» length-обрыва. Адаптивный (12k старт / 20k после
            # подтверждения GPU) и не более окна модели минус генерация.
            budget = max(2048, min(window_cap - gen_reserve, prompt_budget))
            # FIX (логи 00:56/10:48): раньше префлайт строил probe БЕЗ
            # истории диалога, а реальный запрос — С историей; при втором+
            # вопросе оценка занижалась на 1-3k токенов, фильтр пропускал
            # перегруз и Ollama снова молчала (eval_count=None,
            # done_reason='length'). Теперь метрика probe идентична
            # финальному messages.
            hist = _to_dicts(self.history)

            def _probe_est(chunks):
                """Оценка промпта «как в бою»: через probe_fit движка — тот
                же chat-шаблон, история и скользящее окно, что реально
                уйдут в Ollama (FIX логов 11:29: голова оценка по тексту
                контекста занижала метрику на ~15%, и префлайт пропускал
                перегруз -> пустые ответы при num_ctx 12-16k)."""
                msgs = self.prompt_builder.build_rag_prompt(
                    question, chunks, history=hist)
                fit = None
                try:
                    fit = self.llm.probe_fit(msgs)
                except Exception:  # noqa: BLE001
                    fit = None
                if fit is not None:
                    return fit[3]
                return self.llm.estimate_tokens(msgs)

            while len(context_chunks) > 2:
                est = _probe_est(context_chunks)
                if est <= budget:
                    break
                dropped = context_chunks.pop()
                logger.info(
                    f"RAG: контекст ({est} токенов) превышает эффективный "
                    f"бюджет окна ({budget}) — блок '{str(dropped.get('text') or '')[:40]}…' "
                    f"исключён (защита от пустого ответа)")
        except Exception as ex:
            logger.debug(f"RAG: префлайт-проверка бюджета контекста "
                         f"пропущена ({ex})")

        messages = self.prompt_builder.build_rag_prompt(
            question, context_chunks, history=_to_dicts(self.history)
        )
        answer = self._chat_with_retry(messages)
        answer = self._strip_code_fences(answer)
        # FIX («ИИ-коллаж»): снятие «коллажа» до извлечения источников —
        # изолированные куски с [N] на каждой строке, дубли предложений и
        # хвостовые списки «Источники» чистятся в _anti_collage.
        answer = self._anti_collage(answer)
        sources = self._extract_sources(answer, context_chunks)

        # AUDIT (система учит, а не молчит): если LLM так и не дала содержательного
        # ответа, не оставляем пустоту — автоматически запускаем полноценный
        # конспект по найденному контексту (summary-промпт сильнее рвёт шаблон
        # «пересказать кусок», чем обычный RAG-промпт).
        if len((answer or "").strip()) < 80 and context_chunks:
            logger.info("RAG: короткий/пустой ответ semantic-ветки — "
                        "авто-апгрейд до полного конспекта (summary-промпт)")
            sum_messages = self.prompt_builder.build_summary_prompt(
                f"Дай полный структурированный ответ на вопрос: {question}",
                context_chunks, history=self.history)
            upgraded = self._strip_code_fences(self._chat_with_retry(sum_messages))
            if len(upgraded.strip()) > len((answer or "").strip()):
                answer = upgraded
                sources = self._extract_sources(answer, context_chunks)

        if not (answer or "").strip():
            # FIX: вместо пустого ответа — явное сообщение; UI больше не
            # рисует пустой пузырь с кнопками экспорта «в никуда».
            return {"answer":
                    "Модель не сгенерировала ответ по найденным материалам. "
                    "Проверьте, что Ollama запущена и chat_model из config.yaml "
                    "присутствует в `ollama list`.",
                    "sources": sources, "context_chunks": context_chunks}

        return {"answer": answer, "sources": sources, "context_chunks": context_chunks}

    def _chat_with_retry(self, messages: List[Dict[str, Any]]) -> str:
        """Запрос к LLM с каскадом спасения при пустом выводе.

        Мелкие модели (qwen3.5:4b) молча возвращают пустоту, когда промпт
        физически не влезает в KV-кэш/лимит слотов Ollama (лог 14:16:
        num_ctx=19624, eval_count=None, done_reason='length', content='').
        Движок llm_engine уже делает один перезапрос с уменьшенным окном;
        если и он дал пустоту — здесь включается каскад:
          1) обычный повтор;
          2) урезанный контекст (берём половину лучших чанков RAG);
          3) «голый» вопрос без контекста — лучше общий ответ, чем пустой
             пузырь (источники в этом случае останутся кликабельными).
        Каждый шаг идемпотентен по стоимости: выполняется только при пустоте
        предыдущего.
        """
        answer = self.llm.chat(messages)
        if (answer or "").strip():
            return answer

        logger.warning("RAG: пустой ответ модели — повторяю запрос один раз")
        answer = self.llm.chat(messages)
        if (answer or "").strip():
            return answer or ""

        # Каскад 2: урезаем контекст вдвое (сохраняем system + последний user).
        try:
            reduced = self._reduce_context_messages(messages)
            if reduced is not None and len(reduced) == len(messages):
                reduced = None
            if reduced is not None:
                logger.warning(
                    "RAG: повтор снова пуст — перезапрос с урезанным "
                    "контекстом (перегруз окна модели)")
                answer = self.llm.chat(reduced)
                if (answer or "").strip():
                    return answer
        except Exception as ex:  # каскад — косметика, не роняем конвейер
            logger.warning(f"RAG: каскад усечения контекста не удался ({ex})")

        # Каскад 3: без контекста вовсе — модель отвечает по своим знаниям.
        try:
            bare = [m for m in messages if m.get("role") != "user"]
            last_user = next((m for m in reversed(messages)
                              if m.get("role") == "user"), None)
            if last_user is not None:
                q = str(last_user.get("content") or "")
                # Выдёргиваем сам вопрос из шаблонного user-промпта.
                m_q = re.search(r"Вопрос:\s*(.+?)(?:\n|$)", q)
                question_only = m_q.group(1).strip() if m_q else q[-500:]
                bare = bare + [{"role": "user",
                                 "content": f"Вопрос: {question_only}\n\n"
                                            "Контекст недоступен — ответь "
                                            "по общим знаниям математики, "
                                            "честно предупредив об этом."}]
                logger.warning(
                    "RAG: контекст не влез в окно модели — финальный "
                    "перезапрос БЕЗ контекста (лучше общий ответ, чем пустота)")
                answer = self.llm.chat(bare)
        except Exception as ex:
            logger.warning(f"RAG: каскад без контекста не удался ({ex})")
        return answer or ""

    @staticmethod
    def _reduce_context_messages(messages: List[Dict[str, Any]]
                                 ) -> Optional[List[Dict[str, Any]]]:
        """Половина чанков RAG в user-промпте (для каскада перезапроса).

        Контекст в build_rag_prompt отделён маркером «Контекст из книг:» от
        инструкции; режем его по разделителю блоков '\\n\\n---\\n\\n', оставляя
        первые ceil(n/2) блока. Структура сообщений сохраняется.
        """
        idx = None
        for i, m in enumerate(messages):
            if m.get("role") == "user" and "Контекст из книг:" in str(
                    m.get("content") or ""):
                idx = i
                break
        if idx is None:
            return None
        content = str(messages[idx].get("content") or "")
        head, sep, rest = content.partition("Контекст из книг:")
        if not sep:
            return None
        body, marker, tail = rest.partition("\n\nВопрос:")
        blocks = [b for b in body.split("\n\n---\n\n") if b.strip()]
        if len(blocks) < 2:
            return None
        keep = blocks[:max(1, (len(blocks) + 1) // 2)]
        new_content = (head + sep + "\n\n".join(keep)
                       + ("\n\n[...часть контекста опущена: перегруз окна "
                          "модели...]\n" if len(keep) < len(blocks) else "")
                       + marker + tail)
        reduced = [dict(m) for m in messages]
        reduced[idx]["content"] = new_content
        return reduced

    # ------------------------------------------------------------------
    # FIX («ИИ-коллаж»): анти-фрагментация ответа на перегруженном контексте
    # ------------------------------------------------------------------

    # Маркер типа сущности в НАЧАЛЕ абзаца ([[DEFINITION]] …), который модель
    # обязана ставить только для определений/теорем из контекста. При коллаже
    # qwen3.5 обвешивает ими каждый кусок — чистим.
    _ENTITY_MARKER_RE = re.compile(
        r"^[\s>*\-]*\[\[(?:DEFINITION|THEOREM|PROOF|LEMMA|AXIOM|COROLLARY)\]\]\s*",
        re.I | re.M)

    # Цепочки скобок [3][7] / [3] [7] / [1][2][3].
    _BRACKET_RUN_RE = re.compile(r"(?:\[\d{1,2}\][ \t]*){2,}")

    # Заголовок-обманка «Источники» (дублирование списка — запрещено ТЗ).
    _SOURCES_HEAD_RE = re.compile(r"^\s{0,3}#{1,6}\s*[\U0001F4DA\s]*[Ии]сточники.*$",
                                  re.M)

    def _dedupe_sentences(self, text: str) -> str:
        """Удаляет подряд идущие дубликаты предложений (главный симптом
        коллажа: модель печатает одно и то же предложение дважды)."""
        changed = True
        while changed:
            changed = False
            for m in re.finditer(r"(?P<s>[^.!?\n]{20,}[.!?])(?P<ws>[ \t]*|\n+)(?=\.!\?A-Za-zА-Яа-я«”\\$])",
                                 text):
                s = m.group("s").strip()
                after = text[m.end():]
                nxt = re.match(r"\s*(?P<t>[^.!?\n]{20,}[.!?])", after)
                if not nxt:
                    continue
                t = nxt.group("t").strip()
                key_s = re.sub(r"[\W_]+", "", s.lower())[:80]
                key_t = re.sub(r"[\W_]+", "", t.lower())[:80]
                if key_s and key_s == key_t:
                    text = text[:m.start()] + text[m.end():]
                    changed = True
                    break
        return text

    def _anti_collage(self, answer: str) -> str:
        """Постобработка «ИИ-коллажа» (см. жалобу: изолированные куски, перед
        каждым — индекс чанка [3], [7], [3], [2]).

        1. Схлопывает одиночные строки-«осколки» (короткие предложения без
           знаков конца) со следующим абзацем — это обрезки цитат, разорванные
           моделью.
        2. Удаляет служебные [[TYPE]]-маркеры, расставленные не по делу.
        3. Сливает цепочки [N][M] в одну [N] (не более двух) — плотные ссылки
           читаются как шум.
        4. Срезает хвостовой раздел «Источники» (дублирование списков).
        5. Убирает дословные повторы предложений подряд.
        Все шаги идемпотентны и не трогают $$…$$-блоки и markdown-таблицы.
        """
        if not answer:
            return answer
        try:
            # 2. маркеры типов
            answer = self._ENTITY_MARKER_RE.sub("", answer)

            # 1. склейка осколков: абзац из <=4 слов без терминальной точки
            lines = answer.split("\n")
            out: List[str] = []
            buf = ""
            for ln in lines:
                stripped = ln.strip()
                is_frag = (0 < len(stripped.split()) <= 4
                           and not stripped.endswith((".", "!", "?", ":", ";"))
                           and not stripped.startswith(("#", "|", "-", "*", "$", ">", "["))
                           and "**" not in stripped)
                if is_frag:
                    buf = (buf + " " + stripped).strip()
                    continue
                if buf:
                    if stripped:
                        stripped = buf + ", " + stripped[0].lower() + stripped[1:] \
                            if stripped[0].isalpha() and stripped[0].isascii() and stripped[0].islower() is False \
                            else buf + " " + stripped
                    else:
                        stripped = buf
                    buf = ""
                out.append(stripped if (stripped or (out and out[-1] == "")) else "")
            if buf:
                out.append(buf)
            answer = "\n".join(out)

            # 3. цепочки ссылок [1][2][3] -> [1][2]
            def _fold_run(m: re.Match) -> str:
                nums = re.findall(r"\d+", m.group(0))[:2]
                return "".join(f"[{n}]" for n in nums)
            answer = self._BRACKET_RUN_RE.sub(_fold_run, answer)

            # 4. хвостовой список источников
            answer = self._SOURCES_HEAD_RE.sub("", answer)

            # 5. дубли предложений
            answer = self._dedupe_sentences(answer)

            # финальная полировка: не более одного пустого ряда, пробелы перед знаками
            answer = re.sub(r"\n{3,}", "\n\n", answer)
            answer = re.sub(r"[ \t]+([.,;:])", r"\1", answer)
            return answer.strip()
        except Exception as ex:  # никогда не роняем ответ из-за косметики
            logger.warning(f"_anti_collage: пропуск постобработки ({ex})")
            return answer

    # ------------------------------------------------------------------
    # GRAPH-RAG (Этап 2): Context-Expansion Pipeline (Graph Walk)
    # ------------------------------------------------------------------

    _GRAPH_BUDGET_CHARS = 26000   # потолок символов графового расширения

    def _expand_with_graph(
        self, chunks: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Раскрывает подграф знаний от точек входа гибридного поиска.

        Точки входа (первые 8 чанков RRF) используются как seeds для BFS по
        таблице entity_relations: теорема автоматически тянет свои доказательства
        (PROVES) и определения, на которые ссылается (REFERENCES). Добавленные
        вершины получают метку entity_type/hop в metadata, чтобы prompt_builder
        разметил их структурно. Всё остальное поведение пайплайна не меняется;
        при отсутствии графа (старые книги без реимпорта) возвращается исходный
        список — полная обратная совместимость.

        FIX ('str' object has no attribute 'items'): node["metadata"] из
        SQLite может прийти как JSON-строка (сериализованный dict), а не как
        dict — дальнейший .get() падал внутри try и молча глушил весь
        графовый канал (в логах было видно расширение контекста, но метки
        сущностей в промпт не попадали). Нормализуем тип до использования.
        """
        try:
            seeds = []
            # АУДИТ: сначала собираем id ВСЕХ чанков (не только первых 8) —
            # иначе графовые узлы, уже присутствующие в хвосте списка,
            # добавлялись дублями и раздували контекст сверх окна модели.
            existing_ids = {self._item_id(ch) for ch in _to_dicts(chunks)}
            for ch in _to_dicts(chunks)[:8]:
                cid = self._item_id(ch)
                if cid:
                    seeds.append(cid)
            if not seeds:
                return chunks

            extra_nodes = _to_dicts(self.sqlite_store.expand_graph_context(
                seed_ids=seeds, max_hops=2, limit=40,
            ))
            if not extra_nodes:
                return chunks

            budget = self._GRAPH_BUDGET_CHARS
            total_chars = sum(len(str(ch.get("text", ""))) for ch in _to_dicts(chunks))
            added = 0
            expanded = list(chunks)
            for node in extra_nodes:
                bid = node.get("block_id") or node.get("chunk_id") or node.get("id")
                if not bid or bid in existing_ids:
                    continue
                text_len = len(node.get("text") or "")
                if total_chars + text_len > budget:
                    break
                meta = normalize_metadata(node.get("metadata"))
                item = {
                    "id": bid,
                    "chunk_id": bid,
                    "text": node.get("text") or "",
                    "metadata": {
                        "page_number": node.get("page_number"),
                        "chapter": node.get("chapter", ""),
                        "section": node.get("section", ""),
                        "entity_type": node.get("entity_type", "EXPLANATION"),
                        "graph_hop": node.get("hop", 1),
                        "doc_id": self._active_doc_id or "",
                    },
                }
                expanded.append(item)
                existing_ids.add(bid)
                total_chars += text_len
                added += 1
            if added:
                logger.info(
                    f"Graph-RAG: контекст расширен на {added} связанных сущностей "
                    f"(hops<=2)"
                )
            return expanded
        except Exception as e:
            logger.warning(f"Graph-RAG: расширение контекста пропущено: {e}")
            return chunks

    # ------------------------------------------------------------------
    # Главы и оглавление
    # ------------------------------------------------------------------

    def _extract_chapter_name(self, question: str) -> Optional[str]:
        question_lower = question.lower()
        patterns = [r'глав[ауы]\s+["\']?([^"\']+?)["\']?',
                    r'параграф[ауы]?\s+(\d+[\.\d]*)', r'§\s*(\d+[\.\d]*)']
        for pattern in patterns:
            match = re.search(pattern, question_lower)
            if match:
                return match.group(1).strip()
        return None

    def _find_chapter_in_toc(self, query: str) -> Optional[Tuple[int, str, int]]:
        toc = self._get_toc()
        if not toc:
            return None

        query_lower = query.lower().strip()
        best_match, best_score = None, 0.0

        for level, title, page in toc:
            title_lower = title.lower().strip()
            if query_lower in title_lower or title_lower in query_lower:
                return (level, title, page)

            similarity = SequenceMatcher(None, query_lower, title_lower).ratio()
            q_words = set(query_lower.split())
            t_words = set(title_lower.split())
            word_score = len(q_words & t_words) / max(len(q_words), 1)
            combined_score = max(similarity, word_score)

            if combined_score > best_score and combined_score >= 0.5:
                best_score, best_match = combined_score, (level, title, page)

        return best_match

    def _summarize_chapter(self, chapter_title: str, start_page: int,
                           end_page: int, chapter_text: str,
                           question: str) -> Dict[str, Any]:
        """Генерирует структурированное изложение главы через LLM.

        Общий код для обоих путей сборки контекста: новый (SQLite Full-Chapter)
        и старый (PDF). При недоступности модели — честный фолбэк оригиналом
        с нормализованным регистром (как раньше)."""
        try:
            messages = self.prompt_builder.build_chapter_prompt(
                chapter_title, start_page, end_page,
                chapter_text, question, history=self.history,
            )
            answer = self._chat_with_retry(messages)
            if not (answer or "").strip():
                raise RuntimeError("пустой ответ модели")
            answer = self._strip_code_fences(answer)
        except Exception as ex:
            logger.warning(f"LLM недоступна для главы — показываю оригинал: {ex}")
            normalized = self.prompt_builder.normalize_ocr_case(chapter_text)
            answer = (
                f"⚠️ LLM недоступна ({ex}); ниже — оригинальный текст главы "
                f"(регистр нормализован):\n\n### {chapter_title}\n\n{normalized}"
            )
        return {
            "answer": answer,
            "raw_chapter_text": chapter_text,
            "sources": [{"title": chapter_title,
                         "page_number": f"{start_page}-{end_page}",
                         "chapter": chapter_title, "file_format": "pdf"}],
            "context_chunks": [{"text": chapter_text, "metadata": {
                "title": chapter_title,
                "page_number": f"{start_page}-{end_page}",
                "chapter": chapter_title}}],
        }

    def _extract_chapter_text(self, chapter_title: str, start_page: int,
                              level: int,
                              question: Optional[str] = None) -> Dict[str, Any]:
        pdf_path = self._get_current_pdf_path()
        if not pdf_path or pdf_path.suffix.lower() != ".pdf":
            # Не-PDF (EPUB): отдаём гибридный поиск по формулировке вопроса.
            return self._handle_semantic_request(question or chapter_title)

        try:
            doc = pymupdf.open(str(pdf_path))
            toc = self._get_toc()
            end_page = doc.page_count

            for i, (lvl, title, page) in enumerate(toc):
                if title == chapter_title and page == start_page:
                    for j in range(i + 1, len(toc)):
                        if toc[j][0] <= level:
                            end_page = toc[j][2] - 1
                            break
                    break

            # FIX (обрезка "…(текст обрезан)"): раньше главу рубали по голове
            # (первые 15k символов) — конец главы терялся, а модель получала
            # усечённый контекст. Теперь берём ВСЕ страницы главы, но при
            # перегрузе окна распределяем бюджет равномерно по всей главе
            # (окно вокруг каждой ключевой страницы), ничего не выбрасывая
            # молча; о сжатии честно сообщаем в самом тексте контекста.
            pages_text = []
            for page_num in range(start_page - 1, min(end_page, doc.page_count)):
                pages_text.append(doc[page_num].get_text("text"))
            doc.close()

            chapter_text = self._fit_context("\n\n".join(pages_text), budget=48000)

            # Общий путь генерации (тот же код, что и для SQLite-холста):
            # глава подаётся как контекст, структурированное изложение делает
            # LLM; при недоступности модели — фолбэк оригиналом с нормализацией.
            # FIX (гибкость): задание для модели — исходный вопрос
            # («законспектируй § 13 подробно»), а не только заголовок главы.
            result = self._summarize_chapter(chapter_title, start_page, end_page,
                                             chapter_text, question)
            result["pdf_path"] = str(pdf_path)
            return result
        except Exception as e:
            return {"answer": f"Ошибка при извлечении текста: {e}",
                    "sources": [], "context_chunks": []}

    def _get_toc(self) -> List[Tuple[int, str, int]]:
        """TOC активного документа: из PDF-метаданных, иначе из blocks (EPUB)."""
        if self._toc_cache is not None:
            return self._toc_cache

        pdf_path = self._get_current_pdf_path()
        toc: List[Tuple[int, str, int]] = []

        if pdf_path and pdf_path.suffix.lower() == ".pdf" and pdf_path.exists():
            try:
                doc = pymupdf.open(str(pdf_path))
                toc = doc.get_toc()
                doc.close()
            except Exception as e:
                logger.error(f"Ошибка чтения TOC: {e}")

        if not toc:
            toc = self._get_toc_from_blocks()

        self._toc_cache = toc
        return toc

    def _get_toc_from_blocks(self) -> List[Tuple[int, str, int]]:
        """Восстанавливает оглавление из HEADING-блоков в SQLite (для EPUB)."""
        if self._active_doc_id:
            doc_id = self._active_doc_id
        else:
            row = self.sqlite_store.conn.execute(
                "SELECT doc_id FROM documents ORDER BY created_at DESC LIMIT 1"
            ).fetchone()
            if not row:
                return []
            doc_id = row[0]

        rows = self.sqlite_store.conn.execute(
            """SELECT chapter, MIN(page_number) FROM blocks
               WHERE doc_id = ? AND block_type = 'heading' AND chapter != ''
               GROUP BY chapter ORDER BY rowid""",
            (doc_id,),
        ).fetchall()
        return [(1, chapter, page or 0) for chapter, page in rows]

    def _get_current_pdf_path(self) -> Optional[Path]:
        """Путь к файлу активного документа (или последнему добавленному)."""
        if self._current_pdf_path and self._current_pdf_path.exists():
            return self._current_pdf_path

        if self._active_doc_id:
            query = "SELECT source_path FROM documents WHERE doc_id = ?"
            params: Tuple = (self._active_doc_id,)
        else:
            query = "SELECT source_path FROM documents ORDER BY created_at DESC LIMIT 1"
            params = ()

        doc_result = self.sqlite_store.conn.execute(query, params).fetchone()
        if not doc_result:
            return None

        self._current_pdf_path = Path(doc_result[0])
        return self._current_pdf_path

    def _get_toc_from_metadata(self) -> Dict[str, Any]:
        """Оглавление: встроенное TOC PDF, иначе headings из SQLite, иначе OCR."""
        if self._active_doc_id:
            query = "SELECT source_path, title FROM documents WHERE doc_id = ?"
            params: Tuple = (self._active_doc_id,)
        else:
            query = "SELECT source_path, title FROM documents ORDER BY created_at DESC LIMIT 1"
            params = ()

        doc_result = self.sqlite_store.conn.execute(query, params).fetchone()
        if not doc_result:
            return {"answer": "Библиотека пуста.", "sources": [], "context_chunks": []}

        source_path, title = doc_result[0], doc_result[1]
        path = Path(source_path)

        if not path.exists():
            return {"answer": f"Файл не найден: {path}",
                    "sources": [], "context_chunks": []}

        toc: List[Tuple[int, str, int]] = []
        if path.suffix.lower() == ".pdf":
            try:
                doc = pymupdf.open(str(path))
                toc = doc.get_toc()
                doc.close()
            except Exception as e:
                logger.error(f"Ошибка чтения TOC: {e}")

        if not toc:
            # EPUB или PDF без метаданных оглавления - берём headings из базы.
            toc = self._get_toc_from_blocks()

        if not toc and path.suffix.lower() == ".pdf":
            return self._get_toc_via_ocr_fallback()

        if not toc:
            return {"answer": f"Оглавление книги \"{title}\" не найдено.",
                    "sources": [], "context_chunks": []}

        formatted_lines = []
        for level, toc_title, page in toc:
            indent = "  " * (level - 1)
            link = f" |page:{page}|" if page else ""
            formatted_lines.append(
                f"{indent}{toc_title.strip().rstrip('.')} . . . . . . {page}{link}"
            )

        combined_text = "\n".join(formatted_lines)
        return {
            "answer": f"Оглавление книги \"{title}\":\n\n{combined_text}",
            "pdf_path": str(path),
            "sources": [{"title": title, "page_number": "TOC",
                         "chapter": "Оглавление",
                         "file_format": path.suffix.lstrip(".")}],
            "context_chunks": [{"text": combined_text, "metadata": {
                "title": title, "page_number": "TOC", "chapter": "Оглавление"}}],
        }

    def _get_toc_via_ocr_fallback(self) -> Dict[str, Any]:
        pdf_path = self._get_current_pdf_path()
        if not pdf_path or not self.ocr_processor:
            return {"answer": "OCR недоступен.", "sources": [], "context_chunks": []}

        try:
            from ..processing.text_corrector import TextCorrector
            page_texts = self.ocr_processor.extract_text_from_pdf(
                pdf_path, pages=[3, 4, 5, 6, 7, 8]
            )
            combined_text = TextCorrector().correct_toc("\n\n".join(page_texts))

            return {
                "answer": f"Оглавление (распознано через OCR):\n\n{combined_text}",
                "pdf_path": str(pdf_path),
                "sources": [{"title": pdf_path.stem, "page_number": "4-9",
                             "chapter": "Оглавление (OCR)", "file_format": "pdf"}],
                "context_chunks": [{"text": combined_text,
                                    "metadata": {"page_number": "4-9",
                                                 "chapter": "Оглавление"}}],
            }
        except Exception as e:
            return {"answer": f"Ошибка OCR: {e}", "sources": [], "context_chunks": []}

    # ------------------------------------------------------------------
    # Гибридный поиск (FTS5 + векторы + RRF)
    # ------------------------------------------------------------------

    def _hybrid_search(self, question: str,
                       query_embedding: List[float],
                       k: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Гибридный поиск: объединяем FTS5 и векторный каналы через RRF.

        score(d) = sum_channels 1 / (k + rank_channel(d)), k = RRF_K = 60.
        Простая конкатенация списков раньше отсекала векторные результаты
        при малом final_top_k и допускала дубли.

        Args:
            k: переопределяет final_top_k (для summary-ветки нужен более
               широкий контекст — конспект по всей теме, а не 5 чанков).
        """
        fts_results = self._fts5_search(question, top_k=self.settings.search.fts_top_k)
        vector_results = self.chroma_store.search(
            query_embedding=query_embedding,
            top_k=self.settings.search.vector_top_k,
        )

        by_id: Dict[str, Dict[str, Any]] = {}
        rrf_scores: Dict[str, float] = defaultdict(float)

        # FTS обрабатываем первым: его метаданные богаче (title/file_format).
        for results in (fts_results, vector_results):
            for rank, item in enumerate(results, start=1):
                chunk_id = self._item_id(item)
                if not chunk_id:
                    continue
                rrf_scores[chunk_id] += 1.0 / (RRF_K + rank)

                existing = by_id.get(chunk_id)
                if existing is None:
                    item = dict(item)
                    # FIX ('str' object has no attribute 'items'): нормализуем
                    # metadata (JSON-строка из Chroma/SQLite -> dict) на входе
                    # в единый реестр чанков — дальше по конвейеру (merge,
                    # _extract_sources, UI) всегда лежит dict.
                    item["metadata"] = normalize_metadata(item.get("metadata"))
                    by_id[chunk_id] = item
                else:
                    # Дополняем недостающие поля метаданных (FTS уже учтён).
                    merged_meta = {**normalize_metadata(item.get("metadata")),
                                   **existing.get("metadata", {})}
                    by_id[chunk_id] = {**existing, "metadata": merged_meta}

        ranked = sorted(by_id.values(),
                        key=lambda it: rrf_scores[self._item_id(it)],
                        reverse=True)
        # FIX (логи 14:15-14:18): до усечения final_top_k прогоняем список
        # через кросс-эннер rerank (FlashRank, если установлен). RRF по
        # позициям двух каналов путает «похожие по теме» и «отвечают на
        # вопрос»: для точечного запроса («А1 из Теста 2») в топ-8 попадали
        # размытые векторные соседи, а нужный блок страницы с заданиями —
        # нет. Cross-encoder переставляет реально релевантные блоки в начало,
        # и последующее усечение по бюджету токенов отбрасывает уже мусор, а
        # не ответы. Пакет опционален: без него конвейер не меняется.
        final_k = k if k is not None else self.settings.search.final_top_k
        if len(ranked) > final_k:
            ranked = self._rerank_by_relevance(question, ranked)
        return ranked[:final_k]

    _RERANKER = None
    _RERANK_TRIED = False

    @classmethod
    def _get_reranker(cls):
        """Ленивая загрузка FlashRank (модель ms-marco-MiniLM-L6-v2, CPU,
        ~30 МБ, onnxruntime уже в зависимостях). При отсутствии пакета или
        любой ошибке загрузки — None (гибридный RRF-порядок как раньше)."""
        if cls._RERANK_TRIED:
            return cls._RERANKER
        cls._RERANK_TRIED = True
        try:
            from flashrank import Ranker
            # FIX 1: у flashrank.Ranker нет параметра cache_path — кэш задаётся
            # через cache_dir; старый вызов падал с TypeError и reranker молча
            # выключался ("используется RRF").
            # FIX 2: на HF-репозитории prithivida в ZIP доступны только
            # ms-marco-MiniLM-L-12-v2 и ms-marco-TinyBERT-L-2-v2 (L-6/L-4 -> 404),
            # поэтому перебираем существующие модели по убыванию качества.
            import os
            cache_dir = os.path.join(
                os.path.expanduser("~"), ".cache", "flashrank"
            )
            try:
                os.makedirs(cache_dir, exist_ok=True)
            except OSError:
                pass
            last_err = None
            for model in ("ms-marco-MiniLM-L-12-v2",
                          "ms-marco-TinyBERT-L-2-v2"):
                try:
                    try:
                        cls._RERANKER = Ranker(model_name=model,
                                               cache_dir=cache_dir)
                    except TypeError:
                        # Старый API без cache_dir — дефолтный кэш.
                        cls._RERANKER = Ranker(model_name=model)
                    logger.info(
                        f"RAG: FlashRank-reranker активен "
                        f"(модель {model}, дополнительная фильтрация контекста)")
                    break
                except Exception as ex:
                    last_err = ex
                    logger.debug(f"RAG: модель rerank '{model}' недоступна "
                                 f"({ex}) — пробую следующую")
            if cls._RERANKER is None and last_err is not None:
                raise last_err
        except Exception as ex:
            logger.debug(f"RAG: reranker недоступен ({ex}) — "
                         "используется чистый RRF-порядок")
            cls._RERANKER = None
        return cls._RERANKER

    def _rerank_by_relevance(self, question: str,
                             ranked: List[Dict[str, Any]]
                             ) -> List[Dict[str, Any]]:
        """Cross-encoder перегруппировка top-N кандидатов перед усечением.

        DEFENSIVE: любой сбой (пустые тексты, ошибка модели, таймаут) —
        молча возвращаем исходный порядок; rerank — улучшение, не зависимость.
        """
        ranker = self._get_reranker()
        if ranker is None or not question:
            return ranked
        try:
            cand = ranked[:max(12, len(ranked))]
            passages = []
            for i, it in enumerate(cand):
                meta = it.get("metadata") or {}
                title = str(meta.get("title") or "") if isinstance(meta, dict) else ""
                text = str(it.get("text") or it.get("document") or "").strip()
                if not text:
                    continue
                passages.append({"id": i, "text": (title + ". " + text)[:1200],
                                 "meta": {},
                                 "title": "",
                                 "doc_id": title})
            if len(passages) < 2:
                return ranked
            # FIX (flashrank 0.2.10): публичного .rank() нет — API:
            #   rerank(RerankRequest(query, passages)) -> список passages
            #   с добавленным "score", отсортированный по убыванию.
            # Старый вызов rank(...) молча кидал AttributeError в except и
            # весь rerank-канал выключался ("используется чистый RRF").
            try:
                from flashrank import RerankRequest
                req = RerankRequest(query=question[:500], passages=passages)
                hits = ranker.rerank(req)
            except ImportError:
                # Очень старые версии (<0.2): был метод rank(query, passages).
                hits = ranker.rank(query=question[:500], passages=passages)
            out: List[Dict[str, Any]] = []
            seen = set()
            for h in hits or []:
                idx = int(h["id"])
                if idx not in seen:
                    seen.add(idx)
                    out.append(cand[idx])
            # кандидаты без текста / пропущенные ранкером — в хвост
            for i, it in enumerate(cand):
                if i not in seen:
                    out.append(it)
            tail = ranked[len(cand):]
            return out + tail
        except Exception as ex:
            logger.debug(f"RAG: rerank пропущен ({ex})")
            return ranked

    # ------------------------------------------------------------------
    # Цифровой двойник: склейка соседних чанков в логические блоки (ТЗ п.3)
    # ------------------------------------------------------------------

    _MERGE_BUDGET = 9000   # максимальный размер склеенного блока, символов

    @staticmethod
    def _page_span(metadata: Dict[str, Any]) -> Tuple[int, int]:
        """(start, end) страниц чанка; '216-220' → (216, 220)."""
        raw = str(metadata.get("page_number") or "").strip()
        m = re.match(r"(\d+)(?:\s*[-–]\s*(\d+))?", raw)
        if not m:
            return (0, 0)
        start = int(m.group(1))
        end = int(m.group(2)) if m.group(2) else start
        return (start, max(start, end))

    def _merge_adjacent_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Склеивает чанки, лежащие рядом в одном разделе, в связные блоки.

        Два чанка объединяются, если у них одинаковые doc_id/глава и номера
        страниц не расходятся больше чем на 2. У resulting-блока сохраняется
        rich-метаданные первого куска + parent_section (иерархия «глава →
        раздел»), чтобы промпт показывал модели структуру учебника. Порядок
        входной RRF-ранжировки соблюдается (блоки идут по убыванию релевантности).
        """
        if len(chunks) < 2:
            return chunks
        merged: List[Dict[str, Any]] = []
        used = [False] * len(chunks)
        for i, ch in enumerate(chunks):
            if used[i]:
                continue
            used[i] = True
            # FIX ('str' object has no attribute 'items'): metadata может быть
            # JSON-строкой — dict() от строки упал бы здесь, нормализуем тип.
            meta = normalize_metadata(ch.get("metadata"))
            text_parts = [ch.get("text", "")]
            size = len(text_parts[0])
            p_start, p_end = self._page_span(meta)
            for j in range(i + 1, len(chunks)):
                if used[j]:
                    continue
                other = chunks[j]
                ometa = normalize_metadata(other.get("metadata"))
                if (ometa.get("doc_id") or meta.get("doc_id")) != meta.get("doc_id"):
                    continue
                if (ometa.get("chapter") or "") != (meta.get("chapter") or ""):
                    continue
                os_, oe_ = self._page_span(ometa)
                if p_start and os_ and abs(os_ - p_end) > 2:
                    continue
                add_len = len(other.get("text", ""))
                if size + add_len > self._MERGE_BUDGET:
                    break
                used[j] = True
                text_parts.append(other.get("text", ""))
                size += add_len
                p_end = max(p_end, oe_)
            page_label = (f"{p_start}-{p_end}" if p_end > p_start
                          else str(p_start or meta.get("page_number") or ""))
            new_meta = dict(meta)
            new_meta["page_number"] = page_label
            new_meta["parent_section"] = (meta.get("section")
                                          or meta.get("chapter") or "")
            merged.append({"text": "\n\n".join(t for t in text_parts if t),
                           "metadata": new_meta})
        logger.debug(f"Склейка контекста: {len(chunks)} чанков → "
                     f"{len(merged)} логических блоков")
        return merged

    @staticmethod
    def _item_id(item: Dict[str, Any]) -> str:
        # FIX ('str' object has no attribute 'items'): metadata-JSON-строка.
        meta = normalize_metadata(item.get("metadata"))
        return meta.get("block_id") or item.get("id", "")

    def _fts5_search(self, question: str, top_k: int) -> List[Dict[str, Any]]:
        """Полнотекстовый поиск (запрос строит и экранирует SQLiteStore)."""
        results = self.sqlite_store.search_fts(question, limit=top_k)
        return [
            {
                "id": r["block_id"],
                "text": r["text"],
                "metadata": {
                    "doc_id": r["doc_id"],
                    "block_id": r["block_id"],
                    "title": r["title"],
                    "page_number": r["page_number"],
                    "chapter": r["chapter"],
                    "file_format": r["file_format"],
                },
                "score": 1.0,
            }
            for r in results
        ]

    # ------------------------------------------------------------------
    # Источники
    # ------------------------------------------------------------------

    def _extract_sources(self, answer: str,
                         context_chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        sources = []
        for match in re.findall(r'\[(\d+)\]', answer):
            index = int(match) - 1
            if 0 <= index < len(context_chunks):
                # FIX ('str' object has no attribute 'items'): metadata чанка
                # может быть JSON-строкой — нормализуем перед .get().
                meta = normalize_metadata(context_chunks[index].get("metadata"))
                raw_page = meta.get("page_number")
                # FIX (переход к странице из чата): приводим страницу к целому.
                # Могут приходить: None (старые импорты), "12", "12-18"
                # (склеенный блок), "" (sanitize_metadata Chroma для None).
                page_int = None
                if isinstance(raw_page, int):
                    page_int = raw_page if raw_page > 0 else None
                elif isinstance(raw_page, str):
                    m = re.search(r"\d+", raw_page)
                    if m:
                        page_int = int(m.group()) or None
                sources.append({
                    "title": meta.get("title", "Без названия"),
                    "page_number": page_int,
                    "page_label": str(raw_page or ""),
                    "chapter": meta.get("chapter"),
                    "file_format": meta.get("file_format"),
                    # REFACTORING (адаптивный layout): клик по источнику в чате
                    # должен открыть КОНКРЕТНУЮ книгу в просмотрщике — без doc_id
                    # UI вынужден был гаддать по «последнему открытому».
                    "doc_id": meta.get("doc_id") or context_chunks[index].get("doc_id"),
                })

        unique = []
        seen = set()
        for s in sources:
            key = (s["title"], s.get("page_number"), s.get("chapter"))
            if key not in seen:
                seen.add(key)
                unique.append(s)
        return unique
