# Конспект проекта lama-loca ( MinerU ) — состояние на 2026-10-30

Этот файл — итог длинной отладочной сессии. Передайте его в новый чат, чтобы продолжить без потери контекста.

## Что за проект
Локальная RAG-библиотека: PDF/DJVU учебники → MinerU (парсинг в Markdown+LaTeX) → чистка/чанкинг → эмбеддинги bge-m3 (Ollama) → ChromaDB → гибрид-поиск (вектор + BM25/токен + точные теги заданий) → FlashRank-reranker → генерация qwen3.5:4b (Ollama) → UI на Flet 1.0.5 (просмотрщик книг + чат).
Железо пользователя: Windows 11, Python 3.11 venv, RTX 5060 Laptop (8 ГБ VRAM), 23 ГБ RAM. Ollama держит обе модели целиком в VRAM.

## Ключевые компоненты и файлы
- `src/ingestion/mineru_parser.py` — парсинг MinerU: is_available (3 независимых признака + CLI рядом с sys.executable), _should_force_gpu (pynvml/nvidia-smi, порог 2560 МБ), _apply_vram_override (MINERU_VIRTUAL_VRAM_SIZE=16 при fast_batch), release_vram(Ollama) перед парсингом + _wait_for_release, _attach_page_numbers по middle.json.
- `src/llm/ollama_vram.py` — выгрузка/загрузка моделей Ollama через /api/ps + /api/generate keep_alive:0 (исправлен битый urllib GET).
- `src/processing/text_cleaner.py` — чистка текста. АГРЕССИВНАЯ конвертация inline-LaTeX→Unicode УДАЛЕНА из конвейера (был корнем бага «проза»); доступна только по флагу convert_inline_latex=True для plain-text экспорта.
- `src/llm/rag_pipeline.py` — answer(): детект намерения (toc/semantic/summary), _hybrid_search (RRF + опц. BM25 + точные теги «А1/Тест 2»), FlashRank-reranker (cache_dir!, rerank(RerankRequest), модель ms-marco-MiniLM-L-12-v2; L-6 нет на сервере — 404), _expand_with_graph (Graph-RAG), _merge_adjacent_chunks, префлайт-фильтр бюджета (_handle_semantic_request ~строка 450–500: budget = живому effective_prompt_budget движка), _chat_with_retry (каскад: повтор → половина чанков → без контекста), _anti_collage (дедуп цитатник-обломков).
- `src/llm/llm_engine.py` — detect_model_window (GGUF 262144 режется до prompt_budget+gen), estimate_tokens (tiktoken cl100k_base, fallback символы/2.7), probe_fit, _build_options (num_ctx/num_predict инвариант prompt+predict<=ctx), _trim_history/_fit цикл, _report_gen_speed (~50–75 ток/с = GPU подтверждён), АДАПТИВНЫЙ бюджет: старт _EFFECTIVE_PROMPT_TOKENS=12000, после первого GPU-ответа _maybe_raise_budget поднимает до _GPU_RAISED_PROMPT_TOKENS=20000 (ручной override config/env не перекрывается), геттер effective_prompt_budget().
- `src/ui/chat_view.py` — рендер ответов: normalize_raw_latex_in_answer (_MATH_RUN_RE — склейка math-runs в один $…$), _restore_bare_commands_global (потерянные бэкслеши у \frac/\cdot…), _brace_frac_args («frac 12» → «\frac{1}{2}»), fix_glued_quotes (« A » → «A»), формулы с командами рендерятся картинками mathtext/PNG (_formula_image), selectable=True для выделения всего текста.
- `config.yaml`: ollama.effective_prompt_tokens: 0 (0 = авто/адаптивно), unload_for_parse: true, final_top_k: 16, pdf_engine.fast_batch/vram_gb_override.

## История исправлений (коммиты, старые → новые)
fea6e50/a248ddb — LaTeX-рендер в чате + кавычки; a3cb8b8 — анти-пустой ответ (tiktoken, бюджет генерации, опц. rerank); 49d8c72/6c763bc — честная диагностика доступности MinerU; 57bb4f9/bc1a584/66bd5ba — динамическая выгрузка Ollama ради GPU-режима MinerU (+фикс GET /api/ps, ожидание реального освобождения VRAM); 741652f — vram_gb_override/fast_batch (Batch Ratio 8, 16 ГБ виртуальной VRAM); fabd1aa — чистка репозитория + актуальный requirements.txt; 9863970 — удаление LaTeX→Unicode из text_cleaner (баг «проза», двойная очистка, дубли); abc840e — FlashRank API 0.2.10 (cache_dir, rerank(), список моделей); ae5c5eb — синхронизация num_predict с историей (_fit), префлайт с историей; 63ecf36 — адаптивный бюджет 12k→20k после подтверждения GPU; 4bd30ec — ПРЕФЛАЙТ RAG теперь читает живой бюджет (effective_prompt_budget()), а не статичные 12k (чинит «ответ не раскрывает тему»: блоки §13 Формула Тейлора больше не отсекаются).

## Технические решения по спорным вопросам
- WebView/MathJax НЕЛЬЗЯ: в Flet 1.0.5 нет ft.WebView. Рендер формул — препроцессор + mathtext-картинки.
- Стек закреплён в requirements.txt: torch cu128 (2.11.0+cu128 работает; CUDA True на RTX 5060), pillow>=11,<13 (gradio 6.8 несовместим с 13), pynvml, tiktoken, flashrank (опционально, но установлен), mineru[core]>=2.5,<4 (3.4.5). Порядок установки: torch-cu128 → requirements → mineru → flashrank.
- MinerU: выбирать источник huggingface, тип pipeline. Прерывать импорт можно безопасно.
- Предупреждения albumentations/pydantic в логе — безвредны.
- Импорт книги после фикса text_cleaner надо ПОВТОРИТЬ (старые чанки содержат испорченный прозой текст). Накатывать архив поверх проекта (не заменять venv/data).

## Текущее состояние проверок
- py_compile всех модулей OK; тесты бюджета: 12000 до GPU-подтверждения / 20000 после — OK.
- Архив dist/lama-loca-latest.zip (282 КБ, 60 файлов) пересобран из HEAD=4bd30ec, все маркеры фиксов проверены внутри архива (ALL MARKERS PRESENT).

## Что осталось сделать / проверить (для нового чата)
1. Пользователь накатывает свежий архив, перезапускает, ПЕРЕИМПОРТИРУЕТ книги (Кудрявцев 714 стр., povt_matem_11 116 стр.).
2. Проверить в новом диалоге: первый вопрос (короткий ответ, GPU подтвердится ~70 ток/с) → второй вопрос по Тейлору: бюджет должен вырасти до 20k, INFO «бюджет контекста повышен 12000 → 20000», полные ответы без отсечения §13.
3. Если снова пустые ответы при num_ctx 12–16k despite eval_count=None done_reason='length' — проблема реальная ёмкость KV-кэша Ollama (VRAM 8ГБ минус bge-m3): варианты — снизить num_predict дефолт, держать qwen выгруженным во время чата? (нет — нужен в VRAM), или reduce keep_alive. Следующий шаг диагностики: замер `ollama ps` + free VRAM ДО запроса.
4. Косметика в чате: проверить рендер формул на реальных ответах (mathtext-картинки, кавычки, выделение текста).
5. Опционально: Graph-RAG даёт 0 рёбер на этих книгах (entity_extractor слаб на мат.текстах) — можно улучшить экстрактор или отключить _expand_with_graph.
6. Не трогали: BM25-канал реализован как LIKE-теги заданий; полноценный BM25 (rank_bm25) не внедрён.

## Известные ограничения
- qwen3.5:4b галлюцинирует « frac 12» без бэкслешей — лечится препроцессором в UI, не идеально.
- Префлайт-оценка токенов ≈ chat-шаблон (probe_fit), но ±10% возможны.
- Чат-история раздувает промпт: при длинных диалогах trim режет user-сообщение.
