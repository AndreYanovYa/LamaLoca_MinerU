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

## Сессия 2026-10-30 (после логов пользователя: CUDA OK, Ollama «без данных»)
Подтверждено железом/логами: torch 2.11.0+cu128 — CUDA на RTX 5060 Laptop (sm_120) работает; модели MinerU pipeline скачаны с huggingface (кэш PDF-Extract-Kit-1.0, mineru.json настроен). Осталось:

1. **Модель Ollama.** В config.yaml было `qwen3.5:4b` — алиаса нет в реестре Ollama (риск 404 и «пустых» ответов). Исправлено на `qwen3:4b`. Проверить: `ollama list`; при отсутствии — `ollama pull qwen3:4b` (~2.5 ГБ). Движок умеет подбирать tag по префиксу (_resolve_model), но точное имя надёжнее. bge-m3:latest — ок.
2. **keep_alive.** В config.yaml стоит `-1`; llm_engine сам кодирует его в `"1h"` (защита от HTTP 400 `missing unit in duration "-1"`). Рекомендация из setup_gpu.py формально верна — менять не нужно.
3. **Диагностика «? данных о размещении нет».** Причина: Ollama 0.40.x в /api/version не отдаёт флаг `cuda`, а /api/ps пуст в простое → скрипт не мог вынести вердикт. setup_gpu.py доработан: если cuda-флага нет и /api/ps пуст/без details, скрипт САМ прогревает чат-модель коротким /api/generate (num_predict=1, keep_alive=10m) и перечитывает /api/ps, печатая GPU/CPU/частичный-offload вердикт. Перезапустить `python scripts/setup_gpu.py` — секция 3 теперь даёт конкретный ответ.
4. **pdf_engine в config.yaml.** Секции не было (только в example) — добавлена: engine:auto, fast_batch:true, vram_gb_override:**8** (не 16: «виртуальные» 16 ГБ на реальной 8-ГБ карте провоцируют CUDA OOM у Unimernet-батча и тихий откат в CPU; 8 ГБ даёт стабильный Batch Ratio ~2–4). Если в логе импорта увидите OOM — снизьте до 6.
5. **MinerU CLI-флаг формул.** Python-API передаёт formula_enable=True; для CLI-пути добавлена передача `--enable_formula` (только если флаг присутствует в `mineru --help` — совместимость 2.5–3.x через динамическую проверку).
6. **Порядок VRAM (справка):** перед импортом книги release_vram выгружает qwen+bge (unload_for_parse:true), MinerU получает всю карту; после импорта модели перезагрузятся сами при первом вопросе/векторизации. При активном парсинге чат будет «медленно стартует» — это ожидаемо (повторная загрузка весов).
7. Все модули: py_compile OK. Старые чанки книг, импортированных до фикса text_cleaner, всё ещё требуют ПЕРЕИМПОРТА (см. пункт «Что осталось сделать»).

## Сессия 2026-10-30 (15:12): «немой» пустой ответ при полном окне — корневая починка thinking

### Симптом (лог)
`qwen3.5:4b` (на машине реально лежит кастомный GGUF `Qwen3.5-9B-Q4_K_M`) при
num_ctx≈20k возвращает `done_reason='length', eval_count=None, content=''`.
Каскады спасения (`_chat_with_retry`, авто-продолжение) лишь повторяли тот же
обречённый запрос — подсказка про `think=False` выводилась в лог, но никогда
не применялась автоматически.

### Корень
Семейство Qwen3/QwQ по умолчанию генерирует скрытые reasoning-токены в поле
`thinking`; они съевают весь бюджет `num_predict`, и в `content` не попадает
ни одного токена → Ollama отвечает exactly как в логах. Для кастомных имён
(`Qwen3.5-9B...`) автодетект по регулярке мог не сработать, а ручного ремонта
не происходило.

### Исправления (проверено py_compile)
1. **llm_engine.py**: рантайм-переключатель `_force_think_off` — при первом же
   «немом» length-обрыве (eval_count=None, пустой content) движок сам ставит
   think=False и возвращает sentinel `done_reason='retry_think_off'`; цикл
   `chat()` тут же повторяет запрос без потери попытки продолжения. Метрики
   читаются теперь из `data['metrics']` ИЛИ корня ответа (разные сборки).
2. **rag_pipeline.py** `_chat_with_retry`: перед слепым повтором добавлен шаг
   принудительного think=False (временная установка `_thinking_override`).
3. **config.yaml**: чат-модель переключена на `qwen3.5:9b` (реальный тег на
   машине; движок всё равно подберёт его по префиксу через `_resolve_model`).

### Вердикт по предложенной матрице решений
| Технология | Нужна ли здесь | Комментарий |
|---|---|---|
| Corrective RAG (CRAG) | Частично уже есть | Гибридный поиск + FlashRank-reranker + графовое расширение + preflight-защита релевантных блоков (_protect_relevant/_fit_block_to_budget). Полноценный CRAG-критик («оценка релевантности чанков моделью до генерации») избыточен на 8 ГБ VRAM: каждый лишний LLM-проход = +N секунд. |
| Late Chunking & KV-Fix | Уже внедрено | Инвариант prompt+num_predict+reserve ≤ num_ctx ≤ реальный window (detect_model_window + adaptive budget 12k→20k), keep_alive=-1→"1h", unload_for_parse. KV-сбои лечатся retry_think_off + reduced_cap каскадом. |
| GraphRAG | Не нужен как замена | Локальный entity_relations/REFERENCES уже реализован (Этапы 2–3); полная пересборка индекса в граф — высокая цена, низкая отдача для учебников с явной нумерацией §. |
| LangChain Map-Reduce / LlamaIndex CompactAndRefine | Реализовано нативно | `_map_reduce_answer()`: map-выжимки по блокам (≤6, ≤900 симв.) → synthesize по ~2–4k токенов. Включается автоматически при guard_est > 70% бюджета. Без внешней зависимости от LangChain. |
| LLMLingua | Не рекомендуется | Требует доп. модель (GPT-2/small llama) в RAM/VRAM поверх qwen+bge — конфликт с 8 ГБ RTX 5060. Роль экстрактивного сжатия выполняет map-проход (бесплатно, на той же модели). |

### Следующие шаги проверки
1. Перезапустить `python main.py`, задать вопрос про формулу Тейлора.
2. Ожидаемая цепочка в логе: `переведена в think=False` → `повтор запроса с
   отключённым thinking` → непустой ответ (~70 ток/с). Если снова пусто —
   сработает map→synthesize или summary-апгрейд.
3. Закрепить навсегда (опционально): `ollama.thinking_enabled: false` в config.

## Сессия 2026-10-30 (15:1x) — косметика ответа LLM (лог «Тейлор», оценка ИИ-кодера 10/10 по математике)
Баги рендеринга из сгенерированного ответа и их лечение (два слоя защиты):
1. **Постобработка** `src/processing/text_cleaner.py` → `sanitize_llm_answer()` (вызывается в
   `RAGPipeline._strip_code_fences`, т.е. на ВСЕХ ветках ответа до извлечения источников):
   - `fix_llm_math_artifacts` — снимает `\^`/`\_` ВНЕ `$$…$$`-блоков (`o((x-x_0)\^n)` → `o((x-x_0)^n)`),
     валидный LaTeX внутри блоков не трогает;
   - `fix_llm_bold_spacing` — чинит broken bold `** п -ю производную**` → `**п -ю производную**`
     (CommonMark требует непробельный символ у маркеров), пустые `** **` удаляются;
   - `drop_empty_list_items` — вырезает пустые маркеры `- ` / `• ` и висячее тире `-  — `
     (потерянная переменная), нормальные списки/таблицы/тире-абзацы целы.
   Все функции идемпотентны (тесты: двойной проход = без изменений).
2. **System prompt** `prompt_builder.py` — правило **9b**: слеши только как начало LaTeX-команд,
   жирный слитно, запрет пустых пунктов списка. Докстринг сделан raw-string (убран SyntaxWarning `\c`).
main.py/запуск не менялись — путь ответа единый через _strip_code_fences, пересборка индекса не нужна.
