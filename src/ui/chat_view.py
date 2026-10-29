# src/ui/chat_view.py
"""
Представление чата с ИИ (тёмная тема).

АУДИТ (исправлено):
1. RAGPipeline инжектируется из Application — чат работает с теми же
   SQLite/Chroma-соединениями, что импорт и библиотека.
2. Формулы: ответы ИИ содержат LaTeX ($...$, $$...$$). Раньше класс
   LatexRenderer был написан, но нигде не вызывался — формулы показывались
   «сырым» текстом. Теперь текст ответа разбивается на сегменты, и каждая
   формула рендерится matplotlib mathtext в PNG и вставляется в ленту чата.
3. Markdown: реализован лёгкий парсер (жирный, курсив, inline-код, списки),
   раньше ft.Markdown использовался без импорта (NameError) и был закомментирован.
4. Кнопки экспорта ответа в Markdown и PDF (AnswerExporter).
5. Убран битый диалог источников (self._page не существовал) — теперь клик по
   источнику переключает просмотрщик на страницу через колбэк on_open_source.
6. Ответы ИИ генерируются в фоновом потоке (run_thread) — UI не блокируется;
   добавлена кнопка «Очистить диалог» (сбрасывает память диалога RAG).
"""
import re
from typing import Any, Dict, List, Optional

import flet as ft
from loguru import logger

from src.core.settings import Settings
from src.llm.rag_pipeline import RAGPipeline
from src.processing.exporter import AnswerExporter
from src.processing.latex_renderer import LatexRenderer


# ---------------------------------------------------------------------------
# Лёгкий разметчик: LaTeX-формулы + Markdown (bold/italic/code/списки)
# ---------------------------------------------------------------------------

FORMULA_RE = re.compile(r"(\$\$.+?\$\$|\$.+?\$)", re.DOTALL)

# FIX (проблема 3 ТЗ — дублирование источников): страховка на уровне UI.
# Даже если мелкая LLM проигнорирует запрет в промпте и выведет в конце
# ответа раздел «### Источники» / «📚 Источники» со списком [N], он удаляется
# из текста перед рендерингом: интерфейс уже показывает источники кликабельными
# чипами под сообщением, и пользователь не должен видеть одно и то же дважды.
_SOURCES_SECTION_RE = re.compile(
    r"\n?[ \t]*(?:#{1,6}[ \t]*)?(?:\U0001f4da[ \t]*)?\*{0,2}"
    r"(?:Источники|Литература|References)[ \t]*\*{0,2}:?[ \t]*(?:\*\*|__)?[ \t]*\n"
    r"(?:[ \t]*(?:\[[0-9]{1,3}\][^\n]*(?:\n|$)|[-*+][ \t]+\[[0-9]{1,3}\][^\n]*(?:\n|$)))+",
    re.IGNORECASE)


# ---------------------------------------------------------------------------
# PREPROCESSOR (ТЗ п.2): сырой ответ ИИ НИКОГДА не передаётся в ft.Markdown
# напрямую — Flet вырезает латинские буквы и знаки, обёрнутые в $ ... $
# (получаются пустые места на экране). Инлайн-блоки $...$ принудительно
# заменяются стандартными текстовыми эквивалентами Юникода ДО передачи во
# Flet; очищенный текст выводится обычными текстовыми контролами.
# Потокобезопасно: чистые функции над строками, без общего состояния.
# ---------------------------------------------------------------------------

_INLINE_MATH_MAP = {
    r"\notin": "\u2209",   # ∉  (ПЕРЕД \in — иначе подстрока съест команду)
    r"\in": "\u2208",      # ∈
    r"\not\in": "\u2209",
    r"\forall": "\u2200",  # ∀
    r"\exists": "\u2203",  # ∃
    r"\nexists": "\u2204",
    r"\dots": "\u2026",    # …
    r"\ldots": "\u2026",
    r"\cdots": "\u2026",
    r"\emptyset": "\u2205",  # ∅
    r"\varnothing": "\u2205",
    r"\subseteq": "\u2286",
    r"\supseteq": "\u2287",
    r"\subset": "\u2282",
    r"\supset": "\u2283",
    r"\bigcup": "\u22c3",
    r"\bigcap": "\u22c2",
    r"\cup": "\u222a",
    r"\cap": "\u2229",
    r"\setminus": "\u2216",
    r"\land": "\u2227",
    r"\lor": "\u2228",
    r"\neg": "\u00ac",
    r"\implies": "\u21d2",
    r"\iff": "\u21d4",
    r"\to": "\u2192",
    r"\rightarrow": "\u2192",
    r"\leq": "\u2264",
    r"\geq": "\u2265",
    r"\neq": "\u2260",
    r"\approx": "\u2248",
    r"\sum": "\u2211",
    r"\prod": "\u220f",
    r"\int": "\u222b",
    r"\infty": "\u221e",
    r"\partial": "\u2202",
    r"\nabla": "\u2207",
    r"\cdot": "\u00b7",
    r"\times": "\u00d7",
    r"\div": "\u00f7",
    r"\pm": "\u00b1",
    r"\alpha": "\u03b1",
    r"\beta": "\u03b2",
    r"\gamma": "\u03b3",
    r"\delta": "\u03b4",
    r"\varepsilon": "\u03b5",
    r"\epsilon": "\u03f5",
    r"\zeta": "\u03b6",
    r"\eta": "\u03b7",
    r"\theta": "\u03b8",
    r"\iota": "\u03b9",
    r"\kappa": "\u03ba",
    r"\lambda": "\u03bb",
    r"\mu": "\u03bc",
    r"\nu": "\u03bd",
    r"\xi": "\u03be",
    r"\pi": "\u03c0",
    r"\rho": "\u03c1",
    r"\sigma": "\u03c3",
    r"\tau": "\u03c4",
    r"\upsilon": "\u03c5",
    r"\phi": "\u03c6",
    r"\varphi": "\u03d5",
    r"\chi": "\u03c7",
    r"\psi": "\u03c8",
    r"\omega": "\u03c9",
    r"\Gamma": "\u0393",
    r"\Delta": "\u0394",
    r"\Theta": "\u0398",
    r"\Lambda": "\u039b",
    r"\Xi": "\u039e",
    r"\Pi": "\u03a0",
    r"\Sigma": "\u03a3",
    r"\Phi": "\u03a6",
    r"\Psi": "\u03a8",
    r"\Omega": "\u03a9",
    r"\colon": ":",
    r"\langle": "\u27e8",   # ⟨
    r"\rangle": "\u27e9",   # ⟩
    r"\mathbb{R}": "\u211d",
    r"\mathbb{N}": "\u2115",
    r"\mathbb{Z}": "\u2124",
    r"\mathbb{Q}": "\u211a",
    r"\mathbb{C}": "\u2102",
}
# Команды сортируются по убыванию длины: \notin раньше \in, \subseteq раньше \subset.
_SORTED_MATH_KEYS = sorted(_INLINE_MATH_MAP, key=len, reverse=True)


def _latex_inline_to_unicode(latex: str) -> str:
    """LaTeX-тело инлайн-формулы → юникод-текст (\\exists → ∃, \\in → ∈…)."""
    s = latex.strip()
    for cmd in _SORTED_MATH_KEYS:
        if cmd in s:
            s = s.replace(cmd, " " + _INLINE_MATH_MAP[cmd] + " ")
    # FIX (T1): декоративные обёртки с аргументом — \mathbf{t}, \text{...} —
    # снимаются ВМЕСТЕ со скобками аргумента ДО общего удаления слэшей,
    # иначе от «\mathbf{t}» остаётся мусор «{t}»/«mathbft».
    for deco in ("mathbf", "boldsymbol", "mathrm", "mathit", "operatorname",
                 "bm", "text", "mbox"):
        s = re.sub(r"\\" + deco + r"\s*\{([^{}]*)\}", r"\1", s)
    # FIX (лог 12:46): декоративная обёртка без фигурного аргумента
    # («\mathbf t», «\mathrm dx») — снимаем команду, оставляя символ.
    s = re.sub(r"\\(?:mathbf|boldsymbol|mathrm|mathit|bm|operatorname)"
               r"(?=\s*\S)", "", s)
    # FIX (скриншот «angle A + angle B = 180^\circ»): команды с ОДНИМ
    # односимвольным аргументом без фигурных скобок (\angle A, \hat x,
    # \vec a, \bar x…) — захватываем аргумент и превращаем в канон
    # \angle{A}, иначе после снятия слэша получалось слово «angle».
    s = re.sub(r"\\(angle|hat|vec|bar|tilde|dot|breve)(?=\s*\S)",
               lambda m: "\\" + m.group(1) + "{", s)
    # аргументы, захваченные выше без закрывающей скобки, закрываем на месте:
    s = re.sub(r"\\(angle|hat|vec|bar|tilde|dot|breve)\{([A-Za-z0-9])"
               r"(?![A-Za-z0-9])", r"\\\1{\2}", s)
    s = s.replace("\\degree", "\u00b0")
    # остаточные backslash-команды (\hat{x}, \vec a…) — просто снимаем слэш,
    # чтобы Flet не вырезал символы: «\alpha» → «alpha», «\;» → пробел
    s = re.sub(r"\\[A-Za-z]+", lambda m: m.group(0)[1:], s)
    s = s.replace("\\", "")
    # одиночные латинские буквы/цифры в формуле ($A$, $x_1$, "$A, B, C$") —
    # обычный текст: убираем служебные скобки LaTeX и склеиваем
    s = re.sub(r"[{}]", "", s)
    s = s.replace("_", "\u2009")  # нижние индексы: «a_1» → «a₁» читаемо тонким пробелом
    s = re.sub(r"\s{2,}", " ", s).strip()
    return s


def clean_inline_math(text: str) -> str:
    """Преобразует инлайн-блоки $...$ в текстовые юникод-эквиваленты.

    ТЗ п.2: полностью исключает передачу сырых $...$ во Flet — латинские
    буквы и математические знаки больше не вырезаются. Display-блоки $$...$$
    не трогает (они уходят в отдельный PNG/MathML-путь рендера).

    FIX (скриншот: сырой «\\mathbf{t} = (x(t), y(t), z(t))» в ответе LLM):
    мелкие модели выдают LaTeX-команды прямо в прозе БЕЗ знаков $.
    Flet 1.x срезает backslash — пользователь видел голый код.
    Теперь голые декоративные/структурные команды сначала
    оборачиваются в инлайн $...$ и сворачиваются в читаемый
    юникод тем же конвертором.
    """
    if not text:
        return text
    # Сначала защищаем display-блоки $$...$$ от разрезания одинарными $.
    blocks: List[str] = []

    def _save_block(m):
        blocks.append(m.group(0))
        return "\x00BLK%d\x00" % (len(blocks) - 1)

    text = re.sub(r"\$\$(.+?)\$\$", _save_block, text, flags=re.DOTALL)
    # FIX (лог 12:41–12:46 + скриншот «\mathbf{t}…»): голые LaTeX-команды
    # (в т.ч. с аргументом {..}) в прозе без знаков $ → инлайн-формулы,
    # которые сворачиваются юникод-конвертером ниже. Чистая проза (без
    # обратных слешей) сюда не попадает.
    text = re.sub(r"(?<!\\)\\(?:[A-Za-z]+(?:\s*\{[^{}]*\})?|[^A-Za-z\s])",
                  lambda m: "$" + m.group(0) + "$", text)
    # FIX: после оборачивания голых команд получаются соседние блоки
    # «$\exists$ $\delta$» — склеиваем их в один, иначе конвертер режет
    # каждый кусок отдельно и теряются составные записи.
    text = re.sub(r"\$\s+\$", " ", text)
    # Инлайн $...$ → unicode-эквиваленты с обрамляющими пробелами
    text = re.sub(r"(?<!\\)\$(?!\$)([^$\n]+?)\$",
                  lambda m: " " + _latex_inline_to_unicode(m.group(1)) + " ",
                  text)
    # Экранированные \$ — просто доллары
    text = text.replace("\\$", "$")
    # Возвращаем display-блоки нетронутыми
    text = re.sub(r"\x00BLK(\d+)\x00", lambda m: blocks[int(m.group(1))], text)
    return re.sub(r"[ \t]{2,}", " ", text)


def strip_duplicate_sources(text: str) -> str:
    """Убирает финальный раздел «Источники» со списком [N] из ответа LLM."""
    if not text or "сточник" not in text.lower():
        return text
    cleaned = _SOURCES_SECTION_RE.sub("\n", text)
    return cleaned.rstrip() if cleaned != text else text


# ---------------------------------------------------------------------------
# FIX (баг 2 скриншотов): сырые LaTeX-команды в прозе («\frac f^{(n+1)}(x0)...
# \quad x \to$ x_0»). Причины две, обе лечатся здесь:
#   а) модель ставит инлайн-математику БЕЗ ограничителей $...$ — ft.Markdown
#      показывает бэкслэши буквально;
#   б) при обрыве генерации по лимиту length и авто-продолжении ограничитель
#      '$' теряется на стыке частей («...= O((x-x_0)^{n+1}), \quad x \to$
#      x_0») — KaTeX видит нечётный маркер, паникует (throwOnError) и вся
#      строка рендерится сырым текстом.
# Нормализатор проходит построчно (блоки $$...$$ не трогаем), для строк с
# «голыми» командами LaTeX: чиним нечётные одиночные $ (закрываем на месте
# следующего логичного разрыва либо снимаем лишние), затем заворачиваем
# оставшиеся голые команды в $...$. Идемпотентен.
# ---------------------------------------------------------------------------

_LATEX_CMD_RUN_RE = re.compile(
    r"(?<!\\)\\(?:frac|dfrac|sum|prod|int|lim|to|infty|quad|qquad|left|right|"
    r"cdot|ldots|dots|mathbf|mathrm|mathbb|mathcal|partial|varepsilon|"
    r"epsilon|delta|alpha|beta|gamma|sigma|theta|phi|psi|omega|lambda|mu|"
    r"forall|exists|in|subseteq|subset|cup|cap|leq|geq|neq|approx|equiv|"
    # FIX (скриншот 2): angle/hat/vec/bar без $ — модель пишет «angle A +
    # angle B = 180^\circ» прямо в прозе; без этих токенов строка уходила
    # в ft.Markdown сырым LaTeX-кодом.
    r"angle|hat|vec|bar|tilde|breve|degree|circ|"
    r"pm|mp|deg|log|ln|sin|cos|tan|arctg|tg|ctg|operatorname)\b"
)

# FIX (скриншот 2): непрерывный «math-run» — участок строки, начинающийся с
# LaTeX-команды и продолжающийся математическими символами/короткими токенами
# («\frac 1 2 \cdot CH \cdot AB», «x \to$ x_0»). Оборачивается ЦЕЛИКОМ в
# $...$, чтобы ни один фрагмент формулы не остался сырым текстом.
_MATH_SEED_CMD = (r"frac|dfrac|tfrac|sum|prod|int|iint|iiint|lim|to|infty|"
                  r"quad|qquad|cdot|times|div|pm|mp|leq|geq|neq|approx|"
                  r"equiv|subset|subseteq|supset|cup|cap|forall|exists|"
                  r"angle|hat|vec|bar|tilde|partial|circ|degree|ldots|"
                  r"left|right|mathbf|mathrm|mathbb|mathcal|operatorname")
# FIX (скриншот 2 — «angle A + angle B = 180^\circ»): модель пишет мат.
# выражения и БЕЗ слэшей у команд («angle A»). Run начинается с slash- или
# bare-командного токена и продолжается математическими символами/короткими
# токенами; полноценная проза не матчится никогда.
_BARE_CMD_WORDS = (r"angle|hat|vec|bar|tilde|breve|degree|circ|frac|"
                   r"dfrac|sum|prod|int|lim|infty|quad|qquad|cdot|"
                   r"times|partial|forall|exists|ldots|mathbf|"
                   r"mathrm|mathbb|leq|geq|neq|approx")
# FIX (скриншот 2): «star» в хвосте формулы («...180^\circ \star») — без
# этого токена run-регулярка не захватывала конец и хвост уходил сырым.
_BARE_CMD_WORDS += r"|star|ast"
_MATH_RUN_RE = re.compile(
    r"(?<![\w\\])(?:\\(?:" + _MATH_SEED_CMD + r"|" + _BARE_CMD_WORDS + r")\b"
    r"|(?<![A-Za-z])(?:" + _BARE_CMD_WORDS + r")(?=\s*[A-Za-z0-9({]|\s*$)"
    r")(?:[\s^_{}()\[\]0-9A-Za-z+\-*/=<>.,;:!|\\]|\\[A-Za-z]+)*"
    r"(?<![\s.,;:!?])"
)
_MATH_SEED_RE = re.compile(r"\\[A-Za-z]+")

# ---------------------------------------------------------------------------
# FIX (скриншот 1, баг склейки): Qwen обкусывает обратный слэш у команд и
# склеивает их со словами: «x = fracyp\%» (было «\frac{y}{p}\%»). Такой
# токен не матчится ни одним run-регэкспом (нет слэша перед frac), строка
# целиком уходила в ft.Markdown сырым кодом. Детектор ниже находит
# «призрачные» команды по математическому окружению и восстанавливает слэш.
# ВАЖНО: только структурные/декоративные команды — однобуквенные/словесные
# токены («in», «to», «int», «hat», «bar») пропуском, иначе они калечат
# английский текст внутри формул («A in AB» → «A \in AB»).
_GHOST_CMDS = (r"frac|dfrac|tfrac|sqrt|text|mathrm|mathbf|mathbb|mathcal|"
               r"left|right|quad|qquad|cdot|cdots|ldots|"
               r"leq|geq|neq|equiv|approx|propto|"
               r"sum|prod|lim|partial|nabla|circ|degree|"
               r"notin|subseteq|supseteq|forall|exists|wedge|"
               r"lor|land|implies|odot|perp|star|ast")
_GHOST_CMD_RE = re.compile(
    r"(?<![A-Za-z\\])(?:" + _GHOST_CMDS + r")(?:[a-z0-9]{1,3})(?![A-Za-z])"
)


def _is_math_context(text: str, start: int, end: int) -> bool:
    """Окружение фрагмента похоже на математику, а не на обычное слово."""
    if start and text[start - 1].isalpha():
        return False
    if end < len(text) and text[end].isalpha():
        return False
    left = text[max(0, start - 24):start]
    right = text[end:end + 24]
    sig = r"[=+\-*/^_{}()\[\]\\<>0-9$]"
    # слева непосредственно предшествует math-знак (или '='/'(' и т.п.)
    if re.search(sig + r"\s*$", left):
        return True
    # справа сразу идёт math-знак или фигурная скобка аргумента
    if re.match(r"\s*(?:[{}]|" + sig + r")", right):
        return True
    return False


def repair_bare_latex_commands(text: str) -> str:
    """Восстанавливает потерянные бэкслеши у «призрачных» LaTeX-команд.

    Работает ТОЛЬКО по строкам, где уже есть настоящая команда (\\буква) или
    маркер $ — чистая проза никогда не затрагивается, ложные срабатывания
    (слово «text» в абзаце) исключены по построению. Идемпотентен.
    """
    if not text:
        return text
    out_lines = []
    for line in text.split("\n"):
        s = line
        if re.search(r"\\[A-Za-z]", s) or "$" in s:
            shift = 0
            for m in _GHOST_CMD_RE.finditer(s):
                st = m.start() + shift
                if s[st - 1] == "\\":       # уже команда
                    continue
                if not _is_math_context(s, m.start(), m.end()):
                    continue
                # вставка слэша ПЕРЕД именем команды; хвост-склейка
                # («fracyp» → «\fracyp») останется run-регулярке: она
                # захватит сегмент целиком и завернёт в $...$, а рендерер
                # разберёт \frac+аргументы.
                s = s[:st] + "\\" + s[st:]
                shift += 1
        out_lines.append(s)
    return "\n".join(out_lines)


def _fix_unbalanced_inline_dollars(line: str) -> str:
    """Нейтрализует НЕЧЁТНОЕ число одиночных $ в строке (обрывок блока).

    Если в конце строки остался незакрытый '$' (типично для склейки после
    done_reason='length') — закрываем его в конце строки; если лишний '$'
    стоит в середине перед продолжением прозы — снимаем его совсем, иначе
    KaTeX «склеит» математику с текстом и строка станет нечитаемой.
    """
    # ignore escaped \$
    tmp = line.replace("\\$", "")
    n = len(re.findall(r"\$", tmp))
    if n % 2 == 0:
        return line
    # нечётно: последний одиночный $ без пары — закрываем строку им, если он
    # в хвосте (обрывок формулы), иначе убираем (случайной «проглоченной»
    # пары при авто-склейке).
    last = tmp.rfind("$")
    tail_after = tmp[last + 1:].strip()
    if not tail_after or len(tail_after) <= 3:
        insert_at = line.rfind("$")
        return line[:insert_at + 1].rstrip(" ") + " $"
    remove_at = line.rfind("$")
    return line[:remove_at] + line[remove_at + 1:]


def fix_glued_quotes(text: str) -> str:
    """FIX (скриншот 1 — ложные цитаты): модель ставит кавычки-ёлочки с
    отбивкой (« A = CB / AB », «Ответ: 3 »). Markdown/Flet такие «полые»
    кавычки не ломает, но выглядит как битая цитата; приводим к канону
    «содержимое» без внутренних пробелов у самих кавычек."""
    if not text or "\u201c" not in text and "\u00ab" not in text:
        return text
    t = re.sub(r"([\u00ab\u201c\"])\s+", "\\1", text)
    t = re.sub(r"\s+([\u00bb\u201d])", "\\1", t)
    return t


def normalize_raw_latex_in_answer(text: str) -> str:
    """Гарантирует, что в ответе ИИ нет сырых LaTeX-команд вне $...$ и нет
    нечётных маркеров $ в строках (см. комментарий выше)."""
    if not text:
        return text
    out_lines: List[str] = []
    for line in text.split("\n"):
        s = line
        stripped = s.strip()
        is_block_math = (stripped.startswith("$$") and stripped.endswith("$$")
                         and len(stripped) > 4)
        if not is_block_math:
            s = _fix_unbalanced_inline_dollars(s)
            # FIX (скриншот 1 — «x = fracyp\%»): восстановить обкусанные
            # Qwen бэкслеши у команд ДО заворачивания math-runs в $...$.
            s = repair_bare_latex_commands(s)
            # FIX (скриншот 2 — «angle A», «S = \frac 1 2 \cdot CH»): модель
            # пишет мат. выражения в прозе БЕЗ $...$ и без слэшей у части
            # команд. Оборачиваем непрерывные math-сегменты целиком, а не по
            # одной команде — иначе между блоками оставалась голая математика
            # («1 2 · CH ·»), доедавшаяся юникод-конвертером.
            def _wrap_run(m: re.Match) -> str:
                before = s[:m.start()]
                dollars = before.count("$") - before.count("\\$") * 2
                if dollars % 2 == 1:
                    return m.group(0)   # уже внутри формулы
                run = m.group(0).strip()
                # если хвост run утёк в полноценную прозу (≥3 слов из 4+ букв)
                # — это предложение после формулы; режем run до предпоследнего
                # пробела перед первым таким словом.
                words = re.findall(r"[A-Za-zа-яА-Я]{4,}", run)
                if len(words) >= 2:
                    mm = re.search(r"\s+[A-Za-zа-яА-Я]{4,}", run)
                    if mm:
                        run = run[:mm.start()].rstrip()
                        if not _MATH_SEED_RE.search(run):
                            return m.group(0)
                return "$" + run + "$"
            s = _MATH_RUN_RE.sub(_wrap_run, s)
            # Прежний точечный путь: одиночные голые команды вне runs.
            def _wrap(m: re.Match) -> str:
                start = m.start()
                before = s[:start]
                dollars = before.count("$") - before.count("\\$") * 2
                inside = dollars % 2 == 1
                return m.group(0) if inside else "$" + m.group(0) + "$"
            s = _LATEX_CMD_RUN_RE.sub(_wrap, s)
        out_lines.append(s)
    return "\n".join(out_lines)


def split_latex(text: str) -> List[Dict[str, str]]:
    """Разбивает текст на сегменты: {'type': 'text'|'formula', 'content': ...}."""
    segments = []
    for part in FORMULA_RE.split(text):
        if not part:
            continue
        if part.startswith("$$") and part.endswith("$$") and len(part) > 4:
            segments.append({"type": "formula", "content": part[2:-2].strip()})
        elif part.startswith("$") and part.endswith("$") and len(part) > 2:
            segments.append({"type": "formula", "content": part[1:-1].strip()})
        else:
            segments.append({"type": "text", "content": part})
    return segments


# Ссылки вида |page:N| в ответе ИИ (оглавление/главы) — для кликабельности.
PAGE_LINK_RE = re.compile(r"\|page:(\d+)\|")
# REFACTORING (ТЗ п.4): альтернативный формат интерактивной ссылки, который
# просит системный промпт — markdown-ссылка на страницу: [Определение 2.1](page:15).
MD_PAGE_LINK_RE = re.compile(r"\[([^\]\n]{1,80}?)\]\(page:\s*(\d+)\)")


def normalize_page_links(text: str) -> str:
    """Приводит [Название](page:N) к внутреннему канону |page:N| Назование.

    Единая точка нормализации: и ответы ИИ, и оглавления проходят через неё,
    поэтому любой из двух форматов становится кликабельным чипом.
    """
    if not text:
        return text
    return MD_PAGE_LINK_RE.sub(lambda m: f"|page:{m.group(2)}| {m.group(1)}",
                               text)

# Блоки $$...$$ рендерятся отдельной строкой целиком — markdown их трогать
# не должен.
# FIX (проблема 1 ТЗ): [^$] вместо .+? — блок не может «перепрыгнуть» через
# соседнюю инлайн-формулу ($$...$$ сразу рядом с $...$ раньше разрезался
# пополам и вся строка protection разъезжалась).
_BLOCK_FORMULA_RE = re.compile(r"\$\$([^$\n]+?)\$\$")
# Инлайн-формулы $...$ (одинарные) — оставляем как есть, они уходят в PNG.
# FIX (проблема 1 ТЗ — «пустые скобки ()»): хвостовой lookahead (?!\$)
# запрещал закрывающему $ стоять перед другим $ (например «($\exists$)» в
# скученных списках, «$A$, $B$»), а парный lookahead (?<!\\) при отсутствии
# экранированных долларов не давал матчиться вовсе — формула НЕ защищалась,
# доживала до ft.Markdown, где одиночные $ съедались как математические
# ограничители, и пользователь видел пустые скобки.
_INLINE_FORMULA_RE = re.compile(r"(?<!\\)\$(?!\$)([^$\n]+?)\$")
# Маркер защищённой инлайн-формулы (не конфликтует с текстом конспектов).
_FORMULA_TOKEN_RE = re.compile(r"\{\{FORMULA:(\d+)\}\}")

# FIX (проблема 1 ТЗ — пустые ячейки таблиц): мелкие LLM часто пишут символы
# «плоским» юникодом без $...$ (∃, ∀, ∑) — такой текст уходит в ft.Markdown
# как есть и теряется/не подсвечивается. Оборачиваем одиночные математические
# юникод-символы в LaTeX-формулы (одиночный символ внутри $...$ не содержит
# других $ — защита ниже гарантированно его подхватит).
_MATH_UNICODE_CHARS = (
    "\u2200\u2203\u2204\u2209\u2208\u220a\u220b"          # ∀ ∃ ∄ ∉ ∈ ∊ ∋
    "\u2227\u2228\u2229\u222a\u2216\u2205\u2201"          # ∧ ∨ ∩ ∪ ∖ ∅ ∁
    "\u2192\u21d2\u2194\u21d4\u21cc\u2261\u2248\u221d"    # → ⇒ ↔ ⇔ ⇌ ≡ ≈ ∝
    "\u2264\u2265\u2260\u224a\u2aa7\u2a7e"                # ≤ ≥ ≠ ≈⊆(var) ⩤ ⩾(var)
    "\u2282\u2286\u2283\u2287\u2288\u2289"                # ⊂ ⊆ ⊃ ⊇ ⊈ ⊉
    "\u2211\u220f\u222b\u222c\u222d\u221e\u2202\u2207"    # ∑ ∏ ∫ ∬ ∭ ∞ ∂ ∇
    "\u221a\u2223\u2225\u22a5\u2208"                      # √ | ‖ ⊥ ∈
)
_BARE_MATH_UNICODE_RE = re.compile(
    r"(?<![A-Za-z0-9$%])(" + _MATH_UNICODE_CHARS + r")(?![A-Za-z0-9$%])")


def wrap_bare_math_unicode(text: str) -> str:
    """Одиночные математические юникод-символы вне $...$ оборачивает в $...$.

    Применяется только к строкам, в НЕТ ни одной LaTeX-формулы: там символы
    заведомо плоский вывод модели (ячейки таблицы «Символ», подписи), а не
    часть формулы. Внутри $...$ ничего не меняется.
    """
    if not text or "$" in text:
        return text
    text = _BARE_MATH_UNICODE_RE.sub(r"$\1$", text)
    # FIX (скриншот 2): одиночные греческие буквы (σ, ε, δ…) в прозе — тоже
    # плоский вывод модели; оборачиваем их, чтобы инлайн-путь их не потерял.
    text = re.sub(r"(?<![A-Za-z0-9$%\u0370-\u03ff])"
                  r"([\u03b1-\u03c9\u0391-\u03a9])(?![A-Za-z0-9$%])",
                  r"$\1$", text)
    return text


def _normalize_formula_tokens(text: str) -> str:
    """FIX (проблема 1 ТЗ): мелкие LLM вставляют пробелы внутрь маркеров —
    «{{ FORMULA:0 }}», «{{Formula: 1}}» — и регулярка замены их не находит,
    пользователь видел сырые плейсхолдеры. Приводим все вариации к канону."""
    return re.sub(r"\{\s*FORMULA\s*:\s*(\d+)\s*\}", r"{{FORMULA:\1}}",
                  text, flags=re.IGNORECASE)


def _protect_formulas(text: str, store: List[str]) -> str:
    """Заменяет формулы маркерами, чтобы ft.Markdown не съедал $...$:
    в Flet 1.x одиночные $ считаются ограничителями математических блоков и
    без защиты формулы исчезают из текста.

    FIX (проблема 1 ТЗ): блоки $$...$$ защищаются ПЕРВЫМИ — иначе инлайн-
    регулярка разрезала блок по внутреннему $ и портила всю строку.
    """
    def repl(m):
        store.append(m.group(1))
        return "{{FORMULA:%d}}" % (len(store) - 1)
    text = _BLOCK_FORMULA_RE.sub(repl, text)
    return _INLINE_FORMULA_RE.sub(repl, text)


# FIX (проблема 1 ТЗ): «голая» LaTeX-команда (\exists, \forall, \sum…) или
# цепочка математических юникод-символов без знаков $ — короткий сегмент,
# который модель вывела «плоско». Такие строки раньше уходили в ft.Markdown
# как есть: ft.Markdown не знает LaTeX, backslash-команды терялись, и вместо
# символа пользователь видел пустые скобки ().
_BARE_LATEX_CMD_RE = re.compile(r"\\[A-Za-z]+")
_MATH_UNICODE_ALL = _MATH_UNICODE_CHARS + "".join(
    "\u03b1\u03b2\u03b3\u03b4\u03b5\u03b6\u03b7\u03b8\u03ba\u03bb\u03bc\u03bd"
    "\u03be\u03c0\u03c1\u03c3\u03c4\u03c5\u03c6\u03c7\u03c8\u03c9"
    "\u0393\u0394\u0398\u039b\u039e\u03a0\u03a3\u03a6\u03a8\u03a9"
    "\u211d\u2115\u2124\u211a\u2102\u03f5\u03d5\u03d6"
    "\u226a\u226b\u22a4\u22a2\u22a8\u2225\u2223"
)
_MATH_UNICODE_RE = re.compile("[" + _MATH_UNICODE_ALL + "]")


def looks_like_bare_formula(seg: str) -> bool:
    """Короткий сегмент выглядит как «плоская» формула без $...$?"""
    s = (seg or "").strip()
    if not s or len(s) > 48 or "\n" in s:
        return False
    if "$" in s or "|" in s or "`" in s:
        return False
    if _BARE_LATEX_CMD_RE.search(s):
        return True
    if _MATH_UNICODE_RE.search(s):
        # только плотная математика («∀x ∃y»), а не текст с одной скобкой-стрелкой
        mathy = sum(1 for ch in s if _MATH_UNICODE_RE.match(ch)
                    or ch in "≤≥≠≈∞→↔[](){}^_")
        return mathy >= max(1, len(s) // 2)
    return False


def _render_bare_formula_segment(seg: str, render_inline):
    """Сегмент-«плоская» формула → контрол (через тот же inline-рендерер)."""
    wrapped = seg.strip()
    if _BARE_LATEX_CMD_RE.search(wrapped):
        # матhtext/LaTeX понимает только команды — переводим юникод в них
        try:
            from src.processing.latex_renderer import LatexRenderer
            wrapped = LatexRenderer.normalize_latex(wrapped)
        except Exception:
            pass
    return render_inline("$" + wrapped + "$")


def _md_block(text: str, extra_inline: bool = True) -> ft.Control:
    """Безопасный блок текста/Markdown с ПРЕПРОЦЕССОРАМИ (ТЗ п.2).

    REFACTORING (ТЗ п.1): если после защиты остались маркеры {{FORMULA:N}},
    для которых нет LaTeX (мелкая LLM скопировала служебные токены из примера
    промпта), они вырезаются — пользователь никогда не увидит сырой
    плейсхолдер. Битые [текст](page:N) ссылки нормализуются в канон |page:N|.

    REFACTORING (ТЗ п.2): инлайн-формулы $...$ принудительно заменяются
    юникод-эквивалентами (clean_inline_math) ДО передачи во Flet — латинские
    буквы и кванторы больше не вырезаются в пустые места. Display-блоки
    $$...$$ сохраняются для PNG/MathML-пути (_protect_formulas).
    """
    formulas: List[str] = []
    text = normalize_page_links(_normalize_formula_tokens(text))
    if extra_inline:
        # ТЗ п.2: сначала «голые» unicode-символы приводятся к канону $...$,
        # затем ВСЕ инлайн-блоки переводятся в текстовые юникод-эквиваленты.
        text = clean_inline_math(wrap_bare_math_unicode(text))
    src_text = _protect_formulas(text, formulas) if extra_inline else text
    # Маркеры без известного LaTeX → удаляем (плейсхолдеры не должны
    # попадать на экран; канон {{FORMULA:N}} уже приведён нормализатором).
    src_text = re.sub(r"\s*\{\{FORMULA:\d+\}\}", "", src_text)
    md = ft.Markdown(src_text, selectable=True,
                     extension_set=ft.MarkdownExtensionSet.GITHUB_FLAVORED)
    md._formula_store = formulas  # снимок для последующего замещения
    return md


def _table_cell_content(cell: str, render_inline):
    """Содержимое ячейки markdown-таблицы (ТЗ п.3 — строгий табличный UI).

    REFACTORING (ТЗ п.2/п.3): каждая ячейка наполняется ЭЛЕМЕНТАМИ ft.Text с
    применением очистки инлайнов clean_inline_math — никаких сырых $...$ в
    контролах, никаких пустых ячеек и чёрных прямоугольников. Если очистка
    не оставила текста, ячейка печатает читаемый прочерк «—».

    Display-формулы $$...$$ внутри ячейки передаются компилятору строк
    (PNG/MathML-путь), текстовые сегменты — обычным ft.Text.
    """
    try:
        c = (cell or "").strip()
        if not c:
            return ft.Text("—", size=13, color="#9E9E9E")
        # Табличные ячейки — короткие: переводим ВСЕ формулы в unicode-текст
        cleaned = clean_inline_math(wrap_bare_math_unicode(c))
        cleaned = re.sub(r"\$\$(.+?)\$\$", lambda m: " " + m.group(1).strip() + " ",
                         cleaned, flags=re.DOTALL)  # display-тело без $ во Text
        cleaned = cleaned.strip()
        if not cleaned:
            return ft.Text("—", size=13, color="#9E9E9E")
        ctrl = ft.Markdown(cleaned, selectable=True,
                           extension_set=ft.MarkdownExtensionSet.GITHUB_FLAVORED)
        # Фолбэк против «чёрного прямоугольника»: если по какой-то причине
        # markdown получился пустым — возвращаем plain Text.
        if not cleaned.replace("#", "").replace("*", "").strip():
            return ft.Text(cleaned, size=13)
        return ctrl
    except Exception as ex:  # pragma: no cover - защитный рендер
        logger.debug(f"Ячейка таблицы не скомпилирована: {ex}")
        return ft.Text((cell or "").strip() or "—", size=13)


def _parse_markdown_table(table_lines: List[str]):
    """СТРОГИЙ парсер markdown-таблицы (ТЗ п.3).

    Извлекает заголовки колонок и строки данных программно; строка-
    разделитель |---|:--:| отбрасывается, строки с другим числом колонок
    выравниваются (недостающие дописываются пустыми, лишние отсекаются).
    Возвращает (headers: List[str], rows: List[List[str]]) или None.
    """
    parsed: List[List[str]] = []
    for ln in table_lines:
        s = ln.strip().strip("|").strip()
        if not s:
            continue
        cells = [c.strip() for c in s.split("|")]
        if all(set(c) <= set("-: ") and c for c in cells):
            continue  # разделитель |---|---|
        parsed.append(cells)
    if not parsed:
        return None
    ncols = max(len(r) for r in parsed)
    norm = [(r + [""] * (ncols - len(r)))[:ncols] for r in parsed]
    return norm[0], norm[1:]


def _build_table_block(table_lines: List[str], render_inline) -> Optional[ft.Control]:
    """Собирает markdown-таблицу (список строк «| a | b |») в ft.DataTable.

    REFACTORING (ТЗ п.3): заголовки и строки извлекаются СТРОГИМ парсером
    _parse_markdown_table; каждая ячейка — ft.DataCell c элементом ft.Text /
    ft.Markdown, наполненным через очистку инлайнов (Шаг 2). Пустые ячейки
    и дублирование таблицы исключены: при любом провале сборки возвращается
    один фолбэк-блок, никогда не две таблицы.
    """
    parsed = _parse_markdown_table(table_lines)
    if not parsed:
        return None
    header_cells, body_rows = parsed
    ncols = len(header_cells)
    try:
        return ft.DataTable(
            columns=[ft.DataColumn(ft.Text(h or "—", size=13,
                                           weight=ft.FontWeight.BOLD))
                     for h in header_cells],
            rows=[
                ft.DataRow(cells=[
                    ft.DataCell(_table_cell_content(c, render_inline))
                    for c in row[:ncols]
                ])
                for row in body_rows
            ],
            column_spacing=14,
            heading_row_height=40,
            data_row_max_height=88,
            show_bottom_border=True,
            border=ft.Border.all(1, "#3A3A3A"),
        )
    except Exception as ex:
        logger.debug(f"DataTable не собралась ({ex}) — фолбэк markdown-блоком")
        return _md_block("\n".join(x.strip() for x in table_lines))


def render_markdown_line(line: str) -> ft.Control:
    """Мини-Markdown строки ответа.

    АУДИТ (Flet 1.x): в Flet 1.x контрола ft.RichText не существует —
    приложение падало на импорте с AttributeError: module 'flet' has no
    attribute 'RichText'. Вместо самописного span-парсера используется
    нативный ft.Markdown (bold/italic/code/списки поддерживаются из коробки,
    как требует ТЗ).

    FIX (проблема 2 ТЗ): строки таблиц («| Тип | Обозначение |», «|---|---|»)
    раньше рендерились по одной строке — ft.Markdown не собирает из них
    таблицу, и пользователь видел сырую разметку. Для табличных строк
    теперь возвращается специальная заглушка (вызывающий код группирует
    подряд идущие табличные строки в один markdown-блок).

    REFACTORING (адаптивный layout): интерактивное оглавление — ссылки
    |page:N| становятся кликабельными чипами, ведущими в просмотрщик.
    ft.Markdown сам не обрабатывает такой синтаксис, поэтому строка
    разрезается на сегменты: текст → Markdown, ссылка → Chip(on_click).
    """
    stripped = line.strip()
    if stripped.startswith(("- ", "• ")):
        stripped = "- " + stripped[2:]

    if is_table_line(stripped):
        return ft.Container(key="__table_row__", visible=False)

    links = list(PAGE_LINK_RE.finditer(stripped))
    if not links:
        return _md_block(stripped)

    parts = []
    pos = 0
    for m in links:
        before = stripped[pos:m.start()].strip(" —·:")
        if before:
            parts.append(_md_block(before))
        page_no = int(m.group(1))
        parts.append(
            ft.Chip(
                label=ft.Text(f"стр. {page_no}", size=12),
                on_click=lambda e, n=page_no: ChatView._open_page_static(n),
                tooltip=f"Открыть страницу {page_no} в просмотрщике",
            )
        )
        pos = m.end()
    after = stripped[pos:].strip(" —·:")
    if after:
        parts.append(_md_block(after))

    # Если строка СОСТОИТ только из ссылок (например "|page:12|"), добавим
    # подпись, чтобы чип не висел без контекста.
    prefix = ([] if PAGE_LINK_RE.sub("", stripped).strip() else
              [ft.Text("Перейти:", size=12, color="grey")])
    return ft.Row([*prefix, *parts], wrap=True, spacing=4)


def is_table_line(line: str) -> bool:
    """Строка markdown-таблицы: '| ... |' (заголовок/данные/разделитель)."""
    s = line.strip()
    return s.startswith("|") and s.count("|") >= 2


def _group_table_lines(lines: List[str]) -> List[List[str]]:
    """Группирует последовательные строки ответа в блоки: таблицы целиком,
    остальные строки по одной. Возвращает список групп-списков строк."""
    groups: List[List[str]] = []
    for ln in lines:
        if is_table_line(ln):
            if groups and all(is_table_line(x) for x in groups[-1]):
                groups[-1].append(ln)
            else:
                groups.append([ln])
        else:
            if groups and not all(is_table_line(x) for x in groups[-1]):
                groups[-1].append(ln)
            else:
                groups.append([ln])
    return groups


class ChatView:
    # Последний активный чат — нужен модульной функции render_markdown_line,
    # чтобы клики по |page:N| шли в просмотрщик текущего чата.
    _instance: Optional["ChatView"] = None

    @classmethod
    def _open_page_static(cls, page_number: int):
        if cls._instance is not None:
            cls._instance._open_page_link(page_number)

    def __init__(self, page: ft.Page, settings: Settings, rag_pipeline: RAGPipeline,
                 on_open_source=None, on_show_toc=None):
        ChatView._instance = self
        self.page = page
        self.settings = settings
        self.rag_pipeline = rag_pipeline
        # REFACTORING (ТЗ п.1): цвет формул зависит от темы контейнера.
        # Чат — тёмный (#121212) → светло-серые глифы; библиотека/документ —
        # светлые подложки → тёмные глифы. LatexRenderer кэширует по (latex, color).
        self._dark_formulas = True

        # Колбэк перехода просмотрщика к источнику:
        # (page_number:int, doc_id:str|None) -> None
        self._on_open_source = on_open_source
        # REFACTORING (адаптивный layout): «обращается к главе через чат» →
        # панель документа должна раскрываться и при запросе оглавления.
        self._on_show_toc = on_show_toc

        # АУДИТ: settings.storage_root возвращает str — оператор "/" падал
        # с TypeError при открытии чата. Используем явное Path-свойство.
        storage_root = settings.storage_root_path
        self.exporter = AnswerExporter(output_dir=str(storage_root / "exports"))
        self.latex_renderer = LatexRenderer(output_dir=str(storage_root / "temp" / "latex"))

        self.chat_controls: List[ft.Control] = []
        self._last_result: Optional[Dict[str, Any]] = None
        self._pending_question: str = ""
        # ТЗ (поток чата): снимок последнего успешного ответа. Заполняется
        # ТОЛЬКО после успешного рендера сообщения — кнопки экспорта никогда
        # не ссылаются на несуществующий/пустой ответ.
        self.last_answer: str = ""
        self.last_question: str = ""
        self.last_sources: List[Dict[str, Any]] = []
        self._busy = False

        self.chat_column = ft.ListView(expand=True, spacing=10, auto_scroll=True)
        self.input_field = ft.TextField(
            hint_text="Задайте вопрос по библиотеке...",
            expand=True,
            multiline=True,
            min_lines=1,
            max_lines=5,
            shift_enter=True,
            on_submit=self._on_send_click,
        )
        # Flet 1.x: ElevatedButton удалён из API — используется FilledButton.
        self.send_btn = ft.FilledButton(
            icon=ft.Icons.SEND,
            content=ft.Text("Отправить"),
            on_click=self._on_send_click,
            bgcolor="#2196F3",
            color="white",
        )
        self.clear_btn = ft.IconButton(
            icon=ft.Icons.DELETE_SWEEP,
            tooltip="Очистить диалог",
            on_click=self._on_clear_click,
        )
        self.active_doc_text = ft.Text("", size=12, color=ft.Colors.GREY)

        self._add_system_message(
            # ТЗ п.3: новый текст стартового баннера строго по шаблону.
            "👋 Привет! Яlama Loca — помощник по твоим учебным дисциплинам. "
            "Загрузите книгу во вкладке Библиотека и задайте вопрос. "
            "Например: покажи оглавление или что такое равномерная сходимость?"
        )

    def build(self) -> ft.Control:
        return ft.Container(
            content=ft.Column(
                [
                    ft.Row(
                        [
                            ft.Text("💬 Чат с ИИ", size=20, weight=ft.FontWeight.BOLD),
                            self.active_doc_text,
                            ft.Container(expand=True),
                            self.clear_btn,
                        ]
                    ),
                    self.chat_column,
                    ft.Row([self.input_field, self.send_btn]),
                ],
                expand=True,
            ),
            padding=10,
            expand=True,
        )

    def set_active_document_label(self, title: Optional[str]):
        self.active_doc_text.value = f"📖 {title}" if title else "все книги"
        try:
            self.page.update()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Отправка и получение ответа
    # ------------------------------------------------------------------

    def _on_send_click(self, e):
        question = (self.input_field.value or "").strip()
        if not question or self._busy:
            return
        self.input_field.value = ""
        # FIX (экспорт): запоминаем вопрос — он относится к ответу, который
        # будет сгенерирован. В потоке _answer_thread после генерации снимок
        # last_question обновится каноническим значением.
        self._pending_question = question
        self._add_user_message(question)
        self._busy = True
        self.send_btn.disabled = True
        self._show_typing(True)
        try:
            self.page.update()
        except Exception:
            pass
        # Тяжёлый запрос к Ollama уходит в поток — UI остаётся отзывчивым.
        self.page.run_thread(self._answer_thread, question)

    def _answer_thread(self, question: str):
        """Поток ответа строго по ТЗ:
        индикатор → rag.answer() → проверка непустого → рендер сообщения →
        сохранение last_answer/last_question/last_sources → кнопки экспорта →
        скрытие индикатора → page.update()."""
        try:
            result = self.rag_pipeline.answer(question)
            self._show_typing(False)
            answer_text = strip_duplicate_sources(
                ((result or {}).get("answer") or "").strip())
            # FIX (баг 2 — сырые \frac/\quad/\mathbf в прозе): финальная
            # нормализация перед рендером — нечётные $ чинятся, голые
            # LaTeX-команды заворачиваются в $...$ (см. normalize_raw_latex).
            answer_text = normalize_raw_latex_in_answer(fix_glued_quotes(answer_text))
            # FIX (пустой пузырь с кнопками экспорта): если LLM вернула пустой
            # ответ, показываем понятную ошибку вместо пустого сообщения.
            if not answer_text:
                self._add_error_message(
                    "❌ Модель не сгенерировала ответ (пустой вывод). "
                    "Проверьте имя chat_model в config.yaml и что Ollama "
                    "доступна: ollama list"
                )
                return
            # Сначала рендерим сообщение ИИ (LaTeX → PNG, markdown-строки);
            # export-кнопки добавляются внутри и получают снимок этого ответа.
            self._add_assistant_message(answer_text, result.get("sources", []))
            # Только после успешного рендера сохраняем состояние для экспорта.
            self._last_result = result
            self.last_answer = answer_text
            self.last_question = question
            self.last_sources = list(result.get("sources", []))
            # REFACTORING (адаптивный layout): запрос оглавления/главы — это
            # «обращение к контексту документа из чата»: панель просмотра
            # раскрывается динамически (если документ уже открыт).
            if result.get("intent") in ("toc", "chapter") and self._on_show_toc:
                self._on_show_toc()
        except Exception as ex:
            # FIX (диагностика): logger.exception печатает полный traceback —
            # без него было непонятно, в какой именно строке рендера падает
            # ошибка (например, «Container.__init__() got an unexpected
            # keyword argument 'max_height'»).
            logger.exception(f"Ошибка обработки вопроса: {ex}")
            self._show_typing(False)
            self._add_error_message(f"❌ Ошибка: {ex}")
        finally:
            self._busy = False
            self.send_btn.disabled = False
            try:
                self.page.update()
            except Exception:
                pass

    def _on_clear_click(self, e):
        self.rag_pipeline.reset_history()
        self.chat_column.controls.clear()
        self._last_result = None
        self.last_answer = ""
        self.last_question = ""
        self.last_sources = []
        self._add_system_message("Диалог очищен.")
        try:
            self.page.update()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Отрисовка сообщений
    # ------------------------------------------------------------------

    def _typing_control(self) -> ft.Control:
        return ft.Container(
            key="typing_indicator",
            content=ft.Row(
                [ft.Text("ИИ печатает", color=ft.Colors.GREY),
                 ft.ProgressRing(width=16, height=16, stroke_width=2)],
            ),
            visible=False,
            padding=8,
        )

    def _show_typing(self, show: bool):
        ctrl = None
        for c in self.chat_column.controls:
            if isinstance(c, ft.Container) and getattr(c, "key", None) == "typing_indicator":
                ctrl = c
                break
        if ctrl is None and show:
            ctrl = self._typing_control()
            self.chat_column.controls.append(ctrl)
        if ctrl is not None:
            ctrl.visible = show
            if not show:
                self.chat_column.controls.remove(ctrl)

    def _add_user_message(self, text: str):
        self.chat_column.controls.append(
            ft.Container(
                content=ft.Text(text),
                bgcolor=ft.Colors.BLUE_GREY_900,
                border_radius=10,
                padding=12,
                margin=ft.Margin.only(left=80),
                alignment=ft.Alignment.CENTER_RIGHT,
            )
        )
        try:
            self.page.update()
        except Exception:
            pass

    def _add_system_message(self, text: str):
        self.chat_column.controls.append(
            ft.Container(
                content=render_markdown_line(text.replace("\n", " ")),
                padding=8,
                alignment=ft.Alignment.CENTER,
            )
        )

    def _add_error_message(self, text: str):
        self.chat_column.controls.append(
            ft.Container(
                content=ft.Text(text, color=ft.Colors.RED),
                bgcolor="#7f1d1d",
                border_radius=10,
                padding=12,
            )
        )
        try:
            self.page.update()
        except Exception:
            pass

    def _add_assistant_message(self, text: str, sources: List[Dict[str, Any]]):
        controls: List[ft.Control] = [
            ft.Text("ИИ", size=12, weight=ft.FontWeight.BOLD, color=ft.Colors.GREEN)
        ]

        # FIX (проблемы 2 и 5 ТЗ): вместо «текст → картинка формулы → текст»
        # отдельными строками ответ собирается ПОСТРОЧНО: каждая строка текста
        # рендерится ft.Markdown, а инлайн-формулы $...$ вставляются в Row
        # той же строки мелкой картинкой baseline — текст больше не разрывается.
        # Блочные формулы $$...$$ остаются отдельной картинкой (по центру).
        # FIX (проблема 2): подряд идущие строки markdown-таблицы группиру-
        # ются в один блок ft.Markdown(extension tables) — таблица рисуется
        # таблицей, а не сырыми «|---|».
        # FIX (проблема 1 ТЗ — плейсхолдеры в тексте ответа): если LLM
        # сама вывела служебные маркеры {{FORMULA:N}} (скопировала из
        # эталонного примера промпта или не поняла инструкцию), они
        # НЕЗАМЕНЯЕМЫ у пользователя — корректных формул для них нет.
        # Убираем маркеры целиком; нормализация пробелов внутри маркеров
        # («{{ FORMULA:0 }}») выполняется в _md_block/_render_inline_line.
        text = re.sub(r"\s*\{\s*FORMULA\s*:\s*\d+\s*\}", "",
                      text or "", flags=re.IGNORECASE)

        lines = normalize_page_links(text or "").split("\n")

        # GRAPH-RAG (Этап 3): парсер блоков ответа. Ответ режется на блоки по
        # пустым строкам; блок, начинающийся с маркера сущности (**Определение**
        # / **Теорема** / Доказательство:), рендерится карточкой с цветной левой
        # рамкой и заголовком типа. Обычные строки идут прежним путём
        # (postrochno, таблицы группами) — полная обратная совместимость.
        entity_blocks = self._split_entity_blocks(lines)
        if entity_blocks is not None:
            controls.extend(entity_blocks)
        else:
            for group in _group_table_lines(lines):
                if not any(x.strip() for x in group):
                    continue
                if all(is_table_line(x) for x in group):
                    # REFACTORING (ТЗ п.1): таблица собирается в ft.DataTable,
                    # ячейки компилируются тем же inline-рендерером — формулы
                    # внутри ячеек становятся картинками, а не {{FORMULA:N}}/сырым $.
                    table = _build_table_block(group, self._render_inline_line)
                    if table is not None:
                        controls.append(table)
                    continue
                for line in group:
                    if not line.strip():
                        continue
                    stripped = line.strip()
                    if stripped.startswith("$$") and stripped.endswith("$$") and len(stripped) > 4:
                        img = self._formula_image(stripped[2:-2].strip(), block=True)
                        controls.append(img or ft.Text(stripped, style=ft.TextStyle(font_family="monospace")))
                        continue
                    controls.append(self._render_inline_line(stripped))

        # Кнопки экспорта ответа. FIX: снимок (вопрос, ответ, источники)
        # замыкается в lambda конкретной кнопки — экспорт всегда соответствует
        # показанному сообщению, а не глобальному состоянию.
        question_for_snap = self._pending_question or (
            self.rag_pipeline.history[-1]["question"]
            if getattr(getattr(self, "rag_pipeline", None), "history", None) else ""
        )
        export_snapshot = {
            "answer": text,
            "question": question_for_snap,
            "sources": list(sources),
        }

        def _do_export(fmt: str, snap=export_snapshot):
            # ТЗ (Task 3): выбор папки через tkinter.filedialog (без
            # ft.FilePicker) в фоновом потоке — диалог не блокирует UI-поток.
            save_dir = None
            try:
                import threading
                box = {}
                def _pick():
                    try:
                        from tkinter import Tk, filedialog
                        root = Tk(); root.withdraw(); root.attributes("-topmost", True)
                        box["d"] = filedialog.askdirectory(title="Куда сохранить файл?") or None
                        root.destroy()
                    except Exception as ex_pick:
                        logger.warning(f"Диалог выбора папки недоступен: {ex_pick}")
                        box["d"] = None
                th = threading.Thread(target=_pick); th.start(); th.join(120)
                save_dir = box.get("d")
            except Exception:
                save_dir = None
            # ТЗ (Task 3): защита от пустого экспорта. Приоритет — снимок
            # конкретного сообщения; если он пуст — берём глобальный last_answer.
            if not (snap.get("answer") or "").strip():
                current = getattr(self, "last_answer", "") or ""
                if current.strip():
                    snap = {"answer": current,
                            "question": getattr(self, "last_question", "") or snap.get("question", ""),
                            "sources": list(getattr(self, "last_sources", []) or [])}
            if not (snap["answer"] or "").strip():
                self._snack("❌ Экспортировать нечего: ответ пустой")
                return
            try:
                if fmt == "md":
                    path = self.exporter.export_to_markdown(
                        snap["answer"], snap["sources"], snap["question"])
                else:
                    path = self.exporter.export_to_pdf(
                        snap["answer"], snap["sources"], snap["question"])
                if save_dir:
                    import shutil
                    from pathlib import Path as _P
                    dst = _P(save_dir) / _P(path).name
                    shutil.move(str(path), str(dst))
                    path = str(dst)
                self._snack(f"✅ Файл сохранён: {path}")
            except Exception as ex:
                logger.error(f"Ошибка экспорта: {ex}")
                self._snack(f"❌ Ошибка экспорта: {ex}")

        # FIX: кнопки скачивания появляются ТОЛЬКО когда есть что скачивать.
        if (text or "").strip():
            controls.append(
                ft.Row(
                    [
                        ft.IconButton(
                            icon=ft.Icons.DOWNLOAD,  # Flet 1.x: ft.Icons.DOWN нет
                            tooltip="Скачать ответ в Markdown",
                            on_click=lambda e: _do_export("md"),
                        ),
                        ft.IconButton(
                            icon=ft.Icons.PICTURE_AS_PDF,
                            tooltip="Скачать ответ в PDF",
                            on_click=lambda e: _do_export("pdf"),
                        ),
                    ]
                )
            )

        # FIX (проблема 3 ТЗ — дублирование источников): раньше внизу висел
        # блок «📚 Источники» с полным списком, тогда как LLM уже ставит в
        # тексте ссылки [N]. Теперь список источников рендерится только если
        # в ОТВЕТЕ НЕТ ни одной ссылки [N] (тогда чипы — единственный способ
        # навигации). Если ссылки есть — показываем компактную строку
        # кликабельных чипов без заголовка-дублера.
        if sources:
            has_inline_refs = bool(re.search(r"\[\d{1,3}\]", text or ""))
            chips = []
            for i, source in enumerate(sources, start=1):
                title = source.get("title", "Без названия")
                page_no = source.get("page_number")
                chapter = source.get("chapter")
                label = (f"[{i}] {title}, стр. {page_no}" if isinstance(page_no, int)
                         else f"[{i}] {title}, {chapter or ''}".strip().rstrip(","))
                # FIX (переход к странице из чата): подсказка честно отражает
                # готовность перехода — если страница не известна (старый импорт
                # без реиндекса), клик покажет только карточку источника.
                tip = (f"Открыть стр. {page_no} в просмотрщике"
                       if isinstance(page_no, int) and page_no > 0
                       else "Источник: страница не известна — переимпортируйте книгу для перехода")
                chips.append(
                    ft.Chip(
                        label=ft.Text(label, size=11),
                        on_click=lambda e, s=source: self._open_source(s),
                        tooltip=tip,
                    )
                )
            if not has_inline_refs:
                controls.append(ft.Text("📚 Источники:", size=12,
                                       weight=ft.FontWeight.BOLD))
            controls.append(ft.Row(chips, wrap=True, spacing=4))

        # FIX (краш «Container.__init__() got an unexpected keyword argument
        # 'max_height'»): в Flet 1.x у ft.Container НЕТ параметра max_height
        # (в отличие от Flet 0.2x). Если сообщение ИИ выросло слишком
        # большим, ограничение высоты делаем легальным способом: Column со
        # scroll=AUTO внутри Container с фиксированным height. Порог —
        # 70% доступной высоты окна; ниже — bubble растёт по содержимому.
        bubble = ft.Column(controls, spacing=6)
        try:
            win_h = getattr(self.page, "window_height", None)
            if isinstance(win_h, (int, float)) and win_h > 0:
                cap = int(win_h * 0.7)
                approx_h = sum(
                    (getattr(c, "height", None) or 40) if isinstance(c, (ft.Text, ft.Image))
                    else 60 for c in controls) + 20 * len(controls)
                if approx_h > cap:
                    bubble = ft.Column(controls, spacing=6,
                                       scroll=ft.ScrollMode.AUTO, height=cap)
        except Exception as ex:  # любая недоступность атрибутов — без капа
            logger.debug(f"Каппинг высоты пузыря пропущен: {ex}")
        self.chat_column.controls.append(
            ft.Container(
                content=bubble,
                bgcolor=ft.Colors.with_opacity(0.15, ft.Colors.GREEN),
                border_radius=10,
                padding=15,
                margin=ft.Margin.only(right=80),
            )
        )
        try:
            self.page.update()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # GRAPH-RAG (Этап 3): адаптивная отрисовка блоков сущностей
    # ------------------------------------------------------------------

    _ENTITY_MARKERS = {
        "ОПРЕДЕЛЕНИЕ": ("#2196f3", "Определение"),
        "ТЕОРЕМА": ("#4caf50", "Теорема"),
        "ДОКАЗАТЕЛЬСТВО": ("#9c27b0", "Доказательство"),
        "ЛЕММА": ("#00bcd4", "Лемма"),
        "АКСИОМА": ("#ff9800", "Аксиома"),
    }
    _ENTITY_HEAD_RE = re.compile(
        r"^\s*(?:\*\*)?\s*(Определение|Теорема|Доказательство|Лемма|Аксиома)"
        r"\s*(?:[\d.§№:]*|\**)\s*[:.]?\s*(.*)$",
        re.IGNORECASE | re.UNICODE,
    )

    def _detect_entity_block(self, first_line: str):
        """Возвращает (ключ_типа, заголовок, остаток_строки) или None."""
        m = self._ENTITY_HEAD_RE.match(first_line or "")
        if not m:
            return None
        word = m.group(1).upper().rstrip(".")
        key = next((k for k in self._ENTITY_MARKERS if k.startswith(word[:4])), None)
        if key is None:
            return None
        rest = (m.group(2) or "").strip()
        title_full = f"{key.capitalize()} {rest}".strip() if rest else key.capitalize()
        return key, title_full, rest

    def _split_entity_blocks(self, lines: List[str]):
        """Режет ответ на карточки сущностей, если в тексте есть ≥1 маркер.

        Возвращает список контролов либо None — тогда вызывающий код рендерит
        ответ прежним построчным способом (обратная совместимость для ответов
        без структурных маркеров).
        """
        try:
            # Сначала режем на блоки по пустым строкам.
            blocks: List[List[str]] = []
            cur: List[str] = []
            for ln in lines:
                if not ln.strip():
                    if cur:
                        blocks.append(cur)
                        cur = []
                else:
                    cur.append(ln)
            if cur:
                blocks.append(cur)

            has_entity = any(self._detect_entity_block(b[0]) for b in blocks)
            if not has_entity:
                return None

            out: List[ft.Control] = []
            for block in blocks:
                det = self._detect_entity_block(block[0])
                if not det:
                    # Обычный блок — строки через прежний механизм.
                    for group in _group_table_lines(block):
                        if all(is_table_line(x) for x in group):
                            t = _build_table_block(group, self._render_inline_line)
                            if t is not None:
                                out.append(t)
                            continue
                        for ln in group:
                            if ln.strip():
                                s = ln.strip()
                                if s.startswith("$$") and s.endswith("$$") and len(s) > 4:
                                    img = self._formula_image(s[2:-2].strip(), block=True)
                                    out.append(img or ft.Text(s))
                                    continue
                                out.append(self._render_inline_line(s))
                    continue

                key, title, rest = det
                color, label = self._ENTITY_MARKERS[key]
                inner: List[ft.Control] = [
                    ft.Text(title, size=14, weight=ft.FontWeight.BOLD),
                ]
                body_lines = ([rest] if rest else []) + block[1:]
                for group in _group_table_lines(body_lines):
                    if all(is_table_line(x) for x in group):
                        t = _build_table_block(group, self._render_inline_line)
                        if t is not None:
                            inner.append(t)
                        continue
                    for ln in group:
                        if not ln.strip():
                            continue
                        s = ln.strip()
                        if s.startswith("$$") and s.endswith("$$") and len(s) > 4:
                            img = self._formula_image(s[2:-2].strip(), block=True)
                            inner.append(img or ft.Text(s))
                            continue
                        inner.append(self._render_inline_line(s))
                out.append(ft.Container(
                    content=ft.Column(inner, spacing=4),
                    border=ft.Border(left=ft.BorderSide(4, color)),
                    border_radius=6,
                    bgcolor="#1e1e24",
                    padding=ft.Padding.only(left=12, top=8, bottom=8, right=8),
                    margin=ft.Margin.only(bottom=6),
                ))
            return out
        except Exception as e:
            logger.warning(f"Graph-RAG UI: разметка блоков пропущена: {e}")
            return None

    def _formula_image(self, latex: str, block: bool = False):
        """Формула для вставки в чат.

        REFACTORING (ТЗ п.3 — AMS-LaTeX без сбоев): приоритетный путь —
        нативный LaTeX-рендер Flet 1.x (ft.Markdown + $$…$$ рисует формулу
        через встроенный MathJax/MathML-движок Flutter: векторно, тему-
        зависимым цветом, поддерживает \\implies, \\leqslant, \\iiint, матрицы).
        Фолбэк — PNG из LatexRenderer (matplotlib mathtext), финальный фолбэк —
        None: вызывающий код печатает юникод-нотацию (unicode_math) текстом.
        """
        # --- Путь 1: нативный рендер Flet (вектор, без PNG) ---
        try:
            clean = latex.replace("$", "").strip()
            if not clean:
                return None
            value = f"$${clean}$$" if block else f"${clean}$"
            md = ft.Markdown(
                value, selectable=True,
                extension_set=ft.MarkdownExtensionSet.GITHUB_FLAVORED,
                latex_scale_factor=1.0 if block else 0.95,
            )
            if block:
                return ft.Container(content=md, alignment=ft.Alignment.CENTER,
                                    padding=ft.Padding.only(top=4, bottom=4))
            return md
        except Exception as ex:  # сигнатура/параметры недоступны?
            logger.debug(f"Нативный LaTeX-рендер Flet недоступен: {ex}")

        # --- Путь 2: чистый MathML через latex2mathml (ТЗ п.2) ---
        ml = self._formula_markdown(latex, block=block)
        if ml is not None:
            return ml

        # --- Путь 3: PNG через matplotlib mathtext (фолбэк-рендер) ---
        try:
            path = self.latex_renderer.render_formula(
                latex, color="#E0E0E0" if self._dark_formulas else "#111111")
        except Exception as ex:
            logger.debug(f"Рендер формулы не удался: {ex}")
            return None
        if not path:
            return None
        if block:
            return ft.Image(src=path, fit=ft.BoxFit.SCALE_DOWN, width=400,
                            border_radius=6)
        # Инлайн: компактная картинка без подложки, по базовой линии строки.
        return ft.Image(src=path, fit=ft.BoxFit.CONTAIN, height=20)

    def _formula_markdown(self, latex: str, block: bool = False):
        """REFACTORING (ТЗ п.2 «векторный движок»): LaTeX → MathML (latex2mathml).

        Возвращает ft.Markdown с готовым <math>…</math> блоком — векторный
        рендер без PNG и без matplotlib. Flet 1.x умеет отображать MathML в
        Markdown; если среда не поддерживает это, вызывающий код деградирует
        к путям _formula_image (PNG-фолбэк) и unicode_math (текст).
        Никогда не бросает наружу: при недоступности конвертера -> None.
        """
        try:
            from src.processing.latex_mathml import get_mathml_converter
            mathml = get_mathml_converter().to_mathml(latex.replace("$", "").strip())
            if not mathml:
                return None
            md = ft.Markdown(mathml, selectable=True,
                             extension_set=ft.MarkdownExtensionSet.GITHUB_FLAVORED)
            if block:
                return ft.Container(content=md, alignment=ft.Alignment.CENTER,
                                    padding=ft.Padding.only(top=4, bottom=4))
            return md
        except Exception as ex:
            logger.debug(f"MathML-рендер недоступен: {ex}")
            return None

    def _render_inline_line(self, stripped: str) -> ft.Control:
        """Строка ответа с инлайн-формулами (ТЗ п.2 — препроцессор).

        Display-формулы $$...$$ рендерятся PNG-картинками в потоке строки;
        ИНЛАЙН-формулы $...$ принудительно переводятся clean_inline_math в
        текстовые юникод-эквиваленты ДО передачи во Flet — латинские буквы и
        кванторы больше никогда не вырезаются в пустые места.
        """
        # Сначала защищаем display-формулы от ft.Markdown ($ в Flet — мат.
        # ограничитель); инлайн-формулы при этом уже конвертированы в unicode.
        formulas: List[str] = []
        stripped = normalize_page_links(_normalize_formula_tokens(stripped))
        blocks: List[str] = []

        def _save_block(m):
            blocks.append(m.group(1).strip())
            return "{{BLOCKEQ%d}}" % (len(blocks) - 1)

        stripped = re.sub(r"\$\$([^$]+?)\$\$", _save_block, stripped)
        # FIX (скриншот 2 — сырой \frac без $): инлайн-формулы, СОДЕРЖАЩИЕ
        # LaTeX-команды (\frac, \angle, \sum…), рендерятся полноценной
        # картинкой mathtext (тот же путь, что display-блоки) — юникод-
        # сверка калечит составные формулы («½ · CH · AB»). Простые инлайны
        # ($\exists$, «$A_1$») по-прежнему сворачиваются в текст конвертером.
        def _renderable_inline(m):
            body = m.group(1)
            if not re.search(r"\\[A-Za-z]+", body):
                return m.group(0)
            formulas.append(body.strip())
            return "{{FORMULA:%d}}" % (len(formulas) - 1)
        stripped = re.sub(r"(?<!\\)\$(?!\$)([^$\n]+?)\$", _renderable_inline,
                          wrap_bare_math_unicode(stripped))
        stripped = clean_inline_math(stripped)
        # Чиним пробелы у кавычек-ёлочек (« X » → «X») — скриншот 1.
        stripped = fix_glued_quotes(stripped)
        # Возвращаем display-блоки маркерами, которые уходят в PNG-путь
        for i, body in enumerate(blocks):
            stripped = stripped.replace("{{BLOCKEQ%d}}" % i, "$$" + body + "$$")
        protected = _protect_formulas(stripped, formulas)
        # Маркеры {{FORMULA:N}}, присланные самой моделью (без известного
        # LaTeX), вырезаем — сырые плейсхолдеры на экране недопустимы (ТЗ п.1).
        protected = re.sub(r"\s*\{\{FORMULA:\d+\}\}", "", protected)

        links = list(PAGE_LINK_RE.finditer(protected))
        parts: List[ft.Control] = []

        def add_md(chunk: str):
            chunk = chunk.strip(" —·:")
            if not chunk:
                return
            # Заменяем маркеры формул на картинки прямо в потоке сегментов.
            pos = 0
            for m in _FORMULA_TOKEN_RE.finditer(chunk):
                before = chunk[pos:m.start()]
                if before.strip():
                    parts.append(ft.Markdown(
                        before.strip(), selectable=True,
                        extension_set=ft.MarkdownExtensionSet.GITHUB_FLAVORED))
                img = self._formula_image(formulas[int(m.group(1))])
                if img is not None:
                    parts.append(img)
                else:
                    # Финальный фолбэк (ТЗ п.3): юникод-нотация формулы как
                    # обычный текст — никогда не падает и читаем в обеих темах.
                    try:
                        from src.processing.latex_unicode import unicode_math
                        uni = unicode_math(formulas[int(m.group(1))])
                    except Exception:
                        uni = ""
                    parts.append(ft.Text(uni or f"$({formulas[int(m.group(1))]})$",
                                         size=13))
                pos = m.end()
            tail = chunk[pos:]
            if tail.strip():
                parts.append(ft.Markdown(
                    tail.strip(), selectable=True,
                    extension_set=ft.MarkdownExtensionSet.GITHUB_FLAVORED))

        if not links:
            add_md(protected)
            if not parts:
                return ft.Container(visible=False)
            if len(parts) == 1 and isinstance(parts[0], ft.Markdown):
                return parts[0]
            return ft.Row(parts, wrap=True, spacing=4,
                          vertical_alignment=ft.VerticalAlignment.CENTER)

        pos = 0
        for m in links:
            add_md(protected[pos:m.start()])
            page_no = int(m.group(1))
            parts.append(
                ft.Chip(
                    label=ft.Text(f"стр. {page_no}", size=12),
                    on_click=lambda e, n=page_no: ChatView._open_page_static(n),
                    tooltip=f"Открыть страницу {page_no} в просмотрщике",
                )
            )
            pos = m.end()
        add_md(protected[pos:])
        prefix = ([] if PAGE_LINK_RE.sub("", protected).strip() else
                  [ft.Text("Перейти:", size=12, color="grey")])
        return ft.Row([*prefix, *parts], wrap=True, spacing=4,
                      vertical_alignment=ft.VerticalAlignment.CENTER)

    def _open_source(self, source: Dict[str, Any]):
        page_number = source.get("page_number")
        doc_id = source.get("doc_id")
        # Страница может приходить строкой ("12" или "12-18") — берём начало.
        first_page = None
        if isinstance(page_number, int):
            first_page = page_number
        elif isinstance(page_number, str):
            m = re.match(r"\s*(\d+)", page_number)
            if m:
                first_page = int(m.group(1))

        if first_page is not None and self._on_open_source:
            # Целевое действие «запрос контекста документа из чата» →
            # панель просмотра раскрывается и листается на нужную страницу.
            self._on_open_source(first_page, doc_id)
        else:
            self._snack(f"Источник: {source.get('title', '?')}, "
                        f"{source.get('chapter') or ('стр. ' + str(page_number) if page_number else '')}")

    def _open_page_link(self, page_number: int):
        """Клик по |page:N| в оглавлении ответа ИИ."""
        if self._on_open_source:
            active = getattr(self.rag_pipeline, "_active_doc_id", None)
            self._on_open_source(int(page_number), active)

    def _export_answer(self, answer: str, fmt: str):
        # Устаревший путь: сохранён для совместимости, делегирует в общий
        # экспорт с актуальным снимком состояния.
        self._pending_question = self._pending_question or (
            self.rag_pipeline.history[-1]["question"] if self.rag_pipeline.history else ""
        )
        sources = (self._last_result or {}).get("sources", [])
        try:
            if fmt == "md":
                path = self.exporter.export_to_markdown(answer, sources, self._pending_question)
            else:
                path = self.exporter.export_to_pdf(answer, sources, self._pending_question)
            self._snack(f"✅ Файл сохранён: {path}")
        except Exception as ex:
            logger.error(f"Ошибка экспорта: {ex}")
            self._snack(f"❌ Ошибка экспорта: {ex}")

    def _snack(self, message: str):
        snack = ft.SnackBar(content=ft.Text(message))
        self.page.overlay.append(snack)
        snack.open = True
        try:
            self.page.update()
        except Exception:
            pass
