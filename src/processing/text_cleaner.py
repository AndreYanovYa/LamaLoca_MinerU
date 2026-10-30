# src/processing/text_cleaner.py
"""Пре-процессор текста после OCR/LaTeX-распознавания.

ТЗ (раздел 1): автоматическая коррекция шума Tesseract OCR на стыке
распознавания окрестностей и проколотых точек. Локальная LLM, питаясь
таким шумом, генерирует ложную «разность множеств» вида ``X \\setminus O(x_0)``
(«все точки, КРОМЕ окрестности») вместо проколотой окрестности
``x \\in \\dot{O}(x_0)`` / ``x \\in O(x_0) \\setminus {x_0}``, что переворачивает
смысл определения предела по Коши.

Правила замены (эталон ТЗ приведён как示意; здесь — синтаксически корректные
Python-regex с тем же смыслом):

    text = re.sub(r"X\\s*\\\\+\\s*setminus\\s+([UO]\\(x_0\\))", r"x \\in \\dot{\\1}", text)
    text = re.sub(r"X\\s*\\\\+\\s*([UO]\\(x_0\\))",             r"x \\in \\dot{\\1}", text)

Модуль потокобезопасен: только чистые функции над строками, без общего
изменяемого состояния (скомпилированные regex неизменяемы).
"""
import json
import re
from loguru import logger
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# 1. Ложная разность множества на окрестностью (шум OCR проколотой точки).
#    Punctured neighbourhood распознаётся Tesseract как "X \ O(x_0)" или
#    "X \setminus U(x_0)" — обе формы переводим в канон \dot{O}/\dot{U}.
# ---------------------------------------------------------------------------
_RE_FALSE_SETMINUS_NBD = re.compile(
    r"X\s*\\+\s*(?:setminus\s*)?([UO])\(\s*x_0\s*\)",
)

# "x \in X \ O(x_0)" без следа setminus — точечный шум вокруг "\":
_RE_BARE_BACKSLASH_NBD = re.compile(
    r"(?<![A-Za-z])([Xx])\s*\\\s*([UO])\(\s*x_0\s*\)",
)


def _punctured_repl(m: re.Match) -> str:
    """Каноническая запись проколотой окрестности для найденного совпадения."""
    nb = m.group(2) if m.re is _RE_BARE_BACKSLASH_NBD else m.group(1)
    return rf"x \in \dot{{{nb}}}(x_0)"


def fix_punctured_neighborhood(text: str) -> str:
    """Исправляет ложную разность множеств на проколотую окрестность.

    Было (OCR-шум / ошибка модели):
        X \\setminus O(x_0), X \\ U(x_0), x \\in X \\ O(x_0)
    Стало (корректный смысл предела по Коши):
        x \\in \\dot{O}(x_0), x \\in \\dot{U}(x_0)
    """
    if not text:
        return text
    text = _RE_FALSE_SETMINUS_NBD.sub(_punctured_repl, text)
    text = _RE_BARE_BACKSLASH_NBD.sub(_punctured_repl, text)
    return text


# ---------------------------------------------------------------------------
# 2. Канонизация альтернативных записей проколотой окрестности.
#    \mathring{O}, O^\circ, \overset{\circ}{O} -> \dot{O} — один формат,
#    который понимает latex_renderer (mathtext) и ожидает системный промпт.
# ---------------------------------------------------------------------------
_RE_RING_VARIANTS = [
    (re.compile(r"\\mathring\{([UO])\}"), r"\\dot{\1}"),
    (re.compile(r"\\overset\s*\{\\circ\}\{([UO])\}"), r"\\dot{\1}"),
    (re.compile(r"([UO])\^\\circ\b"), r"\\dot{\1}"),
]


def normalize_punctured_variants(text: str) -> str:
    for rx, repl in _RE_RING_VARIANTS:
        text = rx.sub(repl, text)
    return text


# ---------------------------------------------------------------------------
# 3. КАНОН математических ограничителей (Flet 1.x / CommonMark + KaTeX).
#    Фронтенд ft.Markdown понимает ТОЛЬКО парные маркеры:
#      $$ ... $$ — блочная формула (отдельной строкой),
#      $  ... $  — инлайн-формула.
#    Любые альтернативные формы вывода модели — \[...\], \(...\), «голые»
#    цепочки Unicode-глифов (∀x ∈ ℝ) — KaTeX не видит вовсе: формула
#    интерпретируется как обычный текст. Поэтому на границе Ingestion ->
#    Чанкинг -> Промпт все ограничители приводятся к канону ДО любых других
#    преобразований.
# ---------------------------------------------------------------------------
_RE_DISPLAY_TEXBRACK = re.compile(r"\\\[(.+?)\\\]", re.DOTALL)
_RE_INLINE_TEXPAR = re.compile(r"\\\((.+?)\\\)", re.DOTALL)

# Голые math-глифы: последовательности вида «∀x ∈ ℝ», «aₙ → a» модель часто
# выводит без $...$. Оборачиваем плотные зацепки в $...$ (только если внутри
# есть хотя бы один «тяжёлый» глиф — квантор/принадлежность/стрелка).
_HEAVY_GLYPHS = "\u2200\u2203\u2204\u2208\u2209\u220b\u2205\u2282\u2286\u2283\u2287\u222a\u2229\u2216\u21d2\u21d4\u2192\u2190\u2261\u2260\u2264\u2265\u2248\u2211\u220f\u222b\u221e\u2202\u2207\u00ac\u2227\u2228"
_LIGHT_GLYPHS = _HEAVY_GLYPHS + "\u03b1\u03b2\u03b3\u03b4\u03b5\u03b6\u03b7\u03b8\u03ba\u03bb\u03bc\u03bd\u03be\u03c0\u03c1\u03c3\u03c4\u03c5\u03c6\u03c7\u03c8\u03c9\u211d\u2115\u2124\u211a\u2102\u2026\u00b7\u00d7\u00f7\u00b1\u2032"
_RE_BARE_MATH_RUN = re.compile(
    "(?<![\\w$])"                                          # не из середины слова/формулы
    "([%s][%s]*(?:[ ,;:.][ ]?[%s][%s]*)*)"                # плотная цепочка глифов
    % (_HEAVY_GLYPHS, _LIGHT_GLYPHS, _HEAVY_GLYPHS, _LIGHT_GLYPHS),
)


# Псевдо-дроби из текстового слоя PDF (PyMuPDF превращает \\frac{...}{...}
# в «fract^(k)(x 0) k!» / «f^(n)(x_0)/n!»). Конвертируем в валидный LaTeX —
# фронтенд (KaTeX) понимает только \\frac{num}{den}.
_RE_PSEUDO_FRACT = re.compile(
    # fract(a)/(b), fract^(k)(x 0) k!, fraction .../...
    r"\bfr(?:act|action)?\s*\^?\(([^()\n]{1,40}?)\)\s*(?:/\s*\(([^()\n]{1,40}?)\)|[ )]\s*([A-Za-z][A-Za-z0-9_ ]{0,20}?![\s!]*))"
    r"|\bfr(?:act|action)?\s*([A-Za-z0-9_^{}()'\-+* ]{1,30}?)\s*/\s*([A-Za-z0-9_^{}()'\-+* ]{1,30}?)(?![\w/])"
)
# num/den где num — «математический» токен (со степенью/индексом/факториалом):
# f^(n)(x_0)/n!, a_n/k!, x^2/y. Обычные слова вида «a/b», «вход/выход» НЕ
# трогаем — это слэши естественного языка.
_MATH_ATOM = r"[A-Za-z](?:\^\([^()\n]{0,24}\)|\^[^ /\t]{1,8}|_\{[^{}\n]{1,16}\}|_\d|[!]\d*)"
_RE_PSEUDO_SLASH_FRAC = re.compile(
    r"(?<![\w$\\./-])(" + _MATH_ATOM + r"(?:\([^()\n]{0,24}\))*)"
    r"\s*/\s*"
    r"(" + _MATH_ATOM + r")"
    r"(?![\w$/])"
)


def _tidy_frac_operand(s: str) -> str:
    """Нормализует операнд псевдо-дроби: снимает мусор, восстанавливает
    потерянные PyMuPDF индексы («x 0» -> «x_0») и функции («sin x» -> «sin{x}»)."""
    s = s.strip().strip("() ")
    # «2n+1 !» -> «2n+1!», сохраняем факториалы
    had_fact = "!" in s
    s = re.sub(r"\s*!\s*", "!", s).strip()
    # Индексы, потерянные при извлечении: одиночная буква + пробел + цифра/буква
    s = re.sub(r"\b([A-Za-z])([ ]+)(\d+|[a-zA-Z])\b", r"\1_\3", s)
    # Известные мат.функции склеиваем с аргументом как \operatorname-подобная форма
    s = re.sub(r"\b(sin|cos|tan|cot|arcsin|arccos|arctan|sh|ch|th|ln|lg|exp|lim|max|min)\s+", r"\1 ", s)
    if had_fact and not s.endswith("!"):
        s += "!"
    return s


def _split_glued_frac_operands(rest: str):
    """Разбирает слитый хвост после «frac» в (числитель, знаменатель).

    Вход — текст сразу за словом frac, например:
      'x^33!'          -> ('x^3', '3!')
      'x^2n+1(2n+1)!'  -> ('x^{2n+1}', '(2n+1)!')
      'sin xx'         -> ('sin x', 'x')
      '00'             -> ('0', '0')
      'f(x)g(x)'       -> ('f(x)', 'g(x)')
      'nn!'            -> ('n', 'n!')
    Эвристики подобраны под текстовый слой Кудрявцева; несовпадение формы
    возвращает None (сегмент останется на совести других правил).
    """
    if not rest:
        return None

    def read_atom(i: int, greedy_exp: bool = True):
        """Читает один «атом» формулы начиная с i; возвращает (текст, след.позиция).

        greedy_exp=False — после '^' читается ОДИН символ: так различаются
        слитые степени из текстового слоя («x^33!» = x³/3!, а не x^{33!}).
        """
        if i >= len(rest):
            return "", i
        ch = rest[i]
        if ch == "(":
            depth, j = 0, i
            while j < len(rest):
                if rest[j] == "(":
                    depth += 1
                elif rest[j] == ")":
                    depth -= 1
                    if depth == 0:
                        j += 1
                        break
                j += 1
            atom = rest[i:j]
        elif ch.isalpha() or ch.isdigit():
            j = i
            while j < len(rest) and (rest[j].isalnum()):
                j += 1
            atom = rest[i:j]
        else:
            return "", i
        # присоединяем степени/индексы/факториалы к атому: x^3, a_n, n!, (2n+1)!
        while j < len(rest):
            if rest[j] == "^":
                j += 1
                if j < len(rest) and rest[j] == "(":
                    _, j = read_atom(j)
                elif j < len(rest) and rest[j] == "{":
                    k = rest.find("}", j)
                    j = (k + 1) if k != -1 else j + 2
                elif greedy_exp:
                    # многосимвольная степень без скобок: x^2n+1 -> берём до конца цепочки
                    while j < len(rest) and (rest[j].isalnum() or rest[j] in "+-"):
                        j += 1
                elif j < len(rest):
                    j += 1
            elif rest[j] == "_":
                j += 1
                if j < len(rest) and rest[j] == "{":
                    k = rest.find("}", j)
                    j = (k + 1) if k != -1 else j + 2
                elif j < len(rest):
                    j += 1
            elif rest[j] == "!":
                j += 1
            else:
                break
        return atom, j

    funcs = ("arcsin", "arccos", "arctan", "sin", "cos", "tan", "cot", "sh", "ch",
              "th", "ln", "lg", "exp", "lim", "max", "min")

    def read_term(i: int, greedy_exp: bool = True):
        """Терм числителя/знаменателя: функция с аргументом либо один атом."""
        for f in funcs:
            if rest.startswith(f, i):
                j = i + len(f)
                arg, j = read_atom(j, greedy_exp)
                return rest[i:j], j
        return read_atom(i, greedy_exp)

    num, p = read_term(0, greedy_exp=False)
    if not num or p >= len(rest):
        return None
    den_start = p
    # Склейка «функция(аргумент)» внутри одного операнда: если сразу после
    # буквы идёт '(' и скобка закрывается ДО конца хвоста — это аргумент
    # функции («f(x)g(x)» = f(x)/g(x)), а не граница num|den. Добираем
    # «(...)» в числитель и продолжаем границу со следующего символа.
    if rest.startswith("(", p):
        depth, k = 0, p
        while k < len(rest):
            if rest[k] == "(":
                depth += 1
            elif rest[k] == ")":
                depth -= 1
                if depth == 0:
                    break
            k += 1
        if depth == 0 and k + 1 < len(rest):
            num = rest[:k + 1]
            den_start = k + 1
    # Если после закрытой скобки числителя идёт новая буква — это конец
    # аргумента функции («f(x)g(x)» -> num='f(x)', den='g(x)'). Иначе
    # («(x)...» без закрывающей до конца) — граница неразличима.
    # Знаменатель обязан разбираться как терм(ы) ПОЛНОСТЬЮ — иначе это не
    # дробь, а слитый обрывок фразы. Цепочка атомов до конца строки:
    # «(2n+1)!», «3!», «x^2».
    j = den_start
    while j < len(rest):
        a2, j2 = read_term(j, greedy_exp=True)
        if not a2 or j2 <= j:
            return None
        j = j2
    den = rest[den_start:]
    # Не угадываем на двусмысленных склейках текстового слоя:
    #  * знаменатель начинается с закрывающей скобки — числитель обрезан
    #    посередине аргумента;
    #  * после числителя сразу идёт имя функции («fracsin xx») — граница
    #    между операндами неразличима.
    if den.startswith(")"):
        return None
    if any(den.startswith(f) for f in funcs):
        return None
    # Баланс скобок в каждом операнде обязателен.
    def _balanced(s: str) -> bool:
        d = 0
        for ch in s:
            if ch == "(":
                d += 1
            elif ch == ")":
                d -= 1
                if d < 0:
                    return False
        return d == 0

    if not (_balanced(num) and _balanced(den)):
        return None
    return num, den


def _convert_bare_frac(text: str) -> str:
    """Слитые из текстового слоя PDF дроби «fracf(x)g(x)», «fracx^33!» ->
    корректные инлайн-формулы $\\frac{...}{...}$ (без пробелов-операндов)."""
    if "frac" not in text:
        return text

    def _repl(m: re.Match) -> str:
        operands = _split_glued_frac_operands(m.group(1))
        if not operands:
            return m.group(0)
        num, den = operands
        num, den = _tidy_frac_operand(num), _tidy_frac_operand(den)
        if not num or not den:
            return m.group(0)
        # защитимся от абсурдно длинных операндов — это точно не дробь
        if len(num) > 40 or len(den) > 40:
            return m.group(0)
        return r"$\frac{%s}{%s}$" % (num, den)

    # «frac» без предшествующего обратного слэша и буквы — именно артефакт
    # извлечения; жадный хвост до пробела/знака препинания.
    return re.sub(r"(?<![A-Za-z\\])frac([A-Za-z0-9_(){}\^\-+.!]{2,60})",
                  _repl, text)


def _convert_pseudo_frac(seg: str) -> str:
    """«fract^(k)(x 0) k!» / «f^(n)(x_0)/n!» / «fracx^33!» -> \\frac{...}{...}."""
    if not seg or ("/" not in seg and "fr" not in seg):
        return seg

    def _tidy(s: str) -> str:
        s = s.strip().rstrip("!). ") + ("!" if "!" in s else "")
        # PyMuPDF теряет индексы: «x 0» -> «x_0», «a n» -> «a_n»
        s = re.sub(r"\b([A-Za-z])\s+(\d|[a-zA-Z])(?=\b)", r"\1_\2", s)
        return s

    def _fract_repl(m: re.Match) -> str:
        num = _tidy(m.group(1) or m.group(4) or "")
        den = _tidy(m.group(2) or m.group(3) or m.group(5) or "")
        if num and den:
            return "$$\\frac{%s}{%s}$$" % (num, den)
        return m.group(0)

    out = _RE_PSEUDO_FRACT.sub(_fract_repl, seg)
    out = _RE_PSEUDO_SLASH_FRAC.sub(
        lambda mm: "$$\\frac{%s}{%s}$$" % (_tidy(mm.group(1)), _tidy(mm.group(2))),
        out,
    )
    # Слитые дроби без разделителей: «fracx^33!», «fracf(x)g(x)», «frac00»
    out = _convert_bare_frac(out)
    return out


def normalize_math_delimiters(text: str) -> str:
    """Приводит ВСЕ математические ограничители к канону KaTeX ($...$, $$...$$).

    Было:  \\[ \\sum ... \\]   \\( x \\in X \\)   ∀x ∈ ℝ (голый юникод)
           fract^(k)(x 0)k! / f^(n)(x_0)/(n!) (псевдо-UnicodeMath из PyMuPDF)
    Стало: $$ \\sum ... $$    $ x \\in X $     $∀x ∈ ℝ$
           $$\\frac{f^{(k)}(x_0)}{k!}$$
    Уже экранированные \\$ и содержимое корректных $...$/$$...$$ блоков
    не трогается (обработка сегментами вне формул).
    """
    if not text:
        return text

    def _seg(seg: str) -> str:
        seg = _RE_DISPLAY_TEXBRACK.sub(lambda m: "$$" + m.group(1).strip() + "$$", seg)
        seg = _RE_INLINE_TEXPAR.sub(lambda m: "$ " + m.group(1).strip() + " $", seg)
        # Псевдо-дроби из текстового слоя PDF (fract^(k)(x 0)k!, a/b c):
        # конвертируем в валидный LaTeX, иначе фронтенд видит «fract^(k)».
        seg = _convert_pseudo_frac(seg)
        seg = _RE_BARE_MATH_RUN.sub(lambda m: f"${m.group(1)}$", seg)
        return seg

    # Идём по строке сегментами «вне формул», существующие $...$/$$...$$ не трогаем
    out_lines = []
    for line in text.split("\n"):
        pos, buf = 0, []
        for dm in re.finditer(r"\$\$.+?\$\$|\$[^$\n]+\$", line):
            buf.append(_seg(line[pos:dm.start()]))
            buf.append(dm.group(0))
            pos = dm.end()
        buf.append(_seg(line[pos:]))
        out_lines.append("".join(buf))
    return "\n".join(out_lines)


def balance_math_delimiters(text: str) -> str:
    """Гарантирует чётность маркеров $ в тексте (защита от «огрызков»).

    Если после чанкинга/генерации остался НЕЗАКРЫТЫЙ маркер (нечётное число
    одиночных $ или нечётное число $$-ограничителей), он нейтрализуется
    (маркер снимается), иначе KaTeX на фронтенде аварийно завершается
    (throwOnError) и весь ответ отображается сырым текстом.
    """
    if not text or "$" not in text:
        return text
    # Блочные ограничители: нечётное число $$ -> снимаем последний одиночный $
    n_blocks = len(re.findall(r"\$\$", text))
    if n_blocks % 2:
        idx = text.rfind("$$")
        text = text[:idx] + "$" + text[idx + 2:]  # превращаем в нейтральный символ
    # Инлайновые: считаем оставшиеся одиночные $ вне $$
    stripped = text.replace("$$", "")
    n_inline = len(re.findall(r"(?<!\\)\$", stripped))
    if n_inline % 2:
        idx = text.rfind("$")
        text = text[:idx] + text[idx + 1:]
    return text


# ---------------------------------------------------------------------------
# FIX (лог ИИ-кодера 15:1x, Тейлор): постобработка ОТВЕТА LLM.
# Мелкие модели систематически выдают три артефакта разметки:
#   1. «лишний слэш» перед операторами LaTeX вне формул и внутри инлайн-блоков:
#      o((x - x_0)\^n)  ->  o((x - x_0)^n)
#   2. пустые маркеры списков («•  — коэффициент...», «- » в начале строки без
#      содержимого) — модель теряет переменную перед тире;
#   3. broken bold: ** п -ю производную** — пробел после открывающего/перед
#      закрывающим ** ломает CommonMark (жирный не применяется).
# Все функции идемпотентны и НЕ трогают валидный LaTeX внутри $$...$$
# (там \^ — допустимая команда), а также код-блоки ``` ... ```.
# ---------------------------------------------------------------------------

_RE_LLM_CODEBLOCK = re.compile(r"```.*?```", re.DOTALL)


def fix_llm_math_artifacts(text: str) -> str:
    """Снимает «косой слэш» перед ^/_ только ВНЕ $$-блоков и код-секций.

    Внутри блочной формулы \\^ — легальная KaTeX-команда, её не трогаем;
    в прозе/инлайн-аргументах (o((x - x_0)\\^n)) это артефакт экранирования.
    Идемпотентна: повторный вызов ничего не меняет.
    """
    if not text or "\\" not in text:
        return text
    parts = re.split(r"(\$\$.*?\$\$)", text, flags=re.DOTALL)
    for i, seg in enumerate(parts):
        if i % 2 == 1:          # блок формулы — не трогаем
            continue
        seg = re.sub(r"\\\^", "^", seg)
        seg = re.sub(r"\\_(?=[A-Za-z0-9(])", "_", seg)
        # FIX (лог 23:xx, «Parser Error: Undefined control sequence: \gdef»):
        # LLM иногда вставляет служебные TeX-макросы (\gdef, \edef, \def,
        # \newcommand…) прямо в инлайн-прозу — KaTeX их не определяет и роняет
        # весь рендер. Вне $$-блоков они всегда артефакт: СНИМАЕМ ВМЕСТЕ С
        # АРГУМЕНТАМИ («\gdef\foo{1} …» → «…» — раньше оставался мусор
        # «\foo{1}», который снова валил парсер). Внутри $$-блоков макросы
        # легальны — сегменты-формулы пропущены выше.
        seg = re.sub(
            r"\\(?:gdef|edef|xdef|global|longdef)\s*"
            r"(?:\\[A-Za-z]+\s*)?(?:\{[^{}]*\})?", "", seg)
        seg = re.sub(
            r"\\(?:def|newcommand|renewcommand|providecommand|let)\b\s*"
            r"(?:\\[A-Za-z]+|[A-Za-z])\s*(?:\[[0-9]\])?\s*\{[^{}]*\}", "", seg)
        parts[i] = seg
    return "".join(parts)


# ---------------------------------------------------------------------------
# FIX (лог 21:10, Тейлор): «сырая» математика БЕЗ $...$ и без бэкслэшей в
# прозе ответа. Модель пишет P n^(k)(x 0) = f^(k)(x 0) вместо P_n^{(k)}(x_0)=…;
# ни math_guard, ни UI-нормализатор chat_view такую запись не видят (нет seed-
# команды со слэшем), и пользователь видит голые символы. Здесь выражение
# канонизируется в инлайн $...$: надстрочные ^{...}/^X → \index{}{},
# подстрочные _ —> _, склеенные аргументы «x 0» → «x_0», функции cos/sin/lim…
# → \cos…. Вызывается ДО fix_glued_quotes в chat_view, поэтому работает даже
# если пользователь ещё не перенёс патчи UI. Идемпотентно: уже обёрнутые $...$
# и $$...$$ блоки не трогаются.
# ---------------------------------------------------------------------------

# Unicode-скрипты текстового слоя PDF (Кудрявцев): «s₀», «rₙ(x)», «s²ⁿ⁺¹».
_SUB_DIGITS = {
    "\u2080": "0",
    "\u2081": "1",
    "\u2082": "2",
    "\u2083": "3",
    "\u2084": "4",
    "\u2085": "5",
    "\u2086": "6",
    "\u2087": "7",
    "\u2088": "8",
    "\u2089": "9",
    "\u2099": "n",
    "\u2098": "m",
    "\u209a": "p",
    "\u209b": "s",
    "\u209c": "t",
    "\u1d62": "i",
    "\u2090": "a",
    "\u2091": "e",
    "\u2093": "x",
    "\u2092": "o",
    "\u2094": "q",
    "\u2095": "k",
    "\u2096": "l",
    "\u2097": "r",
}
_SUP_MAP = {
    "\u2070": "0",
    "\u00b9": "1",
    "\u00b2": "2",
    "\u00b3": "3",
    "\u2074": "4",
    "\u2075": "5",
    "\u2076": "6",
    "\u2077": "7",
    "\u2078": "8",
    "\u2079": "9",
    "\u207f": "n",
    "\u2071": "i",
    "\u1d40": "A",
    "\u1d41": "B",
    "\u1d2c": "D",
    "\u1d47": "G",
    "\u1d49": "h",
    "\u1d4a": "j",
    "\u1d4b": "k",
    "\u1d4d": "m",
    "\u1d50": "p",
    "\u1d51": "r",
    "\u1d52": "s",
    "\u1d56": "t",
    "\u1d57": "u",
    "\u1d58": "v",
    "\u1d5a": "x",
    "\u1dbb": "z",
    "\u207a": "+",
    "\u207b": "-",
    "\u207c": "=",
    "\u207d": "(",
    "\u207e": ")",
    "\u1d54": "R",
    "\u1d43": "a",
    "\u1d45": "c",
    "\u1d46": "e",
    "\u02b0": "h",
}
_SUP_KEYS = ''.join(_SUP_MAP)


_RAW_MATH_FUNC_RE = re.compile(
    r"(?<![A-Za-z\\])(cos|sin|tg|ctg|arctg|arcctg|arcsin|arccos|ln|lg|log|"
    r"exp|lim|max|min|det|dim|arg)(?=[\s_({]|\[)"
)


# ---------------------------------------------------------------------------
# InlineMathNormalizer (архитектурное решение, лог 23:xx): прежняя regex-версия
# давала каскадные вложения ($O($x^{n}$+1)$) и молча пропускала «потерянные
# индексы» без ^/_ («P n(x)», «x 0 = 0», «(x-x 0)»). Новый модуль режет текст
# на КАНДИДАТЫ (сплошные ASCII-прогоны, не пересекающие слова прозы), каждый
# нормализуется детерминированными правилами и проходит SYMPY-валидацию
# (parse_latex + srepr — ловит битый LaTeX, которого regex не видит). Только
# прошедшие валидацию кандидаты оборачиваются в $...$; остальные остаются
# нетронутыми — никогда не выдаём сломанную формулу вместо текста.
# Тот же модуль вызывается на ИНЖЕСТЕ (mineru_parser._add_prose) — формулы
# канонизируются ДО векторизации, и LLM получает из ChromaDB чистый LaTeX.
# ---------------------------------------------------------------------------


class InlineMathNormalizer:
    """Нормализация «голой» инлайн-математики в прозе.

    API:
        normalize(text) -> str        — полный проход (кандидаты + SymPy)
        validate(expr) -> bool|None   — None, если sympy недоступен

    Правила идемпотентны; кандидат обязан содержать скрипт (^/_), оператор
    (= < > ≈ ≤ ≥ ≠) или o-нотацию — иначе это слово прозы и оно не трогается.
    """

    #: известные TeX/KaTeX-команды (символы и функции) — для структурной
    #: проверки кандидатов до обращения к sympy
    _KNOWN_CMDS = {
        "alpha", "beta", "gamma", "delta", "epsilon", "varepsilon", "zeta",
        "eta", "theta", "vartheta", "iota", "kappa", "lambda", "mu", "nu",
        "xi", "pi", "rho", "sigma", "tau", "upsilon", "phi", "varphi",
        "chi", "psi", "omega", "Gamma", "Delta", "Theta", "Lambda", "Xi",
        "Pi", "Sigma", "Upsilon", "Phi", "Psi", "Omega",
        "infty", "partial", "nabla", "forall", "exists", "in", "notin",
        "subset", "subseteq", "supset", "cup", "cap", "emptyset",
        "cdot", "times", "div", "pm", "mp", "leq", "le", "geq", "ge",
        "neq", "ne", "approx", "equiv", "sim", "simeq", "to", "rightarrow",
        "leftarrow", "Rightarrow", "mapsto", "ldots", "cdots", "dots",
        "sum", "prod", "int", "lim", "frac", "sqrt", "text", "mathrm",
        "mathit", "mathbf", "hat", "bar", "vec", "dot", "tilde", "circ",
        "sin", "cos", "tan", "cot", "arcsin", "arccos", "arctan", "log",
        "ln", "exp", "max", "min", "det", "dim", "ker", "deg", "gcd",
        "sup", "inf", "limsup", "liminf", "o", "O",  # o-нотация Кудрявцева
    }

    def __init__(self, use_sympy: bool = True):
        self.use_sympy = use_sympy
        self._sympy: Any = None
        self._cache: Dict[str, Optional[bool]] = {}

    # ---------------- SymPy-валидация ----------------
    def _get_sympy(self):
        if not self.use_sympy:
            return None
        if self._sympy is None:
            try:
                import sympy
                from sympy.parsing.latex import parse_latex
                parse_latex("x^{2}")  # smoke-test: antlr4-python3-runtime>=4.11
                self._sympy = sympy
            except ImportError:
                # нет antlr-рантайма: parse_latex бракует ВСЁ — не валидируем
                self._sympy = False
            except Exception:
                self._sympy = False
        return self._sympy or None

    def _structural_ok(self, s: str) -> bool:
        """Баланс скобок + все \\<cmd> из белого списка KaTeX/MiKTeX-ядра."""
        if s.count("(") != s.count(")") or s.count("{") != s.count("}"):
            return False
        for cmd in re.findall(r"\\([A-Za-z]+)", s):
            if cmd not in self._KNOWN_CMDS:
                return False
        return True

    @staticmethod
    def _tex_to_sympy(expr: str) -> str:
        """KaTeX-валидный текст → синтаксис, понятный sympy.parse_latex.

        \\o/\\O (o-нотация), \\tg/\\ctg не входят в грамматику ANTLR-sympy —
        заменяются на вызовы функций; операторные команды → знаки. Иначе
        валидатор браковал бы корректные формулы (ложные False).
        """
        s = expr
        # «\o (x)»/«\O (x)» → вызов функции o(x)/O(x) для грамматики sympy
        s = re.sub(r"\\([oO])[ \t]+(?=\()", r"\1", s)
        for cmd, rep in ((r"\\tg\b", r"\\tan"), (r"\\ctg\b", r"\\cot"),
                        (r"\\arctg\b", r"\\arctan"), (r"\\arcctg\b", r"\\arccot"),
                        (r"\\lg\b", r"\\log_{10}")):
            s = re.sub(cmd, rep, s)
        for cmd, op in ((r"\\approx", "="), (r"\\le\b", "<="), (r"\\ge\b", ">="),
                        (r"\\neq\b", "!="), (r"\\cdot", "*")):
            s = re.sub(cmd + r"[ \t]*", op + " ", s)
        return s

    def validate(self, expr: str) -> Optional[bool]:
        """True — LaTeX структурно валиден; False — битый; None — нет sympy.

        Первый уровень — структурная проверка (_structural_ok): баланс скобок
        и белый список команд (ловит «\\foo», разорванные «($x$…» — то, что
        regex-валидатор раньше не видел). Второй уровень — sympy.parse_latex
        по частям равенства: части вида «f(x)» (функция от аргумента) легальны
        в TeX, но запрещены грамматикой sympy — они заменяются на X перед
        парсингом. Если antlr4-рантайм не установлен, sympy-уровень
        пропускается (None), но структура всё равно проверяется.
        """
        if not self._structural_ok(expr):
            return False
        sp = self._get_sympy()
        if sp is None:
            return None
        if expr in self._cache:
            return self._cache[expr]
        ok: Optional[bool] = None
        try:
            from sympy.parsing.latex import parse_latex
            parts = [p.strip() for p in re.split(r"[=<>\u2248\u2264\u2265\u2260]", expr)]
            checked_any = False
            for p in parts:
                if not p:
                    continue
                core = re.sub(r"\b[A-Za-z]\((?:[^()]|\([^()]*\))*\)", "X", p)
                if not core.strip():
                    continue
                parse_latex(self._tex_to_sympy(core))
                checked_any = True
            ok = True if checked_any else None
        except Exception:
            ok = False
        self._cache[expr] = ok
        return ok

    #: прозаические слова-функции (не математика даже с цифрой-аргументом:
    #: «n 1 -го порядка», «f 2(x)» — только если это не однобуквенная переменная)
    _PROSE_WORDS = re.compile(
        r"^(?:[a-zA-Z]{2,})$")

    # ---------------- нормализация кандидата ----------------
    def _normalize_candidate(self, cand: str, left: str = "",
                             right: str = "") -> Optional[str]:
        s = cand.strip().strip(".,;:")
        if not s:
            return None
        has_script = bool(re.search(r"[\^_]", s))
        has_op = bool(re.search(r"[=<>\u2248\u2264\u2265\u2260]", s))
        has_o = bool(re.search(r"(^|[\s(/+\-])[oO]\s*\(", s))
        # потерянный индекс/степень в тексте PDF: одиночная переменная + пробел
        # + цифра («x 0», «P n(x)») — маркер математики. Слева обязан быть
        # конец слова прозы (пробел; перед ним не цифра, чтобы «… 2025 x 0»
        # не склеивалось). Справа — конец выражения: разделитель ИЛИ пробел +
        # строчная кириллица (проза Кудрявцева: «… x 0 , то …», «к x 0
        # остаточный …», «многочлен P n(x) степени …»). ЗАПРЕЩЁН хвост с
        # заглавной буквы — это начало нового предложения («f^(k)(x 0). Это
        # равенство…» остаётся сырым, но безопасным). Числительные («n -ю»)
        # отсекаются: за цифрой идёт дефис/буква — не математический хвост.
        _RIGHT_END = r"(?=$|[ \t]*(?:[.,;:!?)\]]|[а-яё]))"
        has_lost_idx = bool(
            (re.fullmatch(r"[A-Za-z][ \t]+[0-9]{1,3}", s)
             or re.fullmatch(r"[A-Za-z][ \t]+[A-Za-z]\([A-Za-z0-9 \t]+\)", s))
            and re.search(r"[ \t]$", left) is not None
            and not re.search(r"[0-9][ \t]$", left)
            and (re.match(_RIGHT_END, right) is not None
                 or right.strip() == "")   # конец строки без знаков — тоже ок
        )
        if not (has_script or has_op or has_o or has_lost_idx):
            return None
        # потерянные индексы (цифровой токен): буква/скобка + пробел + цифра →
        # «x_0», «r_n». Условие конца: конец прогона, разделитель ИЛИ строчная
        # кириллица (в прозе «… x 0 , то существует …» после запятой идёт
        # русское слово — раньше негативный lookahead его не пропускал и
        # индекс терялся). Буквенный индекс («P n(x)») склеивается только если
        # за ним явный скрипт/аргумент либо весь кандидат — паттерн
        # «переменная + индекс» (has_lost_idx выше).
        s = re.sub(r"(?<=[A-Za-z0-9)\]])[ \t]+(?=\d{1,4}(?![A-Za-z0-9])(?=$|[ \t,;)=<>≈≤≥≠]|[А-Яа-яЁё]))",
                   "_", s)
        # потерянные скрипты с явным маркером: «^ (k)» → '^{(k)}', '_ {0}' и т.п.
        s = re.sub(r"([\^_])[ \t]+(?=[({])", r"\1", s)
        # буквенный индекс/показатель ПЕРЕД явным скриптом или скобкой-аргументом:
        # «P n^(k)(x 0)» → «P_n^{(k)}(x_{0})», «r n(x) = ...» → «r_n(x)».
        # Ограничитель в конце (^_|() гарантирует: проза («P n(x) имеет вид»)
        # не трогается — там после индекса идёт кириллица/конец слова без маркера.
        for _ in range(3):  # цепочки «a n m^(k)» обрабатываются за пару проходов
            new = re.sub(r"(?<=[A-Za-z0-9)\]])[ \t]+([A-Za-z])(?=[\^_]|\()", r"_\1", s)
            if new == s:
                break
            s = new
        # одиночная переменная + буквенный индекс с аргументом («P n(x)» →
        # «P_n(x)») — только когда кандидат ВЕСЬ является таким паттерном
        # (has_lost_idx); иначе «не выше n ,»/«f имеет n -ю» не трогаем.
        if has_lost_idx and re.fullmatch(r"[A-Za-z]_[A-Za-z]\([A-Za-z0-9]+\)",
                                         s.replace(" ", "")):
            s = re.sub(r"([A-Za-z])[ \t]+([A-Za-z])(?=\()", r"\1_\2", s)
        # «o (…)», «O (…)» — o-нотация: TeX-команда с пробелом («\o »),
        # KaTeX-совместимо; голые «o(x)»/«O(x^n+1)» без пробела тоже легальны,
        # но нормализуем к единому виду только при явном пробеле.
        s = re.sub(r"(^|(?<=[\s(/+=]))([oO])[ \t]+(?=\()", r"\1\\\2 ", s)
        # unicode-скрипты текстового слоя PDF (x₀, rₙ, x²ⁿ⁺¹)
        s = "".join(_SUB_DIGITS.get(ch, ch) for ch in s)
        s = re.sub("(?<=[A-Za-z0-9)])([" + _SUP_KEYS + "]+)",
                   lambda mm: "^{" + "".join(_SUP_MAP[c] for c in mm.group(1)) + "}",
                   s)
        # скрипты в канон: ^(k)→^{(k)}, ^{n}→^{n}, ^n→^{n}, _k→_{k}
        s = re.sub(r"\^\s*\(([^()]{1,10})\)", r"^{\1}", s)
        s = re.sub(r"\^\s*\{([^{}]{1,12})\}", r"^{\1}", s)
        # многосимвольный «голый» показатель: «^2n+2», «^n+1» — жадно до
        # границы выражения. Раньше матчался один символ → «x^{2}n+2» вместо
        # «x^{2n+2}» (риск неверной формулы из аудита: O(x^n+1) vs O(x^{n+1})).
        s = re.sub(r"\^\s*([A-Za-z0-9](?:[A-Za-z0-9+\-*/][A-Za-z0-9]*)*)",
                   lambda mm: "^{" + mm.group(1) + "}", s)
        s = re.sub(r"_\s*\(([^()]{1,10})\)", r"_{(\1)}", s)
        s = re.sub(r"_(?!\{)([A-Za-z0-9])", r"_{\1}", s)
        # функции/операторы → TeX
        s = _RAW_MATH_FUNC_RE.sub(chr(92) + r"\1", s)
        s = s.replace("\u00b7", "\\cdot ")
        for bad, good in (("\u2248", "\\approx "), ("\u2264", "\\le "),
                          ("\u2265", "\\ge "), ("\u2260", "\\neq ")):
            s = s.replace(bad, good)
        s = re.sub(r"\s+", " ", s).strip()
        if not s:
            return None
        if s.count("(") != s.count(")") or s.count("{") != s.count("}"):
            return None
        # одинокая переменная без оператора/скрипта — шум (например «Pn»)
        if len(s) < 3:
            return None
        return s

    # ---------------- основной проход ----------------
    # Кандидат = сплошной ASCII-прогон, начинающийся с буквы/цифры/«$» и т.д.,
    # НЕ пересекающий слова прозы (границы — кириллица/пробел-слово). Стартовая
    # группа допускает открывающую скобку, чтобы «(x+a)^1 = x + a» захватывался
    # целиком (прежний вариант стартовал с буквы и резал прогон на «x+a)^1…» —
    # разбаланс отбрасывал кандидат, формула оставалась сырой).
    _CAND_RE = re.compile(
        r"(?<![А-Яа-яЁёA-Za-z0-9_$\\])"
        r"(?:[A-Za-z0-9]|[(\[])((?:[A-Za-z0-9 \t^_(){}\[\]/+\u00b7*=<>~"
        r"\u2248\u2264\u2265\u2260,\-.])*[A-Za-z0-9)\]])?"
        r"(?![А-Яа-яЁёA-Za-z0-9_])"
    )

    def normalize(self, text: str) -> str:
        if not text:
            return text
        blocks: List[str] = []

        def _save(m: re.Match) -> str:
            blocks.append(m.group(0))
            return "\x00B%d\x00" % (len(blocks) - 1)

        text = re.sub(r"\$\$.+?\$\$", _save, text, flags=re.DOTALL)
        text = re.sub(r"(?<!\\)\$[^$\n]+?\$", _save, text)
        text = re.sub(r"`[^`\n]+?`", _save, text)

        def _repl(m: re.Match, line: str) -> str:
            raw = m.group(0)
            cand = self._normalize_candidate(raw, left=line[:m.start()],
                                             right=line[m.end():])
            if cand is None:
                return raw
            if self.validate(cand) is False:
                return raw
            # после канона обязан остаться математический маркер
            if not re.search(r"[\^_=\\]", cand):
                return raw
            return "$" + cand + "$"

        out_lines: List[str] = []
        for line in text.split("\n"):
            if line.strip().startswith("$$") or "```" in line:
                out_lines.append(line)
                continue
            out_lines.append(self._CAND_RE.sub(lambda mm: _repl(mm, line), line))
        text = "\n".join(out_lines)
        return re.sub(r"\x00B(\d+)\x00", lambda m: blocks[int(m.group(1))], text)


_DEFAULT_NORMALIZER: Optional[InlineMathNormalizer] = None


def get_inline_math_normalizer(use_sympy: bool = True) -> InlineMathNormalizer:
    global _DEFAULT_NORMALIZER
    if _DEFAULT_NORMALIZER is None:
        _DEFAULT_NORMALIZER = InlineMathNormalizer(use_sympy=use_sympy)
    return _DEFAULT_NORMALIZER


def normalize_bare_math_prose(text: str) -> str:
    """Совместимость: прежняя точка входа делегирует InlineMathNormalizer.

    Вызывается из sanitize_llm_answer (уровень ответа LLM) и из
    mineru_parser._add_prose (уровень импорта книги — ДО чанкинга и
    векторизации). Идемпотентен; никогда не выдаёт сломанную формулу.
    """
    try:
        return get_inline_math_normalizer().normalize(text)
    except Exception as ex:  # косметика не валит импорт/ответ
        logger.warning(f"normalize_bare_math_prose: пропуск ({ex})")
        return text



def fix_llm_bold_spacing(text: str) -> str:
    """Чинит broken bold: '** текст **' -> '**текст**' (CommonMark требует
    непробельный символ сразу после ** и перед ним). Только парные маркеры
    в пределах одной строки; одиночные ** (например в $$-прозе) не трогает."""
    if "**" not in text:
        return text

    def _fix_line(line: str) -> str:
        if line.count("**") % 2:
            return line  # нечётное число — не угадываем пары

        def _repl(m: re.Match) -> str:
            body = m.group(1).strip()
            return f"**{body}**" if body else ""

        # '** текст **' -> '**текст**'; '** **' (пустое содержимое) -> удаляем
        return re.sub(r"\*\*\s*([^*]*?)\s*\*\*", _repl, line)

    return "\n".join(_fix_line(l) for l in text.split("\n"))


def drop_empty_list_items(text: str) -> str:
    """Удаляет «пустые» строки-маркеры списков, где модель потеряла содержимое:
      1. маркер без текста вовсе: '• ', '- ', '* ' (только пробелы после);
      2. висячее тире/дефис после маркера ('-  — ', '- - ') с последующей
         пустотой до конца строки (потеряна переменная).
    Нормальные пункты ('-  — коэффициент при...' = текст после тире),
    строки «— текст» и markdown-таблицы НЕ трогает."""
    prev = None
    while prev != text:
        prev = text
        text = re.sub(
            r"(?m)^[ \t]*[•\-\*](?:[ \t]+|[ \t]*(?:[\u2014\u2013]|-)[ \t]*)+$",
            "", text)
    # схлопнуть серию пустых строк, оставшихся после вырезания, до одной
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def sanitize_llm_answer(text: str) -> str:
    """Единая постобработка ответа LLM (вызывается из RAGPipeline.answer):
    1) снятие косых слэшей перед ^/_ вне $$-блоков;
    2) ремонт broken bold;
    3) чистка пустых маркеров списков.
    Идемпотентна, безопасна для KaTeX/CommonMark фронтенда."""
    if not text:
        return text
    try:
        text = fix_llm_math_artifacts(text)
        # FIX (InlineMathNormalizer): раньше нормализация «голой» математики
        # была написана, но НИКОГДА не вызывалась из sanitize_llm_answer —
        # поэтому импорт с корректными формулами всё равно давал сырой текст
        # в ответе, когда модель копировала плоский текст чанка. Теперь уровень
        # «ответ LLM» закрыт; идемпотентность проверяется тестом.
        text = normalize_bare_math_prose(text)
        text = fix_llm_bold_spacing(text)
        text = drop_empty_list_items(text)
    except Exception as ex:  # косметика никогда не должна ронять ответ
        logger.warning(f"sanitize_llm_answer: пропуск ({ex})")
    # финальная чистка хвостовых переводов строк — гарантирует идемпотентность
    # всего пути (_strip_code_fences -> math_guard -> sanitize_llm_answer):
    # повторный strip() на уже «вымытом» тексте ничего не меняет.
    return text.strip()


def normalize_metadata(meta: Any) -> Dict[str, Any]:
    """Приводит метаданные чанка/узла графа к dict (гарантия типа).

    FIX ('str' object has no attribute 'items' / ...'get'): один и тот же
    блок живёт в двух хранилищах с разной сериализацией metadata:
      * SQLite FTS5 / graph-узлы — JSON-строка;
      * ChromaDB — dict, НО вложенные dict/list значения сериализуются
        sanitize_metadata() обратно в JSON-строки на уровне полей.
    Любой потребитель (prompt_builder._format_source, RRF-слияние,
    _extract_sources, UI-чипы источников) обязан видеть только dict.
    Мусор (скаляр, битый JSON, None) превращается в {} вместо падения
    AttributeError где-то в UI-потоке.

    Потокобезопасно: только локальные вычисления, без общего состояния.
    """
    if isinstance(meta, dict):
        return meta
    if isinstance(meta, (bytes, bytearray)):
        meta = meta.decode("utf-8", errors="ignore")
    if isinstance(meta, str):
        s = meta.strip()
        if s.startswith("{") and s.endswith("}"):
            try:
                parsed = json.loads(s)
                if isinstance(parsed, dict):
                    return parsed
            except (ValueError, TypeError):
                pass
        # Строка не похожа на JSON — теряем поля, но не роняем конвейер.
        return {}
    return {}


# ---------------------------------------------------------------------------
# 4. Уровень 1+2 архитектуры (универсальная техлитература): сырой LaTeX-код
#    не должен попадать ни в UI, ни в векторный контекст. Инлайн-формы
#    \\( ... \\), \\[ ... \\] и одиночные $...$ переводятся в чистый Unicode;
#    тяжёлые display-блоки $$...$$ сохраняются как есть (для PNG-рендера).
#    Таблица команд — общее знание проекта: её использует и chat_view
#    (препроцессор Flet), поэтому импортируется оттуда напрямую при наличии.
# ---------------------------------------------------------------------------
_LATEX_TO_UNICODE = {
    r"\notin": "\u2209", r"\not\in": "\u2209", r"\in": "\u2208",
    r"\forall": "\u2200", r"\exists": "\u2203", r"\nexists": "\u2204",
    r"\dots": "\u2026", r"\ldots": "\u2026", r"\cdots": "\u2026",
    r"\emptyset": "\u2205", r"\varnothing": "\u2205",
    r"\subseteq": "\u2286", r"\supseteq": "\u2287",
    r"\subset": "\u2282", r"\supset": "\u2283",
    r"\cup": "\u222a", r"\cap": "\u2229", r"\setminus": "\u2216",
    r"\land": "\u2227", r"\lor": "\u2228", r"\neg": "\u00ac",
    r"\implies": "\u21d2", r"\iff": "\u21d4", r"\to": "\u2192",
    r"\rightarrow": "\u2192", r"\leq": "\u2264", r"\geq": "\u2265",
    r"\neq": "\u2260", r"\approx": "\u2248", r"\sum": "\u2211",
    r"\prod": "\u220f", r"\int": "\u222b", r"\infty": "\u221e",
    r"\partial": "\u2202", r"\nabla": "\u2207", r"\cdot": "\u00b7",
    r"\times": "\u00d7", r"\div": "\u00f7", r"\pm": "\u00b1",
    r"\mathbb{R}": "\u211d", r"\mathbb{N}": "\u2115", r"\mathbb{Z}": "\u2124",
    r"\mathbb{Q}": "\u211a", r"\mathbb{C}": "\u2102",
    r"\dot{O}": "O\u0307", r"\dot{U}": "U\u0307", r"\lim": "lim",
    # FIX (греческие буквы): \Sigma/\sigma и компания раньше НЕ имели глифов
    # в таблице — конвертер снимал с них бэкслеш оптом, и ft.Markdown/KaTeX
    # получали «Sigma» вместо формулы. Добавлены все греческие буквы.
    r"\alpha": "\u03b1", r"\beta": "\u03b2", r"\gamma": "\u03b3", r"\delta": "\u03b4",
    r"\varepsilon": "\u03b5", r"\epsilon": "\u03b5", r"\zeta": "\u03b6", r"\eta": "\u03b7",
    r"\theta": "\u03b8", r"\vartheta": "\u03d1", r"\iota": "\u03b9", r"\kappa": "\u03ba",
    r"\lambda": "\u03bb", r"\mu": "\u03bc", r"\nu": "\u03bd", r"\xi": "\u03be",
    r"\pi": "\u03c0", r"\varpi": "\u03d6", r"\rho": "\u03c1", r"\varrho": "\u03f1",
    r"\varsigma": "\u03c2", r"\sigma": "\u03c3", r"\tau": "\u03c4", r"\upsilon": "\u03c5",
    r"\phi": "\u03c6", r"\varphi": "\u03d5", r"\chi": "\u03c7", r"\psi": "\u03c8",
    r"\omega": "\u03c9",
    r"\Gamma": "\u0393", r"\Delta": "\u0394", r"\Theta": "\u0398", r"\Lambda": "\u039b",
    r"\Xi": "\u039e", r"\Pi": "\u03a0", r"\Sigma": "\u03a3", r"\Upsilon": "\u03a5",
    r"\Phi": "\u03a6", r"\Psi": "\u03a8", r"\Omega": "\u03a9",
}
_SORTED_LATEX_KEYS = sorted(_LATEX_TO_UNICODE, key=len, reverse=True)

# Команды, которые НИКОГДА не «оголяются» (без таблицы Unicode-замены они
# остаются валидными LaTeX для KaTeX: пробелы набора, акценты, окружения).
_NO_STRIP_CMDS = frozenset({
    "quad", "qquad", "limits", "displaystyle", "textstyle",
    "hat", "vec", "dot", "ddot", "tilde", "bar", "widehat", "overline",
    "frac", "dfrac", "tfrac", "sqrt", "text", "operatorname", "left", "right",
})

_RE_TEX_DELIMS = re.compile(r"\\\[([^\[\]]*?)\\\]|\\\((.+?)\\\)", re.S)
_RE_INLINE_DOLLAR = re.compile(r"(?<!\\)\$(?!\$)([^$\n]+?)\$")


# FIX «quad x ->x 0» / «\Sigma без бэкслеша»: раньше конвейер канонизировал
# $..$/$$..$$ в LaTeX, а затем этот же конвертер СНИМАЛ бэкслеши со всех
# команд (\quad -> quad, \Sigma -> Sigma, \to -> to). ft.Markdown/KaTeX
# такие «оголённые» команды не распознаёт — пользователь видел сырой текст.
# Команды из таблицы LaTeX->Unicode заменяются глифами; неизвестные и все
# канонические LaTeX-команды сохраняются как есть (валидный KaTeX).
_KEEP_LATEX_CMDS = frozenset({
    r"\quad", r"\qquad", r"\,", r"\:", r"\;", r"\ ", r"\!", r"\>", r"\.",
    r"\limits", r"\displaystyle", r"\textstyle",
})


# FIX (баг «прозы»: дублирование предложений и каша после aggressive-шага):
# Ранее clean_text() принудительно разворачивал ВЕСЬ инлайн-LaTeX ($...$,
# \(...\)) в Unicode прямо на этапе ingestion. Это ломало связность текста:
#   * \frac{1}{2} превращалось в «1/2», дроби теряли структуру, а слитые
#     осколки («frac 12 · CH») модель затем цитировала дословно;
#   * замены таблицей вставляли лишние пробелы внутри предложений;
#   * последующие прогоны конвейера (chunker+prompt_builder) применяли
#     конвертацию ПОВТОРНО к уже искалеченному тексту — отсюда зеркальные
#     дубли абзацев в ответах.
# Новая стратегия: в БАЗЕ хранится канонический KaTeX ($...$/$$...$$) —
# ровно тот вид, который MinerU выдаёт и который понимают ft.Markdown/KaTeX,
# bge-m3 и LLM. Принудительная латекс→Unicode конвертация больше НИКОГДА не
# вызывается из clean_text(); она доступна отдельно (флаг convert_inline_latex)
# только для тех редких потребителей, кому нужен plain-Unicode.


def _latex_body_to_unicode(latex: str) -> str:
    """Тело LaTeX-формулы → читаемый Unicode (для инлайновых форм).

    Сохраняет обратные косые черты: снимаются только те бэкслеши, которым в
    таблице замены нет Юникод-глифа, и то поштучно (\\alpha x -> α x), а не
    «оптом» со всей строки.

    ВЫЗЫВАЕТСЯ ТОЛЬКО явно с convert_inline_latex=True (см. clean_text);
    в основном конвейере не участвует — база хранит канонический KaTeX.
    """
    s = latex.strip()
    for cmd in _SORTED_LATEX_KEYS:
        if cmd in s:
            s = s.replace(cmd, " " + _LATEX_TO_UNICODE[cmd] + " ")
    # Остаточные команды: известные глифы уже заменены выше; здесь меняем
    # только полные токены, чьи имена есть в словаре, и оставляем прочие
    # (\mathbb{...} целиком не матчится одиночным именем — оно остаётся).
    def _keep_or_map(m: re.Match) -> str:
        tok = m.group(0)
        name = tok[1:]
        if tok in _KEEP_LATEX_CMDS or name in _NO_STRIP_CMDS:
            return tok
        # Известные команды уже заменены таблицей выше; если имя НЕ в
        # словаре и не в списке сохраняемых — оставляем бэкслеш (пусть
        # KaTeX сам решит), «оголение» было причиной «Sigma/quad».
        return tok
    s = re.sub(r"\\[A-Za-z]+", _keep_or_map, s)
    # Одиночные управляющие символы без имени команды (\ , \{, \}) сохраняем.
    s = s.replace("\\\u00a0", "\\ ")
    # Двойные пробелы после замен схлопываем, {} для инлайна убираем только
    # вокруг одиночных символов (\dot{O} оставим как есть).
    s = re.sub(r"\{([A-Za-z0-9])\}", r"\1", s)
    s = s.replace("_", "\u2009")
    return re.sub(r"(?<!\\)[ \t]{2,}", " ", s).strip()


def latex_inline_to_unicode(text: str) -> str:
    """Инлайн-математику \\(..\\), \\[..\\], $..$ переводит в чистый Unicode.

    Display-блоки $$...$$ (тяжёлые формулы для PNG-рендера) не трогает.
    Идемпотентна: повторно вызванная на уже очищенном тексте ничего не меняет.

    DEPRECATED для основного конвейера: сохранена для совместимости с
    внешними вызовами и флага convert_inline_latex; внутри clean_text()
    по умолчанию НЕ вызывается.
    """
    if not text or "\\" not in text and "$" not in text:
        return text
    # защита $$...$$ от разрезания одинарными $
    blocks: list = []

    def _save(m):
        blocks.append(m.group(0))
        return "\x00BLK%d\x00" % (len(blocks) - 1)

    text = re.sub(r"\$\$(.+?)\$\$", _save, text, flags=re.DOTALL)
    text = _RE_TEX_DELIMS.sub(
        lambda m: " " + _latex_body_to_unicode(m.group(1) or m.group(2) or "") + " ",
        text)
    text = _RE_INLINE_DOLLAR.sub(
        lambda m: " " + _latex_body_to_unicode(m.group(1)) + " ", text)
    text = text.replace("\\$", "$")
    text = re.sub(r"\x00BLK(\d+)\x00", lambda m: blocks[int(m.group(1))], text)
    return re.sub(r"[ \t]{2,}", " ", text)


def clean_text(text: str, convert_inline_latex: bool = False) -> str:
    """Полный конвейер препроцессора (вызывается из TextCorrector/Chunker).

    ПОРЯДОК ВАЖЕН (устранение рассогласования форматов Ingestion/Чанкинг/LLM/UI):
      1. normalize_math_delimiters — ВСЕ ограничители к канону KaTeX
         ($...$ / $$...$$), который понимает ft.Markdown Flet 1.x;
      2. fix_punctured_neighborhood + normalize_punctured_variants —
         коррекция OCR-шума проколотых окрестностей;
      3. balance_math_delimiters — гарантия ЧЁТНОСТИ маркеров: «огрызок»
         открытого $$ после чанкинга ронял KaTeX (throwOnError) и весь
         ответ отображался сырым текстом.

    FIX («проза»): принудительная конвертация инлайн-LaTeX в Unicode из
    конвейера УДАЛЕНА (была шагом 3 раньше). Причина багов: разворот
    \\frac{1}{2} → «1/2», \\to → «→» и т.п. на этапе ingestion ломал
    связность предложений, вставлял лишние пробелы, а повторные прогоны
    конвейера калечили текст до зеркальных дублей. MinerU отдаёт каноничный
    KaTeX — именно его и нужно хранить в базе: его понимают ft.Markdown/KaTeX
    в UI, bge-m3 при векторизации и LLM в контексте. Тяжёлый рендер
    формул — ответственность UI-слоя (chat_view/mathtext/MathML), а не базы.

    :param convert_inline_latex: ЯВНЫЙ opt-in старой агрессивной конвертации
        $..$/\\(..\\) → Unicode. По умолчанию False. Нужен лишь отдельным
        экспортёрам (txt/plain); в основном пути не использовать.
    """
    if not text:
        return text
    text = normalize_math_delimiters(text)
    text = fix_punctured_neighborhood(text)
    text = normalize_punctured_variants(text)
    if convert_inline_latex:
        text = latex_inline_to_unicode(text)
    text = balance_math_delimiters(text)
    return text
