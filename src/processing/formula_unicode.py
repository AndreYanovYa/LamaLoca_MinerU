# -*- coding: utf-8 -*-
"""rule-based конвертер Unicode-математики -> KaTeX-совместимый LaTeX.

Зачем (производительность + качество):
  Текстовый слой PDF Кудрявцева и подобных учебников содержит формулы
  корректными Unicode-глифами (∀ ∃ ∈ → ⇒ ℝ ∑ ∫ ≤ ⋅ ′ ² ₙ). Раньше каждая
  такая зона отправлялась в нейросеть (pix2tex/texify) — тысячи
  последовательных инференсов на сотнях страниц, GPU/CI простаивают или
  задыхаются. Здесь формула собирается ПРАВИЛАМИ за микросекунды:
  качество не ниже сетевого (сеть на печатном Unicode-слое делает те же
  символы, только медленнее и с галлюцинациями), а сеть остаётся для
  настоящего растра (сканы, рукописный ввод, сплющенная графика).

Направление обратное к latex_unicode.py (LaTeX -> Unicode для UI);
символьные таблицы согласованы с ним.

API:
    convert_unicode_math(text) -> Optional[str]
        Валидный LaTeX без ограничителей $, либо None (правила неприменимы —
        зону должен обрабатывать нейродвижок как раньше).
"""
from __future__ import annotations

import re
import unicodedata

# ----------------------------------------------------------------------
# Словари замены. Порядок применения важен: сначала «длинные» конструкции
# (интегралы с пределами, дроби a⁄b), затем глифы по одному символу.
# ----------------------------------------------------------------------

GREEK = {
    "α": r"\alpha", "β": r"\beta", "γ": r"\gamma", "δ": r"\delta",
    "ε": r"\varepsilon", "ζ": r"\zeta", "η": r"\eta", "θ": r"\theta",
    "ϑ": r"\vartheta", "ι": r"\iota", "κ": r"\kappa", "λ": r"\lambda",
    "μ": r"\mu", "ν": r"\nu", "ξ": r"\xi", "π": r"\pi", "ϖ": r"\varpi",
    "ρ": r"\rho", "ς": r"\sigma", "σ": r"\sigma", "τ": r"\tau",
    "υ": r"\upsilon", "φ": r"\phi", "ϕ": r"\varphi", "χ": r"\chi",
    "ψ": r"\psi", "ω": r"\omega",
    "Γ": r"\Gamma", "Δ": r"\Delta", "Θ": r"\Theta", "Λ": r"\Lambda",
    "Ξ": r"\Xi", "Π": r"\Pi", "Σ": r"\Sigma", "Υ": r"\Upsilon",
    "Φ": r"\Phi", "Ψ": r"\Psi", "Ω": r"\Omega",
}

# Математические «полужирные/рукописные» буквы из шрифтов PDF -> обычный
# курсив LaTeX (модель набора не меняет, только представление).
BOLDSCRIPT = {
    "𝐀": "A", "𝐁": "B", "𝐂": "C", "𝐃": "D", "𝐄": "E", "𝐅": "F",
    "𝐆": "G", "𝐇": "H", "𝐈": "I", "𝐉": "J", "𝐊": "K", "𝐋": "L",
    "𝐌": "M", "𝐍": "N", "𝐎": "O", "𝐏": "P", "𝐐": "Q", "𝐑": "R",
    "𝐒": "S", "𝐓": "T", "𝐔": "U", "𝐕": "V", "𝐖": "W", "𝐗": "X",
    "𝐘": "Y", "𝐙": "Z",
    "𝑎": "a", "𝑏": "b", "𝑐": "c", "𝑑": "d", "𝑒": "e", "𝑓": "f",
    "𝑔": "g", "ℎ": "h", "𝑖": "i", "𝑗": "j", "𝑘": "k", "𝑙": "l",
    "𝑚": "m", "𝑛": "n", "𝑜": "o", "𝑝": "p", "𝑞": "q", "𝑟": "r",
    "𝑠": "s", "𝑡": "t", "𝑢": "u", "𝑣": "v", "𝑤": "w", "𝑥": "x",
    "𝑦": "y", "𝑧": "z",
    "ℝ": r"\mathbb{R}", "ℕ": r"\mathbb{N}", "ℤ": r"\mathbb{Z}",
    "ℚ": r"\mathbb{Q}", "ℂ": r"\mathbb{C}", "ℍ": r"\mathbb{H}",
    "ℵ": r"\aleph", "ℓ": r"\ell", "ℏ": r"\hbar", "∂": r"\partial",
    "∇": r"\nabla", "∅": r"\emptyset", "⌀": r"\varnothing",
}

OPERATORS = {
    # отношения
    "≤": r"\leq", "⩽": r"\leqslant", "≥": r"\geq", "⩾": r"\geqslant",
    "≠": r"\neq", "≈": r"\approx",
    "≡": r"\equiv", "∼": r"\sim", "≃": r"\simeq", "≅": r"\cong",
    "≺": r"\prec", "≻": r"\succ", "⪯": r"\preceq", "⪰": r"\succeq",
    "≪": r"\ll", "≫": r"\gg", "∝": r"\propto", "≜": r":=",
    # стрелки
    "→": r"\to", "←": r"\leftarrow", "↔": r"\leftrightarrow",
    "⇒": r"\Rightarrow", "⟹": r"\Longrightarrow", "⇐": r"\Leftarrow",
    "⇔": r"\iff", "⟺": r"\Longleftrightarrow", "↦": r"\mapsto",
    "↑": r"\uparrow", "↓": r"\downarrow",
    # логика/множества
    "∀": r"\forall", "∃": r"\exists", "∄": r"\nexists",
    "∈": r"\in", "∉": r"\notin", "∋": r"\ni",
    "⊂": r"\subset", "⊃": r"\supset", "⊆": r"\subseteq",
    "⊇": r"\supseteq", "∪": r"\cup", "∩": r"\cap", "∖": r"\setminus",
    "∧": r"\land", "∨": r"\lor", "¬": r"\neg", "⊕": r"\oplus",
    "⊗": r"\otimes", "⊥": r"\perp",
    # бинарные операции
    "×": r"\times", "÷": r"\div", "±": r"\pm", "∓": r"\mp",
    "⋅": r"\cdot", "∙": r"\bullet", "∘": r"\circ",
    # крупные операторы
    "∑": r"\sum", "∏": r"\prod", "∫": r"\int", "∬": r"\iint",
    "∭": r"\iiint", "∮": r"\oint", "⋃": r"\bigcup", "⋂": r"\bigcap",
    # разное
    "∞": r"\infty", "′": "'", "″": "''", "‴": "'''",
    "…": r"\dots", "⋯": r"\cdots", "⋮": r"\vdots",
    "°": r"^\circ", "−": "-", "–": "-", "—": "-",
    "⁄": "/", "½": r"\tfrac{1}{2}", "⅓": r"\tfrac{1}{3}",
    "⅔": r"\tfrac{2}{3}", "¼": r"\tfrac{1}{4}", "¾": r"\tfrac{3}{4}",
    # FIX (потеря бэкслешей, «quad x ->x 0»): в PDF-слое Кудрявцева пробелы
    # набора кодируются глифами QUAD/TWO EM SPACE. Без маппинга они оставались
    # в LaTeX как обычные U+2001/U+2005 и KaTeX/Flet их «съедал», а UI-конвертер
    # снимал обратную косую черту — команды терялись целиком.
    "\u2003": r"\quad",  # EM SPACE
    "\u2001": r"\quad",  # EM QUAD
    "\u2005": r"\;",     # FOUR-PER-EM SPACE (TeX \;)
    "\u2004": r"\:",     # THIRD-PER-EM SPACE (TeX \:)
    "\u2002": r"\ ",     # EN SPACE (TeX \ )
    "\u2009": r"\,",     # THIN SPACE (TeX \,)
}

SUPERSCRIPTS = {
    "⁰": "0", "¹": "1", "²": "2", "³": "3", "⁴": "4", "⁵": "5",
    "⁶": "6", "⁷": "7", "⁸": "8", "⁹": "9", "⁺": "+", "⁻": "-",
    "⁼": "=", "⁽": "(", "⁾": ")", "ⁿ": "n", "ⁱ": "i", "ᵀ": "T",
    "ᵗ": "t", "ᵏ": "k", "ˡ": "l", "ᵐ": "m", "ˢ": "s", "ʳ": "r",
}
SUBSCRIPTS = {
    "₀": "0", "₁": "1", "₂": "2", "₃": "3", "₄": "4", "₅": "5",
    "₆": "6", "₇": "7", "₈": "8", "₉": "9", "₊": "+", "₋": "-",
    "₌": "=", "₍": "(", "₎": ")", "ₐ": "a", "ₑ": "e", "ₓ": "x",
    "ₙ": "n", "ₘ": "m", "ₚ": "p", "ₛ": "s", "ₜ": "t", "ᵢ": "i",
    "ⱼ": "j", "ₖ": "k", "ₗ": "l", "ᵣ": "r",
}

# Глифы, наличие которых вообще оправдывает попытку конвертации.
_MATHY_CHARS = set(GREEK) | set(BOLDSCRIPT) | set(OPERATORS) \
    | set(SUPERSCRIPTS) | set(SUBSCRIPTS) \
    | set("∀∃∈∉⊂⊆∪∩∑∏∫√≤≥≠≈⇒⇔→↦∞∂∇′‵⋅±×÷")

_RE_INVIS = re.compile(r"[\u200b\u200c\u200d\ufeff]")
_RE_WS = re.compile(r"[ \t]{2,}")


def _group_scripts(text: str) -> str:
    """Цепочки надстрочных/подстрочных -> ^{...}_{...} c группировкой."""
    # надстрочные цепочки
    def repl_sup(m):
        seq = m.group(0)
        inner = "".join(SUPERSCRIPTS.get(ch, ch) for ch in seq)
        return "^{" + inner + "}"
    text = re.sub(r"(?:[⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾ⁿⁱᵀᵗᵏˡᵐˢʳ])+", repl_sup, text)

    def repl_sub(m):
        seq = m.group(0)
        inner = "".join(SUBSCRIPTS.get(ch, ch) for ch in seq)
        return "_{" + inner + "}"
    text = re.sub(r"(?:[₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎ₐₑₓₙₘₚₛₜᵢⱼₖₗᵣ])+", repl_sub, text)
    return text


def _split_frac_slash(text: str) -> str:
    """a⁄b уже замапится в a/b; обычную '/' не трогаем (KaTeX ок)."""
    return text


def _insert_command_separators(text: str) -> str:
    """KaTeX читает команды жадно: '\\alpha x' слепится в несуществующую
    '\\alphax'. Разделитель вставляется только там, где за командой реально
    идёт буква/цифра (одиночные \\to, \\in, \\infty перед 'x', 'R' и т.п.).

    Реализация без lookahead-regex: в этой среде re движок возвращает для
    r'(\\[A-Za-z]+)(?=[A-Za-z0-9])' НЕжадный матч ('\\exist' вместо
    '\\exists') — эксплицитный скан надёжнее."""
    out = []
    i, n = 0, len(text)
    while i < n:
        if text[i] == "\\":
            j = i + 1
            while j < n and text[j].isalpha():
                j += 1
            out.append(text[i:j])
            if j < n and (text[j].isalnum() or text[j] in "{(["):
                out.append(" ")
            i = j
        else:
            out.append(text[i])
            i += 1
    return "".join(out)


def _tidy(text: str) -> str:
    text = _RE_INVIS.sub("", text)
    text = text.replace("\u00a0", " ")
    # двоеточие после ∀…/∃… блоков: ": "
    text = re.sub(r"\s*:\s*", ": ", text)
    text = _RE_WS.sub(" ", text)
    return text.strip()


def looks_mathy(text: str) -> bool:
    """Достаточно ли в строке математики, чтобы пробовать правила."""
    hits = sum(1 for ch in text if ch in _MATHY_CHARS)
    letters = sum(1 for ch in text if ch.isalpha() and ord(ch) < 128)
    words = len(re.findall(r"[A-Za-zА-Яа-яЁё]{3,}", text))
    if hits == 0:
        return False
    # проза с одним «∈» или «→» не считается формулой
    if words >= 4 and hits < max(3, words // 2):
        return False
    return hits >= 2 or (hits >= 1 and letters <= 12)


def convert_unicode_math(raw: str) -> str | None:
    """Unicode-формула из текстового слоя -> LaTeX (без $). None если
    правила неприменимы/результат сомнительный."""
    if not raw or not raw.strip():
        return None
    text = raw.replace("\r\n", "\n").replace("\r", "\n")
    # PyMuPDF рвёт встроенные формулы переводами строк внутри блока
    text = text.replace("\n", " ")
    if not looks_mathy(text):
        return None

    out = text
    # порядок: скрипты -> глифы (чтобы ⁻¹ не съедалось операторной картой)
    out = _group_scripts(out)
    table = {}
    table.update(OPERATORS)
    table.update(GREEK)
    table.update(BOLDSCRIPT)
    res = []
    i = 0
    n = len(out)
    while i < n:
        ch = out[i]
        rep = table.get(ch)
        if rep is not None:
            res.append(rep)
            # Разделитель в точке подстановки — пока следующий символ ещё
            # Unicode (буква/цифра/скобка). Для глифов-одиночек это надёжно;
            # финальную расщепку слипшихся команд делает _insert_... ниже.
            if rep.startswith("\\") and i + 1 < n and (
                    out[i + 1].isalnum() or out[i + 1] in "{(["):
                res.append(" ")
        else:
            res.append(ch)
        i += 1
    out = "".join(res)

    out = _insert_command_separators(out)
    # sqrt-глиф
    out = out.replace("√", r"\sqrt ")
    # lim_{x→0} уже собран правилами, если был written as lim subscript
    out = _tidy(out)

    # защита: результат обязан быть короче разумного и без «голых» unicode-матемаркеров
    leftover = [ch for ch in out if ch in _MATHY_CHARS and ord(ch) > 127]
    if leftover:
        return None
    if len(out) < 2 or len(out) > 2000:
        return None
    # должен остаться хоть какой-то математический смысл: команда или оператор
    if not re.search(r"\\[A-Za-z]+|[=<>^_{}]", out):
        return None
    return out
