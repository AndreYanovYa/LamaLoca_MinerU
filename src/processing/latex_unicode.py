# src/processing/latex_unicode.py
"""
REFACTORING (ТЗ «AMS-LaTeX без сбоев»): LaTeX → UnicodeMath для mathtext.

matplotlib mathtext НЕ поддерживает AMS-расширения (\\implies, \\leqslant,
\\iiint, \\begin{cases}...) и не читает MathML. Зато он понимает UnicodeMath —
линейную нотацию из юникод-символов (⟹ ⩽ ∭ ∀ ε). Конвертер ниже переводит
LaTeX в эту нотацию: сначала через latex2mathml (полная AMS-таблица символов),
затем MathML → plain-юникод; надстройкой служат правила frac/sqrt/subsup.

Модуль используется экспортером PDF (картинки формул) как основной путь и
как текстовый фолбэк в UI при недоступности рендера.
"""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET

try:
    from src.processing.latex_mathml import get_mathml_converter
except ImportError:  # pragma: no cover
    get_mathml_converter = None

# Невидимые функциональные символы (U+2061 FUNCTION APPLICATION и т.п.):
# latex2mathml вставляет их после \sin/\lim, а шрифты mathtext их не имеют —
# вместо них появляется «dummy» квадратик. Вырезаем при конвертации.
_INVISIBLE_RE = re.compile(r"[\u2061\u2062\u2063\u2064\ufeff]")


def _mathml_to_unicode(mathml: str) -> str:
    """Извлекает текстовое содержимое MathML (<mo>/<mi>/... → символы)."""
    try:
        root = ET.fromstring(mathml)
    except ET.ParseError:
        return ""
    def render(node) -> str:
        tag = node.tag.split("}")[-1]
        kids = list(node)
        if tag in ("mo", "mi", "mn", "mtext"):
            return (node.text or "").replace("\u00a0", " ")
        if tag == "mspace":
            return " "
        if tag == "mfrac" and len(kids) >= 2:
            return "(" + render(kids[0]) + ")/(" + render(kids[1]) + ")"
        if tag == "msqrt" and kids:
            inner = "".join(render(k) for k in kids)
            return "√(" + inner + ")"
        if tag == "mroot" and len(kids) >= 2:
            base = render(kids[0])
            deg = render(kids[1])
            return f"root({deg})({base})"
        if tag == "msup" and len(kids) >= 2:
            return render(kids[0]) + "^(" + render(kids[1]) + ")"
        if tag == "msub" and len(kids) >= 2:
            return render(kids[0]) + "_(" + render(kids[1]) + ")"
        if tag in ("msubsup", "munderover", "moverset", "munder") and len(kids) >= 3:
            base = render(kids[0])
            sub = render(kids[1])
            sup = render(kids[2])
            out = base
            if sub:
                out += "_(" + sub + ")"
            if sup:
                out += "^(" + sup + ")"
            return out
        if tag in ("mrow", "math", "mtable", "mtr", "mtd", "mstyle", "mpadded",
                   "mphantom", "menclose"):
            sep = " ; " if tag == "mtr" else ""
            joined = sep.join(render(k) for k in kids)
            if tag == "mtable":
                rows = [render(k) for k in kids if k.tag.endswith("mtr")]
                return " \\, ".join(rows) if rows else joined
            if tag == "mfenced":
                return "(" + joined + ")"
            return joined
        if tag == "mfenced":
            return "(" + "".join(render(k) for k in kids) + ")"
        # неизвестный контейнер — просто склеиваем детей
        return "".join(render(k) for k in kids) or (node.text or "")

    return render(root)


# Дополнительные замены на случай, если latex2mathml вернул сырую команду
_FALLBACK_SYMBOLS = {
    "\\implies": "⟹", "\\iff": "⟺", "\\impliedby": "⟸",
    "\\leqslant": "⩽", "\\geqslant": "⩾", "\\ll": "≪", "\\gg": "≫",
    "\\iiint": "∭", "\\iint": "∬", "\\oint": "∮",
    "\\bigcup": "⋃", "\\bigcap": "⋂", "\\bigsqcup": "⨆",
    "\\varnothing": "∅", "\\complement": "∁",
    "\\coloneqq": "≔", "\\stackrel": "", "\\overset": "", "\\underset": "",
    "\\colon": ":", "\\cdots": "⋯", "\\vdots": "⋮", "\\ddots": "⋱",
    "\\dots": "…", "\\ldots": "…",
    "\\mathbb{R}": "ℝ", "\\mathbb{N}": "ℕ", "\\mathbb{Z}": "ℤ",
    "\\mathbb{Q}": "ℚ", "\\mathbb{C}": "ℂ", "\\mathbb{P}": "ℙ",
    "\\text": "", "\\mathrm": "", "\\left": "", "\\right": "",
    "\\,": " ", "\\;": " ", "\\!": "", "\\quad": "  ", "\\qquad": "    ",
}


def unicode_math(latex: str) -> str:
    """LaTeX → строка UnicodeMath (для mathtext/плоского текста).

    Никогда не бросает исключение: при провале конвертера возвращает
    очищенный исходный LaTeX.
    """
    s = (latex or "").strip()
    if not s:
        return ""
    out = None
    if get_mathml_converter is not None:
        try:
            ml = get_mathml_converter().to_mathml(s)
            if ml:
                out = _mathml_to_unicode(ml)
        except Exception:
            out = None
    if not out:
        out = s
    for cmd, rep in _FALLBACK_SYMBOLS.items():
        out = out.replace(cmd, rep)
    # снять мусорные обратные слэши от непереведённых команд вида \foo
    out = re.sub(r"\\([A-Za-z]+)", lambda m: m.group(1), out)
    out = out.replace("{", "").replace("}", "")
    # invisible function-application chars (U+2061 и т.п.) — mathtext их не рисует
    out = _INVISIBLE_RE.sub("", out)
    out = re.sub(r"\s{2,}", " ", out).strip()
    return out
