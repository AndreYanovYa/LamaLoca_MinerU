# src/processing/latex_mathml.py
"""
REFACTORING (ТЗ «замена matplotlib.mathtext»): конвертация LaTeX → MathML.

Движок matplotlib mathtext НЕ поддерживает AMS-LaTeX: \\implies, \\leqslant,
\\iiint, окружения cases/aligned/pmatrix и т.п. — рендер падал с
ParseFatalException, а костыльные алиасы в latex_renderer.normalize_latex
лишь маскировали проблему. latex2mathml работает на чистом Python, понимает
стандарт AMS и НИКОГДА не бросает исключение на битом вводе (возвращает
текст как есть) — это ядинственный внешний конвертер, который можно вызывать
без try/except по периметру.

Модуль потокобезопасный (кэш + lock), зависимости только из requirements.txt.
"""
from __future__ import annotations

import re
import threading
import xml.sax.saxutils as _sax

try:  # импорт один раз при загрузке модуля — UI не должен падать без пакета
    from latex2mathml.converter import convert as _l2m_convert
    LATEX2MATHML_AVAILABLE = True
except ImportError:  # pragma: no cover
    _l2m_convert = None
    LATEX2MATHML_AVAILABLE = False

# latex2mathml корректно кодирует символы (&#x0003C; и т.п.), но НЕ
# экранирует спецсимволы XML внутри аргументов \text{...} / \mathrm{...}
# и голых операндов: "\text{A & B}" или "100$ & >50%" дают сырой '&' в
# <mtext>/<mi> — парсер MathML на клиенте падает. Экранируем содержимое
# текстовых команд заранее, а результат конвертации чиним post-санитайзером
# (MathML отобразит &amp; как &, видимой потери нет).
_TEXT_CMD_RE = re.compile(r"\\(?:text|textrm|textnormal|mathrm|mathbf|mbox)\s*\{")


def _escape_text_args(latex: str) -> str:
    """Экранирует &, <, > внутри аргументов \\text{...}/\\mathrm{...}."""
    out = []
    i = 0
    n = len(latex)
    while i < n:
        m = _TEXT_CMD_RE.match(latex, i)
        if not m:
            out.append(latex[i])
            i += 1
            continue
        out.append(latex[i:m.end()])  # сам вызов команды вместе с '{'
        i = m.end()
        depth = 1
        buf = []
        while i < n and depth:
            ch = latex[i]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    break
            buf.append(ch)
            i += 1
        # Вложенные LaTeX-команды (\textbf{\large x}) экранируем точечно,
        # только опасные для XML символы, чтобы не ломать разметку.
        out.append(_sax.escape("".join(buf)))
        if depth == 0:
            out.append("}")
            i += 1
    return "".join(out)


# Post-санитайзер MathML: единичные '&', не являющиеся частью сущности
# (&amp; &#x02200; &lt; и т.д.), заменяем на &amp;. Иначе XML-парсер
# клиента падает на "not well-formed (invalid token)".
_BARE_AMP_RE = re.compile(r"&(?!(?:[A-Za-z][A-Za-z0-9]{1,15}|#\d{1,7}|#[xX][0-9A-Fa-f]{1,6});)")


def _sanitize_mathml(mathml: str) -> str:
    """Гарантирует well-formed XML: чинит голые '&' в результатах конвертации."""
    return _BARE_AMP_RE.sub("&amp;", mathml)


class MathmlConverter:
    """Ленивый потокобезопасный конвертер LaTeX → MathML строки."""

    def __init__(self):
        self._cache: dict[str, str | None] = {}
        self._lock = threading.Lock()

    @property
    def available(self) -> bool:
        return LATEX2MATHML_AVAILABLE

    def to_mathml(self, latex: str) -> str | None:
        """Возвращает готовый <math>…</math> или None, если формула пуста.

        Битый LaTeX не является ошибкой: latex2mathml деградирует сам, а
        вызывающий код (UI/экспортёр) всегда имеет текстовый фолбэк.
        """
        clean = (latex or "").strip()
        if not clean:
            return None
        key = clean
        cached = self._cache.get(key)
        if key in self._cache:
            return cached
        with self._lock:
            if key in self._cache:
                return self._cache[key]
            result: str | None = None
            if LATEX2MATHML_AVAILABLE:
                try:
                    prepared = _escape_text_args(clean)
                    ml = _l2m_convert(prepared)
                    if ml and "<math" in ml:
                        result = _sanitize_mathml(ml)
                except Exception as ex:  # защита от острого края библиотеки
                    from loguru import logger
                    logger.debug(f"latex2mathml не смог сконвертировать '{clean[:60]}': {ex}")
                    result = None
            self._cache[key] = result
            return result


_converter: MathmlConverter | None = None


def get_mathml_converter() -> MathmlConverter:
    """Синглтон конвертера (общий кэш для всего приложения)."""
    global _converter
    if _converter is None:
        _converter = MathmlConverter()
    return _converter
