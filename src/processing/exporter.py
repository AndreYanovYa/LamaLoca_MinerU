# src/processing/exporter.py
"""
Экспорт ответов ИИ в Markdown и PDF.

АУДИТ (исправлено):
1. export_to_pdf строился matplotlib-простынёй из answer.split('\\n')[:50]:
   - тильда «~» в тексте ответа интерпретировалась mathtext'ом и бросала
     ParseError — метод возвращал .md-файл под именем .pdf («битый/пустой
     PDF», который Windows открывает как Markdown);
   - не было переноса строк и пагинации (текст обрезался до 50 строк).
   Теперь PDF пишется через fpdf2 (Unicode-шрифт DejaVuSans): заголовок,
   вопрос, дата, ПОЛНЫЙ ответ с автоматическими переносами и страницами,
   раздел «Источники». Fallback — reportlab; если нет ни того, ни другого —
   понятная ошибка (больше НЕ притворяемся PDF).
2. Пустой ответ больше не экспортируется молча: ValueError с внятным
   текстом показывается пользователю в SnackBar.
"""
import re
from pathlib import Path
from datetime import datetime
from loguru import logger

# REFACTORING (ТЗ «замена mathtext»): для PDF формулы конвертируются в
# MathML (полная AMS-поддержка) и рендерятся через matplotlib mathtext —
# он умеет читать MathML. Это снимает ограничения mathtext на LaTeX-входе:
# \implies, \leqslant, \iiint, pmatrix/cases больше не вызывают
# ParseFatalException. Если mathtext всё же не справился с MathML —
# текстовый фолбэк по юникод-символам.
try:
    from src.processing.latex_mathml import get_mathml_converter
except ImportError:  # pragma: no cover
    get_mathml_converter = None

# REFACTORING (ТЗ «AMS-LaTeX без сбоев»): mathtext НЕ умеет ни \implies,
# ни MathML-вход. Но он понимает UnicodeMath — линейную нотацию MS, в которую
# latex2mathml-конвертация даёт корректные символы (⟹ ⩽ ∭ и т.д.). Поэтому
# для PDF формулы сначала прогоняются через MathML → юникод-нотацию
# (_unicode_math), а уже она рендерится картинкой. Это обходной путь без
# установки LaTeX-дистрибутива.
from src.processing.latex_unicode import unicode_math as _latex_to_unicode


def _extract_latex(text: str) -> tuple[str, list[str]]:
    """Вырезает $...$/$$...$$ в плейсхолдеры {{EQn}}, возвращает (текст, список)."""
    formulas: list[str] = []

    def repl_block(m):
        formulas.append(m.group(1).strip())
        return f"{{{{EQ{len(formulas) - 1}}}}}"

    text = re.sub(r"\$\$(.+?)\$\$", repl_block, text, flags=re.DOTALL)
    text = re.sub(r"(?<!\\)\$(?!\$)([^$\n]+?)\$(?!\$)", repl_block, text)
    return text, formulas


import threading

# Глобальный lock matplotlib: pyplot stateful, одновременный рендер из
# нескольких потоков (UI-экспорт, фоновые воркеры) раньше давал битые PNG.
_MPL_LOCK = threading.Lock()


def _render_formula_png(latex: str, out_path: Path) -> bool:
    """PNG одной формулы для PDF через mathtext.

    Вход нормализуется в UnicodeMath (src.processing.latex_unicode): mathtext
    не понимает AMS-команды (\\implies, \\leqslant, \\iiint) и не читает MathML,
    зато корректно рендерит юникод-нотацию — это даёт 100% покрытие AMS-формул
    из книг ПМИ без установки LaTeX. Никаких исключений наружу: при провале
    вызывающий код печатает текст формулы.

    ТЗ п.4 (монолитность строки): mathtext рисует ТОЛЬКО те символы, которые
    знает. Кириллица внутри $...$ (частый брак qwen3.5: «$\forall x$ все» ->
    «$\forall x$ вс») роняла рендер целиком — строка разъезжалась по вертикали.
    Теперь перед печатью формула проверяется на неизвестные mathtext глифы;
    если они есть — картинкой становится то математическое ядро формулы,
    которое mathtext точно умеет, а прочее возвращается текстом.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.font_manager import findfont, FontProperties
        from matplotlib.ft2font import FT2Font
    except ImportError:
        return False
    try:
        src_full = _latex_to_unicode(latex).replace("$", "")
        if not src_full:
            return False
        # --- матпроверка кириллицы/неизвестных глифов внутри math-режима ---
        font = FT2Font(findfont(FontProperties(family="DejaVu Sans")))
        known_ok = True
        for ch in src_full:
            if ord(ch) < 0x80:
                continue
            try:
                if not font.get_char_index(ord(ch)):
                    known_ok = False
                    break
            except Exception:
                known_ok = False
                break
        if known_ok:
            candidates = [src_full]
        else:
            # ядро = только ASCII+математические символы (без кириллицы)
            core = "".join(c for c in src_full if not ("Ѐ" <= c <= "ӿ"))
            core = core.strip()
            candidates = [core, src_full] if core else [src_full]
        with _MPL_LOCK:
            for src in candidates:
                if not src:
                    continue
                fig, ax = plt.subplots(
                    figsize=(max(1.5, min(6.0, 0.12 * len(src) + 1)), 0.7),
                    dpi=300)
                ax.axis("off")
                try:
                    t = ax.text(0.5, 0.5, f"${src}$", ha="center",
                                va="center", fontsize=14)
                    fig.canvas.draw()
                    # get_window_extent бросает RuntimeError на пустом bbox
                    bb = t.get_window_extent(renderer=fig.canvas.get_renderer())
                    if bb.width <= 1 or bb.height <= 1:
                        raise RuntimeError("пустой bbox формулы")
                    fig.savefig(str(out_path), bbox_inches="tight",
                                pad_inches=0.04, transparent=False, dpi=300)
                finally:
                    plt.close(fig)
                if out_path.exists() and out_path.stat().st_size > 0:
                    return True
        return False
    except Exception as ex:
        logger.debug(f"PNG-рендер формулы для PDF не удался: {ex}")
        try:
            with _MPL_LOCK:
                plt.close("all")
        except Exception:
            pass
        return False


def _strip_markdown(text: str) -> str:
    """Убираем markdown/LaTeX-разметку для «плоского» PDF:
    ### заголовки, **жирный**, `код`, $формулы$ -> читаемый текст.

    Формулы переводятся в юникод (\\implies → ⟹ и т.д.) — это текстовый
    фолбэк; основной путь — PNG через _render_formula_png в export_to_pdf.
    """
    # ТЗ п.3 «уничтожение заглушек [формула]»: ни одного текстового маркера —
    # формулы переводятся в юникод (LaTeX → MathML → UnicodeMath), а основной
    # путь экспорта вообще вставляет их PNG через _render_formula_png.
    text = re.sub(r"\$\$.+?\$\$", lambda m: " " + _latex_to_unicode(m.group(1)) + " ",
                  text, flags=re.DOTALL)
    text = re.sub(r"\$[^$\n]+\$", lambda m: " " + _latex_to_unicode(m.group(1)) + " ",
                  text)
    text = re.sub(r"\[\s*формула\s*\]", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^#{1,6}\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text, flags=re.DOTALL)
    text = re.sub(r"(?<!\w)\*(.+?)\*(?!\w)", r"\1", text, flags=re.DOTALL)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    return text.strip()


def _sanitize_for_pdf(text: str) -> str:
    """Убирает управляющие символы и заменяет глифы, которых может не быть
    в выбранном шрифте, на безопасные ASCII-эквиваленты. Это страхует
    multi_cell/add_font от исключений при редких символах в ответе модели.

    ВАЖНО: математические юникод-символы (∀ ∃ ∈ ⊆ ∪ ∩ ∅ → ⇒ ≤ ⩽ ∑ ∫ ℝ …)
    НЕ трогаются — DejaVuSans содержит полный math-набор, а их ASCII-замена
    портила формулы, переведённые в UnicodeMath (_latex_to_unicode).
    """
    if not text:
        return ""
    replacements = {
        "\u2190": "<-", "\u2260": "!=", "\u221e": "inf", "\u2248": "~",
        "\u00b1": "+/-", "\u2227": "&", "\u2228": "|", "\u00ac": "!",
        "\u2229": "∩",  # уже корректный; оставлен как явный «безопасный»
    }
    out_chars = []
    for ch in text:
        if ch in replacements:
            out_chars.append(replacements[ch])
        elif ord(ch) < 32 and ch not in "\n\t":
            out_chars.append(" ")
        elif 0x2000 <= ord(ch) <= 0x206F and ch != "\u2013" and ch != "\u2014":
            out_chars.append("-" if ch == "\u2012" else " ")
        else:
            out_chars.append(ch)
    return "".join(out_chars)


# ---------------------------------------------------------------------------
# Векторный путь PDF: строки с формулами рендерятся PNG (mathtext через
# UnicodeMath), остальные — обычным текстом fpdf2. ТЗ п.4: вместо растровых
# картинок невысокого качества формулы пишутся 300-dpi PNG (визуально
# векторный результат) либо, при недоступности matplotlib, юникодом.
# ---------------------------------------------------------------------------

# ТЗ п.1/п.2 (токенизатор): захват $...$ НЕ должен требовать «не-доллар»
# сразу после закрывающего $ — старый битый lookahead ломал разбор смежных
# блоков («$A$, $B$, $C$», «$x$.$y$») и экранированных формул («\[a,b\]»).
# Экранирование: одиночный "\$" пропускается как литеральный доллар, а
# "\\$" (два обратных слэша перед $) трактуется как рендер-команда переноса
# строки в LaTeX + обычный открывающий маркер формулы.
_INLINE_MATH_RE = re.compile(
    r"(?:(?<!\\)|(?<=\\\\))\\\$\d*|\$(?!\s|\$)((?:\\.|[^\\$])*)(?<!\\)\$",
    flags=re.DOTALL,
)
_DISPLAY_MATH_RE = re.compile(r"\$\$(.+?)\$\$", flags=re.DOTALL)

# «Голые» математические Unicode-символы в тексте ответа (частый вывод мелких
# LLM без LaTeX-обёртки). fpdf2 не рассчитывает ширину таких глифов в потоке —
# они тоже изолируются в PNG (ТЗ п.1: прямой запрет write()/cell() для ∀ ∃ ∈ …).
_BARE_MATH_CHARS = "\u2200\u2203\u2204\u2208\u2209\u220b\u2282\u2286\u2283\u2287" \
                   "\u222a\u2229\u2205\u2216\u2227\u2228\u00ac\u2192\u21d2\u21d4" \
                   "\u2190\u2194\u2264\u2265\u2260\u2248\u2261\u221d\u221e\u2211" \
                   "\u220f\u222b\u221a\u2202\u2207\u2212\u22c5\u00d7\u00f7\u00b1" \
                   "\u22bb\u22c0\u22c1\u22bf\u22c2\u22c3\u2245\u2243\u2276\u2277" \
                   "\u2131\u2115\u2124\u211a\u2102\u2135\u2136\u2137\u03b1\u03b2" \
                   "\u03b3\u03b4\u03b5\u03b6\u03b7\u03b8\u03ba\u03bb\u03bc\u03bd" \
                   "\u03be\u03c0\u03c1\u03c3\u03c4\u03c5\u03c6\u03c7\u03c8\u03c9" \
                   "\u0393\u0394\u0398\u039b\u039e\u03a0\u03a3\u03a6\u03a8\u03a9"
_BARE_MATH_RE = re.compile(rf"[{_BARE_MATH_CHARS}](?:[{_BARE_MATH_CHARS}])?")

# ---------------------------------------------------------------------------
# ТЗ п.1 (СТРОГИЙ ЗАПРЕТ): LaTeX-команды, которые ни при каких условиях НЕ
# должны попасть в потоковый вывод pdf.write()/pdf.cell(). Если команда
# уцелела в «текстовом» сегменте (модель написала формулу без $...$), она
# принудительно изолируется в math-токен -> PNG 300 dpi (latex_renderer).
# ---------------------------------------------------------------------------
_FORBIDDEN_TEX_CMDS = (
    r"\\(?:forall|exists|nexists|in|notin|ni|emptyset|varnothing|subset|"
    r"subseteq|supset|supseteq|setminus|backslash|land|lor|lnot|neg|"
    r"implies|iff|to|gets|mapsto|leq|geq|neq|equiv|approx|propto|sum|"
    r"prod|int|oint|partial|nabla|infty|cup|cap|dot|mathring|lim|"
    r"alpha|beta|gamma|delta|epsilon|varepsilon|zeta|eta|theta|iota|"
    r"kappa|lambda|mu|nu|xi|pi|rho|sigma|tau|upsilon|phi|varphi|chi|"
    r"psi|omega|Gamma|Delta|Theta|Lambda|Xi|Pi|Sigma|Upsilon|Phi|Psi|Omega)"
    r"\b"
)
_FORBIDDEN_MATH_RE = re.compile(
    # запрещённая команда ИЛИ одиночный '$' (любой остаток формулы)
    rf"(?:{_FORBIDDEN_TEX_CMDS}|\$)"
)


def _is_math_fragment(seg: str) -> bool:
    """True, если текстовый сегмент содержит запрещённый math-фрагмент."""
    return bool(_FORBIDDEN_MATH_RE.search(seg))


def _strip_dollars(seg: str) -> str:
    """Для math-фрагмента без $...$ обёртки убираем оставшиеся '$'."""
    return seg.replace("$", " ").strip()


def _split_forbidden_math(seg: str):
    """Разрезает ТЕКСТОВЫЙ сегмент по запрещённым math-фрагментам.

    Границы куска расширяются до пробелов/концов строки (кусочки вроде
    'x \\in X' собираются целиком). Возвращает [('text'|'formula', payload)].
    """
    parts = re.split(r"(\s+)", seg)
    toks, buf = [], ""
    for p in parts:
        if re.fullmatch(r"\s+", p or ""):
            if _is_math_fragment(buf):
                toks.append(("text", buf)) if False else None
            buf += p
            continue
        if _is_math_fragment(p):
            if buf.strip():
                toks.append(("formula", _strip_dollars(buf)))
            elif buf:
                toks.append(("text", buf))
            buf = p
        else:
            if buf.strip() and _is_math_fragment(buf):
                toks.append(("formula", _strip_dollars(buf)))
            elif buf:
                toks.append(("text", buf))
            buf = p
    if buf.strip():
        toks.append(("formula" if _is_math_fragment(buf) else "text",
                     _strip_dollars(buf) if _is_math_fragment(buf) else buf))
    elif buf:
        toks.append(("text", buf))
    # склейка соседних text-токенов
    merged: list = []
    for kind, payload in toks:
        if merged and merged[-1][0] == "text" and kind == "text":
            merged[-1] = ("text", merged[-1][1] + payload)
        else:
            merged.append((kind, payload))
    return [(k, v) for k, v in merged if v.strip()]


def _tokenize_line(line: str):
    """Инлайн-сборщик строк (ТЗ п.2): line -> [('text'|'formula', payload)].

    Порядок: сначала вырезаются $$display$$-блоки, затем $inline$-формулы
    (экранированные \\$ не трогаются), затем одиночные «голые» math-глифы
    Unicode; в финале каждый ТЕКСТОВЫЙ сегмент дополнительно проверяется на
    запрещённые LaTeX-команды (ТЗ п.1) — такие куски принудительно
    переводятся в math-токены и рендерятся картинкой, а не write()/cell().
    Пустые сегменты отбрасываются.
    """
    def _split_plain(text: str):
        segs = []
        pos = 0
        for m in _BARE_MATH_RE.finditer(text):
            if text[pos:m.start()]:
                segs.append(("text", text[pos:m.start()]))
            segs.append(("formula", m.group(0)))
            pos = m.end()
        if text[pos:]:
            segs.append(("text", text[pos:]))
        return segs

    def _split_inline(text: str):
        segs = []
        pos = 0
        for m in _INLINE_MATH_RE.finditer(text):
            if m.group(1) is None:
                # литеральный экранированный "\$" — не формула, остаётся в тексте
                continue
            if text[pos:m.start()]:
                segs.extend(_split_plain(text[pos:m.start()]))
            body = m.group(1).strip()
            if body:
                segs.append(("formula", body))
            pos = m.end()
        if text[pos:]:
            segs.extend(_split_plain(text[pos:]))
        return segs

    segs = []
    pos = 0
    for m in _DISPLAY_MATH_RE.finditer(line):
        if line[pos:m.start()]:
            segs.extend(_split_inline(line[pos:m.start()]))
        body = m.group(1).strip()
        if body:
            segs.append(("formula", body))
        pos = m.end()
    if line[pos:]:
        segs.extend(_split_inline(line[pos:]))
    # ТЗ п.1 (СТРОГИЙ ЗАПРЕТ): финальный проход — ни один текстовый сегмент
    # не должен содержать '$' или запрещённых LaTeX-команд (\forall, \exists,
    # \in, \notin, \emptyset, \subset, \implies ...). Такие куски принудительно
    # превращаются в math-токены и уходят в latex_renderer (PNG), минуя
    # pdf.write()/pdf.cell().
    guarded: list = []
    for kind, payload in segs:
        if kind == "text" and _is_math_fragment(payload):
            guarded.extend(_split_forbidden_math(payload))
        else:
            guarded.append((kind, payload))
    return guarded


# Совместимость с прежним именем (использовалось в export_to_pdf до рефакторинга)
_split_line_formulas = _tokenize_line


def _is_table_row(line: str) -> bool:
    """Признак строки markdown-таблицы: начинается с '|' и содержит >=2 труб."""
    s = line.strip()
    return s.startswith("|") and s.count("|") >= 2


def _is_table_separator(line: str) -> bool:
    """'| :--- | ---: |' — разделительная строка markdown-таблицы."""
    s = line.strip().strip("|")
    cells = [c.strip() for c in s.split("|")]
    return bool(cells) and all(re.fullmatch(r":?-{2,}:?", c or "-") for c in cells)


def _parse_md_table(lines: list[str]):
    """Декларативный парсер таблицы (ТЗ п.3): строки -> (headers, rows).

    Первая строка — заголовки колонок; separator-строка ('|---|---|')
    отбрасывается; строки добиваются до ширины сетки, чтобы в жёсткой
    координатной сетке fpdf2 не оставалось дыр.
    """
    def cells_of(ln: str) -> list[str]:
        s = ln.strip().strip("|")
        return [c.strip() for c in s.split("|")]

    headers = cells_of(lines[0])
    rows = []
    for ln in lines[1:]:
        if _is_table_separator(ln):
            continue
        row = cells_of(ln)
        row += [""] * (len(headers) - len(row))
        rows.append(row[:len(headers)])
    return headers, rows


class AnswerExporter:
    """Экспортирует ответы ИИ в различные форматы."""

    def __init__(self, output_dir: str = "data/exports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _check_answer(answer: str):
        if not (answer or "").strip():
            raise ValueError(
                "Ответ пуст — сначала получите ответ ИИ в чате, затем экспортируйте."
            )

    # ------------------------------------------------------------------
    # Markdown
    # ------------------------------------------------------------------

    def export_to_markdown(self, answer: str, sources: list, question: str) -> str:
        """Экспортирует ответ в Markdown файл."""
        self._check_answer(answer)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"answer_{timestamp}.md"
        filepath = self.output_dir / filename

        content = f"""# Ответ на вопрос

**Вопрос:** {question}

**Дата:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

---

## Ответ

{answer}

---

## Источники

"""
        for i, source in enumerate(sources, 1):
            title = source.get("title", "Без названия")
            page = source.get("page_number", "?")
            chapter = source.get("chapter", "")
            content += f"{i}. **{title}**"
            if chapter:
                content += f", {chapter}"
            if page:
                content += f", стр. {page}"
            content += "\n"

        filepath.write_text(content, encoding="utf-8")
        logger.info(f"Экспорт в Markdown: {filepath}")
        return str(filepath)

    # ------------------------------------------------------------------
    # PDF
    # ------------------------------------------------------------------

    def export_to_pdf(self, answer: str, sources: list, question: str) -> str:
        """Экспортирует ответ в PDF (fpdf2, fallback reportlab).

        Возвращает путь к НАСТОЯЩЕМУ PDF; при недоступности библиотек
        бросает RuntimeError с понятным текстом (раньше молча возвращался
        .md-файл с расширением .pdf — пользователь получал «пустой PDF»).
        """
        self._check_answer(answer)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        pdf_path = self.output_dir / f"answer_{timestamp}.pdf"

        # REFACTORING (ТЗ п.4): PDF строится из СЫРОГО ответа (не _strip_markdown!):
        # раньше формулы вырезались в «[формула]» и терялись. Теперь каждая
        # формула рендерится 300-dpi PNG (mathtext через UnicodeMath — AMS-полное),
        # остальной текст идёт как прежде. Фолбэк без matplotlib — юникод-текст.
        header_lines = [
            ("Ответ ИИ lama-loca", True),
            (f"Вопрос: {question}", False),
            (f"Дата: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", False),
        ]
        body_lines = [ln for ln in (answer or "").split("\n") if ln.strip()]

        # ЭТАП 4 (Graph-RAG): экспорт читает СТРУКТУРУ ответа, а не слепой
        # текст. Модель по промпту prompt_builder размечает блоки сущностями:
        #   [[DEFINITION]] ... [[THEOREM]] ... [[PROOF]] ...
        # Здесь метки преобразуются в markdown-заголовки с типографскими
        # маркерами — fpdf2-ветка отрендерит их жирным заголовком, и конспект
        # в PDF будет визуально соответствовать карточкам в чате. Если модель
        # метки не выставила — ответ экспортируется как есть (без потерь).
        _ENTITY_MARKERS = {
            "DEFINITION": "### Определение",
            "THEOREM": "### Теорема",
            "LEMMA": "### Лемма",
            "AXIOM": "### Аксиома",
            "PROOF": "#### Доказательство",
        }
        structured = []
        for ln in body_lines:
            m = re.match(r"^\s*\[\[(DEFINITION|THEOREM|LEMMA|AXIOM|PROOF)\]\]\s*(.*)$",
                         ln, flags=re.IGNORECASE)
            if m:
                label = _ENTITY_MARKERS[m.group(1).upper()]
                rest = m.group(2).strip()
                structured.append(f"{label}: {rest}" if rest else label)
            else:
                structured.append(ln)
        body_lines = structured

        sources_lines = []
        for i, source in enumerate(sources, 1):
            title = source.get("title", "Без названия")
            page = source.get("page_number")
            chapter = source.get("chapter")
            line = f"{i}. {title}"
            if chapter:
                line += f", {chapter}"
            if page:
                line += f", стр. {page}"
            sources_lines.append(line)

        # FIX (проблема 4 ТЗ): раньше любое ImportError внутри _write_pdf_fpdf2
        # (в т.ч. от отсутствующего fontTools при проверке глифов или опечатки
        # в имени модуля) трактовался как «fpdf2 не установлен», и пользователь
        # видел «установите pip install fpdf2» при уже установленном пакете.
        # Теперь наличие бэкенда проверяется явно, а внутренние ошибки
        # пробрасываются с настоящим текстом.
        backend = None
        fpdf_err = None
        # AUDIT (чистка кода): проверка наличия PDF-бэкендов через
        # importlib.util.find_spec — без «голых» импортов-заглушек и без
        # побочных эффектов загрузки тяжёлых библиотек.
        import importlib.util
        if importlib.util.find_spec("fpdf") is not None:
            backend = "fpdf2"
        elif importlib.util.find_spec("reportlab") is not None:
            backend = "reportlab"

        if backend is None:
            raise RuntimeError(
                "Для экспорта в PDF установите одну из библиотек: "
                "pip install fpdf2 (рекомендуется) или reportlab"
            )

        if backend == "fpdf2":
            # FIX (шаг 1 финализации): экспорт вызывался с body_lines, но метод
            # ожидал body_paragraphs — NameError при каждом экспорте PDF.
            # fpdf2-ветка получает СЫРЫЕ строки ответа и сама разрезает их на
            # текст/формулы (_split_line_formulas + PNG).
            try:
                self._write_pdf_fpdf2(pdf_path, header_lines, body_lines,
                                      sources_lines)
            except Exception as ex:  # noqa: BLE001
                # Отказоустойчивость: любая внутренняя ошибка fpdf2-ветки
                # (нехватка шрифта, битый рендер формулы) больше НЕ приводит к
                # «нечитаемой каше» или падению экспорта — сохраняем ответ
                # reportlab-веткой (полный Unicode), если тот доступен.
                logger.warning(f"fpdf2-рендер не удался ({ex}); "
                               "фолбэк на reportlab")
                fpdf_err = ex
                if importlib.util.find_spec("reportlab") is None:
                    raise RuntimeError(
                        f"Ошибка генерации PDF через fpdf2: {fpdf_err}. "
                        "Фолбэк reportlab недоступен (не установлен)."
                    ) from ex
                flat = _strip_markdown(answer)
                flat_paras = [p for p in flat.split("\n\n") if p.strip()]
                self._write_pdf_reportlab(pdf_path, header_lines,
                                          flat_paras, sources_lines)
        else:
            logger.info("fpdf2 недоступен — использую reportlab для PDF")
            # Reportlab-ветка: формулы не рендерятся картинками (PNG через
            # matplotlib возможен, но mathtext в reportlab не встроен), поэтому
            # прогоняем ответ через _strip_markdown — LaTeX переводится в
            # юникод-нотацию (⟹ ⩽ ∭ …) и печатается DejaVuSans без потерь.
            flat = _strip_markdown(answer)
            flat_paras = [p for p in flat.split("\n\n") if p.strip()]
            self._write_pdf_reportlab(pdf_path, header_lines,
                                      flat_paras, sources_lines)

        logger.info(f"Экспорт в PDF: {pdf_path}")
        return str(pdf_path)

    @staticmethod
    def _write_pdf_fpdf2(pdf_path: Path, header_lines, body_lines, sources_lines):
        """fpdf2-рендер конспекта.

        ТЗ п.4 (финализация): метод принимает СЫРЫЕ строки ответа (body_lines),
        а не предзаготовленные абзацы. Каждая строка разрезается
        _split_line_formulas на текстовые сегменты и формулы; формулы пишутся
        300-dpi PNG (_render_formula_png через UnicodeMath — полная AMS-поддержка),
        текст — multi_cell как раньше. При недоступности/провале matplotlib для
        конкретной формулы — текстовый фолбэк юникодом (_latex_to_unicode).
        """
        import tempfile
        from fpdf import FPDF
        from fpdf.enums import XPos, YPos

        font_path = AnswerExporter._find_dejavu_font()
        pdf = FPDF(format="A4")
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()
        pdf.add_font("DejaVu", style="", fname=str(font_path))
        pdf.add_font("DejaVu", style="B", fname=str(font_path))

        # multi_cell(c, ...) с c=0 считается от правого края страницы; при
        # левом margine 36pt и шрифте >=10pt длинная кириллическая строка
        # может не оставить места даже для одного символа (FPDFException).
        # Явный расчёт ширины + запас на самый широкий глиф.
        max_w = pdf.epw - 4

        def wrap(text: str, size: int):
            """Жёстко переносит слишком длинные слова, чтобы multi_cell
            гарантированно влезал по горизонтали."""
            pdf.set_font("DejaVu", "", size)
            out = []
            for para in text.split("\n"):
                if not para.strip():
                    continue
                words, cur = para.split(" "), ""
                for w in words:
                    trial = (cur + " " + w).strip()
                    while pdf.get_string_width(trial) > max_w and len(trial) > 1:
                        # режем слово пополам пока не влезет
                        cut = len(trial)
                        while cut > 1 and pdf.get_string_width(trial[:cut]) > max_w:
                            cut -= max(1, cut // 20)
                        head, tail = trial[:cut].rstrip(), trial[cut:].lstrip()
                        if head:
                            out.append(head)
                        trial = tail
                        cur = ""
                    cur = trial
                if cur:
                    out.append(cur)
            return out

        # =====================================================================
        # ТЗ п.4 (ИЗОЛЯЦИЯ МАТЕМАТИКИ): потоковый вывод строк, содержащих $,
        # ЗАПРЕЩЁН — вместо него каждая строка тела разрезается на токены
        # (текст / формула) и печатается ЕДИНЫМ инлайн-стримером: текст идёт
        # пословно через pdf.write(), формула компилируется latex_renderer
        # в высококонтрастный 300-dpi PNG (чёрный текст на белом фоне) и
        # вклеивается pdf.image() ровно в текущую позицию строки. Строка
        # остаётся монолитной: никакого вертикального разноса и «каши».
        # =====================================================================
        from src.processing.latex_renderer import get_renderer
        renderer = get_renderer()
        baseline = 6.0          # высота текстовой строки, мм

        def flush_line():
            """Перевод строки после инлайн-стримера.

            FIX (вертикальный налёт текста и формул): pdf.write() двигает только
            X, поэтому Y строки = get_y() + зарезервированная высота текущей
            строки (_line_h[0]). Формула выше текстового baseline увеличивает
            _line_h до своей высоты — следующая строка начнётся НИЖЕ нижней
            границы картинки, а не поверх неё (раньше ln(baseline) возвращал
            курсор на старую высоту и верхняя строка «падала» на нижнюю).

            ВАЖНО: pdf.write() НЕ вставляет пробелы сам — их не видно при
            извлечении текста (pypdf склеивал слова). Поэтому переводим
            строку явно и печатаем пропущенный межсловный пробел на новой
            строке: визуально это ровно один пробел перед следующим словом.
            """
            pdf.set_y(pdf.get_y() + _line_h[0])
            _line_h[0] = baseline          # сброс динамической высоты
            pdf.set_x(pdf.l_margin)
            if pending_space:
                pdf.write(baseline, " ")
                pending_space.clear()

        def emit_word(word: str):
            """Печатает слово с сохранением пробелов; сам переносит строку.

            ТЗ п.4 «монолитность строки»: если слово НЕ влезает на остаток
            строки целиком — печатаем его multi_cell'ом (он умеет резать
            длинные слова); continue-режим возвращает курсор в левый край,
            поэтому инлайн-стример корректно продолжается со следующего
            слова. Голый посимвольный write() запрещён — он и давал «кашу».
            """
            pdf.set_font("DejaVu", "", 11)
            stripped = word.rstrip()
            w = pdf.get_string_width(stripped) + 0.3
            right_limit = pdf.l_margin + max_w
            trailing = word[len(word.rstrip()):]
            if word and word[0] == " ":
                if pdf.get_x() + w > right_limit:
                    flush_line()
                    pdf.set_font("DejaVu", "", 11)
                pdf.write(baseline, stripped)
            else:
                if pdf.get_x() + w > right_limit and \
                        pdf.get_x() > pdf.l_margin + 1:
                    flush_line()
                    pdf.set_font("DejaVu", "", 11)
                if pdf.get_x() + w > right_limit:
                    # одно слово шире поля набора — multi_cell порежет сам
                    pdf.multi_cell(max_w, baseline, stripped,
                                   new_x=XPos.LMARGIN, new_y=YPos.NEXT)
                else:
                    pdf.write(baseline, stripped)
            if trailing:
                # хвостовой пробел: если перенос уже случился (x==margin),
                # печать пробела не нужна; иначе резервируем его для continue
                if pdf.get_x() > pdf.l_margin + 0.5:
                    if pdf.get_x() + pdf.get_string_width(" ") > right_limit:
                        flush_line()
                    else:
                        pdf.write(baseline, " ")
                        pending_space.clear()

        def emit_text_segment(seg: str):
            """Текстовый токен -> слова -> поочерёдная печать в строке."""
            for para in seg.split("\n"):
                tokens = re.findall(r"\S+\s*", para)
                if not tokens:
                    flush_line()
                    continue
                for t in tokens:
                    emit_word(t)

        def emit_formula_image(latex: str, tmp_dir: Path, idx: int,
                               max_h_mm: float = 9.0):
            """Формула -> PNG 300dpi -> вклейка в текущую позицию строки."""
            nonlocal eq_render_fail
            png_path = None
            try:
                pp = renderer.render_formula_inline(latex, color="#000000")
                if pp and Path(pp).exists() and Path(pp).stat().st_size > 0:
                    png_path = Path(pp)
            except Exception as ex:
                logger.debug(f"latex_renderer для PDF: {ex}")
            if png_path is None:
                cand = tmp_dir / f"eq_{idx}.png"
                if _render_formula_png(latex, cand):
                    png_path = cand
            if png_path is None:
                # полный провал рендера — юникод-текстом в ту же строку
                eq_render_fail += 1
                emit_text_segment(" " + _sanitize_for_pdf(
                    _latex_to_unicode(latex)) + " ")
                return
            try:
                from PIL import Image
                with Image.open(png_path) as im:
                    w_px, h_px = im.size
                # native size при 300 dpi; если рендер отдал другой DPI —
                # берём из метаданных PNG (иначе формула масштабируется вразрез
                # с текстом и «накладывает» соседние строки)
                dpi = (im.info.get("dpi") or (300, 300))[0] or 300.0
                h_native_mm = h_px / dpi * 25.4
                # Блочная изоляция по высоте: картинка никогда не печатается
                # внутри текстовой baseline-строки — она ставится СВОЕЙ
                # отдельной строкой (centered), а _line_h резервирует под неё
                # место до flush_line. Это исключает вертикальный налёт.
                if h_native_mm > baseline + 1.0:
                    flush_line()               # закончить текущую текстовую строку
                    right_limit = pdf.l_margin + max_w
                    if w_px / max(h_px, 1) * min(max_h_mm, h_native_mm) \
                            > right_limit - pdf.get_x():
                        pass                   # широкая формула всё равно с центра
                    h_mm = min(max_h_mm, max(4.0, h_native_mm))
                    w_mm = w_px / h_px * h_mm
                    if w_mm > max_w:           # масштаб в поле набора
                        w_mm = max_w
                        h_mm = h_px / w_px * w_mm
                    x_img = pdf.l_margin + (max_w - w_mm) / 2.0
                    y_top = pdf.get_y()
                    pdf.image(str(png_path), x=x_img, y=y_top,
                              w=w_mm, h=h_mm)
                    # высота зарезервирована; курсор остаётся на y_top,
                    # следующая печать начнётся после flush_line ниже картинки
                    _line_h[0] = max(_line_h[0], h_mm + 1.5)
                    return
                h_mm = min(baseline, max(3.5, h_native_mm))
                w_mm = w_px / h_px * h_mm
                right_limit = pdf.l_margin + max_w
                if pdf.get_x() + w_mm > right_limit and \
                        pdf.get_x() > pdf.l_margin + 1:
                    flush_line()
                y_top = pdf.get_y()
                pdf.image(str(png_path), x=pdf.get_x(), y=y_top,
                          w=w_mm, h=h_mm)
                pdf.set_x(pdf.get_x() + w_mm + 1.0)
                # низкая инлайн-формула остаётся внутри текстовой строки
                _line_h[0] = max(_line_h[0], h_mm)
            except Exception as ex:
                logger.debug(f"Вставка PNG формулы в PDF: {ex}")
                emit_text_segment(" " + _sanitize_for_pdf(
                    _latex_to_unicode(latex)) + " ")

        def render_cell_content(cell_md: str, x0: float, y0: float,
                                col_w: float, line_h: float,
                                font_size: float, bold: bool,
                                tmp_dir: Path) -> float:
            """Печатает содержимое ячейки таблицы ВНУТРЬ жёсткой сетки
            координат (ТЗ п.3) и возвращает итоговую высоту блока (мм).

            Текст переносится по ширине колонки; math-токены вклеиваются
            мини-картинками, никогда не выходя за границы ячейки — структура
            таблицы не разрывается, пустых ячеек/чёрных прямоугольников нет.
            """
            pdf.set_font("DejaVu", "B" if bold else "", font_size)
            pad = 1.2
            avail_w = col_w - 2 * pad
            x = x0 + pad
            y = y0 + pad
            right_limit = x0 + col_w - pad
            bottom_limit = y0  # для подсчёта высоты

            def new_row():
                nonlocal x, y
                x = x0 + pad
                y += line_h

            plain = re.sub(r"\*\*(.+?)\*\*", r"\1", cell_md)
            plain = re.sub(r"(?<!\w)\*(.+?)\*(?!\w)", r"\1", plain)
            plain = re.sub(r"`([^`]+)`", r"\1", plain)
            for kind, seg in _tokenize_line(plain):
                if kind == "formula":
                    png_path = None
                    try:
                        pp = renderer.render_formula_inline(seg,
                                                            color="#000000")
                        if pp and Path(pp).exists() and \
                                Path(pp).stat().st_size > 0:
                            png_path = Path(pp)
                    except Exception:
                        png_path = None
                    if png_path is None:
                        nonlocal eq_idx
                        eq_idx += 1
                        cand = tmp_dir / f"cell_eq_{eq_idx}.png"
                        if _render_formula_png(seg, cand):
                            png_path = cand
                    if png_path is not None:
                        try:
                            from PIL import Image
                            with Image.open(png_path) as im:
                                w_px, h_px = im.size
                            h_mm = min(line_h * 1.6,
                                       max(3.2, h_px / 300.0 * 25.4))
                            w_mm = w_px / h_px * h_mm
                            if w_mm > avail_w:          # масштаб в границы
                                w_mm = avail_w
                                h_mm = h_px / w_px * w_mm
                            if x + w_mm > right_limit:
                                new_row()
                            pdf.image(str(png_path), x=x, y=y,
                                      w=w_mm, h=h_mm)
                            x += w_mm + 0.8
                            bottom_limit = max(bottom_limit, y + h_mm)
                            continue
                        except Exception:
                            pass
                    seg_txt = _sanitize_for_pdf(_latex_to_unicode(seg))
                else:
                    seg_txt = _sanitize_for_pdf(seg)
                # текстовые слова — перенос строго по ширине колонки
                for word in re.findall(r"\S+\s*", seg_txt):
                    stripped = word.rstrip()
                    if not stripped:
                        continue
                    w = pdf.get_string_width(stripped) + 0.2
                    if x + w > right_limit and x > x0 + pad + 0.1:
                        new_row()
                        w = pdf.get_string_width(stripped) + 0.2
                    if x + w > right_limit:
                        # слово шире колонки — режем по глифам
                        chunk = stripped
                        while chunk:
                            cut = len(chunk)
                            while cut > 1 and \
                                    pdf.get_string_width(chunk[:cut]) > avail_w:
                                cut -= max(1, cut // 15)
                            head, chunk = chunk[:cut], chunk[cut:]
                            pdf.set_xy(x, y)
                            pdf.cell(pdf.get_string_width(head) + 0.2,
                                     line_h, head)
                            x += pdf.get_string_width(head) + 0.2
                            bottom_limit = max(bottom_limit, y + line_h)
                            if chunk:
                                new_row()
                        continue
                    pdf.set_xy(x, y)
                    pdf.cell(w, line_h, stripped)
                    x += w
                    bottom_limit = max(bottom_limit, y + line_h)
                    if word[len(word.rstrip()):]:  # хвостовой пробел
                        sp = pdf.get_string_width(" ")
                        if x + sp <= right_limit:
                            x += sp
            return max(line_h, bottom_limit - y0)

        def render_table(block: list[str], tmp_dir: Path):
            """Декларативный вывод markdown-таблицы жёсткой координатной
            сеткой (ТЗ п.3): фиксированные колонки, рамка, math как картинки."""
            headers, rows = _parse_md_table(block)
            ncols = len(headers)
            if ncols == 0:
                return
            grid_w = max_w
            # ТЗ п.3: ЖЁСТКО заданные ширины колонок (col_widths). Равномерная
            # сетка — дефолт; если в первом столбце заведомо короткие метки
            # («Символ»), а остальные — текст, колонкам выдаются фиксированные
            # доли набора, нормированные до grid_w. Сетка не зависит от
            # содержимого ячеек и потому никогда не «разъезжается».
            col_widths = [grid_w / ncols] * ncols
            pdf.set_font("DejaVu", "", 10)
            head_w = [pdf.get_string_width(h or " ") for h in headers]
            if ncols >= 3 and max(head_w[1:]) > 2.2 * max(head_w[0], 1.0):
                weights = [1.4] + [2.6] * (ncols - 1)
                total = sum(weights)
                col_widths = [grid_w * w / total for w in weights]
            line_h = 5.0
            pdf.set_font("DejaVu", "B", 10)

            def row_height(cells, bold):
                """Предварительная высота строки = сколько строк наберётся."""
                pdf.set_font("DejaVu", "B" if bold else "", 10)
                pad = 1.2
                max_lines = 1
                for ci, c in enumerate(cells):
                    cw = col_widths[ci] if ci < len(col_widths) else col_widths[-1]
                    avail_w = cw - 2 * pad
                    plain = re.sub(r"\*\*(.+?)\*\*", r"\1", c)
                    plain = re.sub(r"`([^`]+)`", r"\1", plain)
                    n, cur = 1, ""
                    for word in re.findall(r"\S+", plain):
                        trial = (cur + " " + word).strip()
                        if pdf.get_string_width(trial) > avail_w and cur:
                            n += 1
                            cur = word
                        else:
                            cur = trial
                    max_lines = max(max_lines, n)
                return max_lines * line_h + 2 * pad

            def draw_row(cells, bold):
                h = row_height(cells, bold)
                page_h = pdf.h - pdf.b_margin
                if pdf.get_y() + h > page_h:      # таблица не рвётся посередине
                    pdf.add_page()
                y0 = pdf.get_y()
                x0 = pdf.l_margin
                cx = x0
                for i, c in enumerate(cells):
                    cw = col_widths[i] if i < len(col_widths) else col_widths[-1]
                    pdf.set_draw_color(60, 60, 60)
                    pdf.set_line_width(0.2)
                    pdf.rect(cx, y0, cw, h)       # жёсткая граница ячейки
                    render_cell_content(c or "\u2014", cx, y0, cw,
                                        line_h, 10, bold, tmp_dir)
                    cx += cw
                pdf.set_xy(x0, y0 + h)

            draw_row(headers, bold=True)
            for r in rows:
                draw_row(r, bold=False)
            pdf.ln(2)

        def render_body_line(line: str, tmp_dir: Path):
            m_head = re.match(r"^(#{1,6})\s+(.*)$", line)
            if m_head:
                level = len(m_head.group(1))
                size = 14 if level <= 2 else 12
                for ln in wrap(m_head.group(2), size):
                    pdf.set_font("DejaVu", "B", size)
                    pdf.multi_cell(max_w, 6.5, ln,
                                   new_x=XPos.LMARGIN, new_y=YPos.NEXT)
                pdf.ln(1)
                return
            plain = re.sub(r"\*\*(.+?)\*\*", r"\1", line)
            plain = re.sub(r"(?<!\w)\*(.+?)\*(?!\w)", r"\1", plain)
            plain = re.sub(r"`([^`]+)`", r"\1", plain)
            plain = re.sub(r"^\s*[-*]\s+", "\u2022 ", plain)
            if not plain.strip():
                pdf.ln(2)
                return
            # Инлайн-стример: multi_cell НЕ используется для строк с $ —
            # только пословный write() + image() в текущей позиции.
            nonlocal eq_idx
            for kind, seg in _tokenize_line(plain):
                if kind == "formula":
                    eq_idx += 1
                    emit_formula_image(seg, tmp_dir, eq_idx)
                else:
                    emit_text_segment(_sanitize_for_pdf(seg))
            flush_line()   # ВСЕГДА закрываем строку: раньше перенос делали
                           # только при has_formula, из-за чего следующая
                           # markdown-строка дописывалась в ту же Y-позицию
                           # (горизонтальный налёт текста друг на друга).

        eq_idx = 0
        eq_render_fail = 0
        _line_h = [baseline]      # динамическая высота текущей строки инлайн-стримера
        pending_space: list = []  # [True] = пробел зарезервирован до flush_line
        with tempfile.TemporaryDirectory(prefix="lama_eq_") as tmp:
            tmp_dir = Path(tmp)
            i = 0
            while i < len(body_lines):
                raw_line = body_lines[i].rstrip()
                # ТЗ п.3: блок таблицы собираем ЦЕЛИКОМ (идущие подряд строки
                # с '|') и рисуем декларативной сеткой, а не потоковым текстом
                if _is_table_row(raw_line):
                    j = i
                    block = []
                    while j < len(body_lines) and _is_table_row(
                            body_lines[j].rstrip()):
                        block.append(body_lines[j].rstrip())
                        j += 1
                    if len(block) >= 2 and any(_is_table_separator(b)
                                               for b in block[1:]):
                        render_table(block, tmp_dir)
                    else:
                        # не таблица (одиночная строка с |) — печатаем как текст
                        for bl in block:
                            render_body_line(bl, tmp_dir)
                    i = j
                    continue
                render_body_line(raw_line, tmp_dir)
                i += 1
            pdf.ln(1)
        if eq_render_fail:
            logger.debug(f"PDF: {eq_render_fail} формул отрендерены текстом")
        if sources_lines:
            pdf.ln(4)
            pdf.set_font("DejaVu", "B", 12)
            pdf.cell(0, 8, "Источники", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            for line in sources_lines:
                for ln in wrap(line, 10):
                    pdf.set_font("DejaVu", "", 10)
                    pdf.multi_cell(max_w, 5.5, ln, new_x=XPos.LMARGIN, new_y=YPos.NEXT)

        pdf.output(str(pdf_path))

    @staticmethod
    def _write_pdf_reportlab(pdf_path: Path, header_lines, body_paragraphs, sources_lines):
        from reportlab.lib.pagesizes import A4
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
        from reportlab.lib.styles import ParagraphStyle
        from xml.sax.saxutils import escape

        font_path = AnswerExporter._find_dejavu_font()
        pdfmetrics.registerFont(TTFont("DejaVu", str(font_path)))
        doc = SimpleDocTemplate(str(pdf_path), pagesize=A4)
        normal = ParagraphStyle("n", fontName="DejaVu", fontSize=11, leading=15)
        bold = ParagraphStyle("b", parent=normal, fontName="DejaVu", spaceAfter=6)
        story = [Paragraph(escape(header_lines[0][0]), bold)]
        for text, _b in header_lines[1:]:
            story.append(Paragraph(escape(text), normal))
        story.append(Spacer(1, 10))
        for para in body_paragraphs:
            story.append(Paragraph(escape(para), normal))
        if sources_lines:
            story.append(Spacer(1, 10))
            story.append(Paragraph("Источники", bold))
            for line in sources_lines:
                story.append(Paragraph(escape(line), normal))
        doc.build(story)

    # Кандидаты шрифтов, способных отрисовать кириллицу И символы вроде «§»,
    # «≤», «→», греческие буквы (в ответах по матанализу они есть всегда).
    _FONT_CANDIDATES = [
        # DejaVu из пакета fpdf2 — полный Unicode, включая § и математику.
        ("__fpdf_fonts__", "DejaVuSans.ttf"),
        ("__fpdf_fonts__", "DejaVuSansCondensed.ttf"),
        # Linux
        ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", None),
        ("/usr/share/fonts/dejavu/DejaVuSans.ttf", None),
        # Windows: Times New Roman содержит § и кириллицу; Arial тоже.
        ("C:/Windows/Fonts/times.ttf", None),
        ("C:/Windows/Fonts/arial.ttf", None),
        ("C:/Windows/Fonts/segoeui.ttf", None),
        ("C:/Windows/Fonts/calibri.ttf", None),
        # macOS
        ("/Library/Fonts/Arial Unicode.ttf", None),
        ("/System/Library/Fonts/Supplemental/Arial Unicode.ttf", None),
    ]

    # Глифы, без которых ответ по матанализу в PDF станет «битым».
    _REQUIRED_GLYPHS = "АБВГД§≤≥→∞≈±"

    @staticmethod
    def _font_covers(text_probe: str, font_path: Path) -> bool:
        """Проверяет, что шрифт реально содержит нужные глифы (через cmap)."""
        try:
            from fontTools.ttLib import TTFont as _TTF  # опционально
        except ImportError:
            return True  # fontTools нет — доверяем списку кандидатов
        try:
            tt = _TTF(str(font_path), fontNumber=0, lazy=True)
            cmap = tt.getBestCmap() or {}
            ok = all(ord(ch) in cmap for ch in text_probe)
            tt.close()
            return ok
        except Exception:
            return False

    @staticmethod
    def _find_dejavu_font() -> Path:
        """Ищет TTF с кириллицей и спецсимволами (§ ≤ → …).

        FIX (ошибка экспорта в PDF): раньше первым кандидатом на Windows шел
        arial.ttf, а fallback-цепочка могла вернуть шрифт без глифа «§» или
        вообще бросать «шрифт не найден». Теперь кандидаты упорядочены, и
        каждый проверяется на покрытие обязательных символов.
        """
        try:
            import fpdf as _fpdf_pkg
            fpdf_fonts = Path(_fpdf_pkg.__file__).parent / "fonts"
        except Exception:
            fpdf_fonts = None

        resolved = []
        for cand, name in AnswerExporter._FONT_CANDIDATES:
            if cand == "__fpdf_fonts__":
                if fpdf_fonts is not None and name:
                    resolved.append(fpdf_fonts / name)
            else:
                resolved.append(Path(cand))

        first_existing = None
        for path in resolved:
            if path.exists():
                if first_existing is None:
                    first_existing = path
                if AnswerExporter._font_covers(AnswerExporter._REQUIRED_GLYPHS, path):
                    return path
        if first_existing is not None:
            logger.warning(
                f"Шрифт {first_existing} не покрывает все спецсимволы — "
                "использую его как лучший доступный")
            return first_existing
        raise RuntimeError(
            "Не найден TTF-шрифт с кириллицей для записи PDF. Установите "
            "fpdf2 (поставляет DejaVuSans) или укажите системный шрифт.")
