# src/processing/latex_renderer.py
"""
Рендеринг LaTeX формул в изображения (PNG) — фолбэк-путь для UI и PDF.

REFACTORING (ТЗ п.2 «векторный движок формул»): основным путём отображения
формулок в чате теперь является нативный MathML/LaTeX-рендер Flet 1.x
(ft.Markdown с $$…$$) и конвертер latex2mathml (src/processing/latex_mathml),
а не растровый mathtext. Matplotlib остался здесь ТОЛЬКО как:
  a) фолбэк PNG, если нативный рендер недоступен;
  b) генератор изображений формул для PDF (fpdf2 не умеет MathML/SVG).
Добавлен AMS-алиас резолвер: \\implies, \\leqslant, \\iiint и т.п. до подачи
в mathtext заменяются поддерживаемыми эквивалентами — раньше на них mathtext
падал с ParseFatalException и в UI появлялись «чёрные квадраты».
"""
import re
import base64
import threading
from pathlib import Path
from typing import List, Tuple

import matplotlib
matplotlib.use('Agg')  # Неинтерактивный бэкенд
import matplotlib.pyplot as plt
from loguru import logger

# Потокобезопасность: pyplot — глобальный stateful-объект; одновременный
# рендер из UI-потока и воркера PDF-экспорта давал битые/пустые PNG.
_MPL_LOCK = threading.Lock()


class LatexRenderer:
    """Рендерит LaTeX формулы в PNG изображения."""
    
    _SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9_.-]+")

    def __init__(self, output_dir: str = "data/temp/latex"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._counter = 0
        self._cache: dict = {}
        self._cache_lock = threading.Lock()
        self._render_lock = _MPL_LOCK

        # Настройки matplotlib для красивых формул
        plt.rcParams.update({
            'text.usetex': False,  # Не использовать внешний LaTeX
            'mathtext.fontset': 'stix',  # Красивые математические шрифты
            'font.size': 14,
        })

    # ------------------------------------------------------------------
    # REFACTORING (ТЗ п.2): AMS-LaTeX алиасы для mathtext.
    # matplotlib mathtext НЕ поддерживает часть стандартных AMS-команд
    # (\implies, \impliedby, \leqslant, \geqslant, \iiint, \bigcup...);
    # привозим их к поддерживаемым формам ДО рендера. MathML-путь Flet
    # получает оригинальный LaTeX — там поддержка полная.
    _AMS_ALIASES = {
        "\\implies": "\\;\\Rightarrow\\;",
        "\\impliedby": "\\;\\Leftarrow\\;",
        "\\iff": "\\;\\Leftrightarrow\\;",
        "\\leqslant": "\\leq",
        "\\geqslant": "\\geq",
        "\\lesssim": "\\ll",
        "\\iiint": "\\int\\!\\!\\int\\!\\!\\int",
        "\\idotsint": "\\cdots",
        "\\coloneqq": ":=",
        "\\eqcirc": "=",
        "\\numberthis": "",
    }

    @classmethod
    def resolve_ams(cls, latex: str) -> str:
        """Заменяет AMS-синтаксис на mathtext-совместимые эквиваленты."""
        out = latex or ""
        for src_cmd, dst in cls._AMS_ALIASES.items():
            if src_cmd in out:
                out = out.replace(src_cmd, dst)
        return out

    # ------------------------------------------------------------------
    @staticmethod
    def normalize_latex(latex: str) -> str:
        """FIX (формулы не рендерились): приводит «плоский» текст к mathtext.

        Мелкие LLM часто выводят греческие/математические символы юникодом
        (∀ ∃ ∈ ⊆ ∪ ∩ ∅ → ⇒ ≤ ≈ ℝ), а matplotlib mathtext понимает только
        LaTeX-команды (\\forall, \\in ...). Без нормализации рендер падал и
        вместо формул появлялись чёрные квадраты либо сырой текст.
        Также чинятся типовые артефакты OCR/LLM (sin x -> \\sin x,
        lim x->0 -> \\lim_{x \\to 0}).
        """
        repl = {
            "∀": r"\forall ", "∃": r"\exists ", "¬": r"\neg ", "∧": r"\land ",
            "∨": r"\lor ", "→": r"\to ", "⇒": r"\Rightarrow ", "⇔": r"\Leftrightarrow ",
            "↔": r"\leftrightarrow ", "∈": r"\in ", "∉": r"\notin ", "∋": r"\ni ",
            "⊂": r"\subset ", "⊆": r"\subseteq ", "⊃": r"\supset ", "⊇": r"\supseteq ",
            "∪": r"\cup ", "∩": r"\cap ", "∅": r"\emptyset ", "∖": r"\setminus ",
            "≤": r"\leq ", "≥": r"\geq ", "≠": r"\neq ", "≈": r"\approx ",
            "≡": r"\equiv ", "∼": r"\sim ", "∝": r"\propto ",
            "×": r"\times ", "·": r"\cdot ", "÷": r"\div ", "±": r"\pm ",
            "∞": r"\infty ", "∑": r"\sum ", "∏": r"\prod ", "∫": r"\int ",
            "∮": r"\oint ", "√": r"\sqrt", "∂": r"\partial ", "∇": r"\nabla ",
            "ℝ": r"\mathbb{R}", "ℕ": r"\mathbb{N}", "ℤ": r"\mathbb{Z}",
            "ℚ": r"\mathbb{Q}", "ℂ": r"\mathbb{C}",
            "α": r"\alpha", "β": r"\beta", "γ": r"\gamma", "δ": r"\delta",
            "ε": r"\varepsilon", "ζ": r"\zeta", "η": r"\eta", "θ": r"\theta",
            "λ": r"\lambda", "μ": r"\mu", "ν": r"\nu", "ξ": r"\xi",
            "π": r"\pi", "ρ": r"\rho", "σ": r"\sigma", "τ": r"\tau",
            "φ": r"\varphi", "χ": r"\chi", "ψ": r"\psi", "ω": r"\omega",
            "Γ": r"\Gamma", "Δ": r"\Delta", "Θ": r"\Theta", "Λ": r"\Lambda",
            "Ξ": r"\Xi", "Π": r"\Pi", "Σ": r"\Sigma", "Φ": r"\Phi",
            "Ψ": r"\Psi", "Ω": r"\Omega",
        }
        s = "".join(repl.get(ch, ch) for ch in (latex or ""))
        # FIX (скриншоты 1–2, «съехавшая кодировка» токенизатора Qwen):
        # модель разбивает числа и аргументы пробелами внутри формул:
        # «1 8 0 ^ \circ», «f ^ { \left( n + 1 \right)}». Чиним ТОЛЬКО
        # одиночные цифры в плотной math-окружающей (сосед — цифра или
        # math-знак), чтобы не сливать перечисления («варианты 1, 2, 3»).
        prev = None
        while prev != s:
            prev = s
            s = re.sub(r"(?<=\d)\s(?=\d)", "", s)
            s = re.sub(r"(?<=[_^{}])\s+(?=\d)", "", s)
            s = re.sub(r"(?<=\d)\s+(?=[_^{}\\])", "", s)
        # двойные пробелы после подстановок с хвостовым пробелом
        s = re.sub(r"\s{2,}", " ", s)
        # названия функций без обратного слэша (OCR/LLM артефакты)
        s = re.sub(r"(?<!\\)\b(sin|cos|tg|ctg|arcsin|arccos|arctg|log|ln|lim|max|min|det)\b",
                   r"\\\1", s)
        # lim x->0 -> \lim_{x \to 0} (частый плоский вывод мелких моделей)
        s = re.sub(r"\\lim\s*([a-zA-Z])\s*->\s*(\S)", r"\\lim_{\1 \\to \2}", s)
        # matplotlib mathtext НЕ знает часть стандартных LaTeX-команд — иначе
        # рендер падает с ParseFatalException. Заменяем на поддерживаемые.
        _mathtext_alias = {
            "\\lor": "\\vee", "\\land": "\\wedge",
            "\\le": "\\leq", "\\ge": "\\geq", "\\ne": "\\neq",
            "\\leqslant": "\\le", "\\geqslant": "\\ge",
            "\\gets": "\\leftarrow", "\\colon": ":",
            "\\bigl": "", "\\bigr": "", "\\Bigl": "", "\\Bigr": "",
            "\\bigcup": "\\cup", "\\bigcap": "\\cap",
            "\\ldots": "\\dots", "\\lvert": "|", "\\rvert": "|",
            "\\vert": "|", "\\mid": "|",
        }
        # unsupported env -> readable inline form (матhtext не знает \\begin)
        m = re.search(r"\\(?:big)?begin\{(cases|aligned|array\{[^}]*\})\}(.+?)\\(?:big)?end\{\1\}",
                      s, flags=re.DOTALL)
        if m:
            body = m.group(2).replace("\\\\", "; ").replace("&", " , ")
            s = (s[:m.start()] + body + s[m.end():]).strip()
        for cmd, rep in _mathtext_alias.items():
            s = re.sub(re.escape(cmd) + r"(?![A-Za-z])", lambda m, r=rep: r, s)
        # mathtext не знает \setminus — заменяем на поддерживаемый \backslash
        s = re.sub(r"\\setminus(?![A-Za-z])", lambda m: " \\backslash ", s)
        s = re.sub(r"\s{2,}", " ", s).strip()
        return s

    # ------------------------------------------------------------------
    def render_formula(self, latex: str, filename: str = None,
                       color: str = "#111111", *, dpi: int = 200,
                       fontsize: float = 17.0, opaque: bool = False) -> str:
        """
        Рендерит LaTeX формулу в PNG файл.

        FIX (проблема «чёрные квадраты»): раньше каждая формула рисовалась на
        холсте figsize=(8,2) с белой подложкой — при height=20 в UI это был
        тёмный прямоугольник с огромными полями, разрывающими текст. Теперь
        размер холста подбирается по длине формулы, фон полностью прозрачный,
        подложка убрана, добавлен кэш (повторные формулы не рендерятся заново).

        REFACTORING (ТЗ п.1): цвет глифа параметризован — тёмная тема чата
        требует светлый (#E0E0E0), светлая тема библиотеки — чёрный. Прозрачный
        фон (transparent=True) делает картинку адаптивной к любому bgcolor.
        Кэш учитывает цвет (одна формула в двух темах — два файла).

        Args:
            latex: LaTeX выражение (без $...$)
            filename: Имя файла (генерируется автоматически если None)
            color: цвет линий формулы (по умолчанию тёмный)

        Returns:
            Путь к созданному PNG файлу
        """
        clean = self.resolve_ams(self.normalize_latex((latex or "").strip()))
        if not clean:
            return None

        cache_key = (clean, color, dpi, opaque)
        with self._cache_lock:
            hit = self._cache.get(cache_key)
        if hit and Path(hit).exists():
            return hit

        if filename is None:
            self._counter += 1
            safe = self._SAFE_NAME_RE.sub("_", clean)[:40].strip("_") or "f"
            tag = "" if (dpi == 200 and not opaque) else f"_{dpi}{'o' if opaque else ''}"
            filename = f"formula_{self._counter}_{safe}{tag}.png"

        output_path = self.output_dir / filename

        try:
            # Компактный холст: ширина ~ по длине формулы (inline-рендер
            # без разрывов текста и без чёрных плашек).
            width = max(1.2, min(8.0, 0.13 * len(clean) + 0.7))
            with _MPL_LOCK:  # потокобезопасность pyplot
                fig, ax = plt.subplots(figsize=(width, 0.6), dpi=dpi)
                ax.axis("off")
                t = ax.text(0.5, 0.5, f"${clean}$",
                            fontsize=fontsize,
                            color=color,
                            ha="center",
                            va="center")
                try:
                    # Валидация непустого bbox: раньше битая формула давала
                    # PNG нулевой видимой области — в UI «пустые скобки ()».
                    fig.canvas.draw()
                    bb = t.get_window_extent(
                        renderer=fig.canvas.get_renderer())
                    if bb.width <= 1 or bb.height <= 1:
                        raise RuntimeError("пустой bbox формулы")
                    fig.savefig(str(output_path),
                                bbox_inches="tight",
                                pad_inches=0.03,
                                transparent=not opaque,
                                dpi=dpi)
                finally:
                    plt.close(fig)

            if not output_path.exists() or output_path.stat().st_size == 0:
                return None

            with self._cache_lock:
                self._cache[cache_key] = str(output_path)
            return str(output_path)

        except Exception as e:
            # FIX: битая/неподдерживаемая команда больше не даёт «чёрный
            # квадрат» и не теряет формулу — рисуем её как обычный текст.
            logger.debug(f"Mathtext-рендер не удался ('{latex}'): {e} — "
                         f"фолбэк plain-текстом")
            with _MPL_LOCK:
                plt.close("all")
            try:
                width = max(1.2, min(8.0, 0.11 * len(clean) + 0.7))
                with _MPL_LOCK:
                    fig, ax = plt.subplots(figsize=(width, 0.5), dpi=dpi)
                    ax.axis("off")
                    ax.text(0.5, 0.5, clean.replace("\\", ""), fontsize=14,
                            color=color, ha="center", va="center")
                    fig.savefig(str(output_path), bbox_inches="tight",
                                pad_inches=0.03,
                                transparent=not opaque, dpi=dpi)
                    plt.close(fig)
                if output_path.exists() and output_path.stat().st_size > 0:
                    with self._cache_lock:
                        self._cache[cache_key] = str(output_path)
                    return str(output_path)
            except Exception as e2:
                logger.debug(f"Plain-фолбэк формулы тоже не удался: {e2}")
                plt.close("all")
            return None
    
    def render_to_base64(self, latex: str) -> str:
        """Рендерит формулу и возвращает base64 строку."""
        path = self.render_formula(latex)
        if path and Path(path).exists():
            with open(path, 'rb') as f:
                return base64.b64encode(f.read()).decode('utf-8')
        return None
    
    def clear_cache(self):
        """Очищает кэш отрендеренных формул."""
        for f in self.output_dir.glob("formula_*.png"):
            f.unlink()
        self._counter = 0


    def render_formula_inline(self, latex: str, color: str = "#000000") -> str:
        """PDF-вариант инлайн-формулы: 300 dpi, ЧЁРНЫЙ текст на БЕЛОМ фоне
        (высокий контраст, fpdf2 не поддерживает альфу корректно).

        ТЗ п.4: вызывается из export_to_pdf для каждой $...$-токен-формулы;
        возвращает путь к PNG или None (тогда экспорт печатает юникод-текст).
        """
        return self.render_formula(latex, color=color, dpi=300,
                                   fontsize=14.0, opaque=True)


_RENDERER_SINGLETON: "LatexRenderer | None" = None
_RENDERER_LOCK = threading.Lock()


def get_renderer(output_dir: str = "data/temp/latex") -> "LatexRenderer":
    """Потокобезопасный ленивый синглтон рендера (общий кэш для UI и PDF)."""
    global _RENDERER_SINGLETON
    if _RENDERER_SINGLETON is None:
        with _RENDERER_LOCK:
            if _RENDERER_SINGLETON is None:
                _RENDERER_SINGLETON = LatexRenderer(output_dir=output_dir)
    return _RENDERER_SINGLETON
