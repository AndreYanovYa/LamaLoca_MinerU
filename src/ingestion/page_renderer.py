# src/ingestion/page_renderer.py
"""Ленивый рендеринг зон страницы PDF в PNG (Level 1 архитектуры ТЗ).

Когда текстовый слой PyMuPDF склеивает встроенную сложную графику
(формулы-картинки, схемы, сплющенные LaTeX-блоки) в нечитаемую «кашу»,
парсер активирует этот модуль: зона блока вырезается и рендерится в PNG
высокого разрешения. Текст при этом НЕ выбрасывается — векторный поиск
продолжает работать по текстовому слою, PNG служит зрительным слоем для
viewer'а.

Модуль ленивый: импортируется и вызывается только при срабатывании
детектора «каши» в pdf_parser, а не на каждой странице.
"""
from pathlib import Path
from typing import Iterable, Optional, Tuple

from loguru import logger

Rect = Tuple[float, float, float, float]

# Кэш уже отрендеренных зон: (pdf_path, page_num, rect) -> str(png path).
# Потокобезопасен для нашего однопоточного импорта; dict-запись атомарна в CPython.
_ZONE_CACHE: dict = {}


def _safe_dir(out_dir: Path) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def render_zone_png(page, rect: Rect, out_dir: Path, filename: str,
                    dpi: int = 300) -> Optional[str]:
    """Рендерит прямоугольную зону страницы в PNG. Возвращает путь или None.

    page:   pymupdf.Page
    rect:   (x0, y0, x1, y1) в координатах страницы (points)
    """
    try:
        import pymupdf  # локальный импорт: модуль опционален для EPUB-пайплайна
    except ImportError:
        logger.debug("PyMuPDF недоступен — ленивый PNG пропущен")
        return None

    key = (str(getattr(getattr(page, "parent", None), "name", "")),
           getattr(page, "number", -1), tuple(round(v, 1) for v in rect))
    if key in _ZONE_CACHE:
        return _ZONE_CACHE[key]

    try:
        x0, y0, x1, y1 = rect
        # УМНЫЙ РЕНДЕР: защита от некорректных («нулевых»/«сплющенных»)
        # координат. PyMuPDF отдаёт вырожденные bbox (x1<=x0 или y1<=y0) для
        # блоков, чей текст извлечён «в никуда» (формулы-картинки без вектор
        # -геометрии, нулевые глифы сканов). Пустой clip -> get_pixmap падал
        # с "height and width must be > 0", а инференс pix2tex по такому PNG
        # был бы пустой работой. Вырожденный rect восстанавливаем по границам
        # страницы; если зона всё ещё пуста — рендер НЕ выполняется (None),
        # и парсер не гоняет видеокарту впустую.
        if not all(isinstance(v, (int, float)) for v in (x0, y0, x1, y1)):
            return None
        # Нормализация порядка координат (инвариант к инвертированным bbox).
        if x1 < x0:
            x0, x1 = x1, x0
        if y1 < y0:
            y0, y1 = y1, y0
        pw, ph = float(page.rect.width), float(page.rect.height)
        if x1 <= x0 or y1 <= y0:
            # нулевая ширина/высота: расширяем до разумного минимума в пределах
            # страницы (типичная строка формулы ~ 14 pt высотой).
            if x1 <= x0:
                x0 = max(min(x0, pw - 1), 0.0)
                x1 = min(max(x0 + 120.0, 1.0), pw)
            if y1 <= y0:
                y0 = max(min(y0, ph - 1), 0.0)
                y1 = min(max(y0 + 14.0, 1.0), ph)
        clip = pymupdf.Rect(max(x0 - 2, 0), max(y0 - 2, 0),
                            min(x1 + 2, pw), min(y1 + 2, ph))
        if clip.width <= 0 or clip.height <= 0:
            logger.debug(f"Зона стр. {getattr(page, 'number', '?')} имеет "
                         f"нулевой размер после нормализации ({rect!r}) — "
                         f"рендер и инференс пропущены")
            return None
        zoom = dpi / 72.0
        mat = pymupdf.Matrix(zoom, zoom)
        if int(clip.width * zoom) <= 0 or int(clip.height * zoom) <= 0:
            return None
        pix = page.get_pixmap(matrix=mat, clip=clip, alpha=False)
        if pix.width <= 0 or pix.height <= 0:
            return None
        out_path = _safe_dir(Path(out_dir)) / filename
        pix.save(str(out_path))
        _ZONE_CACHE[key] = str(out_path)
        logger.debug(f"Зона стр. {key[1]} отрендерена в {out_path.name} ({dpi} dpi)")
        return str(out_path)
    except Exception as e:  # noqa: BLE001 — визуальный fallback не роняет импорт
        logger.debug(f"Ленивый PNG зоны не выполнен: {e}")
        return None


def render_page_png(page, out_dir: Path, filename: str,
                    dpi: int = 200) -> Optional[str]:
    """Рендер всей страницы в PNG (используется OCR-fallback и viewer'ом)."""
    try:
        import pymupdf
    except ImportError:
        return None
    try:
        zoom = dpi / 72.0
        pix = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), alpha=False)
        out_path = _safe_dir(Path(out_dir)) / filename
        pix.save(str(out_path))
        return str(out_path)
    except Exception as e:  # noqa: BLE001
        logger.debug(f"PNG страницы не выполнен: {e}")
        return None
