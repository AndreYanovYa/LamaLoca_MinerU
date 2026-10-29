# src/processing/entity_extractor.py
"""
Этап 1 Graph-RAG: разметка математических сущностей и онтологических связей.

Модуль определяет тип сущности (entity_type) для каждого чанка текста учебника
ПМИ и строит связи между сущностями (REFERENCES / PROVES), которые сохраняются
в таблицу SQLite `entity_relations`.

Дизайн-решения (важно для совместимости):
- Тип сущности НЕ добавляется отдельным столбцом в таблицу blocks — схема БД
  не меняется (жёсткое ограничение предыдущих ТЗ). entity_type хранится внутри
  JSON-колонки metadata таблицы entity_relations и вычисляется детерминированно
  по тексту, поэтому его можно пересчитать в любой момент ("ленивая" разметка).
- Связи строятся эвристикой по нумерации («Теорема 2 ... Определение 1» ->
  REFERENCES; «Доказательство» после теоремы -> PROVES). Эвристика намеренно
  консервативна: лучше меньше рёбер, чем ложные циклы в графе.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

from loguru import logger

# ---------------------------------------------------------------------------
# Типы сущностей (онтология ПМИ)
# ---------------------------------------------------------------------------
DEFINITION = "DEFINITION"
THEOREM = "THEOREM"
PROOF = "PROOF"
AXIOM = "AXIOM"
LEMMA = "LEMMA"
EXPLANATION = "EXPLANATION"

ENTITY_TYPES = (DEFINITION, THEOREM, PROOF, AXIOM, LEMMA, EXPLANATION)

# Relations (Ontology Links)
R_REFERENCES = "REFERENCES"
R_PROVES = "PROVES"

# Маркеры начала сущностей в русском математическом тексте (Кудрявцев,
# Фихтенгольц и т.п.). Регистронезависимо; поддерживаем и верхний регистр OCR.
_ENTITY_PATTERNS: List[Tuple[str, str]] = [
    (THEOREM, r"\bтеорем[а-я]*\b"),
    (LEMMA, r"\bлемм[а-я]*\b"),
    (DEFINITION, r"\bопределени[а-я]*\b"),
    (AXIOM, r"\bаксиом[а-я]*\b"),
    (PROOF, r"\bдоказат[а-я]*\b|\bдокажем\b|\bв\s*сам\s*дел[а-я]*\b"),
]

# Числа/номера в заголовке сущности: «Теорема 2.1», «Определение 1», «§ 13».
_NUM_RE = re.compile(
    r"(?:теорем\w*|лемм\w*|определени\w*|аксиом\w*|следств\w*|принцип\w*)\s*"
    r"(\d+(?:\.\d+)*)",
    re.IGNORECASE,
)

_REF_RE = re.compile(
    r"(теорем\w*|лемм\w*|определени\w*|аксиом\w*|следств\w*|раздел\w*|"
    r"пункт\w*|глав\w*|рис\w*|формул\w*|пример\w*)\s*(?:No\.?|№)?\s*"
    r"(\d+(?:[\.\s]\d+)?)",
    re.IGNORECASE,
)


def classify_entity(text: str) -> str:
    """Определяет entity_type фрагмента текста по его началу.

    Возвращает EXPLANATION, если явных маркеров нет — это связующий авторский
    текст, который тоже участвует в графе как вершина.
    """
    if not text:
        return EXPLANATION
    head = text[:200].lower()
    # Доказательство обычно начинается со слова «Доказательство.» или
    # «Докажем...» — проверяем раньше теоремы, чтобы не отнести его к THEOREM.
    if re.search(r"^[\s\-–—]*(доказательств|докажем|докажем\b)", head):
        return PROOF
    best_type: Optional[str] = None
    best_pos = len(head) + 1
    for etype, pattern in _ENTITY_PATTERNS:
        m = re.search(pattern, head)
        if m and m.start() < best_pos:
            # «теорема Пифагора» внутри объяснения — не сигнал типа,
            # если маркер не в начале фразы и далеко от старта.
            best_type, best_pos = etype, m.start()
    if best_type is None or best_pos > 60:
        return EXPLANATION
    return best_type


def extract_number(text: str) -> Optional[str]:
    """Извлекает номер сущности из заголовка («Теорема 2.1» -> '2.1')."""
    m = _NUM_RE.search(text[:200])
    return m.group(1) if m else None


def extract_references(text: str) -> List[Tuple[str, str]]:
    """Находит упоминания других сущностей вида [('ТЕОРЕМА', '2'), ...]."""
    refs: List[Tuple[str, str]] = []
    seen = set()
    for kind, num in _REF_RE.findall(text or ""):
        kind_l = kind.lower()
        if kind_l.startswith(("теорем",)):
            etype = THEOREM
        elif kind_l.startswith(("лемм",)):
            etype = LEMMA
        elif kind_l.startswith(("определени",)):
            etype = DEFINITION
        elif kind_l.startswith(("аксиом",)):
            etype = AXIOM
        else:
            continue  # раздел/глава/рисунки — вне онтологии сущностей
        key = (etype, num.strip().rstrip("."))
        if key not in seen:
            seen.add(key)
            refs.append(key)
    return refs


def build_relations(
    chunks: List[Dict],
) -> Tuple[List[Dict], Dict[str, str]]:
    """Строит рёбра графа знаний по списку чанков [{block_id, text}, ...].

    Правила:
    - REFERENCES: сущность A ссылается на B, если в тексте A есть упоминание
      «<Тип> N» и в корпусе существует сущность того же типа с номером N
      (или B — определение без номера рядом — не используем, слишком шумно).
    - PROOF->THEOREM: чанк-доказательство привязывается к ближайшей предшествующей
      теореме/лемме в порядке следования блоков (это и есть жёсткая связь
      доказательства со своей теоремой).

    Args:
        chunks: список словарей с ключами block_id и text (в порядке книги).

    Returns:
        (relations, types): relations — список словарей для entity_relations,
        types — соответствие block_id -> entity_type.
    """
    types: Dict[str, str] = {}
    numbers: Dict[str, Optional[str]] = {}
    by_key: Dict[Tuple[str, str], str] = {}

    for ch in chunks:
        bid = ch.get("block_id") or ch.get("chunk_id")
        if not bid:
            continue
        text = ch.get("text", "") or ""
        et = classify_entity(text)
        types[bid] = et
        if et != EXPLANATION:
            num = extract_number(text)
            numbers[bid] = num
            if num:
                by_key.setdefault((et, num.rstrip(".")), bid)

    relations: List[Dict] = []
    last_formal: Optional[Tuple[str, str]] = None  # (block_id, type) теорема/лемма
    for ch in chunks:
        bid = ch.get("block_id") or ch.get("chunk_id")
        if not bid:
            continue
        text = ch.get("text", "") or ""
        et = types.get(bid, EXPLANATION)

        # Присваиваем доказательство последней теореме/лемме до него.
        if et == PROOF and last_formal and last_formal[1] in (THEOREM, LEMMA):
            relations.append({
                "source_chunk_id": bid,
                "target_chunk_id": last_formal[0],
                "relation_type": R_PROVES,
            })
        if et in (THEOREM, LEMMA):
            last_formal = (bid, et)

        # Транзитивные ссылки на другие сущности.
        own_num = (numbers.get(bid) or "").rstrip(".")
        for ref_type, ref_num in extract_references(text):
            target = by_key.get((ref_type, ref_num))
            if target and target != bid:
                relations.append({
                    "source_chunk_id": bid,
                    "target_chunk_id": target,
                    "relation_type": R_REFERENCES,
                })

    # Дедупликация рёбер
    uniq, seen_pairs = [], set()
    for r in relations:
        key = (r["source_chunk_id"], r["target_chunk_id"], r["relation_type"])
        if key not in seen_pairs:
            seen_pairs.add(key)
            uniq.append(r)

    logger.debug(
        f"Graph-RAG: {len(chunks)} чанков → {len(uniq)} рёбер, "
        f"{sum(1 for t in types.values() if t != EXPLANATION)} формальных сущностей"
    )
    return uniq, types
