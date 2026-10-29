# src/storage/chroma_store.py
import json

import chromadb
from chromadb.config import Settings as ChromaSettings
from pathlib import Path
from loguru import logger
from typing import List, Dict, Any, Callable, Optional
from tqdm import tqdm


def sanitize_metadata(meta: Dict[str, Any]) -> Dict[str, Any]:
    """
    Приводит метаданные к типу, допустимому ChromaDB.

    Chroma принимает только str/int/float/bool/SparseVector/list/None.
    Вложенные dict (например координаты блока PDF {'x0':..,'y0':..}) раньше
    роняли импорт с ошибкой "Expected metadata value to be a str, int, float,
    bool ... got {...} which is a dict". Такие значения сериализуются в JSON-
    строку — данные не теряются и остаются фильтруемыми/отображаемыми.
    None также конвертируется в "" — некоторые сборки Chroma (rust-биндинг)
    отказываются принимать null-значение поля метаданных.
    """
    clean: Dict[str, Any] = {}
    for k, v in (meta or {}).items():
        if v is None:
            clean[k] = ""
        elif isinstance(v, (str, int, float, bool)):
            clean[k] = v
        elif isinstance(v, list):
            # список должен быть однородным по скалярным типам
            kinds = {type(x) for x in v if not isinstance(x, (str, int, float, bool))}
            if kinds:
                clean[k] = json.dumps(v, ensure_ascii=False, default=str)
            else:
                clean[k] = v
        elif isinstance(v, dict):
            clean[k] = json.dumps(v, ensure_ascii=False, default=str)
        else:
            clean[k] = str(v)
    return clean


class ChromaStore:
    """Обёртка над ChromaDB с батчингом для больших коллекций."""
    
    # ChromaDB имеет лимит на размер батча (~5461). Используем безопасный размер.
    MAX_BATCH_SIZE = 1000
    
    def __init__(self, persist_directory: str):
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        
        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=ChromaSettings(anonymized_telemetry=False)
        )
        
        self.collection_name = "document_chunks"
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        logger.info(f"ChromaDB инициализирована: {self.persist_directory}")
    
    def add_chunks(
        self,
        ids: List[str],
        texts: List[str],
        embeddings: List[List[float]],
        metadatas: List[Dict[str, Any]],
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ):
        """Добавляет чанки в коллекцию с батчингом.

        :param progress_callback: необязательная функция (done, total) —
                                  вызывается после каждого батча для UI-прогресса.
        """
        if not texts:
            return
        
        # Проверяем, что все списки одной длины
        if not (len(ids) == len(texts) == len(embeddings) == len(metadatas)):
            raise ValueError(
                f"Несоответствие размеров: ids={len(ids)}, texts={len(texts)}, "
                f"embeddings={len(embeddings)}, metadatas={len(metadatas)}"
            )
        
        total = len(texts)
        batches_count = (total + self.MAX_BATCH_SIZE - 1) // self.MAX_BATCH_SIZE
        
        logger.info(f"Сохранение {total} чанков в ChromaDB ({batches_count} батчей по {self.MAX_BATCH_SIZE})")
        
        # Разбиваем на батчи по MAX_BATCH_SIZE
        for i in tqdm(range(0, total, self.MAX_BATCH_SIZE), desc="ChromaDB", total=batches_count):
            end = min(i + self.MAX_BATCH_SIZE, total)
            self.collection.add(
                ids=ids[i:end],
                documents=texts[i:end],
                embeddings=embeddings[i:end],
                # АУДИТ: метаданные проходят санитайз — dict-значения
                # (coordinates и т.п.) роняли add() с TypeError от Chroma.
                metadatas=[sanitize_metadata(m) for m in metadatas[i:end]]
            )
            if progress_callback:
                try:
                    progress_callback(end, total)
                except Exception:
                    pass  # прогресс не должен ронять индексацию
        
        logger.info(f"✅ Добавлено {total} чанков в ChromaDB")
    
    def delete_by_doc_id(self, doc_id: str):
        """Удаляет все чанки, принадлежащие документу."""
        self.collection.delete(where={"doc_id": doc_id})
        logger.info(f"Удалены чанки для doc_id: {doc_id}")
    
    def search(
        self,
        query_embedding: List[float],
        top_k: int,
        filter_dict: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """Ищет похожие чанки по вектору запроса."""
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=filter_dict,
            include=["documents", "metadatas", "distances"]
        )
        
        formatted_results = []
        if results['ids'] and results['ids'][0]:
            for i in range(len(results['ids'][0])):
                formatted_results.append({
                    "id": results['ids'][0][i],
                    "text": results['documents'][0][i],
                    "metadata": results['metadatas'][0][i],
                    "distance": results['distances'][0][i]
                })
        
        return formatted_results