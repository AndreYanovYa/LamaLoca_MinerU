# src/utils/hashing.py
import hashlib
from pathlib import Path

def compute_file_sha256(file_path: Path, chunk_size: int = 8192) -> str:
    """
    Вычисляет SHA-256 хеш содержимого файла.
    Используется как стабильный doc_id: одинаковые файлы = одинаковый ID.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Файл не найден: {file_path}")
    
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            sha256.update(chunk)
    return sha256.hexdigest()