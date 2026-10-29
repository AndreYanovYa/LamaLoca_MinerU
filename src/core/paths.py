# src/core/paths.py
import os
from pathlib import Path
from platformdirs import user_cache_dir

class Paths:
    """Централизованное управление путями приложения."""
    
    def __init__(self, config_root: str):
        self.project_root = Path(__file__).resolve().parent.parent.parent
        self.config_root = Path(config_root).resolve()
        
        self.data_dir = self.config_root
        self.library_dir = self.data_dir / "library"
        self.parsed_dir = self.data_dir / "parsed"
        self.storage_dir = self.data_dir / "storage"
        self.backups_dir = self.data_dir / "backups"
        self.logs_dir = self.data_dir / "logs"
        
        self.cache_dir = Path(user_cache_dir("lama-loca", "Woodpecker_Prime"))
        self.ocr_cache_dir = self.cache_dir / "ocr"
        
        self._ensure_dirs_exist()

    def _ensure_dirs_exist(self) -> None:
        for dir_path in [self.library_dir, self.parsed_dir, self.storage_dir, 
                         self.backups_dir, self.logs_dir, self.ocr_cache_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

    def get_doc_parsed_dir(self, doc_id: str) -> Path:
        path = self.parsed_dir / doc_id
        path.mkdir(parents=True, exist_ok=True)
        return path