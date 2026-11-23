import hashlib
import json
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import joblib

ARTIFACTS_DIR = Path(__file__).resolve().parent.parent.parent / "artifacts"
METADATA_FILE = ARTIFACTS_DIR / "metadata.json"


def compute_sha256(file_path: Path) -> str:
    """Calcula el hash SHA-256 de un archivo."""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(65536), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


class ModelRegistry:
    _instance: Optional["ModelRegistry"] = None

    def __init__(self):
        self.catboost_model = None
        self.kmodes_model = None
        self.metadata: Dict[str, Any] = {}
        self.artifact_hashes: Dict[str, str] = {}
        self.loaded: bool = False

    @classmethod
    def get_instance(cls) -> "ModelRegistry":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def load_artifacts(self) -> None:
        """Carga los modelos .joblib y metadata desde artifacts/."""
        if not METADATA_FILE.exists():
            raise FileNotFoundError(f"No se encontró el archivo de metadatos: {METADATA_FILE}")

        with open(METADATA_FILE, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        # Buscar CatBoost
        cb_info = self.metadata.get("models", {}).get("catboost", {})
        cb_filename = cb_info.get("artifact_file", "modelo_catboost.joblib")
        cb_path = ARTIFACTS_DIR / cb_filename
        if not cb_path.exists():
            fallback = cb_info.get("fallback_artifact_file", "catboost_dengue.joblib")
            cb_path = ARTIFACTS_DIR / fallback
            if not cb_path.exists():
                raise FileNotFoundError(f"No se encontró el artefacto de CatBoost en: {ARTIFACTS_DIR}")

        # Buscar KModes
        km_info = self.metadata.get("models", {}).get("kmodes", {})
        km_filename = km_info.get("artifact_file", "modelo_kmodes.joblib")
        km_path = ARTIFACTS_DIR / km_filename
        if not km_path.exists():
            fallback = km_info.get("fallback_artifact_file", "kmodes_dengue.joblib")
            km_path = ARTIFACTS_DIR / fallback
            if not km_path.exists():
                raise FileNotFoundError(f"No se encontró el artefacto de KModes en: {ARTIFACTS_DIR}")

        # Hashes de integridad
        self.artifact_hashes["catboost"] = compute_sha256(cb_path)
        self.artifact_hashes["kmodes"] = compute_sha256(km_path)

        # Cargar con joblib
        self.catboost_model = joblib.load(cb_path)
        raw_kmodes = joblib.load(km_path)
        if isinstance(raw_kmodes, dict) and "modelo" in raw_kmodes:
            self.kmodes_model = raw_kmodes["modelo"]
            self.kmodes_columns = raw_kmodes.get("columnas", ["Fever", "Headache", "Myalgia", "Rash", "Outcome"])
        else:
            self.kmodes_model = raw_kmodes
            self.kmodes_columns = ["Fever", "Headache", "Myalgia", "Rash", "Outcome"]
        self.loaded = True

    def get_status(self) -> Dict[str, Any]:
        return {
            "loaded": self.loaded,
            "models": {
                "catboost": {
                    "loaded": self.catboost_model is not None,
                    "type": str(type(self.catboost_model).__name__),
                    "sha256": self.artifact_hashes.get("catboost"),
                    "threshold": getattr(self.catboost_model, "best_threshold_", None)
                },
                "kmodes": {
                    "loaded": self.kmodes_model is not None,
                    "type": str(type(self.kmodes_model).__name__),
                    "sha256": self.artifact_hashes.get("kmodes"),
                    "n_clusters": getattr(self.kmodes_model, "n_clusters", None)
                }
            },
            "version": self.metadata.get("version", "1.0.0")
        }
