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
        self.kmodes_sintomas_model = None
        self.kmodes_sintomas_columns = []
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

        # Buscar KModes (con Outcome)
        km_info = self.metadata.get("models", {}).get("kmodes", {})
        km_filename = km_info.get("artifact_file", "modelo_kmodes.joblib")
        km_path = ARTIFACTS_DIR / km_filename
        if not km_path.exists():
            fallback = km_info.get("fallback_artifact_file", "kmodes_dengue.joblib")
            km_path = ARTIFACTS_DIR / fallback
            if not km_path.exists():
                raise FileNotFoundError(f"No se encontró el artefacto de KModes en: {ARTIFACTS_DIR}")

        # Buscar KModes Solo Síntomas (para triage sin requerir Outcome)
        kms_info = self.metadata.get("models", {}).get("kmodes_sintomas", {})
        kms_filename = kms_info.get("artifact_file", "modelo_kmodes_sinoutcoome.joblib")
        kms_path = ARTIFACTS_DIR / kms_filename
        if not kms_path.exists():
            fallback = kms_info.get("fallback_artifact_file", "modelo_kmodes_sintomas.joblib")
            kms_path = ARTIFACTS_DIR / fallback

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

        # Cargar K-Modes Solo Síntomas si existe
        if kms_path.exists():
            self.artifact_hashes["kmodes_sintomas"] = compute_sha256(kms_path)
            raw_kms = joblib.load(kms_path)
            if isinstance(raw_kms, dict) and "modelo" in raw_kms:
                self.kmodes_sintomas_model = raw_kms["modelo"]
                self.kmodes_sintomas_columns = raw_kms.get("columnas", ["Fever", "Headache", "Myalgia", "Rash"])
            else:
                self.kmodes_sintomas_model = raw_kms
                self.kmodes_sintomas_columns = ["Fever", "Headache", "Myalgia", "Rash"]

        self.loaded = True

    def get_status(self) -> Dict[str, Any]:
        models_status = {
            "catboost": {
                "name": "CatBoost Classifier (Supervisado)",
                "loaded": self.catboost_model is not None,
                "type": str(type(self.catboost_model).__name__),
                "sha256": self.artifact_hashes.get("catboost"),
                "threshold": getattr(self.catboost_model, "best_threshold_", None),
                "utility": "Predicción probabilística y clasificación binaria de riesgo de dengue."
            },
            "kmodes": {
                "name": "K-Modes Clustering (Interpretabilidad con Outcome)",
                "loaded": self.kmodes_model is not None,
                "type": str(type(self.kmodes_model).__name__),
                "sha256": self.artifact_hashes.get("kmodes"),
                "n_clusters": getattr(self.kmodes_model, "n_clusters", None),
                "features": self.kmodes_columns,
                "utility": "Interpretabilidad y agrupamiento fenotípico sobre hold-out incluyendo diagnóstico."
            }
        }

        if self.kmodes_sintomas_model is not None:
            models_status["kmodes_sintomas"] = {
                "name": "K-Modes Clustering (Solo Síntomas / Triage Fenotípico)",
                "loaded": True,
                "type": str(type(self.kmodes_sintomas_model).__name__),
                "sha256": self.artifact_hashes.get("kmodes_sintomas"),
                "n_clusters": getattr(self.kmodes_sintomas_model, "n_clusters", None),
                "features": self.kmodes_sintomas_columns,
                "utility": "Permite, dado un paciente febril, ubicarlo directamente en un cluster fenotípico a partir de sus síntomas observados ('Fever', 'Headache', 'Myalgia', 'Rash'), sin requerir diagnóstico de laboratorio (Outcome)."
            }

        return {
            "loaded": self.loaded,
            "models": models_status,
            "version": self.metadata.get("version", "1.0.0")
        }
