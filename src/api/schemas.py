from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field


class PatientInput(BaseModel):
    Fever_Duration: int = Field(..., ge=0, le=30, description="Días desde el inicio de la fiebre")
    Age: float = Field(..., gt=0, le=120, description="Edad del paciente en años")
    Headache: int = Field(..., ge=0, le=1, description="Cefalea (0: no, 1: sí)")
    Myalgia: int = Field(..., ge=0, le=1, description="Mialgia (0: no, 1: sí)")
    Rash: int = Field(..., ge=0, le=1, description="Exantema / Rash (0: no, 1: sí)")
    Fever: int = Field(..., ge=0, le=1, description="Fiebre (0: no, 1: sí)")


class SinglePredictionResponse(BaseModel):
    status: str = "ok"
    model_used: str = "CatBoost + K-Modes"
    probability: float
    optimal_threshold: float
    prediction: int
    diagnosis_suggested: str
    clinical_priority: str
    lab_recommendation: str
    assigned_cluster: Optional[int] = None
    cluster_profile: Optional[str] = None
    cluster_model_used: Optional[str] = "K-Modes (Solo Síntomas)"
    disclaimer: str = (
        "Herramienta orientativa de apoyo a la decisión clínica para dengue en Argentina. "
        "No diagnostica, no descarta ni reemplaza la evaluación médica profesional ni las pruebas confirmatorias de laboratorio."
    )


class HealthResponse(BaseModel):
    status: str
    model_registry: Dict[str, Any]
    version: str
