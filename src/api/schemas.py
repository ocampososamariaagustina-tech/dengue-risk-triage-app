from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field


class PatientInput(BaseModel):
    Fever_Duration: Optional[int] = Field(3, ge=0, le=30, description="Días desde el inicio de la fiebre")
    Age: Optional[float] = Field(30.0, gt=0, le=120, description="Edad del paciente en años")
    Headache: int = Field(..., ge=0, le=1, description="Cefalea (0: no, 1: sí)")
    Myalgia: int = Field(..., ge=0, le=1, description="Mialgia (0: no, 1: sí)")
    Rash: int = Field(..., ge=0, le=1, description="Exantema / Rash (0: no, 1: sí)")
    Fever: int = Field(..., ge=0, le=1, description="Fiebre (0: no, 1: sí)")
    model: Optional[str] = Field("both", description="Modelo a ejecutar: 'both', 'catboost' o 'kmodes'")


class SinglePredictionResponse(BaseModel):
    status: str = "ok"
    model_used: str = "CatBoost + K-Modes"
    probability: Optional[float] = None
    optimal_threshold: Optional[float] = None
    prediction: Optional[int] = None
    diagnosis_suggested: Optional[str] = None
    clinical_priority: Optional[str] = None
    lab_recommendation: Optional[str] = None
    assigned_cluster: Optional[int] = None
    cluster_profile: Optional[str] = None
    cluster_description: Optional[str] = None
    disclaimer: str = (
        "Herramienta orientativa de apoyo a la decisión clínica para dengue en Argentina. "
        "No diagnostica, no descarta ni reemplaza la evaluación médica profesional ni las pruebas confirmatorias de laboratorio."
    )


class HealthResponse(BaseModel):
    status: str
    model_registry: Dict[str, Any]
    version: str

