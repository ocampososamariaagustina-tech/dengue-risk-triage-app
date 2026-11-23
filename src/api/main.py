import io
from contextlib import asynccontextmanager
from typing import Dict, Any, Optional
import pandas as pd
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path

from src.ml.model_loader import ModelRegistry
from src.ml.metrics_evaluator import (
    validate_dataframe,
    validate_symptoms_dataframe,
    evaluate_catboost,
    evaluate_kmodes,
    evaluate_kmodes_sintomas,
    generate_predictions_table,
    generate_symptoms_predictions_table,
    FEATURES_CB,
    FEATURES_KM,
    FEATURES_SINTOMAS,
)
from src.api.schemas import PatientInput, SinglePredictionResponse, HealthResponse

WEB_DIR = Path(__file__).resolve().parent.parent / "web"
DEV_DIR = Path(__file__).resolve().parent.parent.parent / "dev"

# Cache para última evaluación generada (para descarga de CSV)
latest_evaluated_df: Optional[pd.DataFrame] = None
latest_symptoms_df: Optional[pd.DataFrame] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Cargar modelos al iniciar la aplicación
    registry = ModelRegistry.get_instance()
    try:
        registry.load_artifacts()
        print("Modelos CatBoost y K-Modes cargados exitosamente.")
    except Exception as e:
        print(f"Error al cargar artefactos: {e}")
    yield


app = FastAPI(
    title="Dengue Risk Triage API",
    description="API de inferencia, evaluación analítica sobre hold-out y perfiles clínicos de dengue (CatBoost + K-Modes)",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS habilitado para desarrollo y producción
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
def health_check():
    registry = ModelRegistry.get_instance()
    return HealthResponse(
        status="ok" if registry.loaded else "loading_or_error",
        model_registry=registry.get_status(),
        version="1.0.0",
    )


@app.post("/evaluate")
def evaluate_dataset(
    file: UploadFile = File(...),
    model: str = Form("both"),  # "catboost", "kmodes", "both"
):
    global latest_evaluated_df
    registry = ModelRegistry.get_instance()
    if not registry.loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Los modelos no están cargados en el servidor.",
        )

    model_clean = model.strip().lower()
    if model_clean not in ["catboost", "kmodes", "both"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Parámetro 'model' inválido: '{model}'. Debe ser 'catboost', 'kmodes' o 'both'.",
        )

    # Leer CSV directamente de forma síncrona
    if hasattr(file, "file") and hasattr(file.file, "read"):
        content = file.file.read()
    elif hasattr(file, "read"):
        content = file.read()
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Objeto de archivo no válido.",
        )
    try:
        df = pd.read_csv(io.BytesIO(content), encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(io.BytesIO(content), encoding="latin1")
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Error al procesar el archivo CSV: {str(e)}",
        )

    # Validar DataFrame
    errors, warnings = validate_dataframe(df, model_clean)
    if errors:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"status": "error", "errors": errors, "warnings": warnings},
        )

    response_payload: Dict[str, Any] = {
        "status": "ok",
        "n_filas": len(df),
        "warnings": warnings,
        "model_evaluated": model_clean,
    }

    # Evaluar CatBoost
    if model_clean in ["catboost", "both"]:
        response_payload["catboost"] = evaluate_catboost(
            registry.catboost_model, df, registry.metadata
        )

    # Evaluar K-Modes
    if model_clean in ["kmodes", "both"]:
        response_payload["kmodes"] = evaluate_kmodes(
            registry.kmodes_model, df, registry.metadata
        )

    # Generar tabla de predicciones para vista previa y descarga
    enriched_df = generate_predictions_table(
        registry.catboost_model if model_clean in ["catboost", "both"] else None,
        registry.kmodes_model if model_clean in ["kmodes", "both"] else None,
        df,
        registry.metadata,
    )
    latest_evaluated_df = enriched_df

    # Preview de las primeras 25 filas
    preview_cols = [c for c in enriched_df.columns if c not in ["source", "Gender"]]
    response_payload["sample_predictions"] = enriched_df[preview_cols].head(25).to_dict(orient="records")

    return response_payload


@app.get("/export-evaluated-csv")
def export_evaluated_csv():
    global latest_evaluated_df
    if latest_evaluated_df is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No hay ningún dataset evaluado recientemente para descargar.",
        )

    stream = io.StringIO()
    latest_evaluated_df.to_csv(stream, index=False)
    stream.seek(0)
    return StreamingResponse(
        iter([stream.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=resultados_evaluacion_dengue.csv"},
    )


@app.post("/evaluate/symptoms-cohort")
def evaluate_symptoms_cohort(file: UploadFile = File(...)):
    """
    Evalúa una cohorte de pacientes sin desenlace (Outcome) usando K-Modes basado exclusivamente en síntomas.
    Identifica el cluster fenotípico de cada paciente y compara métricas con la línea de base.
    """
    global latest_symptoms_df
    registry = ModelRegistry.get_instance()
    if not registry.loaded or registry.kmodes_sintomas_model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="El modelo K-Modes de solo síntomas no está cargado en el servidor.",
        )

    # Leer archivo
    if hasattr(file, "file") and hasattr(file.file, "read"):
        content = file.file.read()
    elif hasattr(file, "read"):
        content = file.read()
    else:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Archivo no válido.")

    try:
        df = pd.read_csv(io.BytesIO(content), encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(io.BytesIO(content), encoding="latin1")
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Error al leer CSV: {str(e)}")

    errors, warnings = validate_symptoms_dataframe(df)
    if errors:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"status": "error", "errors": errors, "warnings": warnings}
        )

    km_results = evaluate_kmodes_sintomas(
        registry.kmodes_sintomas_model, df, registry.metadata
    )

    enriched_df = generate_symptoms_predictions_table(
        registry.kmodes_sintomas_model, df, registry.metadata
    )
    latest_symptoms_df = enriched_df

    preview_cols = [c for c in enriched_df.columns if c not in ["source", "Gender"]]
    return {
        "status": "ok",
        "n_filas": len(df),
        "warnings": warnings,
        "kmodes_sintomas": km_results,
        "sample_predictions": enriched_df[preview_cols].head(30).to_dict(orient="records")
    }


@app.get("/export-symptoms-csv")
def export_symptoms_csv():
    """Descarga el último dataset de síntomas evaluado con los clusters fenotípicos asignados."""
    global latest_symptoms_df
    if latest_symptoms_df is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No hay datos de cohortes de síntomas evaluados recientemente para descargar.",
        )

    stream = io.StringIO()
    latest_symptoms_df.to_csv(stream, index=False, encoding="utf-8")
    stream.seek(0)

    return StreamingResponse(
        iter([stream.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=pacientes_clusters_sintomas.csv"},
    )


@app.get("/sample/symptoms-csv")
def get_sample_symptoms_csv():
    """Descarga el dataset de prueba de síntomas sin Outcome."""
    sample_path = DEV_DIR / "sample_sintomas_sin_outcome.csv"
    if not sample_path.exists():
        raise HTTPException(status_code=404, detail="Archivo de ejemplo no encontrado.")
    with open(sample_path, "r", encoding="utf-8") as f:
        content = f.read()
    return StreamingResponse(
        iter([content]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=sample_sintomas_sin_outcome.csv"},
    )


@app.get("/sample/holdout-csv")
def get_sample_holdout_csv():
    """Descarga el dataset de prueba hold-out con Outcome."""
    sample_path = DEV_DIR / "sample_holdout.csv"
    if not sample_path.exists():
        raise HTTPException(status_code=404, detail="Archivo de ejemplo no encontrado.")
    with open(sample_path, "r", encoding="utf-8") as f:
        content = f.read()
    return StreamingResponse(
        iter([content]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=sample_holdout.csv"},
    )


@app.post("/predict/clinical-support", response_model=SinglePredictionResponse)
def predict_single_patient(patient: PatientInput):
    registry = ModelRegistry.get_instance()
    if not registry.loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Los modelos no están cargados en el servidor.",
        )

    # Inferencia CatBoost
    df_single = pd.DataFrame([{
        "Fever_Duration": patient.Fever_Duration,
        "Age": patient.Age,
        "Headache": patient.Headache,
        "Myalgia": patient.Myalgia,
        "Rash": patient.Rash,
        "Fever": patient.Fever,
    }])[FEATURES_CB]

    proba = float(registry.catboost_model.predict_proba(df_single)[:, 1][0])
    threshold = float(getattr(registry.catboost_model, "best_threshold_", 0.5602))
    pred = int(proba >= threshold)

    # Inferencia de perfil clínico con K-Modes (ubicar al paciente según síntomas)
    if registry.kmodes_sintomas_model is not None:
        sintomas_cols = getattr(registry, "kmodes_sintomas_columns", ["Fever", "Headache", "Myalgia", "Rash"])
        df_km = pd.DataFrame([{
            "Fever": patient.Fever,
            "Headache": patient.Headache,
            "Myalgia": patient.Myalgia,
            "Rash": patient.Rash,
        }])[sintomas_cols].astype(int).values

        cluster_assigned = int(registry.kmodes_sintomas_model.predict(df_km)[0])
        nombres_dict = registry.metadata.get("models", {}).get("kmodes_sintomas", {}).get("cluster_names", {})
        cluster_desc = nombres_dict.get(str(cluster_assigned), f"Cluster {cluster_assigned}")
    else:
        # Fallback al modelo K-Modes general con Outcome
        df_km = pd.DataFrame([{
            "Fever": patient.Fever,
            "Headache": patient.Headache,
            "Myalgia": patient.Myalgia,
            "Rash": patient.Rash,
            "Outcome": pred,
        }])[FEATURES_KM].astype(int).values

        cluster_assigned = int(registry.kmodes_model.predict(df_km)[0])
        nombres_dict = registry.metadata.get("models", {}).get("kmodes", {}).get("cluster_names", {})
        cluster_desc = nombres_dict.get(str(cluster_assigned), f"Cluster {cluster_assigned}")

    # Triage clínico y conducta diagnóstica orientada a guardia
    if pred == 1:
        priority = "Alta"
        diag_desc = "Probable Dengue (Positivo según corte)"
        if patient.Fever_Duration <= 3:
            lab_rec = "Fase temprana (0-3 d): Indicar método directo (Ag NS1 / RT-PCR)."
        elif patient.Fever_Duration <= 6:
            lab_rec = "Fase intermedia (4-6 d): Indicar método combinado (Ag NS1 + IgM)."
        else:
            lab_rec = "Fase tardía (7+ d): Indicar método indirecto (Serología IgM)."
    elif proba >= 0.20:
        priority = "Intermedia"
        diag_desc = "Sospecha Moderada / Observación"
        lab_rec = "No urgente. Hemograma (plaquetas/leucocitos), hidratación y control en 24-48 h."
    else:
        priority = "Baja"
        diag_desc = "Baja Sospecha de Dengue"
        lab_rec = "No indicar NS1 ni PCR (bajo VPP). Manejo sintomático ambulatorio, evaluar foco alternativo y pautas de alarma."

    return SinglePredictionResponse(
        status="ok",
        probability=round(proba, 4),
        optimal_threshold=round(threshold, 4),
        prediction=pred,
        diagnosis_suggested=diag_desc,
        clinical_priority=priority,
        lab_recommendation=lab_rec,
        assigned_cluster=cluster_assigned,
        cluster_profile=cluster_desc,
        cluster_model_used="K-Modes (Solo Síntomas)" if registry.kmodes_sintomas_model is not None else "K-Modes (General)",
    )


# Montar interfaz web estática
if WEB_DIR.exists():
    app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="static")
