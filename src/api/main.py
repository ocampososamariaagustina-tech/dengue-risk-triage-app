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
    evaluate_catboost,
    evaluate_kmodes,
    generate_predictions_table,
    predict_kmodes_symptoms,
    evaluate_clinical_batch,
    FEATURES_CB,
    FEATURES_KM,
    FEATURES_KM_SYMPTOMS,
)
from src.api.schemas import PatientInput, SinglePredictionResponse, HealthResponse

WEB_DIR = Path(__file__).resolve().parent.parent / "web"

# Cache para exportación de datasets evaluados
latest_evaluated_df: Optional[pd.DataFrame] = None
latest_clinical_batch_df: Optional[pd.DataFrame] = None


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
    description="API de inferencia clínica para nuevos casos y evaluación analítica de métricas (CatBoost + K-Modes)",
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


# =====================================================================
# SECCIÓN 1: EVALUACIÓN CLÍNICA (NUEVOS CASOS / SIN OUTCOME)
# =====================================================================

@app.post("/clinical/predict-single", response_model=SinglePredictionResponse)
@app.post("/predict/clinical-support", response_model=SinglePredictionResponse)
def predict_single_patient(patient: PatientInput):
    """
    Evaluación individual de un paciente para triaje y orientación médica.
    Permite elegir entre CatBoost (probabilidad y conducta), K-Modes (sin outcome) o ambos.
    """
    registry = ModelRegistry.get_instance()
    if not registry.loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Los modelos no están cargados en el servidor.",
        )

    model_clean = (patient.model or "both").strip().lower()
    if model_clean not in ["catboost", "kmodes", "both"]:
        model_clean = "both"

    proba: Optional[float] = None
    threshold: Optional[float] = None
    pred: Optional[int] = None
    priority: Optional[str] = None
    diag_desc: Optional[str] = None
    lab_rec: Optional[str] = None
    assigned_cluster: Optional[int] = None
    cluster_desc: Optional[str] = None
    cluster_detail: Optional[str] = None

    # Inferencia CatBoost
    if model_clean in ["catboost", "both"]:
        fever_dur = patient.Fever_Duration if patient.Fever_Duration is not None else 3
        age_val = patient.Age if patient.Age is not None else 30.0

        df_single = pd.DataFrame([{
            "Fever_Duration": fever_dur,
            "Age": age_val,
            "Headache": patient.Headache,
            "Myalgia": patient.Myalgia,
            "Rash": patient.Rash,
            "Fever": patient.Fever,
        }])[FEATURES_CB]

        proba_val = float(registry.catboost_model.predict_proba(df_single)[:, 1][0])
        thresh_val = float(getattr(registry.catboost_model, "best_threshold_", 0.5602))
        pred_val = int(proba_val >= thresh_val)

        # Enrutamiento según Guía Nacional de Dengue (Argentina)
        if fever_dur <= 3:
            lab_rec = "Fase temprana (0-3 días): Priorizar método directo (NS1 / qRT-PCR)."
        elif fever_dur <= 6:
            lab_rec = "Fase intermedia (4-6 días): Combinar método directo (NS1) con método indirecto (IgM)."
        else:
            lab_rec = "Fase tardía (7+ días): Priorizar método indirecto (IgM)."

        priority = "Alta" if (proba_val >= thresh_val or proba_val > 0.7) else ("Intermedia" if proba_val > 0.4 else "Baja")
        diag_desc = "Probable Dengue (Positivo según umbral de corte)" if pred_val == 1 else "Baja sospecha de Dengue"

        proba = round(proba_val, 4)
        threshold = round(thresh_val, 4)
        pred = pred_val

    # Inferencia K-Modes (fenotipo sin outcome)
    if model_clean in ["kmodes", "both"]:
        df_symp = pd.DataFrame([{
            "Fever": patient.Fever,
            "Headache": patient.Headache,
            "Myalgia": patient.Myalgia,
            "Rash": patient.Rash,
        }])[FEATURES_KM_SYMPTOMS]

        clusters, cluster_nombres = predict_kmodes_symptoms(
            registry.kmodes_model, df_symp, registry.metadata
        )
        assigned_cluster = int(clusters[0])
        cluster_desc = cluster_nombres[0]

        km_meta = registry.metadata.get("models", {}).get("kmodes", {})
        centroids = km_meta.get("centroids", [])
        if assigned_cluster < len(centroids):
            c_vals = centroids[assigned_cluster]
            cluster_detail = f"Fiebre: {'sí' if c_vals.get('Fever') else 'no'} | Cefalea: {'sí' if c_vals.get('Headache') else 'no'} | Mialgia: {'sí' if c_vals.get('Myalgia') else 'no'} | Exantema: {'sí' if c_vals.get('Rash') else 'no'}"

    model_labels = {
        "catboost": "CatBoost (Probabilidad y Triaje)",
        "kmodes": "K-Modes (Fenotipo Clínico sin Outcome)",
        "both": "CatBoost + K-Modes (Combinado)"
    }

    return SinglePredictionResponse(
        status="ok",
        model_used=model_labels.get(model_clean, "CatBoost + K-Modes"),
        probability=proba,
        optimal_threshold=threshold,
        prediction=pred,
        diagnosis_suggested=diag_desc,
        clinical_priority=priority,
        lab_recommendation=lab_rec,
        assigned_cluster=assigned_cluster,
        cluster_profile=cluster_desc,
        cluster_description=cluster_detail,
    )


@app.post("/clinical/evaluate-batch")
def evaluate_clinical_batch_endpoint(
    file: UploadFile = File(...),
    model: str = Form("both"),  # "catboost", "kmodes", "both"
):
    """
    Evaluación por lote orientada al ámbito clínico para NUEVOS CASOS sin necesidad de Outcome.
    Calcula triaje, distribución de prioridades y fenotipos de pacientes.
    """
    global latest_clinical_batch_df
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

    # Leer CSV
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

    # Validar DataFrame sin requerir Outcome
    errors, warnings = validate_dataframe(df, model_type=model_clean, require_outcome=False)
    if errors:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"status": "error", "errors": errors, "warnings": warnings},
        )

    response_payload, enriched_df = evaluate_clinical_batch(
        registry.catboost_model if model_clean in ["catboost", "both"] else None,
        registry.kmodes_model if model_clean in ["kmodes", "both"] else None,
        df,
        registry.metadata,
        model_type=model_clean,
    )
    response_payload["warnings"] = warnings
    latest_clinical_batch_df = enriched_df

    # Preview de las primeras 25 filas
    preview_cols = [c for c in enriched_df.columns if c not in ["source", "Gender"]]
    response_payload["sample_predictions"] = enriched_df[preview_cols].head(25).to_dict(orient="records")

    return response_payload


@app.get("/clinical/export-csv")
def export_clinical_batch_csv():
    """Descarga el último lote procesado en Evaluación Clínica."""
    global latest_clinical_batch_df
    if latest_clinical_batch_df is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No hay ningún lote clínico evaluado recientemente para descargar.",
        )

    stream = io.StringIO()
    latest_clinical_batch_df.to_csv(stream, index=False)
    stream.seek(0)
    return StreamingResponse(
        iter([stream.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=resultados_triaje_clinico.csv"},
    )


# =====================================================================
# SECCIÓN 2: MÉTRICAS DE MODELOS (CON OUTCOME / AUDITORÍA TÉCNICA)
# =====================================================================

@app.post("/metrics/evaluate")
@app.post("/evaluate")
def evaluate_dataset(
    file: UploadFile = File(...),
    model: str = Form("both"),  # "catboost", "kmodes", "both"
):
    """
    Evaluación técnica de métricas (F1, matriz de confusión, ROC, PSI, pureza)
    sobre hold-out o casos confirmados con Outcome.
    """
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

    # Leer CSV
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

    # Validar DataFrame exigiendo obligatoriamente Outcome
    errors, warnings = validate_dataframe(df, model_type=model_clean, require_outcome=True)
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


@app.get("/metrics/export-csv")
@app.get("/export-evaluated-csv")
def export_evaluated_csv():
    """Descarga el último dataset evaluado en la sección de Métricas (con Outcome)."""
    global latest_evaluated_df
    if latest_evaluated_df is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No hay ningún dataset de métricas evaluado recientemente para descargar.",
        )

    stream = io.StringIO()
    latest_evaluated_df.to_csv(stream, index=False)
    stream.seek(0)
    return StreamingResponse(
        iter([stream.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=resultados_metricas_holdout.csv"},
    )


# Montar interfaz web estática
if WEB_DIR.exists():
    app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="static")
