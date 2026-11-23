import sys
import io
import asyncio
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from fastapi import UploadFile
from src.ml.model_loader import ModelRegistry
from src.api.main import (
    health_check,
    evaluate_dataset,
    predict_single_patient,
    export_evaluated_csv,
    evaluate_clinical_batch_endpoint,
    export_clinical_batch_csv,
)
from src.api.schemas import PatientInput


def test_full_pipeline():
    print("\n[1/7] Inicializando ModelRegistry y cargando artefactos...")
    registry = ModelRegistry.get_instance()
    registry.load_artifacts()
    assert registry.loaded is True
    print(" -> CatBoost y K-Modes cargados exitosamente.")

    print("\n[2/7] Probando Endpoint GET /health...")
    health_resp = health_check()
    assert health_resp.status == "ok"
    assert health_resp.version == "1.0.0"
    assert health_resp.model_registry["models"]["catboost"]["loaded"] is True
    assert health_resp.model_registry["models"]["kmodes"]["loaded"] is True
    print(f" -> Health OK: CatBoost hash={health_resp.model_registry['models']['catboost']['sha256'][:10]}...")

    print("\n[3/7] Probando Sección MÉTRICAS (Con Outcome) - POST /metrics/evaluate...")
    sample_csv_path = PROJECT_ROOT / "dev" / "sample_holdout.csv"
    with open(sample_csv_path, "rb") as f:
        file_bytes = f.read()

    upload_file = UploadFile(filename="sample_holdout.csv", file=io.BytesIO(file_bytes))
    eval_result = evaluate_dataset(upload_file, model="both")

    assert eval_result["status"] == "ok"
    assert eval_result["n_filas"] == 40
    assert "catboost" in eval_result
    assert "kmodes" in eval_result

    cb = eval_result["catboost"]
    print(" -> Métricas CatBoost obtenidas (Con Outcome):")
    print(f"    - Accuracy: {cb['accuracy']}")
    print(f"    - F1-Score: {cb['f1']}")
    print(f"    - Sensibilidad (Recall): {cb['recall']}")
    print(f"    - ROC-AUC: {cb['roc_auc']}")
    print(f"    - Matriz de Confusión: {cb['confusion_matrix']}")

    km = eval_result["kmodes"]
    print(" -> Métricas K-Modes obtenidas (Con Outcome):")
    print(f"    - Distribución Clusters: {km['tamano_clusters']}")
    print(f"    - Estabilidad Poblacional (PSI): {km['psi_tamanos']}")
    print(f"    - Pureza con Outcome: {km['coincidencia_con_outcome']['pureza']}")

    print("\n[4/7] Probando Sección EVALUACIÓN CLÍNICA POR LOTE (Sin Outcome) - POST /clinical/evaluate-batch...")
    nuevos_csv_path = PROJECT_ROOT / "dev" / "sample_nuevos_casos_sin_outcome.csv"
    with open(nuevos_csv_path, "rb") as f:
        nuevos_bytes = f.read()

    upload_nuevos = UploadFile(filename="sample_nuevos_casos_sin_outcome.csv", file=io.BytesIO(nuevos_bytes))
    clinical_batch_res = evaluate_clinical_batch_endpoint(upload_nuevos, model="both")

    assert clinical_batch_res["status"] == "ok"
    assert clinical_batch_res["n_filas"] == 30
    assert "catboost_clinical" in clinical_batch_res
    assert "kmodes_clinical" in clinical_batch_res

    cb_clin = clinical_batch_res["catboost_clinical"]
    km_clin = clinical_batch_res["kmodes_clinical"]
    print(" -> Triaje Clínico por Lote obtenido (Sin Outcome):")
    print(f"    - Media de probabilidad: {cb_clin['media_probabilidad']}")
    print(f"    - Casos alta sospecha: {cb_clin['casos_sospecha_alta']} ({cb_clin['pct_sospecha_alta']*100}%)")
    print(f"    - Distribución de prioridades: {cb_clin['distribucion_prioridad']}")
    print(f"    - Distribución fenotipos K-Modes: {[v['nombre'] for v in km_clin['distribucion_clusters'].values()]}")

    print("\n[5/7] Probando EVALUACIÓN INDIVIDUAL con opciones (Ambos, Solo CatBoost, Solo K-Modes)...")
    # Ambos
    p_both = PatientInput(Fever_Duration=3, Age=34.0, Headache=1, Myalgia=1, Rash=0, Fever=1, model="both")
    res_both = predict_single_patient(p_both)
    assert res_both.probability is not None
    assert res_both.assigned_cluster is not None
    print(f" -> [Ambos] Prob: {res_both.probability}, Prioridad: {res_both.clinical_priority}, Cluster: {res_both.cluster_profile}")

    # Solo CatBoost
    p_cb = PatientInput(Fever_Duration=2, Age=45.0, Headache=1, Myalgia=0, Rash=0, Fever=1, model="catboost")
    res_cb = predict_single_patient(p_cb)
    assert res_cb.probability is not None
    assert res_cb.assigned_cluster is None
    print(f" -> [Solo CatBoost] Prob: {res_cb.probability}, Ventana: {res_cb.lab_recommendation}")

    # Solo K-Modes (sin outcome)
    p_km = PatientInput(Fever_Duration=0, Age=30.0, Headache=1, Myalgia=1, Rash=1, Fever=1, model="kmodes")
    res_km = predict_single_patient(p_km)
    assert res_km.probability is None
    assert res_km.assigned_cluster == 2  # Con rash -> cluster 2
    print(f" -> [Solo K-Modes sin Outcome] Asignado a: {res_km.cluster_profile} (Cluster {res_km.assigned_cluster})")

    print("\n[6/7] Probando Descargas CSV de Métricas y de Triaje Clínico...")
    export_metrics = export_evaluated_csv()
    assert export_metrics.media_type == "text/csv"
    export_clinical = export_clinical_batch_csv()
    assert export_clinical.media_type == "text/csv"
    print(" -> Ambas descargas de CSV generadas correctamente.")

    print("\n[7/7] Probando validación estricta de Outcome en Métricas...")
    # Intentar subir el archivo sin outcome al endpoint de métricas debe fallar informando la falta de Outcome
    upload_sin_out = UploadFile(filename="sample_nuevos_casos_sin_outcome.csv", file=io.BytesIO(nuevos_bytes))
    res_inval = evaluate_dataset(upload_sin_out, model="both")
    # Devuelve JSONResponse con status 400
    assert hasattr(res_inval, "status_code") and res_inval.status_code == 400
    print(" -> Validación correcta: El endpoint de Métricas rechaza datasets sin Outcome como se espera.")

    print("\n===================================================================")
    print(">>> TODOS LOS TESTS DE INTEGRACIÓN Y VALIDACIÓN PASARON AL 100% <<<")
    print("===================================================================\n")


if __name__ == "__main__":
    test_full_pipeline()


if __name__ == "__main__":
    test_full_pipeline()
