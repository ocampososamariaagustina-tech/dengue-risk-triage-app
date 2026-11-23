import sys
import io
import asyncio
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from fastapi import UploadFile
from src.ml.model_loader import ModelRegistry
from src.api.main import health_check, evaluate_dataset, predict_single_patient, export_evaluated_csv
from src.api.schemas import PatientInput


def test_full_pipeline():
    print("\n[1/5] Inicializando ModelRegistry y cargando artefactos...")
    registry = ModelRegistry.get_instance()
    registry.load_artifacts()
    assert registry.loaded is True
    print(" -> CatBoost y K-Modes cargados exitosamente.")

    print("\n[2/5] Probando Endpoint GET /health...")
    health_resp = health_check()
    assert health_resp.status == "ok"
    assert health_resp.version == "1.0.0"
    assert health_resp.model_registry["models"]["catboost"]["loaded"] is True
    assert health_resp.model_registry["models"]["kmodes"]["loaded"] is True
    print(f" -> Health OK: CatBoost hash={health_resp.model_registry['models']['catboost']['sha256'][:10]}...")

    print("\n[3/5] Probando Endpoint POST /evaluate con sample_holdout.csv...")
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
    print(" -> Métricas CatBoost obtenidas:")
    print(f"    - Accuracy: {cb['accuracy']}")
    print(f"    - F1-Score: {cb['f1']}")
    print(f"    - Sensibilidad (Recall): {cb['recall']}")
    print(f"    - ROC-AUC: {cb['roc_auc']}")
    print(f"    - Matriz de Confusión: {cb['confusion_matrix']}")
    print(f"    - Intervalos Bootstrap 95%: F1={cb['ic95']['f1']}, Recall={cb['ic95']['recall']}")

    km = eval_result["kmodes"]
    print(" -> Métricas K-Modes obtenidas:")
    print(f"    - Distribución Clusters: {km['tamano_clusters']}")
    print(f"    - Estabilidad Poblacional (PSI): {km['psi_tamanos']}")
    print(f"    - Distancia Hamming al Centroide: {km['costo_por_obs']['holdout']} (Train: {km['costo_por_obs']['train']})")
    print(f"    - Silhouette Score: {km['silhouette']['global']}")
    print(f"    - Perfiles identificados: {[c['nombre'] for c in km['centroides']]}")

    assert len(eval_result["sample_predictions"]) > 0
    print(f" -> Filas de vista previa generadas: {len(eval_result['sample_predictions'])}")

    print("\n[4/5] Probando Endpoint POST /predict/clinical-support...")
    patient = PatientInput(
        Fever_Duration=3,
        Age=34.0,
        Headache=1,
        Myalgia=1,
        Rash=0,
        Fever=1
    )
    single_res = predict_single_patient(patient)
    assert single_res.status == "ok"
    assert single_res.prediction == 1
    assert "Fase temprana" in single_res.lab_recommendation
    print(f" -> Caso Alta Probabilidad: {single_res.diagnosis_suggested}")
    print(f"    - Probabilidad: {single_res.probability}")
    print(f"    - Prioridad: {single_res.clinical_priority}")
    print(f"    - Conducta / Laboratorio: {single_res.lab_recommendation}")

    # Caso Baja Probabilidad (Triage negativo / Descarte)
    patient_low = PatientInput(
        Fever_Duration=1,
        Age=22.0,
        Headache=0,
        Myalgia=0,
        Rash=0,
        Fever=0
    )
    res_low = predict_single_patient(patient_low)
    assert res_low.status == "ok"
    assert res_low.prediction == 0
    assert res_low.clinical_priority == "Baja"
    assert "No indicar NS1 ni PCR" in res_low.lab_recommendation
    print(f" -> Caso Baja Probabilidad: {res_low.diagnosis_suggested}")
    print(f"    - Probabilidad: {res_low.probability}")
    print(f"    - Prioridad: {res_low.clinical_priority}")
    print(f"    - Conducta / Laboratorio: {res_low.lab_recommendation}")

    print("\n[5/5] Probando Endpoint GET /export-evaluated-csv...")
    export_res = export_evaluated_csv()
    assert export_res.media_type == "text/csv"
    print(" -> Exportación de CSV lista para descarga.")

    print("\n=======================================================")
    print(">>> TODOS LOS TESTS DE INTEGRACIÓN PASARON AL 100% <<<")
    print("=======================================================\n")


if __name__ == "__main__":
    test_full_pipeline()
