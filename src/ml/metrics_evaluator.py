import math
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    fbeta_score,
    roc_auc_score,
    average_precision_score,
    matthews_corrcoef,
    brier_score_loss,
    balanced_accuracy_score,
    classification_report,
    roc_curve,
    precision_recall_curve,
    silhouette_score,
    silhouette_samples,
    normalized_mutual_info_score,
    adjusted_rand_score,
)

FEATURES_CB = ["Fever_Duration", "Age", "Headache", "Myalgia", "Rash", "Fever"]
FEATURES_KM = ["Fever", "Headache", "Myalgia", "Rash", "Outcome"]


def validate_dataframe(df: pd.DataFrame, model_type: str = "both") -> Tuple[List[str], List[str]]:
    """
    Valida el DataFrame según las especificaciones.
    Retorna (errores, advertencias).
    """
    errors = []
    warnings = []

    # Verificar filas
    if len(df) == 0:
        errors.append("El archivo CSV está vacío.")
        return errors, warnings
    if len(df) < 30:
        warnings.append(f"El dataset tiene solo {len(df)} filas (< 30). Las métricas pueden ser estadísticamente inestables.")

    # Encabezados requeridos
    cols_presentes = set(df.columns)
    if model_type in ["catboost", "both"]:
        cb_missing = [c for c in FEATURES_CB + ["Outcome"] if c not in cols_presentes]
        if cb_missing:
            errors.append(f"Faltan columnas requeridas para CatBoost: {cb_missing}")

    if model_type in ["kmodes", "both"]:
        km_missing = [c for c in FEATURES_KM if c not in cols_presentes]
        if km_missing:
            errors.append(f"Faltan columnas requeridas para K-Modes: {km_missing}")

    if errors:
        return errors, warnings

    # Normalización opcional de tipos/valores booleanos si vienen en texto
    # Outcome
    if df["Outcome"].dtype == object:
        df["Outcome"] = df["Outcome"].astype(str).str.strip().map({
            "Negative": 0, "Positive": 1, "0": 0, "1": 1, "False": 0, "True": 1
        })
    # Fever
    if df["Fever"].dtype == object or df["Fever"].dtype == bool:
        df["Fever"] = df["Fever"].astype(str).str.strip().map({
            "False": 0, "True": 1, "0": 0, "1": 1
        })

    # Validar nulos en requeridas
    req_cols = list(set(FEATURES_CB + FEATURES_KM))
    null_counts = df[req_cols].isnull().sum()
    cols_with_null = null_counts[null_counts > 0]
    if not cols_with_null.empty:
        errors.append(f"Valores nulos detectados en columnas obligatorias: {cols_with_null.to_dict()}")

    # Validar binarias {0, 1}
    binary_cols = ["Headache", "Myalgia", "Rash", "Fever", "Outcome"]
    for col in binary_cols:
        if col in df.columns:
            invalid_vals = set(df[col].dropna().unique()) - {0, 1}
            if invalid_vals:
                errors.append(f"Columna binaria '{col}' contiene valores fuera de {{0, 1}}: {invalid_vals}")

    # Validar numéricas
    if "Age" in df.columns:
        if not pd.api.types.is_numeric_dtype(df["Age"]):
            errors.append("La columna 'Age' debe ser numérica.")
        elif (df["Age"] <= 0).any():
            errors.append("La columna 'Age' contiene valores menores o iguales a 0.")
        elif (df["Age"] > 110).any():
            warnings.append("Valores de 'Age' superiores a 110 años detectados.")

    if "Fever_Duration" in df.columns:
        if not pd.api.types.is_numeric_dtype(df["Fever_Duration"]):
            errors.append("La columna 'Fever_Duration' debe ser numérica.")
        elif (df["Fever_Duration"] < 0).any():
            errors.append("La columna 'Fever_Duration' contiene valores negativos.")
        elif (df["Fever_Duration"] > 30).any():
            warnings.append("Valores de 'Fever_Duration' mayores a 30 días detectados.")

    # Validar diversidad de clases en Outcome
    if "Outcome" in df.columns and df["Outcome"].nunique() < 2:
        warnings.append("El dataset contiene una sola clase en 'Outcome'. Métricas como ROC-AUC y PR-AUC no podrán calcularse.")

    if df.duplicated(subset=FEATURES_CB).sum() > 0:
        warnings.append(f"Se detectaron {df.duplicated(subset=FEATURES_CB).sum()} filas duplicadas en los predictores clínicos.")

    return errors, warnings


def bootstrap_catboost_ci(
    y_true: np.ndarray,
    proba: np.ndarray,
    pred: np.ndarray,
    n_bootstrap: int = 1000,
    random_state: int = 42
) -> Dict[str, List[float]]:
    """Calcula intervalos de confianza al 95% usando Bootstrap estratificado optimizado."""
    rng = np.random.RandomState(random_state)
    n = len(y_true)
    indices = np.arange(n)

    recalls, precisions, f1s, f2s, rocs = [], [], [], [], []

    # Vectorized bootstrap calculation for maximum performance
    for _ in range(n_bootstrap):
        idx_sample = rng.choice(indices, size=n, replace=True)
        y_s = y_true[idx_sample]
        p_s = pred[idx_sample]
        prob_s = proba[idx_sample]

        tp = int(np.sum((y_s == 1) & (p_s == 1)))
        fp = int(np.sum((y_s == 0) & (p_s == 1)))
        fn = int(np.sum((y_s == 1) & (p_s == 0)))

        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        f2 = (5 * prec * rec / (4 * prec + rec)) if (4 * prec + rec) > 0 else 0.0

        recalls.append(rec)
        precisions.append(prec)
        f1s.append(f1)
        f2s.append(f2)

        if len(np.unique(y_s)) == 2:
            try:
                rocs.append(roc_auc_score(y_s, prob_s))
            except Exception:
                pass

    def get_ci(values: List[float]) -> List[float]:
        if not values:
            return [0.0, 0.0]
        return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]

    return {
        "recall": [round(x, 4) for x in get_ci(recalls)],
        "precision": [round(x, 4) for x in get_ci(precisions)],
        "f1": [round(x, 4) for x in get_ci(f1s)],
        "f2": [round(x, 4) for x in get_ci(f2s)],
        "roc_auc": [round(x, 4) for x in get_ci(rocs)],
    }


def evaluate_catboost(model: Any, df: pd.DataFrame, metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Calcula el conjunto exhaustivo de métricas para CatBoost."""
    X = df[FEATURES_CB].copy()
    y = df["Outcome"].astype(int).values

    # Inferencia
    proba = model.predict_proba(X)[:, 1]
    pred = model.predict(X)
    if hasattr(pred, "ravel"):
        pred = pred.ravel().astype(int)

    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    n_total = len(y)
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())

    # Umbral óptimo del modelo
    umbral = getattr(model, "best_threshold_", 0.5)
    if isinstance(umbral, np.generic):
        umbral = float(umbral)

    has_two_classes = (n_pos > 0 and n_neg > 0)
    roc_auc = float(roc_auc_score(y, proba)) if has_two_classes else None
    pr_auc = float(average_precision_score(y, proba)) if has_two_classes else None

    # Curvas (submuestreo si son muchos puntos para respuesta ágil en UI)
    curvas = {"roc": {"fpr": [], "tpr": []}, "pr": {"precision": [], "recall": []}}
    if has_two_classes:
        fpr, tpr, _ = roc_curve(y, proba)
        prec_curve, rec_curve, _ = precision_recall_curve(y, proba)

        # Reducir a máximo 100 puntos espaciados uniformemente si excede
        step = max(1, len(fpr) // 100)
        curvas["roc"]["fpr"] = [round(float(v), 4) for v in fpr[::step]]
        curvas["roc"]["tpr"] = [round(float(v), 4) for v in tpr[::step]]

        step_pr = max(1, len(prec_curve) // 100)
        curvas["pr"]["precision"] = [round(float(v), 4) for v in prec_curve[::step_pr]]
        curvas["pr"]["recall"] = [round(float(v), 4) for v in rec_curve[::step_pr]]

    # Bootstrap CIs
    ic95 = bootstrap_catboost_ci(y, proba, pred, n_bootstrap=1000)

    # Comparación vs. desarrollo
    test_ref = metadata.get("models", {}).get("catboost", {}).get("test_metrics", {})
    vs_desarrollo = {}
    f1_val = f1_score(y, pred, zero_division=0)
    recall_val = recall_score(y, pred, zero_division=0)
    prec_val = precision_score(y, pred, zero_division=0)
    if test_ref:
        if "f1" in test_ref:
            vs_desarrollo["f1"] = round(float(f1_val - test_ref["f1"]), 4)
        if "recall" in test_ref:
            vs_desarrollo["recall"] = round(float(recall_val - test_ref["recall"]), 4)
        if "precision" in test_ref:
            vs_desarrollo["precision"] = round(float(prec_val - test_ref["precision"]), 4)
        if roc_auc is not None and "roc_auc" in test_ref:
            vs_desarrollo["roc_auc"] = round(float(roc_auc - test_ref["roc_auc"]), 4)

    # Métricas por subgrupo
    por_subgrupo = {}
    for subcol in ["source", "Gender"]:
        if subcol in df.columns:
            por_subgrupo[subcol] = {}
            for val, grp in df.groupby(subcol):
                if len(grp) >= 30 and grp["Outcome"].nunique() >= 1:
                    y_g = grp["Outcome"].astype(int).values
                    X_g = grp[FEATURES_CB]
                    p_g = model.predict(X_g).ravel().astype(int)
                    pr_g = model.predict_proba(X_g)[:, 1]
                    por_subgrupo[subcol][str(val)] = {
                        "n": len(grp),
                        "pct_pos": round(float((y_g == 1).mean()), 4),
                        "accuracy": round(float(accuracy_score(y_g, p_g)), 4),
                        "recall": round(float(recall_score(y_g, p_g, zero_division=0)), 4),
                        "precision": round(float(precision_score(y_g, p_g, zero_division=0)), 4),
                        "f1": round(float(f1_score(y_g, p_g, zero_division=0)), 4),
                        "roc_auc": round(float(roc_auc_score(y_g, pr_g)), 4) if grp["Outcome"].nunique() == 2 else None,
                    }

    return {
        "umbral": round(float(umbral), 4),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "accuracy": round(float(accuracy_score(y, pred)), 4),
        "error_rate": round(float(1 - accuracy_score(y, pred)), 4),
        "precision": round(float(prec_val), 4),
        "recall": round(float(recall_val), 4),
        "specificity": round(float(tn / (tn + fp)), 4) if (tn + fp) > 0 else 0.0,
        "npv": round(float(tn / (tn + fn)), 4) if (tn + fn) > 0 else 0.0,
        "f1": round(float(f1_val), 4),
        "f2": round(float(fbeta_score(y, pred, beta=2, zero_division=0)), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y, pred)), 4),
        "mcc": round(float(matthews_corrcoef(y, pred)), 4),
        "roc_auc": round(roc_auc, 4) if roc_auc is not None else None,
        "pr_auc": round(pr_auc, 4) if pr_auc is not None else None,
        "brier": round(float(brier_score_loss(y, proba)), 4),
        "support": {"n": n_total, "pos": n_pos, "neg": n_neg, "pct_pos": round(float(n_pos / n_total), 4)},
        "classification_report": classification_report(y, pred, digits=4, output_dict=True, zero_division=0),
        "ic95": ic95,
        "vs_desarrollo": vs_desarrollo,
        "curvas": curvas,
        "por_subgrupo": por_subgrupo,
    }


def evaluate_kmodes(km: Any, df: pd.DataFrame, metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Calcula las métricas de interpretabilidad y clustering de K-Modes."""
    Xk = df[FEATURES_KM].astype(int).values
    labels = km.predict(Xk)
    if hasattr(labels, "ravel"):
        labels = labels.ravel().astype(int)

    n_total = len(Xk)
    unique_labels, counts = np.unique(labels, return_counts=True)
    tamano_clusters = {}
    pct_holdout = {}
    for c, cnt in zip(unique_labels, counts):
        c_str = str(int(c))
        pct = float(cnt / n_total)
        tamano_clusters[c_str] = {"n": int(cnt), "pct": round(pct, 4)}
        pct_holdout[c_str] = pct

    # PSI (Population Stability Index) contra proportions de train
    km_meta = metadata.get("models", {}).get("kmodes", {})
    train_dist = km_meta.get("train_stats", {}).get("cluster_distribution", {})
    psi = 0.0
    epsilon = 1e-6
    for c_str in ["0", "1", "2"]:
        p_train = train_dist.get(c_str, {}).get("pct", 0.3333)
        p_hold = pct_holdout.get(c_str, 0.0)
        p_hold_adj = max(p_hold, epsilon)
        p_train_adj = max(p_train, epsilon)
        psi += (p_hold - p_train) * math.log(p_hold_adj / p_train_adj)

    # Costo por observación (distancia Hamming promedio)
    centroids = np.asarray(km.cluster_centroids_).astype(int)
    dist = (Xk != centroids[labels]).sum(axis=1)
    costo_hold = float(dist.mean())
    costo_train = float(km_meta.get("train_stats", {}).get("cost_per_obs", 0.62915))

    # Silhouette Hamming
    silhouette_data = {"global": None, "por_cluster": {}}
    if 1 < len(unique_labels) < n_total:
        try:
            sil_global = float(silhouette_score(Xk, labels, metric="hamming"))
            s_samples = silhouette_samples(Xk, labels, metric="hamming")
            silhouette_data["global"] = round(sil_global, 4)
            for c in unique_labels:
                silhouette_data["por_cluster"][str(int(c))] = round(float(s_samples[labels == c].mean()), 4)
        except Exception:
            pass

    # Perfiles por cluster (frecuencia de cada síntoma + outcome)
    df_km = df[FEATURES_KM].copy()
    df_km["cluster"] = labels
    perfil_por_cluster = {}
    for c in sorted(unique_labels):
        c_df = df_km[df_km["cluster"] == c]
        perfil_por_cluster[str(int(c))] = {
            col: round(float((c_df[col] == 1).mean()), 4) for col in FEATURES_KM
        }

    # Nombres interpretables y centroides formateados
    nombres_dict = km_meta.get("cluster_names", {
        "0": "Perfil sin cuadro clínico / sano",
        "1": "Perfil dengue típico (síntomas cardinales sin exantema)",
        "2": "Perfil dengue completo (síntomas cardinales + exantema)"
    })

    centroides_formateados = []
    for i, cent in enumerate(centroids):
        c_str = str(i)
        frase = f"Fiebre: {'sí' if cent[0] else 'no'} | Cefalea: {'sí' if cent[1] else 'no'} | Mialgia: {'sí' if cent[2] else 'no'} | Exantema: {'sí' if cent[3] else 'no'} | Dengue: {'sí' if cent[4] else 'no'}"
        centroides_formateados.append({
            "cluster": i,
            "nombre": nombres_dict.get(c_str, f"Cluster {i}"),
            "valores": dict(zip(FEATURES_KM, [int(x) for x in cent])),
            "descripcion": frase
        })

    # Coincidencia con Outcome
    y = df["Outcome"].astype(int).values
    pct_dengue = {}
    for c in sorted(unique_labels):
        mask = (labels == c)
        pct_dengue[str(int(c))] = round(float((y[mask] == 1).mean()), 4) if mask.sum() > 0 else 0.0

    # Pureza
    purity_sum = sum(
        max(np.sum((labels == c) & (y == 0)), np.sum((labels == c) & (y == 1)))
        for c in unique_labels
    )
    purity = purity_sum / n_total

    coincidencia = {
        "pct_dengue_por_cluster": pct_dengue,
        "pureza": round(float(purity), 4),
        "nmi": round(float(normalized_mutual_info_score(y, labels)), 4),
        "ari": round(float(adjusted_rand_score(y, labels)), 4),
        "nota": "Descriptiva: Outcome fue utilizado intencionalmente en el clustering para caracterizar el perfil completo."
    }

    return {
        "tamano_clusters": tamano_clusters,
        "psi_tamanos": round(float(psi), 4),
        "costo_por_obs": {
            "holdout": round(costo_hold, 4),
            "train": round(costo_train, 4),
            "delta": round(costo_hold - costo_train, 4),
            "estatus": "Estable" if (costo_hold - costo_train) < 0.2 else "Degradación"
        },
        "silhouette": silhouette_data,
        "centroides": centroides_formateados,
        "perfil_por_cluster": perfil_por_cluster,
        "nombres_cluster": nombres_dict,
        "coincidencia_con_outcome": coincidencia
    }


def generate_predictions_table(
    cb_model: Any,
    km_model: Any,
    df: pd.DataFrame,
    metadata: Dict[str, Any]
) -> pd.DataFrame:
    """Genera DataFrame enriquecido con predicciones y clusters para exportación y visualización."""
    out_df = df.copy()

    if cb_model is not None:
        X_cb = df[FEATURES_CB]
        out_df["proba_dengue"] = np.round(cb_model.predict_proba(X_cb)[:, 1], 4)
        pred_cb = cb_model.predict(X_cb)
        if hasattr(pred_cb, "ravel"):
            pred_cb = pred_cb.ravel().astype(int)
        out_df["pred_catboost"] = pred_cb

    if km_model is not None:
        X_km = df[FEATURES_KM].astype(int).values
        clusters = km_model.predict(X_km)
        if hasattr(clusters, "ravel"):
            clusters = clusters.ravel().astype(int)
        out_df["cluster_kmodes"] = clusters
        nombres_dict = metadata.get("models", {}).get("kmodes", {}).get("cluster_names", {})
        out_df["cluster_nombre"] = [nombres_dict.get(str(c), f"Cluster {c}") for c in clusters]

    return out_df
