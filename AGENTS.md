# AGENTS.md — Reglas del proyecto

Herramienta de **evaluación analítica y apoyo a la decisión clínica** para dengue en Argentina basada en modelos de Machine Learning (CatBoost y K-Modes).

Documentos de referencia:
- `especificaciones_metricas_holdout.txt` (especificación técnica canónica para producción y métricas sobre hold-out)
- `TFI_PROYECTO_v4.ipynb` (código fuente de entrenamiento, parámetros óptimos y validación de modelos)
- `informe_modelo_dengue (1).md` (fundamentación clínica y contextual de salud pública)

---

## 1. Rol del agente vs. Responsabilidad humana

- **Rol del agente (Ingeniería de software):**
  - Implementación de la API de backend (`src/api/`) para carga de datos, validación y cálculo de métricas.
  - Implementación de la capa de inferencia y agrupamiento (`src/ml/`) que ejecuta los artefactos `.joblib`.
  - Construcción de la interfaz web (`src/web/`) para carga de datasets, visualización de métricas (CatBoost) y perfiles clínicos (K-Modes).
  - Empaquetado, pruebas automatizadas (`tests/`) y configuración de despliegue (Docker / Local).
- **Prohibición estricta:**
  - El agente **no debe reentrenar** modelos con datos de hold-out ni alterar los hiperparámetros ni umbrales definidos en `TFI_PROYECTO_v4.ipynb`.
  - El hold-out nunca se usa para tunear ni ajustar; únicamente para evaluar.

---

## 2. Arquitectura de Modelos en Producción

El sistema opera con dos modelos complementarios desacoplados:

### 1) CatBoost — Modelo Predictivo (Supervisado)
- **Función:** Predicción del riesgo de dengue (`Outcome`: 1 = Dengue / Positivo, 0 = No dengue / Negativo).
- **Tipo de estimador:** `TunedThresholdClassifierCV` sobre `CatBoostClassifier`.
- **Características de entrada (6 variables, orden estricto por nombre):**
  `['Fever_Duration', 'Age', 'Headache', 'Myalgia', 'Rash', 'Fever']`
- **Umbral óptimo tuneado:** Definido en el estimador (`best_threshold_` ~0.5602).
- **Salida:** Probabilidades crudas (`predict_proba[:, 1]`), etiquetas binarias con umbral tuneado (`predict`), matriz de confusión, métricas completas (Accuracy, Precision, Recall/Sensibilidad, Especificidad, F1, F2, ROC-AUC, PR-AUC, Brier, MCC), intervalos de confianza 95% (bootstrap) y análisis por subgrupos (`Gender`, `source`).

### 2) K-Modes — Modelo Descriptivo de Interpretabilidad (No Supervisado)
- **Función:** Agrupamiento no supervisado en perfiles clínicos típicos para explicar el comportamiento de los pacientes en lenguaje natural.
- **Tipo de estimador:** `KModes(n_clusters=3, init="random", n_init=5, random_state=42)`.
- **Características de entrada (5 variables binarias, orden estricto por nombre):**
  `['Fever', 'Headache', 'Myalgia', 'Rash', 'Outcome']`
  *(Nota clínica: `Outcome` se incluye intencionalmente en el agrupamiento para caracterizar el perfil completo de síntomas + diagnóstico).*
- **Perfiles de clusters identificados:**
  - **Cluster 0:** Perfil asintomático / sin cuadro clínico ni dengue (`Fever=0, Headache=0, Myalgia=0, Rash=0, Outcome=0`).
  - **Cluster 1:** Perfil dengue típico con síntomas cardinales sin exantema (`Fever=1, Headache=1, Myalgia=1, Rash=0, Outcome=1`).
  - **Cluster 2:** Perfil dengue completo / florido con exantema (`Fever=1, Headache=1, Myalgia=1, Rash=1, Outcome=1`).
- **Métricas:** Tamaños de cluster, PSI vs. entrenamiento, costo promedio Hamming respecto al centroide, coeficiente Silhouette (métrica Hamming) global y por cluster, y tasas de positividad por cluster.

---

## 3. Esquema de Variables (Fuente Única de Verdad)

| Variable | Tipo | Valores válidos | Uso en CatBoost | Uso en K-Modes |
| :--- | :--- | :--- | :--- | :--- |
| `Fever_Duration` | Numérica (int/float) | $\ge 0$ (días) | **Requerido** | No |
| `Age` | Numérica (int/float) | $> 0$ (años) | **Requerido** | No |
| `Headache` | Binaria (int) | `0` (no), `1` (sí) | **Requerido** | **Requerido** |
| `Myalgia` | Binaria (int) | `0` (no), `1` (sí) | **Requerido** | **Requerido** |
| `Rash` | Binaria (int) | `0` (no), `1` (sí) | **Requerido** | **Requerido** |
| `Fever` | Binaria (int) | `0` (no), `1` (sí) | **Requerido** | **Requerido** |
| `Outcome` | Binaria (int) | `0` (negativo), `1` (positivo) | **Requerido (como target $y$)** | **Requerido (como feature)** |
| `Gender` | Binaria (int) | `0` (Male), `1` (Female) | Opcional (subgrupos) | Opcional |
| `source` | Texto | `"2025"`, `"2026"` | Opcional (subgrupos) | Opcional |
| `Platelet_Count` | Numérica | Cualquier valor / null | Ignorada | Ignorada |
| `WBC_Count` | Numérica | Cualquier valor / null | Ignorada | Ignorada |

---

## 4. Endpoints Obligatorios de la API (`src/api/`)

1. **`GET /health`**
   - Estado del servicio, versión de API, hashes SHA-256 de los artefactos `.joblib` y verificación de integridad.
2. **`POST /evaluate`** (Multipart form: `file: UploadFile`, `model: str = "both"`)
   - Recibe el CSV de hold-out (`validacion_produccion.csv` u otro lote).
   - Valida esquema, tipos de datos y rangos.
   - Ejecuta CatBoost (si `model` es `"catboost"` o `"both"`).
   - Ejecuta K-Modes (si `model` es `"kmodes"` o `"both"`).
   - Retorna payload JSON estructurado con métricas, curvas y warnings.
3. **`POST /predict/clinical-support`** (JSON payload)
   - Permite ingresar un paciente individual para triage clínico, devolviendo `P(dengue)`, predicción según umbral óptimo y sugerencia de laboratorio/enrutamiento.

---

## 5. Lineamientos de Frontend (`src/web/`)

1. **Módulo de Evaluación Batch (Hold-out / Subida de Datos):**
   - Drag & drop para archivo CSV.
   - Selector de análisis: CatBoost, K-Modes o Ambos.
   - **Dashboard CatBoost:** Tarjetas de KPIs (Accuracy, F1, F2, Recall, Precision, ROC-AUC), Matriz de Confusión gráfica interactiva, Curva ROC, Curva Precision-Recall, Tabla de Intervalos de Confianza al 95%, y análisis por subgrupos.
   - **Dashboard K-Modes:** Gráficos de barras apiladas de síntomas por cluster, tarjetas descriptivas de los 3 perfiles clínicos con texto en lenguaje natural, y semáforo de estabilidad (PSI, Costo Hamming vs Train, Silhouette).
   - Opción para exportar/descargar el CSV enriquecido con las predicciones y clusters asignados.
2. **Módulo de Triage Individual (Apoyo Clínico):**
   - Formulario de entrada rápida con validaciones inmediatas.
   - Tarjeta de resultado con `P(dengue)`, umbral de decisión y disclaimer de no-diagnóstico médico.
