# Dengue Risk Triage & Analytic Evaluation App

Herramienta de **evaluación analítica y apoyo a la decisión clínica** para dengue en Argentina basada en **CatBoost (supervisado)** y **K-Modes (interpretabilidad de perfiles)**.

---

## Estructura Funcional: Evaluación Clínica vs. Métricas de Modelos

La aplicación separa de forma estricta los dos momentos clave de la atención y auditoría médica:

### 1. Pestaña: Evaluación Clínica (Nuevos Casos — Sin Outcome)
Orientada al médico en guardia o consultorio para pacientes sospechosos de dengue **antes de tener los resultados de laboratorio**:
- **Evaluación Individual:**
  - Permite seleccionar:
    - **Ambos (CatBoost + K-Modes):** Probabilidad de dengue, nivel de riesgo/triage, ventana diagnóstica recomendada y perfil fenotípico.
    - **Solo CatBoost:** Estimación probabilística y recomendación de pruebas virológicas/serológicas (NS1/PCR vs IgM según días de fiebre).
    - **Solo K-Modes (sin Outcome):** Asigna al paciente a uno de los 3 conglomerados clínicos según proximidad Hamming de síntomas.
- **Evaluación por Lote (CSV sin Outcome):**
  - Carga de planillas de pacientes febriles en espera de laboratorio (ej. `dev/sample_nuevos_casos_sin_outcome.csv`).
  - **No requiere ni exige la columna `Outcome`**.
  - Opciones: Ambos, Solo CatBoost o Solo K-Modes.
  - Resumen clínico agregado: % y cantidad de casos de alta sospecha, distribución por ventanas diagnósticas y prevalencia de fenotipos K-Modes.
  - Exportación de CSV con triaje clínico (`/clinical/export-csv`).

### 2. Pestaña: Métricas de Modelos (Hold-Out / Auditoría — Con Outcome)
Orientada a la validación retrospectiva cuando **ya se cuenta con el diagnóstico confirmado por laboratorio** (`Outcome = 1 ó 0`):
- Opciones: **Ambos**, **Solo CatBoost (Supervisado)** o **Solo K-Modes (con Outcome)**.
- **CatBoost:**
  - Métricas completas: F1-Score, Sensibilidad (Recall), ROC-AUC, Accuracy, Precisión, F2-Score, Especificidad, VPN, Brier Score y MCC.
  - Matriz de confusión interactiva con anotaciones clínicas (TN, FP, FN, TP).
  - Curvas interactivas ROC y Precision-Recall (PR).
  - Intervalos de confianza al 95% mediante Bootstrap estratificado ($B=1000$).
  - Desglose de rendimiento por subgrupos (fuente, género).
- **K-Modes:**
  - Estabilidad poblacional mediante el índice **PSI** contra la distribución de desarrollo.
  - Distancia promedio de Hamming al centroide (costo por observación vs baseline de train).
  - Coeficiente Silhouette global y por cluster.
  - Coincidencia y pureza con la variable Outcome (pureza, NMI, ARI, % dengue por cluster).
  - Gráfico de prevalencia de síntomas y centroides por cluster.
- Exportación de CSV evaluado (`/metrics/export-csv`).

---

## Estructura del Proyecto

```text
dengue-risk-triage-app/
├── artifacts/
│   ├── metadata.json                          # Configuración, parámetros y baseline de desarrollo
│   ├── modelo_catboost.joblib                 # Modelo CatBoost con TunedThresholdClassifierCV
│   └── modelo_kmodes.joblib                   # Modelo K-Modes con 3 clusters
├── dev/
│   ├── sample_holdout.csv                     # Dataset con Outcome (para auditoría de Métricas)
│   └── sample_nuevos_casos_sin_outcome.csv    # Dataset sin Outcome (para Evaluación Clínica por Lote)
├── src/
│   ├── api/
│   │   ├── main.py                            # Endpoints FastAPI clínicos y de métricas
│   │   └── schemas.py                         # Esquemas Pydantic y validaciones
│   ├── ml/
│   │   ├── model_loader.py                    # Carga y validación SHA-256 de artefactos
│   │   └── metrics_evaluator.py               # Inferencia clínica, métricas CatBoost, K-Modes y PSI
│   └── web/
│       └── index.html                         # Dashboard interactivo con pestañas clínica y métricas
├── tests/
│   └── test_api_integration.py               # Suite de pruebas automatizadas end-to-end
├── Dockerfile                                 # Empaquetado de producción
├── docker-compose.yml                         # Orquestación de contenedor
├── run_app.bat                                # Lanzador rápido para Windows
├── run_app.ps1                                # Script PowerShell de ejecución local
└── requirements.txt                           # Dependencias fijadas
```

---

## Cómo Ejecutar la Aplicación

### Opción 1: Ejecución Local en Windows (Recomendada)

Hacer doble clic en `run_app.bat` o ejecutar en terminal PowerShell:
```powershell
.\run_app.ps1
```
El servidor iniciará en `http://127.0.0.1:8000` y abrirá automáticamente el navegador.

### Opción 2: Con Docker

```bash
docker compose up --build
```
La aplicación quedará disponible en `http://localhost:8000`.

---

## Endpoints de la API

| Método | Endpoint | Descripción |
| :--- | :--- | :--- |
| `GET` | `/health` | Estado del servicio y hashes SHA-256 de los modelos cargados. |
| `POST` | `/clinical/predict-single` | Triaje clínico individual (Ambos, Solo CatBoost o Solo K-Modes sin Outcome). |
| `POST` | `/clinical/evaluate-batch` | Triaje clínico por lote sobre CSV de nuevos casos (**sin requerir Outcome**). |
| `GET` | `/clinical/export-csv` | Descarga del último lote clínico evaluado con recomendaciones y fenotipos. |
| `POST` | `/metrics/evaluate` | Carga de CSV con `Outcome` para cálculo exhaustivo de métricas supervisadas y PSI. |
| `GET` | `/metrics/export-csv` | Descarga del dataset evaluado en métricas con predicciones y clusters. |
| `POST` | `/predict/clinical-support` | *(Alias de compatibilidad)* Inferencia individual de apoyo clínico. |
| `POST` | `/evaluate` | *(Alias de compatibilidad)* Evaluación por lote con Outcome. |
| `GET` | `/export-evaluated-csv` | *(Alias de compatibilidad)* Descarga de CSV evaluado con Outcome. |

