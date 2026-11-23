# Dengue Risk Triage & Analytic Evaluation App

Herramienta de **evaluación analítica y apoyo a la decisión clínica** para dengue en Argentina basada en **CatBoost (supervisado)** y **K-Modes (interpretabilidad de perfiles)**.

---

## Características Principales

1. **Evaluación de Lotes Hold-Out (Supervisada / Con Outcome):**
   - Carga de datasets de validación (ej. `sample_holdout.csv`).
   - Inferencia en memoria con **CatBoost** (6 predictores clínicos, umbral óptimo ~0.5602).
   - Generación de matriz de confusión gráfica, curvas ROC y Precision-Recall interactivas, e Intervalos de Confianza al 95% por Bootstrap (1000 remuestreos).
   - Agrupamiento con **K-Modes (con Outcome)** para evaluar la correspondencia de perfiles y pureza diagnóstica.
   - Exportación de CSV enriquecido con probabilidades y predicciones.

2. **Clustering Fenotípico de Cohortes (No Supervisada / Sin Outcome):**
   - Carga de datasets de pacientes sin diagnóstico confirmado (ej. `sample_sintomas_sin_outcome.csv`).
   - **No requiere columna Outcome**: utiliza exclusivamente los 4 síntomas cardinales (`Fever`, `Headache`, `Myalgia`, `Rash`).
   - Clasificación y ubicación de cada paciente en su cluster fenotípico.
   - Comparación de métricas poblacionales contra la línea de base ($N=1.715$): Índice de Estabilidad Poblacional (PSI), distancia Hamming promedio (costo por observación), coeficientes Silhouette y gráfico de prevalencia de síntomas.
   - Exportación de CSV con columnas `cluster_asignado`, `perfil_clinico` y `distancia_centroide`.

3. **Triage Clínico Individual (Guardia / Consultorio):**
   - Formulario paciente por paciente para estimar probabilidad pre-test con CatBoost.
   - Ubicación fenotípica inmediata con K-Modes Solo Síntomas (Cluster 0: oligosintomático, Cluster 1: típico sin rash, Cluster 2: completo con rash).
   - Conducta y laboratorio racional según probabilidad pre-test (Safe Triage: no sobrecargar con NS1/PCR en probabilidad baja por bajo VPP; priorizar métodos directos/indirectos en alta probabilidad).

---

## Estructura del Proyecto

```text
dengue-risk-triage-app/
├── artifacts/
│   ├── metadata.json                     # Configuración, parámetros y baseline de desarrollo
│   ├── modelo_catboost.joblib            # Modelo CatBoost con TunedThresholdClassifierCV
│   ├── modelo_kmodes.joblib              # Modelo K-Modes con Outcome (para hold-out)
│   └── modelo_kmodes_sinoutcoome.joblib  # Modelo K-Modes solo síntomas (para triage de guardia)
├── dev/
│   └── sample_holdout.csv         # Dataset de prueba para evaluación inmediata
├── src/
│   ├── api/
│   │   ├── main.py                # Endpoints FastAPI (/health, /evaluate, /predict)
│   │   └── schemas.py             # Esquemas Pydantic y validaciones
│   ├── ml/
│   │   ├── model_loader.py        # Carga y validación SHA-256 de artefactos
│   │   └── metrics_evaluator.py   # Métricas CatBoost, Bootstrap, K-Modes y PSI
│   └── web/
│       └── index.html             # Dashboard frontend interactivo (Tailwind + Chart.js)
├── tests/
│   └── test_api_integration.py   # Suite de pruebas automatizadas end-to-end
├── Dockerfile                     # Empaquetado de producción
├── docker-compose.yml             # Orquestación de contenedor
├── run_app.bat                    # Lanzador rápido para Windows
├── run_app.ps1                    # Script PowerShell de ejecución local
└── requirements.txt               # Dependencias fijadas
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
| `POST` | `/evaluate` | Carga de archivo CSV (`multipart/form-data`) para evaluación de CatBoost y/o K-Modes. |
| `GET` | `/export-evaluated-csv` | Descarga del último dataset evaluado con predicciones y clusters asignados. |
| `POST` | `/predict/clinical-support` | Inferencia para un paciente individual (`application/json`). |
