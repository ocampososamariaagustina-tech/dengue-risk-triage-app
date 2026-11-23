# Dengue Risk Triage & Analytic Evaluation App

Herramienta de **evaluación analítica y apoyo a la decisión clínica** para dengue en Argentina basada en **CatBoost (supervisado)** y **K-Modes (interpretabilidad de perfiles)**.

---

## Características Principales

1. **Evaluación de Lotes (Hold-Out / CSV):**
   - Carga de datasets de validación (ej. `validacion_produccion.csv`).
   - Inferencia en memoria con **CatBoost** (6 predictores clínicos, umbral óptimo ~0.5602).
   - Generación de matriz de confusión gráfica, curvas ROC y Precision-Recall interactivas, e Intervalos de Confianza al 95% por Bootstrap (1000 remuestreos).
   - Agrupamiento no supervisado con **K-Modes** (3 clusters descriptivos), índice de estabilidad poblacional (PSI), distancias Hamming y coeficientes Silhouette.
   - Exportación de CSV enriquecido con probabilidades, predicciones y nombres de perfil asignados.

2. **Triage Clínico Individual:**
   - Formulario de guardia/consultorio para estimar probabilidad pre-test de dengue con CatBoost.
   - **Clasificación fenotípica inmediata con K-Modes (Solo Síntomas):** permite, dado un paciente, ubicarlo en uno de los 3 clusters descriptivos a partir de sus 4 síntomas cardinales (`Fever`, `Headache`, `Myalgia`, `Rash`), sin requerir `Outcome` ni pruebas de laboratorio previas.
   - Conducta y laboratorio racional según probabilidad y ventana epidemiológica (Safe Triage: no sobrecargar con NS1/PCR en baja probabilidad pre-test por bajo VPP; priorizar métodos directos/indirectos en alta probabilidad).

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
