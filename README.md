# Dengue Risk Triage App

> **Trabajo Final Integrador (TFI)** — *(MCTD)*  
> **Autora:** María Agustina Ocampo Sosa - Fernando Enzo Yapura
> **Tecnologías:** Python 3.12 | FastAPI | CatBoost | K-Modes | Docker | Tailwind CSS | Chart.js  

---

## Descripción del Proyecto

El dengue representa un desafío crítico de salud pública en Argentina y la región, caracterizado por brotes epidémicos que generan una alta demanda en los servicios de guardia y consultorios ambulatorios. En la práctica clínica diaria, el profesional médico se enfrenta a dos desafíos principales:
1. **En la consulta inicial (guardia/triaje):** El paciente febril acude con sospecha de dengue, pero **aún no se cuenta con los resultados de laboratorio confirmatorios**. El médico necesita estratificar el riesgo en minutos y determinar qué prueba diagnóstica ordenar según el tiempo de evolución de la fiebre.
2. **En la auditoría epidemiológica y calidad de modelos:** Una vez que las pruebas de laboratorio confirman o descartan los casos (`Outcome = 1 ó 0`), se requiere evaluar rigurosamente la calibración, sensibilidad clínica y estabilidad poblacional de los modelos predictivos frente a derivas de datos (*data drift*).

Esta aplicación resuelve ambos momentos mediante una **arquitectura híbrida complementaria** implementada en **FastAPI** y una interfaz web interactiva.

---

##  Enfoque Metodológico de los Modelos

```text
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│                                 DENGUE RISK TRIAGE SYSTEM                                    │
└──────────────────────────────────────┬──────────────────────────────────────────────────────┘
                                       │
            ┌──────────────────────────┴──────────────────────────┐
            ▼                                                     ▼
┌───────────────────────────────────────┐   ┌─────────────────────────────────────────────────┐
│     1. FASE DE EVALUACIÓN CLÍNICA     │   │      2. FASE DE MÉTRICAS Y AUDITORÍA            │
│       (Nuevos Casos / Sin Outcome)    │   │        (Hold-Out / Casos Confirmados)           │
├───────────────────────────────────────┤   ├─────────────────────────────────────────────────┤
│ • Ámbito: Guardia médica / Consultorio│   │ • Ámbito: Control de calidad / Epidemiología    │
│ • Entrada: Síntomas y días de fiebre  │   │ • Entrada: Dataset con Outcome de laboratorio   │
│ • NO requiere confirmación de dengue  │   │ • Métricas: F1, Sensibilidad, ROC-AUC, Brier    │
│ • CatBoost: Riesgo + Guía Ministerial │   │ • Matriz de Confusión + Curvas ROC y PR         │
│ • K-Modes: Fenotipo sindrómico        │   │ • K-Modes: Índice PSI, Silueta y Pureza         │
│ • Formato: Individual y Lote (CSV)    │   │ • Bootstrap estratificado al 95% (B=1000)       │
└───────────────────────────────────────┘   └─────────────────────────────────────────────────┘
```

### 1. CatBoost Classifier (Modelo Supervisado)
- **Rol:** Estimación probabilística de dengue y priorización clínica.
- **Predictores clínicos (6):** `Fever_Duration`, `Age`, `Fever`, `Headache`, `Myalgia`, `Rash`.
- **Calibración y umbral:** Optimizado mediante validación cruzada con `TunedThresholdClassifierCV` fijando un **umbral óptimo de corte en `0.5602`**. Esto balancea la métrica $F_1$ maximizando la **Sensibilidad (Recall > 94%)** para minimizar falsos negativos en el triaje de guardia.
- **Algoritmo oficial (Ministerio de Salud de la Nación Argentina):**
  - **Fase temprana (0 a 3 días de fiebre):** Prioriza métodos directos de detección virológica (Antígeno NS1 / qRT-PCR).
  - **Fase intermedia (4 a 6 días de fiebre):** Combina métodos directos (NS1) con indirectos (serología IgM).
  - **Fase tardía (7 o más días de fiebre):** Prioriza métodos indirectos de detección serológica (IgM).

### 2. K-Modes Clustering (Modelo No Supervisado / Descriptivo)
- **Rol:** Identificación y explicación de fenotipos clínicos sindrómicos mediante distancia de Hamming sobre variables binarias.
- **Fenotipos identificados (3 clusters estables):**
  - **Cluster 0 — Perfil Sin Cuadro Cardinal / Asintomático:** Ausencia del complejo sintomático típico. Muy baja correlación con dengue.
  - **Cluster 1 — Perfil Dengue Típico:** Fiebre + Cefalea + Mialgia sin erupción cutánea (1, 1, 1, 0).
  - **Cluster 2 — Perfil Dengue Completo:** Presentación clásica florida con exantema / erupción cutánea (1, 1, 1, 1).
- **En Evaluación Clínica:** Asigna el fenotipo sindrómico basándose exclusivamente en los 4 síntomas cardinales sin requerir la variable `Outcome`.
- **En Métricas:** Audita la estabilidad de la población mediante el índice **PSI** (*Population Stability Index*), la distancia promedio de Hamming al centroide y la pureza diagnóstica con `Outcome`.

---

## Guía de Descarga y Puesta en Funcionamiento

### Requisitos Previos
- **Git** instalado en el equipo.
- Para **Opción A (Recomendada):** [Docker Desktop](https://www.docker.com/products/docker-desktop/) instalado y en ejecución.
- Para **Opción B (Local sin Docker):** Python 3.10 o superior (recomendado Python 3.12).

---

### Paso 1: Clonar el Repositorio

Abre una terminal (PowerShell, CMD, Bash o Terminal de macOS/Linux) y clona directamente la rama del proyecto:

```bash
git clone -b feature/separacion-metricas-evaluacion https://github.com/ocampososamariaagustina-tech/dengue-risk-triage-app.git
cd dengue-risk-triage-app
```

> Si ya habías clonado el repositorio anteriormente, sitúate en la carpeta y cambia a la rama actualizada:
> ```bash
> git fetch origin
> git checkout feature/separacion-metricas-evaluacion
> git pull origin feature/separacion-metricas-evaluacion
> ```

---

### Opción A: Ejecutar con Docker (La más rápida y recomendada)

Este método compila y ejecuta todo el entorno en un contenedor aislado, sin necesidad de configurar librerías locales de Python:

1. Asegúrate de que **Docker Desktop** esté abierto y en ejecución.
2. En la terminal dentro de `dengue-risk-triage-app`, ejecuta:
   ```bash
   docker compose up --build
   ```
3. Espera a que termine la compilación inicial y aparezca:
   ```text
   INFO:     Application startup complete.
   INFO:     Uvicorn running on http://0.0.0.0:8000
   ```
4. Abre tu navegador en:  
   👉 **`http://localhost:8000`**

*(Para detener el contenedor, presiona `Ctrl + C` o ejecuta `docker compose down` en otra terminal).*

---

### Opción B: Ejecución Local en Windows (con Python)

Si prefieres ejecutarlo nativamente en Windows:

#### Método Rápido (Scripts incluidos)
Hacer doble clic sobre el archivo **`run_app.bat`** o ejecutar en PowerShell:
```powershell
.\run_app.ps1
```
El script creará automáticamente el entorno virtual `.venv`, instalará las dependencias de `requirements.txt`, levantará el servidor y abrirá tu navegador.

#### Método Manual paso a paso:
1. Crear y activar el entorno virtual:
   ```bash
   python -m venv .venv
   # En Windows:
   .venv\Scripts\activate
   # En Linux/macOS:
   source .venv/bin/activate
   ```
2. Instalar dependencias fijadas:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
3. Iniciar el servidor:
   ```bash
   uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
   ```
4. Ingresar en el navegador a **`http://localhost:8000`**.

---

### Paso 2: Verificar la Integridad del Sistema (Suite de Tests)

El proyecto incluye una suite de pruebas de integración automatizadas que valida los 7 componentes críticos del sistema (carga de modelos con hash SHA-256, inferencia clínica, métricas con Bootstrap, etc.):

```bash
# Con el entorno virtual activado o dentro del contenedor:
python tests/test_api_integration.py
```
Salida esperada:
```text
>>> TODOS LOS TESTS DE INTEGRACIÓN Y VALIDACIÓN PASARON AL 100% <<<
```

---

##  Demostración Paso a Paso

Para presentar la aplicación de manera clara y fluida, se recomienda seguir este recorrido demostrativo utilizando los datasets de prueba incluidos en la carpeta `dev/`:

### 1. Triage Individual (Paciente en Guardia)
1. En la pestaña **"Evaluación Clínica (Nuevos Casos)"**, seleccionar **Triage Individual**.
2. Elegir modelo: **Ambos (CatBoost + K-Modes)**.
3. Ingresar los datos de un paciente con sospecha temprana:
   - Días de fiebre: `2` | Edad: `32` | Fiebre: `Sí (1)` | Cefalea: `Sí (1)` | Mialgia: `Sí (1)` | Exantema: `No (0)`.
4. Pulsar **"Calcular Riesgo y Conducta"**:
   - **CatBoost:** Asigna probabilidad alta de dengue (> 85%), clasifica en *Alta Prioridad* y recomienda la solicitud inmediata de **método directo (Antígeno NS1 / qRT-PCR)** por encontrarse en los días 0 a 3.
   - **K-Modes:** Asigna el fenotipo **Cluster 1: Dengue Típico** explicando que presenta la tríada cardinal clásica sin compromiso cutáneo.
5. Cambiar el selector a **"Solo K-Modes"** para mostrar cómo funciona la asignación sindrómica pura sin requerir CatBoost ni Outcome.

### 2. Evaluación Clínica por Lote (Pacientes Nuevos sin Laboratorio)
1. Dentro de la pestaña de Evaluación Clínica, hacer clic en el botón superior **"Evaluación por Lote (CSV)"**.
2. Arrastrar el archivo de prueba: **`dev/sample_nuevos_casos_sin_outcome.csv`**.
3. Notar que el sistema **no requiere ni exige la columna Outcome**, pues emula una tanda de 30 pacientes recién ingresados a triaje.
4. Pulsar **"Procesar Lote y Generar Triaje"**:
   - Se despliega el panel de triaje agregado: % de casos en alta sospecha, desglose de prioridades intermedias y distribución de ventanas diagnósticas según evolución febril.
   - Se muestra la tabla fila a fila con la recomendación médica para cada paciente.
   - Se puede pulsar **"Descargar CSV con Triaje Clínico"** para exportar la planilla enriquecida.

### 3. Auditoría y Métricas Técnicas (Casos Confirmados con Outcome)
1. Hacer clic en la pestaña superior **"Métricas de Modelos (Con Outcome)"**.
2. Arrastrar el archivo de validación: **`dev/sample_holdout.csv`** (este archivo sí contiene la confirmación de laboratorio `Outcome`).
3. Pulsar **"Procesar y Auditar Métricas"**:
   - **CatBoost:** Despliega KPIs (Accuracy: 0.95, F1-Score: 0.9615, Sensibilidad/Recall: 1.0, ROC-AUC: 1.0), la Matriz de Confusión interactiva, las curvas ROC y PR, los intervalos de confianza al 95% obtenidos por Bootstrap ($B=1000$) y el análisis por subgrupos.
   - **K-Modes:** Muestra la estabilidad poblacional mediante el **PSI** ($0.0043 < 0.10$, clasificado en verde como *Población Estable*), la distancia promedio de Hamming al centroide, el coeficiente Silhouette y la pureza diagnóstica (1.0).
4. Probar arrastrar a propósito el archivo sin outcome (`sample_nuevos_casos_sin_outcome.csv`) para mostrarle al profesor que el módulo de métricas cuenta con **validación estricta y control de integridad**, rechazando datasets que carezcan de confirmación diagnóstica.

---

##  Estructura del Repositorio

```text
dengue-risk-triage-app/
├── artifacts/
│   ├── metadata.json                          # Metadatos, umbral (0.5602), baseline de train y métricas CV
│   ├── modelo_catboost.joblib                 # Estimador CatBoost (TunedThresholdClassifierCV)
│   └── modelo_kmodes.joblib                   # Estimador K-Modes (3 clusters sobre variables categóricas)
├── dev/
│   ├── sample_holdout.csv                     # Dataset con Outcome (para auditoría de Métricas)
│   └── sample_nuevos_casos_sin_outcome.csv    # Dataset sin Outcome (para Triaje Clínico por Lote)
├── src/
│   ├── api/
│   │   ├── main.py                            # Endpoints REST en FastAPI con lifespan de artefactos
│   │   └── schemas.py                         # Esquemas Pydantic para request/response de inferencia
│   ├── ml/
│   │   ├── model_loader.py                    # Singleton con verificación de integridad por SHA-256
│   │   └── metrics_evaluator.py               # Lógica de Bootstrap, curvas ROC/PR, PSI y evaluación clínica
│   └── web/
│       └── index.html                         # Frontend SPA con Tailwind CSS y Chart.js
├── tests/
│   └── test_api_integration.py               # Batería de 7 pruebas unitarias e integrales
├── Dockerfile                                 # Definición de contenedor multi-stage en Python 3.12-slim
├── docker-compose.yml                         # Configuración del servicio y mapeo de puertos
├── run_app.bat                                # Lanzador automático de 1 clic para Windows
├── run_app.ps1                                # Script PowerShell de inicialización local
├── requirements.txt                           # Dependencias científicas fijadas
└── README.md                                  # Documentación técnica y académica del proyecto
```

---

##  Endpoints de la API REST

| Método | Endpoint | Entrada | Salida / Rol |
| :--- | :--- | :--- | :--- |
| `GET` | `/health` | Ninguna | Estado del servicio y hashes SHA-256 de los modelos en memoria. |
| `POST` | `/clinical/predict-single` | JSON (`PatientInput`) | Estimación de triaje individual (`model`: `both`, `catboost`, `kmodes`). |
| `POST` | `/clinical/evaluate-batch` | Form-data (CSV sin Outcome) | Triaje clínico por lote con resumen de riesgo y fenotipos. |
| `GET` | `/clinical/export-csv` | Ninguna | Descarga en CSV del último lote de triaje clínico procesado. |
| `POST` | `/metrics/evaluate` | Form-data (CSV con Outcome) | Cálculo de métricas supervisadas, matriz de confusión y PSI. |
| `GET` | `/metrics/export-csv` | Ninguna | Descarga en CSV del último dataset evaluado en métricas. |
| `POST` | `/predict/clinical-support` | JSON (`PatientInput`) | *(Alias retrocompatible)* Inferencia individual de apoyo clínico. |
| `POST` | `/evaluate` | Form-data (CSV con Outcome) | *(Alias retrocompatible)* Evaluación analítica por lote. |
| `GET` | `/export-evaluated-csv` | Ninguna | *(Alias retrocompatible)* Descarga de CSV evaluado. |

---

##  Declaración de Ética y Uso Clínico

Este software ha sido desarrollado con fines académicos y de investigación en el marco del **Trabajo Final Integrador de la materia Metodos Cuantitativos para la Toma de Decisiones**.  
Constituye una **herramienta orientativa de apoyo a la decisión clínica** y **no reemplaza** el criterio médico profesional, la anamnesis completa ni las pruebas confirmatorias de laboratorio según los protocolos de vigilancia epidemiológica vigentes.
