@echo off
echo ========================================================
echo Iniciando Servidor Dengue Risk Triage (CatBoost + K-Modes)
echo ========================================================
cd /d "%~dp0"
if exist .venv\Scripts\uvicorn.exe (
    echo Usando entorno virtual .venv...
    echo Abriendo http://localhost:8000 en el navegador...
    start http://localhost:8000
    .venv\Scripts\uvicorn.exe src.api.main:app --host 127.0.0.1 --port 8000 --reload
) else (
    echo Error: No se encontro el entorno .venv. Ejecute pip install -r requirements.txt
    pause
)
