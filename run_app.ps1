# Script PowerShell para iniciar la aplicación Dengue Risk Triage
$PSScriptRoot = Split-Path -Parent -Path $MyInvocation.MyCommand.Definition
Set-Location -Path $PSScriptRoot

Write-Host "========================================================" -ForegroundColor Cyan
Write-Host "Iniciando Servidor Dengue Risk Triage (CatBoost + K-Modes)" -ForegroundColor Cyan
Write-Host "========================================================" -ForegroundColor Cyan

$uvicorn = Join-Path $PSScriptRoot ".venv\Scripts\uvicorn.exe"

if (Test-Path $uvicorn) {
    Write-Host "Iniciando servidor en http://127.0.0.1:8000 ..." -ForegroundColor Green
    Start-Process "http://127.0.0.1:8000"
    & $uvicorn "src.api.main:app" --host 127.0.0.1 --port 8000 --reload
} else {
    Write-Host "No se encontró el ejecutable uvicorn en .venv. Ejecute: pip install -r requirements.txt" -ForegroundColor Red
}
