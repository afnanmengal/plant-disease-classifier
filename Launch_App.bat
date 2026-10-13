@echo off
title Plant Disease Classifier
echo ======================================================
echo           Plant Disease Classifier Launching...
echo ======================================================
echo.

cd /d "%~dp0"

:: Check if port 8501 is already listening
netstat -ano | findstr /R /C:":8501 .*LISTENING" >nul 2>&1
if %ERRORLEVEL% equ 0 (
    echo App is already running! Opening in your browser...
    start "" "http://localhost:8501"
    ping 127.0.0.1 -n 2 >nul
    exit /b 0
)

echo Starting application engine (this may take a few seconds)...

:: Prefer conda environment with full RTX GPU acceleration
set "CONDA_PY=%USERPROFILE%\miniconda3\envs\plant-disease-classifier\python.exe"
if exist "%CONDA_PY%" (
    start "" "%CONDA_PY%" -m streamlit run app.py
) else (
    cd /d "%~dp0desktop_app\run"
    start "" "run.exe"
    cd /d "%~dp0"
)

echo Waiting for application to initialize...
set /a attempts=0

:check_loop
ping 127.0.0.1 -n 2 >nul
netstat -ano | findstr /R /C:":8501 .*LISTENING" >nul 2>&1
if %ERRORLEVEL% equ 0 (
    echo App is ready! Opening in your browser...
    start "" "http://localhost:8501"
    ping 127.0.0.1 -n 3 >nul
    exit /b 0
)

set /a attempts+=1
if %attempts% geq 25 (
    echo Opening browser...
    start "" "http://localhost:8501"
    exit /b 0
)
goto check_loop
