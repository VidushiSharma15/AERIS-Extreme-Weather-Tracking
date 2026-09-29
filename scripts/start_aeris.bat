@echo off
title AERIS Software Launcher
setlocal

:: Navigate to project root
cd /d "%~dp0.."

:: Run python launcher using existing virtual environment
if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" "scripts\launcher.py" %*
) else (
    python "scripts\launcher.py" %*
)

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [!] AERIS Launcher encountered an error. Exit Code: %ERRORLEVEL%
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo Press any key to close this status window...
pause >nul
