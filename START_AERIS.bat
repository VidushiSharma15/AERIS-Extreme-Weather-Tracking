@echo off
title AERIS One-Click Launcher
setlocal

:: Navigate to project root directory
cd /d "%~dp0"

:: Launch the main AERIS starter script
call "scripts\start_aeris.bat" %*
