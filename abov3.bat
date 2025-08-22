@echo off
REM ABOV3 4 Ollama - Windows Batch Script Entry Point
REM 
REM This batch file provides a Windows-specific entry point for ABOV3.
REM It handles various Python installation scenarios and ensures proper execution.
REM
REM Usage: abov3.bat [commands...]
REM
REM Author: ABOV3 Team
REM Version: 1.0.0
REM License: MIT

setlocal enabledelayedexpansion

REM Get the directory where this batch file is located
set "SCRIPT_DIR=%~dp0"
set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"

REM Try to find Python executable
set "PYTHON_EXE="

REM Check for python command in PATH
python --version >nul 2>&1
if !errorlevel! equ 0 (
    set "PYTHON_EXE=python"
    goto :found_python
)

REM Check for python3 command in PATH
python3 --version >nul 2>&1
if !errorlevel! equ 0 (
    set "PYTHON_EXE=python3"
    goto :found_python
)

REM Check for py launcher
py --version >nul 2>&1
if !errorlevel! equ 0 (
    set "PYTHON_EXE=py"
    goto :found_python
)

REM If no Python found, show error
echo Error: Python is not installed or not in PATH.
echo Please install Python 3.8+ and ensure it's accessible from command line.
echo Visit: https://www.python.org/downloads/
exit /b 1

:found_python
REM Change to script directory
pushd "%SCRIPT_DIR%"

REM Run ABOV3 with the found Python executable
"%PYTHON_EXE%" abov3.py %*

REM Capture exit code
set "EXIT_CODE=%errorlevel%"

REM Return to original directory
popd

REM Exit with the same code as the Python script
exit /b %EXIT_CODE%