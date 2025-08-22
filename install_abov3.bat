@echo off
REM =============================================================================
REM ABOV3 4 Ollama - Windows Batch Installer
REM =============================================================================
REM
REM This batch file provides a Windows-native installer for ABOV3 4 Ollama.
REM It automatically detects Python, installs dependencies, and sets up the
REM complete ABOV3 development environment with Ollama integration.
REM
REM Features:
REM - Automatic Python detection and validation
REM - Administrative privilege elevation when needed
REM - Comprehensive error handling and recovery
REM - Progress reporting and user feedback
REM - Integration with Windows Registry and Start Menu
REM - Firewall and antivirus compatibility checks
REM
REM Author: ABOV3 Enterprise DevOps Agent
REM Version: 1.0.0
REM =============================================================================

setlocal enabledelayedexpansion
set "SCRIPT_DIR=%~dp0"
set "SCRIPT_NAME=%~nx0"
set "LOG_FILE=%TEMP%\abov3_install.log"
set "PYTHON_REQUIRED=3.8"
set "INSTALL_SUCCESS=0"

REM Color codes for console output
set "RED=[91m"
set "GREEN=[92m"
set "YELLOW=[93m"
set "BLUE=[94m"
set "MAGENTA=[95m"
set "CYAN=[96m"
set "WHITE=[97m"
set "BOLD=[1m"
set "RESET=[0m"

REM Initialize log file
echo ABOV3 4 Ollama Windows Installation Log > "%LOG_FILE%"
echo ========================================== >> "%LOG_FILE%"
echo Start Time: %DATE% %TIME% >> "%LOG_FILE%"
echo Script Location: %SCRIPT_DIR% >> "%LOG_FILE%"
echo User: %USERNAME% >> "%LOG_FILE%"
echo Computer: %COMPUTERNAME% >> "%LOG_FILE%"
echo Windows Version: >> "%LOG_FILE%"
ver >> "%LOG_FILE%" 2>&1
echo. >> "%LOG_FILE%"

:MAIN
call :PRINT_BANNER
call :CHECK_ADMIN_PRIVILEGES
call :DETECT_PYTHON
call :VALIDATE_SYSTEM_REQUIREMENTS
call :RUN_PYTHON_INSTALLER
call :VERIFY_INSTALLATION
call :REGISTER_UNINSTALLER
call :PRINT_COMPLETION_MESSAGE
goto :EOF

:PRINT_BANNER
echo %CYAN%%BOLD%
echo ╔══════════════════════════════════════════════════════════════════╗
echo ║                    ABOV3 4 OLLAMA INSTALLER                     ║
echo ║                                                                  ║
echo ║  Revolutionary AI-Powered Coding Platform                       ║
echo ║  Enterprise-Grade Infrastructure ^& Deployment                   ║
echo ║                                                                  ║
echo ║  Version: 1.0.0 ^| Platform: Windows                             ║
echo ║                                                                  ║
echo ╚══════════════════════════════════════════════════════════════════╝
echo %RESET%
echo.
call :LOG "Installation started by %USERNAME% on %COMPUTERNAME%"
goto :EOF

:CHECK_ADMIN_PRIVILEGES
echo %YELLOW%Checking administrative privileges...%RESET%
call :LOG "Checking administrative privileges"

REM Check if running as administrator
net session >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo %YELLOW%Administrative privileges recommended for full installation.%RESET%
    echo %YELLOW%Some features may require elevation:%RESET%
    echo   - System PATH modification
    echo   - Start Menu integration
    echo   - Firewall configuration
    echo.
    set /p "ELEVATE=Request administrative privileges? (y/N): "
    if /i "!ELEVATE!"=="y" (
        call :LOG "Requesting administrative elevation"
        echo %CYAN%Requesting administrative privileges...%RESET%
        powershell -Command "Start-Process '%~f0' -Verb RunAs"
        exit /b 0
    ) else (
        call :LOG "Continuing without administrative privileges"
        echo %YELLOW%Continuing without administrative privileges...%RESET%
    )
) else (
    call :LOG "Running with administrative privileges"
    echo %GREEN%✓ Running with administrative privileges%RESET%
)
echo.
goto :EOF

:DETECT_PYTHON
echo %CYAN%Detecting Python installation...%RESET%
call :LOG "Starting Python detection"

set "PYTHON_EXE="
set "PYTHON_VERSION="

REM Try common Python executables
for %%p in (python python3 py) do (
    where %%p >nul 2>&1
    if !ERRORLEVEL! equ 0 (
        call :LOG "Found %%p executable"
        for /f "tokens=2" %%v in ('%%p --version 2^>^&1') do (
            set "PYTHON_VERSION=%%v"
            set "PYTHON_EXE=%%p"
            call :LOG "Python version: %%v"
            goto :PYTHON_FOUND
        )
    )
)

REM Check specific Python installation paths
for %%d in (
    "%LOCALAPPDATA%\Programs\Python"
    "%PROGRAMFILES%\Python*"
    "%PROGRAMFILES(X86)%\Python*"
    "C:\Python*"
) do (
    if exist "%%d\python.exe" (
        call :LOG "Found Python at %%d"
        for /f "tokens=2" %%v in ('"%%d\python.exe" --version 2^>^&1') do (
            set "PYTHON_VERSION=%%v"
            set "PYTHON_EXE=%%d\python.exe"
            call :LOG "Python version: %%v"
            goto :PYTHON_FOUND
        )
    )
)

:PYTHON_NOT_FOUND
echo %RED%✗ Python not found!%RESET%
call :LOG "ERROR: Python not found on system"
echo.
echo Python 3.8 or higher is required for ABOV3 4 Ollama.
echo.
echo %YELLOW%Download Python from: https://www.python.org/downloads/%RESET%
echo.
echo %CYAN%Installation recommendations:%RESET%
echo - Use Python 3.10+ for best compatibility
echo - Check "Add Python to PATH" during installation
echo - Include pip package manager
echo.
set /p "INSTALL_PYTHON=Open Python download page? (Y/n): "
if /i not "!INSTALL_PYTHON!"=="n" (
    start https://www.python.org/downloads/
)
call :LOG "Installation aborted - Python not found"
pause
exit /b 1

:PYTHON_FOUND
echo %GREEN%✓ Python found: %PYTHON_EXE%%RESET%
echo %GREEN%✓ Version: %PYTHON_VERSION%%RESET%
call :LOG "Python detected successfully: %PYTHON_EXE% (%PYTHON_VERSION%)"

REM Validate Python version
call :VALIDATE_PYTHON_VERSION "%PYTHON_VERSION%"
if !ERRORLEVEL! neq 0 (
    echo %RED%✗ Python version %PYTHON_VERSION% is too old%RESET%
    echo %YELLOW%Python %PYTHON_REQUIRED% or higher is required%RESET%
    call :LOG "ERROR: Python version too old (%PYTHON_VERSION%)"
    pause
    exit /b 1
)

echo %GREEN%✓ Python version validation passed%RESET%
echo.
goto :EOF

:VALIDATE_PYTHON_VERSION
set "VERSION=%~1"
set "REQUIRED=%PYTHON_REQUIRED%"

REM Extract major and minor version numbers
for /f "tokens=1,2 delims=." %%a in ("%VERSION%") do (
    set "VERSION_MAJOR=%%a"
    set "VERSION_MINOR=%%b"
)

for /f "tokens=1,2 delims=." %%a in ("%REQUIRED%") do (
    set "REQUIRED_MAJOR=%%a"
    set "REQUIRED_MINOR=%%b"
)

REM Compare versions
if %VERSION_MAJOR% gtr %REQUIRED_MAJOR% (
    exit /b 0
) else if %VERSION_MAJOR% equ %REQUIRED_MAJOR% (
    if %VERSION_MINOR% geq %REQUIRED_MINOR% (
        exit /b 0
    )
)

exit /b 1

:VALIDATE_SYSTEM_REQUIREMENTS
echo %CYAN%Validating system requirements...%RESET%
call :LOG "Validating system requirements"

REM Check available disk space (minimum 2GB)
for /f "tokens=3" %%a in ('dir /-c "%SCRIPT_DIR%" 2^>nul ^| find "bytes free"') do (
    set "FREE_SPACE=%%a"
)

if defined FREE_SPACE (
    set /a "FREE_GB=!FREE_SPACE!/1073741824"
    if !FREE_GB! lss 2 (
        echo %RED%✗ Insufficient disk space%RESET%
        echo %YELLOW%Required: 2GB, Available: !FREE_GB!GB%RESET%
        call :LOG "ERROR: Insufficient disk space (!FREE_GB!GB available)"
        pause
        exit /b 1
    ) else (
        echo %GREEN%✓ Disk space: !FREE_GB!GB available%RESET%
        call :LOG "Disk space check passed: !FREE_GB!GB available"
    )
) else (
    echo %YELLOW%? Could not determine available disk space%RESET%
    call :LOG "WARNING: Could not check disk space"
)

REM Check network connectivity
echo %CYAN%Testing network connectivity...%RESET%
ping -n 1 8.8.8.8 >nul 2>&1
if %ERRORLEVEL% equ 0 (
    echo %GREEN%✓ Network connectivity verified%RESET%
    call :LOG "Network connectivity test passed"
) else (
    echo %YELLOW%? Network connectivity issues detected%RESET%
    echo %YELLOW%Some installation steps may fail without internet access%RESET%
    call :LOG "WARNING: Network connectivity issues"
    set /p "CONTINUE=Continue anyway? (y/N): "
    if /i not "!CONTINUE!"=="y" (
        call :LOG "Installation aborted by user - network issues"
        exit /b 1
    )
)

REM Check Windows version compatibility
for /f "tokens=4 delims=[]" %%a in ('ver') do set "WIN_VERSION=%%a"
echo %GREEN%✓ Windows version: %WIN_VERSION%%RESET%
call :LOG "Windows version: %WIN_VERSION%"

REM Check for Windows Defender / Antivirus
sc query "WinDefend" >nul 2>&1
if %ERRORLEVEL% equ 0 (
    echo %YELLOW%! Windows Defender detected%RESET%
    echo %YELLOW%  You may need to add exceptions for ABOV3 if installation fails%RESET%
    call :LOG "Windows Defender detected - may require exceptions"
)

echo.
goto :EOF

:RUN_PYTHON_INSTALLER
echo %CYAN%Running Python installer...%RESET%
call :LOG "Starting Python installer"

REM Check if Python installer exists
if not exist "%SCRIPT_DIR%install_abov3.py" (
    echo %RED%✗ Python installer not found: %SCRIPT_DIR%install_abov3.py%RESET%
    call :LOG "ERROR: Python installer script not found"
    pause
    exit /b 1
)

echo %CYAN%Executing: "%PYTHON_EXE%" "%SCRIPT_DIR%install_abov3.py"%RESET%
call :LOG "Executing Python installer: %PYTHON_EXE% %SCRIPT_DIR%install_abov3.py"

REM Run the Python installer with real-time output
"%PYTHON_EXE%" "%SCRIPT_DIR%install_abov3.py" --verbose
set "PYTHON_EXIT_CODE=%ERRORLEVEL%"

call :LOG "Python installer completed with exit code: %PYTHON_EXIT_CODE%"

if %PYTHON_EXIT_CODE% neq 0 (
    echo.
    echo %RED%✗ Python installer failed with exit code: %PYTHON_EXIT_CODE%%RESET%
    call :LOG "ERROR: Python installer failed"
    
    echo.
    echo %YELLOW%Troubleshooting steps:%RESET%
    echo 1. Check the log file: %LOG_FILE%
    echo 2. Ensure you have internet connectivity
    echo 3. Try running as administrator
    echo 4. Check antivirus software settings
    echo 5. Verify Python and pip are working: %PYTHON_EXE% -m pip --version
    echo.
    
    set /p "VIEW_LOG=View installation log? (Y/n): "
    if /i not "!VIEW_LOG!"=="n" (
        if exist "%LOG_FILE%" (
            type "%LOG_FILE%"
        ) else (
            echo Log file not found.
        )
    )
    
    pause
    exit /b %PYTHON_EXIT_CODE%
) else (
    echo %GREEN%✓ Python installer completed successfully%RESET%
    set "INSTALL_SUCCESS=1"
)

echo.
goto :EOF

:VERIFY_INSTALLATION
echo %CYAN%Verifying installation...%RESET%
call :LOG "Starting installation verification"

REM Test ABOV3 command
echo %CYAN%Testing ABOV3 command...%RESET%
"%PYTHON_EXE%" -m abov3 --help >nul 2>&1
if %ERRORLEVEL% equ 0 (
    echo %GREEN%✓ ABOV3 command available%RESET%
    call :LOG "ABOV3 command verification passed"
) else (
    echo %YELLOW%? ABOV3 command test failed%RESET%
    call :LOG "WARNING: ABOV3 command verification failed"
)

REM Test Ollama
echo %CYAN%Testing Ollama...%RESET%
where ollama >nul 2>&1
if %ERRORLEVEL% equ 0 (
    ollama --version >nul 2>&1
    if !ERRORLEVEL! equ 0 (
        echo %GREEN%✓ Ollama available%RESET%
        call :LOG "Ollama verification passed"
        
        REM Test default model
        echo %CYAN%Testing default model...%RESET%
        ollama list 2>nul | find "llama3.2" >nul
        if !ERRORLEVEL! equ 0 (
            echo %GREEN%✓ Default model (llama3.2) available%RESET%
            call :LOG "Default model verification passed"
        ) else (
            echo %YELLOW%? Default model not found%RESET%
            call :LOG "WARNING: Default model not available"
        )
    ) else (
        echo %YELLOW%? Ollama not responding%RESET%
        call :LOG "WARNING: Ollama not responding"
    )
) else (
    echo %YELLOW%? Ollama not found in PATH%RESET%
    call :LOG "WARNING: Ollama not found in PATH"
)

echo.
goto :EOF

:REGISTER_UNINSTALLER
echo %CYAN%Registering uninstaller...%RESET%
call :LOG "Registering uninstaller in Windows Registry"

REM Only register if we have admin privileges
net session >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo %YELLOW%Skipping uninstaller registration (requires admin privileges)%RESET%
    call :LOG "Skipping uninstaller registration - no admin privileges"
    goto :EOF
)

REM Create uninstall script
set "UNINSTALL_SCRIPT=%SCRIPT_DIR%uninstall_abov3.bat"
(
echo @echo off
echo REM ABOV3 4 Ollama Uninstaller
echo echo Uninstalling ABOV3 4 Ollama...
echo "%PYTHON_EXE%" -m pip uninstall -y abov3-ollama
echo echo Cleaning up configuration files...
echo rd /s /q "%%APPDATA%%\ABOV3" 2^>nul
echo echo Removing desktop shortcuts...
echo del "%%USERPROFILE%%\Desktop\ABOV3 4 Ollama.lnk" 2^>nul
echo echo Removing from Start Menu...
echo rd /s /q "%%APPDATA%%\Microsoft\Windows\Start Menu\Programs\ABOV3" 2^>nul
echo echo ABOV3 4 Ollama has been uninstalled.
echo pause
) > "%UNINSTALL_SCRIPT%"

REM Register in Windows Programs and Features
reg add "HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall\ABOV3-Ollama" /v "DisplayName" /t REG_SZ /d "ABOV3 4 Ollama" /f >nul 2>&1
reg add "HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall\ABOV3-Ollama" /v "DisplayVersion" /t REG_SZ /d "1.0.0" /f >nul 2>&1
reg add "HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall\ABOV3-Ollama" /v "Publisher" /t REG_SZ /d "ABOV3 Enterprise" /f >nul 2>&1
reg add "HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall\ABOV3-Ollama" /v "UninstallString" /t REG_SZ /d "\"%UNINSTALL_SCRIPT%\"" /f >nul 2>&1
reg add "HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall\ABOV3-Ollama" /v "InstallLocation" /t REG_SZ /d "%SCRIPT_DIR%" /f >nul 2>&1
reg add "HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall\ABOV3-Ollama" /v "DisplayIcon" /t REG_SZ /d "%SCRIPT_DIR%assets\icon.ico" /f >nul 2>&1
reg add "HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall\ABOV3-Ollama" /v "NoModify" /t REG_DWORD /d 1 /f >nul 2>&1
reg add "HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall\ABOV3-Ollama" /v "NoRepair" /t REG_DWORD /d 1 /f >nul 2>&1

if %ERRORLEVEL% equ 0 (
    echo %GREEN%✓ Uninstaller registered%RESET%
    call :LOG "Uninstaller registered successfully"
) else (
    echo %YELLOW%? Could not register uninstaller%RESET%
    call :LOG "WARNING: Could not register uninstaller"
)

goto :EOF

:PRINT_COMPLETION_MESSAGE
if %INSTALL_SUCCESS% equ 1 (
    echo %GREEN%%BOLD%
    echo ╔══════════════════════════════════════════════════════════════════╗
    echo ║                    INSTALLATION SUCCESSFUL!                     ║
    echo ║                                                                  ║
    echo ║  🎉 ABOV3 4 Ollama has been installed successfully!             ║
    echo ║                                                                  ║
    echo ║  Quick Start:                                                    ║
    echo ║  • Open Command Prompt or PowerShell                            ║
    echo ║  • Run 'abov3 --help' to see available commands                 ║
    echo ║  • Run 'abov3 serve' to start the development server            ║
    echo ║  • Run 'abov3 chat' to start an interactive session             ║
    echo ║                                                                  ║
    echo ║  Desktop shortcut and Start Menu entry have been created.       ║
    echo ║                                                                  ║
    echo ║  Documentation: https://docs.abov3.com                          ║
    echo ║  Support: support@abov3.com                                     ║
    echo ║                                                                  ║
    echo ╚══════════════════════════════════════════════════════════════════╝
    echo %RESET%
    
    call :LOG "Installation completed successfully"
    
    echo.
    echo %CYAN%Next Steps:%RESET%
    echo 1. %YELLOW%Restart your command prompt%RESET% to ensure PATH changes take effect
    echo 2. %YELLOW%Check firewall settings%RESET% if you experience connectivity issues
    echo 3. %YELLOW%Review configuration%RESET% in %%APPDATA%%\ABOV3\config.json
    echo.
    
    set /p "OPEN_DOCS=Open documentation in browser? (Y/n): "
    if /i not "!OPEN_DOCS!"=="n" (
        start https://docs.abov3.com
    )
    
    set /p "START_ABOV3=Start ABOV3 now? (Y/n): "
    if /i not "!START_ABOV3!"=="n" (
        echo %CYAN%Starting ABOV3...%RESET%
        start cmd /k ""%PYTHON_EXE%" -m abov3 --help && echo. && echo Type 'abov3 serve' to start the server"
    )
    
) else (
    echo %RED%%BOLD%
    echo ╔══════════════════════════════════════════════════════════════════╗
    echo ║                    INSTALLATION FAILED!                         ║
    echo ║                                                                  ║
    echo ║  ❌ ABOV3 4 Ollama installation encountered errors.              ║
    echo ║                                                                  ║
    echo ║  Please check the log file for details:                         ║
    echo ║  %LOG_FILE%                         ║
    echo ║                                                                  ║
    echo ║  Common Solutions:                                               ║
    echo ║  • Run as administrator                                         ║
    echo ║  • Check internet connection                                    ║
    echo ║  • Disable antivirus temporarily                                ║
    echo ║  • Update Python and pip                                        ║
    echo ║                                                                  ║
    echo ║  For support: https://docs.abov3.com/troubleshooting            ║
    echo ║                                                                  ║
    echo ╚══════════════════════════════════════════════════════════════════╝
    echo %RESET%
    
    call :LOG "Installation failed"
)

echo.
call :LOG "Installation script completed at %DATE% %TIME%"
echo %CYAN%Installation log saved to: %LOG_FILE%%RESET%
echo.
pause
goto :EOF

:LOG
set "MSG=%~1"
set "TIMESTAMP=%DATE% %TIME%"
echo [%TIMESTAMP%] %MSG% >> "%LOG_FILE%" 2>&1
goto :EOF

:EOF
REM End of script