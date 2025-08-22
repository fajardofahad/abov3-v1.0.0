@echo off
echo Installing ABOV3 globally...
echo.

REM Copy abov3.exe to Windows directory (which is always in PATH)
copy "C:\Users\fajar\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.13_qbz5n2kfra8p0\LocalCache\local-packages\Python313\Scripts\abov3.exe" "%WINDIR%\abov3.exe"

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ✅ SUCCESS: ABOV3 installed globally!
    echo You can now run 'abov3' from anywhere.
    echo.
    echo Test it:
    echo   abov3 --version
    echo   abov3
) else (
    echo.
    echo ❌ ERROR: Failed to copy file. Try running as Administrator.
    echo.
    echo Alternative: Use abov3_direct.bat from this directory
)

echo.
pause