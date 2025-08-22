@echo off
echo Installing ABOV3 globally...
echo.

REM Get the current directory (where ABOV3 is located)
set "ABOV3_DIR=%~dp0"

REM Create a batch file in Windows directory (which is always in PATH)
echo @echo off > "%WINDIR%\abov3.bat"
echo cd /d "%ABOV3_DIR%" >> "%WINDIR%\abov3.bat"
echo python abov3.py %%* >> "%WINDIR%\abov3.bat"

REM Also create it in System32 for good measure
echo @echo off > "%WINDIR%\System32\abov3.bat"
echo cd /d "%ABOV3_DIR%" >> "%WINDIR%\System32\abov3.bat"
echo python abov3.py %%* >> "%WINDIR%\System32\abov3.bat"

echo.
echo ✅ ABOV3 installed globally!
echo You can now run 'abov3' from anywhere in Command Prompt or PowerShell.
echo.
echo Test it by opening a new command prompt and typing:
echo   abov3 --version
echo   abov3
echo.
pause