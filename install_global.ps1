# PowerShell script to install ABOV3 globally
Write-Host "Installing ABOV3 globally..." -ForegroundColor Green

# Get current directory
$abov3Dir = Get-Location

# Create batch file in Windows directory
$batchContent = @"
@echo off
cd /d "$abov3Dir"
python abov3.py %*
"@

$batchContent | Out-File -FilePath "$env:WINDIR\abov3.bat" -Encoding ASCII
$batchContent | Out-File -FilePath "$env:WINDIR\System32\abov3.bat" -Encoding ASCII

Write-Host ""
Write-Host "✅ ABOV3 installed globally!" -ForegroundColor Green
Write-Host "You can now run 'abov3' from anywhere."
Write-Host ""
Write-Host "Test it by opening a new command prompt and typing:" -ForegroundColor Yellow
Write-Host "  abov3 --version" -ForegroundColor Cyan
Write-Host "  abov3" -ForegroundColor Cyan
Write-Host ""
Read-Host "Press Enter to continue"