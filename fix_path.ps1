# PowerShell script to add Python Scripts to PATH permanently
Write-Host "Adding Python Scripts directory to PATH..." -ForegroundColor Green

$scriptsDir = "C:\Users\fajar\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.13_qbz5n2kfra8p0\LocalCache\local-packages\Python313\Scripts"

# Get current user PATH
$currentPath = [Environment]::GetEnvironmentVariable("PATH", "User")

# Check if already in PATH
if ($currentPath -split ";" -contains $scriptsDir) {
    Write-Host "Python Scripts directory is already in PATH!" -ForegroundColor Yellow
} else {
    # Add to PATH
    $newPath = $currentPath + ";" + $scriptsDir
    [Environment]::SetEnvironmentVariable("PATH", $newPath, "User")
    
    Write-Host ""
    Write-Host "✅ Successfully added to PATH!" -ForegroundColor Green
    Write-Host "Python Scripts directory: $scriptsDir" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "🔄 Please restart PowerShell/Command Prompt for changes to take effect." -ForegroundColor Yellow
    Write-Host ""
    Write-Host "After restarting, you can run:" -ForegroundColor Yellow
    Write-Host "  abov3 --version" -ForegroundColor Cyan
    Write-Host "  abov3" -ForegroundColor Cyan
}

Write-Host ""
Write-Host "🚀 ABOV3 is ready to use!" -ForegroundColor Green
Read-Host "Press Enter to continue"