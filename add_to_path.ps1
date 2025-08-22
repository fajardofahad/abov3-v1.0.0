# PowerShell script to add Python Scripts directory to PATH
Write-Host "Adding Python Scripts directory to PATH..." -ForegroundColor Green

# Get the Python Scripts directory
$pythonScriptsDir = "C:\Users\fajar\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.13_qbz5n2kfra8p0\LocalCache\local-packages\Python313\Scripts"

# Get current user PATH
$currentPath = [Environment]::GetEnvironmentVariable("PATH", "User")

# Check if already in PATH
if ($currentPath -split ";" -contains $pythonScriptsDir) {
    Write-Host "Python Scripts directory is already in PATH!" -ForegroundColor Yellow
} else {
    # Add to PATH
    $newPath = $currentPath + ";" + $pythonScriptsDir
    [Environment]::SetEnvironmentVariable("PATH", $newPath, "User")
    
    Write-Host ""
    Write-Host "✅ Added to PATH successfully!" -ForegroundColor Green
    Write-Host "Please restart your command prompt/terminal for changes to take effect."
    Write-Host ""
    Write-Host "After restarting, you can run:" -ForegroundColor Yellow
    Write-Host "  abov3 --version" -ForegroundColor Cyan
    Write-Host "  abov3" -ForegroundColor Cyan
}

Write-Host ""
Read-Host "Press Enter to continue"