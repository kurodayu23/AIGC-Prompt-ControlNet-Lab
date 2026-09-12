param(
    [string]$Python = "python",
    [string]$OutputDirectory = (Join-Path $PSScriptRoot "../dist"),
    [string]$Version = "1.0.0"
)
$ErrorActionPreference = "Stop"
$projectRoot = Split-Path $PSScriptRoot -Parent
$buildPath = Join-Path $projectRoot "build/package"
New-Item -ItemType Directory -Force -Path $buildPath, $OutputDirectory | Out-Null
& $Python -m PyInstaller --noconfirm --clean --windowed --name "AIGCPromptStudio" --distpath $OutputDirectory --workpath $buildPath --specpath $buildPath --add-data "$projectRoot/prompts;prompts" "$projectRoot/prompt_desktop.py"
if ($LASTEXITCODE -ne 0) { throw "PyInstaller failed" }
Copy-Item -LiteralPath (Join-Path $PSScriptRoot "WINDOWS_PACKAGE_README.txt") -Destination (Join-Path $OutputDirectory "AIGCPromptStudio/使用说明.txt")
$zipPath = Join-Path $OutputDirectory "AIGCPromptStudio-v$Version-Windows-x64.zip"
if (Test-Path -LiteralPath $zipPath) { throw "输出包已存在，请使用新的版本号或目录：$zipPath" }
Compress-Archive -LiteralPath (Join-Path $OutputDirectory "AIGCPromptStudio") -DestinationPath $zipPath
Write-Output $zipPath
