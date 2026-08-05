param(
    [switch]$Local,
    [switch]$Dev
)

$ErrorActionPreference = "Stop"

Write-Host "=== G-MASS Setup ==="

$pythonCandidates = @("py -3", "python")
$pythonCmd = $null

foreach ($candidate in $pythonCandidates) {
    $parts = $candidate -split " "
    $exe = $parts[0]
    $args = @()
    if ($parts.Count -gt 1) {
        $args = $parts[1..($parts.Count - 1)]
    }

    $found = Get-Command $exe -ErrorAction SilentlyContinue
    if ($found) {
        $pythonCmd = @{ Exe = $exe; Args = $args }
        break
    }
}

if (-not $pythonCmd) {
    Write-Error "FAIL Python 3.10+ required"
}

& $pythonCmd.Exe @($pythonCmd.Args + @("-c", "import sys; raise SystemExit(0 if sys.version_info >= (3,10) else 1)"))
if ($LASTEXITCODE -ne 0) {
    Write-Error "FAIL Python 3.10+ required"
}

$version = & $pythonCmd.Exe @($pythonCmd.Args + @("--version"))
Write-Host "OK $version"

if (Test-Path ".env") {
    Write-Host "OK .env exists"
} elseif (Test-Path ".env.example") {
    Copy-Item ".env.example" ".env"
    Write-Host "OK .env created from .env.example"
    Write-Host "Fill in your real API keys in .env before running the project."
} else {
    Write-Error "FAIL .env missing and .env.example was not found"
}

if (Test-Path ".env") {
    $envText = Get-Content -Raw ".env"
    if ($envText -match "your_[A-Za-z0-9_]+|YOUR_[A-Z0-9_]+") {
        Write-Host "WARNING: .env contains placeholder values. Update .env with real keys before use."
    }
}

& $pythonCmd.Exe @($pythonCmd.Args + @("-m", "pip", "install", "--upgrade", "pip", "--quiet"))
& $pythonCmd.Exe @($pythonCmd.Args + @("-m", "pip", "install", "-r", "requirements.txt", "--quiet"))
Write-Host "OK base dependencies and editable gmass CLI installed"

if ($Local) {
    & $pythonCmd.Exe @($pythonCmd.Args + @("-m", "pip", "install", "-r", "requirements-local.txt", "--quiet"))
    Write-Host "OK local Transformers backend dependencies installed"
} else {
    Write-Host "SKIP local Transformers dependencies (run .\setup.ps1 -Local to install them)"
}

if ($Dev) {
    & $pythonCmd.Exe @($pythonCmd.Args + @("-m", "pip", "install", "-e", ".[dev]", "--quiet"))
    Write-Host "OK developer dependencies installed"
}

New-Item -ItemType Directory -Force -Path "scorer\models" | Out-Null
if (Test-Path "scorer\models\lid.176.ftz") {
    Write-Host "OK fasttext LID model present"
} else {
    Invoke-WebRequest `
        -Uri "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz" `
        -OutFile "scorer\models\lid.176.ftz"
    Write-Host "OK fasttext LID downloaded"
}

& $pythonCmd.Exe @($pythonCmd.Args + @("scripts/check_environment.py"))
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

Write-Host "=== Setup complete ==="
Write-Host ""
Write-Host "Next steps:"
Write-Host "  1. Fill in .env with your API keys if needed"
Write-Host "  2. python scripts/check_environment.py"
Write-Host "  3. gmass phi3 --probe-file data/probes/simulation_set_6_probes.jsonl --full"
