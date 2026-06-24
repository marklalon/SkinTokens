param(
    [string]$VenvPath = ".venv-skintokens",
    [string]$PythonVersion = "3.11",
    [string]$PypiIndex = "https://pypi.org/simple",
    [string]$FlashAttnWheel = "https://huggingface.co/marcorez8/flash-attn-windows-blackwell/resolve/main/flash_attn-2.7.4.post1-cp311-cp311-win_amd64-torch2.7.0-cu128/flash_attn-2.7.4.post1-cp311-cp311-win_amd64.whl",
    [switch]$SkipFlashAttn
)

$ErrorActionPreference = "Stop"

function Invoke-Step {
    param(
        [string]$Name,
        [scriptblock]$Command
    )

    Write-Host ""
    Write-Host "==> $Name" -ForegroundColor Cyan
    & $Command
}

$RepoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $RepoRoot

Invoke-Step "Create project-local venv: $VenvPath" {
    if (-not (Test-Path $VenvPath)) {
        uv venv $VenvPath --python $PythonVersion
    } else {
        Write-Host "Venv already exists: $VenvPath"
    }
}

$PythonExe = Join-Path $VenvPath "Scripts\python.exe"
if (-not (Test-Path $PythonExe)) {
    throw "Python executable not found: $PythonExe"
}

Invoke-Step "Install PyTorch 2.7.0 CUDA 12.8 wheels" {
    uv pip install --python $PythonExe `
        torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 `
        --index-url https://download.pytorch.org/whl/cu128
}

Invoke-Step "Install project requirements" {
    uv pip install --python $PythonExe `
        -r requirements.txt `
        --default-index $PypiIndex
}

if (-not $SkipFlashAttn) {
    Invoke-Step "Install flash-attn" {
        try {
            if ($FlashAttnWheel) {
                uv pip install --python $PythonExe $FlashAttnWheel
            } else {
                uv pip install --python $PythonExe `
                    flash-attn --no-build-isolation `
                    --default-index $PypiIndex
            }
        } catch {
            Write-Host ""
            Write-Host "flash-attn failed to install." -ForegroundColor Yellow
            Write-Host "The base SkinTokens environment is installed; rerun with -SkipFlashAttn to finish verification."
            Write-Host "If you have a compatible flash-attn wheel, rerun with -FlashAttnWheel path\to\flash_attn.whl."
            throw
        }
    }
}

Invoke-Step "Verify environment" {
    & $PythonExe -c "import sys, torch; print(sys.executable); print('torch', torch.__version__); print('cuda available', torch.cuda.is_available()); print('cuda', torch.version.cuda)"
}

Write-Host ""
Write-Host "Done. Activate with:" -ForegroundColor Green
Write-Host "  .\$VenvPath\Scripts\Activate.ps1"
