param(
    [string]$VolumeName = "models"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
$Source = Join-Path $RepoRoot "experiments"

if (-not (Test-Path $Source)) {
    throw "Checkpoint directory not found: $Source"
}

docker volume inspect $VolumeName *> $null
if ($LASTEXITCODE -ne 0) {
    docker volume create $VolumeName | Out-Null
}

docker run --rm `
    --mount "type=volume,src=$VolumeName,dst=/models" `
    --mount "type=bind,src=$Source,dst=/source,readonly" `
    alpine:3.21 sh -c `
    "mkdir -p /models/SkinTokens/experiments && cp -a /source/. /models/SkinTokens/experiments/"

docker run --rm `
    --mount "type=volume,src=$VolumeName,dst=/models,readonly" `
    alpine:3.21 sh -c `
    "find /models/SkinTokens/experiments -type f -name '*.ckpt' -exec ls -lh {} ;"
