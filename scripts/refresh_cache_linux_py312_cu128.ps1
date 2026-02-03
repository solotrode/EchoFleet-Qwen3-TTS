Param(
    [string]$CacheDir = ".cache",
    [switch]$DeleteIncompatible
)

$ErrorActionPreference = "Stop"

New-Item -ItemType Directory -Force -Path $CacheDir | Out-Null

# We want Linux (manylinux) artifacts for the Docker image.
$platform = "manylinux_2_28_x86_64"
$pythonVersion = "312"
$implementation = "cp"
$abi = "cp312"

# Torch CUDA 12.8 wheel index
$torchIndex = "https://download.pytorch.org/whl/cu128"

if ($DeleteIncompatible) {
    # Delete known-bad cached wheels (wrong Python tag, Windows wheels, etc.)
    # Keep:
    # - cp312 wheels
    # - abi3 wheels (often compatible with cp312)
    # - py3-none-any wheels
    # - sdists (.tar.gz/.zip)
    $keepPatterns = @(
        '*cp312*',
        '*abi3*',
        '*py3-none-any*',
        '*.tar.gz',
        '*.zip'
    )

    Get-ChildItem $CacheDir -File | ForEach-Object {
        $name = $_.Name
        $keep = $false
        foreach ($pat in $keepPatterns) {
            if ($name -like $pat) { $keep = $true; break }
        }

        # Explicitly drop cp310 torch stack + any Windows wheels
        if ($name -like '*cp310*' -or $name -like '*win_amd64*') { $keep = $false }

        if (-not $keep) {
            Remove-Item -Force $_.FullName
        }
    }
}

Write-Host "Downloading torch stack (Linux/$platform, cp312, cu128) into $CacheDir" 
python -m pip download --dest $CacheDir \
  --platform $platform --python-version $pythonVersion --implementation $implementation --abi $abi \
  --only-binary=:all: --extra-index-url $torchIndex \
  -c constraints-cu128-py312.txt \
  torch==2.7.0+cu128 torchvision==0.22.0+cu128 torchaudio==2.7.0+cu128

Write-Host "Downloading project requirements (Linux/$platform, cp312) into $CacheDir" 
# Allow source distributions if a wheel does not exist; Dockerfile installs from cache only.
python -m pip download --dest $CacheDir \
  --platform $platform --python-version $pythonVersion --implementation $implementation --abi $abi \
  --extra-index-url $torchIndex \
  -c constraints-cu128-py312.txt \
  -r requirements.txt

Write-Host "Cache contents:" 
Get-ChildItem $CacheDir -File | Sort-Object Length -Descending | Format-Table Name, @{Label="Size(MB)";Expression={[math]::Round($_.Length/1MB,2)}}
