Param(
    [string]$CacheDir = ".cache"
)

$ErrorActionPreference = "Stop"

New-Item -ItemType Directory -Force -Path $CacheDir | Out-Null

$baseUrl = "https://download.pytorch.org/whl/cu128"

$files = @(
    "torch-2.7.0%2Bcu128-cp312-cp312-manylinux_2_28_x86_64.whl",
    "torchvision-0.22.0%2Bcu128-cp312-cp312-manylinux_2_28_x86_64.whl",
    "torchaudio-2.7.0%2Bcu128-cp312-cp312-manylinux_2_28_x86_64.whl"
)

foreach ($file in $files) {
    $url = "$baseUrl/$file"
    $outName = [System.Uri]::UnescapeDataString($file)
    $outPath = Join-Path $CacheDir $outName

    if (Test-Path $outPath) {
        Write-Host "Already present: $outName"
        continue
    }

    Write-Host "Downloading: $url"
    Invoke-WebRequest -Uri $url -OutFile $outPath
    Write-Host "Saved: $outPath"
}

Write-Host "Done. Cached wheels:" 
Get-ChildItem $CacheDir -File | Sort-Object Length -Descending | Format-Table Name, @{Label="Size(MB)";Expression={[math]::Round($_.Length/1MB,2)}}
