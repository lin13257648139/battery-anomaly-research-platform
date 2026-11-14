$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$sourceDir = (Resolve-Path (Join-Path $projectRoot 'external_data\figshare_ev')).Path
$outputDir = Join-Path $projectRoot 'converted_figshare'
New-Item -ItemType Directory -Path $outputDir -Force | Out-Null
$outputDir = (Resolve-Path $outputDir).Path
docker run --rm --network none --read-only --cap-drop ALL `
  --security-opt no-new-privileges --pids-limit 128 --memory 4g --cpus 2 `
  --user 65534:65534 --tmpfs /tmp:rw,noexec,nosuid,size=256m `
  --mount "type=bind,source=$sourceDir,target=/input,readonly" `
  --mount "type=bind,source=$outputDir,target=/output" `
  --entrypoint python gis-safe-convert:1 /app/convert_brand2.py
if ($LASTEXITCODE -ne 0) { throw 'Brand2 conversion failed.' }
