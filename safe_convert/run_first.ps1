$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$sourceDir = (Resolve-Path (Join-Path $projectRoot 'GIS\GIS')).Path
$outputDir = Join-Path $projectRoot 'converted'
New-Item -ItemType Directory -Path $outputDir -Force | Out-Null
$outputDir = (Resolve-Path $outputDir).Path
docker info --format '{{.ServerVersion}}' | Out-Null
if ($LASTEXITCODE -ne 0) { throw 'Docker Desktop engine is not running.' }
docker run --rm --network none --read-only --cap-drop ALL `
  --security-opt no-new-privileges --pids-limit 128 --memory 4g --cpus 2 `
  --user 65534:65534 --tmpfs /tmp:rw,noexec,nosuid,size=256m `
  --mount "type=bind,source=$sourceDir,target=/input,readonly" `
  --mount "type=bind,source=$outputDir,target=/output" `
  gis-safe-convert:1 --folder 0
if ($LASTEXITCODE -ne 0) { throw 'Conversion failed; check the container output.' }
