$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
python (Join-Path $projectRoot 'stream/prepare_replay.py')
if ($LASTEXITCODE -ne 0) { throw "Replay preparation failed: $LASTEXITCODE" }
docker run --rm --network none --hostname spark-local --add-host spark-local:127.0.0.1 `
    -e SPARK_LOCAL_IP=127.0.0.1 --cpus 4 --memory 6g `
    --security-opt no-new-privileges --cap-drop ALL --tmpfs /tmp:rw,exec,size=2g `
    -v "${projectRoot}:/app" spark:3.5.9-java17-python3 `
    /opt/spark/bin/spark-submit --driver-memory 3g --conf spark.local.dir=/tmp `
    /app/stream/spark_replay.py
if ($LASTEXITCODE -ne 0) { throw "Spark replay failed: $LASTEXITCODE" }
python (Join-Path $projectRoot 'stream/evaluate_replay.py')
if ($LASTEXITCODE -ne 0) { throw "Replay audit failed: $LASTEXITCODE" }
