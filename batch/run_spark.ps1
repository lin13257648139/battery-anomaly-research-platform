$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
python (Join-Path $projectRoot 'tools/audit_feature_data.py')
if ($LASTEXITCODE -ne 0) { throw "Feature quality gate failed: $LASTEXITCODE" }

docker run --rm --network none --hostname spark-local --add-host spark-local:127.0.0.1 `
    -e SPARK_LOCAL_IP=127.0.0.1 --cpus 4 --memory 6g `
    --security-opt no-new-privileges --cap-drop ALL --tmpfs /tmp:rw,exec,size=2g `
    -v "${projectRoot}:/app" spark:3.5.9-java17-python3 `
    /opt/spark/bin/spark-submit --driver-memory 3g --conf spark.local.dir=/tmp `
    /app/batch/spark_features.py
if ($LASTEXITCODE -ne 0) { throw "Spark batch job failed: $LASTEXITCODE" }
python (Join-Path $projectRoot 'batch/check_parity.py')
if ($LASTEXITCODE -ne 0) { throw "Spark parity check failed: $LASTEXITCODE" }
