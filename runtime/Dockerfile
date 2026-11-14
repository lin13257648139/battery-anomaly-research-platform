FROM public.ecr.aws/docker/library/python:3.10-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
RUN pip install --no-cache-dir --default-timeout 120 --retries 5 \
    --index-url https://pypi.tuna.tsinghua.edu.cn/simple \
    fastapi==0.115.6 uvicorn==0.34.0 pandas==2.2.3 pyarrow==19.0.1
WORKDIR /app
COPY service /app/service
