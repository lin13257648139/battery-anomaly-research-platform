FROM public.ecr.aws/docker/library/python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
RUN pip install --no-cache-dir --no-deps 'torch==2.6.0+cpu' --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir --default-timeout 120 --retries 5 \
    --index-url https://pypi.tuna.tsinghua.edu.cn/simple \
    filelock typing-extensions networkx jinja2 fsspec sympy==1.13.1 \
    numpy==2.1.3 pyarrow==19.0.1
COPY safe_convert/convert.py /app/convert.py
COPY safe_convert/inspect_figshare.py /app/inspect_figshare.py
COPY safe_convert/convert_figshare.py /app/convert_figshare.py
COPY safe_convert/inspect_brand2.py /app/inspect_brand2.py
COPY safe_convert/convert_brand2.py /app/convert_brand2.py
WORKDIR /app
ENTRYPOINT ["python", "/app/convert.py"]
