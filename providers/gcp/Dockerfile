FROM --platform=linux/amd64 python:3.12-alpine

RUN apk add --no-cache gcc musl-dev linux-headers g++ gfortran openblas-dev

WORKDIR /app

COPY providers/gcp/requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY shared/ shared/
COPY scripts/generate-report.py scripts/generate-report.py
COPY providers/gcp/app/pipeline.py pipeline.py

ENV PYTHONPATH=/app

CMD ["python", "pipeline.py"]
