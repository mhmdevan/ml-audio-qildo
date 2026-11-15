FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

RUN pip install --no-cache-dir numpy==2.2.6 scipy==1.15.3
ARG WITH_TORCH=1
RUN if [ "$WITH_TORCH" = "1" ]; then \
      pip install --no-cache-dir torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu; \
    fi
ENV QILDO_DEFAULT_MODEL=${WITH_TORCH}

COPY beats.py .
COPY qildo_beats/ qildo_beats/
COPY checkpoints/mixed-residual.pt checkpoints/mixed-residual.pt
COPY checkpoints/quality.pt checkpoints/quality-ranker.json checkpoints/
ENTRYPOINT ["python", "beats.py"]
