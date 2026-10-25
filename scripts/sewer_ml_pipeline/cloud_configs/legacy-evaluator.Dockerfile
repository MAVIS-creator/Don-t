FROM python:3.11-slim
WORKDIR /workspace
RUN python -m pip install --no-cache-dir numpy==1.26.4 pillow==12.0.0 \
    onnxruntime==1.20.1 google-cloud-storage==3.4.1
COPY scripts/sewer_ml_pipeline/ /workspace/scripts/sewer_ml_pipeline/
ENV PYTHONUNBUFFERED=1
ENTRYPOINT ["python", "scripts/sewer_ml_pipeline/cloud_legacy_evaluate.py"]
