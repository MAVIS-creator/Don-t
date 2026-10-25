FROM python:3.11-slim
WORKDIR /workspace
RUN python -m pip install --no-cache-dir google-cloud-storage==3.4.1 requests==2.32.5 pillow==12.0.0
COPY scripts/sewer_ml_pipeline/ /workspace/scripts/sewer_ml_pipeline/
ENV PYTHONUNBUFFERED=1
ENTRYPOINT ["python", "scripts/sewer_ml_pipeline/cloud_select_250k_manifest.py"]
