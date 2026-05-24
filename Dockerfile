# PCIL Job Orchestrator — container image
# ----------------------------------------
# Build (from PCIL/ or PCIL_dev/):
#     docker build -t pcil-orchestrator .
#
# Run (mount your data/ folder so the orchestrator can find trained
# anomaly bundles + the mock shop-floor CSV):
#     docker run --rm -p 8000:8000 \
#         -v "$(pwd)/../data:/app/data" \
#         pcil-orchestrator
#
# Smoke test:
#     curl http://localhost:8000/
#     curl http://localhost:8000/docs        # Swagger UI

FROM python:3.13-slim

# Runtime libs for the scientific Python stack. sklearn relies on libgomp;
# keep the rest minimal so the image stays small.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps in their own layer so source-only edits don't
# invalidate the heavy install cache.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Application code.
COPY pcil/ ./pcil/
COPY machines/ ./machines/

# In dev, orchestrator.py walks up to ITP/ via Path.parents[2]. In the
# container, /app IS the project root, so set it explicitly.
ENV PCIL_PROJECT_ROOT=/app
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Mount the host's data/ folder here at runtime.
# Contents the orchestrator looks for:
#   /app/data/mock_shop_floor.csv         — trigger.source in config.yaml
#   /app/data/non_cyclical_<id>.pkl       — Zi Hin's bundle
#   /app/data/cyclical_<id>.pkl           — Jaymon's bundle
VOLUME ["/app/data"]

EXPOSE 8000

# Single worker is fine for the 12 June test scope. Add --workers 2+ later
# if Winardi's tests hammer the API.
CMD ["uvicorn", "pcil.orchestrator:app", "--host", "0.0.0.0", "--port", "8000"]
