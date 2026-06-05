# PCIL Job Orchestrator — container image
# ----------------------------------------
# Build (from PCIL/ or PCIL_dev/):
#     docker build -t pcil-orchestrator .
#
# Run (mount your data/ folder + pass the Gemini key via --env-file):
#     docker run --rm -p 8000:8000 \
#         --env-file .env \
#         -v "$(pwd)/../data:/app/data" \
#         pcil-orchestrator
#
# `.env` must define GEMINI_API_KEY (see .env.example for the full list
# of accepted vars). Without it, /pipeline/run still returns the
# impacts JSON but operator_recommendation degrades to a fallback
# string instead of an LLM-generated paragraph.
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
# invalidate the heavy install cache. Install the CPU-only torch wheel
# first: the default PyPI torch wheel bundles CUDA and is ~GB, which the
# NUC does not need. The subsequent requirements install then sees
# torch>=2.2 already satisfied and skips it.
COPY requirements.txt ./
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

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
#   /app/data/RAG/*.docx                  — recovery docs for RAG retrieval
VOLUME ["/app/data"]

EXPOSE 8000

# Single worker is fine for the 12 June test scope. Add --workers 2+ later
# if Winardi's tests hammer the API.
CMD ["uvicorn", "pcil.orchestrator:app", "--host", "0.0.0.0", "--port", "8000"]
