FROM python:3.11-slim

WORKDIR /app

# System deps: matplotlib needs a minimal set of fonts/libs even headless.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/

EXPOSE 8000

# PYTHONPATH so `api.py`'s flat imports (`from orchestrator import ...`)
# and the person1/2 packages resolve the same way they do in the test
# suite and the local `uvicorn --app-dir src` invocation.
ENV PYTHONPATH=/app/src

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000", "--app-dir", "src"]
