"""api/main.py -- OWNER: Person 4.  Run: uvicorn api.main:app --reload"""
import os
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile

from profiler.io import load_dataset
from profiler.report import generate_report
from profiler.schemas import ProfilingReport

app = FastAPI(title="Data Profiling API", version="1.0.0")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/profile", response_model=ProfilingReport)
async def profile(file: UploadFile = File(...)):
    suffix = Path(file.filename or "upload.csv").suffix
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name
    try:
        df = load_dataset(tmp_path)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Could not read file: {e}")
    finally:
        os.unlink(tmp_path)
    return generate_report(df, source_name=file.filename or "upload", out_dir=tempfile.mkdtemp())
