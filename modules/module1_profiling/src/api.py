"""
api.py
--------
Person 4 — Profiling API.

    POST /profile   — upload a dataset, get back profiling_report.json
    GET  /health     — liveness check

Run locally:
    uvicorn api:app --reload --app-dir src
Or via Docker (see ../Dockerfile):
    docker build -t module1-profiling . && docker run -p 8000:8000 module1-profiling
"""

from __future__ import annotations

import io
import os
import tempfile
from typing import Any

import pandas as pd
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse

from orchestrator import generate_report

app = FastAPI(
    title="Module 1 — Data Profiling & Metadata Intelligence API",
    description="Profiles an uploaded dataset and returns profiling_report.json "
                "(metadata, statistics, visualisations, and rule-engine flags).",
    version="1.0.0",
)

_SUPPORTED_EXTENSIONS = {".csv", ".tsv", ".xlsx", ".xlsm", ".parquet"}


def _read_dataframe(filename: str, raw: bytes) -> pd.DataFrame:
    ext = os.path.splitext(filename)[1].lower()
    if ext not in _SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Supported: {sorted(_SUPPORTED_EXTENSIONS)}",
        )

    buf = io.BytesIO(raw)
    try:
        if ext in (".csv", ".tsv"):
            sep = "\t" if ext == ".tsv" else ","
            return pd.read_csv(buf, sep=sep, low_memory=False)
        if ext in (".xlsx", ".xlsm"):
            return pd.read_excel(buf, engine="openpyxl")
        if ext == ".parquet":
            return pd.read_parquet(buf)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Could not parse file: {exc}") from exc

    raise HTTPException(status_code=400, detail="Unreachable: unsupported extension slipped through")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/profile")
async def profile_dataset(
    file: UploadFile = File(..., description="A .csv, .tsv, .xlsx or .parquet dataset"),
    include_visualisations: bool = Query(True, description="Set false to skip chart rendering for a faster pass"),
) -> JSONResponse:
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    df = _read_dataframe(file.filename or "upload.csv", raw)
    if df.empty:
        raise HTTPException(status_code=400, detail="Dataset has no rows")

    report: dict[str, Any] = generate_report(df, include_visualisations=include_visualisations)
    return JSONResponse(content=report)
