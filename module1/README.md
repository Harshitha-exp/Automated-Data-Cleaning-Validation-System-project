# Module 1: Data Profiling & Metadata Intelligence

Output: `profiling_report.json` (JSON Schema: `python -m profiler.schemas`).

| Person | Owns | Files |
|---|---|---|
| 1 | Metadata engine | `profiler/metadata.py` |
| 2 | Profiling + charts | `profiler/stats.py`, `profiler/viz.py` |
| 3 | Rule engine + PII | `profiler/rules.py`, `profiler/rules_config.yaml` |
| 4 | Schema, API, docs, CI | `profiler/schemas.py`, `report.py`, `io.py`, `api/`, `docs/` |

## Run
```
pip install -r requirements.txt
pytest
python -c "from profiler.report import profile_file; profile_file('data.csv')"
uvicorn api.main:app --reload     # Swagger UI at /docs
```

## Rules
- Do not edit `schemas.py` without all four people approving.
- Never put raw PII values in the report.
- Every `TODO(Pn)` marks work assigned to that person.
- Colab: see `notebooks/setup_cell.py`.
