# Automated Data Cleaning & Validation System

CadetX project: a backend "data gatekeeper" that profiles, cleans and validates
a dataset before it is used.

```
dataset -> Module 1: Profile -> profiling_report.json
        -> Module 2: Clean   -> cleaned_data.csv + cleaning_log.json
        -> Module 3: Validate -> validation_report.json
        -> Module 4: Automate -> one CLI / Docker pipeline running all of the above
```

## Repo layout

| Path | What lives there | Status |
|---|---|---|
| `modules/module1_profiling/` | Metadata extraction, profiling stats + charts, rule engine, FastAPI | In place, 76 tests |
| `modules/module2_cleaning/` | Schema inference, imputation, normalisation, quality scoring | Not started |
| `modules/module3_validation/` | Rule validation, anomaly detection, NLP column classifier | Not started |
| `modules/module4_automation/` | Orchestrator, YAML config, logging, CLI, DVC | Not started |
| `schemas/` | JSON Schemas for each module's output (the contracts between modules) | Module 1 done |
| `sample_data/` | Small demo CSVs only | Has a 2,000-row RMA sample |
| `docs/` | Architecture, API usage, per-module presentations | Empty |
| `tests/` | Cross-module integration tests | Empty |
| `.github/` | CI, CODEOWNERS, pull request template | CI covers Module 1 |

## Quick start (Module 1)

```bash
pip install -r requirements.txt
cd modules/module1_profiling
pytest tests/ -v
python src/orchestrator.py ../../sample_data/sample_rma_cause_of_loss.csv --out report.json
```

See `modules/module1_profiling/README.md` for details and `CONTRIBUTING.md`
for how we branch, review and merge.
