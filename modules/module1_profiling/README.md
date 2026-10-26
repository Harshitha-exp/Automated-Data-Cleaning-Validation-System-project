# Module 1 — Advanced Data Profiling & Metadata Intelligence

CadetX "Automated Data Cleaning & Validation System" — Module 1.

Profiles a dataset before any cleaning or ML happens: infers column types
and meaning, computes statistics and charts, flags suspicious/PII
columns, and returns one combined `profiling_report.json`.

```
dataset  ──▶  Person 1: metadata        (types, semantics, mixed-type flags)
                 │
                 ▼
             Person 2: statistics        (missing/consistency/cardinality/
             + visualisations             distributions/correlations + 3 charts)
                 │
                 ▼
             Person 3: rule engine       (suspicious columns, bad formats, PII)
                 │
                 ▼
             Person 4: orchestrator  ──▶  profiling_report.json  /  FastAPI
```

Each stage is wrapped so one module's failure doesn't sink the whole
report — a partial result with a populated `"errors"` list beats a 500.

## Repo layout

```
src/
  person1_metadata/     infer_types(df), detect_semantics(df), detect_mixed_types(df)
  person2_profiling/    stats.py (profile_missing, type_consistency_check,
                         compute_cardinality, compute_distributions, compute_correlations)
                         charts.py (plot_missing_heatmap, plot_outlier_distribution,
                         plot_correlation_heatmap)
  person3_rules/        engine.py, column_rules.py, pii.py, formats.py, rules.yaml
  orchestrator.py        generate_report(df) -> profiling_report.json  (Person 4)
  api.py                  FastAPI app: POST /profile, GET /health       (Person 4)
tests/
  person1/, person3/     each person's own test suite, run unmodified
  test_person2_profiling.py
  test_orchestrator_integration.py   full pipeline, real-shaped data, schema validation
  test_api.py
Dockerfile
```

Shared files live at the repo root, not in this folder:
`../../schemas/profiling_report.schema.json` (the output contract, read by
other modules too), `../../sample_data/` (2,000-row demo CSV) and
`../../.github/workflows/ci.yml` (CI for every module).

## Install & run

```bash
# run everything below from modules/module1_profiling/
pip install -r requirements.txt

# CLI — profile a file directly
python src/orchestrator.py ../../sample_data/sample_rma_cause_of_loss.csv --out report.json

# API — local
uvicorn api:app --app-dir src --reload
curl -F "file=@../../sample_data/sample_rma_cause_of_loss.csv" http://localhost:8000/profile

# Docker
docker build -t module1-profiling .
docker run -p 8000:8000 module1-profiling
```

### Run the tests

```bash
pytest tests/ -v
```

76 tests, all passing: 15 of Person 1's own, 40 of Person 3's own (both
run completely unmodified against their real code), 12 for Person 2's
statistics/charts, 4 for the API, and 5 full-pipeline integration tests
— including one that validates a real generated report against
`schemas/profiling_report.schema.json` (repo root).

Verified end-to-end against the real 1,048,575-row USDA RMA Cause-of-Loss
dataset: **0 errors, 4 genuine flags** (extreme skew on the four
dollar-amount columns — expected for payout data), full report (including
all three charts) generated in ~18 seconds.

## Each person's piece

### Person 1 — Metadata Extraction Engine
`infer_types(df)`, `detect_semantics(df)`, `detect_mixed_types(df)`,
combined by `build_metadata(df)`. Structural type + semantic meaning
(id/email/phone/date/currency/zip/...) + mixed-type detection per column.
This is their real, tested code — see `src/person1_metadata/` and
`tests/person1/`.

### Person 2 — Profiling Engine & Visualisations
**Note:** no reusable functions had been submitted for this role at the
time this repo was assembled — only an auto-generated ydata-profiling
HTML/JSON report, which bundles statistics and charts together
internally and doesn't expose the individual functions the brief names.
`src/person2_profiling/` is a from-scratch implementation written to
match the brief's exact function list, so it's a placeholder for the
real Person 2 to replace — every function keeps the signature the
orchestrator expects, so swapping it in later needs no other changes:

- `profile_missing(df)`, `type_consistency_check(df, metadata=None)`,
  `compute_cardinality(df)`, `compute_distributions(df)`,
  `compute_correlations(df)` (Pearson for numeric, bias-corrected
  Cramér's V for categorical) → combined by `compute_statistics(df, metadata)`
- `plot_missing_heatmap(df)`, `plot_outlier_distribution(df)`,
  `plot_correlation_heatmap(correlations)` → combined by
  `generate_visualisations(df, correlations)`, each returning
  `{"format": "png", "encoding": "base64"|"path", "data": ...}`

### Person 3 — Profiling Rule Engine
`run_rules(df, metadata, config)` — suspicious columns (all-null,
constant, near-unique, leakage-named, extreme skew), inconsistent
formats, and PII detection (column name + regex + Presidio + Person 1's
semantic label as a 4th signal), fully YAML-configurable via
`rules.yaml`. This is their real, tested code, run completely unmodified
— see `src/person3_rules/` and `tests/person3/`. (Presidio needs
`en_core_web_sm` to do NLP-based PII detection; without it, PII detection
still works via name + regex signals and just warns once.)

### Person 4 — Orchestration, API & Schema (this integration layer)
`orchestrator.generate_report(df)` runs all three in order — Person 1
first, since Person 3's rules consume Person 1's metadata — and merges
the results into `profiling_report.json`. `api.py` exposes it as
`POST /profile`. `schemas/profiling_report.schema.json` (repo root) is the formal
contract for that output, validated by its own test.

## Interface notes (so nobody has to guess the contract again)

- Person 3's rules read `metadata["columns"][col]["semantic"]` as a
  lowercase string or `None` — confirmed against their actual
  `column_rules.get_semantic()`, not just the standup description.
- Person 3's `engine.py`/`column_rules.py`/`pii.py`/`formats.py` use flat,
  non-relative imports (`import column_rules as cr`) — written and
  tested as a standalone script directory. Rather than editing their
  source, `orchestrator.py` puts `src/person3_rules/` directly on
  `sys.path` and imports it as a top-level module, exactly like their own
  test suite's `conftest.py` does. Nothing in `person3_rules/` was
  changed from what they submitted.
- Person 2's `type_consistency_check(df, metadata=None)` and
  `compute_statistics(df, metadata=None)` accept Person 1's metadata to
  avoid re-deriving structural types, but fall back to inferring dtype
  directly if called standalone.

## Known gaps / honest limitations

- Person 2's module is a stand-in (see above) — swap it for the real
  thing when it exists; the interface won't need to change.
- `plot_missing_heatmap` samples down to 2,000 rows on large files (the
  same practical compromise the `missingno` library makes) rather than
  rendering every row as a pixel.
- `compute_correlations` samples up to 100,000 rows on large files for
  speed; correlation strength is a population estimate and doesn't need
  every row.
- Presidio-based PII detection needs `python -m spacy download en_core_web_sm`
  installed separately; it's not pulled automatically by `requirements.txt`
  because it's a large model download, not a pip package.
