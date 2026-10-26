# Person 2 — Profiling Engine & Visualisations (stand-in implementation)

No reusable Person 2 functions had been submitted when this repo was
assembled -- only an auto-generated ydata-profiling HTML/JSON report.
Everything in this folder (stats.py, charts.py) was written from scratch
to match the brief's exact function list below, so the real Person 2 can
drop in a replacement later without the orchestrator needing to change.

## Key functions

- `profile_missing(df)` -- missing-value matrix, summarised as statistics
- `type_consistency_check(df, metadata=None)` -- does each column's data
  match its declared structural type
- `compute_cardinality(df)` -- unique-value / cardinality analysis
- `compute_distributions(df)` -- per-column distribution stats
- `compute_correlations(df)` -- Pearson (numeric) + bias-corrected
  Cramér's V (categorical)
- `compute_statistics(df, metadata=None)` -- combines all five into the
  'statistics' section of profiling_report.json

- `plot_missing_heatmap(df)`
- `plot_outlier_distribution(df, columns=None)`
- `plot_correlation_heatmap(correlations, kind='numeric')`
- `generate_visualisations(df, correlations=None)` -- combines all three
  into the 'visualisations' section

Each chart function returns `{'format': 'png', 'encoding': 'base64'|'path', 'data': ...}`
-- pass `save_path=...` to get a file on disk instead of a base64 string.

See the repo root README for design notes (row sampling on large files,
Cramér's V correction, etc.) and tests/test_person2_profiling.py for
usage examples.

