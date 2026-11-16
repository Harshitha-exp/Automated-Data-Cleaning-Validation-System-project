# Statistical Imputation and Quality Scoring (Module 2, Person 2)

Fills missing values with simple statistics (the baseline the ML imputers have to beat)
and scores data quality before and after cleaning.

## Files
- impute_stats.py: impute_statistical(df, schema, config, columns) -> (cleaned_df, log)
  evaluate_statistical(...) scores the baseline by hiding known values and filling them
  mask_known_values(...) hides known values the same way for every imputer, for fair comparison
- quality_score.py: score_quality(df), quality_delta(before, after), score_before_after(raw, cleaned)
- ds_config.yaml / ds_config.py: all settings, checked when the file is loaded
- ds_schema_stub.py: stand-in schema until Person 1's schema inference exists
- test/: unit tests (see TESTING.md)
- checker/check.ipynb: demo notebook on the RMA data

## Setup and tests
    pip install pandas numpy pyyaml pytest
    pytest

## Use
    from ds_schema_stub import basic_schema
    import impute_stats as ist, quality_score as qs

    schema = basic_schema(df, overrides={
        "State": {"semantic": "id"}, "County": {"semantic": "id"},
        "Payment indemnity (US$)": {"min": 0},
    })
    config = {"imputation": {
        "group_by": {"County name": ["State", "County"]},   # fill from rows with the same State and County
        "group_fallback": "constant",                       # a group with no known value gets the constant
        "categorical_constant": "Unknown county",
    }}
    cleaned, log = ist.impute_statistical(df, schema, config)
    result = qs.score_before_after(df, cleaned, schema, config)
    print(result["delta"])

After editing a .py file, restart the notebook kernel. A notebook keeps the old version in memory.

## Behaviour
- Never changes the input. Returns a cleaned copy and a log (a list of dicts for cleaning_log.json).
- Numbers: median or mean. Categories: most frequent value or a constant.
- Group-based fill copies from similar rows (group_by). When a group has no known value, group_fallback
  chooses the overall most frequent value (mode) or the constant. n_fallback in the log counts these.
- Skipped and logged: ID and personal-data columns, dates, free text, columns over 60% missing.
- Quality score 0 to 100 from completeness, uniqueness, validity and consistency. Weights are in ds_config.yaml.
- Missing values use the same null_markers as the rule engine (N/A, NA, null, ...).

## Settings (ds_config.yaml)
imputation: max_missing_share, numeric_strategy (mean or median), categorical_strategy (mode or constant),
categorical_constant, group_fallback (mode or constant), boolean_strategy, impute_text, skip_semantics,
skip_columns, group_by.
quality: weights, sample_size, duplicate_subset, categorical_max_ratio.
A misspelt setting or a bare no/yes in a word list gives a clear error. In YAML, put quotes around no, yes, on, off, y, n.

## Results on RMA (first 100,000 rows)
- 375 missing County names, all County code 999 (a placeholder for unknown counties), filled with "Unknown county".
- Quality score 99.88 to 99.89. RMA is already clean, so the score barely moves.
- Statistical baseline: error about the size of each money column's average, so ML imputers should beat it.
  Details in TESTING.md.

## Assumptions to confirm with the team
- Schema format (type, semantic, min, max, allowed_values, format_regex), owned by Person 1.
- Log entry format (step, column, action, details), owned by whoever writes cleaning_log.json.

## Limits
- Completeness rises after imputation by construction. Read it together with validity and consistency.
- Validity only scores columns that have something to check (a min or max, allowed values, a format,
  or text declared as a number or date).
- Validity and consistency use a random sample of 50,000 rows.
- Statistical fills ignore relationships between columns.