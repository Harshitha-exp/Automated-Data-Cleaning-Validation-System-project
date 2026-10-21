# Rule Engine (Person 3)

Checks a dataset for suspicious columns, inconsistent formats and personal data (PII),
and returns a list of flags, each with a severity and a confidence.

## Data
The RMA data is not in the repo (too big). Download it and save it as data/rma_download.csv.
Do not open or save the CSV in Excel: it cuts files at 1,048,575 rows.

## Setup (Python 3.11)

    pip install -r requirements.txt
    python -m spacy download en_core_web_sm

The spaCy model is used by Presidio. If it is missing, the engine still runs: it prints a
warning and uses column names and regex only.

## Run the tests

    pytest

## Use

    import engine                               # in the profiler/ layout: from profiler import rules as engine
    result = engine.run_rules(df, metadata)     # metadata is optional
    # {"flags": [...], "total_flags": 3, "errors": []}

Also available on their own: `detect_pii(df)` and `detect_format_inconsistency(df)`.

Example flag:

    {"column": "email", "rule": "pii", "severity": "high", "confidence": 0.9,
     "evidence": "likely email; signals: column name, regex (100% of values match), presidio (100% of sampled values)",
     "suggestion": "mask, hash or drop this column before sharing or modelling"}

Fields: column, rule, severity (high/medium/low), confidence (0 to 1), evidence, suggestion.
Flags are sorted most serious first, then most confident first.

## What it checks

| Rule | Flags |
|---|---|
| all_null | columns where every value is missing |
| constant_column | columns with a single value |
| near_unique | non-ID columns where almost every value is different |
| leakage_names | column names such as target, label, outcome, result |
| extreme_skew | number columns dominated by a few huge values (ID-like columns are skipped) |
| format_inconsistency | mixed date formats, mixed casing, inconsistent phone patterns, odd-one-out value shapes |
| pii | likely personal data, scored from column name, regex and Presidio |

PII confidence combines the signals: two agreeing signals score higher than one, and Presidio
alone never flags a column. Place names look like people to Presidio, so a person-name flag
also needs a matching column name.

## Settings

Every threshold, weight and on/off switch is in `rules.yaml`. Change a value and run again;
no Python editing is needed. A misspelt setting, an unknown rule or a bad severity raises a
clear error when the file is loaded. In YAML, put quotes around the words no, yes, on, off,
y and n, because bare they are read as true/false.

`null_markers` lists values treated as missing before any rule runs (for example N/A).

## Input from Person 1 (optional)

    metadata["columns"][column_name]["semantic"]    # e.g. "id", "email", "phone"

The rules still work without it.

## Output notes

- Flags never contain real values: only the column name, the type of problem and abstract
  patterns such as 9999-99-99.
- "errors" lists any rule that crashed. The other rules still run.
- Missing values are not reported here (Person 2's missing-value matrix).
- near_unique and extreme_skew need correctly typed columns, so run type inference first.
- Format and PII checks look at a random sample of rows (sample_size in rules.yaml).
- NN/NN/YYYY dates count as one format: day-first and month-first cannot be told apart by shape.

## Demo notebook

notebooks/explore.ipynb runs the engine on the RMA file (no flags except skewed money columns)
and on a synthetic farmer file made with Faker (PII, mixed date format, lost leading zero and
an outlier are all caught). The synthetic file is fake and contains no real people.