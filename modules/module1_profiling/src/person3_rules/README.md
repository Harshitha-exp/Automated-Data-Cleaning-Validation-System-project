# Profiling Rule Engine (Person 3)

Setup: pip install -r requirements.txt
       python -m spacy download en_core_web_sm
Tests: pytest        (40 tests)

Use:   import engine
       result = engine.run_rules(df, metadata)   # metadata is optional
       # {"flags": [...], "total_flags": n, "errors": [...]}

Each flag: column, rule, severity, confidence, evidence, suggestion.
Also available: pii.detect_pii(df), formats.detect_format_inconsistency(df).
Settings (thresholds, on/off switches): rules.yaml
metadata shape expected: metadata["columns"][col]["semantic"]