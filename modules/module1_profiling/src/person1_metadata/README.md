# Person 1 — Metadata Extraction Engine

Column type inference, semantic detection, and mixed-type detection for
Module 1. Produces the `metadata` structure the rest of the team consumes
directly — most importantly, it's a drop-in match for Person 3's rule
engine with **zero adapter code**, proven by an automated test that
imports and runs their actual `engine.py` (see "Interface with Person 3"
below).

Tested against the real RMA Cause-of-Loss dataset (1,048,575 rows × 12
columns): `build_metadata()` runs in ~2 seconds.

## Files

```
type_inference.py    # infer_types(df)        — structural type per column
semantics.py          # detect_semantics(df)    — what the column MEANS
mixed_types.py         # detect_mixed_types(df)  — dirty/inconsistent-type columns
metadata.py            # build_metadata(df)      — orchestrator; the one function everyone else should call
tests/
  test_metadata.py                 # 15 unit tests
  test_integration_with_person3.py  # calls Person 3's REAL engine.run_rules()
```

## Install & run

```bash
pip install pandas numpy pytest
python metadata.py data/rma_cause_of_loss.csv --out metadata.json
pytest tests/ -v
```

To run the Person-3 integration test, point `PERSON3_PATH` at their repo
(it's skipped automatically if not found, so the rest of the suite still
runs without their code present):

```bash
PERSON3_PATH=/path/to/profiling_rules_to_send pytest tests/test_integration_with_person3.py -v
```

## Key functions (as specified in the task breakdown)

```python
from type_inference import infer_types
from semantics import detect_semantics
from mixed_types import detect_mixed_types
from metadata import build_metadata

types = infer_types(df)              # {col: {"dtype", "type", "n_unique", "n_missing", "cardinality_ratio"}}
semantics = detect_semantics(df)      # {col: {"semantic", "confidence", "basis"}}
mixed = detect_mixed_types(df)        # {col: {"is_mixed_type", "type_breakdown"}}

metadata = build_metadata(df)         # combines all three — this is what you hand to everyone else
```

### `infer_types(df) -> dict`
Structural type per column: `integer | float | boolean | datetime |
categorical | text | unknown`. Categorical vs. text is decided by
cardinality (few enough distinct values in absolute terms, OR a low
enough ratio on a larger column — see the threshold comment in
`type_inference.py` for why it's two rules, not one).

### `detect_semantics(df, types=None) -> dict`
What a column means: `id | email | phone | date | url | currency | zip |
address | person_name | national_id | credit_card | ip_address |
categorical | numeric | boolean | None`. Covers the brief's required list
(name, email, phone, date, ID, URL, currency, zip) plus what falls out
naturally from the same approach. `None` means "no confident signal
either way" — deliberately distinct from "text"/"unknown", so downstream
code can tell "we looked and found nothing" apart from "we didn't look".

Detection combines column-name tokens (whole-word, not substring — see
below) with content sampling (regex against up to 500 sampled values).

### `detect_mixed_types(df) -> dict`
Flags object/string-dtype columns where sampled values don't agree on
one underlying type (e.g. 70% integers-as-text, 30% free text) — a
dirty-data signal for Module 2, not a "what does this mean" question.

### `build_metadata(df) -> dict`
```jsonc
{
  "schema_version": "1.0.0",
  "n_rows": 1048575,
  "n_columns": 12,
  "columns": {
    "Subsidy (US$)": {
      "dtype": "float64", "type": "float",
      "semantic": "currency", "semantic_confidence": 0.75, "semantic_basis": ["column_name"],
      "is_mixed_type": false, "type_breakdown": {},
      "n_unique": 487213, "n_missing": 0, "cardinality_ratio": 0.4646
    },
    ...
  }
}
```

## Interface with Person 3 — answered

> "my rules read your labels as `metadata["columns"][col]["semantic"]`
> (for example `"id"`, `"email"`, `"phone"`). Is that the shape you
> output?"

**Yes, exactly that shape**, confirmed by reading your actual
`column_rules.get_semantic()` / `pii.py` / `engine.py` and building to
match, not guessing from the standup description. Specifically:

- `metadata["columns"][col]["semantic"]` is a plain lowercase string or
  `None` — matches `get_semantic()`'s `.lower()` call exactly.
- We emit the literal string `"id"` for identifier columns, which is
  what `check_near_unique` and `check_skew` key off to skip ID columns.
- We emit `"email"`, `"phone"`, `"person_name"`, `"national_id"`,
  `"credit_card"`, `"ip_address"`, `"address"` — all recognised by your
  `pii.py`'s `SEMANTIC_TO_TYPE` map as the bonus 4th signal.
- `tests/test_integration_with_person3.py` imports your real `engine.py`
  (not a mock) and calls `run_rules(df, metadata)` with our output —
  including a test that specifically proves `semantic == "id"` suppresses
  `near_unique`/`extreme_skew` false positives on an identifier column.
  Both pass. We also ran the full 1,048,575-row RMA file through both
  modules back-to-back: 0 errors, 4 flags (all genuine — extreme skew on
  the four dollar-amount columns, which is real and expected for payout
  data).

**One known, minor gap, not a break:** we emit `"zip"` as its own
semantic (more precise, and the brief asks for it by name), but your
`SEMANTIC_TO_TYPE` map only has `"address"`, not `"zip"` — so a
zip-only column won't get the PII bonus "semantic label" signal from us
specifically. It'll still very likely get flagged via your own
column-name + regex signals independently. A one-line addition
(`"zip": "address"`) on your side would close this if exact parity
matters; happy to do it myself if you'd rather hand it over.

## Design notes / bugs found and fixed while testing against real data

Built and fixed iteratively against the actual RMA dataset, not just
synthetic fixtures — three real issues only showed up at that point:

1. **pandas ≥ 3.0 changed the default dtype for string columns** from
   `object` to a dedicated `str` dtype. Every `dtype == object` check
   silently stopped matching any text column. Fixed by checking
   `is_object_dtype OR is_string_dtype` everywhere.
2. **Column-name matching uses whole-word tokens, never substrings.**
   `"id" in "subsidy"` is `True` (sub-s-**id**-y) — a classic false
   positive. All hint-matching here splits the column name into tokens
   first (`_tokenize`) and checks set membership instead.
3. **Geographic codes vs. continuous measures vs. vague money words.**
   `State`/`County` are low-cardinality integer FIPS codes — they needed
   a dedicated geo-code name check, since the usual "high cardinality
   ratio ⇒ id" heuristic only catches the opposite case. Separately,
   `"Payment acreage"` was getting flagged `currency` purely because it
   contains the word "payment" — currency detection was narrowed to
   require an explicit unit token (`usd`, `us`, `gbp`, ...), since by the
   time a column is numeric dtype any `$` formatting has already been
   stripped by CSV/Excel parsing, so the column name is the only
   remaining signal and vague money-adjacent words aren't reliable
   enough on their own.

All three are covered by regression tests in `tests/test_metadata.py` so
they can't silently regress.

## Known limitations

- Semantic detection is heuristic (name tokens + regex content sampling
  on up to 500 values), not a certification — it's meant to triage for a
  human or for Module 2/3, the same spirit as Person 3's rule engine.
- `detect_mixed_types` samples up to 2,000 non-null values per column for
  speed on multi-million-row columns rather than scanning every row.
