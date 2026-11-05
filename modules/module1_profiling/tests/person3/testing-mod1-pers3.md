## Testing

Run `pytest` from the project root. Last run: 42 tests passed.

### Test files

| File | What it tests |
|---|---|
| tests/test_rules.py | the five column rules, the engine (on/off switch, null markers, sort order, flag fields) and config checks |
| tests/test_formats.py | mixed date formats, mixed casing, phone patterns (including extensions such as x123), odd-one-out shapes |
| tests/test_pii.py | name, regex and Presidio signals, confidence scores, the semantic label, fallback when Presidio is missing |

### Approach

- Each rule is tested on a tiny hand-made table where the right answer is known.
- A clean table must produce no flags (this catches false alarms).
- Edge cases: empty table, one-row table, all-null column, camelCase column names (userId).
- Config checks: a misspelt setting, an unknown rule, a bad severity, a confidence above 1,
  and a YAML boolean in a word list (bare no/yes) must each raise a clear error.
- Privacy: tests check that flags never contain real values.
- Presidio: one test deliberately uses a model that does not exist and expects a warning
  and a normal result from the other signals.
- Tests build their own small settings, so they never depend on the real rules.yaml.

### Results on real and synthetic data

| Dataset | Result |
|---|---|
| RMA, first 100,000 rows | 4 low-severity skew flags (payment acreage, indemnity, liability, subsidy), 0 errors. County, State and Year are skipped on purpose because they are codes. |
| RMA, full file (2,714,521 rows) | Not run separately by Person 3 yet. Person 4's full pipeline run on the first 1,048,575 rows gave 0 errors and the same 4 skew flags. |
| Synthetic farmer file (50 fake rows, planted errors) | 14 flags, 0 errors. Every planted error the engine is responsible for was caught: PII columns, mixed date format, lost leading zero, extreme outlier. The planted missing value is not an engine rule (Person 2). |

The synthetic file is made with Faker (see the demo notebook), so it contains no real people.
Its answer key is the table at the end of the notebook.

## Risk assessment

| # | Risk | Likelihood | Impact | Mitigation / status |
|---|---|---|---|---|
| 1 | False alarms on real data (a rule flags a column that is fine) | Medium | Low | Found and fixed on RMA: County was flagged as skewed, so ID-like columns are now skipped (skip_name_patterns). Presidio sees place names as people, so person-name flags also need a matching column name. Near-unique flags on names and emails are expected and low severity. |
| 2 | Missed problems (a rule stays quiet when it should flag) | Medium | Medium | The odd-one-out format check only fires when one shape covers at least 95% of values (dominant_share). If more than 5% of a column is wrong it stays quiet. Adjustable in rules.yaml. Missing values are not covered here (Person 2). |
| 3 | PII hidden in free text or non-English names | Medium | High | PII is found from column names, regex on values and Presidio on a sample of 50 distinct values per column, using an English model. Free-text columns and other languages can be missed. Open: test on real-looking free text. |
| 4 | Sampling misses rare problems | Low | Medium | Formats and PII use a random sample (sample_size in rules.yaml), not the first rows. Raise sample_size for stricter checks. |
| 5 | Wrong or missing column types | Medium | Medium | near_unique and extreme_skew need real number and integer columns. If everything is loaded as text they skip columns silently. Type inference (Person 1) must run first. |
| 6 | Presidio or its language model not installed | Medium | Low | The engine warns and continues with column names and regex only. Covered by a test. |
| 7 | Config mistakes (typos, YAML reads bare no as false) | Medium | Medium | The config is validated when loaded and gives a clear error. Covered by tests. |
| 8 | A rule crashes on unusual data | Low | Medium | Each rule runs separately. A crash is recorded under "errors" and the other rules still run. |
| 9 | Interface mismatch with Person 1 (semantic labels) | Medium | Medium | metadata is optional. Rules read metadata["columns"][col]["semantic"]. Confirmed: Person 1's README and integration test show the same shape. |
| 10 | Real values leaking into the report | Low | High | Flags show column names, problem types and abstract patterns such as 9999-99-99, never values. Constant text values are hidden. Covered by tests. |
| 11 | Slow on large files | Low | Medium | Formats and PII sample the data. A 200,000-row trial took about 5 seconds. Open: record the full RMA run time above. |
| 12 | Thresholds fit only RMA and the synthetic file | Medium | Medium | Thresholds are settings in rules.yaml and should be re-checked when NASS and weather data arrive (for example (D) suppressed values need adding to null_markers). |
| 13 | Imports break if files are moved or converted | Medium | Medium | The modules use flat imports and the orchestrator adds src/person3_rules/ to the path. Do not convert to relative imports. |

## Known limitations

- Dates written NN/NN/YYYY count as one format: day-first and month-first cannot be told apart by shape.
- The synthetic test file is tidier than real data, so it shows the rules work when problems exist,
  not how often they catch messy real-world cases.
- Phone checks recognise formatted numbers with 7 to 15 digits. Plain digit strings are treated as IDs.
