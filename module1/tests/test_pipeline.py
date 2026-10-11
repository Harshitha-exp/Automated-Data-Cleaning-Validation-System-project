import io

import pandas as pd
from fastapi.testclient import TestClient

from api.main import app
from profiler.report import generate_report


def make_df():
    return pd.DataFrame({
        "email": ["a@x.com", "b@y.org", "c@z.net", None],
        "age": [25, 31, 40, 1000],
        "mixed": ["1", "2", "three", "4"],
        "const": ["x"] * 4,
    })


def test_pipeline_runs(tmp_path):
    r = generate_report(make_df(), out_dir=tmp_path)
    cols = {c.name: c for c in r.metadata.columns}
    assert cols["email"].semantic_type == "email"
    assert cols["mixed"].is_mixed_type
    assert any(f.rule_id == "SUSP_CONSTANT" for f in r.flags)
    assert any(f.category == "pii" for f in r.flags)
    assert len(r.visualisations) == 3
    assert not r.warnings


def test_api_profile():
    csv = make_df().to_csv(index=False).encode()
    res = TestClient(app).post("/profile", files={"file": ("t.csv", io.BytesIO(csv), "text/csv")})
    assert res.status_code == 200
    assert res.json()["schema_version"] == "1.0.0"
