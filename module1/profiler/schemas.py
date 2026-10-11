"""
schemas.py  --  OWNER: Person 4.  SINGLE SOURCE OF TRUTH for profiling_report.json.

FREEZE THIS EARLY. Change only via pull request approved by all four people.
Export JSON Schema with:  python -m profiler.schemas
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Literal, Optional

from pydantic import BaseModel, Field

SCHEMA_VERSION = "1.0.0"

InferredType = Literal[
    "integer", "float", "boolean", "datetime", "categorical", "text", "empty", "unknown"
]
SemanticType = Literal[
    "name", "email", "phone", "date", "id", "url", "currency", "zipcode",
    "ip_address", "country", "gender", "age", "address", "none",
]
Severity = Literal["info", "low", "medium", "high", "critical"]
FlagCategory = Literal["suspicious", "format_inconsistency", "pii"]


# ---------- Person 1: metadata ----------
class ColumnMetadata(BaseModel):
    name: str
    position: int
    pandas_dtype: str
    inferred_type: InferredType
    semantic_type: SemanticType = "none"
    semantic_confidence: float = Field(0.0, ge=0, le=1)
    is_mixed_type: bool = False
    # share of non-null values parseable as each type, e.g. {"integer": 0.9, "text": 0.1}
    type_distribution: dict[str, float] = Field(default_factory=dict)


class Metadata(BaseModel):
    n_rows: int
    n_cols: int
    memory_bytes: int
    columns: list[ColumnMetadata]


# ---------- Person 2: statistics ----------
class MissingColumn(BaseModel):
    column: str
    missing_count: int
    missing_pct: float


class MissingSummary(BaseModel):
    total_missing_cells: int
    total_missing_pct: float
    columns: list[MissingColumn]
    # row-pattern matrix is heavy; store only the most common patterns
    top_patterns: list[dict] = Field(default_factory=list)


class CardinalityInfo(BaseModel):
    column: str
    n_unique: int
    unique_ratio: float
    is_constant: bool
    is_high_cardinality: bool
    # top values are masked/omitted for PII columns (see rules.py)
    top_values: Optional[dict[str, int]] = None


class TypeConsistency(BaseModel):
    column: str
    expected_type: str
    consistent_pct: float
    inconsistent_examples: list[str] = Field(default_factory=list)  # masked, max 5


class NumericDistribution(BaseModel):
    column: str
    mean: Optional[float] = None
    std: Optional[float] = None
    min: Optional[float] = None
    q1: Optional[float] = None
    median: Optional[float] = None
    q3: Optional[float] = None
    max: Optional[float] = None
    skew: Optional[float] = None
    kurtosis: Optional[float] = None
    outliers_iqr: int = 0
    outliers_zscore: int = 0
    histogram: Optional[dict[str, list[float]]] = None  # {"bins": [...], "counts": [...]}


class CorrelationMatrix(BaseModel):
    method: str  # "pearson" | "spearman" | "cramers_v"
    columns: list[str]
    matrix: list[list[Optional[float]]]
    strong_pairs: list[dict] = Field(default_factory=list)  # [{"a":..,"b":..,"value":..}]


class Statistics(BaseModel):
    missing: MissingSummary
    cardinality: list[CardinalityInfo]
    type_consistency: list[TypeConsistency]
    distributions: list[NumericDistribution]
    correlations: list[CorrelationMatrix]


# ---------- Person 2: visualisations ----------
class VisualisationRef(BaseModel):
    name: Literal["missing_heatmap", "outlier_distribution", "correlation_heatmap"]
    path: Optional[str] = None          # PNG on disk
    base64_png: Optional[str] = None    # optional inline embed


# ---------- Person 3: rule engine ----------
class Flag(BaseModel):
    column: Optional[str] = None        # None = dataset-level flag
    rule_id: str                        # e.g. "SUSP_CONSTANT", "PII_EMAIL"
    category: FlagCategory
    severity: Severity
    confidence: float = Field(..., ge=0, le=1)
    message: str
    # NEVER put raw PII here. Counts / masked examples only.
    evidence: dict = Field(default_factory=dict)


# ---------- Final report ----------
class DatasetInfo(BaseModel):
    source_name: str
    n_rows: int
    n_cols: int
    sampled: bool = False
    sample_size: Optional[int] = None


class ProfilingReport(BaseModel):
    schema_version: str = SCHEMA_VERSION
    generated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    dataset: DatasetInfo
    metadata: Metadata
    statistics: Statistics
    visualisations: list[VisualisationRef]
    flags: list[Flag]
    warnings: list[str] = Field(default_factory=list)  # non-fatal issues per engine


if __name__ == "__main__":
    print(json.dumps(ProfilingReport.model_json_schema(), indent=2))
