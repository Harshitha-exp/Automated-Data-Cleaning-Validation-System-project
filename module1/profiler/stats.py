"""stats.py -- OWNER: Person 2. Profiling engine."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .schemas import (CardinalityInfo, CorrelationMatrix, MissingColumn, MissingSummary,
                      Metadata, NumericDistribution, Statistics, TypeConsistency)


def profile_missing(df: pd.DataFrame) -> MissingSummary:
    """Missing counts/percentages per column.
    TODO(P2): top_patterns (group rows by their null-mask)."""
    miss = df.isna().sum()
    total = int(miss.sum())
    return MissingSummary(
        total_missing_cells=total,
        total_missing_pct=round(total / max(df.size, 1) * 100, 4),
        columns=[MissingColumn(column=str(c), missing_count=int(n),
                               missing_pct=round(n / max(len(df), 1) * 100, 4))
                 for c, n in miss.items()],
    )


def profile_cardinality(df: pd.DataFrame, high_card_ratio: float = 0.9) -> list[CardinalityInfo]:
    """TODO(P2): top_values for low-cardinality columns (skip PII columns)."""
    out = []
    for c in df.columns:
        n = int(df[c].nunique(dropna=True))
        ratio = n / max(int(df[c].notna().sum()), 1)
        out.append(CardinalityInfo(column=str(c), n_unique=n, unique_ratio=round(ratio, 4),
                                   is_constant=n <= 1, is_high_cardinality=ratio >= high_card_ratio))
    return out


def check_type_consistency(df: pd.DataFrame, metadata: Metadata) -> list[TypeConsistency]:
    """Compare each value to the column's inferred type; report % consistent + masked examples.
    TODO(P2)."""
    return []


def profile_distributions(df: pd.DataFrame) -> list[NumericDistribution]:
    """Numeric summary, skew/kurtosis, IQR + z-score outlier counts, histogram bins."""
    out = []
    for c in df.select_dtypes(include="number").columns:
        s = df[c].dropna()
        if s.empty:
            continue
        q1, med, q3 = s.quantile([.25, .5, .75])
        iqr = q3 - q1
        iqr_out = int(((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum())
        z_out = int((np.abs((s - s.mean()) / (s.std(ddof=0) or 1)) > 3).sum())
        counts, bins = np.histogram(s, bins=20)
        out.append(NumericDistribution(
            column=str(c), mean=float(s.mean()), std=None if pd.isna(s.std()) else float(s.std()),
            min=float(s.min()), q1=float(q1), median=float(med), q3=float(q3), max=float(s.max()),
            skew=None if pd.isna(s.skew()) else float(s.skew()),
            kurtosis=None if pd.isna(s.kurt()) else float(s.kurt()),
            outliers_iqr=iqr_out, outliers_zscore=z_out,
            histogram={"bins": bins.tolist(), "counts": counts.astype(float).tolist()}))
    return out


def compute_correlations(df: pd.DataFrame, threshold: float = 0.7) -> list[CorrelationMatrix]:
    """Pearson for numeric. TODO(P2): Cramer's V (cat-cat), correlation ratio (num-cat)."""
    num = df.select_dtypes(include="number")
    if num.shape[1] < 2:
        return []
    corr = num.corr(method="pearson")
    cols = [str(c) for c in corr.columns]
    pairs = [{"a": cols[i], "b": cols[j], "value": round(float(corr.iloc[i, j]), 4)}
             for i in range(len(cols)) for j in range(i + 1, len(cols))
             if abs(corr.iloc[i, j]) >= threshold]
    matrix = [[None if pd.isna(v) else round(float(v), 4) for v in row] for row in corr.values]
    return [CorrelationMatrix(method="pearson", columns=cols, matrix=matrix, strong_pairs=pairs)]


def run_statistics(df: pd.DataFrame, metadata: Metadata) -> Statistics:
    return Statistics(
        missing=profile_missing(df),
        cardinality=profile_cardinality(df),
        type_consistency=check_type_consistency(df, metadata),
        distributions=profile_distributions(df),
        correlations=compute_correlations(df),
    )
