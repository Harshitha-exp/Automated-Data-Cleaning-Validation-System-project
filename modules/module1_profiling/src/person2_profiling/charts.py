"""
person2_profiling/charts.py
------------------------------
Person 2 — the "Visualisations" half: the three specific backend-rendered
charts the brief names (missing heatmap, outlier distribution, correlation
heatmap). Each function returns a dict shaped like:

    {"format": "png", "encoding": "base64", "data": "<...>"}

or, if `save_path` is given, writes the PNG to disk instead and returns
{"format": "png", "encoding": "path", "data": "<path>"} — the brief says
output can be "image paths or base64", so both are supported from the
same functions rather than picking one and forcing the caller to convert.

Matplotlib's non-interactive Agg backend is used throughout (set at
import time) since this runs on a server with no display.
"""

from __future__ import annotations

import base64
import io
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# A full million-row boolean matrix can't be drawn pixel-for-pixel — this
# caps how many rows are sampled for the missing-value heatmap, the same
# practical compromise the missingno library makes for large datasets.
MISSING_HEATMAP_MAX_ROWS = 2000
MAX_COLUMNS_IN_CHART = 40


def _fig_to_result(fig, save_path: str | None) -> dict[str, Any]:
    if save_path:
        fig.savefig(save_path, format="png", bbox_inches="tight", dpi=110)
        plt.close(fig)
        return {"format": "png", "encoding": "path", "data": save_path}

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=110)
    plt.close(fig)
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return {"format": "png", "encoding": "base64", "data": encoded}


def plot_missing_heatmap(df: pd.DataFrame, save_path: str | None = None) -> dict[str, Any]:
    """
    Render the missing-value matrix as a picture: one row per sampled
    record, one column per field, dark where a value is present, light
    where it's missing. On large files, rows are evenly sampled down to
    MISSING_HEATMAP_MAX_ROWS so the image stays legible and fast to draw
    (this mirrors stats.profile_missing's numeric summary, but as a
    picture rather than a table).
    """
    columns = list(df.columns)[:MAX_COLUMNS_IN_CHART]
    frame = df[columns]

    if len(frame) > MISSING_HEATMAP_MAX_ROWS:
        idx = np.linspace(0, len(frame) - 1, MISSING_HEATMAP_MAX_ROWS).astype(int)
        frame = frame.iloc[idx]

    matrix = frame.notnull().to_numpy().astype(float)  # 1 = present, 0 = missing

    fig_height = max(4, min(10, len(frame) / 200))
    fig, ax = plt.subplots(figsize=(max(6, len(columns) * 0.4), fig_height))
    ax.imshow(matrix, aspect="auto", cmap="Greys", vmin=0, vmax=1, interpolation="nearest")
    ax.set_xticks(range(len(columns)))
    ax.set_xticklabels(columns, rotation=90, fontsize=7)
    ax.set_yticks([])
    ax.set_title("Missing-value matrix (dark = present, light = missing)")
    fig.tight_layout()
    return _fig_to_result(fig, save_path)


def plot_outlier_distribution(df: pd.DataFrame, columns: list[str] | None = None, save_path: str | None = None) -> dict[str, Any]:
    """
    Boxplots (one per numeric column) so outliers are visible as points
    beyond the whiskers. Defaults to every numeric column, capped at
    MAX_COLUMNS_IN_CHART for legibility; pass `columns` to target specific
    ones instead.
    """
    if columns is None:
        columns = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c])]
    columns = columns[:MAX_COLUMNS_IN_CHART]

    if not columns:
        fig, ax = plt.subplots(figsize=(4, 2))
        ax.text(0.5, 0.5, "No numeric columns to plot", ha="center", va="center")
        ax.axis("off")
        return _fig_to_result(fig, save_path)

    n = len(columns)
    ncols = min(4, n)
    nrows = -(-n // ncols)  # ceil division
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.2, nrows * 3))
    axes = np.atleast_1d(axes).flatten()

    for ax, col in zip(axes, columns):
        data = df[col].dropna()
        ax.boxplot(data, orientation="vertical", flierprops={"markersize": 2, "markerfacecolor": "red", "markeredgecolor": "red"})
        ax.set_title(col, fontsize=8)
        ax.tick_params(axis="both", labelsize=6)
    for ax in axes[len(columns):]:
        ax.axis("off")

    fig.suptitle("Outlier distribution (boxplots)", fontsize=10)
    fig.tight_layout()
    return _fig_to_result(fig, save_path)


def plot_correlation_heatmap(correlations: dict[str, Any] | pd.DataFrame, kind: str = "numeric", save_path: str | None = None) -> dict[str, Any]:
    """
    Render a correlation matrix as a heatmap.

    `correlations` can be either:
      - the dict returned by stats.compute_correlations(df) (pass `kind`
        to pick "numeric" or "categorical"), or
      - a plain square pandas DataFrame correlation matrix directly.
    """
    if isinstance(correlations, pd.DataFrame):
        columns = list(correlations.columns)
        matrix = correlations.to_numpy()
    else:
        section = correlations.get(kind, {})
        columns = section.get("columns", [])
        matrix = np.array(section.get("matrix", []), dtype=float) if section.get("matrix") else np.empty((0, 0))

    if matrix.size == 0 or len(columns) < 2:
        fig, ax = plt.subplots(figsize=(4, 2))
        ax.text(0.5, 0.5, f"Not enough {kind} columns to correlate", ha="center", va="center")
        ax.axis("off")
        return _fig_to_result(fig, save_path)

    fig, ax = plt.subplots(figsize=(max(5, len(columns) * 0.6), max(4, len(columns) * 0.6)))
    im = ax.imshow(matrix, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(columns)))
    ax.set_xticklabels(columns, rotation=90, fontsize=7)
    ax.set_yticks(range(len(columns)))
    ax.set_yticklabels(columns, fontsize=7)
    for i in range(len(columns)):
        for j in range(len(columns)):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=6)
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title(f"{kind.capitalize()} correlation heatmap")
    fig.tight_layout()
    return _fig_to_result(fig, save_path)


def generate_visualisations(df: pd.DataFrame, correlations: dict[str, Any] | None = None) -> dict[str, Any]:
    """
    Combine all three backend charts into the single "visualisations"
    section of profiling_report.json, mirroring the other orchestrator
    functions (build_metadata, run_rules, compute_statistics).
    `correlations` should be stats.compute_correlations(df)'s output if
    already computed elsewhere, to avoid recomputing it.
    """
    from .stats import compute_correlations  # local import avoids a hard cycle at module load time

    corr = correlations if correlations is not None else compute_correlations(df)
    return {
        "missing_heatmap": plot_missing_heatmap(df),
        "outlier_distribution": plot_outlier_distribution(df),
        "correlation_heatmap": plot_correlation_heatmap(corr, kind="numeric"),
    }
