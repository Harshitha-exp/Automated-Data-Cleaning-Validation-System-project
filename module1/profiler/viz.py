"""viz.py -- OWNER: Person 2. Backend-generated charts (no display needed)."""
from __future__ import annotations

import base64
import io
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # must be set before pyplot import
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .schemas import VisualisationRef


def _finish(fig, name: str, out_dir: Path, embed: bool) -> VisualisationRef:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    b64 = None
    if embed:
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
        b64 = base64.b64encode(buf.getvalue()).decode()
    plt.close(fig)
    return VisualisationRef(name=name, path=str(path), base64_png=b64)


def plot_missing_heatmap(df: pd.DataFrame, out_dir: Path, embed: bool = False) -> VisualisationRef:
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(df.isna().iloc[:500], cbar=False, yticklabels=False, ax=ax)
    ax.set_title("Missing values (light = missing, first 500 rows)")
    return _finish(fig, "missing_heatmap", out_dir, embed)


def plot_outlier_distribution(df: pd.DataFrame, out_dir: Path, embed: bool = False) -> VisualisationRef:
    num = df.select_dtypes(include="number")
    fig, ax = plt.subplots(figsize=(10, 5))
    if not num.empty:
        std = num.std().replace(0, 1)
        sns.boxplot(data=(num - num.mean()) / std, ax=ax)  # standardised to share an axis
        ax.tick_params(axis="x", rotation=45)
    ax.set_title("Outlier distribution (standardised numeric columns)")
    return _finish(fig, "outlier_distribution", out_dir, embed)


def plot_correlation_heatmap(df: pd.DataFrame, out_dir: Path, embed: bool = False) -> VisualisationRef:
    num = df.select_dtypes(include="number")
    fig, ax = plt.subplots(figsize=(8, 6))
    if num.shape[1] >= 2:
        sns.heatmap(num.corr(), annot=num.shape[1] <= 12, cmap="coolwarm", vmin=-1, vmax=1, ax=ax)
    ax.set_title("Correlation heatmap")
    return _finish(fig, "correlation_heatmap", out_dir, embed)


def generate_all(df: pd.DataFrame, out_dir: str | Path, embed: bool = False) -> list[VisualisationRef]:
    out_dir = Path(out_dir)
    return [plot_missing_heatmap(df, out_dir, embed),
            plot_outlier_distribution(df, out_dir, embed),
            plot_correlation_heatmap(df, out_dir, embed)]
