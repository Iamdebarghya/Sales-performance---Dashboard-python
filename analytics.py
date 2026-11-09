"""Reusable analytics functions for the Sales Performance Dashboard."""
from __future__ import annotations
import pandas as pd


def validate_dataset(data: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of a non-empty tabular dataset with unique column labels."""
    if data is None or data.empty:
        raise ValueError("The dataset is empty.")
    if data.columns.duplicated().any():
        raise ValueError("Duplicate column names are not supported.")
    if len(data.columns) == 0:
        raise ValueError("The dataset has no columns.")
    return data.copy()


def numeric_measure(data: pd.DataFrame, measure: str) -> pd.Series:
    """Coerce a selected measure to numeric values; invalid cells become NaN."""
    if measure not in data.columns:
        raise KeyError(f"Unknown measure column: {measure}")
    return pd.to_numeric(data[measure], errors="coerce")


def calculate_kpis(data: pd.DataFrame, measure: str) -> dict:
    """Calculate total, average and valid observation count for a measure."""
    values = numeric_measure(data, measure).dropna()
    return {
        "total": float(values.sum()),
        "average": float(values.mean()) if not values.empty else 0.0,
        "valid_rows": int(values.count()),
    }


def aggregate_by(data: pd.DataFrame, category: str, measure: str) -> pd.DataFrame:
    """Aggregate a numeric measure by a categorical field, descending by total."""
    if category not in data.columns:
        raise KeyError(f"Unknown category column: {category}")
    work = pd.DataFrame({
        "category": data[category].astype("string"),
        "value": numeric_measure(data, measure),
    }).dropna(subset=["category", "value"])
    return (
        work.groupby("category", as_index=False)["value"]
        .sum()
        .sort_values("value", ascending=False)
        .reset_index(drop=True)
    )


def monthly_trend(data: pd.DataFrame, date_column: str, measure: str) -> pd.DataFrame:
    """Return monthly sums, excluding invalid dates and non-numeric measures."""
    if date_column not in data.columns:
        raise KeyError(f"Unknown date column: {date_column}")
    work = pd.DataFrame({
        "date": pd.to_datetime(data[date_column], errors="coerce"),
        "value": numeric_measure(data, measure),
    }).dropna(subset=["date", "value"])
    if work.empty:
        return pd.DataFrame(columns=["month", "value"])
    work["month"] = work["date"].dt.to_period("M").astype(str)
    return (
        work.groupby("month", as_index=False)["value"]
        .sum()
        .sort_values("month")
        .reset_index(drop=True)
    )
