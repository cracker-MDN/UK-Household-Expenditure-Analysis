"""
data_processing.py
------------------
Data loading, cleaning, and encoding utilities for the
UK Household Expenditure Analysis.

Usage:
    from utils.data_processing import load_data, get_group_stats, COLS, ORDERS, LABELS
"""

import pandas as pd
import numpy as np

# ── Column name constants ────────────────────────────────────────────────────
COLS = {
    "occ":      "NS - SEC 8 Class of household reference person",
    "tenure":   "Tenure type",
    "adults":   "Number of adults",
    "children": "Number of children",
    "exp":      "Expenditure",
}

# ── Category orders (for consistent plotting/grouping) ───────────────────────
ORDERS = {
    "occ": [
        "Higher managerial, administrative and professional occupations",
        "Intermediate occupations",
        "Routine and manual occupations",
        "Never worked and long term unemployed, students and occupation not stated",
        "Not classified for other reasons",
    ],
    "tenure":   ["Owned", "Private rented", "Public rented"],
    "adults":   ["1 adult", "2 adults", "3 adults", "4 and more adults"],
    "children": ["No children", "One child", "Two or more children"],
}

# ── Short display labels for long occupational class strings ─────────────────
LABELS = {
    "Higher managerial, administrative and professional occupations": "Higher managerial",
    "Intermediate occupations":                                        "Intermediate",
    "Routine and manual occupations":                                  "Routine & manual",
    "Never worked and long term unemployed, students and occupation not stated": "Never worked/unemployed",
    "Not classified for other reasons":                                "Not classified",
}


def load_data(filepath: str) -> pd.DataFrame:
    """
    Load the LCF cleaned CSV and validate that all required columns are present.

    Parameters
    ----------
    filepath : str
        Path to LCF_cleaned.csv.

    Returns
    -------
    pd.DataFrame
        Raw dataframe with all original columns intact.

    Raises
    ------
    FileNotFoundError
        If the CSV file does not exist at the given path.
    ValueError
        If one or more required columns are missing from the file.
    """
    df = pd.read_csv(filepath)

    missing = [col for col in COLS.values() if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    print(f"Data loaded: {df.shape[0]:,} households, {df.shape[1]} variables")
    return df


def encode_categories(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cast all four predictor columns to pandas Categorical with the correct
    order. Required before fitting the OLS combined model.

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataframe returned by load_data().

    Returns
    -------
    pd.DataFrame
        Copy of df with predictor columns as ordered Categoricals.
    """
    df = df.copy()
    for key, col in COLS.items():
        if key == "exp":
            continue
        order = ORDERS.get(key)
        if order:
            df[col] = pd.Categorical(df[col], categories=order, ordered=True)
        else:
            df[col] = df[col].astype("category")
    return df


def apply_short_labels(df: pd.DataFrame, col: str) -> pd.Series:
    """
    Map occupational class values to their short display labels.
    Returns the original series unchanged for any other column.

    Parameters
    ----------
    df  : pd.DataFrame
    col : str  — full column name (use COLS["occ"] for occupational class)

    Returns
    -------
    pd.Series
    """
    if col == COLS["occ"]:
        return df[col].map(LABELS)
    return df[col]


def get_group_stats(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """
    Compute descriptive statistics (n, mean, SD, median) grouped by
    a categorical predictor.

    Parameters
    ----------
    df        : pd.DataFrame
    group_col : str  — one of the values in COLS (e.g. COLS["occ"])

    Returns
    -------
    pd.DataFrame
        Index = category labels, columns = [n, Mean (£), SD (£), Median (£)]
    """
    exp_col = COLS["exp"]
    stats = (
        df.groupby(group_col)[exp_col]
        .agg(n="count", mean="mean", std="std", median="median")
        .round(2)
    )
    stats.columns = ["n", "Mean (£)", "SD (£)", "Median (£)"]

    # Apply short labels for occupational class
    if group_col == COLS["occ"]:
        stats.index = [LABELS.get(i, i) for i in stats.index]

    return stats


def expenditure_summary(df: pd.DataFrame) -> dict:
    """
    Return summary statistics for the expenditure column as a dict.

    Returns
    -------
    dict with keys: mean, median, std, min, max, skewness, n
    """
    exp = df[COLS["exp"]]
    return {
        "mean":     round(exp.mean(), 2),
        "median":   round(exp.median(), 2),
        "std":      round(exp.std(), 2),
        "min":      round(exp.min(), 2),
        "max":      round(exp.max(), 2),
        "skewness": round(exp.skew(), 3),
        "n":        len(exp),
    }
