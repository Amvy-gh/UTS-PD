"""
handler.py
==========

Functions for classifying detected outliers and handling them by either
removing suspected errors or retaining legitimate bulk purchases. These
utilities build upon the `outlier_detection` module and assume the
pembelian data has been loaded and an outlier mask computed.

The classification uses two heuristics:
    * Price per unit: compute the median price per unit per product (`kode`).
      An outlier transaction is considered erroneous if its price per unit
      differs from the median by more than ±50% (ratio < 0.5 or > 1.5).
    * Unit consistency: compute the most common unit per `kode`. An outlier
      transaction is considered erroneous if its unit differs from this
      typical unit.

Both criteria are combined with OR logic: if either condition indicates
anomalous behavior, the outlier is treated as an error.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Tuple


def typical_unit_per_code(df: pd.DataFrame) -> pd.Series:
    """Return a Series mapping each `kode` to its most frequent unit.

    Args:
        df: DataFrame containing at least `kode` and `unit` columns.

    Returns:
        Series indexed by `kode` with the most common unit as values.
    """
    def most_common(series: pd.Series):
        # Drop missing values and return the most common unit if present
        vc = series.dropna().value_counts()
        return vc.idxmax() if len(vc) > 0 else None

    return df.groupby("kode")["unit"].agg(most_common)


def classify_outliers(df: pd.DataFrame, outlier_mask: pd.Series) -> pd.DataFrame:
    """Classify outlier rows as errors or legitimate bulk purchases.

    For each outlier transaction, compute a price per unit. Compare this
    to the median price per unit for that product (`kode`). Also determine
    whether the unit used in the transaction matches the typical unit for
    that product. If the price deviates significantly or the unit is
    inconsistent, label the outlier as an error.

    Args:
        df: Full dataset including `qty_klr`, `nilai_klr`, `kode` and `unit`.
        outlier_mask: Boolean Series marking which rows are outliers.

    Returns:
        DataFrame subset of only the outliers with additional columns:
            `price_per_unit`, `median_price_per_unit`, `price_ratio`,
            `typical_unit`, `is_error`.
    """
    outliers_df = df.loc[outlier_mask].copy()
    # price per unit (avoid divide by zero)
    outliers_df["price_per_unit"] = outliers_df["nilai_klr"] / outliers_df["qty_klr"].replace(0, np.nan)
    # median price per unit per code
    price_per_unit_all = df["nilai_klr"] / df["qty_klr"].replace(0, np.nan)
    med_price = price_per_unit_all.groupby(df["kode"]).median()
    outliers_df = outliers_df.join(med_price.rename("median_price_per_unit"), on="kode")
    # ratio of actual price vs median
    outliers_df["price_ratio"] = outliers_df["price_per_unit"] / outliers_df["median_price_per_unit"]
    # typical unit per code
    typ_unit = typical_unit_per_code(df)
    outliers_df = outliers_df.join(typ_unit.rename("typical_unit"), on="kode")
    # classification: error if price ratio outside [0.5, 1.5] or unit mismatch
    outliers_df["is_error"] = (
        (outliers_df["price_ratio"] < 0.5) |
        (outliers_df["price_ratio"] > 1.5) |
        (outliers_df["unit"] != outliers_df["typical_unit"])
    )
    return outliers_df


def handle_outliers(
    df: pd.DataFrame,
    outlier_mask: pd.Series,
    classification_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Handle outliers based on classification.

    Errors (rows where `is_error` is True) are removed from the data. Legitimate
    outliers are retained. A new mask is returned marking only those
    legitimate outliers in the cleaned DataFrame.

    Args:
        df: Original DataFrame.
        outlier_mask: Boolean Series marking outliers in `df`.
        classification_df: DataFrame returned from `classify_outliers`.

    Returns:
        Tuple of (cleaned_df, cleaned_outlier_mask), where cleaned_df has
        erroneous rows removed and cleaned_outlier_mask is a boolean Series
        aligned to cleaned_df marking the remaining legitimate outliers.
    """
    # rows considered errors (by index)
    error_indices = classification_df[classification_df["is_error"]].index
    # remove erroneous outliers
    cleaned_df = df.drop(index=error_indices)
    # recompute mask: outliers that are legitimate (i.e., not in error_indices)
    cleaned_mask = outlier_mask.drop(index=error_indices, errors="ignore")
    # Align cleaned_mask to cleaned_df: set False for rows not flagged
    cleaned_mask = cleaned_mask.reindex(cleaned_df.index, fill_value=False)
    return cleaned_df, cleaned_mask
