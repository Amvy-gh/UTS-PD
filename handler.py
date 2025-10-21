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
	"""
	Kembalikan Series yang memetakan setiap `kode` ke satuan yang paling sering muncul.

	Args:
		df: DataFrame yang memuat kolom `kode` dan `unit`.

	Returns:
		Series terindeks `kode` dengan nilai satuan yang paling umum.
	"""
	def most_common(series: pd.Series):
		vc = series.dropna().value_counts()
		return vc.idxmax() if len(vc) > 0 else None
	return df.groupby("kode")["unit"].agg(most_common)


def classify_outliers(df: pd.DataFrame, outlier_mask: pd.Series) -> pd.DataFrame:
	"""
	Klasifikasikan baris pencilan sebagai kesalahan input atau pembelian besar yang sah.

	Metode:
	- Hitung harga per unit untuk tiap transaksi pencilan.
	- Bandingkan dengan median harga per unit per `kode`.
	- Periksa apakah `unit` transaksi sesuai dengan `typical_unit` untuk `kode`.
	- Jika deviasi harga > 50% atau unit tidak cocok, tandai sebagai error.

	Args:
		df: Dataset lengkap (harus memuat qty_klr, nilai_klr, kode, unit).
		outlier_mask: Series boolean yang menandai baris pencilan.

	Returns:
		DataFrame subset yang hanya berisi baris pencilan dengan kolom tambahan:
		`price_per_unit`, `median_price_per_unit`, `price_ratio`, `typical_unit`, `is_error`.
	"""
	outliers_df = df.loc[outlier_mask].copy()
	outliers_df["price_per_unit"] = outliers_df["nilai_klr"] / outliers_df["qty_klr"].replace(0, np.nan)
	price_per_unit_all = df["nilai_klr"] / df["qty_klr"].replace(0, np.nan)
	med_price = price_per_unit_all.groupby(df["kode"]).median()
	outliers_df = outliers_df.join(med_price.rename("median_price_per_unit"), on="kode")
	outliers_df["price_ratio"] = outliers_df["price_per_unit"] / outliers_df["median_price_per_unit"]
	typ_unit = typical_unit_per_code(df)
	outliers_df = outliers_df.join(typ_unit.rename("typical_unit"), on="kode")
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
	"""
	Tangani pencilan berdasarkan klasifikasi:
	- Hapus baris yang dianggap error (`is_error` True).
	- Pertahankan pencilan yang sah.
	- Kembalikan dataframe bersih dan mask yang menandai pencilan yang tersisa (legitimate).

	Args:
		df: DataFrame asal.
		outlier_mask: Series boolean menandai outlier di df.
		classification_df: DataFrame hasil `classify_outliers`.

	Returns:
		(cleaned_df, cleaned_outlier_mask)
	"""
	error_indices = classification_df[classification_df["is_error"]].index
	cleaned_df = df.drop(index=error_indices)
	cleaned_mask = outlier_mask.drop(index=error_indices, errors="ignore")
	cleaned_mask = cleaned_mask.reindex(cleaned_df.index, fill_value=False)
	return cleaned_df, cleaned_mask
