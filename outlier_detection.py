"""
outlier_detection.py
====================

Module for detecting outliers in the `qty_klr` (Jumlah_Beli) column of a
cleaned pembelian dataset. Outlier detection is univariate and can use
either a z‐score based approach or an interquartile range (IQR) approach.

Functions in this module are written to be composable: they accept data
frames and return masks or plots without side effects. A caller (e.g.,
main.py) can import these functions, apply them to a data set, and then
decide how to handle any flagged outliers.

Usage example (within main.py or an interactive session):

    from outlier_detection import load_pembelian, detect_outliers, plot_outliers

    df = load_pembelian("data_cleaned/pembelian_cleaned.csv")
    mask = detect_outliers(df, method="zscore", threshold=3.0)
    plot_outliers(df, mask, title="Outlier Detection (Z-score)", path="image/before_handler.png")

"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Literal


def load_pembelian(path: str) -> pd.DataFrame:
	"""
	Muat CSV pembelian yang sudah dibersihkan:
	- Baca file dengan pemisah titik koma (fallback ke koma jika diperlukan)
	- Normalisasi nama kolom menjadi lowercase dengan underscore
	- Konversi kolom numerik (qty_klr, qty_msk, nilai_klr, nilai_msk) ke tipe numerik

	Args:
		path: path ke file CSV (semicolon-separated diharapkan)

	Returns:
		DataFrame dengan nama kolom terstandardisasi dan tipe numerik untuk kolom jumlah/nilai.
	"""
	df = pd.read_csv(path, sep=";", low_memory=False)
	if df.shape[1] == 1:
		df = pd.read_csv(path, sep=",", low_memory=False)
	df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
	for col in ["qty_klr", "qty_msk", "nilai_klr", "nilai_msk"]:
		if col in df.columns:
			df[col] = pd.to_numeric(df[col], errors="coerce")
	return df


def detect_outliers(
	df: pd.DataFrame,
	method: Literal["zscore", "iqr"] = "zscore",
	threshold: float = 3.0,
	iqr_factor: float = 1.5,
) -> pd.Series:
	"""
	Mendeteksi pencilan pada kolom `qty_klr`.

	Args:
		df: DataFrame yang mengandung kolom `qty_klr`.
		method: "zscore" atau "iqr".
		threshold: ambang z-score (dipakai saat method="zscore").
		iqr_factor: faktor IQR (dipakai saat method="iqr").

	Returns:
		Series boolean yang menandai baris pencilan (True) sesuai index df.
	"""
	if "qty_klr" not in df.columns:
		raise KeyError("DataFrame harus berisi kolom 'qty_klr' untuk deteksi outlier.")
	series = df["qty_klr"].astype(float)
	if method == "zscore":
		mean = series.mean()
		std = series.std(ddof=0)
		if std == 0:
			return pd.Series([False] * len(series), index=series.index)
		z = (series - mean) / std
		return z.abs() > threshold
	elif method == "iqr":
		q1 = series.quantile(0.25)
		q3 = series.quantile(0.75)
		iqr = q3 - q1
		lower = q1 - iqr_factor * iqr
		upper = q3 + iqr_factor * iqr
		return (series < lower) | (series > upper)
	else:
		raise ValueError(f"Unsupported method '{method}'. Use 'zscore' or 'iqr'.")


def plot_outliers(
	df: pd.DataFrame,
	outlier_mask: pd.Series,
	title: str,
	path: str,
) -> None:
	"""
	Buat scatter plot index transaksi vs qty_klr dan sorot pencilan.

	Args:
		df: DataFrame yang mengandung kolom `qty_klr`.
		outlier_mask: Series boolean yang menandai baris pencilan.
		title: Judul plot.
		path: Path untuk menyimpan gambar.
	"""
	idx = np.arange(len(df))
	plt.figure(figsize=(12, 6))
	plt.scatter(idx[~outlier_mask], df.loc[~outlier_mask, "qty_klr"], color="blue", s=10, alpha=0.3, label="Normal")
	if outlier_mask.any():
		plt.scatter(idx[outlier_mask], df.loc[outlier_mask, "qty_klr"], color="red", s=20, alpha=0.8, label="Outlier")
	plt.title(title)
	plt.xlabel("Transaction Index")
	plt.ylabel("Jumlah Beli (qty_klr)")
	plt.legend()
	plt.tight_layout()
	plt.savefig(path, dpi=150)
	plt.close()
