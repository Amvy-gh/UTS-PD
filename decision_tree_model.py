"""
decision_tree_model.py
======================

Melatih Decision Tree untuk klasifikasi level stok (High/Low) dan
menyediakan utilitas untuk mengekspor hasil prediksi serta feature importance
ke CSV.

Alur:
- prepare_features()  -> siapkan X, y, feature_names, dan tabel basis (kode+fitur+label)
- train_decision_tree() -> latih model
- predict_and_export()  -> buat prediksi (label & probabilitas), simpan ke CSV
- export_feature_importance() -> simpan feature importance ke CSV
- plot_decision_tree() -> simpan visualisasi pohon ke file gambar
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Tuple, List
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt


def _read_csv_auto_delim(path: str) -> pd.DataFrame:
    """
    Coba baca CSV menggunakan pemisah ';' lalu fallback ke ',' jika diperlukan.

    Args:
        path: path ke file CSV.

    Returns:
        DataFrame hasil pembacaan.
    """
    try:
        df = pd.read_csv(path, sep=";", low_memory=False)
        if df.shape[1] == 1:
            df = pd.read_csv(path, sep=",", low_memory=False)
    except Exception:
        df = pd.read_csv(path, sep=",", low_memory=False)
    return df


def prepare_features(
    pembelian_path: str,
    stok_path: str,
) -> Tuple[pd.DataFrame, pd.Series, List[str], pd.DataFrame]:
    """
    Agregasi transaksi pembelian per kode, gabungkan dengan stok, dan siapkan fitur & label.

    Returns:
        X, y, feature_names, base_df
        - X: fitur numerik
        - y: target biner (1=High, 0=Low)
        - feature_names: daftar nama fitur
        - base_df: DataFrame berisi `kode`, fitur, `qty_stok`, dan `stock_high`
    """
    # --- Pembelian ---
    df_pemb = _read_csv_auto_delim(pembelian_path)
    df_pemb.columns = [c.strip().lower().replace(" ", "_") for c in df_pemb.columns]

    for col in ["qty_msk", "qty_klr", "nilai_msk", "nilai_klr"]:
        if col in df_pemb.columns:
            df_pemb[col] = pd.to_numeric(df_pemb[col], errors="coerce").fillna(0.0)
        else:
            df_pemb[col] = 0.0

    agg = df_pemb.groupby("kode").agg({
        "qty_msk": "sum",
        "qty_klr": "sum",
        "nilai_msk": "sum",
        "nilai_klr": "sum",
    }).reset_index()

    # --- Stok ---
    df_stok = _read_csv_auto_delim(stok_path)
    df_stok.columns = [c.strip().lower().replace(" ", "_") for c in df_stok.columns]
    if "qty_stok" not in df_stok.columns:
        raise KeyError("Stok file must contain 'qty_stok' column")
    df_stok["qty_stok"] = pd.to_numeric(df_stok["qty_stok"], errors="coerce").fillna(0.0)

    # --- Join & label ---
    merged = agg.merge(df_stok[["kode", "qty_stok"]], on="kode", how="left")
    merged["qty_stok"] = pd.to_numeric(merged["qty_stok"], errors="coerce").fillna(0.0)

    median_stock = merged["qty_stok"].median()
    merged["stock_high"] = (merged["qty_stok"] > median_stock).astype(int)

    feature_names = ["qty_msk", "qty_klr", "nilai_msk", "nilai_klr"]
    X = merged[feature_names].copy()
    y = merged["stock_high"].copy()

    return X, y, feature_names, merged[["kode"] + feature_names + ["qty_stok", "stock_high"]].copy()


def train_decision_tree(
    cleaned_pembelian_path: str,
    stok_path: str,
    random_state: int = 42,
    max_depth: int | None = None,
) -> Tuple[pd.DataFrame, pd.Series, DecisionTreeClassifier, List[str], pd.DataFrame]:
    """
    Latih classifier Decision Tree.

    Returns:
        X, y, clf, feature_names, base_df
    """
    X, y, feature_names, base_df = prepare_features(cleaned_pembelian_path, stok_path)
    clf = DecisionTreeClassifier(random_state=random_state, max_depth=max_depth)
    clf.fit(X, y)
    return X, y, clf, feature_names, base_df


def predict_and_export(
    model: DecisionTreeClassifier,
    X: pd.DataFrame,
    base_df: pd.DataFrame,
    out_pred_csv: str,
) -> pd.DataFrame:
    """
    Buat prediksi label dan probabilitas, lalu simpan hasil ke CSV.

    Output CSV berisi kolom:
    - kode, qty_msk, qty_klr, nilai_msk, nilai_klr, qty_stok, stock_high (label asli),
      pred_label, prob_low, prob_high
    """
    proba = model.predict_proba(X)
    pred = model.predict(X)

    result = base_df.copy()
    result["pred_label"] = pred
    result["prob_low"] = proba[:, 0]
    result["prob_high"] = proba[:, 1]
    # Simpan
    result.to_csv(out_pred_csv, sep=";", index=False)
    return result


def export_feature_importance(
    model: DecisionTreeClassifier,
    feature_names: List[str],
    out_csv: str,
) -> pd.DataFrame:
    """Simpan feature importance ke CSV dalam Bahasa Indonesia."""
    fi = pd.DataFrame({
        "feature": feature_names,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)
    fi.to_csv(out_csv, sep=";", index=False)
    return fi


def plot_decision_tree(
    model: DecisionTreeClassifier,
    feature_names: List[str],
    output_path: str,
) -> None:
    """Simpan visual pohon ke file gambar."""
    plt.figure(figsize=(16, 10))
    plot_tree(model, feature_names=feature_names, class_names=["Low", "High"],
              filled=True, rounded=True, proportion=False, fontsize=9)
    plt.title("Decision Tree for Stock Level Classification")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
