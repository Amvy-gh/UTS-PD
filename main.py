"""
main.py
=======

Pipeline:
1) Load pembelian & stok (dari data_cleaned).
2) Deteksi outlier (Z-score/IQR) -> plot BEFORE.
3) Klasifikasi outlier -> hapus yang error, simpan hasil ke data_result_preprocessing.
4) Plot AFTER (setelah handler).
5) Latih Decision Tree, simpan:
    - Gambar tree -> image/decision_tree.png
    - CSV prediksi (per kode) -> data_model/tree_predictions.csv
    - CSV feature importance -> data_model/tree_feature_importances.csv
"""

from __future__ import annotations

import argparse
import os
import pandas as pd

from outlier_detection import load_pembelian, detect_outliers, plot_outliers
from handler import classify_outliers, handle_outliers
from decision_tree_model import (
    train_decision_tree,
    plot_decision_tree,
    predict_and_export,
    export_feature_importance,
)


def ensure_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent and not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Pipeline: outlier -> handler -> decision tree (with CSV outputs).")
    parser.add_argument("--input", default="data_cleaned/pembelian_cleaned.csv")
    parser.add_argument("--stok", default="data_cleaned/stok_cleaned.csv")
    parser.add_argument("--method", choices=["zscore", "iqr"], default="zscore")
    parser.add_argument("--threshold", type=float, default=3.0)
    parser.add_argument("--iqr-factor", type=float, default=1.5)
    parser.add_argument("--output-clean", default="data_result_handler/pembelian_final.csv")
    parser.add_argument("--before-image", default="image/before_handler.png")
    parser.add_argument("--after-image", default="image/after_handler.png")
    parser.add_argument("--tree-image", default="image/decision_tree.png")
    # --- output CSV dari algoritma tree ke folder berbeda (default: data_model)
    parser.add_argument("--pred-csv", default="data_model/tree_predictions.csv",
                        help="CSV hasil prediksi Decision Tree (per kode)")
    parser.add_argument("--fi-csv", default="data_model/tree_feature_importances.csv",
                        help="CSV feature importance Decision Tree")

    parser.add_argument("--tree-max-depth", type=int, default=None,
                        help="Maksimal kedalaman pohon (opsional)")

    args = parser.parse_args()

    # pastikan semua folder tujuan ada
    for p in [args.output_clean, args.before_image, args.after_image,
              args.tree_image, args.pred_csv, args.fi_csv]:
        ensure_dir(p)

    # 1) Load pembelian
    df = load_pembelian(args.input)

    # 2) Deteksi outlier
    outlier_mask = detect_outliers(df, method=args.method,
                                   threshold=args.threshold, iqr_factor=args.iqr_factor)
    plot_outliers(df, outlier_mask, f"Outlier detection ({args.method})", args.before_image)

    # 3) Klasifikasi & handling
    classified = classify_outliers(df, outlier_mask)
    cleaned_df, cleaned_outliers_mask = handle_outliers(df, outlier_mask, classified)
    plot_outliers(cleaned_df, cleaned_outliers_mask, f"After handling outliers ({args.method})", args.after_image)
    cleaned_df.to_csv(args.output_clean, sep=";", index=False)
    print(f"[OK] Cleaned dataset -> {args.output_clean}")

    # 4) Train tree + prediksi + ekspor CSV hasil algoritma
    X, y, model, feature_names, base_df = train_decision_tree(
        cleaned_pembelian_path=args.output_clean,
        stok_path=args.stok,
        random_state=42,
        max_depth=args.tree_max_depth,
    )
    plot_decision_tree(model, feature_names, args.tree_image)
    print(f"[OK] Decision tree image -> {args.tree_image}")

    # 5) CSV hasil algoritma
    pred_df = predict_and_export(model, X, base_df, args.pred_csv)
    print(f"[OK] Tree predictions CSV -> {args.pred_csv}  (rows={len(pred_df)})")

    fi_df = export_feature_importance(model, feature_names, args.fi_csv)
    print(f"[OK] Feature importance CSV -> {args.fi_csv}")

if __name__ == "__main__":
    main()
