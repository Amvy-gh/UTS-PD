"""
Script ini membersihkan dan mengkonversi file TSV mentah untuk data transaksi apotek 
dan data stok menjadi file CSV terstruktur. Script dirancang untuk membaca file 
`pembelian.tsv` dan `stok.tsv` yang disediakan, mengurai setiap baris sesuai format 
yang diharapkan, dan menghasilkan data yang sudah dibersihkan ke dalam direktori 
`data_cleaned` sebagai file CSV.

Cara Penggunaan:
    python data_cleaning.py

Script akan membuat direktori `data_cleaned` jika belum ada dan menulis
dua file CSV:

    - data_cleaned/pembelian_cleaned.csv
    - data_cleaned/stok_cleaned.csv

Format TSV diasumsikan dapat dibaca manusia dengan pemisah spasi yang bervariasi.
Header produk dimulai dengan kode yang diawali 'A' diikuti angka; baris transaksi 
dimulai dengan tanggal dalam format `dd-mm-yy`. File stok berisi baris dengan 
kode, nama produk, lokasi, jumlah dan satuan.

"""

import os
import re
import pandas as pd
from typing import List, Dict


def to_float_id(num_str: str) -> float:
    """
    Mengkonversi string angka format Indonesia menjadi float Python.
    Input bisa mengandung pemisah ribuan (titik atau spasi) dan
    koma desimal.

    Contoh:
        '1.234,50' -> 1234.50
        '2,00'     -> 2.00

    Jika string kosong atau tidak bisa dikonversi, mengembalikan 0.0.

    Args:
        num_str: String yang mewakili angka dengan format lokal Indonesia.

    Returns:
        Representasi float dari input.
    """
    if num_str is None or str(num_str).strip() == "":
        return 0.0
    s = str(num_str).strip()
    # remove all characters except digits, dots, commas and minus signs
    s = re.sub(r"[^0-9.,-]", "", s)
    if s == "":
        return 0.0
    # If there is a comma, treat it as the decimal separator
    if "," in s:
        # remove thousand separators represented by dots
        s = s.replace(".", "")
        # replace the decimal comma with a dot
        s = s.replace(",", ".")
    else:
        # If no comma is present, but there are multiple dots, assume all but
        # the last dot are thousand separators
        parts = s.split(".")
        if len(parts) > 2:
            s = "".join(parts[:-1]) + "." + parts[-1]
    try:
        return float(s)
    except ValueError:
        # fallback to zero if conversion fails
        return 0.0


def parse_pembelian_tsv(filepath: str) -> pd.DataFrame:
    """
    Mengurai file TSV pembelian mentah menjadi DataFrame terstruktur.
    DataFrame yang dihasilkan berisi kolom untuk tanggal, nomor transaksi,
    jumlah masuk dan keluar, nilai-nilainya, dan metadata produk.

    Args:
        filepath: Path ke file `pembelian.tsv`.

    Returns:
        DataFrame pandas dengan data transaksi yang sudah diurai.
    """
    pembelian_records: List[Dict[str, any]] = []
    current_code: str = None
    current_name: str = None
    current_unit: str = None

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            raw_line = line.rstrip("\n")
            # lewati baris yang benar-benar kosong
            if not raw_line.strip():
                continue
            stripped = raw_line.strip()
            # lewati baris header yang dimulai dengan KODE atau TANGGAL
            if stripped.startswith("KODE") or stripped.startswith("TANGGAL"):
                continue
            # Deteksi baris header produk: dimulai dengan kode A diikuti angka
            if re.match(r"^A\d+", stripped):
                # pisahkan berdasarkan spasi
                parts = re.split(r"\s+", stripped)
                if not parts:
                    continue
                current_code = parts[0]
                # Bagian terakhir adalah satuan (mis. STRIP, BTL)
                current_unit = parts[-1]
                # Nama adalah semua bagian antara kode dan satuan
                if len(parts) > 2:
                    current_name = " ".join(parts[1:-1])
                elif len(parts) == 2:
                    current_name = parts[1]
                else:
                    current_name = None
                continue
            # Detect transaction lines: they start with a date dd-mm-yy
            trans_match = re.match(r"^(\d{2}-\d{2}-\d{2})\s+(\S+)\s+(.*)$", raw_line.lstrip())
            if not trans_match:
                continue
            tanggal_str = trans_match.group(1)
            no_transaksi = trans_match.group(2)
            remainder = trans_match.group(3)
            # find all numbers that use a comma as decimal separator
            num_tokens = re.findall(r"\d[\d\.]*,\d+", remainder)
            qty_msk = 0.0
            nilai_msk = 0.0
            qty_klr = 0.0
            nilai_klr = 0.0
            if len(num_tokens) == 4:
                # If there are four numbers, treat as: qty_msk, nilai_msk, qty_klr, nilai_klr
                qty_msk = to_float_id(num_tokens[0])
                nilai_msk = to_float_id(num_tokens[1])
                qty_klr = to_float_id(num_tokens[2])
                nilai_klr = to_float_id(num_tokens[3])
            elif len(num_tokens) == 2:
                # Use the position of the first numeric token to decide whether
                # it's a purchase (msk) or sale (klr). Numbers earlier in the line
                # correspond to purchases, later correspond to sales.
                idx = raw_line.lstrip().find(num_tokens[0])
                # threshold empirically derived from sample lines; adjust as needed
                if idx < 60:
                    qty_msk = to_float_id(num_tokens[0])
                    nilai_msk = to_float_id(num_tokens[1])
                else:
                    qty_klr = to_float_id(num_tokens[0])
                    nilai_klr = to_float_id(num_tokens[1])
            elif len(num_tokens) == 3:
                # Rare case: assume first two are msk and the last is qty_klr
                qty_msk = to_float_id(num_tokens[0])
                nilai_msk = to_float_id(num_tokens[1])
                qty_klr = to_float_id(num_tokens[2])
            elif len(num_tokens) == 1:
                # If only one number is present, treat as quantity msk
                qty_msk = to_float_id(num_tokens[0])
            # Append record
            pembelian_records.append({
                "tanggal": tanggal_str,
                "no_transaksi": no_transaksi,
                "qty_msk": qty_msk,
                "nilai_msk": nilai_msk,
                "qty_klr": qty_klr,
                "nilai_klr": nilai_klr,
                "kode": current_code,
                "nama_produk": current_name,
                "unit": current_unit,
            })
    # Convert to DataFrame
    df = pd.DataFrame(pembelian_records)
    # Convert tanggal column to datetime if possible
    df["tanggal"] = pd.to_datetime(df["tanggal"], format="%d-%m-%y", errors="coerce")
    # Ensure numeric columns are floats
    for col in ["qty_msk", "nilai_msk", "qty_klr", "nilai_klr"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    return df


def parse_stok_tsv(filepath: str) -> pd.DataFrame:
    """
    Mengurai file TSV stok mentah menjadi DataFrame terstruktur.
    Setiap baris input berisi kode, nama produk, lokasi, jumlah, dan satuan.

    Args:
        filepath: Path ke file `stok.tsv`.

    Returns:
        DataFrame pandas dengan data stok yang sudah diurai.
    """
    stok_records: List[Dict[str, any]] = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            stripped = line.strip()
            # skip header lines
            if stripped.startswith("KODE"):
                continue
            parts = re.split(r"\s+", stripped)
            # valid stock line starts with code A followed by digits
            if not re.match(r"^A\d+", parts[0]):
                continue
            code = parts[0]
            unit = parts[-1]
            qty_raw = parts[-2]
            location = parts[-3]
            name_tokens = parts[1:-3]
            name = " ".join(name_tokens)
            qty = to_float_id(qty_raw)
            stok_records.append({
                "kode": code,
                "nama_produk": name,
                "lokasi": location,
                "qty_stok": qty,
                "unit": unit,
            })
    return pd.DataFrame(stok_records)


def main() -> None:
    """
    Fungsi utama: mengurai file TSV dan menulis file CSV yang sudah 
    dibersihkan ke dalam direktori `data_cleaned`.
    """
    input_pembelian = os.path.join("data_original", "pembelian.tsv")
    input_stok = os.path.join("data_original", "stok.tsv")
    output_dir = os.path.join("data_cleaned")
    os.makedirs(output_dir, exist_ok=True)

    # Parse input files
    pembelian_df = parse_pembelian_tsv(input_pembelian)
    stok_df = parse_stok_tsv(input_stok)

    # Write output CSV files
    pembelian_output = os.path.join(output_dir, "pembelian_cleaned.csv")
    stok_output = os.path.join(output_dir, "stok_cleaned.csv")
    pembelian_df.to_csv(pembelian_output, index=False)
    stok_df.to_csv(stok_output, index=False)

    print(f"Saved {len(pembelian_df)} pembelian records to {pembelian_output}")
    print(f"Saved {len(stok_df)} stok records to {stok_output}")


if __name__ == "__main__":
    main()