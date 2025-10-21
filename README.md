# Proyek Data Mining: Analisis Pipeline Data Apotek

Repositori ini berisi implementasi *pipeline* (alur kerja) data mining lengkap untuk menganalisis dan memodelkan data penjualan dan stok apotek. Proyek ini mendemonstrasikan proses *end-to-end* mulai dari pembersihan data mentah hingga pembuatan model prediktif.

## 👥 Anggota Kelompok

* Luthfiandri Ardanie (122140089)
* Muhammad Fatih Hanbali (122140112)
* Anjes Bermana (122140190)

## 📝 Deskripsi Proyek

Tujuan dari proyek ini adalah untuk mengambil data apotek yang mentah, tidak terstruktur, dan "human-readable" (ditulis untuk dibaca manusia), kemudian mengubahnya menjadi data terstruktur yang siap dianalisis.

Setelah data bersih, *pipeline* akan melakukan *preprocessing* untuk menangani anomali data (*outlier*), dan akhirnya membangun model *machine learning* (Decision Tree) untuk memprediksi apakah level stok suatu produk tergolong 'High' (Tinggi) atau 'Low' (Rendah).

## ⚙️ Alur Kerja (Pipeline)

Proses ini dibagi menjadi beberapa tahap utama yang saling berhubungan:

### Tahap 1: Pembersihan Data (`data_cleaning.py`)

* **Tujuan:** Mengubah data mentah (.tsv) menjadi data bersih (.csv).
* **Proses:**
    * Membaca file `pembelian.tsv` dan `stok.tsv` baris per baris.
    * Menggunakan **Regular Expressions (Regex)** untuk mem-parsing baris yang tidak konsisten (membedakan baris header produk dan baris transaksi).
    * Menggunakan fungsi `to_float_id` untuk mengonversi format angka Indonesia (misal, `1.234,50`) menjadi format *float* standar (`1234.50`).
* **Output:** `data_cleaned/pembelian_cleaned.csv` dan `data_cleaned/stok_cleaned.csv`.

### Tahap 2: Deteksi Outlier (`outlier_detection.py`)

* **Tujuan:** Mengidentifikasi data anomali (pencilan) pada data penjualan.
* **Input:** `pembelian_cleaned.csv`.
* **Proses:**
    * Fokus pada kolom `qty_klr` (kuantitas keluar).
    * Menerapkan metode statistik **Z-Score** dan **IQR (Interquartile Range)** untuk menandai baris data yang dianggap *outlier*.
* **Output:** Sebuah *mask* (penanda) boolean untuk *outlier* dan visualisasi `image/before_handler.png`.

### Tahap 3: Penanganan Outlier (`handler.py`)

* **Tujuan:** Memutuskan apa yang harus dilakukan terhadap *outlier* yang ditemukan.
* **Input:** Data bersih dan *mask* *outlier* dari Tahap 2.
* **Proses:**
    * Mengklasifikasikan setiap *outlier* sebagai "Error" atau "Legitimate" (Sah).
    * Sebuah *outlier* dianggap **Error** (dan dihapus) jika:
        1.  Harganya (`price_per_unit`) berbeda jauh dari median harga produk tsb.
        2.  Satuannya (`unit`) berbeda dari satuan yang paling umum dipakai produk tsb.
    * *Outlier* yang **Legitimate** (misal, pembelian borongan yang sah) tetap disimpan.
* **Output:** `data_result_handler/pembelian_final.csv` dan visualisasi `image/after_handler.png`.

### Tahap 4: Pemodelan (`decision_tree_model.py`)

* **Tujuan:** Membangun model untuk memprediksi level stok (High/Low).
* **Input:** `pembelian_final.csv` (super bersih) dan `stok_cleaned.csv`.
* **Proses:**
    1.  **Feature Engineering:** Agregasi data transaksi (dikelompokkan per `kode` produk) untuk membuat fitur (X), seperti total `qty_msk`, `qty_klr`, `nilai_msk`, dll.
    2.  **Labeling:** Membuat target (Y) `stock_high` dengan membandingkan `qty_stok` produk dengan median stok keseluruhan (1 jika > median, 0 jika <= median).
    3.  **Training:** Melatih model `DecisionTreeClassifier` untuk belajar pola dari X agar bisa memprediksi Y.

### Tahap 5: Evaluasi & Ekspor (Bagian dari `decision_tree_model.py` dan `main.py`)

* **Tujuan:** Menyimpan semua hasil analisis.
* **Output:**
    * `image/decision_tree.png`: Visualisasi aturan pohon keputusan.
    * `data_model/tree_predictions.csv`: Hasil prediksi High/Low untuk setiap produk.
    * `data_model/tree_feature_importances.csv`: Peringkat fitur yang paling berpengaruh bagi model.

---

## 🚀 Cara Menjalankan

**1. Kebutuhan (Prerequisites)**

Pastikan Anda memiliki Python dan *library* berikut:
* pandas
* scikit-learn (sklearn)
* matplotlib
* seaborn

Anda dapat menginstalnya menggunakan pip:
```bash
pip install pandas scikit-learn matplotlib seaborn
```

**2. Menjalankan Pipeline**

Pipeline ini harus dijalankan dalam dua langkah:

**Langkah A:** Jalankan Pembersihan Data Awal
Script ini dijalankan terpisah untuk menyiapkan data bersih dari data mentah.

```bash
python data_cleaning.py
```
Ini akan menghasilkan file-file di dalam folder data_cleaned/.

**Langkah B:** Jalankan Pipeline Utama (Preprocessing & Modeling)
Setelah data bersih ada, jalankan main.py. Script ini akan otomatis menjalankan Tahap 2, 3, dan 4 secara berurutan.

```bash
python main.py
```
Ini akan menghasilkan semua file di folder data_result_handler/, data_model/, dan image/.

**Contoh Menjalankan dengan Opsi Metode Deteksi Pencilan:**
Anda dapat memilih metode deteksi pencilan (zscore atau iqr) saat menjalankan pipeline utama.

```bash
python main.py --input data_cleaned/pembelian_cleaned.csv --stok data_cleaned/stok_cleaned.csv --method zscore --threshold 3.0
```

Argumen penting:
- `--method {zscore,iqr}`      : Pilih metode deteksi pencilan
- `--threshold (float)`        : Ambang untuk zscore (mis. 3.0)
- `--iqr-factor (float)`       : Faktor IQR untuk metode iqr (mis. 1.5)
- `--output-clean`             : Lokasi CSV hasil pembersihan
- `--pred-csv, --fi-csv`       : Lokasi CSV hasil model

**Cara menjalankan dengan metode berbeda (contoh Bash)**
```bash
# Gunakan default (zscore) — menghasilkan semua file di data_result_handler/, data_model/, dan image/
python main.py

# Jalankan eksplisit dengan metode z-score (atur ambang jika perlu)
python main.py --method zscore --threshold 3.0

# Jalankan menggunakan metode IQR (atur faktor IQR jika perlu)
python main.py --method iqr --iqr-factor 1.5
```

Keterangan singkat:
- zscore: menandai baris dimana |(value - mean)/std| > threshold.
- iqr: menandai baris di luar [Q1 - factor*IQR, Q3 + factor*IQR].

## 📁 Struktur Repositori
.
├── data_original/     # Data mentah asli (.tsv)
├── data_cleaned/      # Output Tahap 1: Data bersih (.csv)
├── data_result_handler/ # Output Tahap 3: Data final setelah penanganan outlier (.csv)
├── data_model/        # Output Tahap 4: Hasil prediksi & feature importance (.csv)
├── image/             # Output visualisasi (plot outlier, decision tree)
│
├── data_cleaning.py   # Script Tahap 1 (Pembersihan Awal)
├── outlier_detection.py # Script Tahap 2 (Deteksi)
├── handler.py         # Script Tahap 3 (Penanganan)
├── decision_tree_model.py # Script Tahap 4 (Pemodelan)
├── main.py            # Script utama (orkestrator) untuk menjalankan Tahap 2, 3, 4
│
└── README.md          # File Cara Menjalankan Proyek