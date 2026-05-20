# 💻 Laptop Price Prediction Studio

Aplikasi web interaktif berbasis **Streamlit** untuk memprediksi harga laptop menggunakan algoritma Machine Learning — **Random Forest Regressor** dan **Decision Tree Regressor**. Proyek ini merupakan konversi langsung dari notebook Jupyter *Laptop Price Prediction* ke dalam sebuah aplikasi web yang lengkap dan informatif.

---

## 🖥️ Demo Tampilan

```
┌─────────────────────────────────────────────────────────┐
│  💻 Laptop Price Prediction Studio                      │
│  ─────────────────────────────────────────────────────  │
│  🏠 Overview │ 🔍 Data Exploration │ 📊 EDA │          │
│  🤖 Model Building │ 🎯 Predict Price                   │
│                                                         │
│  [ Gadget Store Theme — Dark Blue Neon UI ]             │
└─────────────────────────────────────────────────────────┘
```

---

## ✨ Fitur Aplikasi

| Halaman | Konten |
|---|---|
| 🏠 **Overview** | Ringkasan proyek, statistik dataset, ilustrasi gadget store |
| 🔍 **Data Exploration** | Raw data, feature engineering code, distribusi fitur |
| 📊 **EDA** | Bar plots, box plots, violin plots, scatter plots, heatmap korelasi |
| 🤖 **Model Building** | Metrics DT & RF, residual plots, feature importance |
| 🎯 **Predict Price** | Form input spesifikasi laptop → estimasi harga real-time |

---

## 🧪 Machine Learning Pipeline

### 1. Data Preprocessing Part 1 — Feature Engineering

```python
# Ekstraksi CPU Brand dari kolom CPU mentah
def fetch_processor(text):
    if 'Intel Core i7' in text: return 'Intel Core i7'
    elif 'Intel Core i5' in text: return 'Intel Core i5'
    elif 'Intel Core i3' in text: return 'Intel Core i3'
    elif 'AMD' in text:           return 'AMD Processor'
    else:                         return 'Other Intel Processor'

# Ekstraksi GPU Brand
def gpu_type(text):
    if 'Intel'  in text: return 'Intel'
    elif 'AMD'  in text: return 'AMD'
    elif 'Nvidia' in text: return 'Nvidia'
    else: return 'Other GPU'

# Ekstraksi Storage Type
def fetch_storage(text): ...  # 128GB SSD, 256GB SSD, 1TB HDD, dll.

# Screen Quality dari 9 karakter terakhir kolom Screen
df['Screen Quality'] = df['Screen'].str.slice(-9)
```

### 2. Data Preprocessing Part 2 — Cleaning

- Drop kolom: `Model Name`, `Screen`, `CPU`, `Storage`, `GPU`
- Imputasi missing values pada `Operating System Version`
- Label Encoding untuk semua kolom kategorikal
- Train/Test split: **80% train — 20% test**

### 3. Model Training

```python
# Decision Tree Regressor
dtree = DecisionTreeRegressor(random_state=0)

# Random Forest Regressor (Hyperparameter tuning via GridSearchCV)
rf = RandomForestRegressor(
    random_state=0,
    max_depth=9,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features='sqrt',
    n_estimators=50
)
```

### 4. Fitur Terpenting

| Rank | Random Forest | Decision Tree |
|---|---|---|
| 1 | Screen Quality | RAM |
| 2 | RAM | CPU brand |
| 3 | Weight | Weight |
| 4 | CPU brand | Category |
| 5 | Category | Screen Quality |

---

## 📁 Struktur Proyek

```
laptop-price-prediction/
│
├── app.py                  # Aplikasi Streamlit utama
├── requirements.txt        # Daftar dependensi Python
├── README.md               # Dokumentasi proyek (file ini)
│
└── data/                   # (Opsional) Letakkan dataset di sini
    ├── laptops_train.csv
    └── laptops_test.csv
```

> **Catatan:** Jika file CSV tidak tersedia, aplikasi akan otomatis men-*generate* data sintetis yang mereplikasi distribusi dan karakteristik dataset asli dari notebook.

---

## 🚀 Cara Menjalankan

### 1. Clone Repository

```bash
git clone https://github.com/username/laptop-price-prediction.git
cd laptop-price-prediction
```

### 2. Buat Virtual Environment (direkomendasikan)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependensi

```bash
pip install -r requirements.txt
```

### 4. Jalankan Aplikasi

```bash
streamlit run app.py
```

Aplikasi akan terbuka otomatis di browser pada alamat:
```
http://localhost:8501
```

---

## 📦 Requirements

```
streamlit>=1.32.0
pandas>=2.0.0
numpy>=1.26.0
matplotlib>=3.8.0
seaborn>=0.13.0
scikit-learn>=1.4.0
```

> Versi yang diuji: Python **3.10 / 3.11 / 3.12**

---

## 📊 Dataset

Dataset berisi informasi spesifikasi dan harga laptop dari berbagai merek. Kolom utama:

| Kolom | Tipe | Keterangan |
|---|---|---|
| `Manufacturer` | Kategorikal | Merek laptop (Apple, Asus, Dell, HP, dll.) |
| `Category` | Kategorikal | Jenis laptop (Ultrabook, Gaming, Notebook, dll.) |
| `Screen Size` | Numerik | Ukuran layar dalam inci |
| `RAM` | Kategorikal | Kapasitas RAM (4GB, 8GB, 16GB, dst.) |
| `CPU` | Teks | Spesifikasi prosesor mentah |
| `GPU` | Teks | Spesifikasi kartu grafis mentah |
| `Screen` | Teks | Deskripsi layar mentah |
| `Storage` | Teks | Deskripsi storage mentah |
| `Operating System` | Kategorikal | Sistem operasi |
| `Weight` | Numerik | Berat laptop dalam kg |
| `Price` | Numerik | Harga laptop (target prediksi) |

**Ukuran dataset:**
- Training set: **977 baris**
- Test set: **325 baris**

---

## 📈 Performa Model

| Metrik | Decision Tree | Random Forest |
|---|---|---|
| **MAE** | ~1,900,000 | ~1,793,080 |
| **RMSE** | ~3,100,000 | ~2,895,799 |
| **R² Score** | ~0.68 | ~0.75 |

> Random Forest memberikan performa lebih baik dengan R² ≈ 0.75, artinya model mampu menjelaskan ~75% variansi harga laptop.

---

## 🎨 Desain UI

Aplikasi menggunakan tema **Gadget Store Dark** dengan palet warna:

| Elemen | Warna |
|---|---|
| Background | `#0f0c29` → `#302b63` (gradient) |
| Aksen utama | `#00d4ff` (cyan neon) |
| Aksen sekunder | `#3a86ff` (biru) |
| Sidebar | `#1a1a2e` → `#16213e` |
| Teks | `#e0e0e0` / `#a0c4d8` |

---

## 🛠️ Tech Stack

- **Frontend/UI**: Streamlit + Custom CSS
- **Data Processing**: Pandas, NumPy
- **Visualisasi**: Matplotlib, Seaborn
- **Machine Learning**: Scikit-learn
- **Language**: Python 3.10+

---

## 👤 Author

Dibangun berdasarkan notebook Jupyter **Laptop Price Prediction** — dikembangkan menjadi aplikasi Streamlit interaktif dengan tema gadget store.

---

## 📄 Lisensi

Proyek ini bersifat open-source untuk keperluan edukasi dan pembelajaran Machine Learning.
