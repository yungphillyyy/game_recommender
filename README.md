# 🎮 Game Recommendation System

Sistem rekomendasi game berbasis **Classical Machine Learning** menggunakan dataset **vgsales.csv** dari Kaggle.

---

## 📋 Deskripsi

Sistem ini merekomendasikan video game menggunakan tiga pendekatan ML klasik:

| Metode | Teknik | Kegunaan |
|--------|--------|----------|
| **Content-Based Filtering** | Cosine Similarity | Rekomendasi berdasarkan kesamaan fitur game |
| **Clustering** | KMeans (20 cluster) | Rekomendasi dari kelompok game serupa |
| **Neighbor-Based** | KNN (k=11) | Rekomendasi berdasarkan tetangga terdekat |

> ✅ **Catatan**: Hanya menggunakan Classical ML — tanpa Deep Learning.

---

## 📦 Struktur Project

```
game_recommender/
│
├── app.py                    # 🚀 Streamlit Web App (Deployment)
├── requirements.txt          # Dependency Python
├── README.md                 # Dokumentasi ini
│
├── models/
│   ├── __init__.py
│   └── recommender.py        # Core ML Model (Cosine Sim + KMeans + KNN)
│
├── utils/
│   ├── __init__.py
│   └── data_loader.py        # Utility load & validasi dataset
│
└── data/
    └── vgsales.csv           # ← Letakkan dataset di sini
```

---

## 🗃️ Dataset

**Source**: [Kaggle - Video Game Sales](https://www.kaggle.com/code/upadorprofzs/eda-video-game-sales/input)  
**File**: `vgsales.csv`  
**Records**: ~16.598 game

**Kolom Dataset:**
| Kolom | Tipe | Deskripsi |
|-------|------|-----------|
| `Name` | string | Nama game |
| `Platform` | string | Platform (PS4, Xbox, Wii, dll) |
| `Year` | float | Tahun rilis |
| `Genre` | string | Genre game |
| `Publisher` | string | Publisher game |
| `NA_Sales` | float | Penjualan di North America (juta) |
| `EU_Sales` | float | Penjualan di Europe (juta) |
| `JP_Sales` | float | Penjualan di Japan (juta) |
| `Other_Sales` | float | Penjualan di region lain (juta) |
| `Global_Sales` | float | Total penjualan global (juta) |

---

## 🚀 Cara Menjalankan

### 1. Clone & Install Dependencies

```bash
# Masuk ke folder project
cd game_recommender

# Install dependencies
pip install -r requirements.txt
```

### 2. Download Dataset

1. Buka: https://www.kaggle.com/code/upadorprofzs/eda-video-game-sales/input
2. Download file `vgsales.csv`
3. Simpan di folder `data/vgsales.csv`

### 3. Jalankan Aplikasi

```bash
streamlit run app.py
```

Aplikasi akan berjalan di: **http://localhost:8501**

---

## 🧠 Detail Model

### Feature Engineering

Fitur yang digunakan untuk model:

```python
features = [
    "Platform_enc",    # Label Encoded
    "Genre_enc",       # Label Encoded  
    "Publisher_enc",   # Label Encoded
    "Year",            # Tahun rilis
    "NA_Sales",        # Sales North America
    "EU_Sales",        # Sales Europe
    "JP_Sales",        # Sales Japan
    "Other_Sales",     # Sales Other
    "Global_Sales",    # Total sales
]
```

Semua fitur dinormalisasi menggunakan **MinMaxScaler**.

### Metode Rekomendasi

#### 1. Content-Based (Cosine Similarity)
```python
cosine_sim = cosine_similarity(feature_matrix, feature_matrix)
# Score range: 0.0 (tidak mirip) - 1.0 (sangat mirip)
```

#### 2. KMeans Clustering
```python
kmeans = KMeans(n_clusters=20, random_state=42)
# Game dalam cluster yang sama = kemungkinan mirip
```

#### 3. KNN
```python
knn = NearestNeighbors(n_neighbors=11, metric='cosine')
# Temukan 11 game terdekat dalam feature space
```

---

## 📊 Fitur Aplikasi

| Tab | Fitur |
|-----|-------|
| 🔍 Cari by Judul | Input nama game → dapat rekomendasi serupa + similarity chart |
| 🎭 Cari by Genre | Pilih genre → top game berdasarkan global sales |
| ⚙️ Filter Preferensi | Filter genre + platform + tahun → rekomendasi custom |
| 🔵 Cluster Analysis | Lihat cluster game + distribusi genre dalam cluster |
| 📊 EDA & Visualisasi | Analisis data: top games, trend sales, distribusi region |

---

## ⚡ Performa

- **Latency**: < 100ms per request (sesuai requirement)
- **Dataset**: 16.598 game → ~1.300 unique games (setelah dedup)
- **Akurasi**: Cosine Similarity berbasis fitur multidimensi

---

## 🛠️ Tech Stack

```
Python 3.10+
├── scikit-learn    # ML Models (KMeans, KNN, Cosine Similarity)
├── pandas          # Data manipulation
├── numpy           # Numerical computation
├── streamlit       # Web deployment
├── matplotlib      # Visualisasi
└── seaborn         # Statistical plots
```

---

## 📁 Version Control (Git)

```bash
git init
git add .
git commit -m "Initial commit: Game Recommendation System"
git branch -M main
git remote add origin https://github.com/username/game-recommender.git
git push -u origin main
```

---

## 👥 Cara Akses

- **Local Demo**: `streamlit run app.py` → http://localhost:8501
- **Shared Link**: Deploy ke Streamlit Cloud (https://share.streamlit.io)
- **API**: Dapat di-extend menggunakan FastAPI jika dibutuhkan

---

## 📝 Catatan Penting

- ✅ Hanya Classical ML (tidak ada neural network/deep learning)
- ✅ Deployment via Streamlit (web-based)
- ✅ Latency < 100ms
- ✅ Menerima input user, menjalankan model, dan menampilkan output jelas
- ✅ Version control: Git
- ✅ Library: NumPy, Pandas, Scikit-learn, Matplotlib, Seaborn

## Team
- Matthew Vandersen - 2802444813
- Melvin Hansen Zulkarnain - 2802444561
- Philip Andrew Susanto - 2802452443

Course: Machine Learning, BINUS University
