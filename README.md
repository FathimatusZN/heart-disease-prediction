# 🫀 Heart Disease Prediction & Risk Pattern Analysis

> **Integrasi Random Forest dan Association Rule Mining untuk Prediksi dan Pola Risiko Penyakit Jantung**

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-1.0%2B-orange?logo=scikit-learn)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)

---

## 📋 Deskripsi Proyek

Proyek ini merupakan **Final Project Mata Kuliah Data Mining** yang mengembangkan kerangka kerja terintegrasi untuk prediksi risiko penyakit jantung. Proyek menggabungkan dua pendekatan utama:

- **Klasifikasi (Random Forest)** — prediksi risiko penyakit jantung pada tingkat individu
- **Association Rule Mining (Apriori)** — penemuan pola faktor risiko yang muncul bersama pada populasi berisiko tinggi

Dataset yang digunakan adalah [Heart Failure Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction) dari Kaggle, berisi **918 observasi pasien** dengan **11 fitur klinis**.

---

## 👥 Anggota Kelompok

| Nama | NPM |
|------|-----|
| Azzahra Rahmadani | 22082010155 |
| Fathimatus Zahrotun Nisa | 22082010156 |

> Sistem Informasi, Fakultas Ilmu Komputer  
> Universitas Pembangunan Nasional Veteran Jawa Timur

---

## 🎯 Hasil Utama

### Klasifikasi
| Model | Accuracy | Precision | Recall | F1-Score | AUC-ROC |
|-------|----------|-----------|--------|----------|---------|
| **Random Forest** ⭐ | **90.76%** | **92.08%** | **91.18%** | **91.63%** | **0.9481** |
| Logistic Regression | 89.13% | 90.20% | 90.20% | 90.20% | 0.9301 |
| SVM | 88.59% | 90.91% | 88.24% | 89.55% | 0.9250 |
| KNN | 88.04% | 87.74% | 91.18% | 89.42% | 0.9417 |

### Association Rule Mining
| Metrik | Nilai |
|--------|-------|
| Pasien Berisiko Tinggi | 423 (46.1%) |
| Frequent Itemsets | 1.960 |
| Association Rules | 1.598 |
| Avg. Confidence | 73.03% |
| Avg. Lift | 1.28 |
| Max Lift | 1.65 |

### Top 5 Fitur Terpenting (Random Forest)
| Fitur | Importance |
|-------|-----------|
| ST_Slope | 25.17% |
| ChestPain_ASY | 12.05% |
| ExerciseAngina | 9.41% |
| MaxHR | 8.36% |
| Oldpeak | 8.10% |

---

## 📁 Struktur Repositori

```
heart-disease-prediction/
│
├── notebooks/
│   └── heart_disease_prediction.ipynb      # Notebook utama (full pipeline)
│
├── data/
│   └── raw/
│       └── heart_failure_dataset.csv       # Dataset original dari Kaggle
│
├── models/
│   ├── best_classification_model.pkl       # Model Random Forest terbaik
│   ├── all_classification_models.pkl       # Semua model yang dilatih
│   └── scaler.pkl                          # StandardScaler untuk preprocessing
│
├── outputs/
│   ├── model_comparison.csv                # Perbandingan performa semua model
│   ├── feature_importance.csv              # Ranking feature importance RF
│   ├── frequent_itemsets.csv               # Frequent itemsets hasil Apriori
│   ├── association_rules.csv               # Semua association rules
│   ├── top_20_association_rules.csv        # 20 aturan terbaik berdasarkan lift
│   ├── arm_summary.csv                     # Ringkasan hasil ARM
│   ├── executive_summary.csv               # Ringkasan eksekutif keseluruhan
│   └── final_report.txt                    # Laporan akhir teks
│
├── reports/
│   └── FP DM Kel 13.pdf                    # Paper/laporan ilmiah proyek
│
├── requirements.txt                        # Daftar dependensi Python
└── README.md                               # Dokumentasi ini
```

---

## 🔄 Alur Penelitian (Pipeline)

```
Dataset (918 pasien)
        │
        ▼
┌────────────────────────┐
│  1. Data Understanding │  EDA, distribusi, korelasi, outlier detection
└────────────────────────┘
        │
        ▼
┌─────────────────────────────────────┐
│  2. Data Preparation (Klasifikasi)  │
│  • Handle invalid values            │  Cholesterol=0 → impute median
│  • Outlier capping (IQR)            │
│  • Feature engineering              │  5 fitur kategorikal baru
│  • Encoding (Label/OHE/Ordinal)     │
│  • Train-Test Split (80:20)         │
│  • StandardScaler                   │
└─────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────┐
│  3. Classification Modeling       │
│  • Random Forest (GridSearchCV)   │  ← Best model ⭐
│  • SVM (GridSearchCV)             │
│  • KNN (GridSearchCV)             │
│  • Logistic Regression            │
│  Evaluasi: Accuracy, F1, AUC-ROC  │
└───────────────────────────────────┘
        │
        ▼
┌────────────────────────────────────┐
│  4. Data Preparation (ARM)         │
│  • Identifikasi pasien high-risk   │  Pred=1 & Confidence ≥ 70%
│  • Transformasi ke format kategoris│  → 423 pasien, 282 transaksi bersih
│  • TransactionEncoder              │
└────────────────────────────────────┘
        │
        ▼
┌────────────────────────────────┐
│  5. Association Rule Mining    │
│  • Apriori (min_support=0.1)   │  → 1.960 frequent itemsets
│  • Generate rules              │  → 1.598 rules (conf≥0.6, lift≥1.2)
│  • Clinical interpretation     │
└────────────────────────────────┘
        │
        ▼
┌─────────────────────────────┐
│  6. Analysis & Reporting     │
│  • Business objectives check │
│  • Key insights              │
│  • Clinical recommendations  │
└─────────────────────────────┘
```

---

## 🚀 Cara Menjalankan

Notebook ini dibuat dan dijalankan menggunakan **Google Colab**.

### 1. Buka Notebook di Colab
Klik tombol di bawah atau buka langsung file `notebooks/heart_disease_prediction.ipynb` dari repo ini di Google Colab.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/FathimatusZN/heart-disease-prediction/blob/main/notebooks/heart_disease_prediction.ipynb)

### 2. Install Dependensi Tambahan
Library berikut sudah di-install otomatis di dalam notebook:
```bash
!pip install -q kagglehub
!pip install -q mlxtend
```
Library lainnya (`numpy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn`) sudah tersedia secara default di Colab.

### 3. Download Dataset
Dataset diunduh otomatis melalui `kagglehub` di dalam notebook. Atau unduh manual dari [sini](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction) dan letakkan di `data/raw/heart_failure_dataset.csv`.

### 4. Jalankan Semua Cell
Pilih **Runtime → Run all** di menu Colab.

> **Catatan:** GridSearchCV pada Random Forest menguji 576 kombinasi parameter × 5-fold CV = 2.880 fits. Proses ini membutuhkan waktu beberapa menit.

---

## 📦 Dependensi

| Library | Keterangan |
|---------|-----------|
| `numpy`, `pandas` | Manipulasi data |
| `matplotlib`, `seaborn` | Visualisasi |
| `scikit-learn` | Model ML & preprocessing |
| `mlxtend` | Apriori & TransactionEncoder |
| `kagglehub` | Download dataset dari Kaggle |

> `numpy`, `pandas`, `matplotlib`, `seaborn`, dan `scikit-learn` sudah pre-installed di Google Colab. Hanya `mlxtend` dan `kagglehub` yang perlu di-install manual.

---

## 📊 Dataset

| Atribut | Keterangan |
|---------|-----------|
| **Sumber** | [Kaggle - Heart Failure Prediction](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction) |
| **Ukuran** | 918 baris × 12 kolom |
| **Target** | `HeartDisease` (0 = Tidak, 1 = Ya) |
| **Prevalensi** | 55.3% positif penyakit jantung |
| **Lisensi** | Open Database License (ODbL) |

### Fitur Dataset
| Fitur | Tipe | Deskripsi |
|-------|------|-----------|
| Age | int | Usia pasien (tahun) |
| Sex | object | Jenis kelamin (M/F) |
| ChestPainType | object | Tipe nyeri dada (ATA/NAP/ASY/TA) |
| RestingBP | int | Tekanan darah istirahat (mmHg) |
| Cholesterol | int | Kolesterol serum (mg/dl) |
| FastingBS | int | Gula darah puasa > 120 mg/dl (0/1) |
| RestingECG | object | Hasil EKG istirahat |
| MaxHR | int | Detak jantung maksimum |
| ExerciseAngina | object | Angina akibat olahraga (Y/N) |
| Oldpeak | float | Depresi ST akibat olahraga |
| ST_Slope | object | Kemiringan segmen ST puncak |
| **HeartDisease** | int | **Target** (0/1) |

---

## 🔍 Temuan Utama

### Pola Risiko Tertinggi (Lift = 1.65)
```
IF  {Usia Menengah + ECG Normal + MaxHR Rendah + Angina saat Olahraga}
THEN {ST Slope Datar + GulaDarahPuasa Normal + Nyeri Dada Asimptomatik}

Support: 10.64% | Confidence: 90.91% | Lift: 1.65
```

### Insight Klinis
- **63.09%** dari total predictive power berasal dari 5 fitur teratas
- **91.7%** pasien high-risk adalah laki-laki (vs 68.1% pada grup risiko rendah)
- **64.2%** pasien asimptomatik menunjukkan bukti iskemia objektif dari stress test
- Rata-rata usia pasien high-risk: **56.2 tahun** (vs 51.2 tahun pada risiko rendah)

---

## ✅ Pencapaian Business Objectives

### Klasifikasi
| Objektif | Target | Hasil | Status |
|----------|--------|-------|--------|
| Accuracy | ≥ 85% | 90.76% | ✅ |
| Precision | ≥ 80% | 92.08% | ✅ |
| Recall | ≥ 85% | 91.18% | ✅ |
| F1-Score | ≥ 82% | 91.63% | ✅ |
| AUC-ROC | > 0.85 | 0.9481 | ✅ |

### Association Rule Mining
| Objektif | Target | Hasil | Status |
|----------|--------|-------|--------|
| Jumlah Rules | ≥ 10 | 1.598 | ✅ |
| Min Support | ≥ 0.1 | 0.1028 | ✅ |
| Min Confidence | ≥ 0.6 | 0.6000 | ✅ |
| Min Lift | ≥ 1.2 | 1.2000 | ✅ |

---

## 📚 Referensi

- Breiman, L. (2001). Random forests. *Machine Learning*, 45(1), 5–32.
- Agrawal, R., & Srikant, R. (1994). Fast algorithms for mining association rules. *VLDB*, 487–499.
- Fedesoriano. (2021). Heart Failure Prediction Dataset. Kaggle. https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction
- World Health Organization. (2021). Cardiovascular diseases (CVDs). WHO Fact Sheets.

---

## 📝 Lisensi

Proyek ini dibuat untuk keperluan akademis. Dataset menggunakan lisensi [Open Database License (ODbL)](https://opendatacommons.org/licenses/odbl/1-0/).
