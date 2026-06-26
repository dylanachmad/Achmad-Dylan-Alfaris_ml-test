# 🚍 AI Engineer Technical Test – Travel Time Prediction per Segment - Achmad Dylan Alfaris

## Overview

Project ini merupakan solusi untuk **Technical Test AI Engineer** dengan tujuan membangun model Machine Learning untuk memprediksi **traveling_time_sec** pada setiap segmen perjalanan bus Transjakarta.

Pipeline mencakup:

- Exploratory Data Analysis (EDA)
- Data Cleaning
- Feature Engineering
- Machine Learning Modeling
- Ensemble Learning
- Model Evaluation

Model yang diuji:

- XGBoost
- CatBoost
- LSTM
- Ensemble XGBoost + LSTM
- Ensemble CatBoost + LSTM
- Ensemble XGBoost + CatBoost + LSTM

---

# Dataset

Dataset terdiri dari data histori perjalanan bus per segmen.

Target prediksi:

```
traveling_time_sec
```

Beberapa atribut penting:

- bus_body_no
- segment_id
- route_code
- trip_id
- stop_sequence
- traveling_time_sec
- average_time_sec
- arrival_time
- from_arrival_time_str

---

# Project Structure

```
Achmad-Dylan-Alfaris_ml-test/
│
├── README.md
│
├── assets/
│   ├── deviation_ratio_dist.png
│   ├── skewness_comparison.png
│   └── sparsity_segment_hour.png
│
├── writeup/
│   └── Writeup_Analitis.pdf
│
├── notebooks/
│   ├── 01_EDA_exploration.ipynb
│   └── 02_main_pipeline.ipynb
│
├── data/
│   └── AI_Engineer_dataset.parquet
│
├── output/
│   ├── featured_segments.parquet
│   ├── test_predictions.parquet
│   ├── model_comparison_overall.csv
│   ├── mae_per_trip.csv
│   └── mape_per_segment.csv
│
├── models/
│   ├── xgb_model.pkl
│   ├── catboost_model.cbm
│   └── lstm_model.keras
│
└── requirements.txt
```

---

# Cara Menjalankan Project

## 1. Clone Repository

```bash
git clone <repository-url>
cd AI_Engineer_Transjakarta
```

---

## 2. Install Dependency

```bash
pip install -r requirements.txt
```

---

## 3. Siapkan Dataset

Letakkan dataset pada folder:

```
data/
```

dengan nama:

```
AI_Engineer_dataset.parquet
```

---

## 4. Jalankan Notebook

Notebook pertama (opsional)

```
notebooks/01_EDA_exploration.ipynb
```

Notebook ini berisi:

- eksplorasi data
- investigasi kualitas data
- analisis distribusi
- analisis outlier
- analisis sparsity
- analisis trip incompleteness

---

Notebook utama

```
notebooks/02_main_pipeline.ipynb
```

Notebook ini menjalankan seluruh pipeline mulai dari:

1. Load Dataset
2. Data Cleaning
3. Rekonstruksi Real Trip
4. Feature Engineering
5. Train/Test Split
6. Training XGBoost
7. Training CatBoost
8. Training LSTM
9. Ensemble Prediction
10. Model Evaluation
11. Save Model
12. Save Prediction

---

# Feature Engineering

Feature tambahan yang digunakan antara lain:

| Feature | Deskripsi |
|----------|-----------|
| hour | Jam perjalanan |
| day_of_week | Hari dalam seminggu |
| is_weekend | Weekend / Weekday |
| is_peak_hour | Jam sibuk |
| time_bucket | Kelompok waktu perjalanan |
| deviation_ratio | Rasio deviasi terhadap rata-rata historis |
| anomaly_suspected | Indikasi anomali |
| is_gap_suspected | Indikasi trip tidak lengkap |
| prev_deviation_ratio | Deviasi segmen sebelumnya |
| prev_traveling_time_sec | Travel time segmen sebelumnya |
| trip_progress | Posisi kendaraan pada trip |
| segment_time_estimate | Estimasi historis tiap segmen |
| bus_avg_deviation | Deviasi historis tiap armada |

---

# Asumsi dan Threshold

## Rekonstruksi Trip

Trip baru diasumsikan dimulai ketika:

```
stop_sequence < previous_stop_sequence - 2
```

---

## Gap Detection

```
is_gap_suspected = True
```

apabila terdapat loncatan nomor halte lebih dari satu dalam satu perjalanan.

---

## Trip Incomplete

Trip dianggap tidak dimulai dari halte pertama apabila:

```
first_stop_sequence > 1
```

---

## Hierarchical Backoff

Historical statistics menggunakan tiga level:

1. segment + hour + weekend
2. segment + hour
3. segment

Minimum jumlah sampel:

```
MIN_SAMPLES_LEVEL1 = 20
```

Apabila jumlah sampel kurang dari threshold tersebut maka sistem melakukan fallback ke level berikutnya.

---

## Peak Hour

Jam sibuk didefinisikan sebagai:

```
06.00 – 08.59
16.00 – 19.59
```

---

## Time Bucket

- Early Dawn
- Morning Peak
- Midday
- Evening Peak
- Night

---

## Ensemble Strategy

Eksperimen dilakukan menggunakan:

- XGBoost
- CatBoost
- LSTM

serta beberapa kombinasi ensemble menggunakan **Weighted Average**.

Model terbaik dipilih berdasarkan kombinasi performa MAE dan RMSE.

---

# Output

Pipeline menghasilkan file berikut.

| File | Keterangan |
|--------|------------|
| featured_segments.parquet | Dataset hasil feature engineering |
| test_predictions.parquet | Hasil prediksi data test |
| model_comparison_overall.csv | Perbandingan performa model |
| mae_per_trip.csv | MAE tiap putaran perjalanan |
| mape_per_segment.csv | MAPE tiap segmen |

Model yang telah dilatih disimpan pada folder:

```
models/
```

---

# Evaluation Result

| Model | MAE (detik) | RMSE (detik) |
|------|------------:|------------:|
| Naive | 312.22 | 2266.46 |
| XGBoost | 105.35 | 1206.41 |
| CatBoost | 110.23 | 1326.27 |
| LSTM | 108.66 | 1133.04 |
| Ensemble CatBoost + LSTM | 103.61 | 1080.29 |
| Ensemble XGBoost + CatBoost + LSTM | 99.88 | 1041.38 |
| **Ensemble XGBoost + LSTM** | **99.88** | **1027.93** |

---

## Best Model

Model terbaik yang diperoleh adalah:

**Ensemble XGBoost + LSTM**

Hasil evaluasi:

| Metric | Value |
|---------|------:|
| MAE | 99.88 detik |
| RMSE | 1027.93 detik |
| MAPE Segment | 45.07% |
| MAE Putaran | 1670.71 detik |
| Improvement vs Baseline | 68.0% |

Walaupun ensemble tiga model menghasilkan MAE yang hampir identik, kombinasi **XGBoost + LSTM** dipilih sebagai model akhir karena memberikan nilai RMSE yang lebih rendah sehingga lebih baik dalam mengurangi kesalahan prediksi yang besar.

---

# Author

Achmad Dylan Alfaris - MT Transjakarta Candidate
