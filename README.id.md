# Analisis Faktor Risiko dan Profil Pasien untuk Deteksi Dini Penyakit Stroke

[English](README.md) | [Bahasa Indonesia](README.id.md)

Analisis eksploratif atas 5.109 rekam pasien untuk menemukan atribut mana yang benar-benar membedakan
pasien stroke dari yang bukan — dan mana yang hanya terlihat membedakan.

**Stack:** Python · pandas · NumPy · Matplotlib · Seaborn · Jupyter · kagglehub
**Dataset:** [Stroke Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset) (Kaggle, `fedesoriano/stroke-prediction-dataset`)
**Berkas utama:** [`notebooks/stroke_risk_profiling_eda.ipynb`](notebooks/stroke_risk_profiling_eda.ipynb) (Inggris) · [`notebooks/stroke_risk_profiling_eda_original.ipynb`](notebooks/stroke_risk_profiling_eda_original.ipynb) (Indonesia, naskah pertama) · [`reports/Analisis_Faktor_Risiko_Stroke_Report.pdf`](reports/Analisis_Faktor_Risiko_Stroke_Report.pdf)

> **Dua versi notebook.** `stroke_risk_profiling_eda_original.ipynb` adalah naskah pertama analisis ini
> (bahasa Indonesia, redaksi asli). `stroke_risk_profiling_eda.ipynb` adalah analisis yang sama, ditulis
> ulang dalam bahasa Inggris. Logika kode identik di keduanya; yang berbeda hanya teks naratif dan label
> grafik.

---

## Gambaran Proyek & Masalah Bisnis

Stroke merupakan salah satu penyebab utama kematian dan disabilitas permanen, dengan biaya perawatan
tinggi dan kebutuhan respons cepat. Rumah sakit sebenarnya sudah menyimpan data demografi dan riwayat
medis setiap pasien, tetapi data tersebut baru dimanfaatkan setelah gejala muncul.

**Masalah:** fasilitas kesehatan sulit mengidentifikasi pasien berisiko tinggi secara proaktif sebelum
serangan stroke terjadi, sehingga intervensi medis terlambat dan beban biaya meningkat.

**Tujuan:** memetakan profil dan faktor risiko utama pemicu stroke agar tim medis dapat memprioritaskan
program skrining, edukasi, dan pemantauan preventif pada kelompok paling rentan.

**Stakeholder:** manajemen rumah sakit, dokter (prioritas skrining), serta tim kesehatan masyarakat atau
asuransi kesehatan (program preventif).

**Target analisis:** kolom `stroke` (biner — 1 = pasien mengalami stroke, 0 = tidak).

### Asal Proyek

Analisis ini bermula dari tugas kuliah eksplorasi data, lalu dikembangkan lebih lanjut menjadi proyek
utuh:

- Setiap keputusan pembersihan diubah menjadi log yang dapat diaudit, lengkap dengan jumlah baris
  sebelum dan sesudah, bukan diterapkan tanpa jejak.
- Missing value pada BMI diperlakukan sebagai sinyal informatif, bukan dibuang — kelompok BMI kosong
  memiliki stroke rate 19,90%, sehingga ketidaktersediaan data itu sendiri menjadi fitur
  (`bmi_is_missing`) alih-alih alasan menghapus baris.
- Temuan status merokok diuji lebih dulu terhadap usia sebagai variabel perancu sebelum ditarik
  kesimpulan, dan hasil uji itu mengubah interpretasinya sepenuhnya.
- Jalur menuju predictive modeling dirancang konkret, termasuk cara menangani class imbalance.

Naskah pertama disimpan apa adanya di
[`notebooks/stroke_risk_profiling_eda_original.ipynb`](notebooks/stroke_risk_profiling_eda_original.ipynb)
agar titik awal dan pekerjaan tambahannya dapat dibandingkan langsung.

---

## Temuan Utama & Insight Data

Rata-rata kejadian stroke pada data bersih: **4,87%** (249 dari 5.109 pasien).

| # | Temuan | Bukti | Makna |
|---|---|---|---|
| 1 | **Usia adalah pemicu utama** | 0,13% (<30) → 0,79% (30-44) → 4,83% (45-59) → 9,21% (60-74) → **19,69%** (75+) | Kelompok 75+ mencapai ~4 kali rata-rata; risiko melewati rata-rata pada usia sekitar 45 |
| 2 | **Komorbiditas bersifat kumulatif** | 3,39% (tanpa keduanya) → 12,21% (hipertensi saja) → 16,04% (penyakit jantung saja) → **20,31%** (keduanya) | Memiliki kedua riwayat menaikkan risiko ~6 kali dibanding tanpa keduanya |
| 3 | **Glukosa membedakan, BMI tidak** | Glukosa ≥200 mg/dL: **12,90%** vs 3,58% (<100 mg/dL). Median BMI pasien stroke dan non-stroke berimpitan di ~28 | Glukosa layak jadi sinyal skrining; BMI tunggal tidak |
| 4 | **Sinyal merokok teracuni usia** | `formerly smoked` 7,92% vs `Unknown` 3,04%, tetapi median usia 57 vs 23 | Bukan hubungan sebab-akibat — kelompok `Unknown` didominasi anak dan usia muda |

**Implikasi operasional:** mulai edukasi pencegahan pada usia 45, jadikan kelompok 60+ (khususnya 75+)
sebagai prioritas skrining rutin, dan tandai otomatis pasien dengan hipertensi atau penyakit jantung —
dengan skrining berbasis glukosa, bukan BMI.

---

## Metodologi Profiling & Pembersihan Data

Data mentah: **5.110 baris × 12 kolom**. Setelah pembersihan: **5.109 baris × 12 kolom**. Seluruh 249
kasus stroke tetap utuh.

Pemeriksaan profiling yang dilakukan, masing-masing dengan bukti kode di notebook: missing values,
duplikat baris penuh dan duplikat ID, kardinalitas serta konsistensi penulisan kategori, pemeriksaan
logika antar kolom (menikah di bawah 18 tahun, perokok di bawah 10 tahun, rentang usia `children`),
distribusi numerik dengan batas outlier IQR, dan proporsi kelas target.

| # | Kolom | Masalah | Tindakan | Alasan |
|---|---|---|---|---|
| 1 | `id` | Identifier unik, tanpa makna klinis | Kolom dihapus | Tidak punya nilai prediktif; hanya noise untuk model |
| 2 | `gender` | 1 baris `Other` (0,02%) | Baris dihapus | Terlalu kecil untuk dianalisis sebagai kelompok; bukan kasus stroke |
| 3 | `bmi` | 201 missing values (3,93%) | Dibuat flag biner `bmi_is_missing`, lalu diimputasi median per gender (Female 27,8 / Male 28,4) | Pasien dengan BMI kosong punya stroke rate **19,90%** vs **4,26%** pada BMI terisi — menghapus baris berarti kehilangan sinyal. Median dipilih karena distribusi BMI miring kanan |
| 4 | `bmi` | 13 nilai ekstrem di atas 60 (maks 97,6) | Capping pada 60,0 | Tidak wajar secara klinis dan berpotensi salah input; capping menghindari pembuangan baris |
| 5 | `avg_glucose_level` | 627 outlier IQR (maks 271,74 mg/dL) | Dipertahankan | Hiperglikemia adalah kondisi nyata dan justru faktor risiko stroke |
| 6 | `smoking_status` | 1.544 nilai `Unknown` (30,22%) | Dipertahankan sebagai kategori tersendiri | Terlalu besar untuk dihapus; imputasi paksa akan membiaskan analisis gaya hidup |

Catatan penting: dataset sangat tidak seimbang (249 stroke vs 4.861 non-stroke), sehingga model lanjutan
wajib dievaluasi dengan recall atau PR-AUC, bukan accuracy — menebak kelas mayoritas saja sudah memberi
akurasi 95%.

---

## Struktur Repositori

```
stroke-risk-analysis/
├── notebooks/
│   ├── stroke_risk_profiling_eda.ipynb           # versi portofolio (Inggris)
│   └── stroke_risk_profiling_eda_original.ipynb  # versi asli (Indonesia)
├── reports/
│   ├── Analisis_Faktor_Risiko_Stroke_Report.pdf
│   └── figures/                                  # grafik EDA (1-4)
├── requirements.txt
├── .gitignore
├── README.md                                     # English
└── README.id.md                                  # Bahasa Indonesia
```

Tidak ada berkas dataset yang di-commit: notebook mengunduh data secara otomatis lewat `kagglehub`.

---

## Cara Menjalankan Ulang

```bash
git clone https://github.com/Sultvnnnn/stroke-risk-analysis.git
cd stroke-risk-analysis

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

jupyter notebook notebooks/stroke_risk_profiling_eda.ipynb
```

Lalu jalankan **Restart & Run All**. Dataset ditarik otomatis, jadi tidak perlu mengunduh manual atau
menyiapkan API key Kaggle untuk dataset publik ini.

---

## Rencana Pengembangan

Langkah lanjutan dari analisis deskriptif menuju **predictive modeling**:

1. **Model dasar** — regresi logistik untuk koefisien yang mudah diinterpretasi; XGBoost/LightGBM
   sebagai batas atas performa.
2. **Penanganan class imbalance** — class weight, SMOTE yang hanya diterapkan di dalam fold
   cross-validation (tidak pernah sebelum split), serta tuning threshold pada kurva precision-recall.
3. **Evaluasi** — ROC-AUC, PR-AUC, dan recall pada level presisi yang dapat diterima secara klinis,
   dengan stratified k-fold cross-validation dan test set terpisah yang tidak pernah di-resample.
4. **Feature engineering** — validasi `bmi_is_missing` sebagai prediktor mandiri, encoding status
   merokok sebagai ordinal, penambahan interaksi usia × komorbiditas.
5. **Explainability** — nilai SHAP untuk mengonfirmasi faktor pemicu hasil EDA dan menjelaskan skor
   risiko individual kepada dokter.

---

**Penulis:** Sultan Abdul Fatah — [@Sultvnnnn](https://github.com/Sultvnnnn)
