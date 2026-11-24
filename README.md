# Risk Factor Analysis and Patient Profiling for Early Stroke Detection

[English](README.md) | [Bahasa Indonesia](README.id.md)

Exploratory analysis of 5,109 patient records to find which attributes actually separate stroke cases
from non-cases — and which ones only look like they do.

**Stack:** Python · pandas · NumPy · Matplotlib · Seaborn · Jupyter · kagglehub
**Dataset:** [Stroke Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset) (Kaggle, `fedesoriano/stroke-prediction-dataset`)
**Deliverables:** [`notebooks/stroke_risk_profiling_eda.ipynb`](notebooks/stroke_risk_profiling_eda.ipynb) · [`reports/Analisis_Faktor_Risiko_Stroke_Report.pdf`](reports/Analisis_Faktor_Risiko_Stroke_Report.pdf)

---

## Project Overview & Business Problem

Stroke is a leading cause of death and permanent disability, and its treatment cost and response time
are high. Hospitals already hold demographic and medical-history data on every patient, but that data
sits idle until symptoms appear.

**Problem:** care facilities cannot proactively identify high-risk patients before a stroke occurs, so
intervention is late and expensive.

**Objective:** map the patient profile that drives stroke risk so medical teams can prioritise
screening, education, and preventive follow-up for the most vulnerable groups.

**Stakeholders:** hospital management, physicians (screening priority), and public-health or insurance
teams (prevention programs).

**Target variable:** `stroke` (binary — 1 = patient had a stroke, 0 = did not).

---

## Key Findings & Data Insights

Baseline stroke rate across the cleaned dataset: **4.87%** (249 of 5,109 patients).

| # | Finding | Evidence | Read |
|---|---|---|---|
| 1 | **Age is the primary driver** | 0.13% (<30) → 0.79% (30-44) → 4.83% (45-59) → 9.21% (60-74) → **19.69%** (75+) | 75+ sits at ~4x the baseline; risk crosses the average around age 45 |
| 2 | **Comorbidities compound** | 3.39% (neither) → 12.21% (hypertension only) → 16.04% (heart disease only) → **20.31%** (both) | Having both histories is ~6x the risk of having neither |
| 3 | **Glucose separates, BMI does not** | Glucose ≥200 mg/dL: **12.90%** vs 3.58% (<100 mg/dL). BMI medians of stroke and non-stroke patients overlap at ~28 | Glucose is a usable triage signal; BMI alone is not |
| 4 | **The smoking signal is age-confounded** | `formerly smoked` 7.92% vs `Unknown` 3.04%, but median age is 57 vs 23 | Not a causal signal — the `Unknown` group is mostly children and young adults |

**Operational implication:** start preventive education at 45, make 60+ (especially 75+) the priority
screening cohort, and flag patients with hypertension or heart disease automatically — screening on
glucose rather than BMI.

---

## Data Profiling & Cleaning Methodology

Raw data: **5,110 rows × 12 columns**. After cleaning: **5,109 rows × 12 columns**. All 249 stroke
cases preserved.

Profiling checks performed, each with code evidence in the notebook: missingness, full-row and ID
duplicates, category cardinality and spelling consistency, cross-column logic checks (under-18 married,
under-10 smokers, `children` work type age range), numeric distribution with IQR outlier bounds, and
target balance.

| # | Column | Issue found | Action | Why |
|---|---|---|---|---|
| 1 | `id` | Unique identifier, no clinical meaning | Dropped | No predictive value; pure noise for any model |
| 2 | `gender` | 1 row with `Other` (0.02%) | Dropped the row | Too small to analyse as a group; not a stroke case |
| 3 | `bmi` | 201 missing values (3.93%) | Added binary flag `bmi_is_missing`, then imputed the gender-group median (Female 27.8 / Male 28.4) | Patients with missing BMI show a **19.90%** stroke rate vs **4.26%** for recorded BMI — dropping them would destroy signal. Median chosen because BMI is right-skewed |
| 4 | `bmi` | 13 extreme values above 60 (max 97.6) | Capped at 60.0 | Clinically implausible, likely data-entry error; capping avoids discarding rows |
| 5 | `avg_glucose_level` | 627 IQR outliers (max 271.74 mg/dL) | Kept | Hyperglycaemia is a real condition and a stroke risk factor, not an error |
| 6 | `smoking_status` | 1,544 `Unknown` values (30.22%) | Kept as its own category | Too large to drop; forced imputation would bias the lifestyle analysis |

Also worth noting: the dataset is highly imbalanced (249 stroke vs 4,861 non-stroke), so any downstream
model must be evaluated on recall / PR-AUC rather than accuracy — a majority-class guess already scores
95%.

---

## Repository Structure

```
stroke-risk-analysis/
├── notebooks/
│   └── stroke_risk_profiling_eda.ipynb   # profiling → cleaning → EDA → findings
├── reports/
│   ├── Analisis_Faktor_Risiko_Stroke_Report.pdf
│   └── figures/                          # EDA charts (1-4)
├── requirements.txt
├── .gitignore
├── README.md                             # English
└── README.id.md                          # Bahasa Indonesia
```

No dataset files are committed: the notebook downloads the data at runtime through `kagglehub`.

---

## How to Reproduce / Run

```bash
git clone https://github.com/Sultvnnnn/stroke-risk-analysis.git
cd stroke-risk-analysis

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

jupyter notebook notebooks/stroke_risk_profiling_eda.ipynb
```

Then **Restart & Run All**. The notebook pulls the dataset automatically, so no manual download or
Kaggle API key is needed for this public dataset.

---

## Next Steps

Planned progression from descriptive analysis to **predictive modeling**:

1. **Baseline models** — logistic regression for interpretable coefficients; XGBoost/LightGBM as the
   performance ceiling.
2. **Class imbalance handling** — class weights, SMOTE applied strictly inside cross-validation folds
   (never before the split), and decision-threshold tuning on the precision-recall curve.
3. **Evaluation** — ROC-AUC, PR-AUC, and recall at a clinically acceptable precision level, with
   stratified k-fold cross-validation and a held-out test set that is never resampled.
4. **Feature engineering** — validate `bmi_is_missing` as a standalone predictor, encode smoking status
   as ordinal, add age × comorbidity interaction terms.
5. **Explainability** — SHAP values to confirm the drivers found in this EDA and to justify individual
   risk scores to clinicians.

---

**Author:** Sultan Abdul Fatah — [@Sultvnnnn](https://github.com/Sultvnnnn)
