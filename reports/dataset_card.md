# Dataset Card — CFRP (SmartManuAD)

## Source

- **Name:** CFRP (Carbon-Fiber Reinforced Polymer fabrication sensors)
- **Repository:** [SmartManuAD — Smart-Manufacturing-AD](https://github.com/SmartManuAD/Smart-Manufacturing-AD)
- **Format:** `.npz` (compressed NumPy) — features `X`, labels `y`
- **Local file:** `data/raw/5_cfrp.npz` (20.0 MB)
- **Size:** 52,268 samples × 49 features
- **Labels:** binary anomaly labels (real, not synthetic) — 52,006 nominal (0) vs 262 anomalies (1)
- **Anomaly rate:** 0.50% (highly imbalanced — informs metric choice and threshold selection)

## Why this dataset for the Internet of Production

Composite manufacturing — and CFRP in particular — sits at the intersection
of high added value and high sensor density. Fiber-placement processes are
monitored with force, temperature, position and laser sensors at high
sampling rates; a single defect can scrap an expensive part. This makes the
CFRP dataset a clean fit for the **Internet of Production** narrative:

- **Sensor → data → decision loop.** Anomaly detection on streaming sensor
  data is exactly the kind of closed-loop intelligence IoP promises.
- **Traceability.** Each anomaly label is tied to a process state, which
  illustrates how digital twins and process histories can be enriched with
  ML insight.
- **Sustainability.** Detecting defects early avoids scrapping carbon-fiber
  parts, which carry a large embodied energy cost.

*(To be expanded by the student with personal reflection.)*

## Dataset properties

| Property | Value |
|---|---|
| Samples | 52,268 |
| Features | 49 |
| Feature dtype | float64 (integer-valued, range observed: 0 – 614) |
| Feature semantics | anonymized fiber-placement sensor channels — per-feature units not documented in the SmartManuAD release |
| Target | binary anomaly label (0 = nominal, 1 = anomaly) |
| Class balance | 99.50% nominal / 0.50% anomalies (262 positives) |
| Missing values | none (no NaN, no ±inf) |
| Constant / all-zero columns | none |

## Preprocessing decisions

- **Missing values:** none present in CFRP, so imputation is a no-op safeguard (median) rather than a real fix.
- **Scaling:** `RobustScaler` — features are heavy-tailed (medians ~15–54 with maxima reaching 614), so median/IQR-based scaling avoids letting outliers dominate.
- **Leakage avoidance:** imputer and scaler are fit on the training split only, then applied to validation and test (`src/preprocessing.prepare`).
- **Class imbalance:** 0.5% positives means accuracy is misleading. Evaluation focuses on **ROC-AUC**, **PR-AUC**, and F1 at a threshold tuned on the validation split.

## Train / validation / test split

- Ratio: **70 / 20 / 10** (stratified on the anomaly label).
- Implemented in `src/data_loader.py::split_70_20_10`.
- Random seed: `42` (see `config.yaml`).

## Reflection on transparency, traceability, sustainability (Lernziel 6)

*(To be filled by the student.)*

- **Transparency:** what does each feature mean, and can a domain expert audit a model decision?
- **Traceability:** how does the run + dataset version + code commit tie together (MLflow + Git)?
- **Sustainability:** what energy / scrap savings does early anomaly detection unlock in CFRP production?
