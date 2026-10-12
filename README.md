# Egypt Car Market Analysis

Exploratory machine-learning analysis of ~762k Egyptian vehicle registrations: body-type and brand classification, quarterly market forecasting, governorate clustering, and brand-relationship analysis.

![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebooks-F37626?logo=jupyter&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikitlearn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-Gradient%20Boosting-189FDD)
![pandas](https://img.shields.io/badge/pandas-Data-150458?logo=pandas&logoColor=white)

## Overview

This project explores a large, preprocessed table of car registrations in Egypt (one row per registration, with brand, model, body type, quarter, year, governorate and traffic unit). Five Jupyter notebooks each ask a different question of the same data:

| Notebook | Question | Method |
|---|---|---|
| `body_type_by_traffic_unit.ipynb` | Can body type be predicted from brand/model/time/traffic unit, and which traffic units matter most? | Random Forest classifier + traffic-unit (`TU_*`) feature importances |
| `quarterly_market_forecast.ipynb` | Can total quarterly registrations be forecast from previous quarters? | Aggregation to quarterly counts, 4 lag features, XGBoost regressor |
| `brand_classifier_xgboost.ipynb` | Can the brand be predicted from location, time and body type (without the model ID)? | Multi-class XGBoost classifier |
| `governorate_clustering.ipynb` | Do governorates group into market segments? | Per-governorate aggregation, StandardScaler, PCA (95% variance), KMeans vs Agglomerative (Ward) clustering |
| `brand_cannibalization_analysis.ipynb` | Do top brands take market share from each other? | ADF stationarity tests, VAR / Granger causality attempts, correlations, lead-lag correlations, market-share changes |

Each notebook is self-contained: it loads the CSV, builds its own features, trains or analyses, plots, and (for the modelling notebooks) saves the fitted model.

## Tech stack

- **Python 3.11** (the notebooks were run on Python 3.11 and 3.12 kernels)
- **pandas / NumPy**: loading the ~1 GB CSV, one-hot aggregation, groupby time series
- **scikit-learn**: Random Forest, KMeans, Agglomerative clustering, PCA, StandardScaler, metrics
- **XGBoost**: gradient-boosted trees for the brand classifier and the quarterly regressor
- **statsmodels / SciPy**: ADF tests, VAR, Granger causality, Pearson correlation with p-values, linear regression
- **Matplotlib / Seaborn**: confusion matrices, feature-importance bars, elbow/silhouette plots, market-share charts
- **pickle / joblib**: saving fitted models and the scaler

## Demo / Screenshots

<!-- TODO: add a screenshot at docs/screenshot.png -->
![Screenshot](docs/screenshot.png)

The notebooks already contain their rendered plots in the saved outputs (open them on GitHub to view). For this README, export one representative figure, for example the "Top 20 Traffic Unit Feature Importances" bar chart from `body_type_by_traffic_unit.ipynb` or the governorate segment bar chart from `governorate_clustering.ipynb`, and save it as `docs/screenshot.png`.

## Dataset

**The dataset is not included in this repository.**

- **Content:** Egyptian car registration records, already preprocessed (label-encoded and one-hot encoded).
- **Size / format:** a single CSV, `car_sales_preprocessed_2.csv`, about 1.07 GB, **761,973 rows x 236 columns** (shape printed in the notebook outputs).
- **Columns:**
  - `quarter` (1-4)
  - `Brand_LE` (93 brands), `Model_LE`, `Body Type_LE`: label-encoded integer IDs (the mapping back to names is not part of this repo)
  - `Year_2022`, `Year_2023`: boolean dummies. 66,325 rows have both set to False, i.e. they belong to a third year that was dropped as the reference category
  - `Gov_*`: 26 one-hot governorate columns (Arabic names)
  - `TU_*`: 204 one-hot traffic-unit columns (Arabic names)
- **Source / license:** [TODO: confirm source] / [TODO: confirm license]

Put the file here (the notebooks read `../data/car_sales_preprocessed_2.csv` relative to `notebooks/`):

```
egypt-car-market-analysis/
├── data/
│   └── car_sales_preprocessed_2.csv
└── notebooks/
```

**Models:** no trained models are included. Running the notebooks writes them to `models/` (git-ignored):

| File | Produced by |
|---|---|
| `models/random_forest_tu_classifier.pkl` | `body_type_by_traffic_unit.ipynb` |
| `models/total_market_forecaster_xgb.pkl` | `quarterly_market_forecast.ipynb` |
| `models/xgb_top_brand.joblib` | `brand_classifier_xgboost.ipynb` |
| `models/gov_scaler.joblib`, `models/gov_clustering_model.joblib` | `governorate_clustering.ipynb` |

## Installation

```bash
git clone https://github.com/aliyahseri/egypt-car-market-analysis.git
cd egypt-car-market-analysis
python -m venv .venv
```

Activate the environment:

```bash
# Windows (PowerShell)
.venv\Scripts\Activate.ps1
# macOS / Linux
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

1. Place `car_sales_preprocessed_2.csv` in `data/` (see [Dataset](#dataset)).
2. Start Jupyter from the repository root and open any notebook in `notebooks/`:

   ```bash
   jupyter notebook notebooks/
   ```

3. Run all cells. The notebooks use relative paths (`../data/`, `../models/`), so the kernel's working directory must be `notebooks/` (the default when you open a notebook from Jupyter).

Memory note: every notebook loads the full 1.07 GB CSV with 236 columns into pandas; expect to need several GB of RAM. The Random Forest (571,479 training rows x 209 features) and the 93-class XGBoost model are the slowest steps.

## Results and limitations

All numbers below are copied from the saved notebook outputs. They describe what the experiments showed, including where they fall short.

**1. Body type from traffic unit (`body_type_by_traffic_unit.ipynb`)**
- Random Forest, 25% stratified hold-out: accuracy **0.9965**, macro precision 0.8816, macro recall 0.8756, macro F1 0.8786.
- The high accuracy is not evidence that location predicts body type: `Model_LE` is a feature, and a car model almost always has a single body type, so the model is largely looking it up (target leakage).
- The highest traffic-unit importances are only **0.0004-0.0006** each (the bottom ten round to 0.0000). The notebook's "rougher road / smoother road" interpretation of these importances is therefore weak and not supported by a dedicated test.

**2. Quarterly total-market forecast (`quarterly_market_forecast.ipynb`)**
- Aggregation yields 11 "quarters"; after 4 lags only **7 rows** remain (5 train, 2 test).
- Train RMSE 273.45 / MAE 212.70 vs **test RMSE 15,960.41 / MAE 14,404.00** on 2 test points: the model overfits, and with this little data the result is a proof of concept for the lag-feature pipeline, not a usable forecaster. Four of the 11 time points are artefacts (see Known issues).

**3. Brand classifier (`brand_classifier_xgboost.ipynb`)**
- XGBoost on location/time/body-type features (with `Model_LE` deliberately removed): **test accuracy 0.15697** (~15.7%) over 93 brands. Most brands get zero recall; the model mostly predicts the largest brands. Location and quarter alone carry little brand information.

**4. Governorate clustering (`governorate_clustering.ipynb`)**
- 26 governorates, 209 aggregated features, PCA to 95% variance, k = 3.
- Silhouette: **KMeans 0.0770 vs Agglomerative (Ward) 0.2386**; the Agglomerative labels are used for the Low/Mid/High sales segments. Even 0.24 indicates weak cluster structure, and the features are dominated by raw registration volume (Cairo: 234,455; Giza: 110,844).

**5. Brand cannibalization (`brand_cannibalization_analysis.ipynb`)**
- Grouping by `quarter` alone gives only 4 time points (3 after differencing), so the VAR model and Granger causality tests raise errors and produce no results. The correlations that are reported (e.g. 0.939 between brands 56 and 61, 95% CI [-0.229, 0.999]) are based on 4 points and are not statistically meaningful.

## Project structure

```
egypt-car-market-analysis/
├── data/
│   └── README.md                          # where to put the CSV (data itself is git-ignored)
├── models/                                # created by the notebooks (git-ignored)
├── notebooks/
│   ├── body_type_by_traffic_unit.ipynb    # Random Forest body-type classifier + TU importance
│   ├── quarterly_market_forecast.ipynb    # XGBoost regressor on lagged quarterly totals
│   ├── brand_classifier_xgboost.ipynb     # 93-class XGBoost brand classifier
│   ├── governorate_clustering.ipynb       # PCA + KMeans vs Agglomerative clustering
│   └── brand_cannibalization_analysis.ipynb  # ADF, VAR/Granger attempts, correlations, market share
├── requirements.txt
├── LICENSE
└── README.md
```

## Challenges & what I learned

- **Working with a wide, one-hot table at scale.** 762k rows x 236 mostly boolean columns needed converting booleans to integers for scikit-learn and reconstructing categorical labels (e.g. governorate via `idxmax`) for aggregation.
- **Target leakage.** The near-perfect body-type accuracy versus the ~15.7% brand accuracy (once `Model_LE` was removed) shows how much a single identifier column can dominate a result.
- **Small-sample time series.** Turning registrations into a time series exposed how few distinct periods the data covers; VAR and Granger tests fail outright, and a 4-lag XGBoost model has only 7 usable rows.
- **Choosing a clustering method by metric.** Comparing KMeans and Agglomerative clustering on silhouette score rather than assuming KMeans.
- [TODO: confirm] Any additional lessons the owner wants to highlight (e.g. how the preprocessing pipeline that produced the CSV was built).

## Known issues & limitations

- `quarterly_market_forecast.ipynb`: `np.select(..., default=0)` assigns `Year = 0` to the 66,325 rows of the dropped year, creating fake time points `01`-`04` that are sorted before 2022 and used as lags.
- `quarterly_market_forecast.ipynb`: comments say "6 train, 1 test", but the split is 5 train / 2 test.
- `body_type_by_traffic_unit.ipynb`: `Model_LE` is a feature, so accuracy reflects model-to-body-type lookup (leakage); TU importances (<= 0.0006) don't support the road-quality interpretation printed in the notebook.
- `body_type_by_traffic_unit.ipynb`: if the CSV is missing, the synthetic-data fallback is commented out, so the next line fails with `NameError: df`.
- `brand_classifier_xgboost.ipynb`: `use_label_encoder=False` is ignored by current XGBoost (warning in output); the final model-saving cell was never executed in the saved notebook.
- `governorate_clustering.ipynb`: despite the original "k-prototypes" name, no k-prototypes is used (KMeans and Agglomerative only).
- `governorate_clustering.ipynb`: `brand_cols` matches `Brand_LE`, so the "brand" feature is a sum of label-encoded IDs (meaningless), and `brand_diversity` is always 1.
- `governorate_clustering.ipynb`: `body_cols` looks for the prefix `BodyType_`, which never matches `Body Type_LE`, so `total_body_types` is 0 and `dominant_body_type` is "None" everywhere.
- `governorate_clustering.ipynb`: `idxmax` over the 26 `Gov_*` columns assigns rows of the dropped governorate to the first column (اسيوط): its count is 22,432 vs 14,685 rows actually flagged `Gov_اسيوط`.
- `governorate_clustering.ipynb`: the KMeans model is saved as `gov_clustering_model.joblib`, but the Agglomerative labels are the ones used.
- `brand_cannibalization_analysis.ipynb`: groups by `quarter` only (merging all years), giving 4 observations; VAR (`maxlags is too large`) and Granger (`Insufficient observations`) cells end in tracebacks.
- `brand_cannibalization_analysis.ipynb`: the first Granger cell calls `grangercausality_test` (misspelled; should be `grangercausalitytests`), it is never reached because VAR fails first.
- `brand_cannibalization_analysis.ipynb`: market-share changes are fractions printed with `:+.2f}%` (e.g. `+0.02%` actually means +2 percentage points); the y-axis label "Market Share (%)" has the same problem.
- `brand_cannibalization_analysis.ipynb`: the pandas lead-lag correlation (`iloc[:-1].corr(iloc[1:])`) aligns on index, so it correlates the same 2 overlapping quarters (always 1.000) instead of a real lag.
- `brand_cannibalization_analysis.ipynb`: comment says Brand 56 vs 61 is "strongest correlation found: -0.52", but the computed correlation is 0.939 (the -0.52 values were lead-lag correlations for other pairs).
- `brand_cannibalization_analysis.ipynb`: ADF tests on 4 points per brand are not meaningful; one cell is a duplicate of an earlier cell.
- All notebooks: brands, models and body types appear only as label-encoded IDs; without the encoder mapping the results can't be tied to real brand names.

## Future improvements

- Build the time index from the real year (including the dropped year) and quarter, and aggregate by year-quarter everywhere.
- Re-run the body-type experiment without `Model_LE` (and `Brand_LE`) to measure what location alone explains.
- Use a simpler, better-suited forecaster (e.g. seasonal naive or ARIMA) and evaluate with rolling-origin validation once more quarters are available.
- Fix the governorate feature engineering (body-type counts from `Body Type_LE`, brand counts per brand ID, reconstruct the dropped governorate) and normalise by volume so clusters reflect market mix rather than size.
- Save the clustering model that is actually used, and publish the label-encoder mappings so results are readable.

## Credits / Acknowledgements

- Dataset: Egyptian car registration data, [TODO: confirm source].
- Libraries: pandas, NumPy, scikit-learn, XGBoost, statsmodels, SciPy, Matplotlib, Seaborn.

## License

MIT, see [LICENSE](LICENSE).

## Author

**Aliyah Seri**
- GitHub: [aliyahseri](https://github.com/aliyahseri)
