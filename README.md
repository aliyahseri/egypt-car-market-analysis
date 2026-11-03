# Egypt Car Market Analysis

## Introduction
This project explores about 762,000 Egyptian car registration records with machine learning. It classifies body types and brands, forecasts quarterly market size, groups governorates by their sales patterns, and analyzes relationships between brands.

## Models Used
* Random Forest: body type classification
* XGBoost: brand classification and quarterly market forecasting
* K-Means and Agglomerative clustering: governorate segmentation
* VAR and Granger causality: brand relationship analysis

## Notebooks
* `notebooks/body_type_by_traffic_unit.ipynb`
* `notebooks/brand_classifier_xgboost.ipynb`
* `notebooks/quarterly_market_forecast.ipynb`
* `notebooks/governorate_clustering.ipynb`
* `notebooks/brand_cannibalization_analysis.ipynb`

## Dataset
Preprocessed Egyptian vehicle registration data (CSV, 236 columns), placed at `data/car_sales_preprocessed_2.csv`.

## Requirements
* python 3.10+
* pandas
* numpy
* scikit-learn
* xgboost
* statsmodels
* matplotlib
* jupyter
