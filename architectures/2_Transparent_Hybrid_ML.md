# Model 1: Transparent Hybrid ML Architecture

## Overview
This architecture replaces black-box deep learning operations with traditional, explainable machine learning. It trades off slight accuracy margins for maximal interpretability and computational efficiency.

## Pipeline Architecture
1. **Statistical Feature Engineering:** Extracts hard-coded temporal and spectral statistics directly from the flattened signal windows (mean, standard deviation, max, min, energy).
2. **Random Forest Classifier:** Uses a highly tuned ensemble of 200 decision trees to perform classification on the engineered features.
3. **SHAP Explainability:** A `TreeExplainer` maps the exact contribution of each statistical feature toward the final decision, allowing operators to understand *why* a leak was detected.

## Datasets
* **Mendeley In-Domain Dataset:** [https://data.mendeley.com/datasets/xw44wv2g88/2](https://data.mendeley.com/datasets/xw44wv2g88/2)
* **Hong Kong (HK) Zero-Shot Transfer Dataset:** Processed subset included locally in `/data/processed/test_data_hk.npz`
