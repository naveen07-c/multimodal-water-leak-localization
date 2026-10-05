# Model 0: Baseline (Full Proposed DAE-CNN-LSTM-GNN) Architecture

## Overview
This is the original full proposed multi-modal architecture. It is the most robust and complex model in the suite, utilizing spatial, temporal, and topological data to achieve highly accurate predictions even under extreme sparsity and noise conditions.

## Pipeline Architecture
1. **Denoising Autoencoders (DAE):** A standalone 1D-Conv DAE that filters realistic moving saw and traffic acoustic noise.
2. **Multi-Scale 1D-CNN:** Extracts deep spatial features from the raw accelerometer and hydrophone signals.
3. **Temporal Bi-LSTM:** Models time-series dependencies to capture transient flow and sequence-specific information.
4. **Topological Pipeline GNN:** Models the spatial physical network of the pipes, allowing it to withstand up to 50% sensor sparsity.
5. **Attention-Based Multimodal Fusion:** Dynamically weights the importance of different modalities (vibration vs acoustic).
6. **Ensemble Decision Layer:** Predicts Binary Detection, 5-Class Leak Geometry, and Spatial Localization simultaneously.

## Datasets
* **Mendeley In-Domain Dataset:** [https://data.mendeley.com/datasets/xw44wv2g88/2](https://data.mendeley.com/datasets/xw44wv2g88/2)
* **Hong Kong (HK) Zero-Shot Transfer Dataset:** Processed subset included locally in `/data/processed/test_data_hk.npz`
