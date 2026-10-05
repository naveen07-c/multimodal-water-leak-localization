# Model 3: SAFNet (Self-Attention Fusion Network) Architecture

## Overview
SAFNet is a modern transformer-inspired architecture that relies exclusively on Self-Attention mechanisms rather than convolutions or recurrent units to find long-range acoustic dependencies in the leak signal.

## Pipeline Architecture
1. **Linear Projection:** Maps the raw multimodal input sequence into a higher-dimensional embedding space.
2. **Multi-Head Self Attention (MHSA):** Uses multiple attention heads to globally correlate distant parts of the 8000-length sequence, bypassing the bottleneck of sequential recurrent networks.
3. **Layer Normalization & Residuals:** Standard transformer blocks ensure stable gradients across the attention mechanisms.
4. **Global Average Pooling:** Flattens the attention-weighted sequence into a single context vector.
5. **Dense Output Head:** Predicts the final 5-class geometrical leak type and binary detection status.

## Datasets
* **Mendeley In-Domain Dataset:** [https://data.mendeley.com/datasets/xw44wv2g88/2](https://data.mendeley.com/datasets/xw44wv2g88/2)
* **Hong Kong (HK) Zero-Shot Transfer Dataset:** Processed subset included locally in `/data/processed/test_data_hk.npz`
