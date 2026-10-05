# Model 2: IoT Transfer CNN Architecture

## Overview
This architecture is designed for lightweight deployment on edge IoT devices. It prioritizes minimal parameter count and fast inference speed, sacrificing some of the robustness of the full baseline model.

## Pipeline Architecture
1. **Lightweight Convolutional Blocks:** Consists of purely 1D Convolutions with Batch Normalization and ReLU activations.
2. **Channel Reduction:** Progressively scales down the channel dimensions (`in_channels -> 32 -> 64 -> 128 -> 128`) to keep memory footprint extremely small.
3. **Adaptive Pooling:** Reduces arbitrary sequence lengths to a fixed dimension before passing to the classifier.
4. **IoT Inference Head:** A simple 2-layer MLP classifier maps the pooled features to the multi-class outputs.

## Datasets
* **Mendeley In-Domain Dataset:** [https://data.mendeley.com/datasets/xw44wv2g88/2](https://data.mendeley.com/datasets/xw44wv2g88/2)
* **Hong Kong (HK) Zero-Shot Transfer Dataset:** Processed subset included locally in `/data/processed/test_data_hk.npz`
