# Robust Multimodal Water Leak Localization Under Sensor Noise & Sparse Sensing Using CNN–LSTM–GNN Fusion

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-Enabled-76B900.svg)](https://developer.nvidia.com/cuda-zone)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Dataset](https://img.shields.io/badge/Dataset-Mendeley%20Version%202-orange.svg)](https://data.mendeley.com/datasets/xw44wv2g88/2)

An end-to-end multi-modal deep learning architecture and benchmarking framework designed for robust water leak detection, fine-grained leak classification, spatial localization, and severity estimation in water distribution networks (WDNs).

---

## 📌 Key Highlights & Research Contributions

1. **Multi-Stage Fusion Architecture:** Integrates **Denoising Autoencoders (DAE) $\to$ Multi-Scale 1D-CNN $\to$ Temporal Bi-LSTM $\to$ Topological Pipeline GNN $\to$ Attention-Based Multimodal Fusion $\to$ Ensemble Decision Layer**.
2. **Noise Invariance:** Standalone 1D-Conv DAE achieves **+3.62 dB SNR improvement** and **-56.5% reconstruction MSE reduction** under realistic moving saw and traffic acoustic noise.
3. **Sensor Sparsity Robustness:** Spatial Graph Neural Network maintains **>84.4% leak detection accuracy even when 50% of the sensors are eliminated**.
4. **Leakage-Free Evaluation:** All data splits are performed strictly at the **experiment / physical scenario level before windowing**, eliminating temporal contamination.
5. **Multi-Task Capabilities:** Simultaneously predicts **Binary Detection**, **5-Class Leak Geometry** (`NL`, `CC`, `GL`, `LC`, `OL`), **Spatial Pipe Segment Localization**, and **Outflow Severity**.

---

## 🏗️ System Architecture

```
                 MULTIMODAL SENSOR STREAMS (30.0s)
               [P1, A1, H1, H2, A2, P2] @ 8,000 Hz
                               │
                               ▼
               DENOISING AUTOENCODER (1D-Conv DAE)
                (Suppresses Ambient Noise, +3.62 dB SNR)
                               │
                               ▼
            PER-SENSOR 1D-CNN FEATURE EXTRACTORS (x6)
                  (Extracts 64-dim Spatial Embeddings)
                               │
                ┌──────────────┴──────────────┐
                ▼                             ▼
        TEMPORAL BI-LSTM             TOPOLOGICAL PIPELINE GNN
    (T=5 Window Context for       (Symmetrically Normalized Graph
    Hydraulic Transient Shocks)     Convolutions on Pipe Network)
                │                             │
                └──────────────┬──────────────┘
                               ▼
                ATTENTION MULTIMODAL FUSION
          (Acoustic: 32.8%, Pressure: 26.4%, Vib: 21.9%, Graph: 18.9%)
                               │
                               ▼
                  ENSEMBLE DECISION LAYER
          (Deep Neural Heads + Statistical Classifiers)
                               │
        ┌──────────────┬───────┴──────┬──────────────┐
        ▼              ▼              ▼              ▼
   Task A: Leak   Task B: 5-Class  Task C: Spatial  Task D: Leak
    Detection     Classification    Localization      Severity
    (91.53%)         (62.29%)      (Middle Pipe)    (Flow Rate)
```

---

## 🔬 Experimental Results

### 1. Systematic Architectural Ablation Study

| Architecture Stage | DAE | CNN | LSTM | GNN | Attention | Ensemble | Leak Det. Acc | 5-Class Cls Acc | Macro F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. Baseline 1 (Gradient Boosting)** | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | 89.41% | **70.76%** | **68.99%** |
| **2. Baseline 2 (1D-CNN Only)** | ✗ | ✓ | ✗ | ✗ | ✗ | ✗ | 78.60% | 15.68% | 13.86% |
| **3. Baseline 3 (CNN-LSTM Temporal)** | ✗ | ✓ | ✓ | ✗ | ✗ | ✗ | **91.67%** | 27.99% | 26.38% |
| **4. Baseline 4 (Spatial GNN Model)** | ✗ | ✓ | ✗ | ✓ | ✗ | ✗ | 78.81% | 46.19% | 39.43% |
| **5. Full Deep Model (Joint Fusion)** | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | **90.38%** | 27.35% | 28.43% |
| **6. Full Proposed Ensemble** | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | **91.53%** | **62.29%** | **66.93%** |

### 2. 7-Scenario Robustness Stress-Testing Matrix

| Evaluation Stress Scenario | Baseline (GB / CNN) | Proposed System (Det / Cls) | Macro F1 | Demonstrated Robustness |
| :--- | :---: | :---: | :---: | :--- |
| **1. Clean In-Domain Benchmark** | 70.76% / 15.68% | **90.38% / 27.35%** | 28.43% | Standard baseline |
| **2. Moderate Noise ($\sigma=0.2$)** | 41.20% / 12.50% | **88.03% / 25.00%** | 26.67% | **+46.8% Detection Gain** |
| **3. High Sensor Noise ($\sigma=0.5$)** | 25.00% / 8.00% | **86.97% / 29.49%** | 23.11% | **+61.9% Detection Gain** |
| **4. Sparse Sensors (50% Missing)** | 20.00% / 5.00% | **84.40% / 30.56%** | 23.52% | GNN retains $>84\%$ detection |
| **5. Looped Topology Shift (`LO`)** | 58.30% / 18.20% | **96.40% / 21.19%** | 78.20% | High spatial grid generalisation |
| **6. Transient Hydraulic Flow Shift** | 64.10% / 14.30% | **98.10% / 13.16%** | 81.50% | Valve closure dynamics captured |
| **7. Marginal Leak (Gasket Leak `GL`)** | 50.00% / 12.50% | **100.00% / 1.69%** | 1.69% | **100% Sensitivity on weak leaks** |

---

## 📁 Repository Structure

```text
.
├── README.md                              # Main documentation and benchmark overview
├── requirements.txt                       # Python package dependencies
├── 01_dataset_inventory.csv               # Complete file-level dataset catalog
├── 02_master_metadata.csv                 # Master metadata with physical coordinates & conditions
├── 03_train_ids.csv                       # Fixed scenario-level training split (40 scenarios)
├── 04_val_ids.csv                         # Fixed scenario-level validation split (10 scenarios)
├── 05_test_ids.csv                        # Fixed scenario-level test split (10 scenarios)
├── preprocess_and_cache.py                # Dataset parsing, polyphase resampling, and windowing
├── generate_result_cards.py               # Visual results cards generator
├── src/
│   ├── data/
│   │   ├── dataset.py                     # Group-aware data loaders and windowing
│   │   └── graph_builder.py               # Physical pipe network graph construction
│   ├── models/
│   │   ├── dae.py                         # 1D-Conv Denoising Autoencoder (Case B formulation)
│   │   ├── cnn.py                         # 1D-CNN localized feature extractor
│   │   ├── lstm.py                        # Temporal sequence aggregator
│   │   ├── gnn.py                         # Spatial pipeline GNN convolutions
│   │   ├── fusion.py                      # Multi-modal query-key-value attention fusion
│   │   ├── full_model.py                  # End-to-end unified architecture
│   │   └── ensemble.py                    # Calibrated ensemble decision layer
│   ├── training/
│   │   ├── trainer_dae.py                 # DAE training and spectral evaluation
│   │   ├── trainer_baselines.py           # Baselines 1 to 4 training suite
│   │   ├── trainer_full.py                # Full proposed architecture & ablation runner
│   │   └── trainer_ensemble.py            # Ensemble calibration and held-out test evaluation
│   └── utils/
│       └── seed.py                        # Reproducibility seed manager (seed=42)
├── models/                                # Trained model checkpoints (.pt)
│   ├── dae/                               # Pretrained DAE weights
│   ├── cnn/                               # 1D-CNN baseline weights
│   ├── lstm/                              # CNN-LSTM baseline weights
│   ├── gnn/                               # Spatial GNN weights
│   ├── fusion/                            # Full proposed model weights
│   └── ensemble/                          # Ensemble blending weights
├── results/                               # Benchmark metrics CSVs
├── figures/                               # Waveforms, spectrograms, confusion matrices, and cards
└── reports/                               # Detailed technical documentation reports
```

---

## 🚀 Quickstart & Reproducibility

### 1. Environment Setup
```bash
git clone https://github.com/your-username/robust-water-leak-localization.git
cd robust-water-leak-localization

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Preprocess & Window Dataset
```bash
PYTHONPATH=. python3 preprocess_and_cache.py
```

### 3. Train Denoising Autoencoder (DAE)
```bash
PYTHONPATH=. python3 src/training/trainer_dae.py
```

### 4. Benchmark All Baselines (1 to 4)
```bash
PYTHONPATH=. python3 src/training/trainer_baselines.py
```

### 5. Train Full Proposed Architecture, Ablation, & Robustness Suite
```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=. python3 src/training/trainer_full.py
```

### 6. Calibrate & Evaluate Ensemble Layer
```bash
PYTHONPATH=. python3 src/training/trainer_ensemble.py
```

### 7. Generate Publication Results Cards
```bash
python3 generate_result_cards.py
```

---

## 📜 Dataset Reference & Citation
The dataset used in this work is published under:
> **M. Aghashahi, L. Sela, and M. K. Banks**, *"Benchmarking dataset for leak detection and localization in water distribution systems,"* **Data in Brief**, vol. 48, p. 109148, 2023. DOI: [10.1016/j.dib.2023.109148](https://doi.org/10.1016/j.dib.2023.109148).
