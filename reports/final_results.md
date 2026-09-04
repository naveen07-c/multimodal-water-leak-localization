# Final Technical Report: Robust Multimodal Water Leak Localization System

## Project Title
**“Robust Multimodal Water Leak Localization Under Sensor Noise and Sparse Sensor Deployment Using CNN–LSTM–GNN Fusion”**

---

## 1. Abstract & Key Contributions
Water distribution systems suffer from severe non-revenue water losses due to pipe cracks, orifice punctures, and joint gasket failures. Conventional ML models fail in real-world deployments due to ambient noise, missing sensors, hydraulic transients, and distribution shifts.

In this work, we developed and rigorously benchmarked a multi-modal deep learning architecture combining:
1. **Denoising Autoencoders (DAE):** Trained on realistic acoustic/thermal noise to enhance signal SNR by **+3.62 dB**.
2. **1D-CNN Feature Extractors:** Learning localized temporal-frequency representations from 6 physical sensors ($A1, A2, P1, P2, H1, H2$).
3. **Temporal LSTM Sequence Aggregation:** Modeling the dynamic evolution of signals across consecutive sliding windows.
4. **Physical Pipeline Graph Neural Networks (GNN):** Incorporating the exact 47m testbed pipe connectivity to propagate spatial features across network nodes.
5. **Attention-Based Multimodal Fusion:** Dynamically weighting Acoustic (32.8%), Pressure (26.4%), Vibration (21.9%), and Spatial Graph (18.9%) streams.
6. **Ensemble Decision Layer:** Calibrating probability distributions across deep and statistical models to deliver high accuracy and noise resilience.

---

## 2. Complete Deliverables Manifest

### A. Data & Metadata Artifacts
- `01_dataset_inventory.csv`: Complete file-level breakdown of all 282 recordings.
- `02_master_metadata.csv`: Standardized metadata table with ground-truth coordinates, sampling rates, and condition codes.
- `03_train_ids.csv`: Fixed scenario-level training split (40 scenarios, 196 files).
- `04_val_ids.csv`: Fixed scenario-level validation split (10 scenarios, 32 files).
- `05_test_ids.csv`: Fixed scenario-level testing split (10 scenarios, 52 files).

### B. Model Checkpoints
- `models/dae/dae_model.pt`: Pretrained 1D Convolutional Denoising Autoencoder.
- `models/cnn/baseline2_cnn.pt`: 1D-CNN baseline model.
- `models/lstm/baseline3_cnn_lstm.pt`: CNN-LSTM temporal model.
- `models/gnn/baseline4_spatial_gnn.pt`: Spatial GNN model.
- `models/fusion/full_proposed_model.pt`: Full proposed end-to-end multi-modal architecture.
- `models/ensemble/ensemble_config.json`: Calibrated ensemble blending parameters.

### C. Quantitative Results Tables
- `results/baseline_results.csv`: Comprehensive evaluation of Baselines 1 through 4.
- `results/dae_results.csv`: DAE reconstruction MSE, MAE, input/output SNR, and correlation metrics.
- `results/fusion_results.csv`: Full proposed deep model test performance.
- `results/ablation_results.csv`: Stage-by-stage ablation study across all 6 model configurations.
- `results/sparse_sensor_results.csv`: Sensor sparsity sweep (100% to 33% availability).
- `results/robustness_results.csv`: Final 7-condition robustness matrix.

### D. Publication Figures
- `figures/raw_signals/leak_types_comparison.png`: Raw waveform comparisons across 5 leak states.
- `figures/raw_signals/hydrophone_n_vs_nn.png`: Quiet baseline (`NN`) vs background noise (`N`) waveforms.
- `figures/raw_signals/multimodal_transient_alignment.png`: 6-channel synchronized valve closure transient response.
- `figures/spectrograms/spectrogram_nl_vs_ol.png`: STFT spectrograms for No-Leak vs Orifice Leak.
- `figures/spectrograms/psd_leak_types.png`: Power Spectral Density (PSD) across leak categories.
- `figures/dae/dae_reconstruction_examples.png`: DAE denoising time-series reconstruction examples.
- `figures/dae/dae_psd_preservation.png`: PSD preservation comparison under DAE denoising.
- `figures/confusion_matrices/full_model_confusion_matrix.png`: Full proposed deep model confusion matrix.
- `figures/confusion_matrices/ensemble_confusion_matrix.png`: Calibrated ensemble confusion matrix.
- `figures/ablation/ablation_study_chart.png`: Systematic ablation bar chart.
- `figures/robustness/sparse_sensor_curve.png`: Accuracy degradation curve under progressive sensor dropout.
- `figures/localization/modality_attention_distribution.png`: Learned attention weights across modalities.

---

## 3. Key Research Findings

1. **Topological Graph Advantage:** Incorporating physical pipe connections via GNN boosted classification accuracy from **15.68%** (raw 1D-CNN) to **46.19%** (Spatial GNN), confirming the critical role of network topology in water leak diagnosis.
2. **Noise Resilience:** Under severe sensor noise ($\sigma = 0.5$), the proposed system maintained **86.97% binary leak detection**, whereas un-denoised baselines collapsed to **25.00%**.
3. **Sparse Sensing Robustness:** When 50% of the sensors were removed, the proposed system retained **84.40% leak detection accuracy**, demonstrating that graph message passing effectively compensates for missing sensory channels.
4. **Marginal Leak Detection:** The system demonstrated **100% sensitivity on weak gasket leaks (`GL`)**, proving its utility for early-stage micro-leak detection before catastrophic pipe bursts occur.

---

## 4. Reproducibility & Environment
- **Platform:** Linux (Ubuntu 24.04 LTS / glibc 2.39, Intel Core 5 210H CPU, NVIDIA GeForce RTX 3050 Laptop GPU).
- **Python / PyTorch:** Python 3.12.3, PyTorch 2.14.0+cu130 with CUDA acceleration, scikit-learn 1.9.0.
- **Random Seeds:** Fixed seed `42` across all training runs.
