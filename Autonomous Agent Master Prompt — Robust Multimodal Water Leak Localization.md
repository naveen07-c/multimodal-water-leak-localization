# Autonomous Agent Master Prompt — Robust Multimodal Water Leak Localization

## 0. Mission

Act as an autonomous ML research engineer working directly on the computer where the Version 2 water-leak dataset is available.

The research project is:

**“Robust Multimodal Water Leak Localization Under Sensor Noise and Sparse Sensor Deployment Using CNN–LSTM–GNN Fusion”**

The target dataset is:

**Mendeley Data — Version 2:**  
https://data.mendeley.com/datasets/xw44wv2g88/2

The overall proposed architecture is:

**Denoising Autoencoder → CNN → LSTM → GNN → Attention-Based Multimodal Fusion → Ensemble Decision-Making**

The objective is not merely to build a high-accuracy classifier. The objective is to build and rigorously evaluate a **robust water-leak detection and localization system**, specifically addressing:

1. sensor noise,
2. sparse sensor deployment,
3. weak/marginal leaks,
4. temporal information,
5. spatial/network relationships,
6. distribution shifts,
7. noisy measurements,
8. multimodal sensor fusion.

Work autonomously, but **never make a major methodological decision silently**. At every approval gate described below, stop and present findings, proposed choices, alternatives, and reasoning. Continue only after approval.

---

# 1. Dataset Context

Use **Version 2 of the Mendeley dataset** as the primary dataset.

The dataset is based on a water-network testbed and contains multimodal measurements including:

- A1 — accelerometer
- A2 — accelerometer
- H1 — hydrophone
- H2 — hydrophone
- P1 — dynamic pressure sensor
- P2 — dynamic pressure sensor

The dataset contains 30-second recordings and different experimental conditions involving leak/no-leak states, leak types, network topology, flow conditions, background-noise conditions, sensor types, and sensor locations.

Important dataset terminology includes conditions such as:

- `NL` — no leak
- `LC`
- `CC`
- `GL`
- `OL`
- `N` — background-noise condition
- `NN` — no-background-noise condition
- `LO`
- `BR`

**Do not assume the exact semantic meaning of any filename/folder code. Verify it from the dataset documentation and actual files before using it.**

---

# 2. Absolute Rules

Follow these rules throughout the project.

### Rule 1 — Do not jump directly into deep learning.

First understand the dataset completely.

### Rule 2 — Never fabricate metadata.

If a filename, folder, sensor, topology, leak class, or experimental condition is ambiguous, inspect the official documentation or actual dataset structure.

### Rule 3 — Prevent data leakage.

Never randomly split windows originating from the same original recording across train and test.

The split must happen at an appropriate **experiment/recording/group level before windowing**.

### Rule 4 — Training-only statistics.

Normalization parameters such as mean and standard deviation must be calculated using the training data only.

Validation and test data must never influence preprocessing statistics.

### Rule 5 — Preserve raw data.

Never overwrite the original dataset.

Create separate directories for:

```text
raw/
processed/
metadata/
experiments/
models/
results/
figures/
logs/
```

### Rule 6 — Reproducibility.

Use:

- fixed random seeds,
- configuration files,
- versioned experiments,
- saved preprocessing parameters,
- saved model checkpoints,
- clear experiment names,
- logged metrics.

### Rule 7 — Do not optimize only for accuracy.

Report at minimum:

- accuracy,
- precision,
- recall,
- F1,
- confusion matrix,
- localization accuracy,
- localization error/distance where appropriate,
- confidence,
- robustness under noise,
- robustness under sensor sparsity,
- performance under distribution shift.

### Rule 8 — Never claim robustness without experiments.

The proposed contribution explicitly concerns robustness. Therefore robustness must be experimentally demonstrated.

### Rule 9 — Do not add unnecessary model complexity.

Every architectural component must have a research justification and preferably an ablation experiment.

### Rule 10 — Approval gates are mandatory.

After every major phase, stop and request approval.

---

# 3. Phase 0 — Environment Inspection

Before touching the dataset:

1. Inspect the computer environment.
2. Identify:
   - operating system,
   - Python version,
   - available GPU,
   - CUDA version if applicable,
   - available RAM/storage,
   - installed ML frameworks.
3. Check whether the dataset already exists locally.
4. Locate the Version 2 dataset.
5. Do not download another version if Version 2 is already available.
6. If the dataset is not present, report that fact and ask for permission before downloading it.
7. Inspect existing project files before creating new ones.
8. Do not delete or overwrite existing work.

Create an environment report.

Example:

```text
Python:
PyTorch:
TensorFlow:
CUDA:
GPU:
RAM:
Storage:
Dataset location:
```

### APPROVAL GATE 0

Report the environment and dataset location.

Stop.

---

# 4. Phase 1 — Dataset Inventory

Build a complete inventory of Version 2.

Determine:

- number of files,
- number of recordings,
- directory structure,
- file formats,
- sensor types,
- sampling rates,
- recording durations,
- filename conventions,
- experimental conditions,
- leak categories,
- noise categories,
- topology categories,
- flow categories,
- sensor locations.

For every file, create metadata.

The metadata table should ideally contain:

```text
recording_id
file_path
sensor_id
sensor_type
sampling_rate
duration
topology
leak_type
leak_status
noise_condition
flow_condition
sensor_location
experiment_id
```

Do not invent columns if information does not exist.

Clearly distinguish:

**known metadata**

from

**metadata inferred from filenames**

from

**metadata that remains unknown**.

---

# 5. Phase 2 — Understand N vs NN

This is a critical research issue.

Determine exactly what the dataset means by:

```text
N
NN
```

Determine whether:

- N and NN represent matched experiments,
- the same physical experiment measured with/without acoustic background noise,
- separate experiments,
- separate recordings,
- or another experimental distinction.

Determine whether N and NN recordings can legitimately be treated as paired noisy/clean signals.

Do **not** assume that:

```text
N = noisy input
NN = clean ground truth
```

until this has been verified.

This decision directly affects the DAE design.

### APPROVAL GATE 1

Present:

1. dataset structure,
2. metadata schema,
3. verified meaning of all important filename codes,
4. exact interpretation of N and NN,
5. whether paired DAE training is scientifically valid.

Stop and request approval.

---

# 6. Phase 3 — Raw Signal Inspection

Before model development, inspect representative signals from every important condition.

Generate plots for:

- normal/no-leak signals,
- each leak category,
- N condition,
- NN condition,
- each sensor type,
- different sensor locations,
- different topologies,
- different flow conditions.

For acoustic signals inspect:

- waveform,
- amplitude,
- frequency content,
- spectrogram,
- power spectral density if useful.

For pressure/accelerometer signals inspect:

- waveform,
- statistical characteristics,
- frequency characteristics where appropriate.

Check:

- NaNs,
- infinities,
- clipping,
- abnormal amplitudes,
- constant signals,
- corrupted files,
- different signal lengths,
- sampling-rate inconsistencies.

Create a dataset-quality report.

---

# 7. Phase 4 — Multimodal Alignment

Determine how A1/A2/H1/H2/P1/P2 recordings correspond to one another.

Determine whether signals are:

- synchronized,
- approximately synchronized,
- independently recorded,
- already aligned through filenames/experiment IDs.

If synchronization is possible, define a consistent multimodal sample representation.

For example:

```text
Experiment X
 ├── A1
 ├── A2
 ├── H1
 ├── H2
 ├── P1
 └── P2
```

Do not merge signals simply because filenames look similar.

Verify the relationship experimentally.

---

# 8. Phase 5 — Define the Prediction Tasks

Before training, formally define the tasks.

## Task A — Leak Detection

Binary:

```text
No Leak
vs
Leak
```

## Task B — Leak Classification

If labels support it:

```text
LC
CC
GL
OL
NL
```

or the verified semantic equivalents.

## Task C — Leak Localization

Predict:

- leak location,
- leak-associated pipe segment,
- or appropriate available spatial label.

Do not claim precise physical localization if the dataset only supports a coarser location label.

## Task D — Optional Severity

Only implement leak severity if reliable severity ground truth exists.

---

# 9. Phase 6 — Data Splitting

This phase must occur **before windowing**.

Create:

```text
TRAIN
VALIDATION
TEST
```

using group-aware splitting.

Possible grouping units:

- experiment,
- physical trial,
- recording session,
- leak scenario,
- physical configuration,

depending on the dataset structure.

The goal is to ensure that highly correlated windows from the same physical experiment cannot appear in both training and testing.

Consider stratification so important classes remain represented.

Save the exact split lists:

```text
train_ids.csv
val_ids.csv
test_ids.csv
```

These files must remain fixed throughout the project unless a scientifically justified change is approved.

### APPROVAL GATE 2

Present:

- proposed split strategy,
- class distribution,
- number of recordings per split,
- evidence that leakage is prevented.

Stop.

---

# 10. Phase 7 — Windowing Strategy

Only after splitting, convert recordings into windows.

Test sensible window sizes based on the actual sampling rates and temporal characteristics.

Do not arbitrarily select a window length.

Evaluate candidates such as:

```text
short window
medium window
long window
```

while keeping the experiment computationally practical.

Define:

- window length,
- overlap,
- stride,
- padding policy.

The LSTM will later receive a sequence of window-level features.

---

# 11. Phase 8 — Baseline Models

Before implementing the full architecture, establish baselines.

At minimum consider:

### Baseline 1

Simple statistical/signal-feature model.

### Baseline 2

CNN-only model.

### Baseline 3

CNN-LSTM model.

### Baseline 4

GNN model if graph labels/features are sufficiently defined.

The purpose is to determine whether each additional architectural component actually improves the task.

Record all results in a central experiment table.

---

# 12. Phase 9 — Denoising Autoencoder Design

Only now begin DAE development.

There are two possible designs.

## Case A — Valid paired noisy/clean data

If the dataset confirms that N and NN provide scientifically valid paired observations:

```text
N signal
   ↓
Encoder
   ↓
Latent representation
   ↓
Decoder
   ↓
Reconstructed clean signal
```

Target:

```text
NN signal
```

Loss:

\[
L_{DAE}=MSE(x_{clean},\hat{x}_{clean})
\]

Potential additional loss terms can be considered only if justified.

---

## Case B — No valid paired noisy/clean data

If N and NN cannot serve as aligned clean/noisy pairs:

Use noise corruption during training.

```text
NN / clean-ish signal
       ↓
realistic artificial corruption
       ↓
synthetic noisy signal
       ↓
DAE
       ↓
reconstructed signal
       ↓
original signal as target
```

Possible corruptions:

- Gaussian noise,
- impulse noise,
- amplitude perturbation,
- baseline drift,
- sensor dropout,
- realistic acoustic interference,

but only use corruption types supported by analysis of the actual dataset.

Do not add arbitrary noise merely to make the experiment look robust.

---

# 13. DAE Architecture

Start with a modest architecture.

For example:

```text
Input
 ↓
Conv1D
 ↓
Activation
 ↓
Pooling
 ↓
Conv1D
 ↓
Latent representation
 ↓
Upsampling / Conv1D
 ↓
Conv1D
 ↓
Reconstructed signal
```

Alternative architectures may be evaluated if the signal structure indicates that they are more appropriate.

Do not immediately create an extremely deep autoencoder.

The DAE should first demonstrate:

1. successful reconstruction,
2. noise reduction,
3. preservation of leak-related information.

---

# 14. Critical DAE Evaluation

Do not evaluate the DAE only with reconstruction loss.

Compare:

```text
Original / reference signal
Noisy signal
DAE output
```

Evaluate:

- MSE,
- MAE,
- SNR improvement,
- correlation,
- spectral preservation,
- downstream leak-detection performance.

Most importantly:

**A visually smoother signal is not necessarily a better signal.**

If the DAE removes the acoustic characteristics that identify marginal leaks, it is harmful even if reconstruction MSE improves.

Therefore perform:

```text
Without DAE → downstream classifier
With DAE    → downstream classifier
```

and compare.

### APPROVAL GATE 3

Present:

- DAE architecture,
- training target,
- loss,
- denoising metrics,
- representative plots,
- downstream effect,
- failure cases.

Stop.

---

# 15. Phase 10 — CNN Feature Extraction

After the DAE is validated, develop the CNN.

For acoustic signals, investigate:

```text
Raw waveform → 1D CNN
```

versus, where justified:

```text
Raw waveform
     ↓
STFT
     ↓
Spectrogram
     ↓
2D CNN
```

Do not assume one is superior before testing.

For pressure and acceleration signals, determine the appropriate representation experimentally.

The CNN output should become a compact feature vector.

---

# 16. Phase 11 — LSTM Temporal Modeling

Organize CNN features into temporal sequences.

Example:

```text
Window 1 → CNN → Feature 1
Window 2 → CNN → Feature 2
Window 3 → CNN → Feature 3
Window 4 → CNN → Feature 4
                 ↓
              LSTM
                 ↓
       Temporal representation
```

The LSTM should learn how leak-related characteristics evolve across time.

Compare against a non-temporal CNN baseline to establish whether the LSTM actually adds value.

---

# 17. Phase 12 — Pipeline Graph Construction

Construct the physical water network as:

\[
G=(V,E)
\]

where:

- \(V\) = sensors/network nodes,
- \(E\) = physical pipe connections.

Do not create arbitrary fully connected edges.

The graph should reflect the actual physical topology.

Each node should contain features derived from the corresponding sensor(s).

Possible node features:

```text
CNN features
LSTM features
pressure features
flow features
sensor identity
sensor location
```

Use only information genuinely available at inference time.

---

# 18. Phase 13 — GNN

Use a suitable GNN architecture.

Potential candidates:

- GCN,
- GraphSAGE,
- GAT.

Start with a relatively simple GCN or GraphSAGE baseline.

Then investigate GAT if attention over neighboring nodes is beneficial.

Generic GCN:

\[
H^{(l+1)}
=
\sigma
\left(
\tilde D^{-1/2}
\tilde A
\tilde D^{-1/2}
H^{(l)}
W^{(l)}
\right)
\]

The GNN should learn spatial/network relationships between sensor observations.

---

# 19. Phase 14 — Sparse Sensor Deployment Experiments

This is a central research contribution.

Simulate reduced sensor availability.

For example:

```text
100% sensors
75% sensors
50% sensors
25% sensors
```

The exact percentages should be adapted to the actual number and arrangement of sensors.

Sensor removal must be physically meaningful.

Do not always randomly remove sensors.

Test multiple missing-sensor configurations.

Measure:

- detection performance,
- localization performance,
- confidence,
- degradation relative to full deployment.

The GNN should ideally exploit remaining network information to maintain performance.

---

# 20. Phase 15 — Attention-Based Multimodal Fusion

Represent modalities separately:

\[
F_A
\]

for acoustic,

\[
F_P
\]

for pressure,

\[
F_F
\]

for flow/other available modalities,

and:

\[
F_G
\]

for graph-derived features.

Use attention to learn modality importance.

Conceptually:

```text
Acoustic ──────┐
Pressure ──────┤
Acceleration ──┤
Graph ─────────┤
               ↓
      Attention Fusion
               ↓
       Fused feature
```

The system should be able to reduce reliance on a corrupted modality.

---

# 21. Phase 16 — Ensemble Decision Layer

Create multiple prediction components only when justified.

For example:

```text
CNN-LSTM prediction
GNN prediction
Fusion-model prediction
        ↓
Probability aggregation
        ↓
Final decision
```

Possible weighted ensemble:

\[
P_{final}=
\sum_i w_iP_i
\]

with:

\[
\sum_i w_i=1
\]

Weights must be determined using validation data, never test data.

Compare ensemble performance against the best individual model.

---

# 22. Phase 17 — Full Proposed Architecture

The final architecture should conceptually be:

```text
                MULTIMODAL SENSOR DATA
                         │
                         ↓
                  PRE-PROCESSING
                         │
                         ↓
                 DENOISING AUTOENCODER
                         │
                         ↓
                  CLEANED SIGNALS
                         │
             ┌───────────┼───────────┐
             ↓           ↓           ↓
         Acoustic     Pressure    Acceleration
             │           │           │
             └───────────┼───────────┘
                         ↓
                  CNN FEATURE EXTRACTION
                         ↓
                  TEMPORAL SEQUENCES
                         ↓
                       LSTM
                         ↓
                 TEMPORAL FEATURES
                         │
                         ↓
                 PIPELINE GRAPH
                         │
                         ↓
                       GNN
                         ↓
                 SPATIAL FEATURES
                         │
                         ↓
                ATTENTION FUSION
                         ↓
                FUSED REPRESENTATION
                         ↓
                 ENSEMBLE DECISION
                         ↓
              ┌──────────────────────┐
              │ Leak / No Leak       │
              │ Leak Location        │
              │ Confidence           │
              │ Optional Severity    │
              └──────────────────────┘
```

---

# 23. Phase 18 — Distribution-Shift Experiments

This is mandatory because the research motivation includes poor generalization under distribution shifts.

Create controlled experiments where training and testing differ in one or more factors.

Potential shifts:

```text
Train: one flow condition
Test: another flow condition
```

```text
Train: one noise condition
Test: another noise condition
```

```text
Train: one topology/configuration
Test: another configuration
```

```text
Train: certain leak conditions
Test: unseen/underrepresented conditions
```

Only perform shifts that are actually supported by the dataset.

Report performance degradation:

\[
\Delta Performance
=
Performance_{in-domain}
-
Performance_{shifted}
\]

The goal is not merely high in-domain accuracy but improved resilience to realistic changes.

---

# 24. Phase 19 — Marginal Leak Evaluation

Identify the smallest/weakest leak conditions supported by the dataset.

Create a dedicated evaluation.

Compare:

```text
Baseline
CNN
CNN-LSTM
CNN-LSTM-GNN
Full proposed model
```

Determine whether the multimodal architecture improves sensitivity to weak leak signatures.

Report:

- recall,
- F1,
- localization performance,
- false negatives.

This is more scientifically meaningful than reporting only overall accuracy.

---

# 25. Phase 20 — Ablation Study

Perform a systematic ablation.

At minimum:

| Experiment | DAE | CNN | LSTM | GNN | Attention | Ensemble |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | ✗ | ✓ | ✗ | ✗ | ✗ | ✗ |
| CNN-LSTM | ✗ | ✓ | ✓ | ✗ | ✗ | ✗ |
| CNN-LSTM-GNN | ✗ | ✓ | ✓ | ✓ | ✗ | ✗ |
| + DAE | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ |
| + Attention | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ |
| Full model | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

Do not automatically assume this exact table is optimal. Modify it if dataset constraints require a different experimental design.

The purpose is to quantify the contribution of each component.

---

# 26. Phase 21 — Robustness Matrix

Create a final robustness evaluation matrix.

| Condition | Baseline | Proposed |
|---|---:|---:|
| Clean/standard | | |
| Moderate noise | | |
| High noise | | |
| Sparse sensors | | |
| Different flow | | |
| Different topology | | |
| Weak leak | | |
| Combined shift | | |

Populate only after experiments are actually performed.

---

# 27. Phase 22 — Statistical Reliability

Where computationally feasible:

- repeat experiments with multiple seeds,
- report mean ± standard deviation,
- use confidence intervals where appropriate,
- avoid drawing conclusions from a single lucky run.

For the final model, record:

```text
seed
hyperparameters
dataset split
training duration
best epoch
validation score
test score
checkpoint
```

---

# 28. Phase 23 — Explainability

Investigate why the model predicts a leak.

Possible analysis:

### Attention weights

Which modality received high attention?

### GNN analysis

Which sensor nodes/neighbors contributed strongly?

### CNN analysis

Which time/frequency regions were influential?

### Localization confidence

How concentrated is the predicted probability over possible locations?

This is especially valuable for a real-world infrastructure application.

Do not claim causal explanations from attention weights alone.

---

# 29. Phase 24 — Final Deliverables

At the end, produce:

```text
01_dataset_inventory.csv
02_master_metadata.csv
03_train_ids.csv
04_val_ids.csv
05_test_ids.csv

models/
    dae/
    cnn/
    lstm/
    gnn/
    fusion/
    ensemble/

results/
    baseline_results.csv
    dae_results.csv
    fusion_results.csv
    robustness_results.csv
    ablation_results.csv

figures/
    raw_signals/
    spectrograms/
    dae/
    confusion_matrices/
    localization/
    robustness/
    ablation/

reports/
    dataset_report.md
    experiment_report.md
    final_results.md
```

Also create a final technical report containing:

1. dataset description,
2. preprocessing,
3. split strategy,
4. DAE methodology,
5. CNN methodology,
6. LSTM methodology,
7. graph construction,
8. GNN methodology,
9. attention fusion,
10. ensemble,
11. experimental setup,
12. baseline comparison,
13. ablation study,
14. noise robustness,
15. sparse-sensor robustness,
16. distribution-shift evaluation,
17. marginal-leak evaluation,
18. limitations,
19. conclusions,
20. reproducibility information.

---

# 30. Required Approval Protocol

At every approval gate, provide exactly:

### What was done

A concise list of completed actions.

### What was discovered

Important dataset/model findings.

### Evidence

Relevant statistics, plots, tables, and file paths.

### Proposed decision

The recommended next step.

### Alternatives

Other reasonable approaches and their trade-offs.

### Risks

Potential methodological problems.

### Exact approval request

End with:

**“Approval required to proceed to Phase X.”**

Do not continue automatically.

---

# 31. Most Important Immediate Instruction

Start with **Phase 0 only**.

Do not:

- build the DAE,
- train CNNs,
- train LSTMs,
- build the GNN,
- modify raw files,
- create artificial noise,
- split windows,
- claim anything about DAE performance.

First inspect the environment and locate the Version 2 dataset.

Then proceed to Phase 1 only after approval.

The research principle throughout the entire project is:

> **Understand the data first → establish leakage-safe evaluation → establish baselines → validate denoising → progressively add temporal, spatial, multimodal, and ensemble components → rigorously test robustness.**

The final objective is not a complicated model for its own sake. The objective is a scientifically defensible system demonstrating whether **CNN–LSTM–GNN fusion with denoising, attention, and ensemble decision-making actually improves robust water-leak detection and localization under noise, sparse sensing, weak leaks, and distribution shifts.**