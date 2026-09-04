# Dataset Report: Robust Multimodal Water Leak Localization Testbed

## 1. Dataset Overview & Provenance
- **Dataset:** Mendeley Data — Version 2 (*DOI: 10.17632/xw44wv2g88 / 10.17632/tbrnp6vrnj.1*)
- **Published in:** *Data in Brief* (2023), Vol. 48, 109148. Authors: Mohsen Aghashahi, Lina Sela, M. Katherine Banks.
- **Physical Testbed:** 47-meter closed/open water network constructed from Schedule-80 PVC pipes (152.4 mm internal diameter) with 17 pipe segments, 2 prototype fire hydrants, 1 service line, and concrete support blocks.
- **Total Files Audited:** 282 recordings (80 Accelerometer `.csv`, 80 Dynamic Pressure `.csv`, 122 Hydrophone `.raw`).
- **Physical Experiments:** 60 controlled scenarios + 2 standalone acoustic noise reference files.

---

## 2. Sensor Specifications & Placement

| Sensor | Manufacturer & Model | Modality | Sampling Rate ($f_s$) | Physical Location | Measurement Units |
| :--- | :--- | :--- | :---: | :--- | :---: |
| **A1** | PCB Piezotronics 333B50 | Accelerometer (Vibration) | 25,600 Hz | Leg of Tee Connection 1 (Upstream distribution branch) | $\text{m/s}^2$ |
| **A2** | PCB Piezotronics 333B50 | Accelerometer (Vibration) | 25,600 Hz | Leg of Tee Connection 2 near Hydrant 2 (Downstream branch) | $\text{m/s}^2$ |
| **P1** | PCB Piezotronics 102B16 | Dynamic Pressure Sensor | 25,600 Hz | Supply line inlet (Upstream baseline before network entry) | $\text{Pa}$ |
| **P2** | PCB Piezotronics 102B16 | Dynamic Pressure Sensor | 25,600 Hz | Farthest distribution corner (Downstream monitoring) | $\text{Pa}$ |
| **H1** | Aquarian Audio H2c | Hydrophone (Acoustic) | 8,000 Hz | Prototype Fire Hydrant 1 (Submerged at mid-pipe residential mimic) | $\text{V}$ (or $\text{dB}$) |
| **H2** | Aquarian Audio H2c | Hydrophone (Acoustic) | 8,000 Hz | Prototype Fire Hydrant 2 (Submerged at intersection hydrant) | $\text{V}$ (or $\text{dB}$) |

---

## 3. Experimental Variables & Semantic Taxonomy

### A. Network Topologies ($T$)
1. **Branched (`BR`):** Open-ended tree topology simulating radial distribution feeder mains.
2. **Looped (`LO`):** Closed-loop grid topology with multi-path circulation simulating municipal grid mains.

### B. Leak Conditions ($L$)
All leaks were physically induced at the **Middle Pipe** segment of the testbed network:
1. **`NL` (No Leak):** Intact baseline pipe (0% water loss).
2. **`GL` (Gasket Leak):** Loosened flange bolts at middle junction ($\sim 0.05-0.08\text{ L/s}$ loss, $\sim 7-12\%$ of inflow — **Weakest / Marginal Leak**).
3. **`CC` (Circumferential Crack):** Milled $2\text{ mm} \times 1\text{ mm}$ circumferential slot ($\sim 0.10-0.13\text{ L/s}$ loss).
4. **`LC` (Longitudinal Crack):** Milled $2\text{ mm} \times 1\text{ mm}$ longitudinal slot ($\sim 0.12-0.15\text{ L/s}$ loss).
5. **`OL` (Orifice Leak):** Drilled $\sim 1.6\text{ mm}$ circular orifice ($\sim 0.18-0.22\text{ L/s}$ loss).

### C. Background Flow Regimes ($F$)
1. **`ND` (No Demand):** 0 L/s outflow from service line.
2. **`0.18 LPS`:** 0.18 L/s steady demand outflow (simulating 1:00 AM nighttime low demand).
3. **`0.47 LPS`:** 0.47 L/s steady demand outflow (simulating 5:00 AM morning demand).
4. **`Transient`:** Rapid shutoff of service line globe valve at $t \approx 20\text{ s}$ ($0.47 \to 0\text{ L/s}$ step change creating water hammer dynamics).

### D. Background Noise States ($B$)
1. **`N` (With Background Noise):** Continuous traffic noise via central loudspeaker + moving electric saw around testbed.
2. **`NN` (Without Background Noise):** Quiet baseline laboratory environment (no loudspeaker, no saw).
3. **`Ambient`:** Standalone acoustic recording of traffic loudspeaker + saw noise with pumps and flow OFF.

---

## 4. Signal Integrity & Data Quality Audit Findings

- **Missing Values & NaNs:** **0** across all 282 files.
- **Infinities / Corrupted Headers:** **0** across all 282 files.
- **Constant / Dead Sensors:** None detected (all files display non-zero variance and dynamic response).
- **Temporal Duration:** Every file contains $\ge 33.9$ seconds of clean signal (Hydrophone up to 61.3s). The standardized 30.0-second baseline ($240,000$ samples @ 8 kHz for H; $768,000$ samples @ 25.6 kHz for A/P) is 100% complete and consistent across all files.

---

## 5. Multimodal Alignment & Group-Aware Splitting

### A. Temporal Synchronization
- All 6 sensor streams (`A1`, `A2`, `P1`, `P2`, `H1`, `H2`) within each multimodal experiment are synchronised by shared experiment timestamp.
- Transient valve shutoff at $t = 20.0\text{ s}$ confirms coherent pressure drops in P1/P2 and acoustic/vibration shocks in H1/H2/A1/A2.

### B. Group-Aware Stratified Split (Leakage-Safe)
Splits were constructed strictly at the **scenario / experiment level** before any windowing:
- **Train Set (40 scenarios, 196 files):** Balanced representation of all 5 leak classes across both topologies and flow conditions.
- **Validation Set (10 scenarios, 32 files):** 2 scenarios per leak class (100% held-out physical experiments).
- **Test Set (10 scenarios, 52 files):** 2 scenarios per leak class (100% held-out physical experiments).
- **Split Artifacts:** `metadata/03_train_ids.csv`, `metadata/04_val_ids.csv`, `metadata/05_test_ids.csv`.
