# Research Prompt: Detailed Literature Survey for Robust Multimodal Water Leak Localization

## Role and deadline
Act as a careful academic researcher specializing in water distribution systems, leak detection/localization, signal processing, time-series deep learning, graph neural networks, sensor fusion, and domain generalization.

The submission deadline is **Sunday, 11 October 2026**. Work efficiently but do not sacrifice source verification. The finished survey must contain **20–25 distinct research papers, inclusive**. Aim for **22 papers** so the work stays within the rules while leaving room to replace weak or unverifiable sources.

You are starting with no prior context about the project. Read the complete brief below before searching for literature. Do not ask the user to restate the project.

## 1. Project context

Project repository: https://github.com/naveen07-c/multimodal-water-leak-localization

Project title:
**Robust Multimodal Water Leak Localization Under Sensor Noise and Sparse Sensing Using CNN–LSTM–GNN Fusion**

### Problem being addressed
Water distribution networks experience real water losses and operating costs from leaks. A practical automated system must do more than detect that a leak exists: it should, where possible, distinguish leak types, identify their spatial location, estimate severity, and operate reliably when sensors are noisy, missing, or deployed in different network/flow conditions.

Important challenges:
- Environmental acoustic noise from traffic and tools, along with sensor noise and drift.
- Sparse sensing and sensor failure, where not all instruments are available.
- Weak or marginal leaks, especially gasket leaks with relatively low-amplitude signatures.
- Non-stationary hydraulic behavior, including demand changes and transient/water-hammer events.
- Physical pipe topology: measurements are collected at particular locations in a connected network, not on a regular image grid.
- Generalization from a controlled testbed to different datasets, sites, topologies, and operating regimes.
- Class imbalance and the difference between high binary detection accuracy and reliable multi-class leak classification/localization.

### Dataset and physical setting
The project uses Version 2 of the Mendeley water leak detection/localization dataset:
https://data.mendeley.com/datasets/xw44wv2g88/2

The dataset is described in:
M. Aghashahi, L. Sela, and M. K. Banks, “Benchmarking dataset for leak detection and localization in water distribution systems,” Data in Brief, vol. 48, article 109148, 2023, DOI: 10.1016/j.dib.2023.109148.

The repository documentation describes a controlled 47-m PVC testbed, six synchronized sensor channels, and 60 physical experiment scenarios plus noise-reference recordings. Verify exact dataset details against the original source instead of trusting this brief blindly.

Six sensor channels:
- A1, A2: accelerometers / pipe vibration.
- H1, H2: hydrophones / acoustic signals.
- P1, P2: dynamic pressure sensors / hydraulic pressure variation.

The source recordings have different rates (the repository reports 25.6 kHz for accelerometers and dynamic pressure, and 8 kHz for hydrophones). The preprocessing pipeline resamples channels to a common 8 kHz rate, normalizes using training-only statistics, and creates 1-second windows with 0.5-second stride. The LSTM uses a context of five consecutive windows. Confirm current implementation details in the repository before describing them as facts.

Leak categories represented include:
- NL: no leak
- GL: gasket leak
- CC: circumferential crack
- LC: longitudinal crack
- OL: orifice leak

Flow conditions include no demand, steady demand conditions, and a transient/valve-closure regime. Topologies include branched and looped arrangements. Acoustic-background conditions include quiet and deliberately noisy recordings.

### Proposed / implemented model family
The repository describes the principal pipeline as:
1. Denoising autoencoder (1D convolutional DAE).
2. Per-sensor 1D-CNN feature extraction.
3. Temporal LSTM / Bi-LSTM modeling.
4. Graph neural network over the physical pipe/testbed topology.
5. Attention-based fusion across acoustic, pressure, vibration, and graph representations.
6. Ensemble/decision layers and task-specific heads.

The intended tasks include binary leak detection, five-class leak-type classification, spatial localization, and leak-severity/outflow estimation. Do not assume each task is fully validated simply because it is named in documentation; distinguish implemented, evaluated, and planned components.

Repository reports include in-domain experiments, model ablations, synthetic/noise stress tests, sensor-dropout tests, hydraulic-transient and topology conditions, and cross-dataset evaluation involving a Hong Kong dataset. Inspect the latest repository state and treat results as project-reported results, not independently verified scientific facts.

### Repository links to inspect first
- README: https://github.com/naveen07-c/multimodal-water-leak-localization
- System architecture report: https://github.com/naveen07-c/multimodal-water-leak-localization/blob/main/models/baseline/reports/architecture_and_pipeline_report.md
- Dataset report: https://github.com/naveen07-c/multimodal-water-leak-localization/blob/main/models/baseline/reports/dataset_report.md
- Experiment report: https://github.com/naveen07-c/multimodal-water-leak-localization/blob/main/models/baseline/reports/experiment_report.md
- Final results report: https://github.com/naveen07-c/multimodal-water-leak-localization/blob/main/models/baseline/reports/final_results.md
- Comparison table: https://github.com/naveen07-c/multimodal-water-leak-localization/blob/main/final_comparison_table.md
- Architecture variants: inspect the files in https://github.com/naveen07-c/multimodal-water-leak-localization/tree/main/architectures
- Model implementations and result CSVs are under models/baseline/src/ and models/baseline/results/.

## 2. Main research question

Conduct a critical literature survey answering:

**How can multimodal signal processing and learning, temporal modeling, physical-topology-aware graph learning, and robust sensor fusion be combined to improve water leak detection, leak-type classification, localization, and severity estimation under noise, sparse sensing, hydraulic transients, and dataset/domain shifts?**

The survey must identify what existing studies solve well, where their evidence is weak or incomplete, and what gaps remain relevant to this project's design and evaluation.

## 3. Strict paper-count and recency constraints

1. Include **at least 20 and no more than 25 distinct papers**. Target **22**.
2. Give priority to work published in **2025 and 2026**, especially peer-reviewed 2025–2026 papers directly relevant to water leak detection/localization, multimodal sensing, graph learning, robustness, or transfer/generalization.
3. Because the field may not have enough directly relevant 2025–2026 papers, use a balanced selection of relevant recent work and foundational/high-quality work from earlier years. Do not pad with irrelevant papers just to hit the number.
4. Search for works published online in 2026 up to the actual date of research, **9 October 2026**, but verify publication status and date. A 2026 preprint must be labeled as a preprint, not described as peer-reviewed.
5. There is no fixed minimum number of 2025–2026 papers specified by the supervisor. Therefore, maximize meaningful coverage of 2025–2026 work, report exactly how many of the selected papers are from 2025 and 2026, and explain if direct field-specific evidence is limited.
6. Count each distinct scholarly work only once. A preprint and later journal version of the same work are not two papers. Avoid multiple papers from the same research group unless their contributions are meaningfully different.
7. Do not include the project’s own repository or its numerical results as literature papers. The 2023 dataset paper may be cited for dataset provenance, but count it among the selected papers only if it genuinely contributes to the survey and fits the final count.
8. Avoid predatory outlets and low-quality sources. Prefer peer-reviewed journals/conferences, reputable publishers, established professional societies, and high-quality, clearly relevant preprints where necessary.

## 4. Search and source-verification process

Search several sources; do not rely on a single search engine:
- Google Scholar
- Semantic Scholar
- Crossref
- OpenAlex
- IEEE Xplore
- ScienceDirect / Elsevier
- SpringerLink
- ASCE Library
- MDPI only where the paper is credible and relevant
- Scopus or Web of Science if available
- arXiv for recent preprints
- publisher pages and DOI landing pages for final verification

Suggested search-query families:
- "water distribution network leak detection multimodal sensors deep learning 2025"
- "water leak localization graph neural network 2025 2026"
- "water pipe leak detection acoustic vibration pressure sensor fusion"
- "water leakage detection weak leak sensor noise deep learning"
- "water distribution leak localization domain adaptation transfer learning"
- "graph neural network water distribution network fault localization"
- "GNN sensor placement missing sensors infrastructure monitoring"
- "multimodal time series fusion missing modalities sensor failure"
- "denoising autoencoder acoustic leak detection pipe"
- "hydraulic transient leak detection machine learning water distribution"
- "water distribution network leak detection benchmark dataset"
- "cross-domain generalization industrial time-series anomaly detection sensor"
- "robust graph learning missing nodes sensor dropout physical systems"
- "uncertainty calibration imbalanced classification fault diagnosis"

Search broadly enough to find papers on the exact application and closely related methods. Distinguish:
(A) direct water-pipe leak detection/localization literature;
(B) closely related water-distribution-network diagnosis/monitoring;
(C) transferable methodological literature on multimodal fusion, temporal modeling, GNNs, domain generalization, noise robustness, and missing sensors.

Direct application papers should form the core. Methodological papers should be included only when they provide an explicitly transferable technique relevant to this system.

### Verification rules — mandatory
For every candidate paper, confirm as many details as possible from the publisher page or DOI metadata:
- Exact title
- Complete author list or the publisher’s canonical author representation
- Year and actual publication status
- Journal/conference/preprint venue
- Volume, issue, pages/article number if available
- DOI or stable publisher/arXiv URL
- The paper exists and its metadata matches the source
- Its abstract/full text supports every technical claim made about it

Never invent citations, authors, DOIs, venue names, years, accuracy figures, datasets, or methods. Do not cite a search-result snippet as the sole evidence for a detailed claim. Use the abstract at minimum, and read the full text for the most central papers when accessible. Keep an audit trail with the source used to verify each paper.

If a paper cannot be verified, exclude it and replace it. Do not hide uncertain records in the bibliography. Mark paywalled or abstract-only papers honestly and avoid claiming details unavailable from the accessible text.

## 5. Research coverage and allocation

Select 20–25 papers, ideally approximately 22, with coverage across these categories. These are targets, not permission to include irrelevant work:

1. **Water-pipeline leak detection/localization (6–8 papers):** acoustic methods, vibration, pressure transients, multimodal sensors, leak geometry/severity/localization.
2. **Signal processing and denoising (2–3 papers):** robust features, time-frequency analysis, learned denoising, weak leak signatures, noise resilience.
3. **Temporal deep learning (2–3 papers):** CNN, LSTM/GRU, temporal convolution, transformer or state-space methods for non-stationary sensor time series.
4. **Graph/topology-aware learning (3–4 papers):** GNNs for water networks or analogous physical infrastructures; spatial localization, graph construction, topology shift, missing nodes/sensors.
5. **Multimodal fusion and sensor sparsity (3–4 papers):** sensor-level/feature-level/decision-level fusion, attention, modality reliability, missing or corrupted modalities.
6. **Domain generalization, transfer, and robust evaluation (3–4 papers):** testbed-to-field transfer, cross-site generalization, distribution shift, calibration, class imbalance, and leakage-safe experimental design.

Categories may overlap, but the final list must still contain 20–25 unique papers. Do not force exact category counts where high-quality literature is scarce. Explicitly note thin evidence areas.

## 6. Required deliverable and structure

Produce a detailed, formal academic literature survey suitable for a university project report. The main narrative should synthesize papers rather than present 22 isolated mini-summaries.

### Title
Use a precise title such as:
**“A Critical Literature Survey on Robust Multimodal Water Leak Detection and Localization Under Sensor Noise, Sparse Sensing, and Network-Domain Shift.”**

### Section 1 — Abstract
Approximately 200–250 words. State the problem, scope, review method, selected-paper count and year distribution, major themes, main gaps, and implications for the project. Do not claim a systematic review unless a documented reproducible protocol was actually followed.

### Section 2 — Introduction and scope
Explain:
- Why water leakage detection and localization matter.
- Why single-sensor methods can fail under noise and operational variation.
- The differences among detection, leak-type classification, spatial localization, and severity estimation.
- Why multimodal sensing, temporal dynamics, physical topology, robustness, and generalization matter.
- The exact research question and inclusion/exclusion criteria.

### Section 3 — Literature search methodology
Provide a transparent and reproducible search strategy:
- Databases and publisher sources actually searched.
- Actual date of the search.
- Search strings used.
- Date range, language, document types, and eligibility criteria.
- Duplicate-removal and source-verification process.
- How many records were found, screened, excluded, and included, but only if these counts were actually tracked. Never fabricate PRISMA-style counts.
- If full-text access was limited, disclose this.
- State that 2026 coverage is current only through 9 October 2026.

### Section 4 — Thematic literature review
Organize by research question/theme, not simply year or author:

4.1 Conventional leak detection and signal-processing methods
4.2 Acoustic, vibration, and pressure sensing; multimodal measurement
4.3 Denoising, noise robustness, and weak/marginal leak detection
4.4 Deep temporal models for non-stationary hydraulic and sensor signals
4.5 Graph neural networks and physical topology
4.6 Attention and multimodal sensor fusion
4.7 Sensor sparsity, failure, and missing modalities
4.8 Cross-dataset evaluation, domain adaptation, and real-world generalization
4.9 Localization, leak-type classification, severity estimation, uncertainty, and class imbalance

For each theme:
- Compare methods, datasets, assumptions, and evaluation protocols across multiple papers.
- Explain what is genuinely demonstrated versus asserted by authors.
- Compare metrics only when tasks, datasets, and splits are comparable.
- Highlight contradictory findings, trade-offs, and reproducibility weaknesses.
- Explain concrete relevance to the repository’s proposed architecture.
- Identify limitations that remain unresolved.

### Section 5 — Comparative paper matrix
Create one master table containing all 20–25 papers. Required columns:

1. ID
2. Full paper title
3. Authors
4. Year
5. Venue/status
6. DOI or stable URL
7. Problem/task
8. Dataset/physical setting
9. Sensors/modalities
10. Method/architecture
11. Main findings/metrics (accurately sourced)
12. Limitations/threats to validity
13. Relevance to this project (High/Medium/Low plus a specific reason)

Avoid gigantic, unstructured text in table cells. Use concise but informative summaries and elaborate in the thematic review.

### Section 6 — Cross-paper synthesis
Provide an evidence-based synthesis, including:
- Which sensor modality or combinations seem useful for which leak tasks, and in what settings.
- What denoising approaches do and do not establish.
- Whether and when temporal models help over static classifiers.
- What GNNs add over conventional models or CNN/LSTM approaches, and whether the evidence directly supports physical pipe graphs.
- Fusion designs and handling of sensor reliability.
- Robustness to noise versus robustness to domain shift (do not treat them as equivalent).
- Differences between detection and five-class classification/localization performance.
- Whether published work evaluates real sensor dropout or only synthetic corruption.
- How often papers use scenario-level splits, independent external tests, repeated seeds, confidence intervals, calibration, and statistical significance.

Separate findings that are strongly supported by direct water-network evidence from plausible findings inferred from adjacent domains.

### Section 7 — Research gaps
Identify 5–8 specific, defensible gaps derived from the reviewed evidence. For each gap, include:
- What current papers have already addressed.
- What remains unresolved.
- The exact evidence for calling it a gap (not just “few papers exist”).
- Why it matters in realistic water networks.
- How this project could investigate it.
- A measurable evaluation protocol or metric.

Prioritize gaps such as:
- Joint use of acoustic, vibration, pressure, and physical graph topology.
- Robustness to realistic sensor noise and missing sensor channels.
- Generalization across topologies, flow regimes, and independent datasets.
- Weak gasket/marginal leaks and class imbalance.
- Reliable localization and severity estimation rather than detection alone.
- Fair comparison of complex deep fusion models against strong engineered-feature baselines.
- Uncertainty/calibration and confidence under out-of-distribution conditions.
- Transparent, leakage-safe evaluation on limited physical experiments.

Only call these gaps “novel” if the selected papers genuinely leave them open. Avoid absolute novelty claims without comprehensive evidence.

### Section 8 — Mapping literature to the project
Create a design-justification matrix with columns:
- Project component or decision
- Supporting paper IDs
- Evidence strength (direct / adjacent / preliminary)
- Rationale
- Known risk/limitation
- Recommended experiment or design adjustment

Map at least:
- DAE denoising
- Per-sensor CNN features
- LSTM temporal context
- Physical pipe GNN
- Attention-based modality fusion
- Ensemble and multi-task heads
- Scenario/experiment-level splitting before windowing
- Noise robustness experiments
- Sensor-dropout experiments
- External/cross-dataset evaluation
- Metrics for imbalanced multi-class performance

Be critical. Do not argue that a method is justified merely because it is fashionable or appears in another paper.

### Section 9 — Recommended experiments arising from the survey
Recommend a prioritized, feasible set of experiments grounded in the literature. Explain their evidence-based rationale. Include appropriate metrics:
- Binary detection: precision, recall/sensitivity, specificity, F1, PR-AUC where relevant.
- Five-class classification: macro-F1, per-class precision/recall, confusion matrix, balanced accuracy.
- Localization: clearly defined segment/location accuracy or distance-based error, if ground truth supports it.
- Severity: MAE/RMSE or an appropriate regression metric only if reliable targets exist.
- Denoising: SNR improvement and distortion/signal-preservation measures using genuinely available clean references.
- Robustness: performance across specified noise levels and sensor-dropout patterns.
- Domain shift: untouched external dataset/site and explicit protocol.
- Calibration/uncertainty: calibration error or proper scoring rules if probability outputs are available.

Avoid recommending metrics for tasks when ground-truth labels are not available. Keep recommendations realistic for the limited dataset size and the Sunday deadline.

### Section 10 — Conclusion
Summarize the strongest evidence, remaining uncertainty, and the clearest research direction for this particular project. Do not overstate novelty.

### References
List all selected papers in a consistent citation style (IEEE preferred), with verified DOI/stable links. Ensure one-to-one correspondence between in-text citations, matrix entries, and references. Use numbered citations [1]–[N] consistently.

## 7. Quality and integrity requirements

- Write with the rigor of a good graduate-level engineering review, clear enough for a project team.
- Synthesize, compare, and critique; avoid a sequence of disconnected abstracts.
- Every claim about a specific study must cite the corresponding paper.
- Do not fabricate or overgeneralize quantitative findings.
- Explain experimental setting differences when comparing results.
- Do not interpret high accuracy on a single laboratory/testbed split as proof of deployment readiness.
- Distinguish ground-truth spatial localization from leak classification or detection.
- Distinguish physical sensor removal tests from randomly masking signal values.
- Distinguish added synthetic noise from measured real-world environmental noise.
- Distinguish simulated WDNs from real physical deployments.
- Distinguish a preprint from peer-reviewed work.
- Discuss negative or mixed results, not just successes.
- Do not cite papers solely because their titles contain “leak,” “GNN,” or “multimodal.”
- Do not invent search counts, data, or bibliography metadata.
- Do not treat the repository's claimed project metrics as peer-reviewed evidence.
- Keep project-specific reported figures separate from literature findings.
- Use exact publication year rather than arXiv upload year where a journal/conference version exists; explain the convention.
- Check that all 20–25 references are distinct and traceable.

## 8. Final quality-control checklist — complete before returning

1. Count unique papers. Confirm the final count is between 20 and 25 inclusive.
2. Report the number from 2025 and the number from 2026 separately.
3. Confirm each selected paper has a verified title, author list, year, venue/status, DOI/URL.
4. Confirm each reference is cited in the body and included in the matrix.
5. Confirm no citation was invented and no paper appears twice under alternate versions.
6. Confirm 2026 preprints are labeled accurately and the cutoff date is explicit.
7. Confirm the central narrative synthesizes all key themes and includes project-specific critical analysis.
8. Confirm all claims about model performance are attributed and fairly qualified.
9. Confirm gaps are specific and evidence-based.
10. Deliver the survey with its complete references and the comparative matrix; do not return only a plan or a list of candidate papers.

## 9. Required output files
Deliver:
- `literature_survey.md` — the complete editable survey.
- `paper_matrix.csv` — the same 20–25 papers in structured tabular form.
- `source_verification_log.md` — per-paper metadata verification source, access status (full text / abstract only / paywalled), and any unresolved caveat.
- If possible, a PDF or DOCX version, but the Markdown and CSV are mandatory.

Begin by inspecting the repository and verifying its current technical claims, then search and verify candidate literature. Do not wait for more project context. Start now and prioritize finishing a complete, defensible deliverable before **Sunday, 11 October 2026**.
