# Final Comparison of All Models Across Datasets

## Table 1: Mendeley Dataset (Original In-Domain) Comparison

| Model | Accuracy | F1_Score | Precision | Recall | AUC_ROC | Confusion_Matrix |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Original_Repo_Model** | 0.9380 | 0.4344 | 0.4247 | 0.4986 | 0.9975 | `[[59, 0, 0, 0, 0], [0, 90, 19, 4, 1], [0, 33, 0, 26, 0], [0, 55, 25, 15, 23], [44, 0, 2, 4, 68]]` |
| **Model_1_Transparent_Hybrid** | 0.7903 | 0.4414 | 0.4317 | 0.4516 | 0.4250 | `[[0, 59], [40, 373]]` |
| **Model_2_IoT_Transfer_CNN** | 0.8729 | 0.4661 | 0.4374 | 0.4988 | 0.9171 | `[[0, 59], [1, 412]]` |
| **Model_3_SAFNet** | 0.7585 | 0.4313 | 0.4293 | 0.4334 | 0.8295 | `[[0, 59], [55, 358]]` |

## Table 2: Hong Kong (hk) Dataset (Zero-Shot Transfer) Comparison

| Model | Accuracy | F1_Score | Precision | Recall | AUC_ROC | Confusion_Matrix |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Original_Repo_Model** | 0.5167 | 0.3407 | 0.2583 | 0.5000 | 0.5000 | `[[186, 0], [174, 0]]` |
| **Model_1_Transparent_Hybrid** | 0.4808 | 0.3247 | 0.2404 | 0.5000 | 0.5000 | `[[0, 189], [0, 175]]` |
| **Model_2_IoT_Transfer_CNN** | 0.5192 | 0.3418 | 0.2596 | 0.5000 | 0.5000 | `[[189, 0], [175, 0]]` |
| **Model_3_SAFNet** | 0.5192 | 0.3418 | 0.2596 | 0.5000 | 0.5000 | `[[189, 0], [175, 0]]` |
