import json
import pandas as pd
from pathlib import Path

root_dir = Path(r'd:\deep learning project\multimodal-water-leak-localization')
comp_dir = root_dir / 'results'

def extract_original_mendeley():
    # From results/fusion_results.csv
    try:
        df = pd.read_csv(root_dir / 'models/baseline' / 'results' / 'fusion_results.csv')
        return {
            'Model': 'Original_Repo_Model',
            'Dataset': 'Mendeley',
            'Accuracy': df['leak_det_acc'].values[0] if 'leak_det_acc' in df.columns else None,
            'F1_Score': df['f1_macro'].values[0] if 'f1_macro' in df.columns else None,
            'Precision': df['precision'].values[0] if 'precision' in df.columns else None,
            'Recall': df['recall'].values[0] if 'recall' in df.columns else None,
            'AUC_ROC': df['auc_roc'].values[0] if 'auc_roc' in df.columns else None,
            'Confusion_Matrix': df['confusion_matrix'].values[0] if 'confusion_matrix' in df.columns else None
        }
    except Exception as e:
        return {'Model': 'Original_Repo_Model', 'Dataset': 'Mendeley', 'Error': str(e)}

def load_json_result(filepath):
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        return {
            'Model': data.get('model', 'Unknown'),
            'Dataset': data.get('dataset', 'Unknown'),
            'Accuracy': data.get('accuracy', None),
            'F1_Score': data.get('f1_score', None),
            'Precision': data.get('precision', None),
            'Recall': data.get('recall', None),
            'AUC_ROC': data.get('auc_roc', None),
            'Confusion_Matrix': data.get('confusion_matrix', None)
        }
    except Exception as e:
        return {'Error': str(e), 'File': str(filepath)}

results = []

# 1. Original Mendeley
results.append(extract_original_mendeley())

# 2. Original HK
results.append(load_json_result(root_dir / 'models/baseline' / 'results_original_hk.json'))

# 3. Model 1
results.append(load_json_result(root_dir / 'models/hybrid_ml' / 'results_mendeley.json'))
results.append(load_json_result(root_dir / 'models/hybrid_ml' / 'results_hk.json'))

# 4. Model 2
results.append(load_json_result(root_dir / 'models/transfer_cnn' / 'results_mendeley.json'))
results.append(load_json_result(root_dir / 'models/transfer_cnn' / 'results_hk.json'))

# 5. Model 3
results.append(load_json_result(root_dir / 'models/safnet' / 'results_mendeley.json'))
results.append(load_json_result(root_dir / 'models/safnet' / 'results_hk.json'))

df = pd.DataFrame(results)
print("\n" + "="*50)
print("FINAL COMPARISON OF ALL 4 MODELS ACROSS 2 DATASETS")
print("="*50)
print(df.to_string(index=False))

df.to_csv(root_dir / 'results' / 'final_comparison_results.csv', index=False)
