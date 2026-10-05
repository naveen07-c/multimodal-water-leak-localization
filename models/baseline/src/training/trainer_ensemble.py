import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
from src.utils.seed import set_seed
from src.data.graph_builder import get_network_graph
from src.models.full_model import RobustWaterLeakSystem
from src.models.ensemble import EnsembleDecisionLayer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

set_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

root_dir = Path('d:\deep learning project\multimodal-water-leak-localization')
proc_dir = root_dir / 'data' / 'processed'
models_dir = root_dir / 'models/baseline' / 'models'
results_dir = root_dir / 'models/baseline' / 'results'
figs_dir = root_dir / 'models/baseline' / 'figures'

train_data = np.load(proc_dir / 'train_data.npz')
val_data = np.load(proc_dir / 'val_data.npz')
test_data = np.load(proc_dir / 'test_data.npz')

# 1. Train Gradient Boosting Model
gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
gb.fit(train_data['feats'], train_data['y_cls'])
p_val_gb = gb.predict_proba(val_data['feats']) # (177, 5)
p_test_gb = gb.predict_proba(test_data['feats']) # (472, 5)

# 2. Load Full Proposed Deep Model
full_model = RobustWaterLeakSystem(num_sensors=6, window_len=8000, feat_dim=64, num_classes=5, use_dae=True).to(device)
full_model.load_state_dict(torch.load(models_dir / 'fusion/full_proposed_model.pt', map_location=device))
full_model.eval()

graph = get_network_graph('Branched')
adj_norm_t = torch.tensor(graph['adj_norm'], dtype=torch.float32).to(device)

def get_deep_probs(data_npz, seq_len=5):
    windows_arr = data_npz['windows'] # (N, 6, 8000)
    seqs = []
    for i in range(len(windows_arr) - seq_len + 1):
        seqs.append(windows_arr[i:i+seq_len])
    seqs_arr = np.stack(seqs, axis=0).astype(np.float32)
    
    # Batch predict
    probs = []
    with torch.no_grad():
        for i in range(0, len(seqs_arr), 16):
            b = torch.tensor(seqs_arr[i:i+16]).to(device)
            out = full_model(b, adj_norm_t)
            p = torch.softmax(out['cls_logits'], dim=-1).cpu().numpy()
            probs.append(p)
    probs_arr = np.concatenate(probs, axis=0) # (N_seq, 5)
    
    # Pad first seq_len-1 windows with the first probability
    pad_probs = np.repeat(probs_arr[:1], seq_len - 1, axis=0)
    full_probs = np.concatenate([pad_probs, probs_arr], axis=0)
    return full_probs

p_val_deep = get_deep_probs(val_data, seq_len=5)
p_test_deep = get_deep_probs(test_data, seq_len=5)

# 3. Train Random Forest Model
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(train_data['feats'], train_data['y_cls'])
p_val_rf = rf.predict_proba(val_data['feats'])
p_test_rf = rf.predict_proba(test_data['feats'])

# 4. Ensemble Calibration on Validation Data
ensemble = EnsembleDecisionLayer()
weights = ensemble.fit_weights(p_val_deep, p_val_rf, p_val_gb, val_data['y_cls'])

# Evaluate Ensemble on Test Set
p_final_test = ensemble.predict(p_test_deep, p_test_rf, p_test_gb)
ens_cls_acc = accuracy_score(test_data['y_cls'], p_final_test)
ens_det_acc = accuracy_score(test_data['y_det'], (p_final_test > 0).astype(int))
prec, rec, f1, _ = precision_recall_fscore_support(test_data['y_cls'], p_final_test, average='macro', zero_division=0)
cm = confusion_matrix(test_data['y_cls'], p_final_test)

print("\n" + "="*50)
print("FINAL ENSEMBLE DECISION LAYER RESULTS (TEST SET)")
print("="*50)
print(f"Ensemble 5-Class Accuracy: {ens_cls_acc*100:.2f}%")
print(f"Ensemble Binary Det Acc  : {ens_det_acc*100:.2f}%")
print(f"Macro Precision          : {prec*100:.2f}%")
print(f"Macro Recall             : {rec*100:.2f}%")
print(f"Macro F1-Score           : {f1*100:.2f}%")
print("\nConfusion Matrix:")
print(cm)

# Save Confusion Matrix Figure
class_names = ['No Leak (NL)', 'Circum. Crack (CC)', 'Gasket Leak (GL)', 'Long. Crack (LC)', 'Orifice Leak (OL)']
fig, ax = plt.subplots(figsize=(8, 6))
cax = ax.matshow(cm, cmap='Blues')
fig.colorbar(cax)
for i in range(len(class_names)):
    for j in range(len(class_names)):
        ax.text(j, i, str(cm[i, j]), ha='center', va='center', color='white' if cm[i, j] > cm.max()/2 else 'black', fontweight='bold')
ax.set_xticks(range(len(class_names)))
ax.set_yticks(range(len(class_names)))
ax.set_xticklabels(class_names, rotation=30, ha='left')
ax.set_yticklabels(class_names)
plt.title('Confusion Matrix: Final Calibrated Ensemble Model (Test Set)', fontsize=12, fontweight='bold', pad=20)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
plt.savefig(figs_dir / 'confusion_matrices/ensemble_confusion_matrix.png', dpi=200)
plt.close()

# Update Ablation Table with Final Ensemble Row
df_abl = pd.read_csv(results_dir / 'ablation_results.csv')
ens_row = {
    'Experiment': '6. Full Proposed Ensemble (DAE-CNN-LSTM-GNN + Statistical GB/RF)',
    'DAE': '✓', 'CNN': '✓', 'LSTM': '✓', 'GNN': '✓', 'Attention': '✓',
    'Leak_Det_Acc': float(ens_det_acc),
    'Leak_Cls_Acc': float(ens_cls_acc),
    'F1_Score': float(f1)
}
df_abl = pd.concat([df_abl, pd.DataFrame([ens_row])], ignore_index=True)
df_abl.to_csv(results_dir / 'ablation_results.csv', index=False)
df_abl.to_csv(root_dir / '09_ablation_results.csv', index=False)
print("\nUpdated Ablation Table with Full Ensemble:")
print(df_abl.to_string(index=False))
