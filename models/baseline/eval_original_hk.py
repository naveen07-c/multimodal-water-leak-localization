import torch
import numpy as np
import json
from pathlib import Path
from src.models.full_model import RobustWaterLeakSystem
from src.data.graph_builder import get_network_graph
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
root_dir = Path(r'd:\deep learning project\multimodal-water-leak-localization')
proc_dir = root_dir / 'data' / 'processed'

def evaluate_hk():
    print("--- Evaluating Original Repo Model on Hong Kong Dataset ---")
    test = np.load(proc_dir / 'test_data_hk.npz')
    windows = test['windows'] # (N, 1, 8000)
    y_det = test['y_det']
    
    # Need to reshape to (N_seq, T=5, 6, 8000)
    seq_len = 5
    seqs = []
    y_det_list = []
    
    for i in range(len(windows) - seq_len + 1):
        seq = windows[i:i+seq_len] # (5, 1, 8000)
        # Pad to 6 sensors. A1 is index 1.
        seq_6 = np.zeros((5, 6, 8000), dtype=np.float32)
        seq_6[:, 1:2, :] = seq
        seqs.append(seq_6)
        y_det_list.append(y_det[i + seq_len - 1])
        
    X_seq = torch.tensor(np.stack(seqs, axis=0))
    y_target = np.array(y_det_list)
    
    model = RobustWaterLeakSystem(num_sensors=6, window_len=8000, feat_dim=64, num_classes=5, use_dae=True).to(device)
    model_path = root_dir / 'models/baseline' / 'models' / 'fusion' / 'full_proposed_model.pt'
    if model_path.exists():
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("Loaded trained Original Model.")
    else:
        print("Original Model weights not found! Run trainer_full.py first.")
        return
        
    model.eval()
    graph = get_network_graph('Branched')
    adj_norm_t = torch.tensor(graph['adj_norm'], dtype=torch.float32).to(device)
    
    preds = []
    probs = []
    with torch.no_grad():
        for i in range(0, len(X_seq), 16):
            batch = X_seq[i:i+16].to(device)
            out = model(batch, adj_norm_t, apply_dae=False) # DAE might fail on all-zero sensors
            p = torch.sigmoid(out['det_logit']).view(-1)
            probs.extend(p.cpu().numpy())
            preds.extend((p > 0.5).int().cpu().numpy())
            
    acc = accuracy_score(y_target, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(y_target, preds, average='macro', zero_division=0)
    
    probs = np.nan_to_num(np.array(probs), nan=0.0)
    
    auc = roc_auc_score(y_target, probs)
    cm = confusion_matrix(y_target, preds).tolist()
    
    print(f"[Hong Kong Dataset] Original Model -> Acc: {acc:.4f}, Prec: {prec:.4f}, Rec: {rec:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")
    
    res = {
        'model': 'Original_Repo_Model',
        'dataset': 'hk',
        'accuracy': float(acc),
        'f1_score': float(f1),
        'precision': float(prec),
        'recall': float(rec),
        'auc_roc': float(auc),
        'confusion_matrix': cm
    }
    with open(root_dir / 'models/baseline' / 'results_original_hk.json', 'w') as f:
        json.dump(res, f, indent=4)

if __name__ == '__main__':
    evaluate_hk()
