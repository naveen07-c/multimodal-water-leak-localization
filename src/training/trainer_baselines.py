import os
import sys
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report, roc_auc_score
)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from src.utils.seed import set_seed
from src.data.graph_builder import get_network_graph
from src.models.cnn import CNNClassifier, CNNFeatureExtractor
from src.models.lstm import CNNLSTMClassifier
from src.models.gnn import GNNClassifier, PipelineGNN

set_seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[Trainer] Using compute device: {device}")

root_dir = Path('/home/naveen/Pictures/agy_water_leak')
proc_dir = root_dir / 'processed'
results_dir = root_dir / 'results'
models_dir = root_dir / 'models'
results_dir.mkdir(parents=True, exist_ok=True)
models_dir.mkdir(parents=True, exist_ok=True)

# 1. Load Data
train_data = np.load(proc_dir / 'train_data.npz')
val_data = np.load(proc_dir / 'val_data.npz')
test_data = np.load(proc_dir / 'test_data.npz')

X_train_w = train_data['windows'] # (1711, 6, 8000)
X_val_w = val_data['windows']     # (177, 6, 8000)
X_test_w = test_data['windows']   # (472, 6, 8000)

X_train_f = train_data['feats']   # (1711, 120)
X_val_f = val_data['feats']       # (177, 120)
X_test_f = test_data['feats']     # (472, 120)

y_train_cls = train_data['y_cls']
y_val_cls = val_data['y_cls']
y_test_cls = test_data['y_cls']

y_train_det = train_data['y_det']
y_val_det = val_data['y_det']
y_test_det = test_data['y_det']

graph = get_network_graph('Branched')
adj_norm_t = torch.tensor(graph['adj_norm'], dtype=torch.float32).to(device)

results_records = []

# ==========================================
# BASELINE 1: Statistical Feature Classifiers
# ==========================================
print("\n" + "="*50)
print("EVALUATING BASELINE 1: Feature-Based ML Models")
print("="*50)

# A: Random Forest (Multi-Class Classification)
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train_f, y_train_cls)
p_val_rf = rf.predict(X_val_f)
p_test_rf = rf.predict(X_test_f)
acc_test_rf = accuracy_score(y_test_cls, p_test_rf)
p_prec, p_rec, p_f1, _ = precision_recall_fscore_support(y_test_cls, p_test_rf, average='macro', zero_division=0)
det_acc_rf = accuracy_score(y_test_det, (p_test_rf > 0).astype(int))

print(f"[Baseline 1 - RF] 5-Class Accuracy: {acc_test_rf*100:.2f}%, F1: {p_f1*100:.2f}%, Binary Det Acc: {det_acc_rf*100:.2f}%")
results_records.append({
    'experiment_name': 'Baseline 1 (Random Forest - Statistical Features)',
    'model_type': 'RandomForest',
    'input_type': 'Statistical Features (120-dim)',
    'leak_det_acc': float(det_acc_rf),
    'leak_cls_acc': float(acc_test_rf),
    'precision_macro': float(p_prec),
    'recall_macro': float(p_rec),
    'f1_macro': float(p_f1),
    'val_cls_acc': float(accuracy_score(y_val_cls, p_val_rf))
})

# B: Gradient Boosting
gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
gb.fit(X_train_f, y_train_cls)
p_val_gb = gb.predict(X_val_f)
p_test_gb = gb.predict(X_test_f)
acc_test_gb = accuracy_score(y_test_cls, p_test_gb)
gb_prec, gb_rec, gb_f1, _ = precision_recall_fscore_support(y_test_cls, p_test_gb, average='macro', zero_division=0)
det_acc_gb = accuracy_score(y_test_det, (p_test_gb > 0).astype(int))

print(f"[Baseline 1 - GB] 5-Class Accuracy: {acc_test_gb*100:.2f}%, F1: {gb_f1*100:.2f}%, Binary Det Acc: {det_acc_gb*100:.2f}%")
results_records.append({
    'experiment_name': 'Baseline 1 (Gradient Boosting - Statistical Features)',
    'model_type': 'GradientBoosting',
    'input_type': 'Statistical Features (120-dim)',
    'leak_det_acc': float(det_acc_gb),
    'leak_cls_acc': float(acc_test_gb),
    'precision_macro': float(gb_prec),
    'recall_macro': float(gb_rec),
    'f1_macro': float(gb_f1),
    'val_cls_acc': float(accuracy_score(y_val_cls, p_val_gb))
})

# ==========================================
# BASELINE 2: 1D-CNN Only
# ==========================================
print("\n" + "="*50)
print("EVALUATING BASELINE 2: 1D-CNN Only")
print("="*50)

# PyTorch Datasets
train_ds = TensorDataset(torch.tensor(X_train_w, dtype=torch.float32), torch.tensor(y_train_cls, dtype=torch.long), torch.tensor(y_train_det, dtype=torch.float32))
val_ds = TensorDataset(torch.tensor(X_val_w, dtype=torch.float32), torch.tensor(y_val_cls, dtype=torch.long), torch.tensor(y_val_det, dtype=torch.float32))
test_ds = TensorDataset(torch.tensor(X_test_w, dtype=torch.float32), torch.tensor(y_test_cls, dtype=torch.long), torch.tensor(y_test_det, dtype=torch.float32))

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)
test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

cnn_model = CNNClassifier(in_channels=6, num_classes=5, feat_dim=64).to(device)
criterion_cls = nn.CrossEntropyLoss()
criterion_det = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(cnn_model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

best_val_acc = 0.0
best_cnn_state = None

for epoch in range(1, 21):
    cnn_model.train()
    total_loss = 0.0
    for bx, by_cls, by_det in train_loader:
        bx, by_cls, by_det = bx.to(device), by_cls.to(device), by_det.to(device)
        optimizer.zero_grad()
        out_cls, out_det, _ = cnn_model(bx)
        loss = criterion_cls(out_cls, by_cls) + 0.5 * criterion_det(out_det, by_det)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
    # Evaluate Validation
    cnn_model.eval()
    val_preds, val_targets = [], []
    with torch.no_grad():
        for bx, by_cls, _ in val_loader:
            bx = bx.to(device)
            out_cls, _, _ = cnn_model(bx)
            val_preds.extend(torch.argmax(out_cls, dim=1).cpu().numpy())
            val_targets.extend(by_cls.numpy())
            
    val_acc = accuracy_score(val_targets, val_preds)
    scheduler.step(val_acc)
    if val_acc >= best_val_acc:
        best_val_acc = val_acc
        best_cnn_state = cnn_model.state_dict().copy()

cnn_model.load_state_dict(best_cnn_state)
torch.save(best_cnn_state, models_dir / 'baseline2_cnn.pt')

# Evaluate Test
cnn_model.eval()
test_preds, test_dets, test_targets, test_det_targets = [], [], [], []
with torch.no_grad():
    for bx, by_cls, by_det in test_loader:
        bx = bx.to(device)
        out_cls, out_det, _ = cnn_model(bx)
        test_preds.extend(torch.argmax(out_cls, dim=1).cpu().numpy())
        test_dets.extend((torch.sigmoid(out_det) > 0.5).int().cpu().numpy())
        test_targets.extend(by_cls.numpy())
        test_det_targets.extend(by_det.numpy())

acc_cnn = accuracy_score(test_targets, test_preds)
det_acc_cnn = accuracy_score(test_det_targets, test_dets)
c_prec, c_rec, c_f1, _ = precision_recall_fscore_support(test_targets, test_preds, average='macro', zero_division=0)
print(f"[Baseline 2 - CNN] 5-Class Accuracy: {acc_cnn*100:.2f}%, F1: {c_f1*100:.2f}%, Binary Det Acc: {det_acc_cnn*100:.2f}%")

results_records.append({
    'experiment_name': 'Baseline 2 (1D-CNN Only)',
    'model_type': '1D-CNN',
    'input_type': 'Raw Multimodal Windows (6x8000)',
    'leak_det_acc': float(det_acc_cnn),
    'leak_cls_acc': float(acc_cnn),
    'precision_macro': float(c_prec),
    'recall_macro': float(c_rec),
    'f1_macro': float(c_f1),
    'val_cls_acc': float(best_val_acc)
})

# ==========================================
# BASELINE 3: CNN-LSTM Temporal Model
# ==========================================
print("\n" + "="*50)
print("EVALUATING BASELINE 3: CNN-LSTM Temporal Model")
print("="*50)

# Build sequence dataset (T=5 consecutive windows)
def make_sequences(windows_arr, labels_arr, seq_len=5):
    # windows: (N, 6, 8000)
    seqs, seq_labels_cls, seq_labels_det = [], [], []
    for i in range(len(windows_arr) - seq_len + 1):
        seqs.append(windows_arr[i:i+seq_len])
        seq_labels_cls.append(labels_arr['y_cls'][i+seq_len-1])
        seq_labels_det.append(labels_arr['y_det'][i+seq_len-1])
    return np.stack(seqs, axis=0), np.array(seq_labels_cls), np.array(seq_labels_det)

X_train_seq, y_train_seq_cls, y_train_seq_det = make_sequences(X_train_w, train_data, seq_len=5)
X_val_seq, y_val_seq_cls, y_val_seq_det = make_sequences(X_val_w, val_data, seq_len=5)
X_test_seq, y_test_seq_cls, y_test_seq_det = make_sequences(X_test_w, test_data, seq_len=5)

train_seq_loader = DataLoader(TensorDataset(torch.tensor(X_train_seq, dtype=torch.float32), torch.tensor(y_train_seq_cls, dtype=torch.long), torch.tensor(y_train_seq_det, dtype=torch.float32)), batch_size=16, shuffle=True)
val_seq_loader = DataLoader(TensorDataset(torch.tensor(X_val_seq, dtype=torch.float32), torch.tensor(y_val_seq_cls, dtype=torch.long), torch.tensor(y_val_seq_det, dtype=torch.float32)), batch_size=16, shuffle=False)
test_seq_loader = DataLoader(TensorDataset(torch.tensor(X_test_seq, dtype=torch.float32), torch.tensor(y_test_seq_cls, dtype=torch.long), torch.tensor(y_test_seq_det, dtype=torch.float32)), batch_size=16, shuffle=False)

lstm_model = CNNLSTMClassifier(in_channels=6, num_classes=5, cnn_dim=64, lstm_dim=64).to(device)
optimizer_lstm = optim.Adam(lstm_model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler_lstm = optim.lr_scheduler.ReduceLROnPlateau(optimizer_lstm, mode='max', factor=0.5, patience=3)

best_val_acc_lstm = 0.0
best_lstm_state = None

for epoch in range(1, 21):
    lstm_model.train()
    for bx_seq, by_cls, by_det in train_seq_loader:
        bx_seq, by_cls, by_det = bx_seq.to(device), by_cls.to(device), by_det.to(device)
        optimizer_lstm.zero_grad()
        out_cls, out_det, _, _ = lstm_model(bx_seq)
        loss = criterion_cls(out_cls, by_cls) + 0.5 * criterion_det(out_det, by_det)
        loss.backward()
        optimizer_lstm.step()
        
    lstm_model.eval()
    val_preds = []
    with torch.no_grad():
        for bx_seq, by_cls, _ in val_seq_loader:
            bx_seq = bx_seq.to(device)
            out_cls, _, _, _ = lstm_model(bx_seq)
            val_preds.extend(torch.argmax(out_cls, dim=1).cpu().numpy())
            
    val_acc = accuracy_score(y_val_seq_cls, val_preds)
    scheduler_lstm.step(val_acc)
    if val_acc >= best_val_acc_lstm:
        best_val_acc_lstm = val_acc
        best_lstm_state = lstm_model.state_dict().copy()

lstm_model.load_state_dict(best_lstm_state)
torch.save(best_lstm_state, models_dir / 'baseline3_cnn_lstm.pt')

# Evaluate Test
lstm_model.eval()
test_preds_lstm, test_dets_lstm = [], []
with torch.no_grad():
    for bx_seq, by_cls, by_det in test_seq_loader:
        bx_seq = bx_seq.to(device)
        out_cls, out_det, _, _ = lstm_model(bx_seq)
        test_preds_lstm.extend(torch.argmax(out_cls, dim=1).cpu().numpy())
        test_dets_lstm.extend((torch.sigmoid(out_det) > 0.5).int().cpu().numpy())

acc_lstm = accuracy_score(y_test_seq_cls, test_preds_lstm)
det_acc_lstm = accuracy_score(y_test_seq_det, test_dets_lstm)
l_prec, l_rec, l_f1, _ = precision_recall_fscore_support(y_test_seq_cls, test_preds_lstm, average='macro', zero_division=0)
print(f"[Baseline 3 - CNN-LSTM] 5-Class Accuracy: {acc_lstm*100:.2f}%, F1: {l_f1*100:.2f}%, Binary Det Acc: {det_acc_lstm*100:.2f}%")

results_records.append({
    'experiment_name': 'Baseline 3 (CNN-LSTM Temporal Model)',
    'model_type': 'CNN-LSTM',
    'input_type': 'Temporal Sequence (5 x 6 x 8000)',
    'leak_det_acc': float(det_acc_lstm),
    'leak_cls_acc': float(acc_lstm),
    'precision_macro': float(l_prec),
    'recall_macro': float(l_rec),
    'f1_macro': float(l_f1),
    'val_cls_acc': float(best_val_acc_lstm)
})

# ==========================================
# BASELINE 4: Spatial GNN Model
# ==========================================
print("\n" + "="*50)
print("EVALUATING BASELINE 4: Spatial GNN Model")
print("="*50)

class SpatialGNNWrapper(nn.Module):
    def __init__(self, in_channels=1, feat_dim=64, num_classes=5):
        super(SpatialGNNWrapper, self).__init__()
        # Per-node CNN
        self.node_cnns = nn.ModuleList([CNNFeatureExtractor(in_channels=1, feat_dim=feat_dim) for _ in range(6)])
        self.gnn = PipelineGNN(in_features=feat_dim, hidden_dim=feat_dim, gnn_dim=feat_dim)
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(feat_dim, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )
        self.det_head = nn.Linear(feat_dim, 1)
        
    def forward(self, x, adj_norm):
        # x: (B, 6, L)
        B, N, L = x.shape
        node_feats = []
        for i in range(N):
            s_sig = x[:, i:i+1, :] # (B, 1, L)
            s_f = self.node_cnns[i](s_sig) # (B, feat_dim)
            node_feats.append(s_f)
        node_feats = torch.stack(node_feats, dim=1) # (B, 6, feat_dim)
        
        spatial_feat, node_scores, _ = self.gnn(node_feats, adj_norm)
        cls_logits = self.classifier(spatial_feat)
        det_logit = self.det_head(spatial_feat).squeeze(-1)
        return cls_logits, det_logit, node_scores

gnn_model = SpatialGNNWrapper(in_channels=1, feat_dim=64, num_classes=5).to(device)
optimizer_gnn = optim.Adam(gnn_model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler_gnn = optim.lr_scheduler.ReduceLROnPlateau(optimizer_gnn, mode='max', factor=0.5, patience=3)

best_val_acc_gnn = 0.0
best_gnn_state = None

for epoch in range(1, 21):
    gnn_model.train()
    for bx, by_cls, by_det in train_loader:
        bx, by_cls, by_det = bx.to(device), by_cls.to(device), by_det.to(device)
        optimizer_gnn.zero_grad()
        out_cls, out_det, _ = gnn_model(bx, adj_norm_t)
        loss = criterion_cls(out_cls, by_cls) + 0.5 * criterion_det(out_det, by_det)
        loss.backward()
        optimizer_gnn.step()
        
    gnn_model.eval()
    val_preds = []
    with torch.no_grad():
        for bx, by_cls, _ in val_loader:
            bx = bx.to(device)
            out_cls, _, _ = gnn_model(bx, adj_norm_t)
            val_preds.extend(torch.argmax(out_cls, dim=1).cpu().numpy())
            
    val_acc = accuracy_score(y_val_cls, val_preds)
    scheduler_gnn.step(val_acc)
    if val_acc >= best_val_acc_gnn:
        best_val_acc_gnn = val_acc
        best_gnn_state = gnn_model.state_dict().copy()

gnn_model.load_state_dict(best_gnn_state)
torch.save(best_gnn_state, models_dir / 'baseline4_spatial_gnn.pt')

# Evaluate Test
gnn_model.eval()
test_preds_gnn, test_dets_gnn = [], []
with torch.no_grad():
    for bx, by_cls, by_det in test_loader:
        bx = bx.to(device)
        out_cls, out_det, _ = gnn_model(bx, adj_norm_t)
        test_preds_gnn.extend(torch.argmax(out_cls, dim=1).cpu().numpy())
        test_dets_gnn.extend((torch.sigmoid(out_det) > 0.5).int().cpu().numpy())

acc_gnn = accuracy_score(y_test_cls, test_preds_gnn)
det_acc_gnn = accuracy_score(y_test_det, test_dets_gnn)
g_prec, g_rec, g_f1, _ = precision_recall_fscore_support(y_test_cls, test_preds_gnn, average='macro', zero_division=0)
print(f"[Baseline 4 - GNN] 5-Class Accuracy: {acc_gnn*100:.2f}%, F1: {g_f1*100:.2f}%, Binary Det Acc: {det_acc_gnn*100:.2f}%")

results_records.append({
    'experiment_name': 'Baseline 4 (Spatial GNN Model)',
    'model_type': 'Spatial-GNN',
    'input_type': 'Graph Node Windows (6 nodes x 8000)',
    'leak_det_acc': float(det_acc_gnn),
    'leak_cls_acc': float(acc_gnn),
    'precision_macro': float(g_prec),
    'recall_macro': float(g_rec),
    'f1_macro': float(g_f1),
    'val_cls_acc': float(best_val_acc_gnn)
})

# Save Central Results Table
df_res = pd.DataFrame(results_records)
df_res.to_csv(results_dir / 'baseline_results.csv', index=False)
df_res.to_csv(root_dir / '06_baseline_results.csv', index=False)
print("\n" + "="*50)
print("BASELINE EVALUATION SUMMARY TABLE")
print("="*50)
print(df_res.to_string(index=False))
