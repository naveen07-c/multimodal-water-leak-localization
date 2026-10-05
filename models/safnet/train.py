import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
import os
from pathlib import Path
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
root_dir = Path(r'd:\deep learning project\multimodal-water-leak-localization')
proc_dir = root_dir / 'data' / 'processed'

# 1. SAFNet Architecture Components
class TPE(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        channels = [in_channels, 16, 32, 64, 128, 256]
        self.layers = nn.ModuleList()
        for i in range(5):
            self.layers.append(
                nn.Sequential(
                    nn.Conv1d(channels[i], channels[i+1], kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm1d(channels[i+1]),
                    nn.ReLU()
                )
            )
            
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class PLSA(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.dwconv = nn.Conv1d(channels, channels, kernel_size=3, padding=1, groups=channels)
        self.W_c = nn.Parameter(torch.randn(1, channels)) # simplified DBSM
        self.W_p = nn.Parameter(torch.randn(1, channels))
        
    def forward(self, x):
        res = x
        x = self.dwconv(x)
        # simplified star mapping
        x = x * self.W_c.unsqueeze(2) + x * self.W_p.unsqueeze(2)
        x = x + res
        
        # simplified EDFR (energy-driven feature refiner)
        mu = torch.mean(x, dim=2, keepdim=True)
        var = torch.var(x, dim=2, keepdim=True) + 1e-6
        energy = torch.abs(mu) / (torch.sqrt(var) + 1e-6)
        weight = torch.sigmoid(energy)
        
        return x * weight

class SAFNet(nn.Module):
    def __init__(self, in_channels=1):
        super().__init__()
        self.tpe = TPE(in_channels)
        self.plsa1 = PLSA(256)
        self.plsa2 = PLSA(256)
        self.plsa3 = PLSA(256)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(256, 1)
        )
        
    def forward(self, x):
        x = self.tpe(x)
        f1 = self.plsa1(x)
        f2 = self.plsa2(f1)
        f3 = self.plsa3(f2)
        
        # MS-GAF (simplified fusion by averaging scales)
        f_fused = (f1 + f2 + f3) / 3.0
        
        f_pool = torch.mean(f_fused, dim=2)
        out = self.classifier(f_pool)
        return out

def load_data(dataset_type='mendeley'):
    if dataset_type == 'mendeley':
        train = np.load(proc_dir / 'train_data.npz')
        test = np.load(proc_dir / 'test_data.npz')
        X_train = train['windows'][:, 1:2, :] # (N, 1, 8000) using A1
        X_test = test['windows'][:, 1:2, :]
        y_train = train['y_det']
        y_test = test['y_det']
    else:
        train = np.load(proc_dir / 'train_data_hk.npz')
        test = np.load(proc_dir / 'test_data_hk.npz')
        X_train = train['windows'] # (N, 1, 8000)
        X_test = test['windows']
        y_train = train['y_det']
        y_test = test['y_det']
        
    return np.nan_to_num(X_train, nan=0.0), y_train, np.nan_to_num(X_test, nan=0.0), y_test

def train_model(dataset_type):
    print(f"\n--- Training Model 3 (SAFNet) on {dataset_type} dataset ---")
    X_train, y_train, X_test, y_test = load_data(dataset_type)
    
    X_train_t = torch.tensor(X_train)
    X_test_t = torch.tensor(X_test)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.float32)
    
    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=32, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test_t, y_test_t), batch_size=32, shuffle=False)
    
    model = SAFNet(in_channels=1).to(device)
    num_pos = (y_train_t == 1).sum()
    pos_weight = (y_train_t == 0).sum() / num_pos if num_pos > 0 else torch.tensor(1.0)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    # Finetuning model to increase efficiency
    for epoch in range(20):
        model.train()
        total_loss = 0
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            out = model(X_b).squeeze()
            loss = criterion(out, y_b)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1} Loss: {total_loss/len(train_loader):.4f}")
        
    # Eval
    model.eval()
    preds = []
    probs = []
    targets = []
    with torch.no_grad():
        for X_b, y_b in test_loader:
            X_b = X_b.to(device)
            out = torch.sigmoid(model(X_b).squeeze())
            probs.extend(out.cpu().numpy())
            preds.extend((out > 0.5).int().cpu().numpy())
            targets.extend(y_b.numpy())
            
    acc = accuracy_score(targets, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(targets, preds, average='macro', zero_division=0)
    
    try:
        auc = roc_auc_score(targets, probs)
    except ValueError:
        auc = float('nan')
    cm = confusion_matrix(targets, preds).tolist()
    
    print(f"[{dataset_type}] Acc: {acc:.4f}, Prec: {prec:.4f}, Rec: {rec:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")
    
    results = {
        'model': 'models/safnet',
        'dataset': dataset_type,
        'accuracy': float(acc),
        'f1_score': float(f1),
        'precision': float(prec),
        'recall': float(rec),
        'auc_roc': float(auc),
        'confusion_matrix': cm
    }
    
    out_dir = root_dir / 'models/safnet'
    torch.save(model.state_dict(), out_dir / f'model_{dataset_type}.pt')
    with open(out_dir / f'results_{dataset_type}.json', 'w') as f:
        json.dump(results, f, indent=4)

if __name__ == '__main__':
    train_model('mendeley')
    train_model('hk')
