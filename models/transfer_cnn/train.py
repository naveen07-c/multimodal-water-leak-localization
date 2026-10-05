import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
import numpy as np
import json
import os
from pathlib import Path
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
root_dir = Path(r'd:\deep learning project\multimodal-water-leak-localization')
proc_dir = root_dir / 'data' / 'processed'

# 1. Model Definition (IoT Transfer CNN)
class IoTCNN(nn.Module):
    def __init__(self, in_channels=1):
        super(IoTCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def load_data(dataset_type='mendeley'):
    if dataset_type == 'mendeley':
        train = np.load(proc_dir / 'train_data.npz')
        test = np.load(proc_dir / 'test_data.npz')
        # Use A1 sensor (index 1) for Mendeley if we want single channel, or reshape all
        # Let's just use channel 1 (A1) to be similar to the IoT vibro-acoustic setup
        X_train = train['windows'][:, 1:2, :] # (N, 1, 8000)
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
    print(f"\n--- Training Model 2 (IoT Transfer CNN) on {dataset_type} dataset ---")
    X_train, y_train, X_test, y_test = load_data(dataset_type)
    
    # Convert to Mel-Spectrogram using torchaudio
    mel_transform = torchaudio.transforms.MelSpectrogram(sample_rate=8000, n_fft=400, hop_length=160, n_mels=64)
    
    X_train_t = torch.tensor(X_train)
    X_test_t = torch.tensor(X_test)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.float32)
    
    # Apply transform
    X_train_mel = mel_transform(X_train_t) # (N, 1, 64, time_steps)
    X_test_mel = mel_transform(X_test_t)
    
    train_loader = DataLoader(TensorDataset(X_train_mel, y_train_t), batch_size=32, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test_mel, y_test_t), batch_size=32, shuffle=False)
    
    model = IoTCNN(in_channels=1).to(device)
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
        'model': 'models/transfer_cnn',
        'dataset': dataset_type,
        'accuracy': float(acc),
        'f1_score': float(f1),
        'precision': float(prec),
        'recall': float(rec),
        'auc_roc': float(auc),
        'confusion_matrix': cm
    }
    
    out_dir = root_dir / 'models/transfer_cnn'
    torch.save(model.state_dict(), out_dir / f'model_{dataset_type}.pt')
    with open(out_dir / f'results_{dataset_type}.json', 'w') as f:
        json.dump(results, f, indent=4)

if __name__ == '__main__':
    train_model('mendeley')
    train_model('hk')
