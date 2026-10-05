import os
import glob
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

root_dir = Path(r'd:\deep learning project\multimodal-water-leak-localization')
hk_dir = root_dir / 'datasets' / 'hong kong data set' / 'MEMS Accelerometers'
proc_dir = root_dir / 'data' / 'processed'
proc_dir.mkdir(parents=True, exist_ok=True)

WINDOW_LEN = 8000
STRIDE = 4000

def process_hk_data():
    windows = []
    labels = []
    
    # Process Leak (Label 1)
    leak_files = glob.glob(str(hk_dir / 'Leak' / '*.xlsx'))
    for f in leak_files:
        df = pd.read_excel(f)
        if 'acceleration value' in df.columns:
            sig = df['acceleration value'].values.astype(np.float32)
        else:
            sig = df.iloc[:, 1].values.astype(np.float32)
            
        # Pad if too short
        if len(sig) < WINDOW_LEN:
            sig = np.pad(sig, (0, WINDOW_LEN - len(sig)), mode='edge')
            
        num_windows = (len(sig) - WINDOW_LEN) // STRIDE + 1
        for w_i in range(num_windows):
            start = w_i * STRIDE
            w = sig[start:start+WINDOW_LEN]
            windows.append(w)
            labels.append(1)

    # Process No-Leak (Label 0)
    noleak_files = glob.glob(str(hk_dir / 'No-Leak' / '*.xlsx'))
    for f in noleak_files:
        df = pd.read_excel(f)
        if 'acceleration value' in df.columns:
            sig = df['acceleration value'].values.astype(np.float32)
        else:
            sig = df.iloc[:, 1].values.astype(np.float32)
            
        if len(sig) < WINDOW_LEN:
            sig = np.pad(sig, (0, WINDOW_LEN - len(sig)), mode='edge')
            
        num_windows = (len(sig) - WINDOW_LEN) // STRIDE + 1
        for w_i in range(num_windows):
            start = w_i * STRIDE
            w = sig[start:start+WINDOW_LEN]
            windows.append(w)
            labels.append(0)
            
    windows = np.array(windows)
    labels = np.array(labels)
    
    # Expand dims to simulate channel (N, 1, 8000)
    windows = np.expand_dims(windows, axis=1)
    
    print(f"Total HK Windows: {len(windows)}, Leak: {np.sum(labels==1)}, No-Leak: {np.sum(labels==0)}")
    
    # Split into train/val/test
    # 80/10/10 split
    X_temp, X_test, y_temp, y_test = train_test_split(windows, labels, test_size=0.1, stratify=labels, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.1111, stratify=y_temp, random_state=42)
    
    print(f"HK Splits -> Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # Normalization
    ch_mean = np.mean(X_train, axis=(0, 2), keepdims=True)
    ch_std = np.std(X_train, axis=(0, 2), keepdims=True) + 1e-8
    
    X_train_norm = ((X_train - ch_mean) / ch_std).astype(np.float32)
    X_val_norm = ((X_val - ch_mean) / ch_std).astype(np.float32)
    X_test_norm = ((X_test - ch_mean) / ch_std).astype(np.float32)
    
    # Save
    np.savez_compressed(proc_dir / 'train_data_hk.npz', windows=X_train_norm, y_det=y_train)
    np.savez_compressed(proc_dir / 'val_data_hk.npz', windows=X_val_norm, y_det=y_val)
    np.savez_compressed(proc_dir / 'test_data_hk.npz', windows=X_test_norm, y_det=y_test)
    
    print("Saved HK dataset successfully.")

if __name__ == '__main__':
    process_hk_data()
