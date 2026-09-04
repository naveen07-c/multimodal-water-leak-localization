import os
import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd


from src.data.dataset import build_experiment_tensor, slice_windows, NODE_NAMES
from src.models.baselines import extract_multimodal_features

root_dir = Path('/home/naveen/Pictures/agy_water_leak')
meta_dir = root_dir / 'metadata'
proc_dir = root_dir / 'processed'
proc_dir.mkdir(parents=True, exist_ok=True)

# Load split files
train_df = pd.read_csv(meta_dir / '03_train_ids.csv')
val_df = pd.read_csv(meta_dir / '04_val_ids.csv')
test_df = pd.read_csv(meta_dir / '05_test_ids.csv')

print(f"Loaded splits -> Train: {train_df['experiment_id'].nunique()} exps, Val: {val_df['experiment_id'].nunique()} exps, Test: {test_df['experiment_id'].nunique()} exps")

def process_split(split_df, split_name):
    # Only multimodal S1 experiments (which contain all 6 sensors)
    s1_exps = []
    for exp_id, group in split_df.groupby('experiment_id'):
        if set(group['sensor_id']) == set(NODE_NAMES):
            s1_exps.append(group)
            
    print(f"[{split_name}] Found {len(s1_exps)} full multimodal (S1) scenarios")
    
    all_windows = []
    all_feats = []
    all_y_det = []
    all_y_cls = []
    all_y_loc = []
    all_y_sev = []
    all_top = []
    all_flow = []
    all_noise = []
    all_exp_ids = []
    
    for grp in s1_exps:
        exp_tensor, meta = build_experiment_tensor(root_dir, grp)
        windows, labels = slice_windows(exp_tensor, meta, window_len=8000, stride=4000)
        
        # Extract features for each window
        for w in windows:
            feat_vec = extract_multimodal_features(w, fs=8000)
            all_feats.append(feat_vec)
            
        all_windows.append(windows)
        all_y_det.append(labels['y_det'])
        all_y_cls.append(labels['y_cls'])
        all_y_loc.append(labels['y_loc'])
        all_y_sev.append(labels['y_sev'])
        all_top.append(labels['topology'])
        all_flow.append(labels['flow'])
        all_noise.append(labels['noise'])
        all_exp_ids.append(labels['experiment_id'])
        
    windows_arr = np.concatenate(all_windows, axis=0) # (Total_windows, 6, 8000)
    feats_arr = np.stack(all_feats, axis=0) # (Total_windows, 6*20)
    
    labels_dict = {
        'y_det': np.concatenate(all_y_det, axis=0),
        'y_cls': np.concatenate(all_y_cls, axis=0),
        'y_loc': np.concatenate(all_y_loc, axis=0),
        'y_sev': np.concatenate(all_y_sev, axis=0),
        'topology': np.concatenate(all_top, axis=0),
        'flow': np.concatenate(all_flow, axis=0),
        'noise': np.concatenate(all_noise, axis=0),
        'experiment_id': np.concatenate(all_exp_ids, axis=0)
    }
    
    return windows_arr, feats_arr, labels_dict

print("\n--- Processing Raw Signals and Windowing ---")
train_windows, train_feats, train_labels = process_split(train_df, 'Train')
val_windows, val_feats, val_labels = process_split(val_df, 'Val')
test_windows, test_feats, test_labels = process_split(test_df, 'Test')

print(f"\nWindow counts -> Train: {len(train_windows)}, Val: {len(val_windows)}, Test: {len(test_windows)}")
print(f"Window shape: {train_windows.shape}, Feature vector shape: {train_feats.shape}")

# 3. Compute Normalization Statistics (TRAINING SPLIT ONLY - Rule 4)
print("\n--- Computing Normalization Parameters (TRAINING ONLY) ---")
# Channel-wise mean and std across training windows
# train_windows: (N_train, 6, 8000)
ch_mean = np.mean(train_windows, axis=(0, 2), keepdims=True) # (1, 6, 1)
ch_std = np.std(train_windows, axis=(0, 2), keepdims=True) + 1e-8 # (1, 6, 1)

# Feature-wise mean and std for baseline feature matrix
feat_mean = np.mean(train_feats, axis=0, keepdims=True)
feat_std = np.std(train_feats, axis=0, keepdims=True) + 1e-8

norm_stats = {
    'channel_mean': ch_mean.squeeze().tolist(),
    'channel_std': ch_std.squeeze().tolist(),
    'feat_mean': feat_mean.squeeze().tolist(),
    'feat_std': feat_std.squeeze().tolist()
}

with open(proc_dir / 'norm_stats.json', 'w') as f:
    json.dump(norm_stats, f, indent=2)

print("Saved normalization parameters to processed/norm_stats.json.")

# Apply normalization
train_windows_norm = ((train_windows - ch_mean) / ch_std).astype(np.float32)
val_windows_norm = ((val_windows - ch_mean) / ch_std).astype(np.float32)
test_windows_norm = ((test_windows - ch_mean) / ch_std).astype(np.float32)

train_feats_norm = ((train_feats - feat_mean) / feat_std).astype(np.float32)
val_feats_norm = ((val_feats - feat_mean) / feat_std).astype(np.float32)
test_feats_norm = ((test_feats - feat_mean) / feat_std).astype(np.float32)

# Save compressed npz files
print("\n--- Saving Processed Dataset Arrays ---")
np.savez_compressed(
    proc_dir / 'train_data.npz',
    windows=train_windows_norm,
    raw_windows=train_windows,
    feats=train_feats_norm,
    **train_labels
)

np.savez_compressed(
    proc_dir / 'val_data.npz',
    windows=val_windows_norm,
    raw_windows=val_windows,
    feats=val_feats_norm,
    **val_labels
)

np.savez_compressed(
    proc_dir / 'test_data.npz',
    windows=test_windows_norm,
    raw_windows=test_windows,
    feats=test_feats_norm,
    **test_labels
)

print(f"Successfully processed and saved all splits to {proc_dir}!")
