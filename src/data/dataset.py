import os
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.signal as signal

NODE_NAMES = ['P1', 'A1', 'H1', 'H2', 'A2', 'P2']

LEAK_CODE_MAP = {
    'NL': 0,
    'CC': 1,
    'GL': 2,
    'LC': 3,
    'OL': 4
}

SEVERITY_MAP = {
    'NL': 0.0,
    'GL': 0.07,
    'CC': 0.12,
    'LC': 0.14,
    'OL': 0.20
}

def load_recording_signal(fpath, sensor_type, max_duration=30.0, target_fs=8000):
    """
    Loads raw recording, ensures 30.0s duration, and resamples to target_fs (8000 Hz).
    Returns 1D float32 numpy array of length target_fs * max_duration = 240,000 samples.
    """
    target_samples = int(target_fs * max_duration)
    
    if sensor_type == 'Hydrophone':
        with open(fpath, 'rb') as f:
            data = np.fromfile(f, dtype=np.int32)
        # PCM 32-bit normalization
        sig = data[:target_samples].astype(np.float32) / (2**31)
        if len(sig) < target_samples:
            sig = np.pad(sig, (0, target_samples - len(sig)), mode='edge')
        return sig
    else: # Accelerometer or Dynamic Pressure Sensor (fs = 25600 Hz)
        # 30.0s @ 25600 Hz = 768,000 samples
        raw_samples = int(25600 * max_duration)
        df = pd.read_csv(fpath, nrows=raw_samples)
        val = df['Value'].values.astype(np.float32)
        if len(val) < raw_samples:
            val = np.pad(val, (0, raw_samples - len(val)), mode='edge')
        # Resample from 25600 Hz to 8000 Hz (ratio: 8000 / 25600 = 5 / 16)
        resampled = signal.resample_poly(val, 5, 16).astype(np.float32)
        if len(resampled) > target_samples:
            resampled = resampled[:target_samples]
        elif len(resampled) < target_samples:
            resampled = np.pad(resampled, (0, target_samples - len(resampled)), mode='edge')
        return resampled

def build_experiment_tensor(root_dir, exp_df, target_fs=8000, max_duration=30.0):
    """
    Builds a multimodal synchronized tensor for one experiment scenario.
    Returns:
      tensor: shape (6, 240000) for [P1, A1, H1, H2, A2, P2]
      meta: dict with labels and condition codes
    """
    sensor_signals = {}
    
    for _, row in exp_df.iterrows():
        sid = row['sensor_id']
        stype = row['sensor_type']
        fpath = Path('/home/naveen/Pictures/agy_water_leak') / row['file_path']
        sig = load_recording_signal(fpath, stype, max_duration=max_duration, target_fs=target_fs)
        sensor_signals[sid] = sig
        
    # Stack in fixed canonical order: ['P1', 'A1', 'H1', 'H2', 'A2', 'P2']
    tensor = np.stack([sensor_signals[name] for name in NODE_NAMES], axis=0) # Shape (6, 240000)
    
    first_row = exp_df.iloc[0]
    leak_code = first_row['leak_type_code']
    top_code = first_row['topology_code']
    flow_code = first_row['flow_condition_code']
    noise_code = first_row['noise_condition']
    
    meta = {
        'experiment_id': first_row['experiment_id'],
        'topology_code': top_code,
        'leak_type_code': leak_code,
        'y_det': 0 if leak_code == 'NL' else 1,
        'y_cls': LEAK_CODE_MAP[leak_code],
        'y_loc': 0 if leak_code == 'NL' else 1, # Middle pipe index / leak presence
        'y_sev': SEVERITY_MAP[leak_code],
        'flow_code': flow_code,
        'noise_code': noise_code
    }
    return tensor, meta

def slice_windows(tensor, meta, window_len=8000, stride=4000):
    """
    Slices (6, 240000) experiment tensor into overlapping windows.
    Returns:
      windows: shape (N_windows, 6, window_len)
      labels: dict of arrays (y_det, y_cls, y_loc, y_sev, exp_idx, window_idx)
    """
    total_len = tensor.shape[1]
    num_windows = (total_len - window_len) // stride + 1
    
    window_list = []
    for w_i in range(num_windows):
        start = w_i * stride
        end = start + window_len
        w = tensor[:, start:end] # (6, window_len)
        window_list.append(w)
        
    windows = np.stack(window_list, axis=0) # (num_windows, 6, window_len)
    
    labels = {
        'y_det': np.full(num_windows, meta['y_det'], dtype=np.int64),
        'y_cls': np.full(num_windows, meta['y_cls'], dtype=np.int64),
        'y_loc': np.full(num_windows, meta['y_loc'], dtype=np.int64),
        'y_sev': np.full(num_windows, meta['y_sev'], dtype=np.float32),
        'topology': np.full(num_windows, meta['topology_code']),
        'flow': np.full(num_windows, meta['flow_code']),
        'noise': np.full(num_windows, meta['noise_code']),
        'experiment_id': np.full(num_windows, meta['experiment_id'])
    }
    return windows, labels
