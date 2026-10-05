import os
import sys
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy.signal as signal

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from src.utils.seed import set_seed
from src.models.dae import ConvDAE1D, evaluate_denoising, compute_snr

set_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[DAE Trainer] Using compute device: {device}")

root_dir = Path('d:\deep learning project\multimodal-water-leak-localization')
proc_dir = root_dir / 'data' / 'processed'
models_dir = root_dir / 'models/baseline' / 'models'
results_dir = root_dir / 'models/baseline' / 'results'
figs_dae = root_dir / 'figures/dae'

models_dir.mkdir(parents=True, exist_ok=True)
results_dir.mkdir(parents=True, exist_ok=True)
figs_dae.mkdir(parents=True, exist_ok=True)

# 1. Load Data
train_data = np.load(proc_dir / 'train_data.npz')
val_data = np.load(proc_dir / 'val_data.npz')
test_data = np.load(proc_dir / 'test_data.npz')

# Use normalized raw windows: (N, 6, 8000)
X_train = train_data['windows']
X_val = val_data['windows']
X_test = test_data['windows']

# Load reference background noise from Hydrophone/Background Noise/
noise_f1 = root_dir / 'data' / 'Dataset of Leak Simulations in Experimental Testbed Water Network/Hydrophone/Background Noise/Background Noise_H1.raw'
with open(noise_f1, 'rb') as f:
    bg_noise_raw = np.fromfile(f, dtype=np.int32).astype(np.float32) / (2**31)
# Normalize background noise to unit variance
bg_noise = (bg_noise_raw - np.mean(bg_noise_raw)) / (np.std(bg_noise_raw) + 1e-8)

def corrupt_signal(clean_arr, noise_level=0.3, thermal_std=0.15):
    """
    Simulates physically realistic corruption:
    1. Real acoustic saw/traffic noise from testbed recording
    2. Gaussian thermal sensor noise
    3. Low-frequency sensor baseline wander
    """
    N, C, L = clean_arr.shape
    corrupted = np.copy(clean_arr)
    
    # 1. Sample random slices of real acoustic background noise
    bg_len = len(bg_noise)
    for i in range(N):
        idx = np.random.randint(0, bg_len - L - 1)
        noise_slice = bg_noise[idx:idx+L]
        for c in range(C):
            # Scale acoustic noise
            corrupted[i, c] += noise_level * np.random.uniform(0.5, 1.5) * noise_slice
            
    # 2. Additive Gaussian thermal noise
    gaussian_noise = np.random.normal(0, thermal_std, size=clean_arr.shape).astype(np.float32)
    corrupted += gaussian_noise
    
    # 3. Low-frequency baseline drift (sine drift)
    t = np.linspace(0, 1.0, L)
    drift = 0.2 * np.sin(2 * np.pi * np.random.uniform(0.5, 3.0, (N, C, 1)) * t)
    corrupted += drift.astype(np.float32)
    
    return corrupted.astype(np.float32)

print("\n--- Generating Physically Realistic Corrupted Noise Datasets ---")
X_train_noisy = corrupt_signal(X_train, noise_level=0.3, thermal_std=0.15)
X_val_noisy = corrupt_signal(X_val, noise_level=0.3, thermal_std=0.15)
X_test_noisy = corrupt_signal(X_test, noise_level=0.3, thermal_std=0.15)

# Flatten channels for 1D single-channel DAE training (N*6, 1, 8000)
def to_channel_tensor(arr):
    N, C, L = arr.shape
    return arr.reshape(N * C, 1, L)

train_clean_t = torch.tensor(to_channel_tensor(X_train), dtype=torch.float32)
train_noisy_t = torch.tensor(to_channel_tensor(X_train_noisy), dtype=torch.float32)

val_clean_t = torch.tensor(to_channel_tensor(X_val), dtype=torch.float32)
val_noisy_t = torch.tensor(to_channel_tensor(X_val_noisy), dtype=torch.float32)

test_clean_t = torch.tensor(to_channel_tensor(X_test), dtype=torch.float32)
test_noisy_t = torch.tensor(to_channel_tensor(X_test_noisy), dtype=torch.float32)

train_loader = DataLoader(TensorDataset(train_noisy_t, train_clean_t), batch_size=64, shuffle=True)
val_loader = DataLoader(TensorDataset(val_noisy_t, val_clean_t), batch_size=64, shuffle=False)
test_loader = DataLoader(TensorDataset(test_noisy_t, test_clean_t), batch_size=64, shuffle=False)

# 2. Instantiate and Train DAE
dae = ConvDAE1D(in_channels=1, hidden_channels=[16, 32, 64], kernel_size=15).to(device)
criterion_mse = nn.MSELoss()
criterion_l1 = nn.L1Loss()
optimizer = optim.AdamW(dae.parameters(), lr=1e-3, weight_decay=1e-5)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30, eta_min=1e-5)

print("\n--- Training 1D Convolutional DAE ---")
best_val_loss = float('inf')
best_dae_state = None

for epoch in range(1, 31):
    dae.train()
    train_loss = 0.0
    for bx_noisy, bx_clean in train_loader:
        bx_noisy, bx_clean = bx_noisy.to(device), bx_clean.to(device)
        optimizer.zero_grad()
        recon = dae(bx_noisy)
        loss = criterion_mse(recon, bx_clean) + 0.1 * criterion_l1(recon, bx_clean)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * len(bx_noisy)
        
    train_loss /= len(train_clean_t)
    
    dae.eval()
    val_loss = 0.0
    with torch.no_grad():
        for bx_noisy, bx_clean in val_loader:
            bx_noisy, bx_clean = bx_noisy.to(device), bx_clean.to(device)
            recon = dae(bx_noisy)
            loss = criterion_mse(recon, bx_clean)
            val_loss += loss.item() * len(bx_noisy)
            
    val_loss /= len(val_clean_t)
    scheduler.step()
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_dae_state = dae.state_dict().copy()
        
    if epoch % 5 == 0 or epoch == 30:
        print(f"Epoch {epoch:02d}/30 | Train Loss: {train_loss:.5f} | Val MSE: {val_loss:.5f} (Best: {best_val_loss:.5f})")

dae.load_state_dict(best_dae_state)
torch.save(best_dae_state, models_dir / 'dae_model.pt')
print("Saved best DAE checkpoint to models/dae_model.pt.")

# 3. Quantitative Evaluation on Test Set
dae.eval()
test_recons = []
with torch.no_grad():
    for bx_noisy, _ in test_loader:
        bx_noisy = bx_noisy.to(device)
        recon = dae(bx_noisy)
        test_recons.append(recon.cpu().numpy())
        
test_recon_arr = np.concatenate(test_recons, axis=0) # (N_test*6, 1, 8000)
test_clean_arr = test_clean_t.numpy()
test_noisy_arr = test_noisy_t.numpy()

metrics = evaluate_denoising(test_clean_arr, test_noisy_arr, test_recon_arr)
print("\n" + "="*50)
print("DAE TEST EVALUATION METRICS")
print("="*50)
print(f"Input Noise MSE       : {metrics['mse_noisy']:.6f}")
print(f"Reconstruction MSE    : {metrics['mse_recon']:.6f}")
print(f"Reconstruction MAE    : {metrics['mae_recon']:.6f}")
print(f"Input SNR (dB)        : {metrics['snr_in_db']:.2f} dB")
print(f"Output SNR (dB)       : {metrics['snr_out_db']:.2f} dB")
print(f"SNR Improvement (Gain): +{metrics['snr_gain_db']:.2f} dB")
print(f"Pearson Correlation   : {metrics['correlation']:.4f}")

# Save metrics
df_dae_metrics = pd.DataFrame([metrics])
df_dae_metrics.to_csv(results_dir / 'dae_results.csv', index=False)
df_dae_metrics.to_csv(root_dir / '07_dae_results.csv', index=False)

# 4. Generate Visual Verification Plots (Figures)
print("\n--- Generating DAE Denoising Figures ---")
sample_indices = [0, 50, 100, 200]
fig, axes = plt.subplots(len(sample_indices), 3, figsize=(18, 10), sharex=True)
fig.suptitle('DAE Denoising Performance: Clean vs Noisy vs Reconstructed Signals', fontsize=15, fontweight='bold')

for row_i, s_idx in enumerate(sample_indices):
    c_sig = test_clean_arr[s_idx, 0]
    n_sig = test_noisy_arr[s_idx, 0]
    r_sig = test_recon_arr[s_idx, 0]
    t = np.linspace(0, 1.0, len(c_sig))
    
    snr_i = compute_snr(c_sig, n_sig)
    snr_o = compute_snr(c_sig, r_sig)
    
    axes[row_i, 0].plot(t[:1000], c_sig[:1000], color='#2ca02c', lw=0.8)
    axes[row_i, 0].set_title(f'Sample {s_idx}: Ground Truth Clean', fontsize=10)
    axes[row_i, 0].set_ylabel('Amplitude')
    axes[row_i, 0].grid(True, alpha=0.3)
    
    axes[row_i, 1].plot(t[:1000], n_sig[:1000], color='#d62728', lw=0.8)
    axes[row_i, 1].set_title(f'Corrupted Noisy (SNR: {snr_i:.1f} dB)', fontsize=10)
    axes[row_i, 1].grid(True, alpha=0.3)
    
    axes[row_i, 2].plot(t[:1000], r_sig[:1000], color='#1f77b4', lw=0.8)
    axes[row_i, 2].set_title(f'DAE Denoised (SNR: {snr_o:.1f} dB | Gain: +{snr_o-snr_i:.1f} dB)', fontsize=10)
    axes[row_i, 2].grid(True, alpha=0.3)

axes[-1, 0].set_xlabel('Time (s)')
axes[-1, 1].set_xlabel('Time (s)')
axes[-1, 2].set_xlabel('Time (s)')

plt.tight_layout()
plt.savefig(figs_dae / 'dae_reconstruction_examples.png', dpi=200)
plt.close()

# PSD Preservation Comparison
f_c, psd_c = signal.welch(test_clean_arr[:20, 0].flatten(), fs=8000, nperseg=1024)
f_n, psd_n = signal.welch(test_noisy_arr[:20, 0].flatten(), fs=8000, nperseg=1024)
f_r, psd_r = signal.welch(test_recon_arr[:20, 0].flatten(), fs=8000, nperseg=1024)

fig, ax = plt.subplots(figsize=(10, 5))
ax.semilogy(f_c[:300], psd_c[:300], label='Original Clean Reference', color='#2ca02c', lw=1.5)
ax.semilogy(f_n[:300], psd_n[:300], label='Corrupted Noisy Input', color='#d62728', lw=1.0, alpha=0.7)
ax.semilogy(f_r[:300], psd_r[:300], label='DAE Reconstructed Output', color='#1f77b4', lw=1.5, linestyle='--')
ax.set_title('Power Spectral Density (PSD) Preservation under DAE Denoising', fontsize=12, fontweight='bold')
ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel('Power Spectral Density')
ax.grid(True, which='both', alpha=0.3)
ax.legend()
plt.tight_layout()
plt.savefig(figs_dae / 'dae_psd_preservation.png', dpi=200)
plt.close()

print("DAE plots saved to figures/dae/!")
