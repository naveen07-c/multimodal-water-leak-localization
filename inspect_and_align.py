import os
import sys
import numpy as np
import pandas as pd
import scipy.signal as signal
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

root = Path('/home/naveen/Pictures/agy_water_leak/Dataset of Leak Simulations in Experimental Testbed Water Network')
figs_raw = Path('/home/naveen/Pictures/agy_water_leak/figures/raw_signals')
figs_spec = Path('/home/naveen/Pictures/agy_water_leak/figures/spectrograms')
reports_dir = Path('/home/naveen/Pictures/agy_water_leak/reports')

figs_raw.mkdir(parents=True, exist_ok=True)
figs_spec.mkdir(parents=True, exist_ok=True)
reports_dir.mkdir(parents=True, exist_ok=True)

# 1. Exhaustive Data Quality Audit
master_df = pd.read_csv('/home/naveen/Pictures/agy_water_leak/metadata/02_master_metadata.csv')
print(f"Loaded master metadata: {len(master_df)} records")

quality_records = []

def read_raw_hydrophone(filepath, max_samples=None):
    with open(filepath, 'rb') as f:
        data = np.fromfile(f, dtype=np.int32)
    if max_samples:
        data = data[:max_samples]
    # Normalize 32-bit int to float [-1.0, 1.0] or keep as calibrated float
    # Standard 32-bit PCM: divide by 2**31
    return data.astype(np.float32) / (2**31)

def read_csv_signal(filepath, max_samples=None):
    df = pd.read_csv(filepath)
    if max_samples:
        df = df.iloc[:max_samples]
    time = df['Sample'].values
    val = df['Value'].values.astype(np.float32)
    return time, val

print("Auditing all 282 files for anomalies, NaNs, Infs, clipping...")
for idx, row in master_df.iterrows():
    fpath = Path('/home/naveen/Pictures/agy_water_leak') / row['file_path']
    stype = row['sensor_type']
    
    try:
        if stype == 'Hydrophone':
            val = read_raw_hydrophone(fpath)
            t = np.arange(len(val)) / row['sampling_rate']
        else:
            t, val = read_csv_signal(fpath)
            
        n_nans = int(np.isnan(val).sum())
        n_infs = int(np.isinf(val).sum())
        val_min = float(np.min(val))
        val_max = float(np.max(val))
        val_mean = float(np.mean(val))
        val_std = float(np.std(val))
        val_rms = float(np.sqrt(np.mean(val**2)))
        # Check if constant
        is_constant = bool(val_std == 0.0)
        
        # Check clipping (e.g. hydrophone reaching +/- 1.0 or repetitive max values)
        if stype == 'Hydrophone':
            clip_count = int((np.abs(val) >= 0.999).sum())
        else:
            clip_count = int(((val == val_min) | (val == val_max)).sum() - 2)
            
        quality_records.append({
            'recording_id': row['recording_id'],
            'file_name': row['file_name'],
            'sensor_id': row['sensor_id'],
            'sensor_type': stype,
            'experiment_id': row['experiment_id'],
            'total_samples': len(val),
            'duration_s': float(t[-1] - t[0]) if len(t) > 1 else 0.0,
            'nans': n_nans,
            'infs': n_infs,
            'is_constant': is_constant,
            'clip_count': clip_count,
            'min': val_min,
            'max': val_max,
            'mean': val_mean,
            'std': val_std,
            'rms': val_rms
        })
    except Exception as e:
        quality_records.append({
            'recording_id': row['recording_id'],
            'file_name': row['file_name'],
            'sensor_id': row['sensor_id'],
            'sensor_type': stype,
            'experiment_id': row['experiment_id'],
            'error': str(e)
        })

df_quality = pd.DataFrame(quality_records)
df_quality.to_csv(reports_dir / 'data_quality_audit.csv', index=False)
print("Data quality audit complete. Any NaNs:", df_quality['nans'].sum(), "Any Infs:", df_quality['infs'].sum())

# 2. Generate Representative Signal Plots
print("Generating representative waveform and spectrogram plots...")

# Plot 1: Compare 5 Leak States for Accelerometer (A1), Hydrophone (H1), Dynamic Pressure (P1)
# Condition: BR, 0.18 LPS, N
leak_types = ['NL', 'OL', 'LC', 'CC', 'GL']
leak_names = ['No Leak (NL)', 'Orifice Leak (OL)', 'Longitudinal Crack (LC)', 'Circumferential Crack (CC)', 'Gasket Leak (GL)']
colors = ['#2ca02c', '#d62728', '#1f77b4', '#ff7f0e', '#9467bd']

fig, axes = plt.subplots(3, 5, figsize=(20, 10), sharex='row')
fig.suptitle('Raw Waveforms across Leak States (Branched Topology, 0.18 LPS Flow, Sensor Set S1)', fontsize=16, fontweight='bold')

for col_idx, (lt, lname) in enumerate(zip(leak_types, leak_names)):
    # A1
    f_a1 = root / f'Accelerometer/Branched/{master_df[master_df.leak_type_code==lt].leak_type.iloc[0]}/BR_{lt}_0.18 LPS_A1.csv'
    t_a, v_a = read_csv_signal(f_a1, max_samples=25600*3) # 3 seconds
    axes[0, col_idx].plot(t_a[:25600*2], v_a[:25600*2], color=colors[col_idx], lw=0.5)
    axes[0, col_idx].set_title(lname, fontsize=11, fontweight='bold')
    if col_idx == 0:
        axes[0, col_idx].set_ylabel('Accelerometer A1\n(m/s²)', fontsize=11)
    axes[0, col_idx].grid(True, alpha=0.3)
    
    # P1
    f_p1 = root / f'Dynamic Pressure Sensor/Branched/{master_df[master_df.leak_type_code==lt].leak_type.iloc[0]}/BR_{lt}_0.18 LPS_P1.csv'
    t_p, v_p = read_csv_signal(f_p1, max_samples=25600*3)
    axes[1, col_idx].plot(t_p[:25600*2], v_p[:25600*2], color=colors[col_idx], lw=0.5)
    if col_idx == 0:
        axes[1, col_idx].set_ylabel('Dynamic Pressure P1\n(Pa)', fontsize=11)
    axes[1, col_idx].grid(True, alpha=0.3)
    
    # H1
    f_h1 = root / f'Hydrophone/Branched/{master_df[master_df.leak_type_code==lt].leak_type.iloc[0]}/BR_{lt}_0.18 LPS_N_H1.raw'
    v_h = read_raw_hydrophone(f_h1, max_samples=8000*3)
    t_h = np.arange(len(v_h)) / 8000.0
    axes[2, col_idx].plot(t_h[:8000*2], v_h[:8000*2], color=colors[col_idx], lw=0.5)
    if col_idx == 0:
        axes[2, col_idx].set_ylabel('Hydrophone H1\n(Normalized V)', fontsize=11)
    axes[2, col_idx].set_xlabel('Time (s)', fontsize=11)
    axes[2, col_idx].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(figs_raw / 'leak_types_comparison.png', dpi=200)
plt.close()

# Plot 2: N vs NN for Hydrophones under No-Demand (ND) across Leak Types
fig, axes = plt.subplots(2, 5, figsize=(20, 7), sharex=True, sharey=True)
fig.suptitle('Hydrophone H1: Background Noise (N) vs Quiet Baseline (NN) under No-Demand (ND)', fontsize=15, fontweight='bold')

for col_idx, (lt, lname) in enumerate(zip(leak_types, leak_names)):
    leak_folder = master_df[master_df.leak_type_code==lt].leak_type.iloc[0]
    f_nn = root / f'Hydrophone/Branched/{leak_folder}/BR_{lt}_ND_NN_H1.raw'
    f_n = root / f'Hydrophone/Branched/{leak_folder}/BR_{lt}_ND_N_H1.raw'
    
    v_nn = read_raw_hydrophone(f_nn, max_samples=8000*2)
    v_n = read_raw_hydrophone(f_n, max_samples=8000*2)
    t_h = np.arange(len(v_nn)) / 8000.0
    
    axes[0, col_idx].plot(t_h, v_nn, color='#2ca02c', lw=0.6)
    axes[0, col_idx].set_title(f'{lname}\n(NN: Quiet)', fontsize=10, fontweight='bold')
    axes[0, col_idx].grid(True, alpha=0.3)
    if col_idx == 0:
        axes[0, col_idx].set_ylabel('Amplitude (NN)', fontsize=11)
        
    axes[1, col_idx].plot(t_h, v_n, color='#d62728', lw=0.6)
    axes[1, col_idx].set_title(f'{lname}\n(N: Traffic + Saw)', fontsize=10, fontweight='bold')
    axes[1, col_idx].grid(True, alpha=0.3)
    axes[1, col_idx].set_xlabel('Time (s)', fontsize=11)
    if col_idx == 0:
        axes[1, col_idx].set_ylabel('Amplitude (N)', fontsize=11)

plt.tight_layout()
plt.savefig(figs_raw / 'hydrophone_n_vs_nn.png', dpi=200)
plt.close()

# Plot 3: Transient Response (Valve shutoff at t ~ 20s) across all 6 sensors (A1, A2, P1, P2, H1, H2)
fig, axes = plt.subplots(6, 1, figsize=(14, 12), sharex=True)
fig.suptitle('Multimodal Transient Response During Valve Shutoff (BR_NL_Transient)', fontsize=15, fontweight='bold')

f_a1 = root / 'Accelerometer/Branched/No-leak/BR_NL_Transient_A1.csv'
f_a2 = root / 'Accelerometer/Branched/No-leak/BR_NL_Transient_A2.csv'
f_p1 = root / 'Dynamic Pressure Sensor/Branched/No-leak/BR_NL_Transient_P1.csv'
f_p2 = root / 'Dynamic Pressure Sensor/Branched/No-leak/BR_NL_Transient_P2.csv'
f_h1 = root / 'Hydrophone/Branched/No-leak/BR_NL_Transient_N_H1.raw'
f_h2 = root / 'Hydrophone/Branched/No-leak/BR_NL_Transient_N_H2.raw'

ta1, va1 = read_csv_signal(f_a1)
ta2, va2 = read_csv_signal(f_a2)
tp1, vp1 = read_csv_signal(f_p1)
tp2, vp2 = read_csv_signal(f_p2)
vh1 = read_raw_hydrophone(f_h1)
vh2 = read_raw_hydrophone(f_h2)
th1 = np.arange(len(vh1)) / 8000.0
th2 = np.arange(len(vh2)) / 8000.0

axes[0].plot(ta1, va1, color='#1f77b4', lw=0.4)
axes[0].set_ylabel('A1 (m/s²)')
axes[0].grid(True, alpha=0.3)
axes[0].axvline(x=20.0, color='r', linestyle='--', label='Valve Event ~20s')
axes[0].legend(loc='upper right')

axes[1].plot(ta2, va2, color='#1f77b4', lw=0.4)
axes[1].set_ylabel('A2 (m/s²)')
axes[1].grid(True, alpha=0.3)
axes[1].axvline(x=20.0, color='r', linestyle='--')

axes[2].plot(tp1, vp1, color='#ff7f0e', lw=0.4)
axes[2].set_ylabel('P1 (Pa)')
axes[2].grid(True, alpha=0.3)
axes[2].axvline(x=20.0, color='r', linestyle='--')

axes[3].plot(tp2, vp2, color='#ff7f0e', lw=0.4)
axes[3].set_ylabel('P2 (Pa)')
axes[3].grid(True, alpha=0.3)
axes[3].axvline(x=20.0, color='r', linestyle='--')

axes[4].plot(th1, vh1, color='#2ca02c', lw=0.4)
axes[4].set_ylabel('H1 (Norm V)')
axes[4].grid(True, alpha=0.3)
axes[4].axvline(x=20.0, color='r', linestyle='--')

axes[5].plot(th2, vh2, color='#2ca02c', lw=0.4)
axes[5].set_ylabel('H2 (Norm V)')
axes[5].set_xlabel('Time (seconds)')
axes[5].grid(True, alpha=0.3)
axes[5].axvline(x=20.0, color='r', linestyle='--')

plt.tight_layout()
plt.savefig(figs_raw / 'multimodal_transient_alignment.png', dpi=200)
plt.close()

# Plot 4: Spectrograms (STFT) for Accelerometer, Dynamic Pressure, and Hydrophone (NL vs Leak)
fig, axes = plt.subplots(3, 2, figsize=(14, 10))
fig.suptitle('STFT Spectrogram Comparison: No-Leak (NL) vs Orifice Leak (OL)', fontsize=15, fontweight='bold')

# A1 NL vs OL
ta_nl, va_nl = read_csv_signal(root / 'Accelerometer/Branched/No-leak/BR_NL_0.18 LPS_A1.csv', 25600*10)
ta_ol, va_ol = read_csv_signal(root / 'Accelerometer/Branched/Orifice Leak/BR_OL_0.18 LPS_A1.csv', 25600*10)
f_a_nl, t_a_nl, Sxx_a_nl = signal.spectrogram(va_nl, fs=25600, nperseg=1024, noverlap=512)
f_a_ol, t_a_ol, Sxx_a_ol = signal.spectrogram(va_ol, fs=25600, nperseg=1024, noverlap=512)

axes[0, 0].pcolormesh(t_a_nl, f_a_nl[:200], 10*np.log10(Sxx_a_nl[:200, :] + 1e-12), shading='gouraud', cmap='inferno')
axes[0, 0].set_title('Accelerometer A1 — No Leak (NL)', fontsize=11, fontweight='bold')
axes[0, 0].set_ylabel('Frequency (Hz)')

axes[0, 1].pcolormesh(t_a_ol, f_a_ol[:200], 10*np.log10(Sxx_a_ol[:200, :] + 1e-12), shading='gouraud', cmap='inferno')
axes[0, 1].set_title('Accelerometer A1 — Orifice Leak (OL)', fontsize=11, fontweight='bold')
axes[0, 1].set_ylabel('Frequency (Hz)')

# P1 NL vs OL
tp_nl, vp_nl = read_csv_signal(root / 'Dynamic Pressure Sensor/Branched/No-leak/BR_NL_0.18 LPS_P1.csv', 25600*10)
tp_ol, vp_ol = read_csv_signal(root / 'Dynamic Pressure Sensor/Branched/Orifice Leak/BR_OL_0.18 LPS_P1.csv', 25600*10)
f_p_nl, t_p_nl, Sxx_p_nl = signal.spectrogram(vp_nl, fs=25600, nperseg=1024, noverlap=512)
f_p_ol, t_p_ol, Sxx_p_ol = signal.spectrogram(vp_ol, fs=25600, nperseg=1024, noverlap=512)

axes[1, 0].pcolormesh(t_p_nl, f_p_nl[:200], 10*np.log10(Sxx_p_nl[:200, :] + 1e-12), shading='gouraud', cmap='inferno')
axes[1, 0].set_title('Dynamic Pressure P1 — No Leak (NL)', fontsize=11, fontweight='bold')
axes[1, 0].set_ylabel('Frequency (Hz)')

axes[1, 1].pcolormesh(t_p_ol, f_p_ol[:200], 10*np.log10(Sxx_p_ol[:200, :] + 1e-12), shading='gouraud', cmap='inferno')
axes[1, 1].set_title('Dynamic Pressure P1 — Orifice Leak (OL)', fontsize=11, fontweight='bold')
axes[1, 1].set_ylabel('Frequency (Hz)')

# H1 NL vs OL
vh_nl = read_raw_hydrophone(root / 'Hydrophone/Branched/No-leak/BR_NL_0.18 LPS_N_H1.raw', 8000*10)
vh_ol = read_raw_hydrophone(root / 'Hydrophone/Branched/Orifice Leak/BR_OL_0.18 LPS_N_H1.raw', 8000*10)
f_h_nl, t_h_nl, Sxx_h_nl = signal.spectrogram(vh_nl, fs=8000, nperseg=512, noverlap=256)
f_h_ol, t_h_ol, Sxx_h_ol = signal.spectrogram(vh_ol, fs=8000, nperseg=512, noverlap=256)

axes[2, 0].pcolormesh(t_h_nl, f_h_nl[:150], 10*np.log10(Sxx_h_nl[:150, :] + 1e-12), shading='gouraud', cmap='inferno')
axes[2, 0].set_title('Hydrophone H1 — No Leak (NL)', fontsize=11, fontweight='bold')
axes[2, 0].set_ylabel('Frequency (Hz)')
axes[2, 0].set_xlabel('Time (s)')

axes[2, 1].pcolormesh(t_h_ol, f_h_ol[:150], 10*np.log10(Sxx_h_ol[:150, :] + 1e-12), shading='gouraud', cmap='inferno')
axes[2, 1].set_title('Hydrophone H1 — Orifice Leak (OL)', fontsize=11, fontweight='bold')
axes[2, 1].set_ylabel('Frequency (Hz)')
axes[2, 1].set_xlabel('Time (s)')

plt.tight_layout()
plt.savefig(figs_spec / 'spectrogram_nl_vs_ol.png', dpi=200)
plt.close()

# Plot 5: Power Spectral Density across all 5 leak states for Hydrophone
fig, ax = plt.subplots(figsize=(10, 6))
for lt, lname, c in zip(leak_types, leak_names, colors):
    f_h = root / f'Hydrophone/Branched/{master_df[master_df.leak_type_code==lt].leak_type.iloc[0]}/BR_{lt}_0.18 LPS_N_H1.raw'
    vh = read_raw_hydrophone(f_h, 8000*30)
    freqs, psd = signal.welch(vh, fs=8000, nperseg=2048)
    ax.semilogy(freqs[:1000], psd[:1000], label=lname, color=c, lw=1.2)

ax.set_title('Power Spectral Density (PSD) of Hydrophone H1 across Leak States', fontsize=13, fontweight='bold')
ax.set_xlabel('Frequency (Hz)', fontsize=11)
ax.set_ylabel('Power Spectral Density (V²/Hz)', fontsize=11)
ax.grid(True, which='both', alpha=0.3)
ax.legend()
plt.tight_layout()
plt.savefig(figs_spec / 'psd_leak_types.png', dpi=200)
plt.close()

print("All plots generated successfully!")
