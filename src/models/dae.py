import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvDAE1D(nn.Module):
    """
    1D Convolutional Denoising Autoencoder.
    Input: (B, C, L) where C is number of channels (e.g. 1 or 6), L is window length (8000).
    Output: Reconstructed clean signal (B, C, L).
    """
    def __init__(self, in_channels=1, hidden_channels=[16, 32, 64], kernel_size=15):
        super(ConvDAE1D, self).__init__()
        
        padding = kernel_size // 2
        # Encoder
        self.enc1 = nn.Conv1d(in_channels, hidden_channels[0], kernel_size, stride=2, padding=padding)
        self.bn1 = nn.BatchNorm1d(hidden_channels[0])
        self.enc2 = nn.Conv1d(hidden_channels[0], hidden_channels[1], kernel_size, stride=2, padding=padding)
        self.bn2 = nn.BatchNorm1d(hidden_channels[1])
        self.enc3 = nn.Conv1d(hidden_channels[1], hidden_channels[2], kernel_size, stride=2, padding=padding)
        self.bn3 = nn.BatchNorm1d(hidden_channels[2])
        
        # Bottleneck
        self.bottleneck = nn.Conv1d(hidden_channels[2], hidden_channels[2], kernel_size, stride=1, padding=padding)
        self.bn_b = nn.BatchNorm1d(hidden_channels[2])
        
        # Decoder
        self.dec3 = nn.ConvTranspose1d(hidden_channels[2], hidden_channels[1], kernel_size, stride=2, padding=padding, output_padding=1)
        self.bn_d3 = nn.BatchNorm1d(hidden_channels[1])
        self.dec2 = nn.ConvTranspose1d(hidden_channels[1], hidden_channels[0], kernel_size, stride=2, padding=padding, output_padding=1)
        self.bn_d2 = nn.BatchNorm1d(hidden_channels[0])
        self.dec1 = nn.ConvTranspose1d(hidden_channels[0], in_channels, kernel_size, stride=2, padding=padding, output_padding=1)
        
    def forward(self, x):
        # x: (B, C, L)
        orig_len = x.shape[-1]
        
        # Encode
        e1 = F.leaky_relu(self.bn1(self.enc1(x)), 0.1)
        e2 = F.leaky_relu(self.bn2(self.enc2(e1)), 0.1)
        e3 = F.leaky_relu(self.bn3(self.enc3(e2)), 0.1)
        
        # Bottleneck
        b = F.leaky_relu(self.bn_b(self.bottleneck(e3)), 0.1)
        
        # Decode
        d3 = F.leaky_relu(self.bn_d3(self.dec3(b)), 0.1)
        d2 = F.leaky_relu(self.bn_d2(self.dec2(d3)), 0.1)
        out = self.dec1(d2)
        
        # Ensure exact length matching
        if out.shape[-1] != orig_len:
            out = F.interpolate(out, size=orig_len, mode='linear', align_corners=False)
            
        return out

def compute_snr(clean, noisy):
    """
    Computes Signal-to-Noise Ratio (SNR) in dB.
    """
    clean_pwr = np.mean(clean**2) + 1e-12
    noise_pwr = np.mean((clean - noisy)**2) + 1e-12
    return 10.0 * np.log10(clean_pwr / noise_pwr)

def evaluate_denoising(clean, noisy, reconstructed):
    """
    Computes denoising metrics: MSE, MAE, SNR in, SNR out, SNR improvement, Correlation.
    """
    mse_noisy = np.mean((clean - noisy)**2)
    mse_recon = np.mean((clean - reconstructed)**2)
    mae_recon = np.mean(np.abs(clean - reconstructed))
    
    snr_in = compute_snr(clean, noisy)
    snr_out = compute_snr(clean, reconstructed)
    snr_gain = snr_out - snr_in
    
    c_flat = clean.flatten()
    r_flat = reconstructed.flatten()
    corr = np.corrcoef(c_flat, r_flat)[0, 1] if np.std(c_flat) > 0 and np.std(r_flat) > 0 else 0.0
    
    return {
        'mse_noisy': float(mse_noisy),
        'mse_recon': float(mse_recon),
        'mae_recon': float(mae_recon),
        'snr_in_db': float(snr_in),
        'snr_out_db': float(snr_out),
        'snr_gain_db': float(snr_gain),
        'correlation': float(corr)
    }
