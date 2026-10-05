import numpy as np
import scipy.stats as stats
import scipy.signal as signal

def extract_signal_features(sig, fs=8000):
    """
    Extracts comprehensive time-domain and frequency-domain statistical features
    from a 1D windowed signal.
    """
    # Time domain
    val = sig.astype(np.float64)
    mean_val = np.mean(val)
    std_val = np.std(val) + 1e-12
    var_val = np.var(val)
    rms_val = np.sqrt(np.mean(val**2)) + 1e-12
    peak_val = np.max(np.abs(val))
    p2p_val = np.ptp(val)
    
    crest_factor = peak_val / rms_val
    kurt = stats.kurtosis(val)
    skew = stats.skew(val)
    abs_mean = np.mean(np.abs(val)) + 1e-12
    shape_factor = rms_val / abs_mean
    impulse_factor = peak_val / abs_mean
    
    # Zero crossing rate
    zero_crossings = np.sum(np.diff(np.signbit(val)) != 0) / len(val)
    
    # Energy
    energy = np.sum(val**2)
    
    # Frequency domain via FFT
    n = len(val)
    fft_vals = np.abs(np.fft.rfft(val))
    freqs = np.fft.rfftfreq(n, d=1.0/fs)
    
    fft_sum = np.sum(fft_vals) + 1e-12
    norm_fft = fft_vals / fft_sum
    
    # Spectral moments
    spec_centroid = np.sum(freqs * norm_fft)
    spec_spread = np.sqrt(np.sum(((freqs - spec_centroid)**2) * norm_fft))
    
    # Band energy ratios
    # Low band (0 - 500 Hz), Mid band (500 - 1500 Hz), High band (1500 - 4000 Hz)
    mask_low = freqs <= 500
    mask_mid = (freqs > 500) & (freqs <= 1500)
    mask_high = freqs > 1500
    
    energy_low = np.sum(fft_vals[mask_low]**2) / (np.sum(fft_vals**2) + 1e-12)
    energy_mid = np.sum(fft_vals[mask_mid]**2) / (np.sum(fft_vals**2) + 1e-12)
    energy_high = np.sum(fft_vals[mask_high]**2) / (np.sum(fft_vals**2) + 1e-12)
    
    # Spectral entropy
    spec_entropy = -np.sum(norm_fft * np.log(norm_fft + 1e-12))
    
    # Peak frequency
    peak_freq = freqs[np.argmax(fft_vals)]
    
    return [
        mean_val, std_val, var_val, rms_val, peak_val, p2p_val,
        crest_factor, kurt, skew, shape_factor, impulse_factor,
        zero_crossings, energy, spec_centroid, spec_spread,
        energy_low, energy_mid, energy_high, spec_entropy, peak_freq
    ]

def extract_multimodal_features(window_tensor, fs=8000):
    """
    window_tensor: shape (num_sensors, window_len), e.g. (6, 8000)
    Returns concatenated feature vector of shape (num_sensors * 20,)
    """
    all_feats = []
    for s_idx in range(window_tensor.shape[0]):
        feats = extract_signal_features(window_tensor[s_idx], fs=fs)
        all_feats.extend(feats)
    return np.array(all_feats, dtype=np.float32)
