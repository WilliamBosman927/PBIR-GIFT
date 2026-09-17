import numpy as np

def revise_state(s):
    x = np.asarray(s, dtype=np.float64)
    closes = x[0::6]
    opens = x[1::6]
    highs = x[2::6]
    lows = x[3::6]
    volumes = x[4::6]
    adjusted_closes = x[5::6]
    
    returns = np.diff(closes) / (np.abs(closes[:-1]) + 1e-8)
    
    # f1: 5-day price momentum, normalized by 20-day price std (robust to scale)
    f1 = (closes[-1] - closes[-6]) / (np.std(closes) + 1e-8) if len(closes) >= 6 else 0.0
    
    # f2: downside deviation — std of negative returns only; zero if none exist
    neg_returns = returns[returns < 0]
    f2 = float(np.std(neg_returns)) if len(neg_returns) > 0 else 0.0
    
    # f3: 5-day average normalized high-low range: (high - low) / close
    hl_ranges = (highs[-5:] - lows[-5:]) / (np.abs(closes[-5:]) + 1e-8)
    f3 = float(np.mean(hl_ranges)) if len(hl_ranges) > 0 else 0.0
    
    # f4: volume acceleration — z-score of current volume vs 20-day mean/std
    vol_mean = float(np.mean(volumes))
    vol_std = float(np.std(volumes)) + 1e-8
    f4 = (volumes[-1] - vol_mean) / vol_std if len(volumes) > 0 else 0.0
    
    extra = np.array([f1, f2, f3, f4], dtype=np.float64)
    extra = np.nan_to_num(extra, nan=0.0, posinf=0.0, neginf=0.0)
    return np.concatenate((x, extra))

def intrinsic_reward(updated_s):
    f1 = float(updated_s[120])
    f2 = float(updated_s[121])
    f3 = float(updated_s[122])
    f4 = float(updated_s[123])
    
    # Phi: reward trend (f1), penalize downside risk (f2), penalize wide ranges (f3), neutral on volume (f4)
    # Scale f2 and f3 using regime-informed typical values (0.02 for downside vol, 0.03 for range)
    phi = np.tanh(0.8 * f1 - 0.5 * np.tanh(f2 / 0.02) - 0.4 * np.tanh(f3 / 0.03))
    
    phi = float(np.nan_to_num(phi, nan=0.0, posinf=0.0, neginf=0.0))
    return phi