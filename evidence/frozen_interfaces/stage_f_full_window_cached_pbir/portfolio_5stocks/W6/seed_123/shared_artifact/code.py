import numpy as np

def revise_state(s):
    x = np.asarray(s, dtype=np.float64)
    closes = x[0::6]
    opens = x[1::6]
    highs = x[2::6]
    lows = x[3::6]
    volumes = x[4::6]
    adjusted_closes = x[5::6]
    
    # 20-day simple returns
    returns = np.diff(closes) / (np.abs(closes[:-1]) + 1e-8)
    
    # f1: 10-day momentum (cumulative return over last 10 days)
    f1 = float(np.sum(returns[-10:]) if len(returns) >= 10 else 0.0)
    
    # f2: downside deviation (std of negative returns only; 0 if no negatives)
    neg_returns = returns[returns < 0]
    f2 = float(np.std(neg_returns)) + 1e-8 if len(neg_returns) > 0 else 0.0
    
    # f3: normalized high-low range (current day's range relative to 20-day avg range)
    ranges = highs - lows
    avg_range = float(np.mean(ranges)) + 1e-8
    f3 = float((ranges[-1] - avg_range) / avg_range)
    
    # f4: volume z-score (current volume vs 20-day mean/std)
    vol_mean = float(np.mean(volumes)) + 1e-8
    vol_std = float(np.std(volumes)) + 1e-8
    f4 = float((volumes[-1] - vol_mean) / vol_std)
    
    extra = np.array([f1, f2, f3, f4], dtype=np.float64)
    extra = np.nan_to_num(extra, nan=0.0, posinf=0.0, neginf=0.0)
    return np.concatenate((x, extra))

def intrinsic_reward(updated_s):
    f1 = float(updated_s[120])
    f2 = float(updated_s[121])
    f3 = float(updated_s[122])
    f4 = float(updated_s[123])
    
    # Phi(s): reward trend (f1), penalize downside risk (f2), reward expansion of range (f3), neutral on volume (f4)
    # Emphasize trend and range expansion, suppress downside risk; tanh bounds output to [-1,1]
    phi = np.tanh(1.2 * f1 - 1.0 * f2 + 0.7 * f3)
    phi = float(np.nan_to_num(phi, nan=0.0, posinf=0.0, neginf=0.0))
    return phi