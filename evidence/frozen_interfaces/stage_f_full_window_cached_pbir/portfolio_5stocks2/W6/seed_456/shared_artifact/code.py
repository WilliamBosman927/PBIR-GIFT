import numpy as np

def revise_state(s):
    x = np.asarray(s, dtype=np.float64)
    closes = x[0::6]
    opens = x[1::6]
    highs = x[2::6]
    lows = x[3::6]
    volumes = x[4::6]
    adjusted_closes = x[5::6]
    
    # f1: 3-day log return momentum (sum of last 3 daily log returns)
    log_returns = np.log(closes[1:] / (closes[:-1] + 1e-8))
    f1 = float(np.sum(log_returns[-3:]) if len(log_returns) >= 3 else 0.0)
    
    # f2: 20-day realized volatility (std of absolute log returns)
    f2 = float(np.std(np.abs(log_returns)) + 1e-8) if len(log_returns) > 1 else 1e-8
    
    # f3: normalized mean-reversion signal: (close - 20-day median) / (20-day IQR + 1e-8)
    med = np.median(closes)
    q75, q25 = np.percentile(closes, [75, 25])
    iqr = q75 - q25 + 1e-8
    f3 = float((closes[-1] - med) / iqr)
    
    # f4: volume acceleration: (current volume - 5-day avg volume) / (5-day avg volume + 1e-8)
    vol_5d_avg = np.mean(volumes[-5:]) + 1e-8
    f4 = float((volumes[-1] - vol_5d_avg) / vol_5d_avg)
    
    extra = np.array([f1, f2, f3, f4], dtype=np.float64)
    extra = np.nan_to_num(extra, nan=0.0, posinf=0.0, neginf=0.0)
    return np.concatenate((x, extra))

def intrinsic_reward(updated_s):
    f1 = float(updated_s[120])
    f2 = float(updated_s[121])
    f3 = float(updated_s[122])
    f4 = float(updated_s[123])
    
    # Phi: reward short-term momentum (f1), penalize volatility (f2), reward mean-reversion (f3), neutral on volume acceleration (f4)
    # Emphasize f1 and f3 (strongest SHAP alignment in prior iteration), de-emphasize f2/f4; bound via tanh
    phi = np.tanh(0.8 * f1 - 0.6 * f2 + 0.7 * f3 + 0.05 * np.clip(f4, -2.0, 2.0))
    
    phi = float(np.nan_to_num(phi, nan=0.0, posinf=0.0, neginf=0.0))
    return phi