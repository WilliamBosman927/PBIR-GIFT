import numpy as np

def revise_state(s):
    x = np.asarray(s, dtype=np.float64)
    closes = x[0::6]
    opens = x[1::6]
    highs = x[2::6]
    lows = x[3::6]
    volumes = x[4::6]
    adjusted_closes = x[5::6]
    
    # Log returns for stability
    returns = np.log(closes[1:] / closes[:-1] + 1e-8)
    
    # f1: 5-day momentum (sum of last 5 log returns)
    f1 = float(np.sum(returns[-5:]) if len(returns) >= 5 else 0.0)
    
    # f2: realized volatility (std of last 10 log returns)
    vol_window = returns[-10:] if len(returns) >= 10 else returns
    f2 = float(np.std(vol_window)) + 1e-8
    
    # f3: high-low range normalized by recent close (volatility proxy)
    hl_range = highs - lows
    f3 = float(np.mean(hl_range[-5:] / (closes[-5:] + 1e-8))) if len(closes) >= 5 else 0.0
    
    # f4: volume z-score relative to 20-day mean and std
    vol_mean = float(np.mean(volumes))
    vol_std = float(np.std(volumes)) + 1e-8
    f4 = float((volumes[-1] - vol_mean) / vol_std) if vol_std > 1e-8 else 0.0
    
    extra = np.array([f1, f2, f3, f4], dtype=np.float64)
    extra = np.nan_to_num(extra, nan=0.0, posinf=0.0, neginf=0.0)
    return np.concatenate((x, extra))

def intrinsic_reward(updated_s):
    f1 = float(updated_s[120])
    f2 = float(updated_s[121])
    f3 = float(updated_s[122])
    f4 = float(updated_s[123])
    
    # Phi: reward trend (f1), penalize volatility (f2), reward price dispersion (f3),
    # and neutral on volume signal (f4) — bounded via tanh
    phi = np.tanh(0.9 * f1 - 0.8 * f2 + 0.4 * f3)
    
    phi = float(np.nan_to_num(phi, nan=0.0, posinf=0.0, neginf=0.0))
    return phi