import numpy as np

def revise_state(s):
    x = np.asarray(s, dtype=np.float64)
    closes = x[0::6]
    opens = x[1::6]
    highs = x[2::6]
    lows = x[3::6]
    volumes = x[4::6]
    adjusted_closes = x[5::6]
    
    # f1: 5-day log return momentum (more stable than 3-day, less noisy)
    log_returns = np.log(closes[1:] / (closes[:-1] + 1e-8))
    f1 = float(np.sum(log_returns[-5:]) if len(log_returns) >= 5 else 0.0)
    
    # f2: 20-day downside deviation (robust to zero/negative returns)
    neg_returns = log_returns[log_returns < 0]
    f2 = float(np.std(neg_returns)) if len(neg_returns) > 1 else 0.0
    
    # f3: current close vs 10-day SMA z-score (faster trend alignment)
    sma10 = float(np.mean(closes[-10:])) + 1e-8
    std10 = float(np.std(closes[-10:])) + 1e-8
    f3 = (closes[-1] - sma10) / std10
    
    # f4: high-low range ratio (normalized volatility proxy, regime-robust)
    hl_ranges = (highs - lows) / (closes + 1e-8)
    f4 = float(np.mean(hl_ranges)) if len(hl_ranges) > 0 else 0.0
    
    extra = np.array([f1, f2, f3, f4], dtype=np.float64)
    extra = np.nan_to_num(extra, nan=0.0, posinf=0.0, neginf=0.0)
    return np.concatenate((x, extra))

def intrinsic_reward(updated_s):
    f1 = float(updated_s[120])
    f2 = float(updated_s[121])
    f3 = float(updated_s[122])
    f4 = float(updated_s[123])
    # Phi: reward momentum (f1) and trend alignment (f3), penalize downside risk (f2) and price dispersion (f4)
    phi = 0.7 * np.tanh(f1) - 0.5 * np.tanh(f2) + 0.4 * np.tanh(f3) - 0.3 * np.tanh(f4)
    phi = float(np.nan_to_num(phi, nan=0.0, posinf=0.0, neginf=0.0))
    return np.tanh(phi)