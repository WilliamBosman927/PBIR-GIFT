import numpy as np

def revise_state(s):
    x = np.asarray(s, dtype=np.float64)
    closes = x[0::6]
    opens = x[1::6]
    highs = x[2::6]
    lows = x[3::6]
    volumes = x[4::6]
    adjusted_closes = x[5::6]
    
    # 1. Short-term momentum: sum of last 5 daily returns, normalized by 20-day return volatility
    returns = np.diff(closes) / (np.abs(closes[:-1]) + 1e-8)
    f1 = float(np.sum(returns[-5:]) if len(returns) >= 5 else 0.0) / (np.std(returns) + 1e-8)
    
    # 2. Realized volatility: std of absolute daily returns over full 20-day window
    f2 = float(np.std(np.abs(returns)) if len(returns) > 1 else 0.0)
    
    # 3. Mean-reversion signal: z-score of current close vs 20-day SMA, using ATR(14) for scaling
    sma20 = np.mean(closes)
    tr = np.maximum(highs[1:] - lows[1:], np.maximum(
        np.abs(highs[1:] - closes[:-1]), 
        np.abs(lows[1:] - closes[:-1])
    ))
    atr14 = np.mean(tr[-14:]) if len(tr) >= 14 else np.std(closes) + 1e-8
    f3 = float((closes[-1] - sma20) / (np.abs(atr14) + 1e-8))
    
    # 4. Volume stability: log ratio of current volume to 10-day median volume (robust to outliers)
    vol_window = volumes[-10:] if len(volumes) >= 10 else volumes
    vol_median = np.median(vol_window) + 1e-8 if len(vol_window) > 0 else 1e-8
    f4 = float(np.log(volumes[-1] / vol_median + 1e-8)) if len(vol_window) > 0 else 0.0
    
    extra = np.array([f1, f2, f3, f4], dtype=np.float64)
    extra = np.nan_to_num(extra, nan=0.0, posinf=0.0, neginf=0.0)
    return np.concatenate((x, extra))

def intrinsic_reward(updated_s):
    f1 = float(updated_s[120])
    f2 = float(updated_s[121])
    f3 = float(updated_s[122])
    f4 = float(updated_s[123])
    phi = np.tanh(0.7 * f1 - 0.8 * f2 - 0.3 * f3 + 0.1 * np.clip(f4, -2.0, 2.0))
    phi = float(np.nan_to_num(phi, nan=0.0, posinf=0.0, neginf=0.0))
    return phi