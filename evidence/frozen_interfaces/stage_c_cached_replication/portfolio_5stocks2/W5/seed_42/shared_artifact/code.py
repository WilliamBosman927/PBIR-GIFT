import numpy as np

def revise_state(s):
    x = np.asarray(s, dtype=np.float64)
    closes = x[0::6]
    opens = x[1::6]
    highs = x[2::6]
    lows = x[3::6]
    volumes = x[4::6]
    adjusted_closes = x[5::6]
    
    # 1. 5-day risk-adjusted momentum: sum of last 5 daily returns / (5-day vol + eps)
    returns = np.diff(closes) / (np.abs(closes[:-1]) + 1e-8)
    if len(returns) >= 5:
        vol_5d = np.std(returns[-5:]) + 1e-8
        f1 = float(np.sum(returns[-5:]) / vol_5d)
    else:
        f1 = 0.0
    
    # 2. Downside deviation: std of negative returns over full 20 days
    neg_returns = returns[returns < 0]
    f2 = float(np.std(neg_returns)) if len(neg_returns) > 0 else 0.0
    
    # 3. Normalized high-low range: (high - low) / close, averaged over last 5 days
    hl_ranges = (highs[-5:] - lows[-5:]) / (closes[-5:] + 1e-8)
    f3 = float(np.mean(hl_ranges)) if len(hl_ranges) > 0 else 0.0
    
    # 4. Volume-normalized price change: avg(|return| * volume) over last 5 days
    abs_returns = np.abs(returns[-5:]) if len(returns) >= 5 else np.abs(returns)
    vol_weights = volumes[-len(abs_returns):] if len(volumes) >= len(abs_returns) else volumes
    f4 = float(np.mean(abs_returns * (vol_weights / (np.mean(vol_weights) + 1e-8)))) if len(abs_returns) > 0 else 0.0
    
    extra = np.array([f1, f2, f3, f4], dtype=np.float64)
    extra = np.nan_to_num(extra, nan=0.0, posinf=0.0, neginf=0.0)
    return np.concatenate((x, extra))

def intrinsic_reward(updated_s):
    f1 = float(updated_s[120])
    f2 = float(updated_s[121])
    f3 = float(updated_s[122])
    f4 = float(updated_s[123])
    
    # Potential: reward strong short-term risk-adjusted momentum (f1),
    # penalize downside risk (f2), penalize wide ranges (f3 → volatility proxy),
    # and reward volume-backed moves (f4 → signal confidence)
    phi = 0.7 * f1 - 0.5 * f2 - 0.3 * f3 + 0.4 * f4
    phi = np.tanh(phi)
    phi = float(np.nan_to_num(phi, nan=0.0, posinf=0.0, neginf=0.0))
    return phi