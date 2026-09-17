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
    
    # f1: 5-day momentum (cumulative return over last 5 days) — more responsive to recent trend
    f1 = float(np.sum(returns[-5:]) if len(returns) >= 5 else 0.0)
    
    # f2: downside deviation (std of negative returns only; captures tail risk)
    negative_returns = returns[returns < 0]
    f2 = float(np.std(negative_returns)) if len(negative_returns) > 1 else 0.0
    
    # f3: current close relative to 20-day moving average (normalized by std)
    ma20 = float(np.mean(closes))
    std20 = float(np.std(closes)) + 1e-8
    f3 = float((closes[-1] - ma20) / std20)
    
    # f4: volume acceleration (current volume vs 10-day avg volume, normalized by 10-day vol std)
    vol_window = volumes[-10:] if len(volumes) >= 10 else volumes
    vol_mean = float(np.mean(vol_window))
    vol_std = float(np.std(vol_window)) + 1e-8
    f4 = float((volumes[-1] - vol_mean) / vol_std)
    
    extra = np.array([f1, f2, f3, f4], dtype=np.float64)
    extra = np.nan_to_num(extra, nan=0.0, posinf=0.0, neginf=0.0)
    return np.concatenate((x, extra))

def intrinsic_reward(updated_s):
    f1 = float(updated_s[120])
    f2 = float(updated_s[121])
    f3 = float(updated_s[122])
    f4 = float(updated_s[123])
    
    # Phi(s): reward recent trend (f1), penalize downside risk (f2), 
    # reward price above MA (f3), and reward volume acceleration (f4)
    # Balanced weighting: trend and price position dominate; downside risk strongly penalized
    phi = np.tanh(0.7 * np.tanh(f1) - 1.0 * np.tanh(f2) + 0.5 * np.tanh(f3) + 0.3 * np.tanh(f4))
    
    phi = float(np.nan_to_num(phi, nan=0.0, posinf=0.0, neginf=0.0))
    return phi