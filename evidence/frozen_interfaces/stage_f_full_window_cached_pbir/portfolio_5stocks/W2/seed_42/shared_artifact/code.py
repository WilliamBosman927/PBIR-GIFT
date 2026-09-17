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
    
    # f1: 5-day momentum (cumulative return over last 5 days)
    f1 = float(np.sum(returns[-5:]) if len(returns) >= 5 else 0.0)
    
    # f2: downside volatility (std of negative returns only; 0 if no negatives)
    negative_returns = returns[returns < 0]
    f2 = float(np.std(negative_returns)) + 1e-8 if len(negative_returns) > 1 else 1e-8
    
    # f3: normalized close-to-open gap (average |close - open| / close over 20 days)
    gaps = np.abs(closes - opens) / (np.abs(closes) + 1e-8)
    f3 = float(np.mean(gaps))
    
    # f4: current volume z-score relative to 20-day mean
    vol_mean = float(np.mean(volumes))
    vol_std = float(np.std(volumes)) + 1e-8
    f4 = float((volumes[-1] - vol_mean) / vol_std) if len(volumes) > 1 else 0.0
    
    extra = np.array([f1, f2, f3, f4], dtype=np.float64)
    extra = np.nan_to_num(extra, nan=0.0, posinf=0.0, neginf=0.0)
    return np.concatenate((x, extra))

def intrinsic_reward(updated_s):
    f1 = float(updated_s[120])
    f2 = float(updated_s[121])
    f3 = float(updated_s[122])
    f4 = float(updated_s[123])
    
    # Phi(s): reward trend (f1), penalize downside risk (f2), penalize price instability (f3),
    # and mildly reward volume confirmation (f4) — all bounded via tanh
    phi = np.tanh(0.9 * f1 - 0.7 * f2 - 0.3 * f3 + 0.2 * np.tanh(f4))
    
    phi = float(np.nan_to_num(phi, nan=0.0, posinf=0.0, neginf=0.0))
    return phi