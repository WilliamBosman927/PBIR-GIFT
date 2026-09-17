import numpy as np

def revise_state(s):
    x = np.asarray(s, dtype=np.float64)
    closes = x[0::6]
    opens = x[1::6]
    highs = x[2::6]
    lows = x[3::6]
    volumes = x[4::6]
    adjusted_closes = x[5::6]
    
    # f1: 20-day log return (trend strength)
    f1 = float(np.log(closes[-1] / (closes[0] + 1e-8)))
    
    # f2: downside deviation (std of negative daily log returns only)
    log_returns = np.diff(np.log(closes + 1e-8))
    neg_returns = log_returns[log_returns < 0]
    f2 = float(np.std(neg_returns) + 1e-8) if len(neg_returns) > 1 else 0.0
    
    # f3: normalized 5-day high-low range (volatility proxy)
    hl_spreads = (highs[-5:] - lows[-5:]) / (closes[-5:] + 1e-8)
    f3 = float(np.mean(hl_spreads))
    
    # f4: volume acceleration (recent avg vol vs prior avg vol)
    vol_recent = np.mean(volumes[-5:])
    vol_prior = np.mean(volumes[:-5]) if len(volumes[:-5]) > 0 else np.mean(volumes)
    f4 = float((vol_recent - vol_prior) / (np.std(volumes) + 1e-8)) if np.std(volumes) > 0 else 0.0
    
    extra = np.array([f1, f2, f3, f4], dtype=np.float64)
    extra = np.nan_to_num(extra, nan=0.0, posinf=0.0, neginf=0.0)
    return np.concatenate((x, extra))

def intrinsic_reward(updated_s):
    f1 = float(updated_s[120])
    f2 = float(updated_s[121])
    f3 = float(updated_s[122])
    f4 = float(updated_s[123])
    
    # Phi(s): reward upward trend, penalize downside risk and price dispersion,
    # weakly reward volume acceleration (liquidity signal)
    phi = np.tanh(1.2 * f1 - 1.0 * f2 - 0.8 * f3 + 0.3 * f4)
    
    phi = float(np.nan_to_num(phi, nan=0.0, posinf=0.0, neginf=0.0))
    return phi