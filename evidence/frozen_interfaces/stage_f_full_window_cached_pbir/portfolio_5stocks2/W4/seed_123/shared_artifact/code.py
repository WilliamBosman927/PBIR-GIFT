import numpy as np

def revise_state(s):
    x = np.asarray(s, dtype=np.float64)
    closes = x[0::6]
    opens = x[1::6]
    highs = x[2::6]
    lows = x[3::6]
    volumes = x[4::6]
    adjusted_closes = x[5::6]
    
    # f1: 5-day momentum (cumulative return over last 5 days) — more stable than 3-day, better IC alignment
    returns = np.diff(closes) / (np.abs(closes[:-1]) + 1e-8)
    f1 = float(np.sum(returns[-5:]) if len(returns) >= 5 else 0.0)
    
    # f2: 10-day downside deviation (std of negative returns in last 10 days) — longer horizon improves IC for s[121]
    recent_returns = returns[-10:] if len(returns) >= 10 else returns
    neg_returns = recent_returns[recent_returns < 0]
    f2 = float(np.std(neg_returns)) if len(neg_returns) > 1 else 0.0
    
    # f3: normalized 20-day high-low range volatility (std(high-low)/mean(close)) — robust, matches IC pattern
    hl_range = highs - lows
    f3 = float(np.std(hl_range) / (np.mean(closes) + 1e-8))
    
    # f4: volume acceleration ratio (current volume / 10-day avg volume), clipped to [0.1, 10.0] — avoids z-score overfitting seen in s[123]
    vol_avg_10 = float(np.mean(volumes[-10:])) + 1e-8 if len(volumes) >= 10 else float(np.mean(volumes)) + 1e-8
    f4 = float(np.clip(volumes[-1] / vol_avg_10, 0.1, 10.0))
    
    extra = np.array([f1, f2, f3, f4], dtype=np.float64)
    extra = np.nan_to_num(extra, nan=0.0, posinf=0.0, neginf=0.0)
    return np.concatenate((x, extra))

def intrinsic_reward(updated_s):
    f1 = float(updated_s[120])
    f2 = float(updated_s[121])
    f3 = float(updated_s[122])
    f4 = float(updated_s[123])
    
    # Phi(s): reward longer momentum (f1), penalize downside risk (f2), penalize range volatility (f3),
    # and reward moderate volume acceleration (f4 near 1.0 → highest potential; |log(f4)| penalty)
    phi = np.tanh(1.0 * f1 - 0.8 * f2 - 0.6 * f3 - 0.4 * np.abs(np.log(f4 + 1e-8)))
    
    phi = float(np.nan_to_num(phi, nan=0.0, posinf=0.0, neginf=0.0))
    return phi