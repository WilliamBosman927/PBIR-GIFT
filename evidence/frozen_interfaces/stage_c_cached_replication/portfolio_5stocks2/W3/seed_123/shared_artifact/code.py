import numpy as np

def revise_state(s):
    x = np.asarray(s, dtype=np.float64)
    closes = x[0::6]
    opens = x[1::6]
    highs = x[2::6]
    lows = x[3::6]
    volumes = x[4::6]
    adjusted_closes = x[5::6]
    
    # 1. 5-day normalized momentum: sum of last 5 returns, scaled by 10-day vol (trend strength)
    returns = np.diff(closes) / (np.abs(closes[:-1]) + 1e-8)
    mom_5d = np.sum(returns[-5:]) if len(returns) >= 5 else 0.0
    vol_10d = np.std(returns[-10:]) + 1e-8 if len(returns) >= 10 else 1e-8
    f1 = mom_5d / vol_10d
    
    # 2. Downside deviation: std of negative returns only (asymmetric risk)
    neg_returns = returns[returns < 0]
    f2 = np.std(neg_returns) + 1e-8 if len(neg_returns) > 1 else 0.0
    
    # 3. 20-day realized volatility (full-window defensive signal)
    f3 = np.std(returns) + 1e-8 if len(returns) >= 2 else 1e-8
    
    # 4. Price z-score vs 20-day SMA (mean-reversion anchor, robust in neutral/trending regimes)
    sma20 = np.mean(closes)
    std20 = np.std(closes) + 1e-8
    f4 = (closes[-1] - sma20) / std20
    
    extra = np.array([f1, f2, f3, f4], dtype=np.float64)
    extra = np.nan_to_num(extra, nan=0.0, posinf=0.0, neginf=0.0)
    return np.concatenate((x, extra))

def intrinsic_reward(updated_s):
    f1 = float(updated_s[120])
    f2 = float(updated_s[121])
    f3 = float(updated_s[122])
    f4 = float(updated_s[123])
    
    # Phi(s): reward strong normalized momentum (f1), penalize downside risk (f2),
    # penalize high volatility (f3), and mildly penalize extreme price deviation (f4^2)
    phi = np.tanh(0.9 * f1 - 0.6 * f2 - 0.5 * f3 - 0.25 * f4**2)
    phi = float(np.nan_to_num(phi, nan=0.0, posinf=0.0, neginf=0.0))
    return phi