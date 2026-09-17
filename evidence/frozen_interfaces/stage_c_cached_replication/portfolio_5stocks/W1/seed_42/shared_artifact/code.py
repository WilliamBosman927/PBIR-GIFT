import numpy as np

def revise_state(s):
    x = np.asarray(s, dtype=np.float64)
    closes = x[0::6]
    opens = x[1::6]
    highs = x[2::6]
    lows = x[3::6]
    volumes = x[4::6]
    adjusted_closes = x[5::6]
    
    # f1: 5-day momentum (sum of last 5 daily returns), normalized by 20-day volatility
    returns = np.diff(closes) / (np.abs(closes[:-1]) + 1e-8)
    vol_20d = float(np.std(returns)) + 1e-8
    f1 = float(np.sum(returns[-5:]) / vol_20d) if len(returns) >= 5 else 0.0
    
    # f2: realized volatility (std of 20 daily returns), robustified
    f2 = float(np.std(returns)) + 1e-8
    
    # f3: high-low range ratio (normalized by close) — measure of intraday dispersion
    hl_range = highs - lows
    hl_ratio = hl_range / (np.abs(closes) + 1e-8)
    f3 = float(np.mean(hl_ratio))  # 20-day average range intensity
    
    # f4: volume trend — slope of linear fit to last 10 volume observations (robust z-score style)
    vol_recent = volumes[-10:]
    n = len(vol_recent)
    if n >= 2:
        t = np.arange(n)
        t_centered = t - np.mean(t)
        vol_centered = vol_recent - np.mean(vol_recent)
        slope = np.sum(t_centered * vol_centered) / (np.sum(t_centered**2) + 1e-8)
        f4 = float(slope / (np.std(vol_recent) + 1e-8))
    else:
        f4 = 0.0
    
    extra = np.array([f1, f2, f3, f4], dtype=np.float64)
    extra = np.nan_to_num(extra, nan=0.0, posinf=0.0, neginf=0.0)
    return np.concatenate((x, extra))

def intrinsic_reward(updated_s):
    f1 = float(updated_s[120])
    f2 = float(updated_s[121])
    f3 = float(updated_s[122])
    f4 = float(updated_s[123])
    # Phi: reward trend strength (f1), penalize volatility (f2), penalize excessive intraday dispersion (f3), penalize aggressive volume trend (|f4|)
    phi = np.tanh(0.8 * f1 - 0.6 * f2 - 0.4 * f3 - 0.3 * np.abs(f4))
    phi = float(np.nan_to_num(phi, nan=0.0, posinf=0.0, neginf=0.0))
    return phi