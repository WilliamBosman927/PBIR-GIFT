import numpy as np

def revise_state(s):
    x = np.asarray(s, dtype=np.float64)
    closes = x[0::6]
    opens = x[1::6]
    highs = x[2::6]
    lows = x[3::6]
    volumes = x[4::6]
    adjusted_closes = x[5::6]
    
    # f1: 5-day momentum normalized by 10-day realized volatility (robust to volatile regimes)
    returns = np.diff(closes) / (np.abs(closes[:-1]) + 1e-8)
    vol_10d = float(np.std(returns[-10:])) + 1e-8 if len(returns) >= 10 else 1e-8
    f1 = (closes[-1] - closes[-5]) / (vol_10d * (np.abs(closes[-5]) + 1e-8)) if len(closes) >= 5 else 0.0
    
    # f2: downside deviation (std of negative returns only, captures tail risk)
    negative_returns = returns[returns < 0]
    f2 = float(np.std(negative_returns)) if len(negative_returns) > 0 else 0.0
    
    # f3: normalized high-low range (volatility proxy, stable across regimes)
    hl_ranges = (highs - lows) / (np.abs(closes) + 1e-8)
    f3 = float(np.mean(hl_ranges))
    
    # f4: price z-score relative to 20-day mean (mean-reversion signal)
    price_mean = float(np.mean(closes))
    price_std = float(np.std(closes)) + 1e-8
    f4 = (closes[-1] - price_mean) / price_std if len(closes) > 1 else 0.0
    
    extra = np.array([f1, f2, f3, f4], dtype=np.float64)
    extra = np.nan_to_num(extra, nan=0.0, posinf=0.0, neginf=0.0)
    return np.concatenate((x, extra))

def intrinsic_reward(updated_s):
    f1 = float(updated_s[120])
    f2 = float(updated_s[121])
    f3 = float(updated_s[122])
    f4 = float(updated_s[123])
    # Phi: reward trend & mean-reversion, penalize downside risk & range volatility
    phi = np.tanh(0.5 * f1 - 0.4 * f2 - 0.3 * f3 + 0.3 * f4)
    phi = float(np.nan_to_num(phi, nan=0.0, posinf=0.0, neginf=0.0))
    return phi