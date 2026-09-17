import numpy as np

def revise_state(s):
    x = np.asarray(s, dtype=np.float64)
    closes = x[0::6]
    opens = x[1::6]
    highs = x[2::6]
    lows = x[3::6]
    volumes = x[4::6]
    adjusted_closes = x[5::6]
    
    # 1. 5-day momentum normalized by 10-day volatility (robust to volatile regimes)
    returns = np.diff(closes) / (np.abs(closes[:-1]) + 1e-8)
    vol_10d = np.std(returns[-10:]) + 1e-8 if len(returns) >= 10 else 1e-8
    f1 = float(np.sum(returns[-5:]) / vol_10d) if len(returns) >= 5 else 0.0
    
    # 2. Downside deviation: std of negative returns only (defensive risk measure)
    neg_returns = returns[returns < 0]
    f2 = float(np.std(neg_returns)) if len(neg_returns) > 0 else 0.0
    
    # 3. Price deviation from 20-day simple moving average (mean-reversion signal)
    sma_20 = np.mean(closes)
    f3 = float((closes[-1] - sma_20) / (sma_20 + 1e-8)) if len(closes) > 0 else 0.0
    
    # 4. Volume z-score relative to full 20-day history (stable baseline)
    vol_mean = np.mean(volumes)
    vol_std = np.std(volumes) + 1e-8
    f4 = float((volumes[-1] - vol_mean) / vol_std) if len(volumes) > 1 else 0.0
    
    extra = np.array([f1, f2, f3, f4], dtype=np.float64)
    extra = np.nan_to_num(extra, nan=0.0, posinf=0.0, neginf=0.0)
    return np.concatenate((x, extra))

def intrinsic_reward(updated_s):
    f1 = float(updated_s[120])
    f2 = float(updated_s[121])
    f3 = float(updated_s[122])
    f4 = float(updated_s[123])
    # Phi: reward trend (f1), penalize downside risk (f2), reward mean-reversion (f3), neutral volume (f4)
    phi = 0.5 * np.tanh(f1) - 0.4 * np.tanh(f2) + 0.3 * np.tanh(-f3) - 0.1 * np.tanh(np.abs(f4))
    phi = float(np.nan_to_num(phi, nan=0.0, posinf=0.0, neginf=0.0))
    return phi