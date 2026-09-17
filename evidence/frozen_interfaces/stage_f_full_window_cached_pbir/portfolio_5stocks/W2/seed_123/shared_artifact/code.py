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
    
    # f2: realized volatility (std of 20-day returns, clipped for stability)
    f2 = float(np.std(returns)) + 1e-8
    
    # f3: price deviation from 20-day SMA, normalized by realized vol (mean-reversion signal)
    sma20 = float(np.mean(closes))
    f3 = float((closes[-1] - sma20) / (f2 + 1e-8))
    
    # f4: volume z-score relative to 20-day volume distribution
    vol_mean = float(np.mean(volumes))
    vol_std = float(np.std(volumes)) + 1e-8
    f4 = float((volumes[-1] - vol_mean) / vol_std)
    
    extra = np.array([f1, f2, f3, f4], dtype=np.float64)
    extra = np.nan_to_num(extra, nan=0.0, posinf=0.0, neginf=0.0)
    return np.concatenate((x, extra))

def intrinsic_reward(updated_s):
    f1 = float(updated_s[120])
    f2 = float(updated_s[121])
    f3 = float(updated_s[122])
    f4 = float(updated_s[123])
    
    # Phi: reward trend (f1), penalize volatility (f2), reward mean-reversion (f3), moderate volume surprise (f4)
    # Bounded via tanh; emphasizes stability in balanced regime per diagnostics
    phi = np.tanh(0.7 * f1 - 0.6 * f2 + 0.5 * f3 + 0.2 * np.clip(f4, -1.5, 1.5))
    
    phi = float(np.nan_to_num(phi, nan=0.0, posinf=0.0, neginf=0.0))
    return phi