import numpy as np

def revise_state(s):
    x = np.asarray(s, dtype=np.float64)
    closes = x[0::6]
    highs = x[2::6]
    lows = x[3::6]
    volumes = x[4::6]
    
    # f1: 10-day log momentum (normalized by 20-day std of closes)
    f1 = float(np.log(closes[-1] / (closes[-10] + 1e-8))) / (np.std(closes) + 1e-8) if len(closes) >= 10 else 0.0
    
    # f2: downside volatility (std of negative returns over last 20 days)
    returns = np.diff(closes) / (np.abs(closes[:-1]) + 1e-8)
    neg_returns = returns[returns < 0]
    f2 = float(np.std(neg_returns)) if len(neg_returns) > 0 else 0.0
    
    # f3: realized volatility (std of daily returns over last 20 days)
    f3 = float(np.std(returns)) if len(returns) > 0 else 0.0
    
    # f4: volume z-score vs 20-day mean/std
    vol_mean = float(np.mean(volumes)) + 1e-8
    vol_std = float(np.std(volumes)) + 1e-8
    f4 = float((volumes[-1] - vol_mean) / vol_std) if vol_std > 1e-8 else 0.0
    
    extra = np.array([f1, f2, f3, f4], dtype=np.float64)
    extra = np.nan_to_num(extra, nan=0.0, posinf=0.0, neginf=0.0)
    return np.concatenate((x, extra))

def intrinsic_reward(updated_s):
    f1 = float(updated_s[120])
    f2 = float(updated_s[121])
    f3 = float(updated_s[122])
    f4 = float(updated_s[123])
    
    # Phi: reward trend strength, penalize downside risk and total volatility,
    # mildly reward moderate positive volume conviction; bounded via tanh
    phi = np.tanh(0.9 * f1 - 0.7 * f2 - 0.5 * f3 + 0.2 * np.tanh(f4))
    
    phi = float(np.nan_to_num(phi, nan=0.0, posinf=0.0, neginf=0.0))
    return phi