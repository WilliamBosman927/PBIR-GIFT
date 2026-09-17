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
    
    # f2: 20-day realized volatility (std of daily returns)
    f2 = float(np.std(returns)) + 1e-8
    
    # f3: normalized high-low range (average (high - low) / close over 20 days)
    hl_ranges = (highs - lows) / (np.abs(closes) + 1e-8)
    f3 = float(np.mean(hl_ranges))
    
    # f4: volume z-score relative to 20-day mean (normalized deviation)
    vol_mean = float(np.mean(volumes))
    vol_std = float(np.std(volumes)) + 1e-8
    f4 = float((volumes[-1] - vol_mean) / vol_std) if len(volumes) > 0 else 0.0
    
    extra = np.array([f1, f2, f3, f4], dtype=np.float64)
    extra = np.nan_to_num(extra, nan=0.0, posinf=0.0, neginf=0.0)
    return np.concatenate((x, extra))

def intrinsic_reward(updated_s):
    f1 = float(updated_s[120])
    f2 = float(updated_s[121])
    f3 = float(updated_s[122])
    f4 = float(updated_s[123])
    
    # Phi(s): reward trend, penalize volatility & range, neutral on volume
    # Scale and bound to [-1, 1] via tanh
    phi = np.tanh(0.8 * f1 - 0.6 * f2 - 0.4 * f3)
    
    phi = float(np.nan_to_num(phi, nan=0.0, posinf=0.0, neginf=0.0))
    return phi