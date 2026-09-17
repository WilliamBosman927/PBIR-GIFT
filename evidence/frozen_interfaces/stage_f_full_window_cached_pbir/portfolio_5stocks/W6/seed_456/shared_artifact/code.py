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
    
    # f1: 10-day momentum (sum of last 10 returns)
    f1 = float(np.sum(returns[-10:]) if len(returns) >= 10 else 0.0)
    
    # f2: realized volatility (std of all 19 returns)
    f2 = float(np.std(returns)) + 1e-8
    
    # f3: normalized high-low range (current day's range / 20-day avg range)
    ranges = highs - lows
    avg_range = float(np.mean(ranges)) + 1e-8
    f3 = float(ranges[-1] / avg_range)
    
    # f4: volume z-score (current volume vs 10-day avg volume, normalized by vol std)
    vol_mean = float(np.mean(volumes[-10:])) + 1e-8
    vol_std = float(np.std(volumes[-10:])) + 1e-8
    f4 = float((volumes[-1] - vol_mean) / vol_std) if vol_std > 0 else 0.0
    
    extra = np.array([f1, f2, f3, f4], dtype=np.float64)
    extra = np.nan_to_num(extra, nan=0.0, posinf=0.0, neginf=0.0)
    return np.concatenate((x, extra))

def intrinsic_reward(updated_s):
    f1 = float(updated_s[120])
    f2 = float(updated_s[121])
    f3 = float(updated_s[122])
    f4 = float(updated_s[123])
    
    # Phi: reward sustained trend (f1), penalize volatility (f2), reward tight range (f3 near 0), moderate volume deviation (f4 near 0)
    phi = np.tanh(0.8 * f1 - 0.7 * f2 + 0.3 * (1.0 - np.abs(f3 - 1.0)) - 0.2 * np.abs(f4))
    
    phi = float(np.nan_to_num(phi, nan=0.0, posinf=0.0, neginf=0.0))
    return phi