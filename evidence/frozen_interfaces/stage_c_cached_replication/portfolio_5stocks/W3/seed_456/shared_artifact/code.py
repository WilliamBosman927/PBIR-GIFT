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
    
    # f1: 5-day momentum (sum of last 5 returns)
    f1 = float(np.sum(returns[-5:]) if len(returns) >= 5 else 0.0)
    
    # f2: realized volatility (std of all 19 returns)
    f2 = float(np.std(returns) + 1e-8)
    
    # f3: z-score of current close vs 20-day mean/std (mean-reversion signal)
    sma20 = float(np.mean(closes))
    std20 = float(np.std(closes) + 1e-8)
    f3 = float((closes[-1] - sma20) / std20)
    
    # f4: normalized average high-low range over 20 days: (high - low) / close
    hl_ranges = (highs - lows) / (np.abs(closes) + 1e-8)
    f4 = float(np.mean(hl_ranges))
    
    extra = np.array([f1, f2, f3, f4], dtype=np.float64)
    extra = np.nan_to_num(extra, nan=0.0, posinf=0.0, neginf=0.0)
    return np.concatenate((x, extra))

def intrinsic_reward(updated_s):
    f1 = float(updated_s[120])
    f2 = float(updated_s[121])
    f3 = float(updated_s[122])
    f4 = float(updated_s[123])
    
    # Phi(s): reward trend (f1), penalize total volatility (f2), reward moderate mean-reversion (f3), 
    # and penalize extreme price range (f4); bounded via tanh to [-1.0, 1.0]
    phi = np.tanh(0.8 * f1 - 0.7 * f2 - 0.3 * f3**2 - 0.4 * np.tanh(f4))
    
    phi = float(np.nan_to_num(phi, nan=0.0, posinf=0.0, neginf=0.0))
    return phi