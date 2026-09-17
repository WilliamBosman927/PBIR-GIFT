import numpy as np

def revise_state(s):
    x = np.asarray(s, dtype=np.float64)
    closes = x[0::6]
    opens = x[1::6]
    highs = x[2::6]
    lows = x[3::6]
    volumes = x[4::6]
    adjusted_closes = x[5::6]
    
    # 1. 10-day price momentum (normalized by 10-day volatility)
    returns = np.diff(closes) / (np.abs(closes[:-1]) + 1e-8)
    mom_10d = (closes[-1] - closes[-11]) / (np.std(closes[-11:-1]) + 1e-8) if len(closes) >= 11 else 0.0
    
    # 2. Downside deviation: std of negative returns over last 10 days
    neg_returns = returns[-10:][returns[-10:] < 0]
    f2 = float(np.std(neg_returns)) if len(neg_returns) > 0 else 0.0
    
    # 3. Current price vs 20-day simple moving average (normalized by recent vol)
    sma_20 = np.mean(closes) + 1e-8
    f3 = (closes[-1] - sma_20) / (np.std(closes) + 1e-8)
    
    # 4. Volume acceleration: z-score of volume change (current vs prior 5-day avg)
    vol_change = volumes[-1] - np.mean(volumes[-6:-1]) if len(volumes) >= 6 else 0.0
    vol_change_std = np.std(np.diff(volumes[-10:])) + 1e-8 if len(volumes) >= 10 else 1e-8
    f4 = vol_change / vol_change_std
    
    f1 = float(mom_10d)
    extra = np.array([f1, f2, f3, f4], dtype=np.float64)
    extra = np.nan_to_num(extra, nan=0.0, posinf=0.0, neginf=0.0)
    return np.concatenate((x, extra))

def intrinsic_reward(updated_s):
    f1 = float(updated_s[120])
    f2 = float(updated_s[121])
    f3 = float(updated_s[122])
    f4 = float(updated_s[123])
    phi = 0.7 * np.tanh(f1) - 0.5 * np.tanh(f2) + 0.4 * np.tanh(f3) - 0.2 * np.tanh(np.abs(f4))
    phi = float(np.nan_to_num(phi, nan=0.0, posinf=0.0, neginf=0.0))
    return phi