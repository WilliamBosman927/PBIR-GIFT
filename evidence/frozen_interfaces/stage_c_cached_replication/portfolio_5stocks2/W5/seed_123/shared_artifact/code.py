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
    
    # f2: downside deviation (std of negative returns only; captures tail risk)
    negative_returns = returns[returns < 0]
    f2 = float(np.std(negative_returns)) + 1e-8 if len(negative_returns) > 1 else 0.0
    
    # f3: current price vs 20-day SMA (normalized by volatility to improve stability)
    sma20 = float(np.mean(closes)) + 1e-8
    price_dev = (closes[-1] - sma20) / (np.std(closes) + 1e-8)
    f3 = float(price_dev)
    
    # f4: volume acceleration: (last vol - prev vol) / prev vol, capped and smoothed
    if len(volumes) >= 2:
        vol_change = (volumes[-1] - volumes[-2]) / (np.abs(volumes[-2]) + 1e-8)
        f4 = float(np.clip(vol_change, -3.0, 3.0))
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
    
    # Potential: reward recent momentum (f1), penalize downside risk (f2), 
    # reward mean-reversion signal (f3: positive when below SMA → buy opportunity),
    # moderate volume acceleration (f4: reward confirmation, penalize extremes)
    phi = np.tanh(0.9 * f1 - 0.7 * f2 + 0.5 * f3 - 0.3 * np.abs(f4))
    
    phi = float(np.nan_to_num(phi, nan=0.0, posinf=0.0, neginf=0.0))
    return phi