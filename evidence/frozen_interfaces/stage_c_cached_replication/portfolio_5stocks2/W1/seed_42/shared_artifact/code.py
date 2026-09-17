import numpy as np

def revise_state(s):
    x = np.asarray(s, dtype=np.float64)
    closes = x[0::6]
    opens = x[1::6]
    highs = x[2::6]
    lows = x[3::6]
    volumes = x[4::6]
    adjusted_closes = x[5::6]
    
    # 20-day log returns
    returns = np.log(closes[1:] / closes[:-1] + 1e-8)
    
    # f1: 5-day momentum (mean return over last 5 days)
    f1 = float(np.mean(returns[-5:]) if len(returns) >= 5 else 0.0)
    
    # f2: 20-day realized volatility (std of daily returns)
    f2 = float(np.std(returns) + 1e-8)
    
    # f3: current close vs 20-day SMA deviation (normalized by vol)
    sma20 = float(np.mean(closes))
    price_dev = (closes[-1] - sma20) / (f2 + 1e-8)
    f3 = float(price_dev)
    
    # f4: normalized volume (current volume / 20-day avg volume)
    avg_vol = float(np.mean(volumes)) + 1e-8
    f4 = float(volumes[-1] / avg_vol)
    
    extra = np.array([f1, f2, f3, f4], dtype=np.float64)
    extra = np.nan_to_num(extra, nan=0.0, posinf=0.0, neginf=0.0)
    return np.concatenate((x, extra))

def intrinsic_reward(updated_s):
    f1 = float(updated_s[120])
    f2 = float(updated_s[121])
    f3 = float(updated_s[122])
    f4 = float(updated_s[123])
    
    # Phi: reward trend, penalize volatility & extreme deviations, moderate volume boost
    # Bounded via tanh to [-1, 1]
    phi = np.tanh(2.0 * f1 - 1.5 * f2 - 0.8 * abs(f3) + 0.3 * f4)
    
    phi = float(np.nan_to_num(phi, nan=0.0, posinf=0.0, neginf=0.0))
    return phi