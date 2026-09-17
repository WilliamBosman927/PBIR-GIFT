import numpy as np

def revise_state(s):
    x = np.asarray(s, dtype=np.float64)
    closes = x[0::6]
    opens = x[1::6]
    highs = x[2::6]
    lows = x[3::6]
    volumes = x[4::6]
    adjusted_closes = x[5::6]
    
    # 20-day log returns (stable denominator)
    returns = np.log(closes[1:] / (closes[:-1] + 1e-8))
    
    # f1: 5-day momentum (mean return over last 5 days)
    f1 = float(np.mean(returns[-5:]) if len(returns) >= 5 else 0.0)
    
    # f2: 20-day realized volatility (std of daily returns)
    f2 = float(np.std(returns) + 1e-8)
    
    # f3: current price deviation from 20-day simple moving average (normalized by vol)
    sma20 = float(np.mean(closes))
    f3 = float((closes[-1] - sma20) / (f2 + 1e-8)) if f2 > 0 else 0.0
    
    # f4: normalized volume ratio (last volume / 20-day avg volume)
    avg_vol = float(np.mean(volumes))
    f4 = float(volumes[-1] / (avg_vol + 1e-8)) if avg_vol > 0 else 0.0
    
    extra = np.array([f1, f2, f3, f4], dtype=np.float64)
    extra = np.nan_to_num(extra, nan=0.0, posinf=0.0, neginf=0.0)
    return np.concatenate((x, extra))

def intrinsic_reward(updated_s):
    f1 = float(updated_s[120])
    f2 = float(updated_s[121])
    f3 = float(updated_s[122])
    f4 = float(updated_s[123])
    
    # Phi(s): reward trend, penalize high volatility and extreme price deviation,
    # moderate volume boost (but cap extreme values via tanh)
    phi = np.tanh(1.0 * f1 - 0.8 * f2 - 0.3 * abs(f3) + 0.2 * np.clip(f4, 0.0, 3.0))
    
    phi = float(np.nan_to_num(phi, nan=0.0, posinf=0.0, neginf=0.0))
    return phi