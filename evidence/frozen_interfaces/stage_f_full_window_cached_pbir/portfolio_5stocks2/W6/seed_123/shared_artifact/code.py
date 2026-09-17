import numpy as np

def revise_state(s):
    x = np.asarray(s, dtype=np.float64)
    closes = x[0::6]
    highs = x[2::6]
    lows = x[3::6]
    volumes = x[4::6]
    returns = np.diff(closes) / (np.abs(closes[:-1]) + 1e-8)
    
    # f1: 5-day simple momentum (robust to volatility regime shifts)
    f1 = (closes[-1] - closes[-5]) / (np.abs(closes[-5]) + 1e-8) if len(closes) >= 5 else 0.0
    
    # f2: downside volatility (std of negative returns only; zero if no negatives)
    negative_returns = returns[returns < 0]
    f2 = float(np.std(negative_returns)) + 1e-8 if len(negative_returns) > 1 else 1e-8
    
    # f3: current price deviation from 20-day mean (normalized by std for stability)
    price_mean = float(np.mean(closes))
    price_std = float(np.std(closes)) + 1e-8
    f3 = (closes[-1] - price_mean) / price_std if price_std > 1e-8 else 0.0
    
    # f4: volume z-score (deviation from 20-day volume mean, normalized by std)
    vol_mean = float(np.mean(volumes))
    vol_std = float(np.std(volumes)) + 1e-8
    f4 = (volumes[-1] - vol_mean) / vol_std if vol_std > 1e-8 else 0.0
    
    extra = np.array([f1, f2, f3, f4], dtype=np.float64)
    extra = np.nan_to_num(extra, nan=0.0, posinf=0.0, neginf=0.0)
    return np.concatenate((x, extra))

def intrinsic_reward(updated_s):
    f1 = float(updated_s[120])
    f2 = float(updated_s[121])
    f3 = float(updated_s[122])
    f4 = float(updated_s[123])
    
    # Phi(s): reward trend (f1), penalize downside risk (f2), reward mean-reversion (f3 sign matters),
    # and reward volume confirmation (f4); all bounded via tanh for stability
    phi = np.tanh(1.0 * f1 - 0.9 * f2 + 0.7 * np.tanh(f3) + 0.3 * f4)
    
    phi = float(np.nan_to_num(phi, nan=0.0, posinf=0.0, neginf=0.0))
    return phi