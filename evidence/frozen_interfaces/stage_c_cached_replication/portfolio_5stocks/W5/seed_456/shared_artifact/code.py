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
    
    # f1: 5-day momentum (log return over last 5 days)
    f1 = float(np.sum(returns[-5:]) if len(returns) >= 5 else 0.0)
    
    # f2: realized volatility (std of 20-day returns, bounded)
    f2 = float(np.std(returns) + 1e-8)
    
    # f3: downside deviation (std of negative returns only)
    neg_returns = returns[returns < 0]
    f3 = float(np.std(neg_returns) + 1e-8) if len(neg_returns) > 1 else 0.0
    
    # f4: normalized high-low range (average (high - low) / close over 20 days)
    hl_ranges = (highs - lows) / (closes + 1e-8)
    f4 = float(np.mean(hl_ranges))
    
    extra = np.array([f1, f2, f3, f4], dtype=np.float64)
    extra = np.nan_to_num(extra, nan=0.0, posinf=0.0, neginf=0.0)
    return np.concatenate((x, extra))

def intrinsic_reward(updated_s):
    f1 = float(updated_s[120])
    f2 = float(updated_s[121])
    f3 = float(updated_s[122])
    f4 = float(updated_s[123])
    
    # Potential: reward momentum, penalize volatility and downside risk, moderate by price range
    # Scale to [-1, 1] via tanh; avoid division by zero with safe denominators
    phi = np.tanh(1.0 * f1 - 0.5 * f2 - 0.7 * f3 - 0.3 * f4)
    
    phi = float(np.nan_to_num(phi, nan=0.0, posinf=0.0, neginf=0.0))
    return phi