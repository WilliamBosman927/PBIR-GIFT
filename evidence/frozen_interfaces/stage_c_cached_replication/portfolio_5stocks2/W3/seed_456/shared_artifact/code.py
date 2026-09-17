import numpy as np

def revise_state(s):
    x = np.asarray(s, dtype=np.float64)
    closes = x[0::6]
    opens = x[1::6]
    highs = x[2::6]
    lows = x[3::6]
    volumes = x[4::6]
    adjusted_closes = x[5::6]
    
    # Use adjusted_close for robustness (accounts for splits/dividends)
    prices = adjusted_closes
    
    # 1. 5-day momentum: sum of daily returns, normalized by 10-day return volatility
    returns = np.diff(prices) / (np.abs(prices[:-1]) + 1e-8)
    mom_5d = np.sum(returns[-5:]) if len(returns) >= 5 else 0.0
    vol_10d = np.std(returns[-10:]) if len(returns) >= 10 else 1e-8
    f1 = mom_5d / (vol_10d + 1e-8)
    
    # 2. Downside risk: std of negative returns only (semi-deviation)
    neg_returns = returns[returns < 0]
    f2 = np.std(neg_returns) if len(neg_returns) > 1 else 0.0
    
    # 3. Realized volatility: std of daily returns over full 20 days
    f3 = float(np.std(returns)) if len(returns) > 1 else 0.0
    
    # 4. Price z-score relative to 20-day SMA (mean-reversion signal)
    sma_20 = np.mean(prices)
    f4 = (prices[-1] - sma_20) / (np.std(prices) + 1e-8)
    
    extra = np.array([f1, f2, f3, f4], dtype=np.float64)
    extra = np.nan_to_num(extra, nan=0.0, posinf=0.0, neginf=0.0)
    return np.concatenate((x, extra))

def intrinsic_reward(updated_s):
    f1 = float(updated_s[120])
    f2 = float(updated_s[121])
    f3 = float(updated_s[122])
    f4 = float(updated_s[123])
    # Reward strong positive momentum (f1), penalize downside risk (f2), 
    # penalize high realized volatility (f3), and reward moderate mean-reversion (f4 near zero)
    phi = np.tanh(0.8 * f1 - 0.5 * f2 - 0.4 * f3 - 0.3 * np.abs(f4))
    phi = float(np.nan_to_num(phi, nan=0.0, posinf=0.0, neginf=0.0))
    return phi