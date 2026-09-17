import numpy as np

def revise_state(s):
    x = np.asarray(s, dtype=np.float64)
    closes = x[0::6]
    opens = x[1::6]
    highs = x[2::6]
    lows = x[3::6]
    volumes = x[4::6]
    adjusted_closes = x[5::6]
    
    # 1. 5-day momentum normalized by 10-day return volatility (trend strength)
    returns = np.diff(closes) / (np.abs(closes[:-1]) + 1e-8)
    vol_10d = np.std(returns[-10:]) + 1e-8 if len(returns) >= 10 else 1e-8
    f1 = float(np.sum(returns[-5:]) / vol_10d) if len(returns) >= 5 else 0.0
    
    # 2. Downside deviation: std of negative returns over last 10 days
    recent_returns = returns[-10:] if len(returns) >= 10 else returns
    neg_returns = recent_returns[recent_returns < 0]
    f2 = float(np.std(neg_returns)) if len(neg_returns) > 1 else 0.0
    
    # 3. Normalized high-low range (volatility proxy), median-filtered over last 5 days
    hl_ranges = (highs - lows) / (np.abs(closes) + 1e-8)
    f3 = float(np.median(hl_ranges[-5:])) if len(hl_ranges) >= 5 else 0.0
    
    # 4. Z-score of current close vs 10-day SMA, normalized by 10-day return volatility
    sma10 = np.mean(closes[-10:]) if len(closes) >= 10 else closes[-1]
    ret_vol_10 = np.std(returns[-10:]) + 1e-8 if len(returns) >= 10 else 1e-8
    f4 = float((closes[-1] - sma10) / (ret_vol_10 + 1e-8))
    
    extra = np.array([f1, f2, f3, f4], dtype=np.float64)
    extra = np.nan_to_num(extra, nan=0.0, posinf=0.0, neginf=0.0)
    return np.concatenate((x, extra))

def intrinsic_reward(updated_s):
    f1 = float(updated_s[120])
    f2 = float(updated_s[121])
    f3 = float(updated_s[122])
    f4 = float(updated_s[123])
    
    # Phi(s): reward strong positive trend (f1), penalize downside risk (f2),
    # reward low-moderate volatility (f3 near typical ~0.02), penalize extreme mean-reversion (|f4|)
    # Emphasize f1 and f3 (strong IC/SHAP in neutral & trending regimes), de-emphasize f4 (low IC, high SHAP overfitting)
    phi = np.tanh(1.2 * f1 - 0.7 * np.abs(f2) + 0.5 * (1.0 - np.abs(f3 - 0.02)) - 0.3 * np.abs(f4))
    phi = float(np.nan_to_num(phi, nan=0.0, posinf=0.0, neginf=0.0))
    return phi