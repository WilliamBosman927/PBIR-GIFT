import numpy as np

def revise_state(s):
    x = np.asarray(s, dtype=np.float64)
    closes = x[0::6]
    opens = x[1::6]
    highs = x[2::6]
    lows = x[3::6]
    volumes = x[4::6]
    adjusted_closes = x[5::6]
    
    # Log returns (19 deltas from 20 closes)
    returns = np.diff(np.log(closes + 1e-8))
    
    # f1: 5-day momentum — more robust to noise than 3-day, aligns with defensive regime need for stability
    f1 = float(np.sum(returns[-5:]) if len(returns) >= 5 else 0.0)
    
    # f2: realized volatility — std of all log returns, stabilized
    f2 = float(np.std(returns)) + 1e-8
    
    # f3: price z-score relative to 20-day mean/std — defensive level signal (as before, high SHAP)
    mu = float(np.mean(closes))
    sigma = float(np.std(closes)) + 1e-8
    f3 = float((closes[-1] - mu) / sigma)
    
    # f4: downside deviation — std of negative log returns only; zero if insufficient data
    neg_returns = returns[returns < 0]
    f4 = float(np.std(neg_returns)) if len(neg_returns) > 1 else 0.0
    
    extra = np.array([f1, f2, f3, f4], dtype=np.float64)
    extra = np.nan_to_num(extra, nan=0.0, posinf=0.0, neginf=0.0)
    return np.concatenate((x, extra))

def intrinsic_reward(updated_s):
    f1 = float(updated_s[120])
    f2 = float(updated_s[121])
    f3 = float(updated_s[122])
    f4 = float(updated_s[123])
    
    # Phi: reward momentum, penalize volatility & downside risk, and strongly penalize extreme levels
    # Use f3^2 for symmetric penalty on price extremes (defensive), and invert f2/f4 (harmful IC) to reward low-risk states
    phi = 0.7 * np.tanh(f1) - 0.6 * np.tanh(f2) - 0.5 * np.tanh(np.abs(f4)) - 0.4 * np.tanh(f3**2)
    
    phi = float(np.nan_to_num(phi, nan=0.0, posinf=0.0, neginf=0.0))
    return phi