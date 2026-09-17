"""Look-ahead-safe traditional portfolio baselines for Stage-H.

Every action is computed from the current close and earlier observations.  It
is then passed through :class:`portfolio_env.PortfolioEnv`, so the evaluation
uses exactly the same action timing, long-only simplex, cash asset, and
transaction-cost accounting as GIFT/PPO.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from metrics import calmar_ratio, max_drawdown, sharpe_ratio, sortino_ratio
from portfolio_env import PortfolioEnv
from supplemental_experiment_utils import PROJECT_DIR


METHODS = ("sma", "wma", "atr", "bollinger", "turn_of_month", "xgboost")
DETERMINISTIC_METHODS = ("sma", "wma", "atr", "bollinger", "turn_of_month")


def load_market_frames(data_path: Path, tickers: list[str]) -> dict[str, pd.DataFrame]:
    with data_path.open("rb") as handle:
        raw = pickle.load(handle)
    rows: dict[str, list[dict[str, Any]]] = {ticker: [] for ticker in tickers}
    for date in sorted(raw):
        price = raw[date].get("price", {})
        for ticker in tickers:
            item = price.get(ticker, {})
            close = float(item.get("close", 0.0) or 0.0)
            adjusted = float(item.get("adjusted_close", close) or close)
            ratio = adjusted / close if close > 0 else 1.0
            rows[ticker].append({
                "date": str(date),
                "close": adjusted,
                "high": float(item.get("high", close) or close) * ratio,
                "low": float(item.get("low", close) or close) * ratio,
                "volume": float(item.get("volume", 0.0) or 0.0),
            })
    frames = {}
    for ticker, values in rows.items():
        frame = pd.DataFrame(values).set_index("date").sort_index()
        if frame.empty or (frame["close"] <= 0).any():
            raise ValueError(f"Invalid price history for {ticker} in {data_path}")
        frames[ticker] = frame
    return frames


def _series_until(frame: pd.DataFrame, date: str, column: str = "close") -> np.ndarray:
    if date not in frame.index:
        raise KeyError(f"Date {date} not found in market frame")
    location = int(frame.index.get_loc(date))
    return frame[column].iloc[:location + 1].to_numpy(dtype=float)


def _wma(values: np.ndarray, window: int) -> float:
    recent = values[-window:]
    weights = np.arange(1, len(recent) + 1, dtype=float)
    return float(np.dot(recent, weights) / weights.sum())


def _atr(frame: pd.DataFrame, date: str, window: int) -> float:
    location = int(frame.index.get_loc(date))
    part = frame.iloc[:location + 1]
    high = part["high"].to_numpy(dtype=float)
    low = part["low"].to_numpy(dtype=float)
    close = part["close"].to_numpy(dtype=float)
    if len(close) < 2:
        return 0.0
    previous = np.roll(close, 1)
    true_range = np.maximum.reduce([
        high - low,
        np.abs(high - previous),
        np.abs(low - previous),
    ])[1:]
    return float(np.mean(true_range[-window:])) if true_range.size else 0.0


def _positive_weights(scores: list[float]) -> np.ndarray:
    risky = np.maximum(np.asarray(scores, dtype=float), 0.0)
    output = np.zeros(len(scores) + 1, dtype=float)
    total = float(risky.sum())
    if total <= 1e-12:
        output[-1] = 1.0
    else:
        output[:-1] = risky / total
    return output


def _turn_of_month_active(
    all_dates: list[str], date: str, first_days: int, last_days: int,
) -> bool:
    stamp = pd.Timestamp(date)
    same_month = [
        value for value in all_dates
        if pd.Timestamp(value).year == stamp.year and pd.Timestamp(value).month == stamp.month
    ]
    if date not in same_month:
        return False
    position = same_month.index(date)
    return position < first_days or position >= max(0, len(same_month) - last_days)


def rule_weights(
    method: str,
    frames: dict[str, pd.DataFrame],
    tickers: list[str],
    date: str,
    parameters: dict[str, Any],
) -> np.ndarray:
    """Return an N+1 long-only vector using data no later than ``date``."""
    scores: list[float] = []
    if method == "turn_of_month":
        calendar = list(frames[tickers[0]].index)
        active = _turn_of_month_active(
            calendar, date,
            int(parameters["first_trading_days"]),
            int(parameters["last_trading_days"]),
        )
        if active:
            return np.concatenate([
                np.full(len(tickers), 1.0 / len(tickers)), np.array([0.0])])
        return np.concatenate([np.zeros(len(tickers)), np.array([1.0])])

    for ticker in tickers:
        frame = frames[ticker]
        close = _series_until(frame, date)
        if method in {"sma", "wma"}:
            short = int(parameters["short_window"])
            long = int(parameters["long_window"])
            if len(close) < long:
                scores.append(0.0)
                continue
            if method == "sma":
                short_value = float(np.mean(close[-short:]))
                long_value = float(np.mean(close[-long:]))
            else:
                short_value = _wma(close, short)
                long_value = _wma(close, long)
            scores.append(short_value / max(long_value, 1e-12) - 1.0)
        elif method == "atr":
            atr_window = int(parameters["atr_window"])
            momentum = int(parameters["momentum_window"])
            if len(close) <= max(atr_window, momentum):
                scores.append(0.0)
                continue
            volatility = _atr(frame, date, atr_window)
            score = (close[-1] - close[-momentum - 1]) / max(volatility, 1e-12)
            scores.append(score - float(parameters.get("minimum_score", 0.0)))
        elif method == "bollinger":
            window = int(parameters["window"])
            if len(close) < window:
                scores.append(0.0)
                continue
            recent = close[-window:]
            center = float(np.mean(recent))
            scale = float(np.std(recent, ddof=1))
            z_score = (close[-1] - center) / max(scale, 1e-12)
            entry = float(parameters.get("entry_z", 1.0))
            mode = str(parameters.get("mode", "breakout"))
            if mode == "breakout":
                scores.append(z_score - entry)
            elif mode == "mean_reversion":
                scores.append(-z_score - entry)
            else:
                raise ValueError(f"Unknown Bollinger mode: {mode}")
        else:
            raise ValueError(f"rule_weights does not handle method={method}")
    return _positive_weights(scores)


def _rsi(close: np.ndarray, window: int = 14) -> float:
    if len(close) <= window:
        return 50.0
    changes = np.diff(close[-window - 1:])
    gains = float(np.mean(np.maximum(changes, 0.0)))
    losses = float(np.mean(np.maximum(-changes, 0.0)))
    if losses <= 1e-12:
        return 100.0
    return float(100.0 - 100.0 / (1.0 + gains / losses))


def xgboost_features(
    frame: pd.DataFrame, date: str, ticker_index: int, ticker_count: int,
) -> np.ndarray | None:
    location = int(frame.index.get_loc(date))
    if location < 60:
        return None
    part = frame.iloc[:location + 1]
    close = part["close"].to_numpy(dtype=float)
    volume = part["volume"].to_numpy(dtype=float)
    returns = np.diff(close) / np.maximum(close[:-1], 1e-12)
    values = [
        close[-1] / close[-2] - 1.0,
        close[-1] / close[-6] - 1.0,
        close[-1] / close[-11] - 1.0,
        close[-1] / close[-21] - 1.0,
        float(np.std(returns[-10:], ddof=1)),
        float(np.std(returns[-20:], ddof=1)),
        close[-1] / float(np.mean(close[-20:])) - 1.0,
        close[-1] / float(np.mean(close[-60:])) - 1.0,
        (_rsi(close, 14) - 50.0) / 50.0,
        (volume[-1] - float(np.mean(volume[-20:])))
        / max(float(np.std(volume[-20:], ddof=1)), 1e-12),
    ]
    one_hot = [1.0 if ticker_index == index else 0.0 for index in range(ticker_count)]
    vector = np.asarray(values + one_hot, dtype=float)
    return vector if np.all(np.isfinite(vector)) else None


def fit_xgboost(
    frames: dict[str, pd.DataFrame], tickers: list[str],
    train_period: tuple[str, str], parameters: dict[str, Any], seed: int,
):
    try:
        from xgboost import XGBRegressor
    except ImportError as exc:
        raise RuntimeError(
            "Stage-H XGBoost requires the xgboost package. Install requirements.txt first."
        ) from exc
    features: list[np.ndarray] = []
    targets: list[float] = []
    start, end = train_period
    for ticker_index, ticker in enumerate(tickers):
        frame = frames[ticker]
        dates = list(frame.index)
        for index, date in enumerate(dates[:-1]):
            next_date = dates[index + 1]
            if date < start or next_date > end:
                continue
            vector = xgboost_features(frame, date, ticker_index, len(tickers))
            if vector is None:
                continue
            close_now = float(frame.loc[date, "close"])
            close_next = float(frame.loc[next_date, "close"])
            target = close_next / close_now - 1.0
            if np.isfinite(target):
                features.append(vector)
                targets.append(float(target))
    if len(features) < 200:
        raise ValueError(f"Insufficient XGBoost training samples: {len(features)}")
    allowed = {
        "n_estimators", "max_depth", "learning_rate", "subsample",
        "colsample_bytree", "reg_lambda", "objective", "n_jobs",
    }
    model_kwargs = {key: value for key, value in parameters.items() if key in allowed}
    model = XGBRegressor(random_state=seed, **model_kwargs)
    model.fit(np.vstack(features), np.asarray(targets, dtype=float))
    return model, len(features)


def xgboost_weights(
    model: Any, frames: dict[str, pd.DataFrame], tickers: list[str],
    date: str, minimum_prediction: float,
) -> np.ndarray:
    vectors = [
        xgboost_features(frames[ticker], date, index, len(tickers))
        for index, ticker in enumerate(tickers)
    ]
    if any(vector is None for vector in vectors):
        return np.concatenate([np.zeros(len(tickers)), np.array([1.0])])
    predictions = np.asarray(model.predict(np.vstack(vectors)), dtype=float)
    return _positive_weights((predictions - minimum_prediction).tolist())


def run_traditional_baseline(
    config: dict[str, Any], method: str, parameters: dict[str, Any],
    evaluation_period: tuple[str, str], adaptation_period: tuple[str, str],
    seed: int | None = None,
) -> dict[str, Any]:
    if method not in METHODS:
        raise ValueError(f"Unknown traditional baseline: {method}")
    tickers = list(config["data"]["tickers"])
    configured_path = Path(str(config["data"]["pickle_file"]))
    data_path = configured_path if configured_path.is_absolute() else PROJECT_DIR / configured_path
    frames = load_market_frames(data_path, tickers)
    transaction_cost = float(config.get("portfolio", {}).get("transaction_cost", 0.001))
    env = PortfolioEnv(
        str(data_path), config, train_period=evaluation_period,
        transaction_cost=transaction_cost)

    model = None
    training_samples = None
    training_period = None
    if method == "xgboost":
        if seed is None:
            raise ValueError("XGBoost requires an explicit seed")
        training_period = (
            str(config["experiment"]["train_period"][0]), adaptation_period[1])
        model, training_samples = fit_xgboost(
            frames, tickers, training_period, parameters, seed)

    env.reset()
    returns: list[float] = []
    weights_history: list[np.ndarray] = []
    turnovers: list[float] = []
    done = False
    while not done:
        date = str(env.dates[env.current_step])
        if method == "xgboost":
            weights = xgboost_weights(
                model, frames, tickers, date,
                float(parameters.get("minimum_prediction", 0.0)))
        else:
            weights = rule_weights(method, frames, tickers, date, parameters)
        _, _, done, info = env.step(weights)
        if info:
            returns.append(float(info["portfolio_return"]))
            turnovers.append(float(info["turnover"]))
            weights_history.append(np.asarray(info["weights"], dtype=float))
    if not returns:
        raise ValueError(f"Traditional baseline produced no returns: {method}")
    average = np.mean(np.vstack(weights_history), axis=0)
    return {
        "test_result": {
            "test_sharpe": float(sharpe_ratio(returns)),
            "test_sortino": float(sortino_ratio(returns)),
            "test_max_drawdown": float(max_drawdown(returns)),
            "test_calmar": float(calmar_ratio(returns)),
            "test_total_return": float((env.portfolio_value - 1.0) * 100),
            "test_avg_weights": {
                **{ticker: float(average[index]) for index, ticker in enumerate(tickers)},
                "CASH": float(average[-1]),
            },
            "test_average_turnover": float(np.mean(turnovers)),
        },
        "daily_returns": returns,
        "strategy_parameters": parameters,
        "xgboost_training_period": list(training_period) if training_period else None,
        "xgboost_training_samples": training_samples,
        "lookahead_safe": True,
        "action_timing": "signal_at_t_applied_by_shared_env_to_return_t_to_t_plus_1",
    }
