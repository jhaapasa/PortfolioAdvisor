from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from portfolio_advisor.models.market import ReferenceTickerMetrics


def compute_trailing_returns(ohlc: dict[str, Any]) -> dict[str, Any]:
    rows = ohlc.get("data", []) or []
    closes = [float(r.get("close", 0.0)) for r in rows]

    def trailing(n: int) -> float | None:
        if len(closes) <= n:
            return None
        c0 = closes[-n - 1]
        ct = closes[-1]
        if c0 == 0:
            return None
        return (ct / c0) - 1.0

    return {
        "instrument_id": ohlc.get("instrument_id"),
        "primary_ticker": ohlc.get("primary_ticker"),
        "as_of": ohlc.get("coverage", {}).get("end_date"),
        "windows": {
            "d1": trailing(1),
            "d5": trailing(5),
            "d21": trailing(21),
            "d252": trailing(252),
        },
        "method": "simple_total_return",
        "depends_on": ["primary.ohlc_daily"],
    }


def compute_volatility_annualized(ohlc: dict[str, Any], window: int = 21) -> dict[str, Any]:
    rows = ohlc.get("data", []) or []
    closes = [float(r.get("close", 0.0)) for r in rows]
    if len(closes) < window + 1:
        vol = None
    else:
        returns = []
        for i in range(1, len(closes)):
            prev = closes[i - 1]
            curr = closes[i]
            if prev > 0 and curr > 0:
                returns.append(math.log(curr / prev))
        if len(returns) >= window:
            tail = returns[-window:]
            mean = sum(tail) / len(tail)
            var = sum((x - mean) ** 2 for x in tail) / (len(tail) - 1) if len(tail) > 1 else 0.0
            vol = math.sqrt(var) * math.sqrt(252)
        else:
            vol = None
    return {
        "instrument_id": ohlc.get("instrument_id"),
        "primary_ticker": ohlc.get("primary_ticker"),
        "as_of": ohlc.get("coverage", {}).get("end_date"),
        "window": window,
        "annualization_factor": "sqrt(252)",
        "volatility": vol,
        "method": "std(log_returns) * annualization_factor",
        "depends_on": ["primary.ohlc_daily"],
    }


def compute_sma_series(ohlc: dict[str, Any], windows: list[int] | None = None) -> dict[str, Any]:
    win = windows or [20, 50, 100, 200]
    rows = ohlc.get("data", []) or []
    closes = [float(r.get("close", 0.0)) for r in rows]
    dates = [r.get("date") for r in rows]
    out_rows: list[dict[str, Any]] = []
    for i in range(len(closes)):
        row: dict[str, Any] = {"date": dates[i]}
        for w in win:
            if i + 1 >= w:
                window_vals = closes[i + 1 - w : i + 1]
                row[f"sma{w}"] = sum(window_vals) / w
        out_rows.append(row)
    start = None
    for r in out_rows:
        if any(k.startswith("sma") for k in r.keys() if k != "date"):
            start = r.get("date")
            break
    return {
        "instrument_id": ohlc.get("instrument_id"),
        "primary_ticker": ohlc.get("primary_ticker"),
        "windows": win,
        "data": out_rows,
        "coverage": {
            "start_date": start,
            "end_date": ohlc.get("coverage", {}).get("end_date"),
        },
        "depends_on": ["primary.ohlc_daily"],
    }


class MarketMetricsService:
    """
    Centralized service for computing market metrics (returns, Sharpe, volatility, beta).

    Design principles:
    - Single entry point for computing metrics for any ticker
    - Internal caching to avoid redundant computation within a run
    - Stateless API (except cache); safe for concurrent use
    - All calculations use same methodology for consistency
    """

    def __init__(self, stock_db_root: Path, risk_free_rate_daily: float):
        self.stock_db_root = stock_db_root
        self.risk_free_rate_daily = risk_free_rate_daily
        self._cache: dict[str, ReferenceTickerMetrics] = {}

    def compute_metrics_for_symbol(
        self, symbol: str, slug: str, horizons: list[int]
    ) -> ReferenceTickerMetrics:
        """
        Compute returns, Sharpe ratios, and volatility for a ticker.

        Args:
            symbol: ticker symbol (e.g., "SPY", "AAPL")
            slug: canonical slug for file paths
            horizons: list of horizons in days (e.g., [63, 126, 252, 504])

        Returns:
            ReferenceTickerMetrics with all computed metrics

        Caches result internally to avoid redundant computation.
        """
        cache_key = f"{symbol}:{','.join(map(str, horizons))}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Load OHLC data
        ohlc_path = self.stock_db_root / "tickers" / slug / "primary" / "ohlc_daily.json"
        ohlc_data = self._load_ohlc(ohlc_path)

        if not ohlc_data:
            raise ValueError(f"No OHLC data found for {symbol} at {ohlc_path}")

        # Compute daily log returns
        daily_returns = self._compute_daily_returns(ohlc_data)

        # Compute returns for each horizon
        returns = {}
        for horizon in horizons:
            returns[horizon] = self._compute_horizon_return(ohlc_data, horizon)

        # Compute Sharpe ratios for each horizon
        sharpe_ratios = {}
        for horizon in horizons:
            if horizon <= len(daily_returns):
                sharpe_ratios[horizon] = self._compute_sharpe_ratio(
                    daily_returns[-horizon:], self.risk_free_rate_daily, annualize=True
                )
            else:
                sharpe_ratios[horizon] = None

        # Compute 21-day annualized volatility
        volatility = self._compute_volatility(daily_returns, window=21)

        result = ReferenceTickerMetrics(
            symbol=symbol,
            returns=returns,
            sharpe_ratios=sharpe_ratios,
            volatility_annualized=volatility,
            as_of=ohlc_data[-1]["date"] if ohlc_data else None,
        )

        self._cache[cache_key] = result
        return result

    def compute_beta(
        self,
        stock_symbol: str,
        stock_slug: str,
        benchmark_symbol: str,
        benchmark_slug: str,
        window_days: int = 252,
    ) -> tuple[float, float]:
        """
        Compute beta and R-squared for stock vs. benchmark.

        Args:
            stock_symbol, stock_slug: stock identifier
            benchmark_symbol, benchmark_slug: benchmark identifier
            window_days: lookback window (default 252 = 1 year)

        Returns:
            (beta, r_squared)

        Beta = cov(stock, benchmark) / var(benchmark)
        R-squared = coefficient of determination from linear regression
        """
        # Load daily returns for alignment
        stock_returns = self._load_daily_returns(stock_slug, window_days)
        benchmark_returns = self._load_daily_returns(benchmark_slug, window_days)

        if len(stock_returns) < window_days or len(benchmark_returns) < window_days:
            # Insufficient data
            return None, None

        # Align returns (ensure same dates)
        aligned_stock, aligned_benchmark = self._align_returns(stock_returns, benchmark_returns)

        if len(aligned_stock) < 20:  # Need at least 20 points for meaningful beta
            return None, None

        # Compute beta via covariance
        covariance = np.cov(aligned_stock, aligned_benchmark)[0, 1]
        benchmark_variance = np.var(aligned_benchmark, ddof=1)
        beta = covariance / benchmark_variance if benchmark_variance > 0 else None

        # Compute R-squared via linear regression
        correlation = np.corrcoef(aligned_stock, aligned_benchmark)[0, 1]
        r_squared = correlation**2 if not np.isnan(correlation) else None

        return beta, r_squared

    def clear_cache(self):
        """Clear internal cache. Call between portfolio runs."""
        self._cache.clear()

    # Private helper methods
    def _load_ohlc(self, path: Path) -> list[dict]:
        """Load OHLC data from JSON."""
        if not path.exists():
            return []

        try:
            with open(path) as f:
                data = json.load(f)
                return data.get("data", [])
        except Exception as e:
            raise ValueError(f"Failed to load OHLC data from {path}: {e}")

    def _compute_daily_returns(self, ohlc_data: list[dict]) -> np.ndarray:
        """Compute daily log returns from OHLC data."""
        if not ohlc_data:
            return np.array([])

        closes = np.array([float(bar["close"]) for bar in ohlc_data])
        if len(closes) < 2:
            return np.array([])

        # Filter out invalid prices
        valid_mask = closes > 0
        if not np.any(valid_mask[:-1] & valid_mask[1:]):
            return np.array([])

        # Compute log returns only where both prices are valid
        returns = []
        for i in range(1, len(closes)):
            if closes[i - 1] > 0 and closes[i] > 0:
                returns.append(np.log(closes[i] / closes[i - 1]))

        return np.array(returns)

    def _compute_horizon_return(self, ohlc_data: list[dict], horizon: int) -> float | None:
        """Compute total return over horizon (simple return, not annualized)."""
        if not ohlc_data or len(ohlc_data) < horizon + 1:
            return None

        start_price = float(ohlc_data[-horizon - 1]["close"])
        end_price = float(ohlc_data[-1]["close"])

        if start_price <= 0:
            return None

        return (end_price / start_price) - 1.0

    def _compute_sharpe_ratio(
        self, returns: np.ndarray, risk_free_rate_daily: float, annualize: bool = True
    ) -> float | None:
        """
        Compute Sharpe ratio.

        Sharpe = (mean(returns) - risk_free_rate) / std(returns)
        If annualize=True, scale by sqrt(252) for daily data.
        """
        if len(returns) < 2:
            return None

        excess_returns = returns - risk_free_rate_daily
        mean_excess = np.mean(excess_returns)
        std_excess = np.std(excess_returns, ddof=1)

        if std_excess == 0:
            return None

        sharpe = mean_excess / std_excess
        return sharpe * np.sqrt(252) if annualize else sharpe

    def _compute_volatility(self, returns: np.ndarray, window: int = 21) -> float | None:
        """Compute annualized volatility over trailing window."""
        if len(returns) < window:
            return None

        recent_returns = returns[-window:]
        return np.std(recent_returns, ddof=1) * np.sqrt(252)

    def _load_daily_returns(self, slug: str, window_days: int) -> np.ndarray:
        """Load and compute daily returns for alignment."""
        ohlc_path = self.stock_db_root / "tickers" / slug / "primary" / "ohlc_daily.json"
        ohlc_data = self._load_ohlc(ohlc_path)

        if not ohlc_data:
            return np.array([])

        # Get the last window_days + 1 data points (need +1 for returns calculation)
        relevant_data = (
            ohlc_data[-(window_days + 1) :] if len(ohlc_data) > window_days else ohlc_data
        )
        return self._compute_daily_returns(relevant_data)

    def _align_returns(
        self, returns1: np.ndarray, returns2: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Align two return series by date (assumes both have date metadata)."""
        # For now, assume both series cover same trading days and just truncate to min length
        min_len = min(len(returns1), len(returns2))
        if min_len == 0:
            return np.array([]), np.array([])

        return returns1[-min_len:], returns2[-min_len:]
