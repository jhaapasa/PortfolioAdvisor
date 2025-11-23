"""Market comparison nodes for benchmark analysis and report generation."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from portfolio_advisor.config import MarketComparisonSettings
from portfolio_advisor.graphs.stocks import update_instrument
from portfolio_advisor.models.market import (
    MarketContext,
    PortfolioMarketMetrics,
    StockMarketComparison,
)
from portfolio_advisor.stocks.analysis import MarketMetricsService
from portfolio_advisor.utils.slug import instrument_id_to_slug, slugify

logger = logging.getLogger(__name__)


def ensure_reference_fresh_node(state: dict[str, Any]) -> dict[str, Any]:
    """Ensure all reference tickers are up-to-date.

    This node runs early in the pipeline to update reference ticker data
    (SPY, QQQ, IWM, etc.) using the same mechanisms as portfolio stocks.
    """
    settings = state["settings"]
    market_context = state.get("market_context", MarketContext())

    # Get reference tickers from config
    market_settings = MarketComparisonSettings()
    reference_symbols = market_settings.reference_symbols

    logger.info("Ensuring %d reference tickers are fresh", len(reference_symbols))

    # Update each reference ticker
    for symbol in reference_symbols:
        try:
            logger.debug("Updating reference ticker: %s", symbol)
            # Create instrument dict for the reference ticker
            # Assuming ETFs for reference tickers
            instrument = {"instrument_id": f"cid:etf:us:{symbol.lower()}", "primary_ticker": symbol}
            update_instrument(settings, instrument)
        except Exception as e:
            logger.warning("Failed to update reference ticker %s: %s", symbol, e)
            continue

    # Don't modify market_context here, just ensure data is fresh
    return {"market_context": market_context}


def compute_reference_metrics_node(state: dict[str, Any]) -> dict[str, Any]:
    """Compute metrics for all reference tickers.

    This node computes returns, Sharpe ratios, and volatility for all
    configured reference tickers and stores them in the market context.
    """
    settings = state["settings"]
    market_context = state.get("market_context", MarketContext())

    # Get configuration
    market_settings = MarketComparisonSettings()
    horizons = market_settings.time_horizons_days
    risk_free_rate_daily = market_settings.risk_free_rate_daily

    # Get stock database root
    stock_db_root = Path(settings.output_dir) / "stocks"

    # Initialize service
    service = MarketMetricsService(stock_db_root, risk_free_rate_daily)

    logger.info(
        "Computing metrics for %d reference tickers", len(market_settings.reference_tickers)
    )

    # Compute metrics for each reference ticker
    for ticker_config in market_settings.reference_tickers:
        symbol = ticker_config.symbol
        # Generate slug from instrument_id
        instrument_id = f"cid:etf:us:{symbol.lower()}"
        slug = instrument_id_to_slug(instrument_id)

        try:
            logger.debug("Computing metrics for reference ticker: %s", symbol)
            metrics = service.compute_metrics_for_symbol(symbol, slug, horizons)
            market_context.reference_metrics[symbol] = metrics
        except Exception as e:
            logger.warning("Failed to compute metrics for %s: %s", symbol, e)
            continue

    logger.info("Computed metrics for %d reference tickers", len(market_context.reference_metrics))

    # Clear service cache for next run
    service.clear_cache()

    return {"market_context": market_context}


def compute_stock_market_comparisons_node(state: dict[str, Any]) -> dict[str, Any]:
    """Compute market comparisons for all portfolio stocks.

    For each portfolio stock, this node computes:
    - Beta coefficients vs. default benchmarks (SPY, QQQ, IWM)
    - Sharpe ratios across multiple horizons
    - Returns and volatility

    Results are written to analysis/market_comparison.json for each stock.
    """
    settings = state["settings"]
    market_context = state.get("market_context", MarketContext())
    instruments = state.get("instruments", [])

    if not market_context.is_ready_for_stock_comparisons():
        logger.warning("Reference metrics not ready, skipping stock comparisons")
        return {"market_context": market_context}

    # Get configuration
    market_settings = MarketComparisonSettings()
    horizons = market_settings.time_horizons_days
    risk_free_rate_daily = market_settings.risk_free_rate_daily
    default_benchmarks = market_settings.default_benchmarks

    # Get stock database root
    stock_db_root = Path(settings.output_dir) / "stocks"

    # Initialize service
    service = MarketMetricsService(stock_db_root, risk_free_rate_daily)

    logger.info("Computing market comparisons for %d portfolio stocks", len(instruments))

    # Process each portfolio stock
    for instrument in instruments:
        ticker = instrument.get("primary_ticker")
        if not ticker:
            continue

        # Derive slug from instrument_id
        instrument_id = instrument.get("instrument_id", "")
        slug = slugify(instrument_id)

        try:
            logger.debug("Computing comparisons for %s", ticker)

            # Compute stock metrics
            stock_metrics = service.compute_metrics_for_symbol(ticker, slug, horizons)

            # Compute beta vs. each benchmark
            betas = {}
            for benchmark_symbol in default_benchmarks:
                benchmark_id = f"cid:etf:us:{benchmark_symbol.lower()}"
                benchmark_slug = instrument_id_to_slug(benchmark_id)
                try:
                    beta, r_squared = service.compute_beta(
                        ticker, slug, benchmark_symbol, benchmark_slug
                    )
                    if beta is not None:
                        betas[benchmark_symbol] = {
                            "value": beta,
                            "window_days": 252,
                            "r_squared": r_squared,
                        }
                except Exception as e:
                    logger.warning(
                        "Failed to compute beta for %s vs %s: %s", ticker, benchmark_symbol, e
                    )

            # Create comparison object
            comparison = StockMarketComparison(
                ticker=ticker,
                slug=slug,
                betas={k: v["value"] for k, v in betas.items()},  # Simplified for state
                sharpe_ratios=stock_metrics.sharpe_ratios,
                returns=stock_metrics.returns,
                as_of=stock_metrics.as_of,
            )

            # Store in state
            market_context.stock_comparisons[ticker] = comparison

            # Write to disk
            analysis_dir = stock_db_root / "tickers" / slug / "analysis"
            analysis_dir.mkdir(parents=True, exist_ok=True)

            comparison_data = {
                "ticker": ticker,
                "slug": slug,
                "as_of": stock_metrics.as_of,
                "betas": betas,  # Full beta data with r_squared
                "sharpe_ratios": stock_metrics.sharpe_ratios,
                "returns": stock_metrics.returns,
                "volatility_annualized": stock_metrics.volatility_annualized,
                "depends_on": ["primary.ohlc_daily"],
                "generated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            }

            with open(analysis_dir / "market_comparison.json", "w") as f:
                json.dump(comparison_data, f, indent=2)

        except Exception as e:
            logger.warning("Failed to compute comparisons for %s: %s", ticker, e)
            continue

    logger.info("Computed comparisons for %d stocks", len(market_context.stock_comparisons))

    # Clear service cache
    service.clear_cache()

    return {"market_context": market_context}


def compute_portfolio_market_metrics_node(state: dict[str, Any]) -> dict[str, Any]:
    """Compute portfolio-level market metrics.

    Aggregates per-stock comparisons to compute:
    - Portfolio-weighted average beta
    - Portfolio Sharpe ratio
    - Stocks outperforming benchmarks
    - Top contributors to outperformance
    """
    market_context = state.get("market_context", MarketContext())
    holdings = state.get("resolved_holdings", [])

    if not market_context.is_ready_for_portfolio_metrics():
        logger.warning("Stock comparisons not ready, skipping portfolio metrics")
        return {"market_context": market_context}

    # Get configuration
    market_settings = MarketComparisonSettings()
    default_benchmarks = market_settings.default_benchmarks

    # Build position weights from holdings
    position_weights = {}
    total_value = 0.0
    for holding in holdings:
        ticker = holding.get("primary_ticker")
        value = holding.get("total_value", 0)
        if ticker and value:
            position_weights[ticker] = value
            total_value += value

    # Normalize weights
    if total_value > 0:
        for ticker in position_weights:
            position_weights[ticker] /= total_value

    # Calculate portfolio-weighted average betas
    average_betas = {}
    for benchmark in default_benchmarks:
        weighted_beta = 0.0
        weight_sum = 0.0

        for ticker, comparison in market_context.stock_comparisons.items():
            if ticker in position_weights and benchmark in comparison.betas:
                beta = comparison.betas[benchmark]
                if beta is not None:
                    weight = position_weights[ticker]
                    weighted_beta += beta * weight
                    weight_sum += weight

        if weight_sum > 0:
            average_betas[benchmark] = weighted_beta / weight_sum

    # Calculate average stock Sharpe (1yr)
    sharpe_values = []
    for comparison in market_context.stock_comparisons.values():
        if 252 in comparison.sharpe_ratios and comparison.sharpe_ratios[252] is not None:
            sharpe_values.append(comparison.sharpe_ratios[252])

    average_stock_sharpe = sum(sharpe_values) / len(sharpe_values) if sharpe_values else 0.0

    # Count stocks outperforming benchmarks (based on 1yr Sharpe)
    stocks_outperforming = {}
    spy_sharpe = None

    if "SPY" in market_context.reference_metrics:
        spy_metrics = market_context.reference_metrics["SPY"]
        if 252 in spy_metrics.sharpe_ratios:
            spy_sharpe = spy_metrics.sharpe_ratios[252]

    for benchmark in default_benchmarks:
        count = 0
        if benchmark in market_context.reference_metrics:
            benchmark_metrics = market_context.reference_metrics[benchmark]
            if 252 in benchmark_metrics.sharpe_ratios:
                benchmark_sharpe = benchmark_metrics.sharpe_ratios[252]
                if benchmark_sharpe is not None:
                    for comparison in market_context.stock_comparisons.values():
                        if 252 in comparison.sharpe_ratios:
                            stock_sharpe = comparison.sharpe_ratios[252]
                            if stock_sharpe is not None and stock_sharpe > benchmark_sharpe:
                                count += 1
        stocks_outperforming[benchmark] = count

    # Find top contributors (vs. SPY based on excess Sharpe)
    contributors = []
    if spy_sharpe is not None:
        for ticker, comparison in market_context.stock_comparisons.items():
            if 252 in comparison.sharpe_ratios and comparison.sharpe_ratios[252] is not None:
                stock_sharpe = comparison.sharpe_ratios[252]
                excess_sharpe = stock_sharpe - spy_sharpe

                # Get stock return
                excess_return = None
                if 252 in comparison.returns and comparison.returns[252] is not None:
                    spy_return = market_context.reference_metrics["SPY"].returns.get(252, 0)
                    if spy_return is not None:
                        excess_return = comparison.returns[252] - spy_return

                contributors.append(
                    {
                        "ticker": ticker,
                        "excess_sharpe": excess_sharpe,
                        "excess_return": excess_return,
                    }
                )

    # Sort by excess Sharpe and take top 5
    contributors.sort(key=lambda x: x["excess_sharpe"], reverse=True)
    top_contributors = contributors[:5]

    # TODO: Compute portfolio-level Sharpe (requires aggregate portfolio returns)
    # For now, use average of stock Sharpes as approximation
    portfolio_sharpe = average_stock_sharpe

    # Create portfolio metrics
    portfolio_metrics = PortfolioMarketMetrics(
        average_beta_vs_benchmarks=average_betas,
        portfolio_sharpe=portfolio_sharpe,
        average_stock_sharpe=average_stock_sharpe,
        stocks_outperforming=stocks_outperforming,
        total_stocks=len(market_context.stock_comparisons),
        top_contributors=top_contributors,
        as_of=datetime.now(UTC).strftime("%Y-%m-%d"),
    )

    market_context.portfolio_metrics = portfolio_metrics

    logger.info(
        "Computed portfolio market metrics for %d stocks", len(market_context.stock_comparisons)
    )

    return {"market_context": market_context}


def _build_market_assessment(
    market_context: MarketContext, market_settings: MarketComparisonSettings
) -> list[str]:
    """Build the Market Assessment section with structured analysis.

    Generates:
    1. Performance Summary (1yr) - structured templated content
    2. Risk-Adjusted Returns (1yr Sharpe) - structured templated content
    3. Volatility Environment - structured templated content
    4. Market Themes and Context - LLM-generated narrative (future enhancement)
    """
    lines = []

    # Get reference metrics
    ref_metrics = market_context.reference_metrics
    if not ref_metrics:
        return [
            "### Market Assessment",
            "",
            "*Insufficient reference data available for market assessment.*",
        ]

    # Helper to categorize tickers by type
    equity_tickers = ["SPY", "QQQ", "IWM", "VTI"]
    intl_tickers = ["EFA", "EEM"]
    bond_tickers = ["AGG", "TLT"]

    # 1. Performance Summary (1yr)
    lines.extend(["### Performance Summary (1yr)", ""])

    # Equity performance
    equity_perf = []
    for symbol in equity_tickers:
        if symbol in ref_metrics:
            metrics = ref_metrics[symbol]
            ret_1yr = metrics.returns.get(252)
            if ret_1yr is not None:
                ticker_config = next(
                    (t for t in market_settings.reference_tickers if t.symbol == symbol), None
                )
                name = ticker_config.name if ticker_config else symbol
                equity_perf.append((name, symbol, ret_1yr))

    if equity_perf:
        equity_perf.sort(key=lambda x: x[2], reverse=True)
        perf_str = ", ".join([f"{name} ({ret:+.1%})" for name, sym, ret in equity_perf[:3]])
        lines.append(f"- **Equities**: {perf_str}")

    # International performance
    intl_perf = []
    for symbol in intl_tickers:
        if symbol in ref_metrics:
            metrics = ref_metrics[symbol]
            ret_1yr = metrics.returns.get(252)
            if ret_1yr is not None:
                ticker_config = next(
                    (t for t in market_settings.reference_tickers if t.symbol == symbol), None
                )
                name = ticker_config.name if ticker_config else symbol
                intl_perf.append((name, symbol, ret_1yr))

    if intl_perf:
        perf_str = ", ".join([f"{name} ({ret:+.1%})" for name, sym, ret in intl_perf])
        lines.append(f"- **International**: {perf_str}")

    # Fixed income performance
    bond_perf = []
    for symbol in bond_tickers:
        if symbol in ref_metrics:
            metrics = ref_metrics[symbol]
            ret_1yr = metrics.returns.get(252)
            if ret_1yr is not None:
                ticker_config = next(
                    (t for t in market_settings.reference_tickers if t.symbol == symbol), None
                )
                name = ticker_config.name if ticker_config else symbol
                bond_perf.append((name, symbol, ret_1yr))

    if bond_perf:
        perf_str = ", ".join([f"{name} ({ret:+.1%})" for name, sym, ret in bond_perf])
        lines.append(f"- **Fixed Income**: {perf_str}")

    # 2. Risk-Adjusted Returns (1yr Sharpe)
    lines.extend(["", "### Risk-Adjusted Returns (1yr Sharpe)", ""])

    # Collect all Sharpe ratios
    all_sharpes = []
    for symbol, metrics in ref_metrics.items():
        sharpe_1yr = metrics.sharpe_ratios.get(252)
        if sharpe_1yr is not None:
            ticker_config = next(
                (t for t in market_settings.reference_tickers if t.symbol == symbol), None
            )
            name = ticker_config.name if ticker_config else symbol
            all_sharpes.append((name, symbol, sharpe_1yr))

    if all_sharpes:
        all_sharpes.sort(key=lambda x: x[2], reverse=True)

        # Top performers
        if len(all_sharpes) >= 3:
            top_3 = all_sharpes[:3]
            top_str = ", ".join([f"{name} ({sharpe:.2f})" for name, sym, sharpe in top_3])
            lines.append(f"- **Best**: {top_str}")

        # Middle performers
        if len(all_sharpes) >= 5:
            mid_start = 3
            mid_end = min(5, len(all_sharpes))
            mid = all_sharpes[mid_start:mid_end]
            if mid:
                mid_str = ", ".join([f"{name} ({sharpe:.2f})" for name, sym, sharpe in mid])
                lines.append(f"- **Moderate**: {mid_str}")

        # Lower performers
        if len(all_sharpes) > 5:
            low = all_sharpes[5:]
            if low:
                low_str = ", ".join([f"{name} ({sharpe:.2f})" for name, sym, sharpe in low])
                lines.append(f"- **Lower**: {low_str}")

    # 3. Volatility Environment
    lines.extend(["", "### Volatility Environment", ""])

    # Collect volatilities
    all_vols = []
    for symbol, metrics in ref_metrics.items():
        vol = metrics.volatility_annualized
        if vol is not None:
            ticker_config = next(
                (t for t in market_settings.reference_tickers if t.symbol == symbol), None
            )
            name = ticker_config.name if ticker_config else symbol
            # Categorize by asset class
            if symbol in equity_tickers:
                asset_class = "equity"
            elif symbol in intl_tickers:
                asset_class = "intl_equity"
            elif symbol in bond_tickers:
                asset_class = "bond"
            else:
                asset_class = "other"
            all_vols.append((name, symbol, vol, asset_class))

    if all_vols:
        # Equity volatility
        equity_vols = [v for v in all_vols if v[3] in ["equity", "intl_equity"]]
        if equity_vols:
            equity_vols.sort(key=lambda x: x[2], reverse=True)
            highest = equity_vols[0]
            lowest = equity_vols[-1]
            lines.append(
                f"- **Equity volatility**: {highest[0]} highest ({highest[2]:.1%}), "
                f"{lowest[0]} lowest ({lowest[2]:.1%})"
            )

        # Fixed income volatility
        bond_vols = [v for v in all_vols if v[3] == "bond"]
        if bond_vols:
            bond_str = ", ".join([f"{name} ({vol:.1%})" for name, sym, vol, ac in bond_vols])
            lines.append(f"- **Fixed income volatility**: {bond_str}")

    # 4. Market Themes and Context (LLM-generated - future enhancement)
    lines.extend(
        [
            "",
            "### Market Themes and Context",
            "*[LLM-Generated Analysis - To be implemented in future phase]*",
        ]
    )

    return lines


def generate_market_overview_report_node(state: dict[str, Any]) -> dict[str, Any]:
    """Generate standalone market overview report.

    Creates a market overview report with:
    - Market performance summary table
    - Market risk metrics table
    - Market assessment and themes
    """
    settings = state["settings"]
    market_context = state.get("market_context", MarketContext())

    if not market_context.reference_metrics:
        logger.warning("No reference metrics available, skipping market overview report")
        return {"market_context": market_context}

    # Get configuration
    market_settings = MarketComparisonSettings()

    # Prepare report data
    now = datetime.now(UTC)
    report_date = now.strftime("%Y-%m-%d")
    report_timestamp = now.strftime("%Y-%m-%d %H:%M UTC")

    # Get last trading day from any reference ticker
    last_trading_day = None
    for metrics in market_context.reference_metrics.values():
        if metrics.as_of:
            last_trading_day = metrics.as_of
            break

    # Build performance summary table
    performance_lines = []
    performance_lines.append("| Index | Symbol | 3mo | 6mo | 1yr | 2yr |")
    performance_lines.append("|-------|--------|-----|-----|-----|-----|")

    for ticker_config in market_settings.reference_tickers:
        symbol = ticker_config.symbol
        name = ticker_config.name

        if symbol not in market_context.reference_metrics:
            continue

        metrics = market_context.reference_metrics[symbol]

        returns_3mo = metrics.returns.get(63)
        returns_6mo = metrics.returns.get(126)
        returns_1yr = metrics.returns.get(252)
        returns_2yr = metrics.returns.get(504)

        def fmt_return(r):
            if r is None:
                return "N/A"
            return f"{r:+.1%}"

        performance_lines.append(
            f"| {name} | {symbol} | {fmt_return(returns_3mo)} | "
            f"{fmt_return(returns_6mo)} | {fmt_return(returns_1yr)} | "
            f"{fmt_return(returns_2yr)} |"
        )

    # Build risk metrics table
    risk_lines = []
    risk_lines.append("| Index | Symbol | Volatility (21d) | Sharpe (1yr) | Sharpe (2yr) |")
    risk_lines.append("|-------|--------|------------------|--------------|--------------|")

    for ticker_config in market_settings.reference_tickers:
        symbol = ticker_config.symbol
        name = ticker_config.name

        if symbol not in market_context.reference_metrics:
            continue

        metrics = market_context.reference_metrics[symbol]

        vol = metrics.volatility_annualized
        sharpe_1yr = metrics.sharpe_ratios.get(252)
        sharpe_2yr = metrics.sharpe_ratios.get(504)

        def fmt_vol(v):
            if v is None:
                return "N/A"
            return f"{v:.1%}"

        def fmt_sharpe(s):
            if s is None:
                return "N/A"
            return f"{s:.2f}"

        risk_lines.append(
            f"| {name} | {symbol} | {fmt_vol(vol)} | "
            f"{fmt_sharpe(sharpe_1yr)} | {fmt_sharpe(sharpe_2yr)} |"
        )

    # Build market assessment section
    assessment_lines = _build_market_assessment(market_context, market_settings)

    # Generate report
    report_lines = [
        "# Market Overview Report",
        "",
        f"Generated: {report_timestamp}",
        f"As of: {last_trading_day or 'Unknown'}",
        "",
        "## Market Performance Summary",
        "",
    ]
    report_lines.extend(performance_lines)
    report_lines.extend(
        [
            "",
            "## Market Risk Metrics",
            "",
        ]
    )
    report_lines.extend(risk_lines)
    report_lines.extend(
        [
            "",
            "## Market Assessment",
            "",
        ]
    )
    report_lines.extend(assessment_lines)
    report_lines.extend(
        [
            "",
            "Notes:",
            "- All returns are total returns including dividends",
            f"- Sharpe ratios assume {market_settings.risk_free_rate_annual:.1%} "
            "annual risk-free rate",
            "- Volatility is annualized 21-day historical volatility",
            f"- Data as of {last_trading_day or 'last complete trading day'}",
        ]
    )

    report_content = "\n".join(report_lines)

    # Write report
    portfolio_dir = Path(settings.portfolio_dir)
    report_path = portfolio_dir / f"market_overview_{report_date}.md"

    with open(report_path, "w") as f:
        f.write(report_content)

    # Also write JSON data
    json_data = {
        "generated_at": now.isoformat() + "Z",
        "as_of": last_trading_day,
        "reference_tickers": {
            symbol: {
                "returns": metrics.returns,
                "sharpe_ratios": metrics.sharpe_ratios,
                "volatility": metrics.volatility_annualized,
            }
            for symbol, metrics in market_context.reference_metrics.items()
        },
    }

    json_path = portfolio_dir / f"market_overview_{report_date}.json"
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2)

    market_context.market_overview_generated = str(report_path)

    logger.info("Generated market overview report: %s", report_path)

    return {"market_context": market_context}
