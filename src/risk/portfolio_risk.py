"""
Portfolio-Level Risk Manager
===============================
Aggregates risk across all active pairs and enforces portfolio constraints.

FIXED: Proper return calculation using price-based spread returns
scaled by capital allocation weights.

Correct P&L for a pair trade (Y-X with hedge ratio beta):
    Dollar P&L = position_weight * capital * (ret_Y - beta * ret_X)
    Portfolio return = sum of all pair dollar P&Ls / total_capital
"""

import numpy as np
import pandas as pd
from pathlib import Path

from src.risk.position_sizer import PositionSizer
from src.risk.var_calculator import VaRCalculator
from src.risk.drawdown_monitor import DrawdownMonitor


class PortfolioRiskManager:
    """
    Manages risk across the entire portfolio of pair trades.

    Parameters
    ----------
    total_capital : float
        Total portfolio capital.
    max_portfolio_var_pct : float
        Maximum allowable daily 95% VaR as fraction of capital.
        Default: 0.02 (2%).
    """

    def __init__(
        self,
        total_capital: float = 1_000_000,
        max_portfolio_var_pct: float = 0.02
    ):
        self.total_capital = total_capital
        self.max_portfolio_var_pct = max_portfolio_var_pct

        # Sub-components
        self.sizer = PositionSizer(total_capital=total_capital)
        self.var_calc = VaRCalculator()
        self.dd_monitor = DrawdownMonitor()

        # Portfolio state
        self.portfolio_returns = None
        self.portfolio_equity = None
        self.risk_metrics = None

    def compute_portfolio_returns(
        self,
        pair_signals: dict,
        prices_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Compute daily portfolio returns from all pair signals.

        Correct calculation:
            For each pair Y-X:
                spread_return_t = ret_Y_t - beta_t * ret_X_t
                pair_pnl_t = position_{t-1} * spread_return_t * weight
            
            portfolio_return_t = sum(pair_pnl_t for all pairs)

        Weight per pair = 1 / n_total_pairs (equal weight)
        This ensures total portfolio exposure is bounded.

        Parameters
        ----------
        pair_signals : dict
            pair_name -> signals DataFrame (from signal pipeline)
        prices_df : pd.DataFrame
            Price data for all assets

        Returns
        -------
        pd.DataFrame
            Portfolio-level daily returns and equity curve
        """
        n_pairs = len(pair_signals)
        if n_pairs == 0:
            raise ValueError("No pair signals provided!")

        # Weight per pair: equal weight, capped so total <= 100%
        # With 14 pairs, each gets ~7% of capital
        weight_per_pair = min(1.0 / n_pairs, 0.15)

        pair_returns = {}

        for pair_name, signals_df in pair_signals.items():
            if 'position' not in signals_df.columns:
                continue

            # Parse pair assets
            assets = pair_name.split('-')
            if len(assets) != 2:
                continue
            asset_y, asset_x = assets

            if asset_y not in prices_df.columns or asset_x not in prices_df.columns:
                continue

            try:
                # Get daily PRICE returns for both assets
                ret_y = prices_df[asset_y].pct_change()
                ret_x = prices_df[asset_x].pct_change()

                # Get hedge ratio (beta) from signals
                valid = signals_df.dropna(subset=['zscore']).copy()

                if 'beta' in valid.columns:
                    beta = valid['beta']
                else:
                    # Fallback: estimate beta from price ratio
                    beta = pd.Series(1.0, index=valid.index)

                # Align all series to common dates
                common = valid.index.intersection(ret_y.dropna().index).intersection(ret_x.dropna().index)
                if len(common) < 100:
                    continue

                ret_y_aligned = ret_y.reindex(common).fillna(0)
                ret_x_aligned = ret_x.reindex(common).fillna(0)
                beta_aligned = beta.reindex(common).ffill().fillna(1.0)
                position = valid['position'].reindex(common).fillna(0)

                # Spread return: what you earn from the pair trade
                spread_return = ret_y_aligned - beta_aligned * ret_x_aligned

                # Clip extreme spread returns (data errors or splits)
                spread_return = spread_return.clip(-0.10, 0.10)

                # Position-weighted return:
                # Use PREVIOUS day's position (can't trade on today's info)
                # Multiply by weight_per_pair to get portfolio contribution
                pair_contribution = position.shift(1).fillna(0) * spread_return * weight_per_pair

                pair_returns[pair_name] = pair_contribution

            except Exception as e:
                print(f"   Warning: {pair_name} return calc failed: {str(e)[:60]}")
                continue

        if not pair_returns:
            raise ValueError("No valid pair returns computed!")

        # Build portfolio DataFrame
        returns_df = pd.DataFrame(pair_returns)
        returns_df = returns_df.fillna(0)

        # Portfolio return = sum of all pair contributions
        portfolio_return = returns_df.sum(axis=1)

        # Sanity clip: no single day should exceed +-10%
        portfolio_return = portfolio_return.clip(-0.10, 0.10)

        result = pd.DataFrame(index=returns_df.index)
        result['portfolio_return'] = portfolio_return
        result['cumulative_return'] = (1 + portfolio_return).cumprod() - 1
        result['equity'] = self.total_capital * (1 + result['cumulative_return'])
        result['n_active_pairs'] = (returns_df != 0).sum(axis=1)

        # Add individual pair returns for analysis
        for pair_name in pair_returns:
            result[f'return_{pair_name}'] = returns_df[pair_name]

        self.portfolio_returns = result
        self.portfolio_equity = result['equity']

        return result

    def compute_risk_report(self) -> pd.DataFrame:
        """
        Generate comprehensive risk report combining all risk metrics.

        Returns
        -------
        pd.DataFrame
            Daily risk metrics: VaR, CVaR, drawdown, circuit breaker status
        """
        if self.portfolio_returns is None:
            raise ValueError("No portfolio returns. Call compute_portfolio_returns() first.")

        returns = self.portfolio_returns['portfolio_return']
        equity = self.portfolio_returns['equity']

        # VaR metrics
        var_df = self.var_calc.compute_all_metrics(returns, self.total_capital)

        # Drawdown metrics
        dd_df = self.dd_monitor.compute_drawdowns(equity)

        # Combine
        report = pd.DataFrame(index=returns.index)
        report['daily_return'] = returns
        report['equity'] = equity
        report['drawdown_pct'] = dd_df['drawdown_pct']
        report['drawdown_duration'] = dd_df['drawdown_duration']
        report['circuit_breaker'] = dd_df['circuit_breaker_level']
        report['position_scalar'] = dd_df['position_scalar']
        report['n_active_pairs'] = self.portfolio_returns['n_active_pairs']

        # VaR columns
        for col in var_df.columns:
            if col != 'daily_return':
                report[col] = var_df[col]

        # Rolling Sharpe ratio (annualized)
        rolling_mean = returns.rolling(252, min_periods=63).mean()
        rolling_std = returns.rolling(252, min_periods=63).std()
        report['rolling_sharpe'] = (rolling_mean / rolling_std.replace(0, np.nan)) * np.sqrt(252)

        self.risk_metrics = report
        return report

    def get_risk_summary(self) -> dict:
        """Get a comprehensive risk summary."""
        if self.risk_metrics is None:
            raise ValueError("No risk metrics. Call compute_risk_report() first.")

        returns = self.risk_metrics['daily_return'].dropna()
        equity = self.risk_metrics['equity']

        # Remove zero-return days for cleaner stats
        active_returns = returns[returns != 0]

        # Performance metrics
        total_return = (equity.iloc[-1] / equity.iloc[0]) - 1
        n_years = len(returns) / 252
        ann_return = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0
        ann_vol = returns.std() * np.sqrt(252)
        sharpe = ann_return / ann_vol if ann_vol > 0 else 0

        # Drawdown
        dd_summary = self.dd_monitor.get_drawdown_summary()

        # VaR snapshot
        var_snapshot = self.var_calc.get_current_risk_snapshot(
            returns, self.total_capital
        )

        # Win rate
        winning_days = (active_returns > 0).sum()
        total_active = len(active_returns)
        win_rate = winning_days / total_active * 100 if total_active > 0 else 0

        # Profit factor
        gross_profit = active_returns[active_returns > 0].sum()
        gross_loss = abs(active_returns[active_returns < 0].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf

        max_dd = dd_summary['max_drawdown_pct']

        return {
            'total_return_pct': total_return * 100,
            'annualized_return_pct': ann_return * 100,
            'annualized_volatility_pct': ann_vol * 100,
            'sharpe_ratio': sharpe,
            'max_drawdown_pct': max_dd,
            'max_drawdown_duration': dd_summary['max_drawdown_duration'],
            'calmar_ratio': ann_return / abs(max_dd / 100) if max_dd != 0 else 0,
            'win_rate_pct': win_rate,
            'profit_factor': profit_factor,
            'var_snapshot': var_snapshot,
            'circuit_breakers': dd_summary['circuit_breaker_triggers'],
            'positive_days_pct': (returns > 0).mean() * 100,
            'best_day_pct': returns.max() * 100,
            'worst_day_pct': returns.min() * 100,
            'avg_daily_return_bps': returns.mean() * 10000,
            'n_trading_days': len(returns),
            'n_active_days': total_active,
        }

    def save_risk_report(self, output_dir: str = "data/results") -> str:
        """Save risk report to CSV."""
        if self.risk_metrics is None:
            raise ValueError("No risk metrics computed.")

        path = Path(output_dir) / "risk_report.csv"
        self.risk_metrics.to_csv(path)
        print(f"Saved risk report to {path} ({len(self.risk_metrics)} rows)")
        return str(path)


# ============================================================
# Standalone test
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("PORTFOLIO RISK MANAGER TEST")
    print("=" * 60)

    prices = pd.read_csv("data/processed/close_prices.csv", index_col=0, parse_dates=True)

    signals_dir = Path("data/results/signals")
    signal_files = list(signals_dir.glob("signals_*.csv"))

    if not signal_files:
        print("No signal files found. Run Step 4 first.")
        exit(1)

    pair_signals = {}
    for f in signal_files:
        pair_name = f.stem.replace("signals_", "")
        pair_signals[pair_name] = pd.read_csv(f, index_col=0, parse_dates=True)

    print(f"\nLoaded {len(pair_signals)} pairs")

    manager = PortfolioRiskManager(total_capital=1_000_000)

    portfolio_df = manager.compute_portfolio_returns(pair_signals, prices)
    report = manager.compute_risk_report()
    summary = manager.get_risk_summary()

    print(f"\nPortfolio Risk Summary:")
    print(f"   Total return: {summary['total_return_pct']:.2f}%")
    print(f"   Annualized return: {summary['annualized_return_pct']:.2f}%")
    print(f"   Annualized vol: {summary['annualized_volatility_pct']:.2f}%")
    print(f"   Sharpe ratio: {summary['sharpe_ratio']:.3f}")
    print(f"   Max drawdown: {summary['max_drawdown_pct']:.2f}%")
    print(f"   Win rate: {summary['win_rate_pct']:.1f}%")
    print(f"   Profit factor: {summary['profit_factor']:.2f}")

    manager.save_risk_report()
    print(f"\nPortfolio risk test complete!")