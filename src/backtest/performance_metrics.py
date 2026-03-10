"""
Performance Metrics Calculator
================================
Comprehensive performance analytics for backtested strategies.

Metrics computed:
    - Risk-adjusted: Sharpe, Sortino, Calmar, Information Ratio
    - Return: CAGR, total return, monthly/yearly returns
    - Risk: Max drawdown, VaR, volatility, downside deviation
    - Trade: Win rate, profit factor, avg win/loss, expectancy
    - Consistency: % positive months, longest win/loss streaks
"""

import numpy as np
import pandas as pd


class PerformanceAnalyzer:
    """
    Comprehensive performance analysis for a backtested strategy.

    Parameters
    ----------
    risk_free_rate : float
        Annual risk-free rate for Sharpe ratio calculation.
        Default: 0.04 (4% — approximate T-bill rate).
    """

    def __init__(self, risk_free_rate: float = 0.04):
        self.risk_free_rate = risk_free_rate
        self.daily_rf = (1 + risk_free_rate) ** (1/252) - 1

    def compute_all_metrics(
        self,
        equity_curve: pd.Series,
        daily_returns: pd.Series = None,
        trades: list = None
    ) -> dict:
        """
        Compute all performance metrics.

        Parameters
        ----------
        equity_curve : pd.Series
            Portfolio equity over time
        daily_returns : pd.Series, optional
            Daily returns. If None, computed from equity curve.
        trades : list, optional
            List of trade dicts with 'pnl' key.

        Returns
        -------
        dict
            Comprehensive performance metrics
        """
        if daily_returns is None:
            daily_returns = equity_curve.pct_change().fillna(0)

        metrics = {}

        # Return metrics
        metrics.update(self._return_metrics(equity_curve, daily_returns))

        # Risk metrics
        metrics.update(self._risk_metrics(equity_curve, daily_returns))

        # Risk-adjusted metrics
        metrics.update(self._risk_adjusted_metrics(daily_returns))

        # Consistency metrics
        metrics.update(self._consistency_metrics(daily_returns))

        # Trade metrics
        if trades:
            metrics.update(self._trade_metrics(trades))

        return metrics

    def _return_metrics(self, equity: pd.Series, returns: pd.Series) -> dict:
        """Core return metrics."""
        total_return = equity.iloc[-1] / equity.iloc[0] - 1
        n_years = len(returns) / 252
        cagr = (1 + total_return) ** (1/n_years) - 1 if n_years > 0 else 0

        return {
            'total_return_pct': total_return * 100,
            'cagr_pct': cagr * 100,
            'avg_daily_return_bps': returns.mean() * 10000,
            'median_daily_return_bps': returns.median() * 10000,
            'best_day_pct': returns.max() * 100,
            'worst_day_pct': returns.min() * 100,
            'final_equity': equity.iloc[-1],
        }

    def _risk_metrics(self, equity: pd.Series, returns: pd.Series) -> dict:
        """Risk metrics."""
        # Volatility
        ann_vol = returns.std() * np.sqrt(252)

        # Drawdown analysis
        peak = equity.cummax()
        drawdown = (equity - peak) / peak
        max_dd = drawdown.min()

        # Drawdown duration
        max_dd_duration = 0
        current_duration = 0
        for dd in drawdown:
            if dd < -1e-6:
                current_duration += 1
                max_dd_duration = max(max_dd_duration, current_duration)
            else:
                current_duration = 0

        # Average drawdown
        avg_dd = drawdown[drawdown < -1e-6].mean() if (drawdown < -1e-6).any() else 0

        # Downside deviation (only negative returns)
        downside = returns[returns < 0]
        downside_dev = downside.std() * np.sqrt(252) if len(downside) > 0 else 0

        # Skew and kurtosis
        skew = returns.skew()
        kurt = returns.kurtosis()

        # VaR
        var_95 = -np.percentile(returns.dropna(), 5)
        var_99 = -np.percentile(returns.dropna(), 1)

        return {
            'annualized_volatility_pct': ann_vol * 100,
            'max_drawdown_pct': max_dd * 100,
            'max_dd_duration_days': max_dd_duration,
            'avg_drawdown_pct': avg_dd * 100,
            'downside_deviation_pct': downside_dev * 100,
            'daily_var_95_pct': var_95 * 100,
            'daily_var_99_pct': var_99 * 100,
            'return_skewness': skew,
            'return_kurtosis': kurt,
        }

    def _risk_adjusted_metrics(self, returns: pd.Series) -> dict:
        """Risk-adjusted performance metrics."""
        excess_returns = returns - self.daily_rf
        ann_excess = excess_returns.mean() * 252
        ann_vol = returns.std() * np.sqrt(252)

        # Sharpe
        sharpe = ann_excess / ann_vol if ann_vol > 0 else 0

        # Sortino (using downside deviation)
        downside = returns[returns < self.daily_rf]
        downside_vol = downside.std() * np.sqrt(252) if len(downside) > 0 else ann_vol
        sortino = ann_excess / downside_vol if downside_vol > 0 else 0

        # Omega ratio
        threshold = self.daily_rf
        gains = returns[returns > threshold] - threshold
        losses = threshold - returns[returns <= threshold]
        omega = gains.sum() / losses.sum() if losses.sum() > 0 else float('inf')

        return {
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'omega_ratio': omega,
        }

    def _consistency_metrics(self, returns: pd.Series) -> dict:
        """How consistent is the strategy?"""
        # Monthly returns
        monthly = returns.resample('ME').sum()
        positive_months = (monthly > 0).sum()
        total_months = len(monthly)
        pct_positive_months = positive_months / total_months * 100 if total_months > 0 else 0

        # Yearly returns
        yearly = returns.resample('YE').sum()

        # Streaks
        is_positive = (returns > 0).astype(int)
        longest_win = self._longest_streak(is_positive, 1)
        longest_loss = self._longest_streak(is_positive, 0)

        return {
            'positive_days_pct': (returns > 0).mean() * 100,
            'positive_months_pct': pct_positive_months,
            'positive_years': (yearly > 0).sum(),
            'total_years': len(yearly),
            'longest_win_streak_days': longest_win,
            'longest_loss_streak_days': longest_loss,
            'best_month_pct': monthly.max() * 100,
            'worst_month_pct': monthly.min() * 100,
        }

    def _trade_metrics(self, trades: list) -> dict:
        """Trade-level analytics."""
        if not trades:
            return {}

        pnls = [t['pnl'] for t in trades]
        winners = [p for p in pnls if p > 0]
        losers = [p for p in pnls if p <= 0]

        win_rate = len(winners) / len(trades) * 100
        avg_win = np.mean(winners) if winners else 0
        avg_loss = np.mean(losers) if losers else 0

        gross_profit = sum(winners)
        gross_loss = abs(sum(losers))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        # Expectancy per trade
        expectancy = np.mean(pnls)

        # Payoff ratio
        payoff = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')

        # Holding periods
        holding_days = [t.get('holding_days', 0) for t in trades]

        return {
            'total_trades': len(trades),
            'win_rate_pct': win_rate,
            'avg_win_dollar': avg_win,
            'avg_loss_dollar': avg_loss,
            'largest_win': max(pnls),
            'largest_loss': min(pnls),
            'profit_factor': profit_factor,
            'payoff_ratio': payoff,
            'expectancy_per_trade': expectancy,
            'avg_holding_days': np.mean(holding_days) if holding_days else 0,
            'max_holding_days': max(holding_days) if holding_days else 0,
        }

    def _longest_streak(self, binary_series: pd.Series, value: int) -> int:
        """Find longest consecutive streak of a value."""
        max_streak = 0
        current = 0
        for v in binary_series:
            if v == value:
                current += 1
                max_streak = max(max_streak, current)
            else:
                current = 0
        return max_streak

    def generate_monthly_table(self, daily_returns: pd.Series) -> pd.DataFrame:
        """Generate monthly return heatmap table."""
        monthly = daily_returns.resample('ME').sum() * 100  # Convert to %

        table = pd.DataFrame()
        for date, ret in monthly.items():
            year = date.year
            month = date.month
            table.loc[year, month] = ret

        table.columns = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                         'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

        # Add yearly total
        yearly = daily_returns.resample('YE').sum() * 100
        table['Year Total'] = [yearly.get(pd.Timestamp(f'{y}-12-31'), np.nan)
                                for y in table.index]

        return table.round(2)