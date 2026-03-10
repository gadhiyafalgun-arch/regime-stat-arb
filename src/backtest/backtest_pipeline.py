"""
Backtesting Pipeline
======================
Orchestrates full walk-forward backtesting:
1. Load signals and price data
2. Run pair-level backtests with transaction costs
3. Aggregate into portfolio
4. Compute comprehensive performance metrics
5. Generate reports and plots
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

from src.backtest.backtest_engine import PortfolioBacktester
from src.backtest.performance_metrics import PerformanceAnalyzer


class BacktestPipeline:
    """
    End-to-end backtesting pipeline.

    Parameters
    ----------
    total_capital : float
        Starting capital.
    transaction_cost_bps : float
        Transaction cost in basis points.
    slippage_bps : float
        Slippage in basis points.
    output_dir : str
        Directory for output files.
    plot_dir : str
        Directory for plots.
    """

    def __init__(
        self,
        total_capital: float = 1_000_000,
        transaction_cost_bps: float = 10.0,
        slippage_bps: float = 5.0,
        output_dir: str = "data/results",
        plot_dir: str = "data/results/plots"
    ):
        self.total_capital = total_capital
        self.transaction_cost_bps = transaction_cost_bps
        self.slippage_bps = slippage_bps
        self.output_dir = Path(output_dir)
        self.plot_dir = Path(plot_dir)

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.plot_dir.mkdir(parents=True, exist_ok=True)

        # Components
        self.backtester = None
        self.analyzer = PerformanceAnalyzer(risk_free_rate=0.04)

        # Results
        self.portfolio_df = None
        self.metrics = None
        self.monthly_table = None

    def run(self, verbose: bool = True) -> dict:
        """Execute the full backtest pipeline."""
        print("=" * 70)
        print("🏗️  BACKTESTING PIPELINE")
        print("=" * 70)

        # Step 1: Load data
        prices, pair_signals = self._load_data(verbose)

        # Step 2: Run backtest
        self._run_backtest(pair_signals, prices, verbose)

        # Step 3: Performance analysis
        self._analyze_performance(verbose)

        # Step 4: Save results
        self._save_all(verbose)

        # Step 5: Generate plots
        self._plot_results(verbose)

        # Step 6: Print final report
        self._print_final_report()

        return self.metrics

    def _load_data(self, verbose: bool) -> tuple:
        """Load all required data."""
        if verbose:
            print(f"\n📥 Step 1: Loading data...")

        prices = pd.read_csv("data/processed/close_prices.csv",
                              index_col=0, parse_dates=True)

        signals_dir = Path("data/results/signals")
        signal_files = list(signals_dir.glob("signals_*.csv"))

        if not signal_files:
            raise FileNotFoundError("No signal files found. Run Step 4 first!")

        pair_signals = {}
        for f in signal_files:
            pair_name = f.stem.replace("signals_", "")
            pair_signals[pair_name] = pd.read_csv(f, index_col=0, parse_dates=True)

        if verbose:
            print(f"   Prices: {prices.shape[0]} days x {prices.shape[1]} assets")
            print(f"   Signal files: {len(pair_signals)} pairs")

        return prices, pair_signals

    def _run_backtest(self, pair_signals: dict, prices: pd.DataFrame, verbose: bool):
        """Run the backtest engine."""
        if verbose:
            print(f"\n⚙️  Step 2: Running backtest...")
            print(f"   Capital: ${self.total_capital:,.0f}")
            print(f"   Costs: {self.transaction_cost_bps} + {self.slippage_bps} bps")

        self.backtester = PortfolioBacktester(
            total_capital=self.total_capital,
            transaction_cost_bps=self.transaction_cost_bps,
            slippage_bps=self.slippage_bps
        )

        self.portfolio_df = self.backtester.run(
            pair_signals, prices, verbose=verbose
        )

    def _analyze_performance(self, verbose: bool):
        """Compute comprehensive performance metrics."""
        if verbose:
            print(f"\n📊 Step 3: Performance analysis...")

        self.metrics = self.analyzer.compute_all_metrics(
            equity_curve=self.portfolio_df['equity'],
            daily_returns=self.portfolio_df['daily_return'],
            trades=self.backtester.all_trades
        )

        # Add portfolio-level metrics from backtester
        portfolio_summary = self.backtester.get_portfolio_summary()
        self.metrics['total_transaction_costs'] = portfolio_summary.get('total_transaction_costs', 0)
        self.metrics['avg_active_pairs'] = portfolio_summary.get('avg_active_pairs', 0)
        self.metrics['trades_per_year'] = portfolio_summary.get('trades_per_year', 0)

        # Monthly returns table
        self.monthly_table = self.analyzer.generate_monthly_table(
            self.portfolio_df['daily_return']
        )

    def _save_all(self, verbose: bool):
        """Save all backtest results."""
        if verbose:
            print(f"\n💾 Step 4: Saving results...")

        # Portfolio equity curve
        equity_path = self.output_dir / "backtest_equity.csv"
        self.portfolio_df.to_csv(equity_path)
        if verbose:
            print(f"   Saved equity curve to {equity_path}")

        # Performance metrics
        metrics_path = self.output_dir / "backtest_metrics.txt"
        with open(metrics_path, 'w') as f:
            f.write("BACKTEST PERFORMANCE REPORT\n")
            f.write("=" * 60 + "\n\n")
            for key, val in sorted(self.metrics.items()):
                if isinstance(val, float):
                    f.write(f"{key:>40s}: {val:>12.4f}\n")
                else:
                    f.write(f"{key:>40s}: {val}\n")
        if verbose:
            print(f"   Saved metrics to {metrics_path}")

        # Monthly table
        monthly_path = self.output_dir / "backtest_monthly_returns.csv"
        self.monthly_table.to_csv(monthly_path)
        if verbose:
            print(f"   Saved monthly returns to {monthly_path}")

        # Pair comparison
        pair_comp = self.backtester.get_pair_comparison()
        pair_path = self.output_dir / "backtest_pair_comparison.csv"
        pair_comp.to_csv(pair_path, index=False)
        if verbose:
            print(f"   Saved pair comparison to {pair_path}")

        # Trade log
        if self.backtester.all_trades:
            trades_df = pd.DataFrame(self.backtester.all_trades)
            trades_path = self.output_dir / "backtest_trades.csv"
            trades_df.to_csv(trades_path, index=False)
            if verbose:
                print(f"   Saved {len(trades_df)} trades to {trades_path}")

    def _plot_results(self, verbose: bool):
        """Generate backtest plots."""
        if verbose:
            print(f"\n📊 Step 5: Generating plots...")

        try:
            self._plot_equity_curve()
            self._plot_monthly_heatmap()
            self._plot_pair_comparison()
            self._plot_drawdown_analysis()
            if verbose:
                print(f"   Saved plots to {self.plot_dir}/")
        except Exception as e:
            if verbose:
                print(f"   Plotting error (non-critical): {e}")

    def _plot_equity_curve(self):
        """Plot portfolio equity curve with drawdown."""
        fig, axes = plt.subplots(2, 1, figsize=(16, 10),
                                  gridspec_kw={'height_ratios': [3, 1]})

        equity = self.portfolio_df['equity']
        peak = equity.cummax()
        drawdown = (equity - peak) / peak * 100

        # Equity curve
        ax1 = axes[0]
        ax1.plot(equity.index, equity, color='steelblue', linewidth=1.2, label='Portfolio')
        ax1.axhline(y=self.total_capital, color='gray', linestyle='--', alpha=0.5,
                    label='Starting Capital')
        ax1.fill_between(equity.index, self.total_capital, equity,
                        where=equity >= self.total_capital, alpha=0.15, color='green')
        ax1.fill_between(equity.index, self.total_capital, equity,
                        where=equity < self.total_capital, alpha=0.15, color='red')
        ax1.set_title('Portfolio Equity Curve — Regime-Aware Stat Arb', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Equity ($)')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

        # Drawdown
        ax2 = axes[1]
        ax2.fill_between(drawdown.index, drawdown, 0, alpha=0.6, color='red')
        ax2.set_title('Drawdown (%)', fontsize=12)
        ax2.set_ylabel('Drawdown %')
        ax2.grid(True, alpha=0.3)
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

        plt.tight_layout()
        plt.savefig(self.plot_dir / 'backtest_equity.png', dpi=150, bbox_inches='tight')
        plt.close()

    def _plot_monthly_heatmap(self):
        """Plot monthly returns heatmap."""
        if self.monthly_table is None or len(self.monthly_table) == 0:
            return

        # Use only monthly columns (exclude 'Year Total')
        table = self.monthly_table.drop(columns=['Year Total'], errors='ignore')

        fig, ax = plt.subplots(figsize=(14, max(6, len(table) * 0.5)))

        # Create color array
        data = table.values
        mask = np.isnan(data)

        im = ax.imshow(data, cmap='RdYlGn', aspect='auto',
                       vmin=-3, vmax=3)

        ax.set_xticks(range(len(table.columns)))
        ax.set_xticklabels(table.columns, fontsize=10)
        ax.set_yticks(range(len(table.index)))
        ax.set_yticklabels(table.index, fontsize=10)

        # Add text annotations
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                if not np.isnan(data[i, j]):
                    color = 'white' if abs(data[i, j]) > 2 else 'black'
                    ax.text(j, i, f'{data[i, j]:.1f}%', ha='center', va='center',
                           fontsize=8, color=color)

        plt.colorbar(im, ax=ax, label='Monthly Return (%)')
        ax.set_title('Monthly Returns Heatmap (%)', fontsize=14, fontweight='bold')

        plt.tight_layout()
        plt.savefig(self.plot_dir / 'backtest_monthly.png', dpi=150, bbox_inches='tight')
        plt.close()

    def _plot_pair_comparison(self):
        """Plot pair-by-pair performance comparison."""
        pair_comp = self.backtester.get_pair_comparison()
        if pair_comp.empty:
            return

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        pairs = pair_comp['pair']
        x = range(len(pairs))

        # Total PnL
        ax1 = axes[0]
        colors = ['green' if v > 0 else 'red' for v in pair_comp['total_pnl']]
        ax1.barh(x, pair_comp['total_pnl'], color=colors, alpha=0.7)
        ax1.set_yticks(x)
        ax1.set_yticklabels(pairs, fontsize=9)
        ax1.set_title('Total PnL ($)', fontweight='bold')
        ax1.axvline(x=0, color='black', linewidth=0.5)
        ax1.grid(True, alpha=0.3)

        # Sharpe
        ax2 = axes[1]
        colors = ['green' if v > 0 else 'red' for v in pair_comp['sharpe']]
        ax2.barh(x, pair_comp['sharpe'], color=colors, alpha=0.7)
        ax2.set_yticks(x)
        ax2.set_yticklabels(pairs, fontsize=9)
        ax2.set_title('Sharpe Ratio', fontweight='bold')
        ax2.axvline(x=0, color='black', linewidth=0.5)
        ax2.grid(True, alpha=0.3)

        # Win Rate
        ax3 = axes[2]
        ax3.barh(x, pair_comp['win_rate'], color='steelblue', alpha=0.7)
        ax3.set_yticks(x)
        ax3.set_yticklabels(pairs, fontsize=9)
        ax3.set_title('Win Rate (%)', fontweight='bold')
        ax3.axvline(x=50, color='orange', linestyle='--', alpha=0.7, label='50%')
        ax3.grid(True, alpha=0.3)
        ax3.legend()

        plt.suptitle('Pair-by-Pair Performance', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.plot_dir / 'backtest_pair_comparison.png', dpi=150, bbox_inches='tight')
        plt.close()

    def _plot_drawdown_analysis(self):
        """Plot drawdown duration and recovery analysis."""
        equity = self.portfolio_df['equity']
        peak = equity.cummax()
        drawdown = (equity - peak) / peak * 100

        fig, axes = plt.subplots(1, 2, figsize=(16, 5))

        # Drawdown distribution
        ax1 = axes[0]
        dd_values = drawdown[drawdown < -0.01]
        if len(dd_values) > 0:
            ax1.hist(dd_values, bins=50, color='red', alpha=0.6, edgecolor='darkred')
        ax1.set_title('Drawdown Distribution', fontweight='bold')
        ax1.set_xlabel('Drawdown (%)')
        ax1.set_ylabel('Frequency')
        ax1.grid(True, alpha=0.3)

        # Rolling returns
        ax2 = axes[1]
        rolling_30 = self.portfolio_df['daily_return'].rolling(30).sum() * 100
        rolling_90 = self.portfolio_df['daily_return'].rolling(90).sum() * 100
        ax2.plot(rolling_30.index, rolling_30, color='steelblue', alpha=0.7,
                label='30-day', linewidth=0.8)
        ax2.plot(rolling_90.index, rolling_90, color='darkblue', alpha=0.9,
                label='90-day', linewidth=1.2)
        ax2.axhline(y=0, color='black', linewidth=0.5)
        ax2.set_title('Rolling Returns (%)', fontweight='bold')
        ax2.set_ylabel('Cumulative Return (%)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

        plt.tight_layout()
        plt.savefig(self.plot_dir / 'backtest_drawdown_analysis.png', dpi=150, bbox_inches='tight')
        plt.close()

    def _print_final_report(self):
        """Print the final backtest report."""
        m = self.metrics

        print(f"\n{'=' * 70}")
        print(f"📋 BACKTEST FINAL REPORT")
        print(f"{'=' * 70}")

        print(f"\n  RETURNS")
        print(f"  {'─' * 45}")
        print(f"  Total Return:          {m.get('total_return_pct', 0):>10.2f}%")
        print(f"  CAGR:                  {m.get('cagr_pct', 0):>10.2f}%")
        print(f"  Best Day:              {m.get('best_day_pct', 0):>10.2f}%")
        print(f"  Worst Day:             {m.get('worst_day_pct', 0):>10.2f}%")
        print(f"  Best Month:            {m.get('best_month_pct', 0):>10.2f}%")
        print(f"  Worst Month:           {m.get('worst_month_pct', 0):>10.2f}%")

        print(f"\n  RISK")
        print(f"  {'─' * 45}")
        print(f"  Annualized Vol:        {m.get('annualized_volatility_pct', 0):>10.2f}%")
        print(f"  Max Drawdown:          {m.get('max_drawdown_pct', 0):>10.2f}%")
        print(f"  Max DD Duration:       {m.get('max_dd_duration_days', 0):>10d} days")
        print(f"  Daily VaR (95%):       {m.get('daily_var_95_pct', 0):>10.2f}%")
        print(f"  Daily VaR (99%):       {m.get('daily_var_99_pct', 0):>10.2f}%")

        print(f"\n  RISK-ADJUSTED")
        print(f"  {'─' * 45}")
        print(f"  Sharpe Ratio:          {m.get('sharpe_ratio', 0):>10.3f}")
        print(f"  Sortino Ratio:         {m.get('sortino_ratio', 0):>10.3f}")
        print(f"  Calmar Ratio:          {m.get('calmar_ratio', 0) if 'calmar_ratio' in m else 0:>10.3f}")
        print(f"  Omega Ratio:           {m.get('omega_ratio', 0):>10.3f}")

        print(f"\n  TRADES")
        print(f"  {'─' * 45}")
        print(f"  Total Trades:          {m.get('total_trades', 0):>10d}")
        print(f"  Trades/Year:           {m.get('trades_per_year', 0):>10.1f}")
        print(f"  Win Rate:              {m.get('win_rate_pct', 0):>10.1f}%")
        print(f"  Profit Factor:         {m.get('profit_factor', 0):>10.2f}")
        print(f"  Avg Win:             ${m.get('avg_win_dollar', 0):>10.2f}")
        print(f"  Avg Loss:            ${m.get('avg_loss_dollar', 0):>10.2f}")
        print(f"  Expectancy/Trade:    ${m.get('expectancy_per_trade', 0):>10.2f}")

        print(f"\n  CONSISTENCY")
        print(f"  {'─' * 45}")
        print(f"  Positive Days:         {m.get('positive_days_pct', 0):>10.1f}%")
        print(f"  Positive Months:       {m.get('positive_months_pct', 0):>10.1f}%")
        print(f"  Longest Win Streak:    {m.get('longest_win_streak_days', 0):>10d} days")
        print(f"  Longest Loss Streak:   {m.get('longest_loss_streak_days', 0):>10d} days")

        print(f"\n  COSTS")
        print(f"  {'─' * 45}")
        print(f"  Total Txn Costs:     ${m.get('total_transaction_costs', 0):>10,.2f}")
        print(f"  Avg Active Pairs:      {m.get('avg_active_pairs', 0):>10.1f}")

        print(f"\n  MONTHLY RETURNS TABLE")
        print(f"  {'─' * 45}")
        if self.monthly_table is not None:
            print(self.monthly_table.to_string())

        print(f"\n{'=' * 70}")
        print(f"✅ BACKTESTING COMPLETE!")
        print(f"{'=' * 70}")


# ============================================================
# Main entry point
# ============================================================
if __name__ == "__main__":
    pipeline = BacktestPipeline(
        total_capital=1_000_000,
        transaction_cost_bps=10,
        slippage_bps=5
    )
    metrics = pipeline.run(verbose=True)

    print(f"\n📋 OUTPUT FILES:")
    print(f"   data/results/backtest_equity.csv")
    print(f"   data/results/backtest_metrics.txt")
    print(f"   data/results/backtest_monthly_returns.csv")
    print(f"   data/results/backtest_pair_comparison.csv")
    print(f"   data/results/backtest_trades.csv")
    print(f"   data/results/plots/backtest_equity.png")
    print(f"   data/results/plots/backtest_monthly.png")
    print(f"   data/results/plots/backtest_pair_comparison.png")
    print(f"   data/results/plots/backtest_drawdown_analysis.png")
    print(f"\n🎯 Ready for Step 7: Analytics & Dashboard!")