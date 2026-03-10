"""
Risk Management Pipeline
===========================
Orchestrates the full risk management workflow:
1. Load signals for all pairs
2. Size positions based on volatility and regime
3. Compute portfolio returns
4. Calculate VaR and drawdowns
5. Apply circuit breakers
6. Save risk report and plots
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

from src.risk.position_sizer import PositionSizer
from src.risk.var_calculator import VaRCalculator
from src.risk.drawdown_monitor import DrawdownMonitor
from src.risk.portfolio_risk import PortfolioRiskManager


class RiskPipeline:
    """
    End-to-end risk management pipeline.
    
    Parameters
    ----------
    total_capital : float
        Starting capital. Default: \$1,000,000.
    output_dir : str
        Directory for output files.
    plot_dir : str
        Directory for plots.
    """
    
    def __init__(
        self,
        total_capital: float = 1_000_000,
        output_dir: str = "data/results",
        plot_dir: str = "data/results/plots"
    ):
        self.total_capital = total_capital
        self.output_dir = Path(output_dir)
        self.plot_dir = Path(plot_dir)
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.plot_dir.mkdir(parents=True, exist_ok=True)
        
        self.manager = PortfolioRiskManager(total_capital=total_capital)
        self.sizer = PositionSizer(total_capital=total_capital)
        
        # Results
        self.pair_signals = {}
        self.sized_signals = {}
        self.portfolio_df = None
        self.risk_report = None
        self.risk_summary = None
        
    def run(self, verbose: bool = True) -> dict:
        """Execute the full risk pipeline."""
        print("=" * 70)
        print("🛡️ RISK MANAGEMENT PIPELINE")
        print("=" * 70)
        
        # Step 1: Load data
        prices, pair_signals, regime_df = self._load_data(verbose)
        
        # Step 2: Position sizing per pair
        self._size_positions(pair_signals, verbose)
        
        # Step 3: Portfolio-level scaling
        self._portfolio_scaling(verbose)
        
        # Step 4: Compute portfolio returns
        self._compute_portfolio(pair_signals, prices, verbose)
        
        # Step 5: Risk metrics
        self._compute_risk_metrics(verbose)
        
        # Step 6: Save
        self._save_all(verbose)
        
        # Step 7: Plot
        self._plot_results(verbose)
        
        print(f"\n{'=' * 70}")
        print(f"✅ RISK MANAGEMENT COMPLETE!")
        print(f"{'=' * 70}")
        
        return self.risk_summary
    
    def _load_data(self, verbose: bool) -> tuple:
        """Load prices, signals, and regime data."""
        if verbose:
            print(f"\n📥 Step 1: Loading data...")
        
        prices = pd.read_csv("data/processed/close_prices.csv", index_col=0, parse_dates=True)
        
        regime_df = pd.read_csv("data/results/regime_labels.csv", index_col=0, parse_dates=True)
        
        signals_dir = Path("data/results/signals")
        signal_files = list(signals_dir.glob("signals_*.csv"))
        
        if not signal_files:
            raise FileNotFoundError("No signal files found. Run Step 4 first!")
        
        pair_signals = {}
        for f in signal_files:
            pair_name = f.stem.replace("signals_", "")
            pair_signals[pair_name] = pd.read_csv(f, index_col=0, parse_dates=True)
        
        if verbose:
            print(f"   Prices: {prices.shape}")
            print(f"   Regime labels: {len(regime_df)} days")
            print(f"   Pair signals: {len(pair_signals)} pairs")
        
        self.pair_signals = pair_signals
        return prices, pair_signals, regime_df
    
    def _size_positions(self, pair_signals: dict, verbose: bool):
        """Apply position sizing to each pair."""
        if verbose:
            print(f"\n📐 Step 2: Position sizing...")
        
        for pair_name, signals_df in pair_signals.items():
            try:
                sized = self.sizer.compute_position_sizes(signals_df)
                self.sized_signals[pair_name] = sized
                
                summary = self.sizer.get_sizing_summary(sized)
                if verbose and summary.get('status') != 'no_active_positions':
                    print(f"   {pair_name}: avg ${summary['avg_position_size']:,.0f}, "
                          f"max ${summary['max_position_size']:,.0f}, "
                          f"util {summary['utilization_pct']:.1f}%")
                          
            except Exception as e:
                if verbose:
                    print(f"   ⚠️ {pair_name}: sizing failed - {str(e)[:50]}")
    
    def _portfolio_scaling(self, verbose: bool):
        """Apply portfolio-level position constraints."""
        if verbose:
            print(f"\n⚖️ Step 3: Portfolio-level scaling...")
        
        if self.sized_signals:
            self.sized_signals = self.sizer.compute_portfolio_sizes(self.sized_signals)
            
            if verbose:
                # Check how often scaling was applied
                for pair_name, df in self.sized_signals.items():
                    if 'portfolio_scale' in df.columns:
                        scaled_days = (df['portfolio_scale'] < 0.999).sum()
                        if scaled_days > 0:
                            print(f"   {pair_name}: scaled down on {scaled_days} days")
        
        if verbose:
            print(f"   ✅ Portfolio constraints applied")
    
    def _compute_portfolio(self, pair_signals: dict, prices: pd.DataFrame, verbose: bool):
        """Compute portfolio-level returns."""
        if verbose:
            print(f"\n💰 Step 4: Computing portfolio returns...")
        
        self.portfolio_df = self.manager.compute_portfolio_returns(
            pair_signals, prices
        )
        
        if verbose:
            equity = self.portfolio_df['equity']
            total_ret = (equity.iloc[-1] / equity.iloc[0] - 1) * 100
            print(f"   Period: {equity.index[0].date()} → {equity.index[-1].date()}")
            print(f"   Starting equity: ${equity.iloc[0]:,.0f}")
            print(f"   Ending equity: ${equity.iloc[-1]:,.0f}")
            print(f"   Total return: {total_ret:.2f}%")
    
    def _compute_risk_metrics(self, verbose: bool):
        """Compute all risk metrics."""
        if verbose:
            print(f"\n📊 Step 5: Computing risk metrics...")
        
        self.risk_report = self.manager.compute_risk_report()
        self.risk_summary = self.manager.get_risk_summary()
        
        if verbose:
            s = self.risk_summary
            print(f"\n   {'─' * 45}")
            print(f"   PORTFOLIO RISK SUMMARY")
            print(f"   {'─' * 45}")
            print(f"   Annualized return:  {s['annualized_return_pct']:>8.2f}%")
            print(f"   Annualized vol:     {s['annualized_volatility_pct']:>8.2f}%")
            print(f"   Sharpe ratio:       {s['sharpe_ratio']:>8.3f}")
            print(f"   Max drawdown:       {s['max_drawdown_pct']:>8.2f}%")
            print(f"   Max DD duration:    {s['max_drawdown_duration']:>5d} days")
            print(f"   Calmar ratio:       {s['calmar_ratio']:>8.3f}")
            print(f"   Positive days:      {s['positive_days_pct']:>8.1f}%")
            print(f"   Best day:           {s['best_day_pct']:>8.2f}%")
            print(f"   Worst day:          {s['worst_day_pct']:>8.2f}%")
            print(f"   {'─' * 45}")
            
            print(f"\n   VaR Snapshot (current):")
            for key, val in s['var_snapshot'].items():
                if '$' in key:
                    print(f"     {key:>15s}: ${val:>12,.2f}")
                elif not np.isnan(val):
                    print(f"     {key:>15s}: {val:>12.4%}")
            
            print(f"\n   Circuit Breaker Triggers:")
            for level, count in s['circuit_breakers'].items():
                print(f"     {level}: {count} days")
    
    def _save_all(self, verbose: bool):
        """Save all risk results."""
        if verbose:
            print(f"\n💾 Step 6: Saving results...")
        
        # Save risk report
        self.manager.save_risk_report(str(self.output_dir))
        
        # Save portfolio returns
        portfolio_path = self.output_dir / "portfolio_returns.csv"
        self.portfolio_df.to_csv(portfolio_path)
        if verbose:
            print(f"✅ Saved portfolio returns to {portfolio_path}")
        
        # Save sized signals
        sized_dir = self.output_dir / "sized_signals"
        sized_dir.mkdir(exist_ok=True)
        for pair_name, df in self.sized_signals.items():
            path = sized_dir / f"sized_{pair_name}.csv"
            df.to_csv(path)
        if verbose:
            print(f"✅ Saved {len(self.sized_signals)} sized signal files")
        
        # Save risk summary as text
        summary_path = self.output_dir / "risk_summary.txt"
        with open(summary_path, 'w') as f:
            f.write("PORTFOLIO RISK SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            for key, val in self.risk_summary.items():
                if isinstance(val, dict):
                    f.write(f"\n{key}:\n")
                    for k2, v2 in val.items():
                        f.write(f"  {k2}: {v2}\n")
                else:
                    f.write(f"{key}: {val}\n")
        if verbose:
            print(f"✅ Saved risk summary to {summary_path}")
    
    def _plot_results(self, verbose: bool):
        """Generate risk visualizations."""
        if verbose:
            print(f"\n📊 Step 7: Generating plots...")
        
        try:
            self._plot_equity_and_drawdown()
            self._plot_risk_dashboard()
            if verbose:
                print(f"   ✅ Plots saved to {self.plot_dir}/")
        except Exception as e:
            if verbose:
                print(f"   ⚠️ Plotting error (non-critical): {e}")
    
    def _plot_equity_and_drawdown(self):
        """Plot equity curve with drawdown overlay."""
        fig, axes = plt.subplots(3, 1, figsize=(16, 12),
                                  gridspec_kw={'height_ratios': [3, 1, 1]})
        
        report = self.risk_report
        
        # Equity curve
        ax1 = axes[0]
        ax1.plot(report.index, report['equity'], color='steelblue', linewidth=1.2)
        ax1.fill_between(report.index, self.total_capital, report['equity'],
                         where=report['equity'] >= self.total_capital,
                         alpha=0.2, color='green', label='Profit')
        ax1.fill_between(report.index, self.total_capital, report['equity'],
                         where=report['equity'] < self.total_capital,
                         alpha=0.2, color='red', label='Loss')
        ax1.axhline(y=self.total_capital, color='gray', linestyle='--', alpha=0.5)
        ax1.set_title('Portfolio Equity Curve', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Equity ($)')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        
        # Drawdown
        ax2 = axes[1]
        ax2.fill_between(report.index, report['drawdown_pct'] * 100, 0,
                         alpha=0.6, color='red')
        ax2.axhline(y=-5, color='orange', linestyle='--', alpha=0.7, label='Warning (5%)')
        ax2.axhline(y=-10, color='red', linestyle='--', alpha=0.7, label='Defensive (10%)')
        ax2.set_title('Drawdown', fontsize=12)
        ax2.set_ylabel('Drawdown (%)')
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        
        # Active pairs
        ax3 = axes[2]
        ax3.fill_between(report.index, report['n_active_pairs'],
                         alpha=0.5, color='steelblue')
        ax3.set_title('Active Pairs', fontsize=12)
        ax3.set_ylabel('# Pairs')
        ax3.grid(True, alpha=0.3)
        ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        
        plt.tight_layout()
        plt.savefig(self.plot_dir / 'equity_drawdown.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_risk_dashboard(self):
        """Plot VaR and rolling Sharpe."""
        fig, axes = plt.subplots(2, 1, figsize=(16, 8))
        
        report = self.risk_report
        
        # Rolling VaR
        ax1 = axes[0]
        if 'var_95_dollar' in report.columns:
            ax1.plot(report.index, report['var_95_dollar'],
                     color='orange', linewidth=1, label='95% VaR ($)')
        if 'var_99_dollar' in report.columns:
            ax1.plot(report.index, report['var_99_dollar'],
                     color='red', linewidth=1, label='99% VaR ($)')
        ax1.set_title('Rolling Value at Risk (Dollar)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('VaR ($)')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        
        # Rolling Sharpe
        ax2 = axes[1]
        if 'rolling_sharpe' in report.columns:
            ax2.plot(report.index, report['rolling_sharpe'],
                     color='steelblue', linewidth=1)
            ax2.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
            ax2.axhline(y=1, color='green', linestyle='--', alpha=0.5, label='Sharpe = 1')
            ax2.axhline(y=2, color='darkgreen', linestyle='--', alpha=0.5, label='Sharpe = 2')
        ax2.set_title('Rolling Sharpe Ratio (1Y)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Sharpe')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        
        plt.tight_layout()
        plt.savefig(self.plot_dir / 'risk_dashboard.png', dpi=150, bbox_inches='tight')
        plt.close()


# ============================================================
# Main entry point
# ============================================================
if __name__ == "__main__":
    pipeline = RiskPipeline(total_capital=1_000_000)
    risk_summary = pipeline.run(verbose=True)
    
    print(f"\n{'=' * 70}")
    print(f"📋 OUTPUT FILES:")
    print(f"{'=' * 70}")
    print(f"   data/results/risk_report.csv         — Daily risk metrics")
    print(f"   data/results/portfolio_returns.csv    — Portfolio equity + returns")
    print(f"   data/results/risk_summary.txt         — Summary statistics")
    print(f"   data/results/sized_signals/           — Position-sized signals")
    print(f"   data/results/plots/equity_drawdown.png")
    print(f"   data/results/plots/risk_dashboard.png")
    print(f"\n🎯 Ready for Step 6: Backtesting Engine!")