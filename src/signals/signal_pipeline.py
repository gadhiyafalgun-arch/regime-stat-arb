"""
Signal Generation Pipeline
=============================
Orchestrates the full signal generation workflow for ALL selected pairs:
1. Load pair list and price data
2. For each pair: Kalman Filter → Spread → Z-score → Signals
3. Merge with regime data
4. Save all results and generate plots
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

from src.signals.kalman_filter import KalmanHedgeRatio
from src.signals.spread_calculator import SpreadCalculator
from src.signals.signal_generator import RegimeAdaptiveSignalGenerator


class SignalPipeline:
    """
    End-to-end signal generation for all selected pairs.

    Parameters
    ----------
    delta : float
        Kalman Filter process noise (default: 1e-4)
    zscore_lookback : int
        Rolling window for z-score (default: 63)
    crisis_trading : bool
        Allow new entries during Crisis regime (default: False)
    output_dir : str
        Directory for output files
    plot_dir : str
        Directory for plots
    """

    def __init__(
        self,
        delta: float = 1e-4,
        zscore_lookback: int = 63,
        crisis_trading: bool = False,
        output_dir: str = "data/results",
        plot_dir: str = "data/results/plots"
    ):
        self.delta = delta
        self.zscore_lookback = zscore_lookback
        self.crisis_trading = crisis_trading
        self.output_dir = Path(output_dir)
        self.plot_dir = Path(plot_dir)

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.plot_dir.mkdir(parents=True, exist_ok=True)

        # Stored results
        self.pair_results = {}
        self.pair_summaries = {}
        self.pair_trades = {}

    def run(self, verbose: bool = True) -> dict:
        """Run signal generation for all selected pairs."""
        print("=" * 70)
        print("🎯 SIGNAL GENERATION PIPELINE")
        print("=" * 70)

        prices, regime_df, pairs = self._load_data(verbose)

        if verbose:
            print(f"\n📋 Processing {len(pairs)} pairs...")

        successful = 0
        failed = 0

        for idx, row in pairs.iterrows():
            asset_y = row['asset1']
            asset_x = row['asset2']
            pair_name = f"{asset_y}-{asset_x}"

            try:
                if verbose:
                    print(f"\n{'─' * 50}")
                    print(f"   Processing {pair_name} ({idx+1}/{len(pairs)})...")

                if asset_y not in prices.columns or asset_x not in prices.columns:
                    if verbose:
                        print(f"   ⚠️ Skipping — assets not in price data")
                    failed += 1
                    continue

                # Step 1: Kalman Filter
                kf = KalmanHedgeRatio(delta=self.delta)
                kalman_df = kf.fit_dataframe(prices, asset_y, asset_x)

                # Step 2: Spread and Z-score
                calc = SpreadCalculator(zscore_lookback=self.zscore_lookback)
                spread_df = calc.compute_from_kalman(
                    kalman_df, prices, asset_y, asset_x
                )

                # Step 3: Regime-adaptive signals
                gen = RegimeAdaptiveSignalGenerator(
                    crisis_trading=self.crisis_trading
                )
                signals_df = gen.generate_signals(spread_df, regime_df)

                # Store results
                self.pair_results[pair_name] = signals_df
                self.pair_summaries[pair_name] = gen.get_signal_summary()
                self.pair_trades[pair_name] = gen.get_trade_log_df()

                kf_diag = kf.get_diagnostics()
                spread_diag = calc.get_spread_diagnostics()
                summary = self.pair_summaries[pair_name]

                if verbose:
                    print(f"   beta range: [{kf_diag['beta_min']:.3f}, {kf_diag['beta_max']:.3f}]")
                    print(f"   Halflife: {spread_diag['mean_reversion_halflife']:.1f} days")
                    print(f"   Trades: {summary['total_trades']} ({summary['trades_per_year']:.1f}/yr)")
                    print(f"   Time in market: {100 - summary['flat_pct']:.1f}%")
                    print(f"   Stop losses: {summary['total_stop_losses']}")

                successful += 1

            except Exception as e:
                if verbose:
                    print(f"   ❌ Failed: {str(e)[:80]}")
                failed += 1

        print(f"\n{'=' * 70}")
        print(f"📊 PIPELINE SUMMARY: {successful} succeeded, {failed} failed")
        print(f"{'=' * 70}")

        self._print_comparison_table()
        self._save_all(verbose)
        self._plot_all(verbose)

        print(f"\n✅ Signal generation complete for {successful} pairs!")
        return self.pair_results

    def _load_data(self, verbose: bool) -> tuple:
        """Load prices, regime labels, and pair list."""
        prices_path = Path("data/processed/close_prices.csv")
        if not prices_path.exists():
            raise FileNotFoundError(f"Prices not found at {prices_path}")
        prices = pd.read_csv(prices_path, index_col=0, parse_dates=True)

        regime_path = Path("data/results/regime_labels.csv")
        if not regime_path.exists():
            raise FileNotFoundError(f"Regime labels not found at {regime_path}. Run Step 3 first!")
        regime_df = pd.read_csv(regime_path, index_col=0, parse_dates=True)

        pairs_path = Path("data/results/stable_pairs_final.csv")
        if not pairs_path.exists():
            raise FileNotFoundError(f"Pairs not found at {pairs_path}. Run Step 2 first!")
        pairs = pd.read_csv(pairs_path)

        if verbose:
            print(f"\n📥 Loaded data:")
            print(f"   Prices: {prices.shape[0]} days x {prices.shape[1]} assets")
            print(f"   Regime labels: {len(regime_df)} days")
            print(f"   Selected pairs: {len(pairs)}")

        return prices, regime_df, pairs

    def _print_comparison_table(self):
        """Print summary comparison table across all pairs."""
        if not self.pair_summaries:
            return

        print(f"\n{'Pair':<15} {'Trades':>7} {'Trades/Yr':>10} {'In Market':>10} "
              f"{'Avg Hold':>9} {'Stops':>6} {'Long%':>7} {'Short%':>7}")
        print("-" * 82)

        for pair_name, summary in sorted(self.pair_summaries.items()):
            print(f"{pair_name:<15} "
                  f"{summary['total_trades']:>7} "
                  f"{summary['trades_per_year']:>10.1f} "
                  f"{100 - summary['flat_pct']:>9.1f}% "
                  f"{summary['avg_holding_days']:>8.1f}d "
                  f"{summary['total_stop_losses']:>6} "
                  f"{summary['long_pct']:>6.1f}% "
                  f"{summary['short_pct']:>6.1f}%")

    def _save_all(self, verbose: bool):
        """Save all signal results."""
        if verbose:
            print(f"\n💾 Saving results...")

        # Save individual pair signals
        signals_dir = self.output_dir / "signals"
        signals_dir.mkdir(exist_ok=True)

        for pair_name, signals_df in self.pair_results.items():
            path = signals_dir / f"signals_{pair_name}.csv"
            signals_df.to_csv(path)

        if verbose:
            print(f"   ✅ Saved {len(self.pair_results)} signal files to {signals_dir}/")

        # Save trade logs
        trades_dir = self.output_dir / "trades"
        trades_dir.mkdir(exist_ok=True)

        all_trades = []
        for pair_name, trades_df in self.pair_trades.items():
            if len(trades_df) > 0:
                trades_df_copy = trades_df.copy()
                trades_df_copy['pair'] = pair_name
                all_trades.append(trades_df_copy)

                path = trades_dir / f"trades_{pair_name}.csv"
                trades_df.to_csv(path, index=False)

        if all_trades:
            all_trades_df = pd.concat(all_trades, ignore_index=True)
            all_trades_path = self.output_dir / "all_trades.csv"
            all_trades_df.to_csv(all_trades_path, index=False)
            if verbose:
                print(f"   ✅ Saved {len(all_trades_df)} total trades to {all_trades_path}")

        # Save summary
        summary_rows = []
        for pair_name, summary in self.pair_summaries.items():
            row = {
                'pair': pair_name,
                'total_trades': summary['total_trades'],
                'trades_per_year': summary['trades_per_year'],
                'time_in_market_pct': 100 - summary['flat_pct'],
                'long_pct': summary['long_pct'],
                'short_pct': summary['short_pct'],
                'avg_holding_days': summary['avg_holding_days'],
                'stop_losses': summary['total_stop_losses'],
            }
            summary_rows.append(row)

        summary_df = pd.DataFrame(summary_rows)
        summary_path = self.output_dir / "signal_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        if verbose:
            print(f"   ✅ Saved signal summary to {summary_path}")

    def _plot_all(self, verbose: bool):
        """Generate plots for top pairs."""
        if verbose:
            print(f"\n📊 Generating plots...")

        try:
            sorted_pairs = sorted(
                self.pair_summaries.items(),
                key=lambda x: x[1]['total_trades'],
                reverse=True
            )

            for pair_name, _ in sorted_pairs[:3]:
                self._plot_pair_signals(pair_name)

            if verbose:
                print(f"   ✅ Plots saved to {self.plot_dir}/")

        except Exception as e:
            if verbose:
                print(f"   ⚠️ Plotting error (non-critical): {e}")

    def _plot_pair_signals(self, pair_name: str):
        """Plot signal chart for a single pair."""
        signals_df = self.pair_results[pair_name]
        valid = signals_df.dropna(subset=['zscore'])

        if len(valid) == 0:
            return

        fig, axes = plt.subplots(3, 1, figsize=(16, 10),
                                  gridspec_kw={'height_ratios': [2, 2, 1]})

        # Plot 1: Z-score with thresholds
        ax1 = axes[0]
        ax1.plot(valid.index, valid['zscore'], color='steelblue',
                 linewidth=0.8, alpha=0.8, label='Z-Score')
        ax1.axhline(y=0, color='black', linewidth=0.5)
        ax1.axhline(y=2.0, color='red', linestyle='--', alpha=0.5, label='Entry (Bear)')
        ax1.axhline(y=-2.0, color='red', linestyle='--', alpha=0.5)
        ax1.axhline(y=1.5, color='orange', linestyle='--', alpha=0.5, label='Entry (Bull)')
        ax1.axhline(y=-1.5, color='orange', linestyle='--', alpha=0.5)

        # Mark entries
        entries_long = valid[valid['signal_type'] == 'entry_long']
        entries_short = valid[valid['signal_type'] == 'entry_short']
        stops = valid[valid['signal_type'] == 'stop_loss']

        if len(entries_long) > 0:
            ax1.scatter(entries_long.index, entries_long['zscore'],
                       color='green', marker='^', s=50, zorder=5, label='Long Entry')
        if len(entries_short) > 0:
            ax1.scatter(entries_short.index, entries_short['zscore'],
                       color='red', marker='v', s=50, zorder=5, label='Short Entry')
        if len(stops) > 0:
            ax1.scatter(stops.index, stops['zscore'],
                       color='black', marker='x', s=50, zorder=5, label='Stop Loss')

        ax1.set_title(f'{pair_name} — Z-Score & Signals', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Z-Score')
        ax1.legend(loc='upper right', fontsize=8)
        ax1.grid(True, alpha=0.3)

        # Plot 2: Position over time
        ax2 = axes[1]
        ax2.fill_between(valid.index, valid['position'], 0,
                         where=valid['position'] > 0, alpha=0.4, color='green', label='Long')
        ax2.fill_between(valid.index, valid['position'], 0,
                         where=valid['position'] < 0, alpha=0.4, color='red', label='Short')
        ax2.set_title('Position', fontsize=12)
        ax2.set_ylabel('Position')
        ax2.set_ylim(-1.5, 1.5)
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)

        # Plot 3: Regime
        ax3 = axes[2]
        regime_colors = {'Bull': '#2ecc71', 'Bear': '#e74c3c', 'Crisis': '#9b59b6'}
        for regime_name, color in regime_colors.items():
            mask = valid['regime'] == regime_name
            if mask.sum() > 0:
                ax3.fill_between(valid.index, 0, 1,
                                where=mask, alpha=0.5, color=color, label=regime_name)
        ax3.set_title('Market Regime', fontsize=12)
        ax3.set_yticks([])
        ax3.legend(loc='upper right', fontsize=9)

        for ax in axes:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
            ax.xaxis.set_major_locator(mdates.YearLocator())

        plt.tight_layout()
        safe_name = pair_name.replace('/', '_')
        plt.savefig(self.plot_dir / f'signals_{safe_name}.png', dpi=150, bbox_inches='tight')
        plt.close()


# ============================================================
# Main entry point
# ============================================================
if __name__ == "__main__":
    pipeline = SignalPipeline(delta=1e-4, zscore_lookback=63, crisis_trading=False)
    pair_results = pipeline.run(verbose=True)

    print(f"\n{'=' * 70}")
    print(f"📋 OUTPUT FILES:")
    print(f"{'=' * 70}")
    print(f"   data/results/signals/         — Signal files per pair")
    print(f"   data/results/trades/          — Trade logs per pair")
    print(f"   data/results/all_trades.csv   — Combined trade log")
    print(f"   data/results/signal_summary.csv")
    print(f"   data/results/plots/signals_*.png")
    print(f"\n🎯 Ready for Step 5: Risk Management!")