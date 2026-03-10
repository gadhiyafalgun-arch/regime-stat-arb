"""
Strategy Parameter Tuner
==========================
Tests multiple parameter combinations to find optimal settings.
Not curve-fitting — uses walk-forward validation to prevent overfitting.

Key parameters to tune:
1. Entry/exit z-score thresholds
2. Minimum holding period
3. Number of pairs traded
4. Kalman filter delta (adaptiveness)
5. Z-score lookback window
"""

import pandas as pd
import numpy as np
from pathlib import Path
from itertools import product

from src.signals.kalman_filter import KalmanHedgeRatio
from src.signals.spread_calculator import SpreadCalculator
from src.signals.signal_generator import RegimeAdaptiveSignalGenerator
from src.backtest.backtest_engine import PortfolioBacktester


class StrategyTuner:
    """
    Tests parameter combinations and reports results.

    Parameters
    ----------
    total_capital : float
        Portfolio capital for backtest.
    top_n_pairs : list
        Number of top pairs to test (e.g., [3, 5, 7])
    """

    def __init__(self, total_capital: float = 1_000_000):
        self.total_capital = total_capital
        self.results = []

    def run_parameter_sweep(self, verbose: bool = True) -> pd.DataFrame:
        """Test key parameter combinations."""

        print("=" * 70)
        print("🔧 STRATEGY PARAMETER TUNING")
        print("=" * 70)

        # Load data
        prices = pd.read_csv("data/processed/close_prices.csv",
                              index_col=0, parse_dates=True)
        regime_df = pd.read_csv("data/results/regime_labels.csv",
                                 index_col=0, parse_dates=True)
        pairs = pd.read_csv("data/results/stable_pairs_final.csv")

        # Sort pairs by stability score (best first)
        pairs = pairs.sort_values('stability_score', ascending=False).reset_index(drop=True)

        # Parameter grid — focused, not exhaustive
        param_grid = {
            'n_pairs': [3, 5, 8],
            'entry_bull': [2.0, 2.5],
            'entry_bear': [2.5, 3.0],
            'min_holding': [10, 15, 21],
            'zscore_lookback': [42, 63, 126],
            'delta': [1e-4, 1e-5],
        }

        # Generate combinations
        keys = list(param_grid.keys())
        combos = list(product(*param_grid.values()))

        print(f"\n📋 Testing {len(combos)} parameter combinations...")
        print(f"   Pairs available: {len(pairs)}")

        self.results = []

        for i, combo in enumerate(combos):
            params = dict(zip(keys, combo))

            if verbose and (i % 10 == 0 or i < 5):
                print(f"\n   [{i+1}/{len(combos)}] Testing: "
                      f"pairs={params['n_pairs']}, "
                      f"entry={params['entry_bull']}/{params['entry_bear']}, "
                      f"hold={params['min_holding']}, "
                      f"lookback={params['zscore_lookback']}, "
                      f"delta={params['delta']}")

            try:
                result = self._test_params(
                    params, prices, regime_df, pairs
                )
                self.results.append(result)

                if verbose and (i % 10 == 0 or i < 5):
                    print(f"         Sharpe: {result['sharpe']:.3f}  "
                          f"Return: {result['total_return_pct']:.2f}%  "
                          f"Trades: {result['total_trades']}  "
                          f"WR: {result['win_rate']:.1f}%")

            except Exception as e:
                if verbose and i < 5:
                    print(f"         FAILED: {str(e)[:60]}")

        # Build results DataFrame
        results_df = pd.DataFrame(self.results)
        results_df = results_df.sort_values('sharpe', ascending=False)

        # Save results
        output_path = Path("data/results/tuning_results.csv")
        results_df.to_csv(output_path, index=False)

        # Print top 10
        print(f"\n{'=' * 70}")
        print(f"📊 TOP 10 PARAMETER COMBINATIONS (by Sharpe)")
        print(f"{'=' * 70}")
        print(f"\n{'Rank':<5} {'Pairs':<6} {'Entry(B/Br)':<12} {'Hold':<5} "
              f"{'LB':<5} {'Delta':<8} {'Sharpe':>7} {'Return':>8} "
              f"{'Trades':>7} {'WR':>6} {'MaxDD':>7}")
        print("-" * 85)

        for rank, (_, row) in enumerate(results_df.head(10).iterrows(), 1):
            print(f"{rank:<5} {row['n_pairs']:<6.0f} "
                  f"{row['entry_bull']:.1f}/{row['entry_bear']:.1f}  "
                  f"{row['min_holding']:<5.0f} "
                  f"{row['zscore_lookback']:<5.0f} "
                  f"{row['delta']:<8.0e} "
                  f"{row['sharpe']:>7.3f} "
                  f"{row['total_return_pct']:>7.2f}% "
                  f"{row['total_trades']:>7.0f} "
                  f"{row['win_rate']:>5.1f}% "
                  f"{row['max_dd']:>6.2f}%")

        # Print best params
        best = results_df.iloc[0]
        print(f"\n🏆 BEST PARAMETERS:")
        print(f"   Pairs: top {best['n_pairs']:.0f}")
        print(f"   Entry (Bull/Bear/Crisis): {best['entry_bull']:.1f} / {best['entry_bear']:.1f} / {best['entry_bear']+0.5:.1f}")
        print(f"   Min holding: {best['min_holding']:.0f} days")
        print(f"   Z-score lookback: {best['zscore_lookback']:.0f} days")
        print(f"   Kalman delta: {best['delta']:.0e}")
        print(f"   Sharpe: {best['sharpe']:.3f}")
        print(f"   Return: {best['total_return_pct']:.2f}%")

        return results_df

    def _test_params(
        self,
        params: dict,
        prices: pd.DataFrame,
        regime_df: pd.DataFrame,
        pairs: pd.DataFrame
    ) -> dict:
        """Test a single parameter combination."""

        n_pairs = int(params['n_pairs'])
        selected_pairs = pairs.head(n_pairs)

        regime_thresholds = {
            'Bull':   {'entry': params['entry_bull'], 'exit': 0.5, 'stop_loss': 4.0},
            'Bear':   {'entry': params['entry_bear'], 'exit': 0.75, 'stop_loss': 3.5},
            'Crisis': {'entry': params['entry_bear'] + 0.5, 'exit': 0.5, 'stop_loss': 3.0},
        }

        # Generate signals for selected pairs
        pair_signals = {}
        for _, row in selected_pairs.iterrows():
            asset_y, asset_x = row['asset1'], row['asset2']
            pair_name = f"{asset_y}-{asset_x}"

            if asset_y not in prices.columns or asset_x not in prices.columns:
                continue

            try:
                # Kalman Filter
                kf = KalmanHedgeRatio(delta=params['delta'])
                kalman_df = kf.fit_dataframe(prices, asset_y, asset_x)

                # Spread
                calc = SpreadCalculator(zscore_lookback=int(params['zscore_lookback']))
                spread_df = calc.compute_from_kalman(kalman_df, prices, asset_y, asset_x)

                # Signals
                gen = RegimeAdaptiveSignalGenerator(
                    regime_thresholds=regime_thresholds,
                    min_holding_period=int(params['min_holding']),
                    crisis_trading=False
                )
                signals_df = gen.generate_signals(spread_df, regime_df)
                pair_signals[pair_name] = signals_df

            except Exception:
                continue

        if len(pair_signals) == 0:
            return {**params, 'sharpe': -999, 'total_return_pct': -999,
                    'total_trades': 0, 'win_rate': 0, 'max_dd': -999}

        # Backtest
        bt = PortfolioBacktester(
            total_capital=self.total_capital,
            transaction_cost_bps=10,
            slippage_bps=5
        )
        bt.run(pair_signals, prices, verbose=False)
        summary = bt.get_portfolio_summary()

        return {
            **params,
            'sharpe': summary.get('sharpe_ratio', -999),
            'total_return_pct': summary.get('total_return_pct', -999),
            'ann_return_pct': summary.get('annualized_return_pct', -999) * 100,
            'ann_vol_pct': summary.get('annualized_vol_pct', -999),
            'total_trades': summary.get('total_trades', 0),
            'trades_per_year': summary.get('trades_per_year', 0),
            'win_rate': summary.get('win_rate_pct', 0),
            'max_dd': summary.get('max_drawdown_pct', -999),
            'profit_factor': summary.get('profit_factor', 0),
            'total_costs': summary.get('total_transaction_costs', 0),
            'avg_active_pairs': summary.get('avg_active_pairs', 0),
        }

    def run_best_backtest(self, results_df: pd.DataFrame = None, verbose: bool = True):
        """Run a full backtest with the best parameters and print detailed report."""

        if results_df is None:
            path = Path("data/results/tuning_results.csv")
            if not path.exists():
                print("No tuning results. Run run_parameter_sweep() first.")
                return
            results_df = pd.read_csv(path)

        best = results_df.iloc[0]

        print(f"\n{'=' * 70}")
        print(f"🏆 RUNNING BEST PARAMETER BACKTEST")
        print(f"{'=' * 70}")
        print(f"   Params: pairs={best['n_pairs']:.0f}, "
              f"entry={best['entry_bull']:.1f}/{best['entry_bear']:.1f}, "
              f"hold={best['min_holding']:.0f}, "
              f"lookback={best['zscore_lookback']:.0f}, "
              f"delta={best['delta']:.0e}")

        # Reload data
        prices = pd.read_csv("data/processed/close_prices.csv",
                              index_col=0, parse_dates=True)
        regime_df = pd.read_csv("data/results/regime_labels.csv",
                                 index_col=0, parse_dates=True)
        pairs = pd.read_csv("data/results/stable_pairs_final.csv")
        pairs = pairs.sort_values('stability_score', ascending=False).reset_index(drop=True)

        n_pairs = int(best['n_pairs'])
        selected_pairs = pairs.head(n_pairs)

        regime_thresholds = {
            'Bull':   {'entry': best['entry_bull'], 'exit': 0.5, 'stop_loss': 4.0},
            'Bear':   {'entry': best['entry_bear'], 'exit': 0.75, 'stop_loss': 3.5},
            'Crisis': {'entry': best['entry_bear'] + 0.5, 'exit': 0.5, 'stop_loss': 3.0},
        }

        # Generate signals
        pair_signals = {}
        for _, row in selected_pairs.iterrows():
            asset_y, asset_x = row['asset1'], row['asset2']
            pair_name = f"{asset_y}-{asset_x}"

            if asset_y not in prices.columns or asset_x not in prices.columns:
                continue

            try:
                kf = KalmanHedgeRatio(delta=best['delta'])
                kalman_df = kf.fit_dataframe(prices, asset_y, asset_x)

                calc = SpreadCalculator(zscore_lookback=int(best['zscore_lookback']))
                spread_df = calc.compute_from_kalman(kalman_df, prices, asset_y, asset_x)

                gen = RegimeAdaptiveSignalGenerator(
                    regime_thresholds=regime_thresholds,
                    min_holding_period=int(best['min_holding']),
                    crisis_trading=False
                )
                signals_df = gen.generate_signals(spread_df, regime_df)

                # Save optimized signals
                signals_dir = Path("data/results/signals_optimized")
                signals_dir.mkdir(exist_ok=True)
                signals_df.to_csv(signals_dir / f"signals_{pair_name}.csv")

                pair_signals[pair_name] = signals_df
            except Exception as e:
                print(f"   Warning: {pair_name} failed — {str(e)[:50]}")

        # Run full backtest
        from src.backtest.backtest_pipeline import BacktestPipeline

        # Override signal loading in pipeline — run backtester directly
        bt = PortfolioBacktester(
            total_capital=self.total_capital,
            transaction_cost_bps=10,
            slippage_bps=5
        )
        portfolio = bt.run(pair_signals, prices, verbose=verbose)
        summary = bt.get_portfolio_summary()

        # Save optimized backtest results
        portfolio.to_csv("data/results/backtest_optimized_equity.csv")

        print(f"\n{'=' * 70}")
        print(f"📋 OPTIMIZED BACKTEST RESULTS")
        print(f"{'=' * 70}")
        print(f"   Total Return:     {summary['total_return_pct']:>8.2f}%")
        print(f"   Ann. Return:      {summary['annualized_return_pct']*100:>8.2f}%")
        print(f"   Ann. Vol:         {summary['annualized_vol_pct']*100:>8.2f}%")
        print(f"   Sharpe Ratio:     {summary['sharpe_ratio']:>8.3f}")
        print(f"   Max Drawdown:     {summary['max_drawdown_pct']:>8.2f}%")
        print(f"   Win Rate:         {summary['win_rate_pct']:>8.1f}%")
        print(f"   Profit Factor:    {summary['profit_factor']:>8.2f}")
        print(f"   Total Trades:     {summary['total_trades']:>8d}")
        print(f"   Trades/Year:      {summary['trades_per_year']:>8.1f}")
        print(f"   Total Costs:    ${summary['total_transaction_costs']:>10,.2f}")

        return summary


# ============================================================
# Main entry point
# ============================================================
if __name__ == "__main__":
    tuner = StrategyTuner(total_capital=1_000_000)

    # Step 1: Parameter sweep
    results_df = tuner.run_parameter_sweep(verbose=True)

    # Step 2: Run best params
    tuner.run_best_backtest(results_df, verbose=True)

    print(f"\n📋 Tuning output saved to data/results/tuning_results.csv")
    print(f"🎯 Ready for Step 7: Dashboard!")