"""
Regime-Specific Performance Analysis
=======================================
Breaks down strategy performance by market regime to answer:
- Does the strategy work better in Bull vs Bear markets?
- Should we trade during Crisis regimes at all?
- How does regime detection add value vs a static strategy?
"""

import numpy as np
import pandas as pd
from pathlib import Path


class RegimePerformanceAnalyzer:
    """
    Analyzes backtest performance segmented by market regime.
    """

    def __init__(self):
        self.regime_stats = None

    def analyze(
        self,
        backtest_df: pd.DataFrame,
        regime_df: pd.DataFrame,
        verbose: bool = True
    ) -> pd.DataFrame:
        """
        Compute performance metrics per regime.

        Parameters
        ----------
        backtest_df : pd.DataFrame
            Backtest results with 'daily_return' column
        regime_df : pd.DataFrame
            Regime labels with 'regime_name' or 'regime_name_smoothed' column

        Returns
        -------
        pd.DataFrame
            Per-regime performance metrics
        """
        # Align dates
        common = backtest_df.index.intersection(regime_df.index)
        returns = backtest_df['daily_return'].reindex(common)

        regime_col = 'regime_name_smoothed' if 'regime_name_smoothed' in regime_df.columns else 'regime_name'
        regimes = regime_df[regime_col].reindex(common)

        results = []
        for regime_name in regimes.unique():
            mask = regimes == regime_name
            r = returns[mask]

            if len(r) < 10:
                continue

            ann_return = r.mean() * 252
            ann_vol = r.std() * np.sqrt(252)
            sharpe = ann_return / ann_vol if ann_vol > 0 else 0

            # Drawdown in this regime
            equity = (1 + r).cumprod()
            peak = equity.cummax()
            dd = ((equity - peak) / peak).min()

            results.append({
                'regime': regime_name,
                'days': len(r),
                'pct_of_total': len(r) / len(returns) * 100,
                'total_return_pct': r.sum() * 100,
                'ann_return_pct': ann_return * 100,
                'ann_vol_pct': ann_vol * 100,
                'sharpe': sharpe,
                'max_dd_pct': dd * 100,
                'positive_days_pct': (r > 0).mean() * 100,
                'avg_daily_return_bps': r.mean() * 10000,
                'best_day_pct': r.max() * 100,
                'worst_day_pct': r.min() * 100,
            })

        self.regime_stats = pd.DataFrame(results)

        if verbose:
            print(f"\n📊 REGIME-SPECIFIC PERFORMANCE")
            print(f"{'─' * 70}")
            print(f"{'Regime':<10} {'Days':>6} {'%Total':>7} {'Return':>8} "
                  f"{'AnnRet':>8} {'Sharpe':>7} {'MaxDD':>7} {'WinDay%':>8}")
            print(f"{'─' * 70}")
            for _, row in self.regime_stats.iterrows():
                print(f"{row['regime']:<10} {row['days']:>6.0f} "
                      f"{row['pct_of_total']:>6.1f}% "
                      f"{row['total_return_pct']:>7.2f}% "
                      f"{row['ann_return_pct']:>7.2f}% "
                      f"{row['sharpe']:>7.3f} "
                      f"{row['max_dd_pct']:>6.2f}% "
                      f"{row['positive_days_pct']:>7.1f}%")

        return self.regime_stats

    def save(self, output_dir: str = "data/results") -> str:
        """Save regime analysis."""
        if self.regime_stats is None:
            raise ValueError("No analysis done yet.")
        path = Path(output_dir) / "regime_performance.csv"
        self.regime_stats.to_csv(path, index=False)
        print(f"✅ Saved regime performance to {path}")
        return str(path)


if __name__ == "__main__":
    print("=" * 60)
    print("REGIME PERFORMANCE ANALYSIS")
    print("=" * 60)

    bt_path = Path("data/results/backtest_equity.csv")
    if not bt_path.exists():
        bt_path = Path("data/results/backtest_optimized_equity.csv")
    regime_path = Path("data/results/regime_labels.csv")

    if not bt_path.exists() or not regime_path.exists():
        print("Missing files. Run Steps 3 and 6 first.")
        exit(1)

    bt_df = pd.read_csv(bt_path, index_col=0, parse_dates=True)
    regime_df = pd.read_csv(regime_path, index_col=0, parse_dates=True)

    analyzer = RegimePerformanceAnalyzer()
    stats = analyzer.analyze(bt_df, regime_df, verbose=True)
    analyzer.save()

    print("\n✅ Regime analysis complete!")