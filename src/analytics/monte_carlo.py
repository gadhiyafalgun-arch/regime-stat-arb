"""
Monte Carlo Stress Testing
=============================
Simulates thousands of possible future paths by resampling
historical returns, answering questions like:
- What's the probability of losing more than 10%?
- What's the range of outcomes over the next year?
- How bad can it get in the 5th percentile scenario?

Method: Block bootstrap
    Instead of resampling individual days (which destroys autocorrelation),
    we resample BLOCKS of consecutive days. This preserves the structure
    of regime clusters and serial correlation in returns.
"""

import numpy as np
import pandas as pd
from pathlib import Path


class MonteCarloSimulator:
    """
    Monte Carlo simulation via block bootstrap of historical returns.

    Parameters
    ----------
    n_simulations : int
        Number of simulated paths. Default: 5000.
    horizon_days : int
        Simulation horizon in trading days. Default: 252 (1 year).
    block_size : int
        Block size for bootstrap. Default: 21 (~1 month).
        Larger blocks preserve more autocorrelation structure.
    confidence_levels : list
        Percentiles to compute. Default: [5, 25, 50, 75, 95].
    """

    def __init__(
        self,
        n_simulations: int = 5000,
        horizon_days: int = 252,
        block_size: int = 21,
        confidence_levels: list = None
    ):
        self.n_simulations = n_simulations
        self.horizon_days = horizon_days
        self.block_size = block_size
        self.confidence_levels = confidence_levels or [5, 25, 50, 75, 95]

        self.simulated_paths = None
        self.final_values = None
        self.statistics = None

    def run(
        self,
        historical_returns: pd.Series,
        initial_capital: float = 1_000_000,
        verbose: bool = True
    ) -> dict:
        """
        Run Monte Carlo simulation.

        Parameters
        ----------
        historical_returns : pd.Series
            Daily returns from backtest
        initial_capital : float
            Starting equity for simulations

        Returns
        -------
        dict with simulation results
        """
        returns = historical_returns.dropna().values
        T = len(returns)

        if verbose:
            print(f"\n🎲 Running Monte Carlo simulation...")
            print(f"   Simulations: {self.n_simulations}")
            print(f"   Horizon: {self.horizon_days} days ({self.horizon_days/252:.1f} years)")
            print(f"   Block size: {self.block_size} days")
            print(f"   Historical returns: {T} days")

        # Block bootstrap
        n_blocks = int(np.ceil(self.horizon_days / self.block_size))
        max_start = T - self.block_size

        if max_start <= 0:
            raise ValueError(f"Not enough data ({T} days) for block size {self.block_size}")

        # Simulate paths
        np.random.seed(42)
        simulated_returns = np.zeros((self.n_simulations, self.horizon_days))

        for sim in range(self.n_simulations):
            path_returns = []
            for _ in range(n_blocks):
                start = np.random.randint(0, max_start)
                block = returns[start:start + self.block_size]
                path_returns.extend(block)
            simulated_returns[sim, :] = path_returns[:self.horizon_days]

        # Convert to equity paths
        cumulative = np.cumprod(1 + simulated_returns, axis=1)
        self.simulated_paths = initial_capital * cumulative
        self.final_values = self.simulated_paths[:, -1]

        # Compute statistics
        self.statistics = self._compute_statistics(initial_capital)

        if verbose:
            self._print_results(initial_capital)

        return self.statistics

    def _compute_statistics(self, initial_capital: float) -> dict:
        """Compute summary statistics from simulations."""
        final = self.final_values
        final_returns = (final / initial_capital - 1) * 100

        # Drawdowns per path
        max_drawdowns = []
        for i in range(self.n_simulations):
            path = self.simulated_paths[i, :]
            peak = np.maximum.accumulate(path)
            dd = (path - peak) / peak
            max_drawdowns.append(dd.min() * 100)

        stats = {
            'mean_final_equity': np.mean(final),
            'median_final_equity': np.median(final),
            'std_final_equity': np.std(final),
            'mean_return_pct': np.mean(final_returns),
            'median_return_pct': np.median(final_returns),
            'prob_profit': (final > initial_capital).mean() * 100,
            'prob_loss_5pct': (final_returns < -5).mean() * 100,
            'prob_loss_10pct': (final_returns < -10).mean() * 100,
            'prob_loss_20pct': (final_returns < -20).mean() * 100,
            'prob_gain_5pct': (final_returns > 5).mean() * 100,
            'prob_gain_10pct': (final_returns > 10).mean() * 100,
            'mean_max_drawdown_pct': np.mean(max_drawdowns),
            'worst_max_drawdown_pct': np.min(max_drawdowns),
            'best_case_return_pct': np.max(final_returns),
            'worst_case_return_pct': np.min(final_returns),
        }

        # Percentiles
        for pct in self.confidence_levels:
            stats[f'percentile_{pct}_equity'] = np.percentile(final, pct)
            stats[f'percentile_{pct}_return_pct'] = np.percentile(final_returns, pct)

        return stats

    def _print_results(self, initial_capital: float):
        """Print Monte Carlo results."""
        s = self.statistics

        print(f"\n   {'─' * 50}")
        print(f"   MONTE CARLO RESULTS ({self.n_simulations} simulations)")
        print(f"   {'─' * 50}")
        print(f"   Probability of profit:     {s['prob_profit']:>6.1f}%")
        print(f"   Probability of >5% gain:   {s['prob_gain_5pct']:>6.1f}%")
        print(f"   Probability of >5% loss:   {s['prob_loss_5pct']:>6.1f}%")
        print(f"   Probability of >10% loss:  {s['prob_loss_10pct']:>6.1f}%")
        print(f"   Probability of >20% loss:  {s['prob_loss_20pct']:>6.1f}%")
        print(f"\n   Return Distribution:")
        print(f"     5th percentile:  {s['percentile_5_return_pct']:>8.2f}%")
        print(f"    25th percentile:  {s['percentile_25_return_pct']:>8.2f}%")
        print(f"    50th (median):    {s['percentile_50_return_pct']:>8.2f}%")
        print(f"    75th percentile:  {s['percentile_75_return_pct']:>8.2f}%")
        print(f"    95th percentile:  {s['percentile_95_return_pct']:>8.2f}%")
        print(f"\n   Equity Distribution:")
        print(f"     Worst case:    ${s['percentile_5_equity']:>12,.0f}")
        print(f"     Median:        ${s['median_final_equity']:>12,.0f}")
        print(f"     Best case:     ${s['percentile_95_equity']:>12,.0f}")
        print(f"\n   Drawdown Risk:")
        print(f"     Mean max DD:     {s['mean_max_drawdown_pct']:>8.2f}%")
        print(f"     Worst max DD:    {s['worst_max_drawdown_pct']:>8.2f}%")

    def save_results(self, output_dir: str = "data/results") -> str:
        """Save simulation results."""
        path = Path(output_dir) / "monte_carlo_results.csv"

        # Save summary stats
        stats_df = pd.DataFrame([self.statistics])
        stats_df.to_csv(path, index=False)

        # Save percentile paths for plotting
        paths_path = Path(output_dir) / "monte_carlo_paths.csv"
        percentile_paths = {}
        for pct in self.confidence_levels:
            percentile_paths[f'p{pct}'] = np.percentile(
                self.simulated_paths, pct, axis=0
            )
        percentile_paths['mean'] = np.mean(self.simulated_paths, axis=0)
        paths_df = pd.DataFrame(percentile_paths)
        paths_df.to_csv(paths_path, index=False)

        print(f"✅ Saved Monte Carlo results to {path}")
        return str(path)


# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("MONTE CARLO STRESS TEST")
    print("=" * 60)

    # Load backtest returns
    bt_path = Path("data/results/backtest_equity.csv")
    if not bt_path.exists():
        bt_path = Path("data/results/backtest_optimized_equity.csv")

    if not bt_path.exists():
        print("No backtest results found. Run Step 6 first.")
        exit(1)

    bt_df = pd.read_csv(bt_path, index_col=0, parse_dates=True)
    returns = bt_df['daily_return']

    mc = MonteCarloSimulator(n_simulations=5000, horizon_days=252)
    stats = mc.run(returns, initial_capital=1_000_000, verbose=True)
    mc.save_results()

    print("\n✅ Monte Carlo complete!")