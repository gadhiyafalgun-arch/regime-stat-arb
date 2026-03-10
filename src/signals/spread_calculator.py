"""
Spread and Z-Score Calculator
================================
Computes the trading spread from Kalman Filter output and normalizes
it into z-scores using rolling statistics.

The z-score is the primary signal driver:
    z_t = (spread_t - μ_spread) / σ_spread

Where μ and σ are computed over a rolling lookback window
to avoid look-ahead bias.

Z-score interpretation:
    z > +2.0  → spread is unusually wide → short the spread
    z < -2.0  → spread is unusually narrow → long the spread
    z ≈  0.0  → spread is at equilibrium → no action / exit
"""

import numpy as np
import pandas as pd
from pathlib import Path


class SpreadCalculator:
    """
    Computes spread and z-scores from Kalman Filter output.
    
    Parameters
    ----------
    zscore_lookback : int
        Rolling window for z-score mean/std calculation.
        Default: 63 (~3 months). Shorter = more responsive, longer = more stable.
    min_periods : int
        Minimum periods before computing z-score.
        Default: 21 (~1 month). Prevents unreliable early estimates.
    halflife : int or None
        If set, uses exponentially-weighted mean/std with this halflife
        instead of simple rolling. Gives more weight to recent data.
        Default: None (use simple rolling).
    """
    
    def __init__(
        self,
        zscore_lookback: int = 63,
        min_periods: int = 21,
        halflife: int = None
    ):
        self.zscore_lookback = zscore_lookback
        self.min_periods = min_periods
        self.halflife = halflife
        
        self.spread_df = None
        
    def compute_zscore(self, spread: pd.Series) -> pd.DataFrame:
        """
        Compute z-score of the spread using rolling statistics.
        
        Parameters
        ----------
        spread : pd.Series
            Raw spread from Kalman Filter (y - alpha - beta*x)
            
        Returns
        -------
        pd.DataFrame
            Columns: spread, spread_mean, spread_std, zscore
        """
        result = pd.DataFrame(index=spread.index)
        result['spread'] = spread
        
        if self.halflife is not None:
            # Exponentially-weighted statistics
            result['spread_mean'] = spread.ewm(
                halflife=self.halflife, min_periods=self.min_periods
            ).mean()
            result['spread_std'] = spread.ewm(
                halflife=self.halflife, min_periods=self.min_periods
            ).std()
        else:
            # Simple rolling statistics
            result['spread_mean'] = spread.rolling(
                window=self.zscore_lookback, min_periods=self.min_periods
            ).mean()
            result['spread_std'] = spread.rolling(
                window=self.zscore_lookback, min_periods=self.min_periods
            ).std()
        
        # Z-score: how many standard deviations from the rolling mean
        result['zscore'] = (
            (result['spread'] - result['spread_mean']) / 
            result['spread_std'].replace(0, np.nan)
        )
        
        # Clip extreme z-scores (beyond ±6 is likely data error)
        result['zscore'] = result['zscore'].clip(-6, 6)
        
        self.spread_df = result
        return result
    
    def compute_from_kalman(
        self,
        kalman_df: pd.DataFrame,
        prices_df: pd.DataFrame,
        asset_y: str,
        asset_x: str
    ) -> pd.DataFrame:
        """
        Full computation: Kalman output → spread → z-score.
        
        Also computes the "clean" spread as:
            spread = price_y - alpha - beta * price_x
        
        Parameters
        ----------
        kalman_df : pd.DataFrame
            Output from KalmanHedgeRatio.fit_dataframe()
        prices_df : pd.DataFrame
            Raw price data
        asset_y, asset_x : str
            Asset column names
            
        Returns
        -------
        pd.DataFrame
            Complete spread analysis with all intermediate values
        """
        result = pd.DataFrame(index=kalman_df.index)
        
        # Prices
        result['price_y'] = prices_df[asset_y]
        result['price_x'] = prices_df[asset_x]
        
        # Kalman parameters
        result['alpha'] = kalman_df['alpha']
        result['beta'] = kalman_df['beta']
        result['state_uncertainty'] = kalman_df['state_uncertainty']
        
        # Spread: residual after removing the estimated linear relationship
        result['spread'] = (
            result['price_y'] - result['alpha'] - result['beta'] * result['price_x']
        )
        
        # Z-score
        zscore_result = self.compute_zscore(result['spread'])
        result['spread_mean'] = zscore_result['spread_mean']
        result['spread_std'] = zscore_result['spread_std']
        result['zscore'] = zscore_result['zscore']
        
        # Additional useful metrics
        result['zscore_abs'] = result['zscore'].abs()
        
        # Rate of z-score change (momentum of spread)
        result['zscore_velocity'] = result['zscore'].diff()
        
        self.spread_df = result
        return result
    
    def get_spread_diagnostics(self, result: pd.DataFrame = None) -> dict:
        """Diagnostic statistics for the spread."""
        if result is None:
            result = self.spread_df
        
        valid = result['zscore'].dropna()
        
        return {
            'spread_mean': result['spread'].mean(),
            'spread_std': result['spread'].std(),
            'zscore_mean': valid.mean(),
            'zscore_std': valid.std(),
            'zscore_skew': valid.skew(),
            'zscore_kurtosis': valid.kurtosis(),
            'pct_above_2': (valid.abs() > 2.0).mean() * 100,
            'pct_above_3': (valid.abs() > 3.0).mean() * 100,
            'mean_reversion_halflife': self._estimate_halflife(result['spread'].dropna()),
            'valid_observations': len(valid),
            'nan_observations': result['zscore'].isna().sum()
        }
    
    def _estimate_halflife(self, spread: pd.Series) -> float:
        """
        Estimate mean-reversion halflife using the Ornstein-Uhlenbeck model.
        
        The spread follows: ds = -θ(s - μ)dt + σdW
        Halflife = ln(2) / θ
        
        We estimate θ from: Δs_t = a + b * s_{t-1} + ε_t
        where θ = -b, halflife = -ln(2) / b
        """
        try:
            spread_lag = spread.shift(1)
            spread_diff = spread.diff()
            
            # Remove NaN
            valid = pd.DataFrame({'diff': spread_diff, 'lag': spread_lag}).dropna()
            
            if len(valid) < 30:
                return np.nan
            
            # OLS: diff = a + b * lag
            X = np.column_stack([np.ones(len(valid)), valid['lag'].values])
            y = valid['diff'].values
            
            beta = np.linalg.lstsq(X, y, rcond=None)[0]
            b = beta[1]
            
            if b >= 0:
                return np.nan  # Not mean-reverting
            
            halflife = -np.log(2) / b
            return max(halflife, 0.5)  # Floor at 0.5 days
            
        except Exception:
            return np.nan


# ============================================================
# Standalone execution
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("SPREAD & Z-SCORE CALCULATOR TEST")
    print("=" * 60)
    
    from src.signals.kalman_filter import KalmanHedgeRatio
    
    prices = pd.read_csv("data/processed/close_prices.csv", index_col=0, parse_dates=True)
    
    # Test pair
    pair = ('EFA', 'VGK')
    if pair[0] not in prices.columns or pair[1] not in prices.columns:
        cols = list(prices.columns)
        pair = (cols[0], cols[1])
    
    print(f"\n📊 Testing on pair: {pair[0]}-{pair[1]}")
    
    # Kalman Filter
    kf = KalmanHedgeRatio(delta=1e-4)
    kalman_df = kf.fit_dataframe(prices, pair[0], pair[1])
    
    # Spread & Z-score
    calc = SpreadCalculator(zscore_lookback=63)
    spread_df = calc.compute_from_kalman(kalman_df, prices, pair[0], pair[1])
    
    # Diagnostics
    diag = calc.get_spread_diagnostics()
    print(f"\n📈 Spread Diagnostics:")
    for key, val in diag.items():
        if isinstance(val, float):
            print(f"   {key:>30s}: {val:.4f}")
        else:
            print(f"   {key:>30s}: {val}")
    
    # Z-score distribution
    valid_z = spread_df['zscore'].dropna()
    print(f"\n📊 Z-Score Distribution:")
    for threshold in [1.0, 1.5, 2.0, 2.5, 3.0]:
        pct = (valid_z.abs() > threshold).mean() * 100
        print(f"   |z| > {threshold}: {pct:.1f}% of days")
    
    print(f"\n✅ Spread calculation complete!")