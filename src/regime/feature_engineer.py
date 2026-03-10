"""
Feature Engineering for Regime Detection
=========================================
Computes observable features from SPY price/returns data
that serve as inputs to the Hidden Markov Model.

Features:
- Smoothed returns (signal of drift/trend)
- Realized volatility (signal of risk regime)
- Volatility of volatility (signal of regime instability)
- Return skewness (signal of tail behavior)
"""

import pandas as pd
import numpy as np
from pathlib import Path


class RegimeFeatureEngineer:
    """
    Transforms raw SPY price data into features suitable for HMM regime detection.
    
    The key insight: we don't feed raw daily returns to the HMM (too noisy).
    Instead, we compute smoothed statistical features over rolling windows
    that capture the *character* of each regime.
    
    Parameters
    ----------
    returns_window : int
        Rolling window for smoothed returns (default: 21 = ~1 month)
    vol_window : int
        Rolling window for realized volatility (default: 21)
    vol_of_vol_window : int
        Rolling window for volatility of volatility (default: 63 = ~3 months)
    skew_window : int
        Rolling window for return skewness (default: 63)
    """
    
    def __init__(
        self,
        returns_window: int = 21,
        vol_window: int = 21,
        vol_of_vol_window: int = 63,
        skew_window: int = 63
    ):
        self.returns_window = returns_window
        self.vol_window = vol_window
        self.vol_of_vol_window = vol_of_vol_window
        self.skew_window = skew_window
        
        # Store computed features
        self.features_df = None
        self.feature_names = []
        
    def compute_features(self, prices: pd.Series, log_returns: pd.Series = None) -> pd.DataFrame:
        """
        Compute all regime features from SPY price series.
        
        Parameters
        ----------
        prices : pd.Series
            SPY daily close prices with DatetimeIndex
        log_returns : pd.Series, optional
            Pre-computed log returns. If None, computed from prices.
            
        Returns
        -------
        pd.DataFrame
            DataFrame with all features, NaN rows dropped.
            Columns: smoothed_returns, realized_vol, vol_of_vol, rolling_skew
        """
        if log_returns is None:
            log_returns = np.log(prices / prices.shift(1))
        
        features = pd.DataFrame(index=prices.index)
        
        # === Feature 1: Smoothed Returns ===
        # Rolling mean of returns captures the "drift" of the market
        # Bull regime: positive drift, Bear: negative drift
        features['smoothed_returns'] = (
            log_returns.rolling(window=self.returns_window, min_periods=self.returns_window)
            .mean()
        )
        
        # === Feature 2: Realized Volatility ===
        # Rolling std of returns — the classic risk measure
        # Bull: low vol, Bear: medium vol, Crisis: very high vol
        features['realized_vol'] = (
            log_returns.rolling(window=self.vol_window, min_periods=self.vol_window)
            .std()
        ) * np.sqrt(252)  # Annualize for interpretability
        
        # === Feature 3: Volatility of Volatility ===
        # How much is volatility itself changing?
        # Stable regimes: low vol-of-vol, Transition periods: high vol-of-vol
        daily_vol = log_returns.rolling(window=self.vol_window, min_periods=self.vol_window).std()
        features['vol_of_vol'] = (
            daily_vol.rolling(window=self.vol_of_vol_window, min_periods=self.vol_of_vol_window)
            .std()
        ) * np.sqrt(252)  # Annualize
        
        # === Feature 4: Rolling Skewness ===
        # Negative skew = more left-tail events (bear/crisis)
        # Positive skew = more right-tail events (recovery rallies)
        features['rolling_skew'] = (
            log_returns.rolling(window=self.skew_window, min_periods=self.skew_window)
            .skew()
        )
        
        # Drop NaN rows (from rolling windows)
        features = features.dropna()
        
        self.features_df = features
        self.feature_names = list(features.columns)
        
        return features
    
    def normalize_features(self, features: pd.DataFrame = None) -> pd.DataFrame:
        """
        Z-score normalize features for HMM stability.
        
        HMMs with Gaussian emissions work best when features are
        roughly standard normal. This prevents features with larger
        magnitudes from dominating the likelihood.
        
        Parameters
        ----------
        features : pd.DataFrame, optional
            If None, uses self.features_df
            
        Returns
        -------
        pd.DataFrame
            Normalized features (zero mean, unit variance)
        """
        if features is None:
            features = self.features_df
            
        if features is None:
            raise ValueError("No features computed yet. Call compute_features() first.")
        
        normalized = (features - features.mean()) / features.std()
        
        return normalized
    
    def get_feature_summary(self, features: pd.DataFrame = None) -> pd.DataFrame:
        """Print summary statistics of computed features."""
        if features is None:
            features = self.features_df
            
        summary = features.describe().round(6)
        return summary
    
    def save_features(self, output_dir: str = "data/processed") -> str:
        """Save computed features to CSV."""
        if self.features_df is None:
            raise ValueError("No features computed yet.")
            
        path = Path(output_dir) / "regime_features.csv"
        self.features_df.to_csv(path)
        print(f"✅ Saved regime features to {path} ({len(self.features_df)} rows)")
        return str(path)


def load_spy_data() -> tuple:
    """
    Load SPY data from the project's data files.
    
    Returns
    -------
    tuple of (pd.Series, pd.Series)
        (close_prices, log_returns) for SPY
    """
    prices_path = Path("data/processed/close_prices.csv")
    returns_path = Path("data/processed/log_returns.csv")
    
    if not prices_path.exists():
        raise FileNotFoundError(f"Close prices not found at {prices_path}. Run Step 1 first.")
    
    prices_df = pd.read_csv(prices_path, index_col=0, parse_dates=True)
    returns_df = pd.read_csv(returns_path, index_col=0, parse_dates=True)
    
    if 'SPY' not in prices_df.columns:
        raise ValueError("SPY not found in close prices. Check your data pipeline.")
    
    return prices_df['SPY'], returns_df['SPY']


# ============================================================
# Standalone execution
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("REGIME FEATURE ENGINEERING")
    print("=" * 60)
    
    # Load SPY data
    spy_prices, spy_returns = load_spy_data()
    print(f"\n📊 SPY data: {len(spy_prices)} trading days")
    print(f"   Period: {spy_prices.index[0].date()} to {spy_prices.index[-1].date()}")
    
    # Compute features
    engineer = RegimeFeatureEngineer()
    features = engineer.compute_features(spy_prices, spy_returns)
    
    print(f"\n✅ Computed {len(features.columns)} features over {len(features)} days")
    print(f"   Features: {list(features.columns)}")
    
    # Summary stats
    print("\n📈 Feature Summary Statistics:")
    print(engineer.get_feature_summary().to_string())
    
    # Normalized features
    normalized = engineer.normalize_features()
    print("\n📐 Normalized Feature Ranges:")
    for col in normalized.columns:
        print(f"   {col}: [{normalized[col].min():.2f}, {normalized[col].max():.2f}]")
    
    # Save
    engineer.save_features()
    
    print("\n🎯 Feature engineering complete! Ready for HMM fitting.")