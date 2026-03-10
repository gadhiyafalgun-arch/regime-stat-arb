"""
Value at Risk (VaR) Calculator
================================
Estimates potential losses at various confidence levels using:
1. Historical VaR: percentile of actual return distribution
2. Parametric VaR: assumes normal distribution (fast but less accurate)
3. Conditional VaR (CVaR / Expected Shortfall): average loss beyond VaR

VaR answers: "What's the worst loss I can expect on 95% of days?"
CVaR answers: "When things DO go bad, how bad on average?"

For stat-arb, CVaR is more useful because:
    - Pair spreads have fat tails (not normal)
    - The worst days matter more than the average day
    - Regime changes cause clustered losses
"""

import numpy as np
import pandas as pd
from pathlib import Path


class VaRCalculator:
    """
    Computes Value at Risk and Conditional VaR for trading strategies.
    
    Parameters
    ----------
    confidence_levels : list
        VaR confidence levels to compute. Default: [0.95, 0.99]
    lookback : int
        Rolling window for VaR estimation. Default: 252 (1 year).
    min_periods : int
        Minimum observations before computing VaR. Default: 63.
    """
    
    def __init__(
        self,
        confidence_levels: list = None,
        lookback: int = 252,
        min_periods: int = 63
    ):
        self.confidence_levels = confidence_levels or [0.95, 0.99]
        self.lookback = lookback
        self.min_periods = min_periods
        
    def compute_historical_var(
        self,
        returns: pd.Series,
        confidence: float = 0.95
    ) -> pd.Series:
        """
        Rolling historical VaR.
        
        VaR_α = -Percentile(returns, 1-α)
        
        Positive values indicate potential loss.
        
        Parameters
        ----------
        returns : pd.Series
            Daily P&L returns
        confidence : float
            Confidence level (e.g., 0.95 means 95% VaR)
            
        Returns
        -------
        pd.Series
            Rolling VaR values (positive = potential loss)
        """
        quantile = 1 - confidence
        
        var = returns.rolling(
            window=self.lookback, min_periods=self.min_periods
        ).quantile(quantile)
        
        return -var  # Convention: positive VaR means potential loss
    
    def compute_parametric_var(
        self,
        returns: pd.Series,
        confidence: float = 0.95
    ) -> pd.Series:
        """
        Rolling parametric (Gaussian) VaR.
        
        VaR_α = -(μ + z_α × σ)
        
        Where z_α is the standard normal quantile.
        """
        from scipy.stats import norm
        
        z = norm.ppf(1 - confidence)
        
        rolling_mean = returns.rolling(
            window=self.lookback, min_periods=self.min_periods
        ).mean()
        rolling_std = returns.rolling(
            window=self.lookback, min_periods=self.min_periods
        ).std()
        
        var = -(rolling_mean + z * rolling_std)
        return var
    
    def compute_cvar(
        self,
        returns: pd.Series,
        confidence: float = 0.95
    ) -> pd.Series:
        """
        Rolling Conditional VaR (Expected Shortfall).
        
        CVaR_α = -E[r | r < -VaR_α]
        
        Average loss on days worse than VaR.
        """
        quantile = 1 - confidence
        
        def cvar_func(window):
            threshold = np.percentile(window, quantile * 100)
            tail = window[window <= threshold]
            return -np.mean(tail) if len(tail) > 0 else np.nan
        
        cvar = returns.rolling(
            window=self.lookback, min_periods=self.min_periods
        ).apply(cvar_func, raw=True)
        
        return cvar
    
    def compute_all_metrics(
        self,
        returns: pd.Series,
        capital: float = 1_000_000
    ) -> pd.DataFrame:
        """
        Compute all VaR metrics at all confidence levels.
        
        Parameters
        ----------
        returns : pd.Series
            Daily fractional returns (e.g., 0.01 = 1%)
        capital : float
            Total capital for dollar VaR computation
            
        Returns
        -------
        pd.DataFrame
            All VaR metrics over time
        """
        result = pd.DataFrame(index=returns.index)
        result['daily_return'] = returns
        
        for conf in self.confidence_levels:
            pct = int(conf * 100)
            
            # Historical VaR (% terms)
            result[f'var_{pct}_hist'] = self.compute_historical_var(returns, conf)
            
            # Parametric VaR
            result[f'var_{pct}_param'] = self.compute_parametric_var(returns, conf)
            
            # CVaR
            result[f'cvar_{pct}'] = self.compute_cvar(returns, conf)
            
            # Dollar VaR
            result[f'var_{pct}_dollar'] = result[f'var_{pct}_hist'] * capital
            result[f'cvar_{pct}_dollar'] = result[f'cvar_{pct}'] * capital
        
        return result
    
    def get_current_risk_snapshot(
        self,
        returns: pd.Series,
        capital: float = 1_000_000
    ) -> dict:
        """
        Get the most recent risk metrics.
        
        Returns
        -------
        dict
            Current VaR, CVaR at all confidence levels
        """
        metrics = self.compute_all_metrics(returns, capital)
        latest = metrics.iloc[-1]
        
        snapshot = {}
        for conf in self.confidence_levels:
            pct = int(conf * 100)
            snapshot[f'VaR_{pct}%'] = latest.get(f'var_{pct}_hist', np.nan)
            snapshot[f'CVaR_{pct}%'] = latest.get(f'cvar_{pct}', np.nan)
            snapshot[f'VaR_{pct}%_$'] = latest.get(f'var_{pct}_dollar', np.nan)
            snapshot[f'CVaR_{pct}%_$'] = latest.get(f'cvar_{pct}_dollar', np.nan)
        
        return snapshot


# ============================================================
# Standalone test
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("VAR CALCULATOR TEST")
    print("=" * 60)
    
    # Generate synthetic returns for testing
    np.random.seed(42)
    dates = pd.date_range('2015-01-01', '2024-12-31', freq='B')
    returns = pd.Series(
        np.random.normal(0.0002, 0.01, len(dates)),
        index=dates,
        name='returns'
    )
    # Add a crisis period
    crisis_mask = (dates >= '2020-02-15') & (dates <= '2020-03-31')
    returns[crisis_mask] = np.random.normal(-0.005, 0.03, crisis_mask.sum())
    
    var_calc = VaRCalculator(confidence_levels=[0.95, 0.99])
    
    snapshot = var_calc.get_current_risk_snapshot(returns, capital=1_000_000)
    print(f"\n📈 Current Risk Snapshot:")
    for key, val in snapshot.items():
        if '$' in key:
            print(f"   {key:>15s}: ${val:>12,.2f}")
        else:
            print(f"   {key:>15s}: {val:>12.4%}")
    
    print(f"\n✅ VaR calculator test complete!")