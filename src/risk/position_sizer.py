"""
Position Sizing Engine
========================
Determines how much capital to allocate to each trade based on:
1. Spread volatility (inverse-vol weighting)
2. Current market regime (regime scalar)
3. Signal confidence (z-score magnitude)
4. Portfolio-level constraints (max exposure per pair, total)

Position Sizing Methods:
    1. Fixed Fractional: risk a fixed % of equity per trade
    2. Inverse Volatility: scale inversely with spread vol
    3. Kelly Criterion: optimal fraction based on win rate and payoff

We use Inverse Volatility as default because:
    - It automatically reduces size in volatile markets
    - Combined with regime scalars, it's doubly adaptive
    - It's the industry standard for stat-arb
"""

import numpy as np
import pandas as pd
from pathlib import Path


class PositionSizer:
    """
    Calculates position sizes adaptive to volatility and regime.
    
    Parameters
    ----------
    total_capital : float
        Total portfolio capital in dollars.
    max_pair_allocation : float
        Maximum fraction of capital allocated to a single pair.
        Default: 0.20 (20%).
    max_total_exposure : float
        Maximum total exposure as fraction of capital.
        Default: 0.80 (80% — keep 20% cash buffer).
    risk_per_trade : float
        Target annualized risk per trade as fraction of capital.
        Default: 0.02 (2%).
    vol_lookback : int
        Rolling window for spread volatility estimation.
        Default: 63 (~3 months).
    regime_scalars : dict
        Multiplier on position size per regime.
        Default: Bull=1.0, Bear=0.5, Crisis=0.25.
    """
    
    def __init__(
        self,
        total_capital: float = 1_000_000,
        max_pair_allocation: float = 0.20,
        max_total_exposure: float = 0.80,
        risk_per_trade: float = 0.02,
        vol_lookback: int = 63,
        regime_scalars: dict = None
    ):
        self.total_capital = total_capital
        self.max_pair_allocation = max_pair_allocation
        self.max_total_exposure = max_total_exposure
        self.risk_per_trade = risk_per_trade
        self.vol_lookback = vol_lookback
        self.regime_scalars = regime_scalars or {
            'Bull': 1.0,
            'Bear': 0.5,
            'Crisis': 0.25
        }
        self.default_regime_scalar = 0.5
        
    def compute_position_sizes(
        self,
        signals_df: pd.DataFrame,
        spread_series: pd.Series = None
    ) -> pd.DataFrame:
        """
        Compute dollar position sizes for each day.
        
        Parameters
        ----------
        signals_df : pd.DataFrame
            Output from SignalGenerator. Must have: 'position', 'regime', 'zscore'
        spread_series : pd.Series, optional
            Raw spread for volatility estimation. If None, extracted from signals_df.
            
        Returns
        -------
        pd.DataFrame
            Original signals_df plus columns:
            - spread_vol: rolling spread volatility
            - regime_scalar: regime-based multiplier
            - raw_size: unconstrained position size
            - position_size: final constrained position size (dollars)
            - position_weight: size as fraction of capital
        """
        result = signals_df.copy()
        
        # Get spread for volatility calculation
        if spread_series is None:
            if 'zscore' in result.columns:
                spread_series = result['zscore']
            else:
                raise ValueError("Need spread or zscore data for volatility estimation")
        
        # === Step 1: Estimate spread volatility ===
        spread_returns = spread_series.diff()
        result['spread_vol'] = spread_returns.rolling(
            window=self.vol_lookback, min_periods=21
        ).std() * np.sqrt(252)  # Annualize
        
        # Floor volatility to avoid division by zero or extreme sizes
        result['spread_vol'] = result['spread_vol'].clip(lower=0.01)
        
        # === Step 2: Regime scalar ===
        result['regime_scalar'] = result['regime'].map(
            lambda r: self.regime_scalars.get(r, self.default_regime_scalar)
        )
        
        # === Step 3: Inverse-vol position sizing ===
        # Size = (risk_budget / annualized_vol) * regime_scalar
        risk_budget = self.total_capital * self.risk_per_trade
        
        result['raw_size'] = (risk_budget / result['spread_vol']) * result['regime_scalar']
        
        # === Step 4: Apply constraints ===
        max_dollar = self.total_capital * self.max_pair_allocation
        result['position_size'] = result['raw_size'].clip(upper=max_dollar)
        
        # Zero size when flat
        result.loc[result['position'] == 0, 'position_size'] = 0.0
        
        # Signed position size (positive for long, negative for short)
        result['signed_size'] = result['position_size'] * result['position']
        
        # Weight as fraction of capital
        result['position_weight'] = result['position_size'] / self.total_capital
        
        return result
    
    def compute_portfolio_sizes(
        self,
        pair_sizes: dict
    ) -> dict:
        """
        Apply portfolio-level constraints across all pairs.
        
        If total exposure exceeds max_total_exposure, scale all positions down
        proportionally.
        
        Parameters
        ----------
        pair_sizes : dict
            pair_name → DataFrame with 'position_size' column
            
        Returns
        -------
        dict
            pair_name → DataFrame with adjusted 'position_size_adj' column
        """
        # Get all dates across all pairs
        all_dates = set()
        for df in pair_sizes.values():
            all_dates.update(df.index)
        all_dates = sorted(all_dates)
        
        # Build total exposure per day
        total_exposure = pd.Series(0.0, index=all_dates)
        for pair_name, df in pair_sizes.items():
            aligned = df['position_size'].reindex(all_dates, fill_value=0)
            total_exposure += aligned.abs()
        
        # Compute scaling factor where exposure exceeds limit
        max_dollar = self.total_capital * self.max_total_exposure
        scale_factor = pd.Series(1.0, index=all_dates)
        over_limit = total_exposure > max_dollar
        scale_factor[over_limit] = max_dollar / total_exposure[over_limit]
        
        # Apply scaling
        result = {}
        for pair_name, df in pair_sizes.items():
            df_adj = df.copy()
            aligned_scale = scale_factor.reindex(df.index, fill_value=1.0)
            df_adj['portfolio_scale'] = aligned_scale
            df_adj['position_size_adj'] = df_adj['position_size'] * aligned_scale
            df_adj['signed_size_adj'] = df_adj['signed_size'] * aligned_scale
            df_adj['position_weight_adj'] = df_adj['position_size_adj'] / self.total_capital
            result[pair_name] = df_adj
        
        return result
    
    def get_sizing_summary(self, sized_df: pd.DataFrame) -> dict:
        """Summary statistics for position sizing."""
        active = sized_df[sized_df['position'] != 0]
        
        if len(active) == 0:
            return {'status': 'no_active_positions'}
        
        return {
            'avg_position_size': active['position_size'].mean(),
            'max_position_size': active['position_size'].max(),
            'avg_weight': active['position_weight'].mean() * 100,
            'max_weight': active['position_weight'].max() * 100,
            'avg_regime_scalar': active['regime_scalar'].mean(),
            'avg_spread_vol': active['spread_vol'].mean(),
            'days_active': len(active),
            'days_total': len(sized_df),
            'utilization_pct': len(active) / len(sized_df) * 100
        }


# ============================================================
# Standalone test
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("POSITION SIZER TEST")
    print("=" * 60)
    
    # Load a signal file
    signals_dir = Path("data/results/signals")
    signal_files = list(signals_dir.glob("signals_*.csv"))
    
    if not signal_files:
        print("❌ No signal files found. Run Step 4 first.")
        exit(1)
    
    # Test with first available pair
    test_file = signal_files[0]
    pair_name = test_file.stem.replace("signals_", "")
    print(f"\n📊 Testing position sizing for: {pair_name}")
    
    signals_df = pd.read_csv(test_file, index_col=0, parse_dates=True)
    
    sizer = PositionSizer(total_capital=1_000_000)
    sized_df = sizer.compute_position_sizes(signals_df)
    
    summary = sizer.get_sizing_summary(sized_df)
    print(f"\n📈 Position Sizing Summary:")
    for key, val in summary.items():
        if isinstance(val, float):
            print(f"   {key:>25s}: {val:,.2f}")
        else:
            print(f"   {key:>25s}: {val}")
    
    print(f"\n✅ Position sizing test complete!")