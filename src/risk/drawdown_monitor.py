"""
Drawdown Monitor & Circuit Breakers
======================================
Tracks portfolio drawdowns and implements automatic shutdown rules.

Drawdown = peak-to-trough decline in portfolio value.

Circuit Breaker Logic:
    Level 1 (Warning):    Drawdown > 5%   → Reduce position sizes by 50%
    Level 2 (Defensive):  Drawdown > 10%  → Close all positions, stop new entries
    Level 3 (Shutdown):   Drawdown > 15%  → Full shutdown, require manual restart

Why circuit breakers?
    - Stat-arb assumes mean reversion. If the spread diverges beyond
      historical bounds, the relationship may have broken.
    - Cascading losses during regime changes can be catastrophic.
    - Better to stop and reassess than ride a losing position to zero.
"""

import numpy as np
import pandas as pd
from pathlib import Path


class DrawdownMonitor:
    """
    Monitors portfolio equity curve for drawdowns and triggers circuit breakers.
    
    Parameters
    ----------
    warning_threshold : float
        Drawdown level for Level 1 warning. Default: 0.05 (5%).
    defensive_threshold : float
        Drawdown level for Level 2 defensive mode. Default: 0.10 (10%).
    shutdown_threshold : float
        Drawdown level for Level 3 shutdown. Default: 0.15 (15%).
    recovery_threshold : float
        Drawdown must recover to this level before resuming from defensive.
        Default: 0.03 (3%).
    """
    
    def __init__(
        self,
        warning_threshold: float = 0.05,
        defensive_threshold: float = 0.10,
        shutdown_threshold: float = 0.15,
        recovery_threshold: float = 0.03
    ):
        self.warning_threshold = warning_threshold
        self.defensive_threshold = defensive_threshold
        self.shutdown_threshold = shutdown_threshold
        self.recovery_threshold = recovery_threshold
        
        self.drawdown_df = None
        
    def compute_drawdowns(self, equity_curve: pd.Series) -> pd.DataFrame:
        """
        Compute drawdown series from equity curve.
        
        Parameters
        ----------
        equity_curve : pd.Series
            Portfolio value over time (starting from initial capital).
            
        Returns
        -------
        pd.DataFrame
            Columns: equity, peak, drawdown, drawdown_pct, drawdown_duration,
                     circuit_breaker_level, position_scalar
        """
        result = pd.DataFrame(index=equity_curve.index)
        result['equity'] = equity_curve
        
        # Running peak (high-water mark)
        result['peak'] = equity_curve.cummax()
        
        # Drawdown in dollars
        result['drawdown_dollar'] = result['equity'] - result['peak']
        
        # Drawdown as percentage of peak
        result['drawdown_pct'] = result['drawdown_dollar'] / result['peak']
        
        # Drawdown duration (consecutive days in drawdown)
        result['in_drawdown'] = (result['drawdown_pct'] < -1e-6).astype(int)
        result['drawdown_duration'] = self._compute_drawdown_duration(
            result['in_drawdown'].values
        )
        
        # Circuit breaker levels and position scalars
        levels, scalars = self._compute_circuit_breakers(result['drawdown_pct'].values)
        result['circuit_breaker_level'] = levels
        result['position_scalar'] = scalars
        
        self.drawdown_df = result
        return result
    
    def _compute_drawdown_duration(self, in_drawdown: np.ndarray) -> np.ndarray:
        """Compute consecutive days in drawdown."""
        duration = np.zeros(len(in_drawdown), dtype=int)
        for i in range(1, len(in_drawdown)):
            if in_drawdown[i]:
                duration[i] = duration[i-1] + 1
            else:
                duration[i] = 0
        return duration
    
    def _compute_circuit_breakers(self, drawdown_pct: np.ndarray) -> tuple:
        """
        Compute circuit breaker levels with hysteresis.
        
        Hysteresis means we don't immediately resume when drawdown
        slightly improves — we wait for meaningful recovery.
        
        Returns
        -------
        tuple of (levels, scalars)
            levels: array of 0, 1, 2, 3
            scalars: array of position size multipliers
        """
        T = len(drawdown_pct)
        levels = np.zeros(T, dtype=int)
        scalars = np.ones(T, dtype=float)
        
        current_level = 0
        
        for i in range(T):
            dd = abs(drawdown_pct[i])  # Make positive for comparison
            
            # Check for escalation
            if dd >= self.shutdown_threshold:
                current_level = 3
            elif dd >= self.defensive_threshold:
                current_level = max(current_level, 2)
            elif dd >= self.warning_threshold:
                current_level = max(current_level, 1)
            
            # Check for de-escalation (with hysteresis)
            if current_level >= 2 and dd < self.recovery_threshold:
                current_level = 0  # Full recovery
            elif current_level == 1 and dd < self.recovery_threshold:
                current_level = 0
            
            levels[i] = current_level
            
            # Position scalar based on level
            if current_level == 0:
                scalars[i] = 1.0
            elif current_level == 1:
                scalars[i] = 0.5
            elif current_level == 2:
                scalars[i] = 0.0  # No new positions
            elif current_level == 3:
                scalars[i] = 0.0  # Full shutdown
        
        return levels, scalars
    
    def get_drawdown_summary(self, drawdown_df: pd.DataFrame = None) -> dict:
        """Compute summary statistics of drawdowns."""
        if drawdown_df is None:
            drawdown_df = self.drawdown_df
        
        if drawdown_df is None:
            raise ValueError("No drawdown data. Call compute_drawdowns() first.")
        
        dd = drawdown_df['drawdown_pct']
        
        # Find individual drawdown periods
        drawdown_periods = self._find_drawdown_periods(drawdown_df)
        
        return {
            'max_drawdown_pct': dd.min() * 100,  # Most negative
            'max_drawdown_dollar': drawdown_df['drawdown_dollar'].min(),
            'avg_drawdown_pct': dd[dd < -1e-6].mean() * 100 if (dd < -1e-6).any() else 0,
            'max_drawdown_duration': drawdown_df['drawdown_duration'].max(),
            'time_in_drawdown_pct': (dd < -1e-6).mean() * 100,
            'num_drawdown_periods': len(drawdown_periods),
            'circuit_breaker_triggers': {
                'level_1': (drawdown_df['circuit_breaker_level'] == 1).sum(),
                'level_2': (drawdown_df['circuit_breaker_level'] == 2).sum(),
                'level_3': (drawdown_df['circuit_breaker_level'] == 3).sum(),
            },
            'top_5_drawdowns': drawdown_periods[:5] if drawdown_periods else []
        }
    
    def _find_drawdown_periods(self, drawdown_df: pd.DataFrame) -> list:
        """Identify distinct drawdown periods with their statistics."""
        dd = drawdown_df['drawdown_pct']
        in_dd = (dd < -1e-6)
        
        periods = []
        start = None
        
        for i in range(len(dd)):
            if in_dd.iloc[i] and start is None:
                start = i
            elif not in_dd.iloc[i] and start is not None:
                # End of drawdown period
                period_dd = dd.iloc[start:i]
                periods.append({
                    'start': dd.index[start],
                    'end': dd.index[i-1],
                    'trough_date': period_dd.idxmin(),
                    'max_dd_pct': period_dd.min() * 100,
                    'duration_days': i - start
                })
                start = None
        
        # Handle ongoing drawdown
        if start is not None:
            period_dd = dd.iloc[start:]
            periods.append({
                'start': dd.index[start],
                'end': dd.index[-1],
                'trough_date': period_dd.idxmin(),
                'max_dd_pct': period_dd.min() * 100,
                'duration_days': len(dd) - start
            })
        
        # Sort by severity
        periods.sort(key=lambda x: x['max_dd_pct'])
        return periods


# ============================================================
# Standalone test
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("DRAWDOWN MONITOR TEST")
    print("=" * 60)
    
    # Synthetic equity curve
    np.random.seed(42)
    dates = pd.date_range('2015-01-01', '2024-12-31', freq='B')
    returns = np.random.normal(0.0003, 0.008, len(dates))
    
    # Add drawdowns
    crisis = (dates >= '2020-02-15') & (dates <= '2020-03-31')
    returns[crisis] = np.random.normal(-0.008, 0.02, crisis.sum())
    
    equity = 1_000_000 * np.cumprod(1 + returns)
    equity_series = pd.Series(equity, index=dates)
    
    monitor = DrawdownMonitor()
    dd_df = monitor.compute_drawdowns(equity_series)
    
    summary = monitor.get_drawdown_summary()
    print(f"\n📈 Drawdown Summary:")
    print(f"   Max drawdown: {summary['max_drawdown_pct']:.2f}%")
    print(f"   Max duration: {summary['max_drawdown_duration']} days")
    print(f"   Time in drawdown: {summary['time_in_drawdown_pct']:.1f}%")
    print(f"   Circuit breaker triggers: {summary['circuit_breaker_triggers']}")
    
    print(f"\n   Top 5 drawdowns:")
    for p in summary['top_5_drawdowns']:
        print(f"     {p['start'].date()} to {p['end'].date()}: "
              f"{p['max_dd_pct']:.2f}%, {p['duration_days']} days")
    
    print(f"\n✅ Drawdown monitor test complete!")