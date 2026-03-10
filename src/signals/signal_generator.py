"""
Regime-Adaptive Signal Generator
==================================
Generates trading signals (long/short/flat) based on z-scores,
with entry/exit thresholds that adapt to the current market regime.

Signal Logic:
    LONG spread  (buy Y, sell X): z-score drops below -entry_threshold
    SHORT spread (sell Y, buy X): z-score rises above +entry_threshold
    EXIT:                         z-score crosses back toward zero
    STOP LOSS:                    z-score exceeds stop_loss_threshold

Regime Adaptation:
    Bull:   Tight thresholds (1.5 entry) → confident mean-reversion
    Bear:   Medium thresholds (2.0 entry) → cautious trading
    Crisis: Wide thresholds (2.5 entry) → very conservative

State Machine:
    FLAT → LONG_SPREAD:   when z < -entry_threshold
    FLAT → SHORT_SPREAD:  when z > +entry_threshold
    LONG_SPREAD → FLAT:   when z > -exit_threshold (crossed toward 0)
    SHORT_SPREAD → FLAT:  when z < +exit_threshold (crossed toward 0)
    ANY → FLAT:           when |z| > stop_loss (regime break)
"""

import numpy as np
import pandas as pd
from pathlib import Path


class RegimeAdaptiveSignalGenerator:
    """
    Generates entry/exit signals adaptive to market regime.
    
    Parameters
    ----------
    regime_thresholds : dict
        Per-regime threshold configuration. Format:
        {
            'Bull':   {'entry': 1.5, 'exit': 0.5, 'stop_loss': 4.0},
            'Bear':   {'entry': 2.0, 'exit': 0.75, 'stop_loss': 3.5},
            'Crisis': {'entry': 2.5, 'exit': 0.5, 'stop_loss': 3.0},
        }
    min_holding_period : int
        Minimum days to hold a position before allowing exit.
        Prevents excessive churn from signal noise. Default: 5.
    min_regime_persistence : int
        Minimum days a regime must persist before we trust it.
        Prevents acting on 1-day regime flickers. Default: 3.
    crisis_trading : bool
        Whether to allow new entries during Crisis regime.
        Default: False (exit-only during crisis).
    """
    
    def __init__(
        self,
        regime_thresholds: dict = None,
        min_holding_period: int = 5,
        min_regime_persistence: int = 3,
        crisis_trading: bool = False
    ):
        self.regime_thresholds = regime_thresholds or {
            'Bull':   {'entry': 1.5, 'exit': 0.5, 'stop_loss': 4.0},
            'Bear':   {'entry': 2.0, 'exit': 0.75, 'stop_loss': 3.5},
            'Crisis': {'entry': 2.5, 'exit': 0.5, 'stop_loss': 3.0},
        }
        self.min_holding_period = min_holding_period
        self.min_regime_persistence = min_regime_persistence
        self.crisis_trading = crisis_trading
        
        # Default thresholds for unknown regimes
        self.default_thresholds = {'entry': 2.0, 'exit': 0.75, 'stop_loss': 3.5}
        
        # Results
        self.signals_df = None
        self.trade_log = []
        
    def generate_signals(
        self,
        spread_df: pd.DataFrame,
        regime_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Generate trading signals by combining z-scores with regime information.
        
        Parameters
        ----------
        spread_df : pd.DataFrame
            Output from SpreadCalculator. Must contain 'zscore' column.
        regime_df : pd.DataFrame
            Output from RegimeLabeler. Must contain 'regime_name' or 'regime_name_smoothed'.
            
        Returns
        -------
        pd.DataFrame
            Signal DataFrame with columns:
            - zscore, regime, entry_threshold, exit_threshold, stop_threshold
            - raw_signal (-1, 0, +1)
            - position (current position state)
            - signal_type (entry_long, entry_short, exit, stop_loss, hold)
        """
        # Align dates
        common_dates = spread_df.index.intersection(regime_df.index)
        if len(common_dates) == 0:
            raise ValueError("No overlapping dates between spread and regime data!")
        
        spread_aligned = spread_df.loc[common_dates].copy()
        regime_aligned = regime_df.loc[common_dates].copy()
        
        # Use smoothed regime to avoid flickers
        regime_col = 'regime_name_smoothed' if 'regime_name_smoothed' in regime_aligned.columns else 'regime_name'
        
        # Build result DataFrame
        result = pd.DataFrame(index=common_dates)
        result['zscore'] = spread_aligned['zscore']
        result['regime'] = regime_aligned[regime_col]
        result['confidence'] = regime_aligned['confidence'] if 'confidence' in regime_aligned.columns else 1.0
        
        if 'regime_persistence' in regime_aligned.columns:
            result['regime_persistence'] = regime_aligned['regime_persistence']
        else:
            result['regime_persistence'] = self.min_regime_persistence + 1
        
        # Get thresholds for each day based on regime
        result['entry_threshold'] = result['regime'].map(
            lambda r: self._get_threshold(r, 'entry')
        )
        result['exit_threshold'] = result['regime'].map(
            lambda r: self._get_threshold(r, 'exit')
        )
        result['stop_threshold'] = result['regime'].map(
            lambda r: self._get_threshold(r, 'stop_loss')
        )
        
        # Run the state machine
        positions, signal_types, trade_log = self._run_state_machine(result)
        
        result['position'] = positions
        result['signal_type'] = signal_types
        
        # Raw signal: just the entry direction indicator
        result['raw_signal'] = 0
        result.loc[result['signal_type'] == 'entry_long', 'raw_signal'] = 1
        result.loc[result['signal_type'] == 'entry_short', 'raw_signal'] = -1
        
        # Position changes (for transaction cost tracking)
        result['position_change'] = result['position'].diff().fillna(0).astype(int)
        
        self.signals_df = result
        self.trade_log = trade_log
        
        return result
    
    def _get_threshold(self, regime: str, threshold_type: str) -> float:
        """Get the appropriate threshold for a given regime."""
        if regime in self.regime_thresholds:
            return self.regime_thresholds[regime].get(
                threshold_type,
                self.default_thresholds[threshold_type]
            )
        return self.default_thresholds[threshold_type]
    
    def _run_state_machine(self, data: pd.DataFrame) -> tuple:
        """
        Run the position state machine day by day.
        
        States:
            0  = FLAT (no position)
            1  = LONG spread (long Y, short X)
            -1 = SHORT spread (short Y, long X)
            
        Returns
        -------
        tuple of (positions, signal_types, trade_log)
        """
        T = len(data)
        positions = np.zeros(T, dtype=int)
        signal_types = ['hold'] * T
        trade_log = []
        
        current_position = 0
        days_in_position = 0
        entry_zscore = 0.0
        entry_date = None
        
        for i in range(T):
            z = data['zscore'].iloc[i]
            regime = data['regime'].iloc[i]
            entry_thresh = data['entry_threshold'].iloc[i]
            exit_thresh = data['exit_threshold'].iloc[i]
            stop_thresh = data['stop_threshold'].iloc[i]
            regime_persist = data['regime_persistence'].iloc[i]
            
            # Skip if z-score is NaN
            if np.isnan(z):
                positions[i] = current_position
                signal_types[i] = 'hold'
                days_in_position += 1
                continue
            
            # === STOP LOSS CHECK (always active) ===
            if current_position != 0 and abs(z) > stop_thresh:
                trade_log.append({
                    'exit_date': data.index[i],
                    'entry_date': entry_date,
                    'direction': current_position,
                    'entry_z': entry_zscore,
                    'exit_z': z,
                    'exit_type': 'stop_loss',
                    'holding_days': days_in_position,
                    'regime': regime
                })
                current_position = 0
                days_in_position = 0
                signal_types[i] = 'stop_loss'
                positions[i] = 0
                continue
            
            # === EXIT CHECK ===
            if current_position != 0 and days_in_position >= self.min_holding_period:
                should_exit = False
                
                if current_position == 1:  # Long spread
                    # Exit when z-score rises back above -exit_threshold (toward 0)
                    should_exit = z > -exit_thresh
                    
                elif current_position == -1:  # Short spread
                    # Exit when z-score falls back below +exit_threshold (toward 0)
                    should_exit = z < exit_thresh
                
                if should_exit:
                    trade_log.append({
                        'exit_date': data.index[i],
                        'entry_date': entry_date,
                        'direction': current_position,
                        'entry_z': entry_zscore,
                        'exit_z': z,
                        'exit_type': 'signal_exit',
                        'holding_days': days_in_position,
                        'regime': regime
                    })
                    current_position = 0
                    days_in_position = 0
                    signal_types[i] = 'exit'
                    positions[i] = 0
                    continue
            
            # === ENTRY CHECK (only when flat) ===
            if current_position == 0:
                # Don't enter during Crisis if crisis_trading is off
                if regime == 'Crisis' and not self.crisis_trading:
                    positions[i] = 0
                    signal_types[i] = 'hold'
                    continue
                
                # Don't enter if regime is too new (might be a flicker)
                if regime_persist < self.min_regime_persistence:
                    positions[i] = 0
                    signal_types[i] = 'hold'
                    continue
                
                # Long spread: z-score is very negative (spread too narrow)
                if z < -entry_thresh:
                    current_position = 1
                    entry_zscore = z
                    entry_date = data.index[i]
                    days_in_position = 1
                    signal_types[i] = 'entry_long'
                    positions[i] = 1
                    continue
                
                # Short spread: z-score is very positive (spread too wide)
                elif z > entry_thresh:
                    current_position = -1
                    entry_zscore = z
                    entry_date = data.index[i]
                    days_in_position = 1
                    signal_types[i] = 'entry_short'
                    positions[i] = -1
                    continue
            
            # === DEFAULT: hold current position ===
            positions[i] = current_position
            signal_types[i] = 'hold'
            days_in_position += 1
        
        return positions, signal_types, trade_log
    
    def get_signal_summary(self, signals_df: pd.DataFrame = None) -> dict:
        """Compute summary statistics of generated signals."""
        if signals_df is None:
            signals_df = self.signals_df
        
        if signals_df is None:
            raise ValueError("No signals generated yet.")
        
        valid = signals_df.dropna(subset=['zscore'])
        
        total_days = len(valid)
        flat_days = (valid['position'] == 0).sum()
        long_days = (valid['position'] == 1).sum()
        short_days = (valid['position'] == -1).sum()
        
        n_entries = ((valid['signal_type'] == 'entry_long') | 
                     (valid['signal_type'] == 'entry_short')).sum()
        n_exits = (valid['signal_type'] == 'exit').sum()
        n_stops = (valid['signal_type'] == 'stop_loss').sum()
        
        # Per-regime breakdown
        regime_breakdown = {}
        for regime in valid['regime'].unique():
            mask = valid['regime'] == regime
            regime_data = valid[mask]
            regime_breakdown[regime] = {
                'days': len(regime_data),
                'trades': ((regime_data['signal_type'] == 'entry_long') | 
                          (regime_data['signal_type'] == 'entry_short')).sum(),
                'avg_zscore': regime_data['zscore'].mean(),
                'time_in_market': (regime_data['position'] != 0).mean() * 100
            }
        
        return {
            'total_days': total_days,
            'flat_pct': flat_days / total_days * 100,
            'long_pct': long_days / total_days * 100,
            'short_pct': short_days / total_days * 100,
            'total_entries': n_entries,
            'total_exits': n_exits,
            'total_stop_losses': n_stops,
            'total_trades': len(self.trade_log),
            'trades_per_year': len(self.trade_log) / (total_days / 252),
            'avg_holding_days': (
                np.mean([t['holding_days'] for t in self.trade_log])
                if self.trade_log else 0
            ),
            'regime_breakdown': regime_breakdown
        }
    
    def get_trade_log_df(self) -> pd.DataFrame:
        """Return trade log as DataFrame."""
        if not self.trade_log:
            return pd.DataFrame()
        return pd.DataFrame(self.trade_log)


# ============================================================
# Standalone execution
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("REGIME-ADAPTIVE SIGNAL GENERATOR TEST")
    print("=" * 60)
    
    from src.signals.kalman_filter import KalmanHedgeRatio
    from src.signals.spread_calculator import SpreadCalculator
    
    # Load data
    prices = pd.read_csv("data/processed/close_prices.csv", index_col=0, parse_dates=True)
    regime_df = pd.read_csv("data/results/regime_labels.csv", index_col=0, parse_dates=True)
    
    pair = ('EFA', 'VGK')
    if pair[0] not in prices.columns or pair[1] not in prices.columns:
        cols = list(prices.columns)
        pair = (cols[0], cols[1])
    
    print(f"\n📊 Generating signals for: {pair[0]}-{pair[1]}")
    
    # Kalman Filter
    kf = KalmanHedgeRatio(delta=1e-4)
    kalman_df = kf.fit_dataframe(prices, pair[0], pair[1])
    
    # Spread
    calc = SpreadCalculator(zscore_lookback=63)
    spread_df = calc.compute_from_kalman(kalman_df, prices, pair[0], pair[1])
    
    # Signals
    gen = RegimeAdaptiveSignalGenerator(crisis_trading=False)
    signals_df = gen.generate_signals(spread_df, regime_df)
    
    # Summary
    summary = gen.get_signal_summary()
    print(f"\n📈 Signal Summary:")
    print(f"   Total days: {summary['total_days']}")
    print(f"   Flat: {summary['flat_pct']:.1f}%  Long: {summary['long_pct']:.1f}%  Short: {summary['short_pct']:.1f}%")
    print(f"   Total trades: {summary['total_trades']}")
    print(f"   Trades/year: {summary['trades_per_year']:.1f}")
    print(f"   Avg holding: {summary['avg_holding_days']:.1f} days")
    print(f"   Stop losses: {summary['total_stop_losses']}")
    
    print(f"\n📊 Per-Regime Breakdown:")
    for regime, stats in summary['regime_breakdown'].items():
        print(f"   {regime:>8s}: {stats['days']} days, {stats['trades']} trades, "
              f"{stats['time_in_market']:.1f}% in market")
    
    # Trade log
    trades = gen.get_trade_log_df()
    if len(trades) > 0:
        print(f"\n📝 First 10 trades:")
        print(trades.head(10).to_string(index=False))
    
    print(f"\n✅ Signal generation complete!")