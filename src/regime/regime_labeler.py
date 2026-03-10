"""
Regime Labeler
===============
Maps HMM numeric states (0, 1, 2) to meaningful regime names
(Bull, Bear, Crisis) based on the learned regime characteristics.

The HMM assigns arbitrary numbers to regimes. We need to map them
to interpretable labels by examining the learned emission parameters:
    - Bull: highest smoothed return mean, lowest volatility mean
    - Crisis: highest volatility mean
    - Bear: everything else (middle ground, or negative returns)
    
Also computes derived regime features useful for downstream modules:
    - Regime confidence (max probability)
    - Regime persistence (days in current regime)
    - Transition detection (regime change flags)
"""

import pandas as pd
import numpy as np
from pathlib import Path


class RegimeLabeler:
    """
    Labels HMM regimes with meaningful names and computes derived features.
    
    Parameters
    ----------
    regime_names : dict, optional
        Custom mapping {regime_int: name}. If None, auto-detected from
        regime characteristics (means of features).
    """
    
    def __init__(self, regime_names: dict = None):
        self.regime_names = regime_names
        self.regime_map = None
        self.labeled_df = None
        
    def auto_label_regimes(self, regime_means: np.ndarray, feature_names: list) -> dict:
        """
        Automatically assign Bull/Bear/Crisis labels based on regime means.
        
        Logic:
        1. Find the regime with highest realized_vol → Crisis
        2. Among remaining, highest smoothed_returns → Bull
        3. The last one → Bear
        
        Parameters
        ----------
        regime_means : np.ndarray
            Shape (n_regimes, n_features) — mean of each feature in each regime
        feature_names : list
            Names of features corresponding to columns of regime_means
            
        Returns
        -------
        dict
            Mapping {regime_int: regime_name}
        """
        n_regimes = regime_means.shape[0]
        
        # Get feature indices
        vol_idx = feature_names.index('realized_vol') if 'realized_vol' in feature_names else 1
        ret_idx = feature_names.index('smoothed_returns') if 'smoothed_returns' in feature_names else 0
        
        # Extract the key metrics per regime
        vol_means = regime_means[:, vol_idx]
        ret_means = regime_means[:, ret_idx]
        
        regime_map = {}
        remaining = list(range(n_regimes))
        
        # Step 1: Highest volatility → Crisis
        crisis_idx = remaining[np.argmax(vol_means[remaining])]
        regime_map[crisis_idx] = "Crisis"
        remaining.remove(crisis_idx)
        
        # Step 2: Among remaining, highest return → Bull
        if len(remaining) >= 2:
            bull_idx = remaining[np.argmax(ret_means[remaining])]
            regime_map[bull_idx] = "Bull"
            remaining.remove(bull_idx)
            
            # Step 3: Last one → Bear
            regime_map[remaining[0]] = "Bear"
        elif len(remaining) == 1:
            # Only 2 regimes case
            regime_map[remaining[0]] = "Bull" if ret_means[remaining[0]] > 0 else "Bear"
        
        self.regime_map = regime_map
        
        return regime_map
    
    def create_regime_dataframe(
        self,
        dates: pd.DatetimeIndex,
        hidden_states: np.ndarray,
        state_probs: np.ndarray,
        regime_map: dict = None
    ) -> pd.DataFrame:
        """
        Create a comprehensive regime DataFrame with labels and derived features.
        
        Parameters
        ----------
        dates : pd.DatetimeIndex
            Trading dates aligned with hidden_states
        hidden_states : np.ndarray
            Integer regime labels from HMM (shape: T,)
        state_probs : np.ndarray
            Regime probabilities from HMM (shape: T x n_regimes)
        regime_map : dict, optional
            Mapping {int: name}. If None, uses self.regime_map.
            
        Returns
        -------
        pd.DataFrame
            Columns: regime_id, regime_name, prob_bull, prob_bear, prob_crisis,
                     confidence, regime_persistence, regime_change
        """
        if regime_map is None:
            regime_map = self.regime_map
        
        if regime_map is None:
            raise ValueError("No regime map available. Call auto_label_regimes() first.")
        
        df = pd.DataFrame(index=dates)
        
        # Basic regime info
        df['regime_id'] = hidden_states
        df['regime_name'] = df['regime_id'].map(regime_map)
        
        # Probabilities for each named regime
        # Map probability columns to regime names
        for regime_id, regime_name in regime_map.items():
            col_name = f"prob_{regime_name.lower()}"
            df[col_name] = state_probs[:, regime_id]
        
        # === Derived Feature 1: Confidence ===
        # How sure is the model about the current regime?
        # High confidence = clearly in one regime
        # Low confidence = transitioning between regimes
        df['confidence'] = np.max(state_probs, axis=1)
        
        # === Derived Feature 2: Regime Persistence ===
        # How many consecutive days in the current regime?
        # Used downstream to filter out regime "flickers"
        df['regime_persistence'] = self._compute_persistence(hidden_states)
        
        # === Derived Feature 3: Regime Change Flag ===
        # Binary flag: did the regime change today?
        df['regime_change'] = (df['regime_id'] != df['regime_id'].shift(1)).astype(int)
        df.iloc[0, df.columns.get_loc('regime_change')] = 0  # First day is not a change
        
        # === Derived Feature 4: Dominant Regime (smoothed) ===
        # 5-day rolling mode to smooth out single-day flickers
        df['regime_smoothed'] = (
            df['regime_id']
            .rolling(window=5, min_periods=1)
            .apply(lambda x: pd.Series(x).mode().iloc[0], raw=False)
            .astype(int)
        )
        df['regime_name_smoothed'] = df['regime_smoothed'].map(regime_map)
        
        self.labeled_df = df
        return df
    
    def _compute_persistence(self, states: np.ndarray) -> np.ndarray:
        """
        Compute consecutive days in the same regime.
        
        Example: [0,0,0,1,1,0] → [1,2,3,1,2,1]
        """
        persistence = np.ones(len(states), dtype=int)
        for i in range(1, len(states)):
            if states[i] == states[i-1]:
                persistence[i] = persistence[i-1] + 1
            else:
                persistence[i] = 1
        return persistence
    
    def get_regime_summary(self, df: pd.DataFrame = None) -> pd.DataFrame:
        """
        Compute summary statistics for each regime.
        
        Returns
        -------
        pd.DataFrame
            Per-regime: count, percentage, avg_duration, avg_confidence
        """
        if df is None:
            df = self.labeled_df
            
        summary_rows = []
        for regime_name in df['regime_name'].unique():
            mask = df['regime_name'] == regime_name
            regime_data = df[mask]
            
            # Calculate average duration of continuous regime periods
            changes = df['regime_change'].copy()
            changes.iloc[0] = 1  # mark start
            period_starts = df.index[changes == 1]
            
            # Compute durations per regime period
            durations = []
            current_start = None
            for idx in range(len(df)):
                if idx == 0 or df['regime_id'].iloc[idx] != df['regime_id'].iloc[idx-1]:
                    if current_start is not None and df['regime_name'].iloc[idx-1] == regime_name:
                        durations.append(idx - current_start)
                    current_start = idx
            # Handle last period
            if current_start is not None and df['regime_name'].iloc[-1] == regime_name:
                durations.append(len(df) - current_start)
            
            avg_duration = np.mean(durations) if durations else 0
            
            summary_rows.append({
                'regime': regime_name,
                'days': len(regime_data),
                'pct': len(regime_data) / len(df) * 100,
                'avg_duration': avg_duration,
                'avg_confidence': regime_data['confidence'].mean(),
                'num_periods': len(durations)
            })
        
        return pd.DataFrame(summary_rows).sort_values('days', ascending=False)
    
    def save_regimes(self, output_dir: str = "data/results") -> str:
        """Save labeled regime data to CSV."""
        if self.labeled_df is None:
            raise ValueError("No labeled data. Call create_regime_dataframe() first.")
            
        path = Path(output_dir) / "regime_labels.csv"
        self.labeled_df.to_csv(path)
        print(f"✅ Saved regime labels to {path} ({len(self.labeled_df)} rows)")
        return str(path)


# ============================================================
# Standalone execution
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("REGIME LABELING")
    print("=" * 60)
    
    # Load features and HMM model
    import pickle
    
    model_path = Path("data/results/hmm_model.pkl")
    features_path = Path("data/processed/regime_features.csv")
    
    if not model_path.exists() or not features_path.exists():
        print("❌ HMM model or features not found. Run hmm_model.py first:")
        print("   python -m src.regime.hmm_model")
        exit(1)
    
    # Load
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    
    features_df = pd.read_csv(features_path, index_col=0, parse_dates=True)
    feature_names = list(features_df.columns)
    
    # Normalize and predict
    normalized = (features_df - features_df.mean()) / features_df.std()
    
    model = model_data['model']
    states = model.predict(normalized.values)
    probs = model.predict_proba(normalized.values)
    
    # Label
    labeler = RegimeLabeler()
    regime_map = labeler.auto_label_regimes(
        regime_means=model_data['regime_means'],
        feature_names=feature_names
    )
    
    print(f"\n🏷️ Regime Mapping: {regime_map}")
    
    df = labeler.create_regime_dataframe(
        dates=features_df.index,
        hidden_states=states,
        state_probs=probs,
        regime_map=regime_map
    )
    
    # Summary
    print("\n📊 Regime Summary:")
    summary = labeler.get_regime_summary()
    print(summary.to_string(index=False))
    
    # Sample of regime transitions
    changes = df[df['regime_change'] == 1].head(15)
    print(f"\n🔄 First 15 regime transitions:")
    print(changes[['regime_name', 'confidence', 'regime_persistence']].to_string())
    
    # Save
    labeler.save_regimes()
    
    print("\n🎯 Regime labeling complete!")