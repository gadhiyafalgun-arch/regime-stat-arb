"""
Regime Detection Pipeline
===========================
Orchestrates the full regime detection workflow:
1. Feature engineering from SPY data
2. HMM fitting with multiple restarts
3. Regime labeling and derived features
4. Validation and sanity checks
5. Visualization of results

This is the main entry point for Step 3.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for servers
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
import json

from src.regime.feature_engineer import RegimeFeatureEngineer, load_spy_data
from src.regime.hmm_model import RegimeHMM
from src.regime.regime_labeler import RegimeLabeler


class RegimePipeline:
    """
    End-to-end regime detection pipeline.
    
    Usage:
        pipeline = RegimePipeline()
        pipeline.run()
    """
    
    def __init__(
        self,
        n_regimes: int = 3,
        n_fits: int = 25,
        output_dir: str = "data/results",
        plot_dir: str = "data/results/plots"
    ):
        self.n_regimes = n_regimes
        self.n_fits = n_fits
        self.output_dir = Path(output_dir)
        self.plot_dir = Path(plot_dir)
        
        # Create directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.plot_dir.mkdir(parents=True, exist_ok=True)
        
        # Components
        self.feature_engineer = RegimeFeatureEngineer()
        self.hmm = RegimeHMM(n_regimes=n_regimes, n_fits=n_fits)
        self.labeler = RegimeLabeler()
        
        # Data
        self.spy_prices = None
        self.spy_returns = None
        self.features = None
        self.normalized_features = None
        self.regime_df = None
        
    def run(self, verbose: bool = True) -> pd.DataFrame:
        """
        Execute the full regime detection pipeline.
        
        Returns
        -------
        pd.DataFrame
            Complete regime labels with probabilities and derived features
        """
        print("=" * 70)
        print("🚀 REGIME DETECTION PIPELINE")
        print("=" * 70)
        
        # Step 1: Load data
        self._load_data(verbose)
        
        # Step 2: Feature engineering
        self._compute_features(verbose)
        
        # Step 3: Fit HMM
        self._fit_hmm(verbose)
        
        # Step 4: Predict and label
        self._predict_and_label(verbose)
        
        # Step 5: Validate
        self._validate(verbose)
        
        # Step 6: Save everything
        self._save_all(verbose)
        
        # Step 7: Plot
        self._plot_results(verbose)
        
        print("\n" + "=" * 70)
        print("✅ REGIME DETECTION COMPLETE!")
        print("=" * 70)
        
        return self.regime_df
    
    def _load_data(self, verbose: bool):
        """Load SPY price and return data."""
        if verbose:
            print("\n📥 Step 1: Loading SPY data...")
            
        self.spy_prices, self.spy_returns = load_spy_data()
        
        if verbose:
            print(f"   Loaded {len(self.spy_prices)} trading days")
            print(f"   Period: {self.spy_prices.index[0].date()} → {self.spy_prices.index[-1].date()}")
            print(f"   Price range: ${self.spy_prices.min():.2f} - ${self.spy_prices.max():.2f}")
    
    def _compute_features(self, verbose: bool):
        """Compute and normalize features."""
        if verbose:
            print("\n🔧 Step 2: Computing features...")
            
        self.features = self.feature_engineer.compute_features(
            self.spy_prices, self.spy_returns
        )
        self.normalized_features = self.feature_engineer.normalize_features()
        
        if verbose:
            print(f"   Features: {list(self.features.columns)}")
            print(f"   Shape: {self.features.shape}")
            print(f"   Date range: {self.features.index[0].date()} → {self.features.index[-1].date()}")
    
    def _fit_hmm(self, verbose: bool):
        """Fit the Hidden Markov Model."""
        if verbose:
            print("\n🧠 Step 3: Fitting HMM...")
            
        self.hmm.fit(self.normalized_features.values, verbose=verbose)
    
    def _predict_and_label(self, verbose: bool):
        """Predict regimes and assign meaningful labels."""
        if verbose:
            print("\n🏷️ Step 4: Predicting and labeling regimes...")
        
        # Predict
        states, probs = self.hmm.predict(self.normalized_features.values)
        
        # Auto-label based on regime characteristics
        feature_names = list(self.features.columns)
        regime_map = self.labeler.auto_label_regimes(
            self.hmm.regime_means, feature_names
        )
        
        if verbose:
            print(f"   Regime mapping: {regime_map}")
        
        # Create comprehensive DataFrame
        self.regime_df = self.labeler.create_regime_dataframe(
            dates=self.features.index,
            hidden_states=states,
            state_probs=probs,
            regime_map=regime_map
        )
        
        if verbose:
            self.hmm.print_regime_summary(feature_names)
            print("\n📊 Regime Distribution:")
            summary = self.labeler.get_regime_summary()
            print(summary.to_string(index=False))
    
    def _validate(self, verbose: bool):
        """Run sanity checks on regime detection results."""
        if verbose:
            print("\n🔍 Step 5: Validation...")
        
        issues = []
        
        # Check 1: All regimes should be present
        unique_regimes = self.regime_df['regime_name'].unique()
        if len(unique_regimes) < self.n_regimes:
            issues.append(f"Only {len(unique_regimes)} of {self.n_regimes} regimes detected")
        
        # Check 2: No single regime should dominate excessively (>80%)
        for regime in unique_regimes:
            pct = (self.regime_df['regime_name'] == regime).mean() * 100
            if pct > 80:
                issues.append(f"Regime '{regime}' dominates at {pct:.1f}%")
        
        # Check 3: Crisis regime should appear during known crisis periods
        known_crises = {
            'COVID': ('2020-02-15', '2020-04-15'),
            'Late 2018': ('2018-10-01', '2018-12-31'),
        }
        
        for name, (start, end) in known_crises.items():
            mask = (self.regime_df.index >= start) & (self.regime_df.index <= end)
            if mask.sum() > 0:
                period_data = self.regime_df.loc[mask]
                crisis_pct = (period_data['regime_name'] == 'Crisis').mean() * 100
                bear_pct = (period_data['regime_name'] == 'Bear').mean() * 100
                stress_pct = crisis_pct + bear_pct
                
                if stress_pct < 30:
                    issues.append(
                        f"Known crisis '{name}' only {stress_pct:.0f}% Bear+Crisis "
                        f"(expected >30%)"
                    )
                elif verbose:
                    print(f"   ✓ {name}: {crisis_pct:.0f}% Crisis, {bear_pct:.0f}% Bear")
        
        # Check 4: Average confidence should be reasonable
        avg_conf = self.regime_df['confidence'].mean()
        if avg_conf < 0.5:
            issues.append(f"Low average confidence: {avg_conf:.3f} (expected >0.5)")
        elif verbose:
            print(f"   ✓ Average confidence: {avg_conf:.3f}")
        
        # Check 5: Regime shouldn't flicker too much
        changes_per_year = (
            self.regime_df['regime_change'].sum() / 
            (len(self.regime_df) / 252)
        )
        if changes_per_year > 100:
            issues.append(f"Too many regime changes: {changes_per_year:.0f}/year")
        elif verbose:
            print(f"   ✓ Regime changes: {changes_per_year:.1f}/year")
        
        if issues:
            print(f"\n   ⚠️ Validation warnings:")
            for issue in issues:
                print(f"      - {issue}")
        else:
            if verbose:
                print(f"   ✅ All validation checks passed!")
    
    def _save_all(self, verbose: bool):
        """Save all outputs."""
        if verbose:
            print("\n💾 Step 6: Saving results...")
        
        # Save features
        self.feature_engineer.save_features()
        
        # Save HMM model
        self.hmm.save_model(str(self.output_dir))
        
        # Save regime labels
        self.labeler.save_regimes(str(self.output_dir))
        
        # Save regime map as JSON for easy loading
        regime_map_path = self.output_dir / "regime_map.json"
        # Convert int keys to strings for JSON
        regime_map_json = {str(k): v for k, v in self.labeler.regime_map.items()}
        with open(regime_map_path, 'w') as f:
            json.dump(regime_map_json, f, indent=2)
        if verbose:
            print(f"✅ Saved regime map to {regime_map_path}")
        
        # Save summary stats
        summary = self.labeler.get_regime_summary()
        summary_path = self.output_dir / "regime_summary.csv"
        summary.to_csv(summary_path, index=False)
        if verbose:
            print(f"✅ Saved regime summary to {summary_path}")
    
    def _plot_results(self, verbose: bool):
        """Generate visualization plots."""
        if verbose:
            print("\n📊 Step 7: Generating plots...")
        
        try:
            self._plot_regime_timeline()
            self._plot_regime_probabilities()
            self._plot_feature_distributions()
            if verbose:
                print(f"   ✅ Plots saved to {self.plot_dir}/")
        except Exception as e:
            print(f"   ⚠️ Plotting failed (non-critical): {e}")
    
    def _plot_regime_timeline(self):
        """Plot SPY prices colored by regime."""
        fig, axes = plt.subplots(2, 1, figsize=(16, 10), 
                                  gridspec_kw={'height_ratios': [3, 1]})
        
        # Align SPY prices with regime dates
        common_dates = self.regime_df.index.intersection(self.spy_prices.index)
        prices_aligned = self.spy_prices.loc[common_dates]
        regime_aligned = self.regime_df.loc[common_dates]
        
        # Top plot: SPY price colored by regime
        ax1 = axes[0]
        colors = {'Bull': '#2ecc71', 'Bear': '#e74c3c', 'Crisis': '#9b59b6'}
        
        for regime_name, color in colors.items():
            mask = regime_aligned['regime_name'] == regime_name
            if mask.sum() > 0:
                ax1.scatter(
                    prices_aligned.index[mask],
                    prices_aligned.values[mask],
                    c=color, s=1, alpha=0.7, label=regime_name
                )
        
        ax1.set_title('SPY Price Colored by Detected Regime', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Price ($)')
        ax1.legend(markerscale=10, fontsize=11)
        ax1.grid(True, alpha=0.3)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax1.xaxis.set_major_locator(mdates.YearLocator())
        
        # Bottom plot: Regime confidence
        ax2 = axes[1]
        ax2.fill_between(
            regime_aligned.index,
            regime_aligned['confidence'],
            alpha=0.5, color='steelblue'
        )
        ax2.set_ylabel('Confidence')
        ax2.set_ylim(0, 1)
        ax2.set_title('Regime Confidence Over Time', fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax2.xaxis.set_major_locator(mdates.YearLocator())
        
        plt.tight_layout()
        plt.savefig(self.plot_dir / 'regime_timeline.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_regime_probabilities(self):
        """Plot stacked regime probabilities over time."""
        fig, ax = plt.subplots(figsize=(16, 5))
        
        prob_cols = [c for c in self.regime_df.columns if c.startswith('prob_')]
        colors = {'prob_bull': '#2ecc71', 'prob_bear': '#e74c3c', 'prob_crisis': '#9b59b6'}
        
        # Stack plot
        bottom = np.zeros(len(self.regime_df))
        for col in prob_cols:
            color = colors.get(col, '#95a5a6')
            label = col.replace('prob_', '').title()
            ax.fill_between(
                self.regime_df.index,
                bottom,
                bottom + self.regime_df[col].values,
                alpha=0.7,
                color=color,
                label=label
            )
            bottom += self.regime_df[col].values
        
        ax.set_title('Regime Probabilities Over Time (Stacked)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Probability')
        ax.set_ylim(0, 1)
        ax.legend(loc='upper right', fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.xaxis.set_major_locator(mdates.YearLocator())
        
        plt.tight_layout()
        plt.savefig(self.plot_dir / 'regime_probabilities.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_feature_distributions(self):
        """Plot feature distributions per regime."""
        feature_names = list(self.features.columns)
        n_features = len(feature_names)
        
        fig, axes = plt.subplots(1, n_features, figsize=(5 * n_features, 5))
        if n_features == 1:
            axes = [axes]
        
        colors = {'Bull': '#2ecc71', 'Bear': '#e74c3c', 'Crisis': '#9b59b6'}
        
        # Align features with regime labels
        common_idx = self.features.index.intersection(self.regime_df.index)
        features_aligned = self.features.loc[common_idx]
        regime_aligned = self.regime_df.loc[common_idx]
        
        for i, fname in enumerate(feature_names):
            ax = axes[i]
            for regime_name, color in colors.items():
                mask = regime_aligned['regime_name'] == regime_name
                if mask.sum() > 0:
                    data = features_aligned.loc[mask, fname]
                    ax.hist(data, bins=50, alpha=0.5, color=color, 
                           label=regime_name, density=True)
            
            ax.set_title(fname, fontsize=12, fontweight='bold')
            ax.set_ylabel('Density')
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
        
        plt.suptitle('Feature Distributions by Regime', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(self.plot_dir / 'feature_distributions.png', dpi=150, bbox_inches='tight')
        plt.close()


# ============================================================
# Main entry point
# ============================================================
if __name__ == "__main__":
    pipeline = RegimePipeline(n_regimes=3, n_fits=25)
    regime_df = pipeline.run(verbose=True)
    
    # Final summary
    print("\n" + "=" * 70)
    print("📋 FINAL OUTPUT FILES:")
    print("=" * 70)
    print(f"   data/processed/regime_features.csv    — 4 features × {len(pipeline.features)} days")
    print(f"   data/results/hmm_model.pkl            — Fitted HMM model")
    print(f"   data/results/regime_labels.csv         — {len(regime_df)} rows with labels + probs")
    print(f"   data/results/regime_map.json           — Regime name mapping")
    print(f"   data/results/regime_summary.csv        — Regime statistics")
    print(f"   data/results/plots/regime_timeline.png — Price + regime visualization")
    print(f"   data/results/plots/regime_probabilities.png")
    print(f"   data/results/plots/feature_distributions.png")
    print(f"\n🎯 Ready for Step 4: Signal Generation with Kalman Filter!")