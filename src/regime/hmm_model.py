"""
Hidden Markov Model for Regime Detection
==========================================
Fits a Gaussian HMM to market features to detect latent regimes.

Mathematical Background:
    A Hidden Markov Model assumes:
    1. There exist K hidden states (regimes) S_t ∈ {0, 1, ..., K-1}
    2. State transitions follow a Markov chain with transition matrix A
       where A[i,j] = P(S_t = j | S_{t-1} = i)
    3. Each state emits observations from a Gaussian distribution:
       O_t | S_t = k ~ N(μ_k, Σ_k)
    
    The model is trained via Expectation-Maximization (Baum-Welch algorithm):
    - E-step: Compute posterior probabilities of each state given observations
    - M-step: Update (μ_k, Σ_k, A) to maximize expected log-likelihood
    
    After fitting:
    - Viterbi algorithm gives the most likely state sequence
    - Forward-backward algorithm gives state probabilities per timestep

Why "full" covariance?
    We use covariance_type="full" because our features (returns, vol, skew)
    are correlated with each other. A bull regime has (positive returns AND
    low vol AND positive skew) — these aren't independent.
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from hmmlearn.hmm import GaussianHMM
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


class RegimeHMM:
    """
    Gaussian Hidden Markov Model for market regime detection.
    
    Fits multiple random initializations and selects the best model
    by log-likelihood to avoid local optima.
    
    Parameters
    ----------
    n_regimes : int
        Number of hidden states/regimes (default: 3 for bull/bear/crisis)
    n_iterations : int
        Max EM iterations per fit (default: 200)
    n_fits : int
        Number of random restarts to avoid local optima (default: 25)
    covariance_type : str
        Type of covariance matrix. 'full' captures feature correlations.
    random_state_base : int
        Base random seed for reproducibility
    """
    
    def __init__(
        self,
        n_regimes: int = 3,
        n_iterations: int = 200,
        n_fits: int = 25,
        covariance_type: str = "full",
        random_state_base: int = 42
    ):
        self.n_regimes = n_regimes
        self.n_iterations = n_iterations
        self.n_fits = n_fits
        self.covariance_type = covariance_type
        self.random_state_base = random_state_base
        
        # Model state
        self.best_model = None
        self.best_score = -np.inf
        self.all_scores = []
        self.is_fitted = False
        
        # Results
        self.hidden_states = None
        self.state_probs = None
        self.regime_means = None
        self.regime_covars = None
        self.transition_matrix = None
        
    def fit(self, features: np.ndarray, verbose: bool = True) -> 'RegimeHMM':
        """
        Fit the HMM with multiple random restarts, keep the best.
        
        The EM algorithm for HMMs is sensitive to initialization —
        different starting points converge to different local optima.
        We run n_fits times and keep the model with highest log-likelihood.
        
        Parameters
        ----------
        features : np.ndarray
            Shape (T, n_features) — the observation matrix.
            Typically normalized features from RegimeFeatureEngineer.
        verbose : bool
            Whether to print progress.
            
        Returns
        -------
        self
            For method chaining
        """
        if verbose:
            print(f"\n🔄 Fitting HMM with {self.n_regimes} regimes, "
                  f"{self.n_fits} random restarts...")
            print(f"   Data shape: {features.shape}")
        
        self.best_model = None
        self.best_score = -np.inf
        self.all_scores = []
        converged_count = 0
        
        for i in range(self.n_fits):
            try:
                model = GaussianHMM(
                    n_components=self.n_regimes,
                    covariance_type=self.covariance_type,
                    n_iter=self.n_iterations,
                    random_state=self.random_state_base + i,
                    tol=1e-4,           # Convergence tolerance
                    verbose=False,
                    init_params="stmc",  # Initialize all params randomly
                    params="stmc"        # Train all params
                )
                
                model.fit(features)
                score = model.score(features)  # Log-likelihood
                self.all_scores.append(score)
                
                if model.monitor_.converged:
                    converged_count += 1
                
                if score > self.best_score:
                    self.best_score = score
                    self.best_model = model
                    
            except Exception as e:
                # Some initializations may fail — that's okay
                self.all_scores.append(np.nan)
                if verbose and i < 3:  # Only warn for first few
                    print(f"   ⚠️ Fit {i+1} failed: {str(e)[:50]}")
        
        if self.best_model is None:
            raise RuntimeError("All HMM fits failed! Check your feature data.")
        
        self.is_fitted = True
        
        # Extract model parameters
        self._extract_parameters()
        
        if verbose:
            valid_scores = [s for s in self.all_scores if not np.isnan(s)]
            print(f"\n✅ HMM fitting complete!")
            print(f"   Converged: {converged_count}/{self.n_fits}")
            print(f"   Best log-likelihood: {self.best_score:.2f}")
            print(f"   Score range: [{min(valid_scores):.2f}, {max(valid_scores):.2f}]")
        
        return self
    
    def _extract_parameters(self):
        """Extract and store key parameters from the fitted model."""
        self.regime_means = self.best_model.means_
        self.regime_covars = self.best_model.covars_
        self.transition_matrix = self.best_model.transmat_
        
    def predict(self, features: np.ndarray) -> tuple:
        """
        Predict regime states and probabilities for given features.
        
        Uses:
        - Viterbi algorithm for most likely state sequence
        - Forward-backward for state posterior probabilities
        
        Parameters
        ----------
        features : np.ndarray
            Shape (T, n_features)
            
        Returns
        -------
        tuple of (np.ndarray, np.ndarray)
            (hidden_states, state_probabilities)
            hidden_states: shape (T,), integer regime labels
            state_probabilities: shape (T, n_regimes), probability of each regime
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted yet. Call fit() first.")
        
        # Viterbi: most likely state sequence
        self.hidden_states = self.best_model.predict(features)
        
        # Forward-backward: posterior probabilities
        self.state_probs = self.best_model.predict_proba(features)
        
        return self.hidden_states, self.state_probs
    
    def get_regime_statistics(self, feature_names: list = None) -> dict:
        """
        Get interpretable statistics for each regime.
        
        Returns
        -------
        dict
            Keys: regime index, Values: dict with means, stds, and transition probs
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted yet.")
        
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(self.regime_means.shape[1])]
        
        stats = {}
        for k in range(self.n_regimes):
            regime_stats = {
                'means': dict(zip(feature_names, self.regime_means[k])),
                'stds': dict(zip(
                    feature_names,
                    np.sqrt(np.diag(self.regime_covars[k]))
                )),
                'stay_probability': self.transition_matrix[k, k],
                'expected_duration_days': 1.0 / (1.0 - self.transition_matrix[k, k] + 1e-10)
            }
            stats[k] = regime_stats
            
        return stats
    
    def print_regime_summary(self, feature_names: list = None):
        """Print a human-readable summary of each regime."""
        stats = self.get_regime_statistics(feature_names)
        
        print("\n" + "=" * 70)
        print("REGIME SUMMARY")
        print("=" * 70)
        
        for k, s in stats.items():
            print(f"\n--- Regime {k} ---")
            print(f"  Expected duration: {s['expected_duration_days']:.1f} days")
            print(f"  Stay probability:  {s['stay_probability']:.4f}")
            print(f"  Feature means:")
            for fname, val in s['means'].items():
                std = s['stds'][fname]
                print(f"    {fname:>20s}: {val:>8.4f}  (±{std:.4f})")
        
        print(f"\n  Transition Matrix:")
        print(f"  {np.array2string(self.transition_matrix, precision=4, suppress_small=True)}")
    
    def save_model(self, output_dir: str = "data/results") -> str:
        """Save the fitted HMM model to disk."""
        if not self.is_fitted:
            raise ValueError("Model not fitted yet.")
            
        path = Path(output_dir) / "hmm_model.pkl"
        with open(path, 'wb') as f:
            pickle.dump({
                'model': self.best_model,
                'best_score': self.best_score,
                'n_regimes': self.n_regimes,
                'regime_means': self.regime_means,
                'regime_covars': self.regime_covars,
                'transition_matrix': self.transition_matrix
            }, f)
        print(f"✅ Saved HMM model to {path}")
        return str(path)
    
    @classmethod
    def load_model(cls, model_path: str = "data/results/hmm_model.pkl") -> 'RegimeHMM':
        """Load a previously saved HMM model."""
        with open(model_path, 'rb') as f:
            data = pickle.load(f)
        
        hmm = cls(n_regimes=data['n_regimes'])
        hmm.best_model = data['model']
        hmm.best_score = data['best_score']
        hmm.regime_means = data['regime_means']
        hmm.regime_covars = data['regime_covars']
        hmm.transition_matrix = data['transition_matrix']
        hmm.is_fitted = True
        
        print(f"✅ Loaded HMM model ({hmm.n_regimes} regimes, score={hmm.best_score:.2f})")
        return hmm


# ============================================================
# Standalone execution: fit HMM on pre-computed features
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("HMM REGIME MODEL FITTING")
    print("=" * 60)
    
    # Load features
    features_path = Path("data/processed/regime_features.csv")
    if not features_path.exists():
        print("❌ Features not found. Run feature_engineer.py first:")
        print("   python -m src.regime.feature_engineer")
        exit(1)
    
    features_df = pd.read_csv(features_path, index_col=0, parse_dates=True)
    feature_names = list(features_df.columns)
    print(f"\n📊 Loaded features: {features_df.shape}")
    
    # Normalize
    normalized = (features_df - features_df.mean()) / features_df.std()
    features_array = normalized.values
    
    # Fit HMM
    hmm = RegimeHMM(n_regimes=3, n_fits=25)
    hmm.fit(features_array, verbose=True)
    
    # Predict
    states, probs = hmm.predict(features_array)
    
    # Summary
    hmm.print_regime_summary(feature_names)
    
    # State distribution
    print("\n📊 Regime Distribution:")
    unique, counts = np.unique(states, return_counts=True)
    for u, c in zip(unique, counts):
        pct = c / len(states) * 100
        print(f"   Regime {u}: {c:>5d} days ({pct:.1f}%)")
    
    # Save
    hmm.save_model()
    
    print("\n🎯 HMM fitting complete!")