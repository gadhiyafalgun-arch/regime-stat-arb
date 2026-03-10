"""
Kalman Filter for Dynamic Hedge Ratio Estimation
==================================================
Implements a linear-regression Kalman Filter where the state vector
is [intercept, hedge_ratio] and it evolves as a random walk.

Mathematical Detail:
    State: θ_t = [α_t, β_t]' (intercept + slope/hedge ratio)
    
    Predict step:
        θ̂_t|t-1 = θ̂_{t-1|t-1}          (state evolves as random walk)
        P_t|t-1  = P_{t-1|t-1} + Q       (covariance grows by process noise)
    
    Update step:
        e_t = y_t - H_t · θ̂_t|t-1       (innovation / prediction error)
        S_t = H_t · P_t|t-1 · H_t' + R   (innovation covariance)
        K_t = P_t|t-1 · H_t' · S_t⁻¹     (Kalman gain)
        θ̂_t|t = θ̂_t|t-1 + K_t · e_t     (state update)
        P_t|t = (I - K_t · H_t) · P_t|t-1 (covariance update)

    The Q/R ratio controls adaptiveness:
        High Q/R → hedge ratio reacts fast, but noisy
        Low  Q/R → hedge ratio is smooth, but slow to adapt
        
    We implement from scratch for full transparency (no filterpy dependency).
"""

import numpy as np
import pandas as pd
from pathlib import Path


class KalmanHedgeRatio:
    """
    Kalman Filter that dynamically estimates the hedge ratio between two assets.
    
    Models: y_t = α_t + β_t * x_t + ε_t
    Where α_t and β_t evolve as random walks.
    
    Parameters
    ----------
    delta : float
        Process noise scaling factor. Controls how fast the hedge ratio
        can change. Typical range: 1e-5 to 1e-3.
        - 1e-5: very smooth, slow adaptation
        - 1e-4: balanced (recommended default)
        - 1e-3: responsive but noisy
    obs_noise : float
        Initial observation noise variance. Gets updated adaptively.
    initial_state_mean : np.ndarray
        Initial estimate of [α, β]. Default [0, 1].
    initial_state_cov : float
        Initial uncertainty in state estimate.
    """
    
    def __init__(
        self,
        delta: float = 1e-4,
        obs_noise: float = 1e-3,
        initial_state_mean: np.ndarray = None,
        initial_state_cov: float = 1.0
    ):
        self.delta = delta
        self.obs_noise = obs_noise
        self.initial_state_mean = initial_state_mean if initial_state_mean is not None else np.array([0.0, 1.0])
        self.initial_state_cov = initial_state_cov
        
        # Results storage
        self.alphas = None       # Intercept time series
        self.betas = None        # Hedge ratio time series
        self.spreads = None      # Residual spread
        self.forecast_errors = None  # Innovation / prediction error
        self.state_covs = None   # State covariance trace (uncertainty)
        
    def fit(self, price_y: np.ndarray, price_x: np.ndarray) -> dict:
        """
        Run the Kalman Filter forward through the price series.
        
        Parameters
        ----------
        price_y : np.ndarray
            Dependent asset prices (shape: T,)
        price_x : np.ndarray
            Independent asset prices (shape: T,)
            
        Returns
        -------
        dict with keys:
            'alpha': intercept time series
            'beta': hedge ratio time series
            'spread': residual spread (y - alpha - beta*x)
            'forecast_error': prediction errors
            'state_cov_trace': uncertainty in estimates over time
        """
        T = len(price_y)
        
        if len(price_x) != T:
            raise ValueError(f"Price series length mismatch: y={T}, x={len(price_x)}")
        
        # === Initialize ===
        n_states = 2  # [alpha, beta]
        
        # State estimate: [α, β]
        theta = self.initial_state_mean.copy()
        
        # State covariance
        P = np.eye(n_states) * self.initial_state_cov
        
        # Process noise covariance (how much hedge ratio can change per step)
        Q = np.eye(n_states) * self.delta
        
        # Observation noise variance
        R = self.obs_noise
        
        # === Storage ===
        alphas = np.zeros(T)
        betas = np.zeros(T)
        spreads = np.zeros(T)
        forecast_errors = np.zeros(T)
        state_cov_traces = np.zeros(T)
        
        # === Run Kalman Filter ===
        for t in range(T):
            # Observation vector: H_t = [1, x_t]
            H = np.array([1.0, price_x[t]])
            
            # === PREDICT STEP ===
            # State prediction (random walk → no change in expectation)
            theta_pred = theta.copy()
            
            # Covariance prediction (grows by Q each step)
            P_pred = P + Q
            
            # === UPDATE STEP ===
            # Prediction of observation
            y_hat = H @ theta_pred
            
            # Innovation (prediction error)
            e = price_y[t] - y_hat
            
            # Innovation covariance
            S = H @ P_pred @ H.T + R
            
            # Kalman gain
            K = (P_pred @ H.T) / S  # 2x1 vector
            
            # State update
            theta = theta_pred + K * e
            
            # Covariance update (Joseph form for numerical stability)
            I_KH = np.eye(n_states) - np.outer(K, H)
            P = I_KH @ P_pred @ I_KH.T + np.outer(K, K) * R
            
            # === Store results ===
            alphas[t] = theta[0]
            betas[t] = theta[1]
            spreads[t] = e  # The forecast error IS the spread
            forecast_errors[t] = e
            state_cov_traces[t] = np.trace(P)
        
        # Store
        self.alphas = alphas
        self.betas = betas
        self.spreads = spreads
        self.forecast_errors = forecast_errors
        self.state_covs = state_cov_traces
        
        return {
            'alpha': alphas,
            'beta': betas,
            'spread': spreads,
            'forecast_error': forecast_errors,
            'state_cov_trace': state_cov_traces
        }
    
    def fit_dataframe(
        self,
        prices_df: pd.DataFrame,
        asset_y: str,
        asset_x: str
    ) -> pd.DataFrame:
        """
        Convenience method that works with DataFrame and returns DataFrame.
        
        Parameters
        ----------
        prices_df : pd.DataFrame
            DataFrame with price columns
        asset_y : str
            Column name for dependent asset
        asset_x : str
            Column name for independent asset
            
        Returns
        -------
        pd.DataFrame
            With columns: alpha, beta, spread, forecast_error, state_uncertainty
        """
        price_y = prices_df[asset_y].values
        price_x = prices_df[asset_x].values
        
        results = self.fit(price_y, price_x)
        
        result_df = pd.DataFrame({
            'alpha': results['alpha'],
            'beta': results['beta'],
            'spread': results['spread'],
            'forecast_error': results['forecast_error'],
            'state_uncertainty': results['state_cov_trace']
        }, index=prices_df.index)
        
        return result_df
    
    def get_diagnostics(self) -> dict:
        """
        Return diagnostic statistics for the Kalman Filter fit.
        
        Returns
        -------
        dict with diagnostic metrics
        """
        if self.betas is None:
            raise ValueError("Not fitted yet. Call fit() first.")
        
        return {
            'beta_mean': np.mean(self.betas),
            'beta_std': np.std(self.betas),
            'beta_min': np.min(self.betas),
            'beta_max': np.max(self.betas),
            'beta_range': np.max(self.betas) - np.min(self.betas),
            'spread_mean': np.mean(self.spreads),
            'spread_std': np.std(self.spreads),
            'spread_autocorr': np.corrcoef(self.spreads[1:], self.spreads[:-1])[0, 1],
            'final_uncertainty': self.state_covs[-1],
            'avg_uncertainty': np.mean(self.state_covs),
        }


# ============================================================
# Standalone execution
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("KALMAN FILTER HEDGE RATIO TEST")
    print("=" * 60)
    
    # Load prices
    prices_path = Path("data/processed/close_prices.csv")
    if not prices_path.exists():
        print("❌ Prices not found. Run Step 1 first.")
        exit(1)
    
    prices = pd.read_csv(prices_path, index_col=0, parse_dates=True)
    
    # Test with best pair from Step 2: EFA-VGK
    pair = ('EFA', 'VGK')
    if pair[0] not in prices.columns or pair[1] not in prices.columns:
        # Fallback to any available pair
        available = [c for c in prices.columns if c in ['XLV', 'XLB', 'XLP', 'XLU', 'XLI', 'KO']]
        if len(available) >= 2:
            pair = (available[0], available[1])
        else:
            print("❌ Insufficient assets in price data.")
            exit(1)
    
    print(f"\n📊 Testing Kalman Filter on pair: {pair[0]}-{pair[1]}")
    print(f"   Data: {len(prices)} trading days")
    
    # Fit Kalman Filter
    kf = KalmanHedgeRatio(delta=1e-4)
    result_df = kf.fit_dataframe(prices, pair[0], pair[1])
    
    # Diagnostics
    diag = kf.get_diagnostics()
    print(f"\n📈 Kalman Filter Diagnostics:")
    for key, val in diag.items():
        print(f"   {key:>25s}: {val:.6f}")
    
    # Show hedge ratio evolution
    print(f"\n📉 Hedge Ratio (β) over time:")
    print(f"   Start: {result_df['beta'].iloc[10]:.4f}")
    print(f"   Middle: {result_df['beta'].iloc[len(result_df)//2]:.4f}")
    print(f"   End: {result_df['beta'].iloc[-1]:.4f}")
    
    print(f"\n✅ Kalman Filter test complete!")