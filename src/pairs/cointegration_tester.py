"""
Cointegration Tester Module
Tests if asset pairs have a statistically significant long-term relationship

Physics Analogy:
- Two coupled oscillators that wander independently
- But their DIFFERENCE oscillates around equilibrium
- The "restoring force" = our trading edge
"""

import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller, coint
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
import os
import warnings
warnings.filterwarnings('ignore')


class CointegrationTester:
    """
    Tests pairs for cointegration using:
    1. Engle-Granger two-step method
    2. Augmented Dickey-Fuller test on spread
    
    Also calculates:
    - Hedge ratio (beta)
    - Half-life of mean reversion
    - Spread statistics
    
    Usage:
        ct = CointegrationTester()
        results = ct.test_all_pairs(close_prices, filtered_pairs)
    """
    
    def __init__(self):
        self.results_path = "data/results/"
        os.makedirs(self.results_path, exist_ok=True)
    
    def engle_granger_test(self, price_a, price_b):
        """
        Engle-Granger Two-Step Cointegration Test.
        
        Step 1: Regress Price_A = alpha + beta * Price_B + residuals
        Step 2: Test if residuals are stationary (ADF test)
        
        If residuals ARE stationary → prices are cointegrated
        The residuals = the SPREAD we will trade
        
        Args:
            price_a (pd.Series): Price series of asset A
            price_b (pd.Series): Price series of asset B
            
        Returns:
            dict: Test results including p-value, hedge ratio, etc.
        """
        # Step 1: OLS Regression
        # Price_A = alpha + beta * Price_B + epsilon
        X = add_constant(price_b.values)
        model = OLS(price_a.values, X).fit()
        
        alpha = model.params[0]       # intercept
        beta = model.params[1]        # hedge ratio
        residuals = model.resid       # spread
        
        # Step 2: ADF test on residuals
        adf_result = adfuller(residuals, maxlag=20, autolag='AIC')
        adf_stat = adf_result[0]
        p_value = adf_result[1]
        critical_values = adf_result[4]
        
        # Also use statsmodels coint function (more robust)
        coint_stat, coint_pvalue, coint_crit = coint(price_a, price_b)
        
        return {
            'alpha': alpha,
            'beta': beta,
            'residuals': residuals,
            'adf_statistic': adf_stat,
            'adf_pvalue': p_value,
            'adf_critical_1pct': critical_values['1%'],
            'adf_critical_5pct': critical_values['5%'],
            'adf_critical_10pct': critical_values['10%'],
            'coint_statistic': coint_stat,
            'coint_pvalue': coint_pvalue,
            'r_squared': model.rsquared,
        }
    
    def calculate_half_life(self, spread):
        """
        Calculate half-life of mean reversion.
        
        Uses AR(1) model: spread(t) = theta * spread(t-1) + noise
        Half-life = -ln(2) / ln(theta)
        
        Physics Analogy:
        - Like radioactive decay half-life
        - How long for spread to revert HALFWAY to its mean
        - Shorter = faster mean reversion = better for trading
        
        Args:
            spread (np.array): Spread (residuals) series
            
        Returns:
            float: Half-life in trading days
        """
        spread_series = pd.Series(spread)
        
        # AR(1): spread(t) = theta * spread(t-1) + noise
        spread_lag = spread_series.shift(1).dropna()
        spread_diff = spread_series.diff().dropna()
        
        # Align
        spread_lag = spread_lag.iloc[1:]
        spread_diff = spread_diff.iloc[1:]
        
        # Regress: delta_spread = phi * spread_lag + noise
        # theta = 1 + phi
        X = add_constant(spread_lag.values)
        model = OLS(spread_diff.values, X).fit()
        
        phi = model.params[1]
        
        if phi >= 0:
            # Not mean reverting
            return np.inf
        
        theta = 1 + phi
        
        if theta <= 0 or theta >= 1:
            return np.inf
        
        half_life = -np.log(2) / np.log(theta)
        
        return half_life
    
    def calculate_spread_stats(self, spread):
        """
        Calculate statistics of the spread.
        
        Args:
            spread (np.array): Spread series
            
        Returns:
            dict: Spread statistics
        """
        return {
            'spread_mean': np.mean(spread),
            'spread_std': np.std(spread),
            'spread_skew': pd.Series(spread).skew(),
            'spread_kurtosis': pd.Series(spread).kurtosis(),
            'spread_min': np.min(spread),
            'spread_max': np.max(spread),
            'zero_crossings': np.sum(np.diff(np.sign(spread - np.mean(spread))) != 0),
        }
    
    def test_single_pair(self, close_prices, ticker_a, ticker_b):
        """
        Run full cointegration analysis on a single pair.
        
        Args:
            close_prices (pd.DataFrame): Close prices
            ticker_a (str): First asset
            ticker_b (str): Second asset
            
        Returns:
            dict: Complete test results
        """
        price_a = close_prices[ticker_a]
        price_b = close_prices[ticker_b]
        
        # Run Engle-Granger test
        eg_results = self.engle_granger_test(price_a, price_b)
        
        # Calculate half-life
        half_life = self.calculate_half_life(eg_results['residuals'])
        
        # Calculate spread stats
        spread_stats = self.calculate_spread_stats(eg_results['residuals'])
        
        # Determine if cointegrated
        is_cointegrated = eg_results['coint_pvalue'] < 0.05
        
        # Determine if tradeable (half-life in reasonable range)
        is_tradeable = 5 <= half_life <= 120
        
        result = {
            'Ticker_A': ticker_a,
            'Ticker_B': ticker_b,
            'Cointegrated': is_cointegrated,
            'Coint_PValue': round(eg_results['coint_pvalue'], 6),
            'ADF_PValue': round(eg_results['adf_pvalue'], 6),
            'Hedge_Ratio': round(eg_results['beta'], 4),
            'Intercept': round(eg_results['alpha'], 4),
            'Half_Life': round(half_life, 1),
            'Is_Tradeable': is_tradeable,
            'R_Squared': round(eg_results['r_squared'], 4),
            'Spread_Mean': round(spread_stats['spread_mean'], 4),
            'Spread_Std': round(spread_stats['spread_std'], 4),
            'Zero_Crossings': spread_stats['zero_crossings'],
        }
        
        return result
    
    def test_all_pairs(self, close_prices, filtered_pairs):
        """
        Test all correlation-filtered pairs for cointegration.
        
        Args:
            close_prices (pd.DataFrame): Close prices
            filtered_pairs (pd.DataFrame): From correlation filter
            
        Returns:
            pd.DataFrame: All results sorted by quality
        """
        print("=" * 60)
        print("🧪 COINTEGRATION TESTING")
        print(f"   Testing {len(filtered_pairs)} pairs")
        print("=" * 60)
        
        results = []
        
        for idx, row in filtered_pairs.iterrows():
            ticker_a = row['Ticker_A']
            ticker_b = row['Ticker_B']
            
            try:
                result = self.test_single_pair(close_prices, ticker_a, ticker_b)
                result['Correlation'] = row['Correlation']
                results.append(result)
            except Exception as e:
                print(f"  ❌ {ticker_a}-{ticker_b}: Error - {e}")
        
        results_df = pd.DataFrame(results)
        
        # Separate cointegrated pairs
        cointegrated = results_df[results_df['Cointegrated'] == True].copy()
        tradeable = cointegrated[cointegrated['Is_Tradeable'] == True].copy()
        
        # Sort by p-value (lower = stronger evidence)
        if len(tradeable) > 0:
            tradeable = tradeable.sort_values('Coint_PValue').reset_index(drop=True)
        
        if len(cointegrated) > 0:
            cointegrated = cointegrated.sort_values('Coint_PValue').reset_index(drop=True)
        
        # Summary
        print(f"\n📊 RESULTS SUMMARY:")
        print(f"   Total pairs tested:     {len(results_df)}")
        print(f"   Cointegrated (p<0.05):  {len(cointegrated)}")
        print(f"   Tradeable (good HL):    {len(tradeable)}")
        
        if len(tradeable) > 0:
            print(f"\n📋 TOP TRADEABLE PAIRS:")
            display_cols = [
                'Ticker_A', 'Ticker_B', 'Coint_PValue', 
                'Hedge_Ratio', 'Half_Life', 'Correlation', 'Zero_Crossings'
            ]
            print(tradeable[display_cols].head(15).to_string(index=False))
        
        if len(tradeable) == 0 and len(cointegrated) > 0:
            print(f"\n⚠️  Cointegrated pairs exist but half-lives are outside tradeable range")
            print(f"   Consider adjusting half-life bounds")
            display_cols = [
                'Ticker_A', 'Ticker_B', 'Coint_PValue',
                'Hedge_Ratio', 'Half_Life', 'Correlation'
            ]
            print(cointegrated[display_cols].head(10).to_string(index=False))
        
        return {
            'all_results': results_df,
            'cointegrated': cointegrated,
            'tradeable': tradeable
        }


# === RUN DIRECTLY ===
if __name__ == "__main__":
    # Load data
    close_prices = pd.read_csv(
        "data/processed/close_prices.csv",
        index_col=0,
        parse_dates=True
    )
    
    # Load correlation filtered pairs
    filtered_pairs = pd.read_csv("data/results/correlation_filtered_pairs.csv")
    
    print(f"📂 Loaded {len(filtered_pairs)} correlation-filtered pairs\n")
    
    # Run cointegration tests
    ct = CointegrationTester()
    results = ct.test_all_pairs(close_prices, filtered_pairs)
    
    # Save results
    results['all_results'].to_csv("data/results/cointegration_all.csv", index=False)
    results['cointegrated'].to_csv("data/results/cointegration_passed.csv", index=False)
    results['tradeable'].to_csv("data/results/cointegration_tradeable.csv", index=False)
    
    print(f"\n💾 Results saved to data/results/")
    print("✅ Cointegration testing complete!")
