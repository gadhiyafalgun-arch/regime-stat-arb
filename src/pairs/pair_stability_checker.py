"""
Pair Stability Checker Module
Tests if cointegration relationship is STABLE over time

Why this matters:
- A pair might be cointegrated over FULL period
- But if relationship breaks down frequently → dangerous to trade
- We need pairs that are CONSISTENTLY cointegrated

Physics Analogy:
- Like checking if a coupled oscillator maintains coupling
- If coupling constant changes → system becomes unpredictable
"""

import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import coint
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings('ignore')


class PairStabilityChecker:
    """
    Tests cointegration stability using rolling windows.
    
    For each pair:
    - Slide a window across time
    - Test cointegration at each window position
    - Calculate % of windows where pair is cointegrated
    - Track how hedge ratio changes over time
    
    Usage:
        psc = PairStabilityChecker()
        stable_pairs = psc.check_all_pairs(close_prices, tradeable_pairs)
    """
    
    def __init__(self, window_size=252, step_size=63, min_stability=0.6):
        """
        Args:
            window_size (int): Rolling window in trading days (252 = 1 year)
            step_size (int): Step forward size (63 = 1 quarter)
            min_stability (float): Minimum % of windows that must be cointegrated
        """
        self.window_size = window_size
        self.step_size = step_size
        self.min_stability = min_stability
        self.results_path = "data/results/"
        os.makedirs(self.results_path, exist_ok=True)
    
    def rolling_cointegration_test(self, price_a, price_b):
        """
        Test cointegration on rolling windows.
        
        Args:
            price_a (pd.Series): Price series A
            price_b (pd.Series): Price series B
            
        Returns:
            dict: Rolling test results
        """
        n = len(price_a)
        
        # Generate window start positions
        starts = range(0, n - self.window_size, self.step_size)
        
        window_results = []
        
        for start in starts:
            end = start + self.window_size
            
            # Extract window
            window_a = price_a.iloc[start:end]
            window_b = price_b.iloc[start:end]
            
            try:
                # Cointegration test on this window
                coint_stat, p_value, _ = coint(window_a, window_b)
                
                # Hedge ratio on this window
                X = add_constant(window_b.values)
                model = OLS(window_a.values, X).fit()
                hedge_ratio = model.params[1]
                
                window_results.append({
                    'window_start': price_a.index[start],
                    'window_end': price_a.index[end - 1],
                    'p_value': p_value,
                    'is_cointegrated': p_value < 0.05,
                    'hedge_ratio': hedge_ratio,
                    'coint_statistic': coint_stat,
                })
            except Exception:
                continue
        
        return pd.DataFrame(window_results)
    
    def calculate_stability_score(self, window_results):
        """
        Calculate overall stability score for a pair.
        
        Score combines:
        - % of windows cointegrated
        - Consistency of hedge ratio
        - Average p-value strength
        
        Args:
            window_results (pd.DataFrame): From rolling test
            
        Returns:
            dict: Stability metrics
        """
        if len(window_results) == 0:
            return {
                'stability_pct': 0,
                'avg_pvalue': 1.0,
                'hedge_ratio_mean': 0,
                'hedge_ratio_std': np.inf,
                'hedge_ratio_cv': np.inf,
                'n_windows': 0,
                'stability_score': 0,
            }
        
        n_windows = len(window_results)
        n_cointegrated = window_results['is_cointegrated'].sum()
        stability_pct = n_cointegrated / n_windows
        
        avg_pvalue = window_results['p_value'].mean()
        
        hr_mean = window_results['hedge_ratio'].mean()
        hr_std = window_results['hedge_ratio'].std()
        hr_cv = hr_std / abs(hr_mean) if hr_mean != 0 else np.inf
        
        # Composite stability score (0 to 1)
        # Higher = more stable
        score_coint = stability_pct
        score_pvalue = max(0, 1 - avg_pvalue)
        score_hedge = max(0, 1 - hr_cv) if hr_cv != np.inf else 0
        
        # Weighted combination
        stability_score = (
            0.50 * score_coint +
            0.25 * score_pvalue +
            0.25 * score_hedge
        )
        
        return {
            'stability_pct': round(stability_pct * 100, 1),
            'avg_pvalue': round(avg_pvalue, 4),
            'hedge_ratio_mean': round(hr_mean, 4),
            'hedge_ratio_std': round(hr_std, 4),
            'hedge_ratio_cv': round(hr_cv, 4),
            'n_windows': n_windows,
            'stability_score': round(stability_score, 4),
        }
    
    def check_single_pair(self, close_prices, ticker_a, ticker_b):
        """
        Full stability check for one pair.
        
        Args:
            close_prices (pd.DataFrame): Close prices
            ticker_a (str): First asset
            ticker_b (str): Second asset
            
        Returns:
            dict: Stability results
        """
        price_a = close_prices[ticker_a]
        price_b = close_prices[ticker_b]
        
        # Run rolling cointegration
        window_results = self.rolling_cointegration_test(price_a, price_b)
        
        # Calculate stability score
        stability = self.calculate_stability_score(window_results)
        
        result = {
            'Ticker_A': ticker_a,
            'Ticker_B': ticker_b,
            **stability,
            'is_stable': stability['stability_pct'] >= self.min_stability * 100,
        }
        
        return result, window_results
    
    def check_all_pairs(self, close_prices, tradeable_pairs):
        """
        Check stability for all tradeable pairs.
        
        Args:
            close_prices (pd.DataFrame): Close prices
            tradeable_pairs (pd.DataFrame): From cointegration tester
            
        Returns:
            pd.DataFrame: Pairs with stability scores
        """
        print("=" * 60)
        print("🔄 PAIR STABILITY ANALYSIS")
        print(f"   Window: {self.window_size} days | Step: {self.step_size} days")
        print(f"   Min stability: {self.min_stability * 100}%")
        print(f"   Testing {len(tradeable_pairs)} pairs")
        print("=" * 60)
        
        all_results = []
        all_window_data = {}
        
        for idx, row in tradeable_pairs.iterrows():
            ticker_a = row['Ticker_A']
            ticker_b = row['Ticker_B']
            
            try:
                result, window_results = self.check_single_pair(
                    close_prices, ticker_a, ticker_b
                )
                
                # Carry forward previous metrics
                result['Coint_PValue'] = row.get('Coint_PValue', None)
                result['Half_Life'] = row.get('Half_Life', None)
                result['Correlation'] = row.get('Correlation', None)
                
                all_results.append(result)
                all_window_data[f"{ticker_a}_{ticker_b}"] = window_results
                
                status = "✅" if result['is_stable'] else "⚠️"
                print(f"  {status} {ticker_a}-{ticker_b}: "
                      f"stability={result['stability_pct']}% | "
                      f"HR={result['hedge_ratio_mean']} ± {result['hedge_ratio_std']} | "
                      f"score={result['stability_score']}")
                
            except Exception as e:
                print(f"  ❌ {ticker_a}-{ticker_b}: Error - {e}")
        
        results_df = pd.DataFrame(all_results)
        
        # Separate stable pairs
        stable_pairs = results_df[results_df['is_stable'] == True].copy()
        
        if len(stable_pairs) > 0:
            stable_pairs = stable_pairs.sort_values(
                'stability_score', ascending=False
            ).reset_index(drop=True)
        
        # Summary
        print(f"\n{'=' * 60}")
        print(f"📊 STABILITY RESULTS:")
        print(f"   Total tested:  {len(results_df)}")
        print(f"   Stable pairs:  {len(stable_pairs)}")
        print(f"   Unstable:      {len(results_df) - len(stable_pairs)}")
        
        if len(stable_pairs) > 0:
            print(f"\n📋 STABLE PAIRS (sorted by stability score):")
            display_cols = [
                'Ticker_A', 'Ticker_B', 'stability_score',
                'stability_pct', 'hedge_ratio_mean', 'hedge_ratio_cv',
                'Half_Life', 'Coint_PValue'
            ]
            available_cols = [c for c in display_cols if c in stable_pairs.columns]
            print(stable_pairs[available_cols].to_string(index=False))
        
        print(f"{'=' * 60}")
        
        return {
            'all_results': results_df,
            'stable_pairs': stable_pairs,
            'window_data': all_window_data
        }
    
    def plot_stability(self, close_prices, stable_pairs, window_data, n_pairs=4, save=True):
        """
        Plot stability analysis for top pairs.
        
        For each pair shows:
        - Price series (normalized)
        - Rolling p-value
        - Rolling hedge ratio
        
        Args:
            close_prices (pd.DataFrame): Close prices
            stable_pairs (pd.DataFrame): Stable pairs
            window_data (dict): Rolling window results
            n_pairs (int): Number of pairs to plot
            save (bool): Save to file
        """
        n_pairs = min(n_pairs, len(stable_pairs))
        
        if n_pairs == 0:
            print("⚠️  No stable pairs to plot")
            return
        
        fig, axes = plt.subplots(n_pairs, 3, figsize=(20, 5 * n_pairs))
        if n_pairs == 1:
            axes = axes.reshape(1, -1)
        
        for i in range(n_pairs):
            row = stable_pairs.iloc[i]
            ticker_a = row['Ticker_A']
            ticker_b = row['Ticker_B']
            key = f"{ticker_a}_{ticker_b}"
            
            wd = window_data.get(key, pd.DataFrame())
            
            # Plot 1: Normalized Prices
            ax1 = axes[i][0]
            pa = close_prices[ticker_a] / close_prices[ticker_a].iloc[0] * 100
            pb = close_prices[ticker_b] / close_prices[ticker_b].iloc[0] * 100
            ax1.plot(pa, label=ticker_a, linewidth=1)
            ax1.plot(pb, label=ticker_b, linewidth=1)
            ax1.set_title(f'{ticker_a} vs {ticker_b} - Prices', fontweight='bold')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            if len(wd) > 0:
                # Plot 2: Rolling P-Value
                ax2 = axes[i][1]
                ax2.plot(wd['window_end'], wd['p_value'], 'b-', linewidth=1)
                ax2.axhline(y=0.05, color='r', linestyle='--', label='5% threshold')
                ax2.fill_between(
                    wd['window_end'],
                    wd['p_value'],
                    0.05,
                    where=wd['p_value'] < 0.05,
                    alpha=0.3,
                    color='green',
                    label='Cointegrated'
                )
                ax2.set_title(f'Rolling Cointegration P-Value', fontweight='bold')
                ax2.set_ylabel('P-Value')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
                
                # Plot 3: Rolling Hedge Ratio
                ax3 = axes[i][2]
                ax3.plot(wd['window_end'], wd['hedge_ratio'], 'g-', linewidth=1)
                ax3.axhline(
                    y=wd['hedge_ratio'].mean(),
                    color='r', linestyle='--',
                    label=f'Mean: {wd["hedge_ratio"].mean():.3f}'
                )
                ax3.fill_between(
                    wd['window_end'],
                    wd['hedge_ratio'].mean() - wd['hedge_ratio'].std(),
                    wd['hedge_ratio'].mean() + wd['hedge_ratio'].std(),
                    alpha=0.2, color='green'
                )
                ax3.set_title(f'Rolling Hedge Ratio', fontweight='bold')
                ax3.set_ylabel('Hedge Ratio (β)')
                ax3.legend()
                ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            filepath = os.path.join(self.results_path, "pair_stability_analysis.png")
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            print(f"\n📊 Stability charts saved: {filepath}")
        
        plt.close()


# === RUN DIRECTLY ===
if __name__ == "__main__":
    # Load data
    close_prices = pd.read_csv(
        "data/processed/close_prices.csv",
        index_col=0,
        parse_dates=True
    )
    
    # Load tradeable pairs from cointegration test
    tradeable_path = "data/results/cointegration_tradeable.csv"
    
    if not os.path.exists(tradeable_path):
        print("❌ No tradeable pairs file found!")
        print("   Run cointegration_tester.py first")
        exit()
    
    tradeable_pairs = pd.read_csv(tradeable_path)
    print(f"📂 Loaded {len(tradeable_pairs)} tradeable pairs\n")
    
    # Run stability check
    psc = PairStabilityChecker(
        window_size=252,    # 1 year windows
        step_size=63,       # step every quarter
        min_stability=0.6   # must be cointegrated 60% of windows
    )
    
    results = psc.check_all_pairs(close_prices, tradeable_pairs)
    
    # Save results
    results['all_results'].to_csv("data/results/stability_all.csv", index=False)
    results['stable_pairs'].to_csv("data/results/stable_pairs_final.csv", index=False)
    
    # Plot
    if len(results['stable_pairs']) > 0:
        psc.plot_stability(
            close_prices,
            results['stable_pairs'],
            results['window_data']
        )
    
    print(f"\n💾 Results saved to data/results/")
    print("✅ Stability analysis complete!")
