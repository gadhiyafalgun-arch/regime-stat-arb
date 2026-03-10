"""
Correlation Filter Module
Quick initial screening to find potentially related asset pairs
"""

import pandas as pd
import numpy as np
import os
import itertools
import seaborn as sns
import matplotlib.pyplot as plt


class CorrelationFilter:
    """
    Filters asset pairs by correlation as a FIRST PASS.
    
    Why: Testing cointegration is expensive.
    Correlation filter reduces N*N pairs to a manageable set.
    
    Note: High correlation does NOT mean cointegration.
    But low correlation almost always means NO cointegration.
    So this is a NECESSARY but not SUFFICIENT filter.
    
    Usage:
        cf = CorrelationFilter()
        candidates = cf.filter_pairs(close_prices, min_corr=0.7)
    """
    
    def __init__(self):
        self.results_path = "data/results/"
        os.makedirs(self.results_path, exist_ok=True)
    
    def calculate_correlation_matrix(self, close_prices, method='pearson'):
        """
        Calculate correlation matrix of log returns.
        
        We use RETURNS not PRICES because:
        - Prices are non-stationary (trend upward)
        - Two unrelated stocks both trending up = high price correlation
        - Returns remove the trend → more meaningful correlation
        
        Args:
            close_prices (pd.DataFrame): Aligned close prices
            method (str): 'pearson' or 'spearman'
            
        Returns:
            pd.DataFrame: Correlation matrix
        """
        # Calculate log returns
        log_returns = np.log(close_prices / close_prices.shift(1)).dropna()
        
        # Calculate correlation
        corr_matrix = log_returns.corr(method=method)
        
        return corr_matrix
    
    def get_all_pairs(self, close_prices):
        """
        Generate all unique pairs from asset list.
        
        For N assets: N*(N-1)/2 pairs
        For 30 assets: 435 pairs
        
        Args:
            close_prices (pd.DataFrame): Close prices
            
        Returns:
            list: List of (ticker_a, ticker_b) tuples
        """
        tickers = close_prices.columns.tolist()
        
        # Remove SPY (we use it for regime detection, not pair trading)
        if 'SPY' in tickers:
            tickers.remove('SPY')
        
        pairs = list(itertools.combinations(tickers, 2))
        print(f"📊 Total possible pairs: {len(pairs)} (from {len(tickers)} assets)")
        
        return pairs
    
    def filter_pairs(self, close_prices, min_corr=0.7, max_corr=0.99):
        """
        Filter pairs by correlation threshold.
        
        Args:
            close_prices (pd.DataFrame): Aligned close prices
            min_corr (float): Minimum correlation (default 0.7)
            max_corr (float): Maximum correlation (default 0.99)
                              Too high = basically same asset
            
        Returns:
            pd.DataFrame: Filtered pairs with correlations
        """
        print("=" * 60)
        print("🔗 CORRELATION FILTER")
        print(f"   Thresholds: {min_corr} ≤ |corr| ≤ {max_corr}")
        print("=" * 60)
        
        # Get correlation matrix
        corr_matrix = self.calculate_correlation_matrix(close_prices)
        
        # Get all pairs
        all_pairs = self.get_all_pairs(close_prices)
        
        # Filter by correlation
        filtered_pairs = []
        
        for ticker_a, ticker_b in all_pairs:
            corr = corr_matrix.loc[ticker_a, ticker_b]
            
            if min_corr <= abs(corr) <= max_corr:
                filtered_pairs.append({
                    'Ticker_A': ticker_a,
                    'Ticker_B': ticker_b,
                    'Correlation': round(corr, 4)
                })
        
        # Create DataFrame and sort
        results = pd.DataFrame(filtered_pairs)
        
        if len(results) > 0:
            results = results.sort_values('Correlation', ascending=False).reset_index(drop=True)
        
        # Summary
        print(f"\n✅ Pairs passing correlation filter: {len(results)}/{len(all_pairs)}")
        print(f"   Reduction: {(1 - len(results)/len(all_pairs))*100:.1f}% filtered out")
        
        if len(results) > 0:
            print(f"\n📋 Top 15 correlated pairs:")
            print(results.head(15).to_string(index=False))
        
        return results, corr_matrix
    
    def plot_correlation_matrix(self, corr_matrix, save=True):
        """
        Plot correlation heatmap.
        
        Args:
            corr_matrix (pd.DataFrame): Correlation matrix
            save (bool): Save to file
        """
        fig, ax = plt.subplots(figsize=(16, 14))
        
        sns.heatmap(
            corr_matrix,
            annot=True,
            fmt='.2f',
            cmap='RdYlBu_r',
            center=0,
            vmin=-1,
            vmax=1,
            square=True,
            linewidths=0.5,
            annot_kws={'size': 7},
            ax=ax
        )
        
        ax.set_title('Asset Return Correlation Matrix', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save:
            filepath = os.path.join(self.results_path, "correlation_matrix.png")
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            print(f"\n📊 Correlation heatmap saved: {filepath}")
        
        plt.close()
    
    def plot_top_pairs(self, close_prices, filtered_pairs, n_pairs=6, save=True):
        """
        Plot price series of top correlated pairs.
        
        Args:
            close_prices (pd.DataFrame): Close prices
            filtered_pairs (pd.DataFrame): Filtered pairs
            n_pairs (int): Number of pairs to plot
            save (bool): Save to file
        """
        n_pairs = min(n_pairs, len(filtered_pairs))
        
        fig, axes = plt.subplots(n_pairs, 1, figsize=(14, 4 * n_pairs))
        if n_pairs == 1:
            axes = [axes]
        
        for i in range(n_pairs):
            row = filtered_pairs.iloc[i]
            ticker_a = row['Ticker_A']
            ticker_b = row['Ticker_B']
            corr = row['Correlation']
            
            ax = axes[i]
            
            # Normalize prices to start at 100
            price_a = close_prices[ticker_a] / close_prices[ticker_a].iloc[0] * 100
            price_b = close_prices[ticker_b] / close_prices[ticker_b].iloc[0] * 100
            
            ax.plot(price_a, label=ticker_a, linewidth=1.5)
            ax.plot(price_b, label=ticker_b, linewidth=1.5)
            ax.set_title(f'{ticker_a} vs {ticker_b} (corr: {corr:.4f})', fontweight='bold')
            ax.legend(loc='upper left')
            ax.set_ylabel('Normalized Price')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            filepath = os.path.join(self.results_path, "top_correlated_pairs.png")
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            print(f"📊 Top pairs chart saved: {filepath}")
        
        plt.close()


# === RUN DIRECTLY ===
if __name__ == "__main__":
    # Load processed data
    close_prices = pd.read_csv(
        "data/processed/close_prices.csv",
        index_col=0,
        parse_dates=True
    )
    
    # Run correlation filter
    cf = CorrelationFilter()
    filtered_pairs, corr_matrix = cf.filter_pairs(close_prices, min_corr=0.7)
    
    # Save results
    filtered_pairs.to_csv("data/results/correlation_filtered_pairs.csv", index=False)
    
    # Generate plots
    cf.plot_correlation_matrix(corr_matrix)
    
    if len(filtered_pairs) > 0:
        cf.plot_top_pairs(close_prices, filtered_pairs)
    
    print("\n✅ Correlation filter complete!")
