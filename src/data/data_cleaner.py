"""
Data Cleaner Module
Cleans and validates raw market data
"""

import pandas as pd
import numpy as np
import os
import yaml


class DataCleaner:
    """
    Cleans raw OHLCV data:
    - Handles missing values
    - Removes outliers
    - Calculates returns
    - Aligns all assets to same date range
    
    Usage:
        cleaner = DataCleaner()
        clean_data = cleaner.clean_all()
    """
    
    def __init__(self, config_path="config/settings.yaml"):
        """Load configuration"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.assets = self.config['data']['assets']
        self.raw_data_path = "data/raw/"
        self.processed_data_path = "data/processed/"
        
        os.makedirs(self.processed_data_path, exist_ok=True)
    
    def load_raw_data(self):
        """
        Load all raw CSV files into a dictionary.
        
        Returns:
            dict: {ticker: pd.DataFrame}
        """
        all_data = {}
        
        for ticker in self.assets:
            filepath = os.path.join(self.raw_data_path, f"{ticker}.csv")
            if os.path.exists(filepath):
                df = pd.read_csv(filepath, index_col=0, parse_dates=True)
                all_data[ticker] = df
            else:
                print(f"⚠️  {ticker}: Raw file not found, skipping")
        
        print(f"📂 Loaded {len(all_data)} raw files")
        return all_data
    
    def clean_single(self, df, ticker):
        """
        Clean a single asset's data.
        
        Steps:
        1. Remove duplicate dates
        2. Sort by date
        3. Handle missing values
        4. Remove extreme outliers
        5. Calculate returns
        
        Args:
            df (pd.DataFrame): Raw OHLCV data
            ticker (str): Asset name for logging
            
        Returns:
            pd.DataFrame: Cleaned data
        """
        original_rows = len(df)
        
        # Step 1: Remove duplicates
        df = df[~df.index.duplicated(keep='first')]
        
        # Step 2: Sort by date
        df = df.sort_index()
        
        # Step 3: Handle missing values
        # Forward fill first (use previous day's price)
        # Then backward fill (for any leading NaNs)
        numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        available_cols = [col for col in numeric_cols if col in df.columns]
        
        missing_before = df[available_cols].isna().sum().sum()
        
        df[available_cols] = df[available_cols].ffill()
        df[available_cols] = df[available_cols].bfill()
        
        # Drop any rows still with NaN (shouldn't happen but safety)
        df = df.dropna(subset=['Close'])
        
        # Step 4: Remove extreme outliers in Close price
        # If daily change > 50%, likely data error
        if len(df) > 1:
            daily_returns = df['Close'].pct_change()
            extreme_mask = daily_returns.abs() > 0.50  # 50% daily move
            extreme_count = extreme_mask.sum()
            
            if extreme_count > 0:
                print(f"  ⚠️  {ticker}: Found {extreme_count} extreme moves (>50%), keeping but flagging")
                df['Extreme_Move'] = extreme_mask
            else:
                df['Extreme_Move'] = False
        
        # Step 5: Calculate returns
        df['Simple_Return'] = df['Close'].pct_change()
        df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))
        
        # Step 6: Calculate rolling volatility (useful later)
        df['Volatility_20d'] = df['Log_Return'].rolling(window=20).std() * np.sqrt(252)
        
        # Drop first row (NaN from return calculation)
        df = df.iloc[1:]
        
        cleaned_rows = len(df)
        
        return df
    
    def build_close_price_matrix(self, all_clean_data):
        """
        Build aligned close price matrix for all assets.
        Only keeps dates where ALL assets have data.
        
        Args:
            all_clean_data (dict): {ticker: pd.DataFrame}
            
        Returns:
            pd.DataFrame: Close prices with tickers as columns
        """
        close_prices = pd.DataFrame()
        
        for ticker, df in all_clean_data.items():
            close_prices[ticker] = df['Close']
        
        # Keep only dates where all assets have data
        original_dates = len(close_prices)
        close_prices = close_prices.dropna()
        aligned_dates = len(close_prices)
        
        print(f"\n📊 Close Price Matrix:")
        print(f"   Original dates: {original_dates}")
        print(f"   Aligned dates:  {aligned_dates}")
        print(f"   Date range:     {close_prices.index[0].strftime('%Y-%m-%d')} to {close_prices.index[-1].strftime('%Y-%m-%d')}")
        print(f"   Assets:         {len(close_prices.columns)}")
        
        return close_prices
    
    def build_returns_matrix(self, close_prices):
        """
        Build log returns matrix from close prices.
        
        Args:
            close_prices (pd.DataFrame): Aligned close prices
            
        Returns:
            pd.DataFrame: Log returns
        """
        log_returns = np.log(close_prices / close_prices.shift(1)).dropna()
        return log_returns
    
    def clean_all(self):
        """
        Master function: Load, clean, align, and save all data.
        
        Returns:
            dict: {
                'individual': {ticker: pd.DataFrame},
                'close_prices': pd.DataFrame,
                'log_returns': pd.DataFrame
            }
        """
        print("=" * 60)
        print("🧹 CLEANING MARKET DATA")
        print("=" * 60)
        
        # Load raw data
        all_raw = self.load_raw_data()
        
        # Clean each asset
        all_clean = {}
        print("\nCleaning individual assets:")
        
        for ticker, df in all_raw.items():
            clean_df = self.clean_single(df, ticker)
            all_clean[ticker] = clean_df
            print(f"  ✅ {ticker}: {len(clean_df)} rows")
        
        # Build aligned matrices
        close_prices = self.build_close_price_matrix(all_clean)
        log_returns = self.build_returns_matrix(close_prices)
        
        # Save everything
        self.save_processed(all_clean, close_prices, log_returns)
        
        result = {
            'individual': all_clean,
            'close_prices': close_prices,
            'log_returns': log_returns
        }
        
        print("\n" + "=" * 60)
        print("✅ DATA CLEANING COMPLETE")
        print("=" * 60)
        
        return result
    
    def save_processed(self, all_clean, close_prices, log_returns):
        """Save all processed data"""
        print("\n💾 Saving processed data...")
        
        # Save individual cleaned files
        for ticker, df in all_clean.items():
            filepath = os.path.join(self.processed_data_path, f"{ticker}_clean.csv")
            df.to_csv(filepath)
        
        # Save aligned matrices
        close_prices.to_csv(os.path.join(self.processed_data_path, "close_prices.csv"))
        log_returns.to_csv(os.path.join(self.processed_data_path, "log_returns.csv"))
        
        print(f"✅ Saved to {self.processed_data_path}")
    
    def load_processed(self):
        """
        Load previously processed data.
        
        Returns:
            dict: {'close_prices': DataFrame, 'log_returns': DataFrame}
        """
        close_path = os.path.join(self.processed_data_path, "close_prices.csv")
        returns_path = os.path.join(self.processed_data_path, "log_returns.csv")
        
        if os.path.exists(close_path) and os.path.exists(returns_path):
            close_prices = pd.read_csv(close_path, index_col=0, parse_dates=True)
            log_returns = pd.read_csv(returns_path, index_col=0, parse_dates=True)
            
            print(f"📂 Loaded processed data:")
            print(f"   Close prices: {close_prices.shape}")
            print(f"   Log returns:  {log_returns.shape}")
            
            return {
                'close_prices': close_prices,
                'log_returns': log_returns
            }
        else:
            print("⚠️  No processed data found. Run clean_all() first.")
            return None


# === RUN DIRECTLY ===
if __name__ == "__main__":
    cleaner = DataCleaner()
    result = cleaner.clean_all()
    
    # Quick summary
    print("\n📋 QUICK SUMMARY:")
    print(f"   Close prices shape: {result['close_prices'].shape}")
    print(f"   Log returns shape:  {result['log_returns'].shape}")
    print(f"\n   Sample close prices (last 3 days):")
    print(result['close_prices'].tail(3))
