"""
Data Downloader Module
Downloads historical price data from Yahoo Finance
"""

import yfinance as yf
import pandas as pd
import yaml
import os
from datetime import datetime
from tqdm import tqdm


class DataDownloader:
    """
    Downloads OHLCV data for all assets defined in config.
    
    Usage:
        downloader = DataDownloader()
        data = downloader.download_all()
    """
    
    def __init__(self, config_path="config/settings.yaml"):
        """Load configuration"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.assets = self.config['data']['assets']
        self.start_date = self.config['data']['start_date']
        self.end_date = self.config['data']['end_date']
        self.raw_data_path = "data/raw/"
        
        # Ensure directory exists
        os.makedirs(self.raw_data_path, exist_ok=True)
    
    def download_single(self, ticker):
        """
        Download data for a single ticker.
        
        Args:
            ticker (str): Asset ticker symbol (e.g., 'SPY')
            
        Returns:
            pd.DataFrame: OHLCV data with datetime index
            None: if download fails
        """
        try:
            data = yf.download(
                ticker,
                start=self.start_date,
                end=self.end_date,
                progress=False,
                auto_adjust=True
            )
            
            if data.empty:
                print(f"  ⚠️  {ticker}: No data returned")
                return None
            
            # Flatten multi-level columns if present
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            
            # Keep only OHLCV columns
            expected_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            data = data[[col for col in expected_cols if col in data.columns]]
            
            # Add ticker column for identification
            data['Ticker'] = ticker
            
            return data
            
        except Exception as e:
            print(f"  ❌ {ticker}: Download failed - {e}")
            return None
    
    def download_all(self):
        """
        Download data for ALL assets in config.
        
        Returns:
            dict: {ticker: pd.DataFrame} for all successful downloads
        """
        print("=" * 60)
        print("📥 DOWNLOADING MARKET DATA")
        print(f"   Assets: {len(self.assets)}")
        print(f"   Period: {self.start_date} to {self.end_date}")
        print("=" * 60)
        
        all_data = {}
        failed = []
        
        for ticker in tqdm(self.assets, desc="Downloading"):
            data = self.download_single(ticker)
            
            if data is not None and not data.empty:
                all_data[ticker] = data
            else:
                failed.append(ticker)
        
        # Summary
        print("\n" + "=" * 60)
        print(f"✅ Successfully downloaded: {len(all_data)}/{len(self.assets)}")
        if failed:
            print(f"❌ Failed: {failed}")
        print("=" * 60)
        
        return all_data
    
    def save_raw(self, all_data):
        """
        Save raw data to CSV files.
        
        Args:
            all_data (dict): {ticker: pd.DataFrame}
        """
        print("\n💾 Saving raw data...")
        
        for ticker, data in all_data.items():
            filepath = os.path.join(self.raw_data_path, f"{ticker}.csv")
            data.to_csv(filepath)
        
        # Also save a combined close prices file (very useful later)
        close_prices = pd.DataFrame()
        for ticker, data in all_data.items():
            close_prices[ticker] = data['Close']
        
        close_prices.to_csv(os.path.join(self.raw_data_path, "_all_close_prices.csv"))
        
        print(f"✅ Saved {len(all_data)} individual files + combined close prices")
        print(f"📁 Location: {self.raw_data_path}")
    
    def load_raw(self, ticker):
        """
        Load previously saved raw data for a ticker.
        
        Args:
            ticker (str): Asset ticker symbol
            
        Returns:
            pd.DataFrame: Saved data
        """
        filepath = os.path.join(self.raw_data_path, f"{ticker}.csv")
        if os.path.exists(filepath):
            return pd.read_csv(filepath, index_col=0, parse_dates=True)
        else:
            print(f"⚠️  No saved data found for {ticker}")
            return None


# === RUN DIRECTLY ===
if __name__ == "__main__":
    downloader = DataDownloader()
    all_data = downloader.download_all()
    downloader.save_raw(all_data)
