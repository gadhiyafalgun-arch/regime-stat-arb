"""
Data Validator Module
Validates data quality before using it in the pipeline
"""

import pandas as pd
import numpy as np
import os
import yaml


class DataValidator:
    """
    Runs quality checks on processed data:
    - No missing values
    - No zero prices
    - Reasonable return ranges
    - Sufficient data length
    - No stale data (stuck prices)
    
    Usage:
        validator = DataValidator()
        report = validator.validate_all()
    """
    
    def __init__(self, config_path="config/settings.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.processed_data_path = "data/processed/"
    
    def load_data(self):
        """Load processed close prices and returns"""
        close_path = os.path.join(self.processed_data_path, "close_prices.csv")
        returns_path = os.path.join(self.processed_data_path, "log_returns.csv")
        
        close_prices = pd.read_csv(close_path, index_col=0, parse_dates=True)
        log_returns = pd.read_csv(returns_path, index_col=0, parse_dates=True)
        
        return close_prices, log_returns
    
    def check_missing_values(self, close_prices):
        """Check for any missing values"""
        missing = close_prices.isna().sum()
        total_missing = missing.sum()
        
        if total_missing == 0:
            return {"status": "PASS", "detail": "No missing values"}
        else:
            bad_tickers = missing[missing > 0].to_dict()
            return {"status": "FAIL", "detail": f"Missing values: {bad_tickers}"}
    
    def check_zero_prices(self, close_prices):
        """Check for zero or negative prices"""
        zeros = (close_prices <= 0).sum()
        total_zeros = zeros.sum()
        
        if total_zeros == 0:
            return {"status": "PASS", "detail": "No zero/negative prices"}
        else:
            bad_tickers = zeros[zeros > 0].to_dict()
            return {"status": "FAIL", "detail": f"Zero/negative prices: {bad_tickers}"}
    
    def check_return_range(self, log_returns):
        """Check for unreasonable returns (> 100% daily)"""
        extreme = (log_returns.abs() > 1.0).sum()  # > 100% daily move
        total_extreme = extreme.sum()
        
        if total_extreme == 0:
            return {"status": "PASS", "detail": "All returns within reasonable range"}
        else:
            bad_tickers = extreme[extreme > 0].to_dict()
            return {"status": "WARN", "detail": f"Extreme returns found: {bad_tickers}"}
    
    def check_data_length(self, close_prices, min_days=500):
        """Check if we have enough data"""
        n_days = len(close_prices)
        
        if n_days >= min_days:
            return {"status": "PASS", "detail": f"{n_days} trading days (min: {min_days})"}
        else:
            return {"status": "FAIL", "detail": f"Only {n_days} days (need {min_days})"}
    
    def check_stale_data(self, close_prices, max_stale_days=10):
        """Check for assets with stuck/unchanging prices"""
        stale_tickers = []
        
        for ticker in close_prices.columns:
            # Count max consecutive days with zero change
            changes = close_prices[ticker].diff()
            is_zero = (changes == 0)
            
            # Find max consecutive zeros
            max_consecutive = 0
            current_streak = 0
            for val in is_zero:
                if val:
                    current_streak += 1
                    max_consecutive = max(max_consecutive, current_streak)
                else:
                    current_streak = 0
            
            if max_consecutive > max_stale_days:
                stale_tickers.append(f"{ticker}({max_consecutive}d)")
        
        if len(stale_tickers) == 0:
            return {"status": "PASS", "detail": "No stale price data detected"}
        else:
            return {"status": "WARN", "detail": f"Stale data: {stale_tickers}"}
    
    def check_date_continuity(self, close_prices):
        """Check for large gaps in dates (beyond normal weekends/holidays)"""
        dates = close_prices.index
        gaps = pd.Series(dates[1:]) - pd.Series(dates[:-1])
        max_gap = gaps.max()
        
        # More than 10 calendar days = suspicious
        if max_gap.days <= 10:
            return {"status": "PASS", "detail": f"Max date gap: {max_gap.days} days"}
        else:
            return {"status": "WARN", "detail": f"Large date gap found: {max_gap.days} days"}
    
    def generate_summary_stats(self, close_prices, log_returns):
        """Generate summary statistics for each asset"""
        stats = pd.DataFrame(index=close_prices.columns)
        
        stats['Start_Date'] = close_prices.apply(lambda x: x.first_valid_index()).dt.strftime('%Y-%m-%d')
        stats['End_Date'] = close_prices.apply(lambda x: x.last_valid_index()).dt.strftime('%Y-%m-%d')
        stats['Trading_Days'] = close_prices.count()
        stats['Min_Price'] = close_prices.min().round(2)
        stats['Max_Price'] = close_prices.max().round(2)
        stats['Last_Price'] = close_prices.iloc[-1].round(2)
        stats['Avg_Daily_Return'] = (log_returns.mean() * 100).round(4)
        stats['Annual_Volatility'] = (log_returns.std() * np.sqrt(252) * 100).round(2)
        stats['Total_Return_Pct'] = ((close_prices.iloc[-1] / close_prices.iloc[0] - 1) * 100).round(2)
        
        return stats
    
    def validate_all(self):
        """
        Run ALL validation checks and print report.
        
        Returns:
            dict: Validation results
        """
        print("=" * 60)
        print("🔍 VALIDATING DATA QUALITY")
        print("=" * 60)
        
        # Load data
        close_prices, log_returns = self.load_data()
        
        print(f"\n📊 Dataset: {close_prices.shape[1]} assets × {close_prices.shape[0]} days")
        print(f"   Range: {close_prices.index[0].strftime('%Y-%m-%d')} to {close_prices.index[-1].strftime('%Y-%m-%d')}")
        
        # Run checks
        checks = {
            "Missing Values": self.check_missing_values(close_prices),
            "Zero Prices": self.check_zero_prices(close_prices),
            "Return Range": self.check_return_range(log_returns),
            "Data Length": self.check_data_length(close_prices),
            "Stale Data": self.check_stale_data(close_prices),
            "Date Continuity": self.check_date_continuity(close_prices),
        }
        
        # Print results
        print("\n" + "-" * 60)
        print("CHECK RESULTS:")
        print("-" * 60)
        
        pass_count = 0
        warn_count = 0
        fail_count = 0
        
        for check_name, result in checks.items():
            status = result['status']
            detail = result['detail']
            
            if status == "PASS":
                icon = "✅"
                pass_count += 1
            elif status == "WARN":
                icon = "⚠️ "
                warn_count += 1
            else:
                icon = "❌"
                fail_count += 1
            
            print(f"  {icon} {check_name}: {detail}")
        
        # Summary stats
        print("\n" + "-" * 60)
        print("ASSET SUMMARY STATISTICS:")
        print("-" * 60)
        
        stats = self.generate_summary_stats(close_prices, log_returns)
        print(stats.to_string())
        
        # Final verdict
        print("\n" + "=" * 60)
        print(f"📋 VALIDATION SUMMARY: {pass_count} PASS | {warn_count} WARN | {fail_count} FAIL")
        
        if fail_count == 0:
            print("✅ DATA IS READY FOR ANALYSIS")
        else:
            print("❌ DATA HAS ISSUES - FIX BEFORE PROCEEDING")
        
        print("=" * 60)
        
        return {
            'checks': checks,
            'stats': stats,
            'passed': fail_count == 0
        }


# === RUN DIRECTLY ===
if __name__ == "__main__":
    validator = DataValidator()
    report = validator.validate_all()
