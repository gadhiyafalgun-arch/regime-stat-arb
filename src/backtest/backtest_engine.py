"""
Walk-Forward Backtesting Engine
=================================
Simulates trading the regime-adaptive stat-arb strategy day by day,
properly accounting for:
1. Transaction costs (commissions + slippage)
2. Capital allocation per pair
3. Regime-based position sizing
4. Circuit breaker shutdowns
5. Walk-forward (no look-ahead bias)

Key Design Principles:
    - Uses PREVIOUS day's signal to trade at TODAY's open
    - All position changes incur transaction costs
    - Capital is tracked in dollars, not just returns
    - Each pair maintains its own P&L ledger
    - Portfolio aggregates all pair P&Ls

Transaction Cost Model:
    cost = |position_change| * price * cost_per_share
    cost_per_share ≈ 0.1% (covers commission + half-spread slippage)
    
    For a \$10,000 position change: cost ≈ \$10
"""

import numpy as np
import pandas as pd
from pathlib import Path


class PairBacktester:
    """
    Backtests a single pair trading strategy.

    Simulates dollar P&L from trading the spread between two assets
    with a dynamic hedge ratio from the Kalman Filter.

    Parameters
    ----------
    pair_name : str
        Name of the pair (e.g., 'EFA-VGK')
    capital_per_pair : float
        Dollar amount allocated to this pair.
    transaction_cost_bps : float
        Round-trip transaction cost in basis points.
        Default: 10 bps (0.10%) — covers commission + slippage.
    slippage_bps : float
        Additional slippage in basis points.
        Default: 5 bps (0.05%).
    """

    def __init__(
        self,
        pair_name: str,
        capital_per_pair: float = 50_000,
        transaction_cost_bps: float = 10.0,
        slippage_bps: float = 5.0
    ):
        self.pair_name = pair_name
        self.capital_per_pair = capital_per_pair
        self.total_cost_bps = (transaction_cost_bps + slippage_bps) / 10_000

        # Parse pair
        parts = pair_name.split('-')
        if len(parts) != 2:
            raise ValueError(f"Invalid pair name: {pair_name}")
        self.asset_y = parts[0]
        self.asset_x = parts[1]

        # Results
        self.results_df = None
        self.trade_results = []

    def run(
        self,
        signals_df: pd.DataFrame,
        prices_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Run the backtest for this pair.

        Parameters
        ----------
        signals_df : pd.DataFrame
            Signal data with columns: position, zscore, regime, beta
        prices_df : pd.DataFrame
            Price data with columns for both assets

        Returns
        -------
        pd.DataFrame
            Daily P&L results
        """
        # Validate inputs
        if self.asset_y not in prices_df.columns or self.asset_x not in prices_df.columns:
            raise ValueError(f"Assets {self.asset_y}, {self.asset_x} not in price data")

        # Align all data to common dates
        common = signals_df.index.intersection(prices_df.index)
        signals = signals_df.reindex(common).copy()
        price_y = prices_df[self.asset_y].reindex(common)
        price_x = prices_df[self.asset_x].reindex(common)

        # Get hedge ratio
        if 'beta' in signals.columns:
            beta = signals['beta'].fillna(method='ffill').fillna(1.0)
        else:
            beta = pd.Series(1.0, index=common)

        # Get position signal
        position = signals['position'].fillna(0)

        # Daily returns of each asset
        ret_y = price_y.pct_change().fillna(0)
        ret_x = price_x.pct_change().fillna(0)

        # Build results
        result = pd.DataFrame(index=common)
        result['price_y'] = price_y
        result['price_x'] = price_x
        result['beta'] = beta
        result['position'] = position
        result['regime'] = signals['regime'] if 'regime' in signals.columns else 'Unknown'

        # Previous day position (what we actually hold entering today)
        result['prev_position'] = position.shift(1).fillna(0)

        # Position change (triggers transaction costs)
        result['position_change'] = (result['position'] - result['prev_position']).abs()

        # Spread return: long Y, short beta*X
        # If position = +1: profit from spread narrowing (Y up, X down)
        # If position = -1: profit from spread widening (Y down, X up)
        result['spread_return'] = ret_y - beta * ret_x

        # Dollar notional: how much capital is at risk
        result['notional'] = self.capital_per_pair * result['prev_position'].abs()

        # Gross P&L (before costs)
        result['gross_pnl'] = result['prev_position'] * result['spread_return'] * self.capital_per_pair

        # Transaction costs
        # Cost proportional to notional value of position change
        result['txn_cost'] = result['position_change'] * self.capital_per_pair * self.total_cost_bps

        # Net P&L
        result['net_pnl'] = result['gross_pnl'] - result['txn_cost']

        # Cumulative P&L
        result['cumulative_pnl'] = result['net_pnl'].cumsum()

        # Pair equity curve
        result['equity'] = self.capital_per_pair + result['cumulative_pnl']

        # Return relative to allocated capital
        result['daily_return'] = result['net_pnl'] / self.capital_per_pair

        # Track trades
        self._extract_trades(result)

        self.results_df = result
        return result

    def _extract_trades(self, result: pd.DataFrame):
        """Extract individual trade results from backtest."""
        self.trade_results = []
        position = result['position']
        pnl = result['net_pnl']

        in_trade = False
        entry_date = None
        entry_direction = 0
        trade_pnl = 0.0

        for i in range(len(result)):
            curr_pos = position.iloc[i]
            daily_pnl = pnl.iloc[i]

            if not in_trade and curr_pos != 0:
                # Entry
                in_trade = True
                entry_date = result.index[i]
                entry_direction = int(curr_pos)
                trade_pnl = daily_pnl

            elif in_trade and curr_pos == entry_direction:
                # Holding
                trade_pnl += daily_pnl

            elif in_trade and curr_pos != entry_direction:
                # Exit or reversal
                self.trade_results.append({
                    'pair': self.pair_name,
                    'entry_date': entry_date,
                    'exit_date': result.index[i],
                    'direction': 'long' if entry_direction > 0 else 'short',
                    'pnl': trade_pnl,
                    'return_pct': trade_pnl / self.capital_per_pair * 100,
                    'holding_days': (result.index[i] - entry_date).days,
                    'regime': result['regime'].iloc[i]
                })

                if curr_pos != 0:
                    # Reversal — start new trade
                    entry_date = result.index[i]
                    entry_direction = int(curr_pos)
                    trade_pnl = daily_pnl
                else:
                    in_trade = False
                    trade_pnl = 0.0

        # Close any open trade
        if in_trade:
            self.trade_results.append({
                'pair': self.pair_name,
                'entry_date': entry_date,
                'exit_date': result.index[-1],
                'direction': 'long' if entry_direction > 0 else 'short',
                'pnl': trade_pnl,
                'return_pct': trade_pnl / self.capital_per_pair * 100,
                'holding_days': (result.index[-1] - entry_date).days,
                'regime': 'open'
            })

    def get_pair_summary(self) -> dict:
        """Get summary stats for this pair's backtest."""
        if self.results_df is None:
            return {}

        r = self.results_df
        trades = self.trade_results

        total_pnl = r['cumulative_pnl'].iloc[-1]
        total_costs = r['txn_cost'].sum()
        active_days = (r['prev_position'] != 0).sum()
        total_days = len(r)

        # Trade stats
        if trades:
            trade_pnls = [t['pnl'] for t in trades]
            winners = [p for p in trade_pnls if p > 0]
            losers = [p for p in trade_pnls if p <= 0]
            win_rate = len(winners) / len(trades) * 100 if trades else 0
            avg_win = np.mean(winners) if winners else 0
            avg_loss = np.mean(losers) if losers else 0
        else:
            win_rate = 0
            avg_win = 0
            avg_loss = 0

        daily_ret = r['daily_return']
        active_ret = daily_ret[daily_ret != 0]

        return {
            'pair': self.pair_name,
            'total_pnl': total_pnl,
            'total_return_pct': total_pnl / self.capital_per_pair * 100,
            'total_txn_costs': total_costs,
            'n_trades': len(trades),
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': abs(avg_win * len([p for p in (t['pnl'] for t in trades) if p > 0]) /
                               (avg_loss * len([p for p in (t['pnl'] for t in trades) if p <= 0])))
                               if avg_loss != 0 and trades else 0,
            'active_days': active_days,
            'active_pct': active_days / total_days * 100,
            'ann_return_pct': daily_ret.mean() * 252 * 100,
            'ann_vol_pct': daily_ret.std() * np.sqrt(252) * 100,
            'sharpe': (daily_ret.mean() / daily_ret.std() * np.sqrt(252))
                       if daily_ret.std() > 0 else 0,
        }


class PortfolioBacktester:
    """
    Runs backtests across all pairs and aggregates into portfolio.

    Parameters
    ----------
    total_capital : float
        Total portfolio capital.
    max_pairs : int
        Maximum number of pairs to trade simultaneously.
    capital_allocation : str
        'equal' — split capital equally across all pairs
        'vol_weighted' — allocate more to lower-vol pairs
    transaction_cost_bps : float
        Transaction cost in basis points per trade.
    slippage_bps : float
        Slippage in basis points per trade.
    """

    def __init__(
        self,
        total_capital: float = 1_000_000,
        max_pairs: int = 14,
        capital_allocation: str = 'equal',
        transaction_cost_bps: float = 10.0,
        slippage_bps: float = 5.0
    ):
        self.total_capital = total_capital
        self.max_pairs = max_pairs
        self.capital_allocation = capital_allocation
        self.transaction_cost_bps = transaction_cost_bps
        self.slippage_bps = slippage_bps

        # Results
        self.pair_results = {}
        self.pair_summaries = {}
        self.all_trades = []
        self.portfolio_df = None

    def run(
        self,
        pair_signals: dict,
        prices_df: pd.DataFrame,
        verbose: bool = True
    ) -> pd.DataFrame:
        """
        Run backtest for all pairs and aggregate.

        Parameters
        ----------
        pair_signals : dict
            pair_name -> signals DataFrame
        prices_df : pd.DataFrame
            Price data for all assets

        Returns
        -------
        pd.DataFrame
            Portfolio-level daily results
        """
        n_pairs = min(len(pair_signals), self.max_pairs)
        capital_per_pair = self.total_capital / n_pairs

        if verbose:
            print(f"\n📊 Backtesting {n_pairs} pairs")
            print(f"   Capital per pair: ${capital_per_pair:,.0f}")
            print(f"   Transaction cost: {self.transaction_cost_bps} bps")
            print(f"   Slippage: {self.slippage_bps} bps")

        # Run each pair
        for pair_name, signals_df in pair_signals.items():
            try:
                bt = PairBacktester(
                    pair_name=pair_name,
                    capital_per_pair=capital_per_pair,
                    transaction_cost_bps=self.transaction_cost_bps,
                    slippage_bps=self.slippage_bps
                )

                result = bt.run(signals_df, prices_df)
                self.pair_results[pair_name] = result
                self.pair_summaries[pair_name] = bt.get_pair_summary()
                self.all_trades.extend(bt.trade_results)

                if verbose:
                    s = self.pair_summaries[pair_name]
                    print(f"   {pair_name:<12s}: PnL ${s['total_pnl']:>10,.2f}  "
                          f"({s['total_return_pct']:>6.2f}%)  "
                          f"Trades: {s['n_trades']:>3d}  "
                          f"WR: {s['win_rate']:>5.1f}%  "
                          f"Sharpe: {s['sharpe']:>6.2f}")

            except Exception as e:
                if verbose:
                    print(f"   {pair_name}: FAILED — {str(e)[:60]}")

        # Aggregate portfolio
        self._aggregate_portfolio()

        return self.portfolio_df

    def _aggregate_portfolio(self):
        """Aggregate all pair results into portfolio-level metrics."""
        if not self.pair_results:
            raise ValueError("No pair results to aggregate!")

        # Collect daily PnL from each pair
        pnl_dict = {}
        cost_dict = {}
        for pair_name, result in self.pair_results.items():
            pnl_dict[f'pnl_{pair_name}'] = result['net_pnl']
            cost_dict[f'cost_{pair_name}'] = result['txn_cost']

        pnl_df = pd.DataFrame(pnl_dict).fillna(0)
        cost_df = pd.DataFrame(cost_dict).fillna(0)

        # Portfolio daily P&L
        portfolio = pd.DataFrame(index=pnl_df.index)
        portfolio['gross_pnl'] = pnl_df.sum(axis=1) + cost_df.sum(axis=1)
        portfolio['total_costs'] = cost_df.sum(axis=1)
        portfolio['net_pnl'] = pnl_df.sum(axis=1)
        portfolio['cumulative_pnl'] = portfolio['net_pnl'].cumsum()
        portfolio['equity'] = self.total_capital + portfolio['cumulative_pnl']
        portfolio['daily_return'] = portfolio['net_pnl'] / self.total_capital
        portfolio['cumulative_return'] = portfolio['equity'] / self.total_capital - 1

        # Active pairs per day
        active_counts = {}
        for pair_name, result in self.pair_results.items():
            active_counts[pair_name] = (result['prev_position'].abs() > 0).astype(int)
        active_df = pd.DataFrame(active_counts).reindex(portfolio.index).fillna(0)
        portfolio['n_active_pairs'] = active_df.sum(axis=1)

        # Add individual pair PnLs
        for col in pnl_df.columns:
            portfolio[col] = pnl_df[col]

        self.portfolio_df = portfolio

    def get_portfolio_summary(self) -> dict:
        """Get comprehensive portfolio summary."""
        if self.portfolio_df is None:
            return {}

        p = self.portfolio_df
        ret = p['daily_return']
        equity = p['equity']

        total_pnl = p['cumulative_pnl'].iloc[-1]
        total_costs = p['total_costs'].sum()
        n_days = len(ret)
        n_years = n_days / 252

        # Core metrics
        ann_return = ret.mean() * 252
        ann_vol = ret.std() * np.sqrt(252)
        sharpe = ann_return / ann_vol if ann_vol > 0 else 0

        # Drawdown
        peak = equity.cummax()
        drawdown = (equity - peak) / peak
        max_dd = drawdown.min()

        # Drawdown duration
        dd_start = None
        max_dd_duration = 0
        current_dd_duration = 0
        for i in range(len(drawdown)):
            if drawdown.iloc[i] < -1e-6:
                current_dd_duration += 1
                max_dd_duration = max(max_dd_duration, current_dd_duration)
            else:
                current_dd_duration = 0

        calmar = ann_return / abs(max_dd) if max_dd != 0 else 0

        # Sortino
        downside_ret = ret[ret < 0]
        downside_vol = downside_ret.std() * np.sqrt(252) if len(downside_ret) > 0 else ann_vol
        sortino = ann_return / downside_vol if downside_vol > 0 else 0

        # Trade stats
        if self.all_trades:
            trade_pnls = [t['pnl'] for t in self.all_trades]
            winners = [p for p in trade_pnls if p > 0]
            losers = [p for p in trade_pnls if p <= 0]
            win_rate = len(winners) / len(trade_pnls) * 100
            avg_win = np.mean(winners) if winners else 0
            avg_loss = np.mean(losers) if losers else 0
            gross_profit = sum(winners)
            gross_loss = abs(sum(losers))
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        else:
            win_rate = avg_win = avg_loss = profit_factor = 0

        return {
            'total_pnl': total_pnl,
            'total_return_pct': total_pnl / self.total_capital * 100,
            'annualized_return_pct': ann_return * 100,
            'annualized_vol_pct': ann_vol * 100,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'max_drawdown_pct': max_dd * 100,
            'max_dd_duration_days': max_dd_duration,
            'calmar_ratio': calmar,
            'total_transaction_costs': total_costs,
            'costs_pct_of_pnl': abs(total_costs / total_pnl * 100) if total_pnl != 0 else 0,
            'total_trades': len(self.all_trades),
            'trades_per_year': len(self.all_trades) / n_years if n_years > 0 else 0,
            'win_rate_pct': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'best_day_pct': ret.max() * 100,
            'worst_day_pct': ret.min() * 100,
            'positive_days_pct': (ret > 0).mean() * 100,
            'avg_active_pairs': p['n_active_pairs'].mean(),
            'n_trading_days': n_days,
        }

    def get_pair_comparison(self) -> pd.DataFrame:
        """Get comparison table across all pairs."""
        rows = []
        for pair_name, summary in sorted(self.pair_summaries.items()):
            rows.append(summary)
        return pd.DataFrame(rows)


# ============================================================
# Standalone test
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("BACKTEST ENGINE TEST")
    print("=" * 60)

    prices = pd.read_csv("data/processed/close_prices.csv",
                          index_col=0, parse_dates=True)

    signals_dir = Path("data/results/signals")
    signal_files = list(signals_dir.glob("signals_*.csv"))

    if not signal_files:
        print("No signal files. Run Step 4 first.")
        exit(1)

    pair_signals = {}
    for f in signal_files:
        pair_name = f.stem.replace("signals_", "")
        pair_signals[pair_name] = pd.read_csv(f, index_col=0, parse_dates=True)

    bt = PortfolioBacktester(
        total_capital=1_000_000,
        transaction_cost_bps=10,
        slippage_bps=5
    )

    portfolio = bt.run(pair_signals, prices, verbose=True)
    summary = bt.get_portfolio_summary()

    print(f"\nPortfolio Summary:")
    for key, val in summary.items():
        if isinstance(val, float):
            print(f"   {key:>30s}: {val:>12.4f}")
        else:
            print(f"   {key:>30s}: {val}")

    print("\nBacktest test complete!")