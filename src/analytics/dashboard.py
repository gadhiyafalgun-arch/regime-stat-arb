"""
Streamlit Dashboard
=====================
Interactive dashboard for the Regime-Aware Stat Arb system.

Run with: streamlit run src/analytics/dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from pathlib import Path
import json


# ============================================================
# Page Configuration
# ============================================================
st.set_page_config(
    page_title="Regime-Aware Stat Arb",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# Data Loading (cached)
# ============================================================
@st.cache_data
def load_all_data():
    """Load all results data."""
    data = {}

    # Prices
    p = Path("data/processed/close_prices.csv")
    if p.exists():
        data['prices'] = pd.read_csv(p, index_col=0, parse_dates=True)

    # Regime labels
    p = Path("data/results/regime_labels.csv")
    if p.exists():
        data['regimes'] = pd.read_csv(p, index_col=0, parse_dates=True)

    # Backtest equity (try optimized first)
    for fname in ['backtest_optimized_equity.csv', 'backtest_equity.csv']:
        p = Path(f"data/results/{fname}")
        if p.exists():
            data['backtest'] = pd.read_csv(p, index_col=0, parse_dates=True)
            break

    # Pair comparison
    p = Path("data/results/backtest_pair_comparison.csv")
    if p.exists():
        data['pair_comparison'] = pd.read_csv(p)

    # Monthly returns
    p = Path("data/results/backtest_monthly_returns.csv")
    if p.exists():
        data['monthly'] = pd.read_csv(p, index_col=0)

    # Trade log
    p = Path("data/results/backtest_trades.csv")
    if p.exists():
        data['trades'] = pd.read_csv(p)

    # Monte Carlo
    p = Path("data/results/monte_carlo_paths.csv")
    if p.exists():
        data['mc_paths'] = pd.read_csv(p)
    p = Path("data/results/monte_carlo_results.csv")
    if p.exists():
        data['mc_stats'] = pd.read_csv(p)

    # Regime performance
    p = Path("data/results/regime_performance.csv")
    if p.exists():
        data['regime_perf'] = pd.read_csv(p)

    # Signal files
    signals_dir = Path("data/results/signals")
    if signals_dir.exists():
        data['signals'] = {}
        for f in signals_dir.glob("signals_*.csv"):
            pair = f.stem.replace("signals_", "")
            data['signals'][pair] = pd.read_csv(f, index_col=0, parse_dates=True)

    # Regime map
    p = Path("data/results/regime_map.json")
    if p.exists():
        with open(p) as f:
            data['regime_map'] = json.load(f)

    # Tuning results
    p = Path("data/results/tuning_results.csv")
    if p.exists():
        data['tuning'] = pd.read_csv(p)

    return data


# ============================================================
# Sidebar
# ============================================================
def render_sidebar(data):
    """Render sidebar with navigation and filters."""
    st.sidebar.title("📊 Regime Stat Arb")
    st.sidebar.markdown("---")

    page = st.sidebar.radio(
        "Navigation",
        ["Overview", "Regime Analysis", "Pair Explorer",
         "Backtest Results", "Monte Carlo", "Parameter Tuning"]
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Project:** Regime-Aware Statistical Arbitrage")
    st.sidebar.markdown("**GitHub:** gadhiyafalgun-arch")

    if 'backtest' in data:
        equity = data['backtest']['equity']
        total_ret = (equity.iloc[-1] / equity.iloc[0] - 1) * 100
        st.sidebar.metric("Total Return", f"{total_ret:.2f}%")

    return page


# ============================================================
# Pages
# ============================================================
def page_overview(data):
    """Main overview dashboard."""
    st.title("🎯 Regime-Aware Statistical Arbitrage Engine")
    st.markdown("*Detecting market regimes and trading cointegrated pairs with adaptive strategies*")

    # Key metrics row
    col1, col2, col3, col4, col5 = st.columns(5)

    if 'backtest' in data:
        bt = data['backtest']
        equity = bt['equity']
        returns = bt['daily_return']

        total_ret = (equity.iloc[-1] / equity.iloc[0] - 1) * 100
        ann_ret = returns.mean() * 252 * 100
        ann_vol = returns.std() * np.sqrt(252) * 100
        sharpe = (returns.mean() * 252) / (returns.std() * np.sqrt(252)) if returns.std() > 0 else 0
        peak = equity.cummax()
        max_dd = ((equity - peak) / peak).min() * 100

        col1.metric("Total Return", f"{total_ret:.2f}%")
        col2.metric("Ann. Return", f"{ann_ret:.2f}%")
        col3.metric("Sharpe Ratio", f"{sharpe:.3f}")
        col4.metric("Max Drawdown", f"{max_dd:.2f}%")
        col5.metric("Ann. Volatility", f"{ann_vol:.2f}%")

    st.markdown("---")

    # Equity curve
    if 'backtest' in data:
        st.subheader("Portfolio Equity Curve")
        bt = data['backtest']

        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                           vertical_spacing=0.05,
                           row_heights=[0.7, 0.3],
                           subplot_titles=("Equity", "Drawdown"))

        fig.add_trace(
            go.Scatter(x=bt.index, y=bt['equity'],
                      mode='lines', name='Portfolio',
                      line=dict(color='steelblue', width=1.5)),
            row=1, col=1
        )

        peak = bt['equity'].cummax()
        dd = (bt['equity'] - peak) / peak * 100
        fig.add_trace(
            go.Scatter(x=bt.index, y=dd,
                      fill='tozeroy', name='Drawdown',
                      line=dict(color='red', width=0.5)),
            row=2, col=1
        )

        fig.update_layout(height=600, showlegend=True)
        fig.update_yaxes(title_text="Equity ($)", row=1, col=1)
        fig.update_yaxes(title_text="DD (%)", row=2, col=1)
        st.plotly_chart(fig, use_container_width=True)

    # Regime timeline
    if 'regimes' in data and 'prices' in data and 'SPY' in data['prices'].columns:
        st.subheader("Market Regime Detection (SPY)")
        regimes = data['regimes']
        spy = data['prices']['SPY']

        common = spy.index.intersection(regimes.index)
        spy_aligned = spy.reindex(common)
        reg_aligned = regimes.reindex(common)

        regime_col = 'regime_name_smoothed' if 'regime_name_smoothed' in reg_aligned.columns else 'regime_name'

        fig = go.Figure()
        colors = {'Bull': '#2ecc71', 'Bear': '#e74c3c', 'Crisis': '#9b59b6'}
        for regime, color in colors.items():
            mask = reg_aligned[regime_col] == regime
            if mask.sum() > 0:
                fig.add_trace(go.Scatter(
                    x=spy_aligned.index[mask],
                    y=spy_aligned.values[mask],
                    mode='markers', marker=dict(size=2, color=color),
                    name=regime
                ))

        fig.update_layout(height=400, title="SPY Colored by Regime",
                         yaxis_title="Price ($)")
        st.plotly_chart(fig, use_container_width=True)


def page_regime_analysis(data):
    """Regime-specific analysis."""
    st.title("🔍 Regime Analysis")

    if 'regime_perf' in data:
        st.subheader("Performance by Regime")
        rp = data['regime_perf']
        st.dataframe(rp.style.format({
            'days': '{:.0f}',
            'pct_of_total': '{:.1f}%',
            'total_return_pct': '{:.2f}%',
            'ann_return_pct': '{:.2f}%',
            'ann_vol_pct': '{:.2f}%',
            'sharpe': '{:.3f}',
            'max_dd_pct': '{:.2f}%',
            'positive_days_pct': '{:.1f}%',
        }), use_container_width=True)

        # Bar charts
        col1, col2 = st.columns(2)
        with col1:
            fig = px.bar(rp, x='regime', y='sharpe',
                        color='regime',
                        color_discrete_map={'Bull': '#2ecc71', 'Bear': '#e74c3c', 'Crisis': '#9b59b6'},
                        title="Sharpe Ratio by Regime")
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = px.bar(rp, x='regime', y='total_return_pct',
                        color='regime',
                        color_discrete_map={'Bull': '#2ecc71', 'Bear': '#e74c3c', 'Crisis': '#9b59b6'},
                        title="Total Return (%) by Regime")
            st.plotly_chart(fig, use_container_width=True)

    # Regime probabilities over time
    if 'regimes' in data:
        st.subheader("Regime Probabilities Over Time")
        regimes = data['regimes']
        prob_cols = [c for c in regimes.columns if c.startswith('prob_')]

        if prob_cols:
            fig = go.Figure()
            colors = {'prob_bull': '#2ecc71', 'prob_bear': '#e74c3c', 'prob_crisis': '#9b59b6'}
            for col in prob_cols:
                name = col.replace('prob_', '').title()
                fig.add_trace(go.Scatter(
                    x=regimes.index, y=regimes[col],
                    stackgroup='one', name=name,
                    line=dict(color=colors.get(col, 'gray'), width=0)
                ))
            fig.update_layout(height=400, yaxis_title="Probability",
                            title="Stacked Regime Probabilities")
            st.plotly_chart(fig, use_container_width=True)


def page_pair_explorer(data):
    """Interactive pair exploration."""
    st.title("🔗 Pair Explorer")

    if 'signals' not in data or not data['signals']:
        st.warning("No signal data found.")
        return

    pair_names = sorted(data['signals'].keys())
    selected_pair = st.selectbox("Select Pair", pair_names)

    if selected_pair:
        signals = data['signals'][selected_pair]
        valid = signals.dropna(subset=['zscore'])

        if len(valid) == 0:
            st.warning("No valid z-score data for this pair.")
            return

        # Z-score chart
        st.subheader(f"Z-Score & Signals: {selected_pair}")

        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                           vertical_spacing=0.05,
                           row_heights=[0.7, 0.3],
                           subplot_titles=("Z-Score", "Position"))

        fig.add_trace(
            go.Scatter(x=valid.index, y=valid['zscore'],
                      mode='lines', name='Z-Score',
                      line=dict(color='steelblue', width=1)),
            row=1, col=1
        )

        # Thresholds
        for thresh in [2.0, -2.0, 1.5, -1.5]:
            fig.add_hline(y=thresh, line_dash="dash",
                         line_color="red" if abs(thresh) == 2.0 else "orange",
                         opacity=0.4, row=1, col=1)
        fig.add_hline(y=0, line_color="black", opacity=0.3, row=1, col=1)

        # Entries
        if 'signal_type' in valid.columns:
            entries_long = valid[valid['signal_type'] == 'entry_long']
            entries_short = valid[valid['signal_type'] == 'entry_short']
            if len(entries_long) > 0:
                fig.add_trace(go.Scatter(
                    x=entries_long.index, y=entries_long['zscore'],
                    mode='markers', name='Long Entry',
                    marker=dict(symbol='triangle-up', size=8, color='green')
                ), row=1, col=1)
            if len(entries_short) > 0:
                fig.add_trace(go.Scatter(
                    x=entries_short.index, y=entries_short['zscore'],
                    mode='markers', name='Short Entry',
                    marker=dict(symbol='triangle-down', size=8, color='red')
                ), row=1, col=1)

        # Position
        if 'position' in valid.columns:
            fig.add_trace(
                go.Scatter(x=valid.index, y=valid['position'],
                          fill='tozeroy', name='Position',
                          line=dict(color='steelblue', width=1)),
                row=2, col=1
            )

        fig.update_layout(height=500, showlegend=True)
        st.plotly_chart(fig, use_container_width=True)

        # Pair stats
        if 'pair_comparison' in data:
            pc = data['pair_comparison']
            pair_row = pc[pc['pair'] == selected_pair]
            if len(pair_row) > 0:
                st.subheader("Pair Statistics")
                stats = pair_row.iloc[0]
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Total PnL", f"${stats.get('total_pnl', 0):,.2f}")
                c2.metric("Sharpe", f"{stats.get('sharpe', 0):.3f}")
                c3.metric("Win Rate", f"{stats.get('win_rate', 0):.1f}%")
                c4.metric("Trades", f"{stats.get('n_trades', 0):.0f}")


def page_backtest_results(data):
    """Detailed backtest results."""
    st.title("🏗️ Backtest Results")

    if 'pair_comparison' in data:
        st.subheader("Pair-by-Pair Performance")
        pc = data['pair_comparison']

        fig = make_subplots(rows=1, cols=3,
                           subplot_titles=("Total PnL ($)", "Sharpe Ratio", "Win Rate (%)"))

        colors = ['green' if v > 0 else 'red' for v in pc['total_pnl']]
        fig.add_trace(go.Bar(x=pc['pair'], y=pc['total_pnl'],
                            marker_color=colors, name='PnL'), row=1, col=1)

        colors = ['green' if v > 0 else 'red' for v in pc['sharpe']]
        fig.add_trace(go.Bar(x=pc['pair'], y=pc['sharpe'],
                            marker_color=colors, name='Sharpe'), row=1, col=2)

        fig.add_trace(go.Bar(x=pc['pair'], y=pc['win_rate'],
                            marker_color='steelblue', name='WR'), row=1, col=3)
        fig.add_hline(y=50, line_dash="dash", line_color="orange",
                     opacity=0.5, row=1, col=3)

        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    if 'monthly' in data:
        st.subheader("Monthly Returns (%)")
        monthly = data['monthly']
        st.dataframe(monthly.style.background_gradient(
            cmap='RdYlGn', vmin=-3, vmax=3
        ).format("{:.2f}%", na_rep=""), use_container_width=True)

    if 'trades' in data:
        st.subheader("Trade Log")
        trades = data['trades']
        st.dataframe(trades.tail(50), use_container_width=True)

        # Trade PnL distribution
        if 'pnl' in trades.columns:
            fig = px.histogram(trades, x='pnl', nbins=50,
                             title="Trade PnL Distribution",
                             color_discrete_sequence=['steelblue'])
            fig.add_vline(x=0, line_dash="dash", line_color="red")
            st.plotly_chart(fig, use_container_width=True)


def page_monte_carlo(data):
    """Monte Carlo simulation results."""
    st.title("🎲 Monte Carlo Stress Test")

    if 'mc_stats' not in data:
        st.warning("No Monte Carlo results found. Run: `python -m src.analytics.monte_carlo`")
        return

    stats = data['mc_stats'].iloc[0]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("P(Profit)", f"{stats.get('prob_profit', 0):.1f}%")
    col2.metric("Median Return", f"{stats.get('median_return_pct', 0):.2f}%")
    col3.metric("P(Loss > 10%)", f"{stats.get('prob_loss_10pct', 0):.1f}%")
    col4.metric("Worst Max DD", f"{stats.get('worst_max_drawdown_pct', 0):.1f}%")

    if 'mc_paths' in data:
        st.subheader("Simulated Equity Paths")
        paths = data['mc_paths']

        fig = go.Figure()
        if 'p5' in paths.columns:
            fig.add_trace(go.Scatter(y=paths['p5'], mode='lines',
                                    name='5th pct (worst)', line=dict(color='red', dash='dash')))
        if 'p25' in paths.columns:
            fig.add_trace(go.Scatter(y=paths['p25'], mode='lines',
                                    name='25th pct', line=dict(color='orange', dash='dot')))
        if 'p50' in paths.columns:
            fig.add_trace(go.Scatter(y=paths['p50'], mode='lines',
                                    name='Median', line=dict(color='steelblue', width=2)))
        if 'p75' in paths.columns:
            fig.add_trace(go.Scatter(y=paths['p75'], mode='lines',
                                    name='75th pct', line=dict(color='green', dash='dot')))
        if 'p95' in paths.columns:
            fig.add_trace(go.Scatter(y=paths['p95'], mode='lines',
                                    name='95th pct (best)', line=dict(color='darkgreen', dash='dash')))

        fig.update_layout(height=500, xaxis_title="Trading Days",
                         yaxis_title="Equity ($)",
                         title="Monte Carlo: 1-Year Simulated Paths")
        st.plotly_chart(fig, use_container_width=True)

    # Return distribution
    st.subheader("Return Distribution (1-Year)")
    pcts = {k: v for k, v in stats.items() if 'percentile' in k and 'return' in k}
    if pcts:
        pct_df = pd.DataFrame([
            {'Percentile': k.replace('percentile_', '').replace('_return_pct', 'th'),
             'Return (%)': v}
            for k, v in sorted(pcts.items())
        ])
        st.dataframe(pct_df, use_container_width=True)


def page_parameter_tuning(data):
    """Parameter tuning results."""
    st.title("🔧 Parameter Tuning")

    if 'tuning' not in data:
        st.warning("No tuning results. Run: `python -m src.backtest.strategy_tuner`")
        return

    tuning = data['tuning'].sort_values('sharpe', ascending=False)

    st.subheader("Top 20 Parameter Combinations")
    display_cols = ['n_pairs', 'entry_bull', 'entry_bear', 'min_holding',
                   'zscore_lookback', 'delta', 'sharpe', 'total_return_pct',
                   'total_trades', 'win_rate', 'max_dd']
    available = [c for c in display_cols if c in tuning.columns]
    st.dataframe(tuning[available].head(20), use_container_width=True)

    # Scatter: Sharpe vs Return
    st.subheader("Sharpe vs Return (all combinations)")
    fig = px.scatter(tuning, x='total_return_pct', y='sharpe',
                    color='n_pairs', size='total_trades',
                    hover_data=['entry_bull', 'entry_bear', 'min_holding'],
                    title="Parameter Space Exploration",
                    color_continuous_scale='viridis')
    fig.add_hline(y=0, line_dash="dash", line_color="red", opacity=0.5)
    fig.add_vline(x=0, line_dash="dash", line_color="red", opacity=0.5)
    st.plotly_chart(fig, use_container_width=True)

    # Best params highlight
    best = tuning.iloc[0]
    st.subheader("🏆 Best Parameters")
    c1, c2, c3 = st.columns(3)
    c1.metric("Top N Pairs", f"{best['n_pairs']:.0f}")
    c2.metric("Entry (Bull)", f"{best['entry_bull']:.1f}")
    c3.metric("Entry (Bear)", f"{best['entry_bear']:.1f}")
    c4, c5, c6 = st.columns(3)
    c4.metric("Min Hold", f"{best['min_holding']:.0f} days")
    c5.metric("Z-Score Lookback", f"{best['zscore_lookback']:.0f}")
    c6.metric("Kalman Delta", f"{best['delta']:.0e}")


# ============================================================
# Main
# ============================================================
def main():
    data = load_all_data()
    page = render_sidebar(data)

    if page == "Overview":
        page_overview(data)
    elif page == "Regime Analysis":
        page_regime_analysis(data)
    elif page == "Pair Explorer":
        page_pair_explorer(data)
    elif page == "Backtest Results":
        page_backtest_results(data)
    elif page == "Monte Carlo":
        page_monte_carlo(data)
    elif page == "Parameter Tuning":
        page_parameter_tuning(data)


if __name__ == "__main__":
    main()