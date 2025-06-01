import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

# Function to calculate Sharpe Ratio
def calculate_sharpe_ratio(returns, risk_free_rate=0.02):
    excess_returns = returns - risk_free_rate / 252  # Annual risk-free rate converted to daily
    return np.mean(excess_returns) / np.std(excess_returns) if np.std(excess_returns) != 0 else 0

# Function to calculate Value at Risk (VaR)
def calculate_var(returns, confidence_level=0.95):
    return np.percentile(returns, (1 - confidence_level) * 100)

# Function to calculate Conditional Value at Risk (CVaR) / Expected Shortfall
def calculate_cvar(returns, confidence_level=0.95):
    var = calculate_var(returns, confidence_level)
    return returns[returns <= var].mean() if len(returns[returns <= var]) > 0 else var

# Function to calculate Maximum Drawdown
def calculate_max_drawdown(prices):
    cumulative_returns = prices / prices.iloc[0] if not prices.empty else pd.Series([1])
    peak = np.maximum.accumulate(cumulative_returns)
    drawdown = (cumulative_returns - peak) / peak
    return np.min(drawdown) if len(drawdown) > 0 else 0

# Function to calculate Beta (market sensitivity)
def calculate_beta(stock_returns, market_returns):
    covariance = stock_returns.cov(market_returns)
    market_variance = market_returns.var()
    return covariance / market_variance if market_variance != 0 else 1

def show():
    st.title("Dynamic Portfolio Management")
    
    # Initialize portfolio data in session state if not present
    if 'portfolio_data' not in st.session_state:
        st.session_state.portfolio_data = {
            'AAPL': {'shares': 10, 'cost_basis': 150.00},
            'MSFT': {'shares': 5, 'cost_basis': 290.00},
            'GOOGL': {'shares': 2, 'cost_basis': 125.00}
        }
    
    # Create tabs for portfolio view and management
    tab1, tab2, tab3 = st.tabs(["Portfolio Dashboard", "Manage Portfolio", "Risk Analysis"])
    
    with tab1:
        # Portfolio Dashboard
        display_portfolio_dashboard(st.session_state.portfolio_data)
    
    with tab2:
        # Manage Portfolio
        manage_portfolio()
    
    with tab3:
        # Risk Analysis
        risk_analysis(st.session_state.portfolio_data)

def display_portfolio_dashboard(portfolio_data):
    st.subheader("Portfolio Dashboard")
    
    # Time period selection
    time_period = st.select_slider(
        "Select Time Period",
        options=["1m", "3m", "6m", "1y", "2y"],
        value="3m",
        key="dashboard_time_period"
    )
    
    # Fetch historical prices and current prices
    historical_prices, current_prices, portfolio_value, cost_basis_total = fetch_portfolio_data(portfolio_data, time_period)
    
    # If no data, show message
    if not historical_prices:
        st.warning("No portfolio data available. Please add stocks in the Manage Portfolio tab.")
        return
    
    # Calculate returns
    returns_df = pd.DataFrame(historical_prices).pct_change().dropna()
    
    # Calculate weighted portfolio returns
    weights = {}
    for ticker, data in portfolio_data.items():
        if ticker in current_prices:
            weights[ticker] = current_prices[ticker] * data['shares'] / portfolio_value if portfolio_value > 0 else 0
    
    portfolio_returns = pd.Series(0, index=returns_df.index)
    for ticker in weights:
        if ticker in returns_df.columns:
            portfolio_returns += returns_df[ticker] * weights.get(ticker, 0)
    
    # Portfolio performance
    total_gain_loss = portfolio_value - cost_basis_total
    total_gain_loss_percent = (total_gain_loss / cost_basis_total) * 100 if cost_basis_total > 0 else 0
    
    # Get market data (S&P 500) for benchmark
    try:
        spy = yf.Ticker("SPY")
        spy_hist = spy.history(period=time_period)['Close']
        spy_returns = spy_hist.pct_change().dropna()
        benchmark_return = ((spy_hist.iloc[-1] / spy_hist.iloc[0]) - 1) * 100
    except:
        spy_returns = pd.Series(0, index=portfolio_returns.index)
        benchmark_return = 0
    
    # Risk Metrics
    sharpe_ratio = calculate_sharpe_ratio(portfolio_returns)
    var_95 = calculate_var(portfolio_returns) * portfolio_value
    max_drawdown = calculate_max_drawdown(portfolio_returns.cumsum() + 1) * portfolio_value
    beta = calculate_beta(portfolio_returns, spy_returns)
    portfolio_return = ((1 + portfolio_returns).prod() - 1) * 100
    
    # Display portfolio summary
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Portfolio Value", f"${portfolio_value:.2f}")
    with col2:
        st.metric("Total Gain/Loss", f"${total_gain_loss:.2f}", f"{total_gain_loss_percent:.2f}%")
    with col3:
        st.metric("Return (Period)", f"{portfolio_return:.2f}%", f"{portfolio_return - benchmark_return:.2f}% vs S&P 500")
    
    # Portfolio composition
    st.subheader("Portfolio Composition")
    
    if portfolio_data:
        portfolio_df = create_portfolio_dataframe(portfolio_data, current_prices)
        
        # Display portfolio table
        st.dataframe(portfolio_df.style.format({
            'Cost Basis': '${:.2f}',
            'Current Price': '${:.2f}',
            'Market Value': '${:.2f}',
            'Gain/Loss': '${:.2f}',
            'Gain/Loss %': '{:.2f}%',
            'Allocation': '{:.2f}%'
        }))
        
        # Portfolio allocation pie chart
        fig = px.pie(portfolio_df, values='Market Value', names='Symbol', title='Portfolio Allocation')
        st.plotly_chart(fig, use_container_width=True)
        
        # Risk metrics
        risk_col1, risk_col2, risk_col3, risk_col4 = st.columns(4)
        with risk_col1:
            st.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")
        with risk_col2:
            st.metric("Beta", f"{beta:.2f}")
        with risk_col3:
            st.metric("95% Daily VaR", f"${abs(var_95):.2f}")
        with risk_col4:
            st.metric("Max Drawdown", f"${abs(max_drawdown):.2f}")
        
        # Historical performance chart
        st.subheader("Performance Comparison")
        
        # Create cumulative return series
        cum_returns = (1 + returns_df).cumprod()
        cum_portfolio = (1 + portfolio_returns).cumprod()
        cum_spy = (1 + spy_returns).cumprod()
        
        # Create performance chart
        fig = go.Figure()
        
        # Add portfolio line
        fig.add_trace(go.Scatter(
            x=cum_portfolio.index,
            y=cum_portfolio.values,
            mode='lines',
            name='Portfolio',
            line=dict(color='blue', width=3)
        ))
        
        # Add S&P 500 line
        fig.add_trace(go.Scatter(
            x=cum_spy.index,
            y=cum_spy.values,
            mode='lines',
            name='S&P 500',
            line=dict(color='red', width=2, dash='dash')
        ))
        
        # Add individual stocks
        for ticker in portfolio_data:
            if ticker in cum_returns.columns:
                fig.add_trace(go.Scatter(
                    x=cum_returns.index,
                    y=cum_returns[ticker].values,
                    mode='lines',
                    name=ticker,
                    line=dict(width=1.5),
                    visible='legendonly'  # Hidden by default
                ))
        
        fig.update_layout(
            title='Cumulative Returns Comparison',
            xaxis_title='Date',
            yaxis_title='Growth of $1',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        st.plotly_chart(fig, use_container_width=True)

def manage_portfolio():
    st.subheader("Manage Your Portfolio")
    
    # Display current portfolio
    st.write("Current Portfolio:")
    if st.session_state.portfolio_data:
        current_data = []
        for symbol, data in st.session_state.portfolio_data.items():
            current_data.append({
                "Symbol": symbol,
                "Shares": data['shares'],
                "Cost Basis": f"${data['cost_basis']:.2f}"
            })
        
        st.table(pd.DataFrame(current_data))
    else:
        st.info("Your portfolio is empty. Add some stocks below.")
    
    # Add new stock
    st.write("Add or Update Stock:")
    with st.form("add_stock_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            new_symbol = st.text_input("Stock Symbol", key="new_symbol").upper()
        with col2:
            new_shares = st.number_input("Number of Shares", min_value=0.0, step=0.01, key="new_shares")
        with col3:
            new_cost_basis = st.number_input("Cost Basis (per share)", min_value=0.0, step=0.01, key="new_cost_basis")
        
        submit_button = st.form_submit_button("Add/Update Stock")
        
        if submit_button and new_symbol and new_shares > 0 and new_cost_basis > 0:
            # Validate stock symbol
            try:
                stock = yf.Ticker(new_symbol)
                info = stock.info
                if 'regularMarketPrice' in info and info['regularMarketPrice'] is not None:
                    st.session_state.portfolio_data[new_symbol] = {
                        'shares': new_shares,
                        'cost_basis': new_cost_basis
                    }
                    st.success(f"Added {new_shares} shares of {new_symbol} at ${new_cost_basis:.2f}")
                else:
                    st.error(f"Invalid stock symbol: {new_symbol}")
            except:
                st.error(f"Could not validate stock symbol: {new_symbol}")
    
    # Remove stock
    st.write("Remove Stock:")
    if st.session_state.portfolio_data:
        stock_to_remove = st.selectbox("Select Stock to Remove", options=list(st.session_state.portfolio_data.keys()))
        if st.button("Remove Selected Stock"):
            if stock_to_remove in st.session_state.portfolio_data:
                del st.session_state.portfolio_data[stock_to_remove]
                st.success(f"Removed {stock_to_remove} from portfolio")
    else:
        st.info("No stocks to remove.")
    
    # Clear entire portfolio
    if st.session_state.portfolio_data and st.button("Clear Entire Portfolio"):
        st.session_state.portfolio_data = {}
        st.success("Portfolio cleared")

def risk_analysis(portfolio_data):
    st.subheader("Advanced Risk Analysis")
    
    # Time period selection for risk analysis
    time_period = st.select_slider(
        "Analysis Time Period",
        options=["1m", "3m", "6m", "1y", "2y"],
        value="1y",
        key="risk_time_period"
    )
    
    # Risk-free rate input
    risk_free_rate = st.slider("Risk-Free Rate (%)", min_value=0.0, max_value=5.0, value=2.0, step=0.1) / 100
    
    # Confidence level for VaR
    confidence_level = st.slider("Confidence Level for VaR/CVaR (%)", min_value=90, max_value=99, value=95) / 100
    
    # Fetch historical prices and current prices
    historical_prices, current_prices, portfolio_value, cost_basis_total = fetch_portfolio_data(portfolio_data, time_period)
    
    if not historical_prices:
        st.warning("No portfolio data available. Please add stocks in the Manage Portfolio tab.")
        return
    
    # Calculate returns
    returns_df = pd.DataFrame(historical_prices).pct_change().dropna()
    
    # Calculate weighted portfolio returns
    weights = {}
    for ticker, data in portfolio_data.items():
        if ticker in current_prices:
            weights[ticker] = current_prices[ticker] * data['shares'] / portfolio_value if portfolio_value > 0 else 0
    
    portfolio_returns = pd.Series(0, index=returns_df.index)
    for ticker in weights:
        if ticker in returns_df.columns:
            portfolio_returns += returns_df[ticker] * weights.get(ticker, 0)
    
    # Get market data (S&P 500) for benchmark
    try:
        spy = yf.Ticker("SPY")
        spy_hist = spy.history(period=time_period)['Close']
        spy_returns = spy_hist.pct_change().dropna()
    except:
        spy_returns = pd.Series(0, index=portfolio_returns.index)
    
    # Calculate risk metrics
    sharpe = calculate_sharpe_ratio(portfolio_returns, risk_free_rate)
    var_daily = calculate_var(portfolio_returns, confidence_level)
    var_dollar = var_daily * portfolio_value
    cvar_daily = calculate_cvar(portfolio_returns, confidence_level)
    cvar_dollar = cvar_daily * portfolio_value
    max_dd = calculate_max_drawdown(portfolio_returns.cumsum() + 1)
    max_dd_dollar = max_dd * portfolio_value
    beta = calculate_beta(portfolio_returns, spy_returns)
    
    # Calculate annualized metrics
    annual_return = ((1 + portfolio_returns).prod() ** (252 / len(portfolio_returns)) - 1) * 100
    annual_volatility = portfolio_returns.std() * np.sqrt(252) * 100
    
    # Display risk metrics
    st.subheader("Portfolio Risk Metrics")
    
    metrics_col1, metrics_col2 = st.columns(2)
    
    with metrics_col1:
        st.metric("Annualized Return", f"{annual_return:.2f}%")
        st.metric("Annualized Volatility", f"{annual_volatility:.2f}%")
        st.metric("Sharpe Ratio", f"{sharpe:.3f}")
        st.metric("Beta (vs S&P 500)", f"{beta:.3f}")
    
    with metrics_col2:
        st.metric(f"Value at Risk ({confidence_level*100:.0f}%)", f"${abs(var_dollar):.2f} / {abs(var_daily*100):.2f}%")
        st.metric(f"Conditional VaR ({confidence_level*100:.0f}%)", f"${abs(cvar_dollar):.2f} / {abs(cvar_daily*100):.2f}%")
        st.metric("Maximum Drawdown", f"${abs(max_dd_dollar):.2f} / {abs(max_dd*100):.2f}%")
    
    # Create correlation heatmap
    st.subheader("Asset Correlation Matrix")
    
    if len(returns_df.columns) > 1:
        corr_matrix = returns_df.corr()
        fig = px.imshow(
            corr_matrix,
            text_auto=True,
            color_continuous_scale='RdBu_r',
            zmin=-1, zmax=1,
            aspect="equal"
        )
        fig.update_layout(title="Correlation Between Assets")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Need at least two assets to calculate correlations.")
    
    # Monte Carlo Simulation
    st.subheader("Portfolio Monte Carlo Simulation")
    
    num_simulations = st.slider("Number of Simulations", min_value=100, max_value=1000, value=200, step=100)
    forecast_days = st.slider("Forecast Days", min_value=30, max_value=365, value=90, step=30)
    
    if st.button("Run Monte Carlo Simulation"):
        with st.spinner("Running simulation..."):
            # Get portfolio expected return and volatility
            port_mean = portfolio_returns.mean()
            port_std = portfolio_returns.std()
            
            # Run simulation
            simulation_results = run_monte_carlo(portfolio_value, port_mean, port_std, num_simulations, forecast_days)
            
            # Calculate percentiles
            percentiles = [0.05, 0.25, 0.5, 0.75, 0.95]
            sim_percentiles = np.percentile(simulation_results[-1, :], [p * 100 for p in percentiles])
            
            # Plot results
            fig = go.Figure()
            
            # Plot some sample paths (first 50 or fewer)
            num_paths = min(50, num_simulations)
            for i in range(num_paths):
                fig.add_trace(go.Scatter(
                    y=simulation_results[:, i],
                    mode='lines',
                    line=dict(width=0.5, color='rgba(70, 130, 180, 0.2)'),
                    showlegend=False
                ))
            
            # Plot percentile lines
            percentile_colors = ['rgba(255,0,0,0.8)', 'rgba(255,165,0,0.8)', 'rgba(0,128,0,0.8)', 
                               'rgba(255,165,0,0.8)', 'rgba(255,0,0,0.8)']
            percentile_labels = ['5th', '25th', '50th (Median)', '75th', '95th']
            
            for i, p in enumerate(percentiles):
                y_values = np.percentile(simulation_results, p * 100, axis=1)
                fig.add_trace(go.Scatter(
                    y=y_values,
                    mode='lines',
                    line=dict(width=2, color=percentile_colors[i]),
                    name=f"{percentile_labels[i]} Percentile"
                ))
            
            # Add initial portfolio value as horizontal line
            fig.add_hline(y=portfolio_value, line_dash="dash", line_color="black", 
                        annotation_text="Current Value", annotation_position="bottom right")
            
            fig.update_layout(
                title=f'Monte Carlo Simulation: Portfolio Value Over {forecast_days} Days',
                xaxis_title='Days',
                yaxis_title='Portfolio Value ($)',
                showlegend=True
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Display forecast metrics
            st.subheader("Forecast Summary (End of Period)")
            
            forecast_cols = st.columns(5)
            for i, (percentile, value) in enumerate(zip(percentile_labels, sim_percentiles)):
                with forecast_cols[i]:
                    change = ((value / portfolio_value) - 1) * 100
                    color = "green" if change >= 0 else "red"
                    st.markdown(f"**{percentile} Percentile:**")
                    st.markdown(f"${value:.2f}")
                    st.markdown(f"<span style='color:{color}'>{change:+.2f}%</span>", unsafe_allow_html=True)

def fetch_portfolio_data(portfolio_data, time_period):
    """Fetch historical and current prices for portfolio stocks"""
    historical_prices = {}
    current_prices = {}
    portfolio_value = 0
    cost_basis_total = 0
    
    for ticker, data in portfolio_data.items():
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period=time_period)['Close']
            if not hist.empty:
                historical_prices[ticker] = hist
                current_price = hist.iloc[-1]
                current_prices[ticker] = current_price
                portfolio_value += current_price * data['shares']
                cost_basis_total += data['cost_basis'] * data['shares']
        except Exception as e:
            st.warning(f"Error fetching data for {ticker}: {e}")
    
    return historical_prices, current_prices, portfolio_value, cost_basis_total

def create_portfolio_dataframe(portfolio_data, current_prices):
    """Create a DataFrame with portfolio information"""
    portfolio_df = pd.DataFrame({
        'Symbol': list(portfolio_data.keys()),
        'Shares': [data['shares'] for data in portfolio_data.values()],
        'Cost Basis': [data['cost_basis'] for data in portfolio_data.values()],
    })
    
    # Add current prices (only for stocks we could get data for)
    portfolio_df['Current Price'] = portfolio_df['Symbol'].apply(lambda x: current_prices.get(x, 0))
    
    # Calculate market value and gain/loss
    portfolio_df['Market Value'] = portfolio_df['Shares'] * portfolio_df['Current Price']
    portfolio_df['Gain/Loss'] = portfolio_df['Market Value'] - (portfolio_df['Shares'] * portfolio_df['Cost Basis'])
    portfolio_df['Gain/Loss %'] = (portfolio_df['Gain/Loss'] / (portfolio_df['Shares'] * portfolio_df['Cost Basis'])) * 100
    
    # Calculate allocation
    total_value = portfolio_df['Market Value'].sum()
    portfolio_df['Allocation'] = (portfolio_df['Market Value'] / total_value) * 100 if total_value > 0 else 0
    
    return portfolio_df

def run_monte_carlo(initial_value, mean_return, std_return, num_simulations, days):
    """Run Monte Carlo simulation for portfolio value"""
    simulation_df = np.zeros((days + 1, num_simulations))
    simulation_df[0] = initial_value
    
    for i in range(num_simulations):
        random_returns = np.random.normal(loc=mean_return, scale=std_return, size=days)
        for day in range(1, days + 1):
            simulation_df[day, i] = simulation_df[day-1, i] * (1 + random_returns[day-1])
    
    return simulation_df