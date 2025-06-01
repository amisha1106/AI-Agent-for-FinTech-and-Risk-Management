import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from plotly.subplots import make_subplots

def show():
    st.title("Company Dashboard")
    
    # Sample company data - in a real app, this would be loaded from a database
    company_data = {
        'name': 'XYZ Technology Inc.',
        'symbol': 'XYZ',
        'sector': 'Technology',
        'listed_date': '2020-01-15',
        'current_price': 45.67,
        'market_cap': 1230000000,
        'risk_percentage': 68,
    }
    
    tabs = st.tabs(["Company Overview", "Stock Management", "Earnings Reports", "Analytics"])
    
    with tabs[0]:
        show_company_overview(company_data)
    
    with tabs[1]:
        show_stock_management(company_data)
    
    with tabs[2]:
        show_earnings_reports(company_data)
    
    with tabs[3]:
        show_company_analytics(company_data)

def show_company_overview(company_data):
    st.header("Company Overview")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader(company_data['name'])
        st.write(f"**Symbol:** {company_data['symbol']}")
        st.write(f"**Sector:** {company_data['sector']}")
        st.write(f"**Listed Date:** {company_data['listed_date']}")
        st.write(f"**Current Stock Price:** ${company_data['current_price']}")
        st.write(f"**Market Cap:** ${company_data['market_cap']:,}")
        
        # Risk disclosure
        st.warning(f"**Risk Assessment:** {company_data['risk_percentage']}% (Moderate to High Risk)")
    
    with col2:
        # Company logo placeholder
        st.image("https://via.placeholder.com/150", caption="Company Logo")
        
        # Quick stats
        st.metric("Daily Change", "+2.3%")
        st.metric("30-Day Volume", "1.2M shares")
    
    # Stock price chart (simulated)
    st.subheader("Stock Performance")
    
    # Generate simulated stock data
    dates = pd.date_range(end=pd.Timestamp.today(), periods=120)
    price = 25  # Starting price
    prices = [price]
    
    for _ in range(len(dates)-1):
        change = np.random.normal(0.001, 0.02)
        price *= (1 + change)
        prices.append(price)
    
    stock_df = pd.DataFrame({
        'Date': dates,
        'Close': prices
    })
    
    fig = px.line(stock_df, x='Date', y='Close', title=f"{company_data['symbol']} Stock Price")
    st.plotly_chart(fig, use_container_width=True)
    
    # Company description
    st.subheader("About the Company")
    st.write("""
    XYZ Technology Inc. is a leading provider of cloud-based software solutions for businesses of all sizes. 
    The company specializes in artificial intelligence, machine learning, and data analytics platforms that help 
    organizations make better decisions and improve operational efficiency.
    
    Founded in 2015, XYZ Technology has grown rapidly to become one of the most innovative companies in the tech sector, 
    with operations in over 30 countries and more than 2,500 employees worldwide.
    """)

def show_stock_management(company_data):
    st.header("Stock Management")
    
    # Add new stock listing
    st.subheader("Add New Stock Listing")
    
    with st.form("new_stock_listing"):
        stock_name = st.text_input("Stock Name", company_data['name'])
        stock_symbol = st.text_input("Stock Symbol", company_data['symbol'])
        stock_sector = st.selectbox("Sector", ["Technology", "Healthcare", "Finance", "Consumer Goods", "Energy", "Manufacturing", "Real Estate"])
        initial_price = st.number_input("Initial Price ($)", min_value=0.01, value=company_data['current_price'])
        shares_outstanding = st.number_input("Shares Outstanding", min_value=1000, value=10000000)
        risk_percentage = st.slider("Risk Percentage", 0, 100, company_data['risk_percentage'])
        
        st.info("Note: All new stock listings require admin approval before becoming active")
        
        submit_button = st.form_submit_button("Submit for Approval")
        if submit_button:
            st.success(f"Stock listing for {stock_name} ({stock_symbol}) has been submitted for approval")
    
    # Update stock price
    st.subheader("Update Stock Price")
    
    with st.form("update_stock_price"):
        current_price = st.number_input("Current Price ($)", min_value=0.01, value=company_data['current_price'])
        price_change_reason = st.text_area("Reason for Price Change", "Quarterly earnings exceeded expectations.")
        
        update_button = st.form_submit_button("Update Price")
        if update_button:
            st.success(f"Stock price for {company_data['symbol']} updated to ${current_price}")
    
    # Stock splits and dividends
    st.subheader("Stock Actions")
    
    col1, col2 = st.columns(2)
    
    with col1:
        with st.form("stock_split"):
            split_ratio = st.selectbox("Split Ratio", ["2:1", "3:1", "3:2", "4:1"])
            split_date = st.date_input("Split Date", datetime.now() + timedelta(days=30))
            
            split_button = st.form_submit_button("Announce Split")
            if split_button:
                st.success(f"Stock split ({split_ratio}) announced for {split_date}")
    
    with col2:
        with st.form("dividend"):
            dividend_amount = st.number_input("Dividend Amount ($)", min_value=0.01, value=0.25)
            dividend_date = st.date_input("Dividend Date", datetime.now() + timedelta(days=15))
            
            dividend_button = st.form_submit_button("Announce Dividend")
            if dividend_button:
                st.success(f"Dividend (${dividend_amount} per share) announced for {dividend_date}")

def show_earnings_reports(company_data):
    st.header("Earnings Reports")
    
    # Historical earnings data (simulated)
    quarters = ["Q1 2024", "Q4 2023", "Q3 2023", "Q2 2023", "Q1 2023", "Q4 2022"]
    revenue = [42.5, 38.7, 35.2, 33.9, 31.2, 28.5]  # In millions
    eps_actual = [0.42, 0.35, 0.31, 0.30, 0.28, 0.25]
    eps_estimate = [0.38, 0.36, 0.32, 0.29, 0.27, 0.26]
    
    earnings_df = pd.DataFrame({
        'Quarter': quarters,
        'Revenue (millions)': revenue,
        'EPS (Actual)': eps_actual,
        'EPS (Estimate)': eps_estimate,
        'Beat/Miss': [actual - estimate for actual, estimate in zip(eps_actual, eps_estimate)]
    })
    
    # Display earnings history
    st.subheader("Earnings History")
    st.dataframe(earnings_df, use_container_width=True)
    
    # Earnings visualization
    st.subheader("Earnings Visualization")
    
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                       vertical_spacing=0.1,
                       subplot_titles=('Revenue', 'Earnings Per Share'))
    
    # Revenue chart
    fig.add_trace(
        go.Bar(x=quarters, y=revenue, name="Revenue", marker_color="skyblue"),
        row=1, col=1
    )
    
    # EPS chart
    fig.add_trace(
        go.Scatter(x=quarters, y=eps_actual, name="EPS (Actual)", marker_color="green", mode="lines+markers"),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=quarters, y=eps_estimate, name="EPS (Estimate)", marker_color="red", line=dict(dash="dot"), mode="lines+markers"),
        row=2, col=1
    )
    
    fig.update_layout(height=500, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    st.plotly_chart(fig, use_container_width=True)
    
    # Submit new earnings report
    st.subheader("Submit New Earnings Report")
    
    with st.form("earnings_report"):
        report_quarter = st.selectbox("Quarter", ["Q2 2024", "Q3 2024", "Q4 2024"])
        report_revenue = st.number_input("Revenue (millions)", min_value=0.1, value=45.0)
        report_eps = st.number_input("Earnings Per Share", min_value=-10.0, value=0.45)
        report_notes = st.text_area("Earnings Notes", "Strong performance in our cloud services division drove revenue growth.")
        
        # Financial highlights
        st.subheader("Financial Highlights")
        col1, col2 = st.columns(2)
        
        with col1:
            gross_margin = st.number_input("Gross Margin (%)", min_value=0.0, max_value=100.0, value=65.0)
            operating_margin = st.number_input("Operating Margin (%)", min_value=-100.0, max_value=100.0, value=18.5)
        
        with col2:
            cash_balance = st.number_input("Cash Balance (millions)", min_value=0.0, value=125.0)
            debt = st.number_input("Total Debt (millions)", min_value=0.0, value=40.0)
        
        submit_button = st.form_submit_button("Submit Earnings Report")
        if submit_button:
            st.success(f"Earnings report for {report_quarter} submitted successfully")

def show_company_analytics(company_data):
    st.header("Company Analytics")
    
    # Stock performance vs market
    st.subheader("Stock Performance vs Market")
    
    # Generate simulated performance data
    dates = pd.date_range(end=pd.Timestamp.today(), periods=252)  # 1 year of trading days
    company_perf = 100 * np.cumprod(1 + np.random.normal(0.0008, 0.018, 252))
    market_perf = 100 * np.cumprod(1 + np.random.normal(0.0005, 0.012, 252))
    sector_perf = 100 * np.cumprod(1 + np.random.normal(0.0006, 0.015, 252))
    
    perf_df = pd.DataFrame({
        'Date': dates,
        'Company': company_perf,
        'Sector': sector_perf,
        'S&P 500': market_perf
    })
    
    fig = px.line(perf_df, x='Date', y=['Company', 'Sector', 'S&P 500'], 
                 title='Relative Performance (Normalized to 100)')
    st.plotly_chart(fig, use_container_width=True)
    
    # Investor demographics
    st.subheader("Investor Demographics")
    
    investor_data = {
        'Type': ['Institutional', 'Individual', 'Insider', 'Other'],
        'Percentage': [65, 25, 8, 2]
    }
    
    fig = px.pie(investor_data, values='Percentage', names='Type', title='Investor Distribution')
    st.plotly_chart(fig, use_container_width=True)
    
    # Trading volume analysis
    st.subheader("Trading Volume Analysis")
    
    # Generate simulated volume data
    volume = np.abs(np.random.normal(500000, 200000, len(dates)))
    price = company_perf / 2  # Scale for visualization
    
    volume_df = pd.DataFrame({
        'Date': dates,
        'Volume': volume,
        'Price': price
    })
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig.add_trace(
        go.Bar(x=volume_df['Date'], y=volume_df['Volume'], name="Volume", marker_color="lightblue"),
        secondary_y=False
    )
    
    fig.add_trace(
        go.Scatter(x=volume_df['Date'], y=volume_df['Price'], name="Price", marker_color="red"),
        secondary_y=True
    )
    
    fig.update_layout(title="Stock Price vs Trading Volume", 
                     xaxis_title="Date",
                     legend=dict(orientation="h"))
    
    fig.update_yaxes(title_text="Volume", secondary_y=False)
    fig.update_yaxes(title_text="Price", secondary_y=True)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Risk analysis
    st.subheader("Risk Analysis")
    
    # Beta calculation (simulated)
    beta = 1.25
    volatility = 28.4  # Annualized volatility in percentage
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Beta", f"{beta:.2f}", "+0.15")
    with col2:
        st.metric("Volatility", f"{volatility:.1f}%")
    with col3:
        st.metric("Risk Rating", f"{company_data['risk_percentage']}%")
    
    # Risk explanation
    st.info("""
    **What these metrics mean:**
    - **Beta**: Measures stock's volatility relative to the market. A beta > 1 indicates higher volatility than the market.
    - **Volatility**: Annualized standard deviation of returns, indicating price fluctuation magnitude.
    - **Risk Rating**: Our proprietary risk assessment combining multiple factors including financial stability, market conditions, and industry outlook.
    """)
    
    # Monthly returns heatmap
    st.subheader("Monthly Returns Heatmap")
    
    # Create monthly returns data (simulated)
    years = ['2021', '2022', '2023', '2024']
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    # Create a list of all year-month combinations
    periods = [(year, month) for year in years for month in months]
    
    # Generate random returns (-15% to +15%)
    returns = np.random.uniform(-15, 15, len(periods))
    
    # Create dataframe
    heatmap_data = []
    for i, ((year, month), ret) in enumerate(zip(periods, returns)):
        # Only include data up to current month
        if year == '2024' and months.index(month) > datetime.now().month - 1:
            continue
        heatmap_data.append({'Year': year, 'Month': month, 'Return': ret})
    
    heatmap_df = pd.DataFrame(heatmap_data)
    
    # Create pivot table for heatmap
    pivot_df = heatmap_df.pivot(index='Year', columns='Month', values='Return')
    
    # Reorder months chronologically
    pivot_df = pivot_df[months]
    
    # Create heatmap using plotly
    fig = go.Figure(data=go.Heatmap(
        z=pivot_df.values,
        x=pivot_df.columns,
        y=pivot_df.index,
        colorscale='RdBu_r',  # Red for negative, blue for positive
        zmid=0,  # Center the color scale at zero
        text=[[f"{value:.1f}%" for value in row] for row in pivot_df.values],
        texttemplate="%{text}",
        textfont={"size":10},
    ))
    
    fig.update_layout(title="Monthly Returns (%)")
    st.plotly_chart(fig, use_container_width=True)