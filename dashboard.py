import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import requests
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import time

# Initialize sentiment analyzer
sentiment_analyzer = SentimentIntensityAnalyzer()

# Add auto-refresh functionality
def get_refresh_interval():
    intervals = {
        "Off": None,
        "5 seconds": 5,
        "10 seconds": 10,
        "30 seconds": 30,
        "1 minute": 60,
        "5 minutes": 300
    }
    return intervals[st.sidebar.selectbox("Auto-refresh", list(intervals.keys()), index=3)]

def show():
    st.title("Real-time Stock Market Dashboard")
    
    # Add last updated timestamp
    now = datetime.now()
    st.sidebar.write(f"Last updated: {now.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Setup auto-refresh
    refresh_interval = get_refresh_interval()
    if refresh_interval:
        st.sidebar.write(f"Refreshing every {refresh_interval} seconds")
        st.sidebar.write("Next refresh in:")
        progress_bar = st.sidebar.progress(0)
        
        # Only show this if JavaScript is enabled in the sidebar
        st.sidebar.markdown("""
        <script>
            let countDown = 0;
            let maxCount = arguments[0];
            
            function updateProgressBar() {
                countDown += 1;
                let progress = countDown / maxCount;
                
                if (window.parent.document.querySelector('div[data-testid="stSidebarUserContent"] progress')) {
                    window.parent.document.querySelector('div[data-testid="stSidebarUserContent"] progress').value = progress;
                }
                
                if (countDown >= maxCount) {
                    window.location.reload();
                } else {
                    setTimeout(updateProgressBar, 1000);
                }
            }
            
            setTimeout(updateProgressBar, 1000);
        </script>
        """, unsafe_allow_html=True)
    
    # Initialize session state for pending listings if it doesn't exist
    if 'pending_stock_listings' not in st.session_state:
        st.session_state.pending_stock_listings = pd.DataFrame({
            'Company': ['XYZ Corp', 'ABC Inc', 'Tech Solutions'],
            'Symbol': ['XYZ', 'ABC', 'TECH'],
            'Date Submitted': ['2025-03-15', '2025-03-17', '2025-03-18'],
            'Initial Price': [25.50, 30.75, 45.20],
            'Risk Percentage': [35, 42, 58],
            'Status': ['Pending', 'Pending', 'Pending'],
            'Submitted By': ['user1', 'user2', 'user3']
        })
    
    # Initialize session state for approved listings if it doesn't exist
    if 'approved_stock_listings' not in st.session_state:
        st.session_state.approved_stock_listings = pd.DataFrame({
            'Company': ['Example Corp', 'Sample Inc'],
            'Symbol': ['EXA', 'SAMP'],
            'Date Approved': ['2025-03-10', '2025-03-12'],
            'Initial Price': [50.25, 32.10],
            'Risk Percentage': [40, 25],
            'Status': ['Active', 'Active'],
            'Approved By': ['admin1', 'admin1']
        })
    
    # Different dashboard views based on user type
    if st.session_state.get('user_role') == 'admin':
        admin_dashboard()
    elif st.session_state.get('user_role') == 'company':
        company_dashboard()
    else:
        investor_dashboard()

def admin_dashboard():
    st.header("Admin Dashboard")
    
    # Pending approvals section with enhanced functionality
    st.subheader("Pending Stock Listings")
    
    if len(st.session_state.pending_stock_listings) == 0:
        st.info("No pending stock listings at this time.")
    else:
        # Display pending listings with status filter
        status_filter = st.radio("Filter by status", ["All", "Pending", "Approved", "Rejected"], horizontal=True)
        
        filtered_df = st.session_state.pending_stock_listings
        if status_filter != "All":
            filtered_df = filtered_df[filtered_df["Status"] == status_filter]
        
        if len(filtered_df) == 0:
            st.info(f"No listings with status '{status_filter}'")
        else:
            # Display with approve/reject buttons
            for i, row in filtered_df.iterrows():
                with st.container():
                    col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
                    with col1:
                        st.write(f"**{row['Company']} ({row['Symbol']})**")
                        st.write(f"Submitted: {row['Date Submitted']} | Initial Price: ${row['Initial Price']} | Risk: {row['Risk Percentage']}%")
                    
                    # Only show approval buttons for pending stocks
                    if row['Status'] == 'Pending':
                        with col2:
                            if st.button("Approve", key=f"approve_{i}"):
                                # Update the status to approved
                                st.session_state.pending_stock_listings.at[i, 'Status'] = 'Approved'
                                
                                # Add to approved listings
                                new_approved = pd.DataFrame([{
                                    'Company': row['Company'],
                                    'Symbol': row['Symbol'],
                                    'Date Approved': datetime.now().strftime('%Y-%m-%d'),
                                    'Initial Price': row['Initial Price'],
                                    'Risk Percentage': row['Risk Percentage'],
                                    'Status': 'Active',
                                    'Approved By': st.session_state.get('username', 'admin')
                                }])
                                
                                st.session_state.approved_stock_listings = pd.concat([
                                    st.session_state.approved_stock_listings, new_approved
                                ], ignore_index=True)
                                
                                st.success(f"Approved {row['Symbol']}")
                                time.sleep(1)  # Brief pause to show the success message
                                st.experimental_rerun()  # Use experimental_rerun() instead of st.rerun()
                                
                        with col3:
                            if st.button("Reject", key=f"reject_{i}"):
                                # Update the status to rejected
                                st.session_state.pending_stock_listings.at[i, 'Status'] = 'Rejected'
                                st.error(f"Rejected {row['Symbol']}")
                                time.sleep(1)  # Brief pause to show the error message
                                st.experimental_rerun()  # Use experimental_rerun() instead of st.rerun()
                    else:
                        with col2:
                            status_color = "green" if row['Status'] == 'Approved' else "red"
                            st.markdown(f"<span style='color:{status_color};font-weight:bold'>{row['Status']}</span>", unsafe_allow_html=True)
                    
                    with col4:
                        if st.button("Details", key=f"details_{i}"):
                            st.session_state.selected_listing = row
                            st.session_state.show_listing_details = True
                    
                    st.markdown("---")
    
    # Show listing details if requested
    if st.session_state.get('show_listing_details', False):
        with st.expander("Listing Details", expanded=True):
            row = st.session_state.selected_listing
            st.write(f"### {row['Company']} ({row['Symbol']})")
            st.write(f"**Submission Date:** {row['Date Submitted']}")
            st.write(f"**Initial Price:** ${row['Initial Price']}")
            st.write(f"**Risk Percentage:** {row['Risk Percentage']}%")
            st.write(f"**Status:** {row['Status']}")
            st.write(f"**Submitted By:** {row['Submitted By']}")
            
            # Additional fields if they exist in the listing
            if 'Sector' in row:
                st.write(f"**Sector:** {row['Sector']}")
            if 'Description' in row:
                st.write(f"**Company Description:**")
                st.write(row['Description'])
            
            if st.button("Close Details"):
                st.session_state.show_listing_details = False
                st.experimental_rerun()  # Use experimental_rerun() instead of st.rerun()
    
    # Approved stocks section
    st.subheader("Approved Stock Listings")
    if len(st.session_state.approved_stock_listings) == 0:
        st.info("No approved stock listings yet.")
    else:
        with st.expander("View Approved Listings", expanded=True):
            # Add search and filter functionality
            search_term = st.text_input("Search by company name or symbol", "")
            
            filtered_approved = st.session_state.approved_stock_listings
            if search_term:
                filtered_approved = filtered_approved[
                    filtered_approved['Company'].str.contains(search_term, case=False) | 
                    filtered_approved['Symbol'].str.contains(search_term, case=False)
                ]
            
            if len(filtered_approved) == 0:
                st.info(f"No approved listings matching '{search_term}'")
            else:
                # Display each approved listing with actions
                for i, row in filtered_approved.iterrows():
                    with st.container():
                        col1, col2 = st.columns([4, 1])
                        with col1:
                            st.write(f"**{row['Company']} ({row['Symbol']})**")
                            st.write(f"Approved: {row['Date Approved']} | Initial Price: ${row['Initial Price']} | Risk: {row['Risk Percentage']}%")
                            st.write(f"Status: {row['Status']} | Approved by: {row['Approved By']}")
                        
                        with col2:
                            if st.button("Suspend", key=f"suspend_{i}"):
                                if row['Status'] == 'Active':
                                    st.session_state.approved_stock_listings.at[i, 'Status'] = 'Suspended'
                                    st.warning(f"Suspended {row['Symbol']}")
                                else:
                                    st.session_state.approved_stock_listings.at[i, 'Status'] = 'Active'
                                    st.success(f"Reactivated {row['Symbol']}")
                                time.sleep(1)
                                st.experimental_rerun()
                        
                        st.markdown("---")
    
    # User management with improved functionality
    st.subheader("User Management")
    with st.expander("Manage Users", expanded=False):
        # Initialize sample users if not already in session state
        if 'users_df' not in st.session_state:
            st.session_state.users_df = pd.DataFrame({
                'User': ['John Doe', 'Jane Smith', 'Bob Johnson', 'Alice Brown', 'Charlie Davis'],
                'Username': ['john_doe', 'jane_smith', 'bob_j', 'alice_b', 'charlie_d'],
                'Role': ['Investor', 'Company', 'Investor', 'Admin', 'Company'],
                'Status': ['Active', 'Active', 'Suspended', 'Active', 'Active'],
                'Joined': ['2025-01-15', '2025-01-20', '2025-02-05', '2025-02-10', '2025-02-15']
            })
        
        # User search
        user_search = st.text_input("Search users", "")
        
        filtered_users = st.session_state.users_df
        if user_search:
            filtered_users = filtered_users[
                filtered_users['User'].str.contains(user_search, case=False) | 
                filtered_users['Username'].str.contains(user_search, case=False)
            ]
        
        if len(filtered_users) == 0:
            st.info(f"No users matching '{user_search}'")
        else:
            # Display users with actions
            for i, row in filtered_users.iterrows():
                with st.container():
                    col1, col2, col3 = st.columns([3, 1, 1])
                    with col1:
                        st.write(f"**{row['User']} (@{row['Username']})**")
                        status_color = "green" if row['Status'] == 'Active' else "red"
                        st.write(f"Role: {row['Role']} | Status: ")
                        st.markdown(f"<span style='color:{status_color};font-weight:bold'>{row['Status']}</span> | Joined: {row['Joined']}", unsafe_allow_html=True)
                    
                    with col2:
                        if st.button("Toggle Status", key=f"toggle_{i}"):
                            new_status = 'Suspended' if row['Status'] == 'Active' else 'Active'
                            st.session_state.users_df.at[i, 'Status'] = new_status
                            st.warning(f"Changed {row['Username']} status to {new_status}")
                            time.sleep(1)
                            st.experimental_rerun()
                    
                    with col3:
                        roles = ['Investor', 'Company', 'Admin']
                        current_role_index = roles.index(row['Role'])
                        new_role = roles[(current_role_index + 1) % len(roles)]
                        
                        if st.button(f"Make {new_role}", key=f"role_{i}"):
                            st.session_state.users_df.at[i, 'Role'] = new_role
                            st.info(f"Changed {row['Username']} role to {new_role}")
                            time.sleep(1)
                            st.experimental_rerun()
                    
                    st.markdown("---")
        
        # Add new user button
        if st.button("Add New User"):
            st.session_state.show_add_user = True
    
    # Show add user form if requested
    if st.session_state.get('show_add_user', False):
        with st.form("add_user_form"):
            st.subheader("Add New User")
            new_name = st.text_input("Full Name")
            new_username = st.text_input("Username")
            new_role = st.selectbox("Role", ['Investor', 'Company', 'Admin'])
            new_password = st.text_input("Password", type="password")
            
            submit = st.form_submit_button("Add User")
            
            if submit:
                if not new_name or not new_username or not new_password:
                    st.error("All fields are required")
                elif new_username in st.session_state.users_df['Username'].values:
                    st.error("Username already exists")
                else:
                    # Add new user
                    new_user = pd.DataFrame([{
                        'User': new_name,
                        'Username': new_username,
                        'Role': new_role,
                        'Status': 'Active',
                        'Joined': datetime.now().strftime('%Y-%m-%d')
                    }])
                    
                    st.session_state.users_df = pd.concat([
                        st.session_state.users_df, new_user
                    ], ignore_index=True)
                    
                    st.session_state.show_add_user = False
                    st.success(f"Added new user: {new_name}")
                    time.sleep(1)
                    st.experimental_rerun()
        
        if st.button("Cancel"):
            st.session_state.show_add_user = False
            st.experimental_rerun()
    
    # Market overview with sentiment analysis
    st.subheader("Market Overview")
    
    # Add market-wide sentiment analysis
    st.subheader("Real-time Market Sentiment Analysis")
    display_market_sentiment()
    
    # Display regular market overview
    display_market_overview()
    
    # Add real-time market activity monitoring
    st.subheader("Real-time Market Activity")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Trading Volume Heat Map")
        # Simulate real-time trading volume heat map
        volume_data = {
            'Technology': fetch_real_time_volume('XLK'),
            'Healthcare': fetch_real_time_volume('XLV'),
            'Financials': fetch_real_time_volume('XLF'),
            'Energy': fetch_real_time_volume('XLE'),
            'Consumer': fetch_real_time_volume('XLY')
        }
        
        volume_df = pd.DataFrame([
            {'Sector': sector, 'Volume': volume} 
            for sector, volume in volume_data.items()
        ])
        
        fig = px.treemap(
            volume_df, 
            path=['Sector'], 
            values='Volume',
            color='Volume',
            title='Current Trading Volume by Sector',
            color_continuous_scale='Blues'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### Real-time Market Alerts")
        alerts = fetch_market_alerts()
        for i, alert in enumerate(alerts):
            with st.container():
                st.markdown(f"**{alert['time']}** - {alert['message']}")
                st.markdown("---")
                
    # Add system logs and audit trail
    st.subheader("System Logs & Audit Trail")
    with st.expander("View System Logs", expanded=False):
        # Simulate system logs
        system_logs = [
            {"timestamp": "2025-04-10 09:15:32", "user": "admin1", "action": "Approved stock listing XYZ"},
            {"timestamp": "2025-04-10 09:02:45", "user": "admin1", "action": "Suspended user bob_j"},
            {"timestamp": "2025-04-09 16:47:12", "user": "admin2", "action": "Rejected stock listing ABC"},
            {"timestamp": "2025-04-09 14:30:05", "user": "system", "action": "Daily market data backup completed"},
            {"timestamp": "2025-04-09 10:12:38", "user": "admin1", "action": "Added new user alice_b with Admin role"}
        ]
        
        # Add filter options
        log_filter = st.selectbox("Filter by action type", ["All", "Stock listings", "User management", "System"])
        
        filtered_logs = system_logs
        if log_filter == "Stock listings":
            filtered_logs = [log for log in system_logs if "stock listing" in log["action"]]
        elif log_filter == "User management":
            filtered_logs = [log for log in system_logs if "user" in log["action"] and "system" != log["user"]]
        elif log_filter == "System":
            filtered_logs = [log for log in system_logs if "system" == log["user"]]
        
        # Display logs
        for log in filtered_logs:
            st.markdown(f"**{log['timestamp']}** - {log['user']}: {log['action']}")
            st.markdown("---")

def company_dashboard():
    st.header("Company Dashboard")
    
    # Add tabs for better organization
    tab1, tab2, tab3 = st.tabs(["Stock Performance", "Stock Listings", "Market Overview"])
    
    with tab1:
        # Company stock performance with enhanced visualization
        st.subheader("Your Stock Performance")
        symbol = st.selectbox("Select your company stock", ["AAPL", "MSFT", "GOOGL"])
        
        # Add period selector
        time_period = st.radio("Select Time Period", ["1M", "3M", "6M", "1Y", "5Y"], horizontal=True)
        period_map = {"1M": "1mo", "3M": "3mo", "6M": "6mo", "1Y": "1y", "5Y": "5y"}
        
        # Add real-time ticker at the top
        display_live_ticker(symbol)
        
        display_stock_chart(symbol, period_map[time_period])
        
        # Add real-time order book
        st.subheader("Real-time Order Book")
        display_order_book(symbol)
        
        # Add company-specific news and sentiment
        st.subheader("Company News & Sentiment")
        display_company_news(symbol)
    
    with tab2:
        # Show status of company listings
        st.subheader("Your Stock Listings")
        
        # Filter pending listings for this company (in a real app, would filter by actual company ID)
        company_name = st.session_state.get('username', 'Current Company')
        
        # Check for existing listings
        company_listings = st.session_state.pending_stock_listings[
            st.session_state.pending_stock_listings['Submitted By'] == company_name
        ]
        
        if len(company_listings) > 0:
            st.write("Your submitted listings:")
            for i, row in company_listings.iterrows():
                status_color = "blue" if row['Status'] == 'Pending' else (
                    "green" if row['Status'] == 'Approved' else "red"
                )
                st.markdown(f"""
                **{row['Company']} ({row['Symbol']})** - 
                <span style='color:{status_color};font-weight:bold'>{row['Status']}</span>
                """, unsafe_allow_html=True)
                st.write(f"Submitted: {row['Date Submitted']} | Initial Price: ${row['Initial Price']} | Risk: {row['Risk Percentage']}%")
                st.markdown("---")
        else:
            st.info("You haven't submitted any stock listings yet.")
        
        # Add new stock listing form with improved validation
        st.subheader("Submit New Stock Listing")
        with st.form("new_stock_form"):
            stock_name = st.text_input("Company Name")
            stock_symbol = st.text_input("Stock Symbol (3-5 letters)")
            initial_price = st.number_input("Initial Price ($)", min_value=0.01, value=25.00)
            risk_percentage = st.slider("Risk Percentage", 0, 100, 50)
            
            # Additional information
            st.write("Additional Information")
            sector = st.selectbox("Sector", [
                "Technology", "Healthcare", "Financial Services", "Consumer Goods", 
                "Energy", "Utilities", "Real Estate", "Communication Services", "Other"
            ])
            company_description = st.text_area("Company Description")
            
            submitted = st.form_submit_button("Submit for Approval")
            
            if submitted:
                # Validate inputs
                if not stock_name:
                    st.error("Company Name is required")
                elif not stock_symbol or len(stock_symbol) < 3 or len(stock_symbol) > 5:
                    st.error("Stock Symbol must be 3-5 letters")
                elif stock_symbol in st.session_state.pending_stock_listings['Symbol'].values:
                    st.error("Symbol already exists. Please choose a different symbol.")
                else:
                    # Create new listing entry
                    new_listing = pd.DataFrame([{
                        'Company': stock_name,
                        'Symbol': stock_symbol.upper(),
                        'Date Submitted': datetime.now().strftime('%Y-%m-%d'),
                        'Initial Price': initial_price,
                        'Risk Percentage': risk_percentage,
                        'Sector': sector,
                        'Description': company_description,
                        'Status': 'Pending',
                        'Submitted By': st.session_state.get('username', company_name)
                    }])
                    
                    # Add to pending listings
                    st.session_state.pending_stock_listings = pd.concat([
                        st.session_state.pending_stock_listings, new_listing
                    ], ignore_index=True)
                    
                    st.success(f"Stock {stock_name} ({stock_symbol.upper()}) submitted for approval")
                    st.balloons()
    
    with tab3:
        # Market overview
        st.subheader("Market Overview")
        display_market_overview()
        
        # Display market news that could affect listings
        st.subheader("Market News")
        display_market_news()

def investor_dashboard():
    st.header("Investor Dashboard")
    
    # Add real-time market overview at the top
    col1, col2, col3 = st.columns(3)
    indices = {'^GSPC': 'S&P 500', '^DJI': 'Dow Jones', '^IXIC': 'NASDAQ'}
    
    for (i, (symbol, name)) in enumerate(indices.items()):
        with [col1, col2, col3][i]:
            current_price, change_pct = fetch_real_time_price(symbol)
            st.metric(
                name, 
                f"${current_price:.2f}", 
                f"{change_pct:.2f}%",
                delta_color="normal" if change_pct >= 0 else "inverse"
            )
    
    # Portfolio summary with enhanced visualization and real-time data
    st.subheader("Portfolio Summary")
    portfolio_df = pd.DataFrame({
        'Stock': ['AAPL', 'MSFT', 'GOOGL'],
        'Shares': [10, 5, 2]
    })
    
    # Get real-time prices for portfolio stocks
    for stock in portfolio_df['Stock']:
        price, change = fetch_real_time_price(stock)
        portfolio_df.loc[portfolio_df['Stock'] == stock, 'Current Price'] = price
        portfolio_df.loc[portfolio_df['Stock'] == stock, 'Gain/Loss %'] = change
    
    # Calculate values
    portfolio_df['Value'] = portfolio_df['Shares'] * portfolio_df['Current Price']
    
    # Add risk score (simulated)
    portfolio_df['Risk Score'] = [35, 42, 58]
    
    # Calculate total portfolio value
    total_value = portfolio_df['Value'].sum()
    
    # Display portfolio metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        # Calculate real-time total portfolio change
        total_change = portfolio_df['Value'].dot(portfolio_df['Gain/Loss %']) / total_value
        st.metric("Total Portfolio Value", f"${total_value:.2f}", f"{total_change:.2f}%")
    with col2:
        avg_gain = portfolio_df['Value'].dot(portfolio_df['Gain/Loss %']) / total_value
        st.metric("Average Return", f"{avg_gain:.1f}%", "")
    with col3:
        weighted_risk = portfolio_df['Value'].dot(portfolio_df['Risk Score']) / total_value
        risk_color = "red" if weighted_risk > 50 else "orange" if weighted_risk > 30 else "green"
        st.markdown(f"**Portfolio Risk**: <span style='color:{risk_color}'>{weighted_risk:.1f}/100</span>", unsafe_allow_html=True)
    
    # Display portfolio as dataframe
    st.dataframe(portfolio_df)
    
    # Portfolio composition pie chart
    st.subheader("Portfolio Composition")
    fig = px.pie(portfolio_df, values='Value', names='Stock', title='Portfolio Allocation')
    st.plotly_chart(fig, use_container_width=True)
    
    # Show available stocks (including newly approved ones)
    st.subheader("Available Stocks")
    
    # Combine built-in stocks with approved listings
    available_stocks = pd.DataFrame({
        'Symbol': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META'],
        'Company': ['Apple Inc', 'Microsoft Corp', 'Alphabet Inc', 'Amazon.com Inc', 'Meta Platforms Inc'],
        'Current Price': [0, 0, 0, 0, 0],
        'Change %': [0, 0, 0, 0, 0]
    })
    
    # Add approved stocks from the admin listings
    if len(st.session_state.approved_stock_listings) > 0:
        for _, row in st.session_state.approved_stock_listings.iterrows():
            if row['Status'] == 'Active' and row['Symbol'] not in available_stocks['Symbol'].values:
                new_row = pd.DataFrame([{
                    'Symbol': row['Symbol'],
                    'Company': row['Company'],
                    'Current Price': 0,
                    'Change %': 0
                }])
                available_stocks = pd.concat([available_stocks, new_row], ignore_index=True)
    
    # Get real-time prices
    for i, row in available_stocks.iterrows():
        price, change = fetch_real_time_price(row['Symbol'])
        available_stocks.at[i, 'Current Price'] = price
        available_stocks.at[i, 'Change %'] = change
    
    # Display with buy button
    for i, row in available_stocks.iterrows():
        col1, col2, col3 = st.columns([3, 1, 1])
        with col1:
            change_color = "green" if row['Change %'] >= 0 else "red"
            st.markdown(f"""
            **{row['Company']} ({row['Symbol']})** - 
            ${row['Current Price']:.2f} 
            <span style='color:{change_color}'>({row['Change %']:.2f}%)</span>
            """, unsafe_allow_html=True)
        with col2:
            shares = st.number_input("Shares", min_value=1, value=1, key=f"shares_{i}")
        with col3:
            if st.button("Buy", key=f"buy_{i}"):
                st.success(f"Purchased {shares} shares of {row['Symbol']} at ${row['Current Price']}")
        st.markdown("---")
    
    # Real-time watchlist
    st.subheader("Real-time Watchlist")
    watchlist = ['NVDA', 'AMZN', 'META', 'TSLA', 'NFLX']
    
    # Create DataFrame for watchlist
    watchlist_data = []
    for symbol in watchlist:
        price, change = fetch_real_time_price(symbol)
        volume = fetch_real_time_volume(symbol)
        watchlist_data.append({
            'Symbol': symbol,
            'Price': price,
            'Change %': change,
            'Volume': volume
        })
    
    watchlist_df = pd.DataFrame(watchlist_data)
    
    # Display watchlist with real-time updates
    st.dataframe(
        watchlist_df.style.apply(
            lambda x: ['color: red' if v < 0 else 'color: green' for v in x], 
            subset=['Change %']
        ),
        use_container_width=True
    )
    
    # Trade signals with enhanced visualization
    st.subheader("AI Trade Signals")
    signals_df = pd.DataFrame({
        'Stock': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META'],
        'Signal': ['Buy', 'Hold', 'Sell', 'Buy', 'Hold'],
        'Confidence': [0.85, 0.65, 0.78, 0.72, 0.55],
        'Expected Return': ['+4.2%', '+1.7%', '-2.8%', '+3.5%', '+0.9%'],
        'Risk Level': ['Medium', 'Low', 'High', 'Medium', 'Low']
    })
    
    # Update signals based on real-time data
    for i, row in signals_df.iterrows():
        current_price, change = fetch_real_time_price(row['Stock'])
        # Adjust signal based on real-time price movement (simplified algorithm)
        if change > 2.0 and row['Signal'] != 'Sell':
            signals_df.at[i, 'Signal'] = 'Sell'
            signals_df.at[i, 'Confidence'] = min(0.9, signals_df.at[i, 'Confidence'] + 0.1)
        elif change < -2.0 and row['Signal'] != 'Buy':
            signals_df.at[i, 'Signal'] = 'Buy'
            signals_df.at[i, 'Confidence'] = min(0.9, signals_df.at[i, 'Confidence'] + 0.1)
    
    # Color-coded signals with confidence visualization
    for i, row in signals_df.iterrows():
        signal_color = 'green' if row['Signal'] == 'Buy' else ('red' if row['Signal'] == 'Sell' else 'orange')
        confidence_label = 'High' if row['Confidence'] > 0.75 else ('Medium' if row['Confidence'] > 0.6 else 'Low')
        
        col1, col2, col3 = st.columns([3, 4, 3])
        with col1:
            st.markdown(f"**{row['Stock']}**: <span style='color:{signal_color}'>{row['Signal']}</span>", unsafe_allow_html=True)
        with col2:
            # Create a confidence bar
            conf_val = int(row['Confidence'] * 100)
            st.progress(row['Confidence'])
            st.text(f"Confidence: {conf_val}% ({confidence_label})")
        with col3:
            st.text(f"Expected: {row['Expected Return']}")
            st.text(f"Risk: {row['Risk Level']}")
        st.markdown("---")
    
    # News feed and sentiment analysis
    st.subheader("Latest Market News")
    display_market_news()
    
    # Market overview
    st.subheader("Market Overview")
    display_market_overview()

def display_live_ticker(symbol):
    """Display a real-time ticker for the selected stock"""
    try:
        # Get real-time price
        current_price, change_pct = fetch_real_time_price(symbol)
        
        # Format the ticker display
        ticker_color = "green" if change_pct >= 0 else "red"
        change_sign = "+" if change_pct >= 0 else ""
        
        # Create ticker container
        st.markdown(
            f"""
            <div style="background-color: #f0f2f6; padding: 10px; border-radius: 5px; margin-bottom: 20px;">
                <span style="font-size: 24px; font-weight: bold;">{symbol}</span>
                <span style="font-size: 24px; margin-left: 15px;">${current_price:.2f}</span>
                <span style="font-size: 18px; margin-left: 10px; color: {ticker_color};">
                    {change_sign}{change_pct:.2f}%
                </span>
                <span style="float: right; font-size: 14px;">Last updated: {datetime.now().strftime('%H:%M:%S')}</span>
            </div>
            """,
            unsafe_allow_html=True
        )
    except Exception as e:
        st.error(f"Error fetching real-time data for {symbol}: {e}")

def display_order_book(symbol):
    """Display a simulated real-time order book"""
    try:
        # Create columns for buy and sell orders
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Buy Orders")
            # Simulate buy orders (would be real API data in production)
            current_price, _ = fetch_real_time_price(symbol)
            
            buy_orders = [
                {"price": round(current_price * (1 - (i * 0.001)), 2), 
                 "quantity": int(np.random.normal(500, 200)),
                 "total": 0}
                for i in range(1, 6)
            ]
            
            # Calculate totals
            for order in buy_orders:
                order["total"] = round(order["price"] * order["quantity"], 2)
            
            # Create DataFrame
            buy_df = pd.DataFrame(buy_orders)
            st.dataframe(buy_df, use_container_width=True)
        
        with col2:
            st.markdown("### Sell Orders")
            # Simulate sell orders
            sell_orders = [
                {"price": round(current_price * (1 + (i * 0.001)), 2), 
                 "quantity": int(np.random.normal(500, 200)),
                 "total": 0}
                for i in range(1, 6)
            ]
            
            # Calculate totals
            for order in sell_orders:
                order["total"] = round(order["price"] * order["quantity"], 2)
            
            # Create DataFrame
            sell_df = pd.DataFrame(sell_orders)
            st.dataframe(sell_df, use_container_width=True)
        
        # Display combined order book as depth chart
        st.markdown("### Order Book Depth")
        
        # Combine orders for visualization
        buy_depth = [{"price": order["price"], "volume": order["quantity"], "type": "buy"} for order in buy_orders]
        sell_depth = [{"price": order["price"], "volume": order["quantity"], "type": "sell"} for order in sell_orders]
        
        combined_depth = pd.DataFrame(buy_depth + sell_depth)
        
        # Create depth chart
        fig = px.bar(
            combined_depth, 
            x="price", 
            y="volume", 
            color="type",
            color_discrete_map={"buy": "green", "sell": "red"},
            title=f"{symbol} Order Book Depth"
        )
        
        fig.update_layout(
            xaxis_title="Price ($)",
            yaxis_title="Volume (shares)"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error generating order book for {symbol}: {e}")

def display_market_overview():
    # Get some major indices
    indices = {
        'S&P 500': '^GSPC',
        'Dow Jones': '^DJI',
        'NASDAQ': '^IXIC'
    }
    
    # Create tabs for different market views
    tab1, tab2 = st.tabs(["Market Summary", "Historical Trends"])
    
    with tab1:
        # Display current values and daily change
        cols = st.columns(len(indices))
        index_data = {}
        
        for i, (name, symbol) in enumerate(indices.items()):
            try:
                # Use real-time data function
                current, change = fetch_real_time_price(symbol)
                index_data[name] = {'current': current, 'change': change}
                
                with cols[i]:
                    st.metric(name, f"{current:.2f}", f"{change:.2f}%")
            except Exception as e:
                with cols[i]:
                    st.metric(name, "Error", "0.00%")
                    st.error(f"Failed to load data: {str(e)}")
        
        # Display sector performance
        st.subheader("Sector Performance")
        sectors = {
            'Technology': 'XLK',
            'Healthcare': 'XLV',
            'Financials': 'XLF',
            'Energy': 'XLE',
            'Consumer Discretionary': 'XLY',
            'Consumer Staples': 'XLP',
            'Utilities': 'XLU',
            'Real Estate': 'XLRE',
            'Materials': 'XLB',
            'Industrials': 'XLI',
            'Communication Services': 'XLC'
        }
        
        # Get performance data for sectors
        sector_data = []
        for name, symbol in sectors.items():
            try:
                current, change = fetch_real_time_price(symbol)
                sector_data.append({
                    'Sector': name,
                    'Change': change
                })
            except:
                sector_data.append({
                    'Sector': name,
                    'Change': 0
                })
        
        sector_df = pd.DataFrame(sector_data)
        sector_df = sector_df.sort_values('Change', ascending=False)
        
        # Create horizontal bar chart for sector performance
        fig = px.bar(
            sector_df,
            y='Sector',
            x='Change',
            title='Daily Sector Performance (%)',
            color='Change',
            color_continuous_scale=['red', 'gray', 'green'],
            color_continuous_midpoint=0,
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        # Historical trends
        st.subheader("Historical Index Performance")
        
        # Select time period
        period = st.radio("Select time period", 
                          ["1M", "3M", "6M", "1Y", "5Y"], 
                          horizontal=True)
        
        period_map = {"1M": "1mo", "3M": "3mo", "6M": "6mo", "1Y": "1y", "5Y": "5y"}
        
        # Download historical data
        try:
            fig = go.Figure()
            
            for name, symbol in indices.items():
                df = yf.download(symbol, period=period_map[period])
                
                # Calculate percentage change from start
                df['Normalized'] = df['Close'] / df['Close'].iloc[0] * 100 - 100
                
                fig.add_trace(go.Scatter(
                    x=df.index,
                    y=df['Normalized'],
                    name=name,
                    mode='lines'
                ))
            
            fig.update_layout(
                title=f"Index Performance Over {period} (% Change)",
                xaxis_title="Date",
                yaxis_title="% Change",
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Failed to load historical data: {str(e)}")

def display_market_sentiment():
    """Display real-time market sentiment analysis"""
    # Create simulated sentiment data
    sentiment_data = {
        'Overall Market': np.random.uniform(-0.3, 0.3),
        'Technology': np.random.uniform(-0.5, 0.5),
        'Healthcare': np.random.uniform(-0.4, 0.4),
        'Financials': np.random.uniform(-0.3, 0.3),
        'Energy': np.random.uniform(-0.2, 0.5),
        'Consumer': np.random.uniform(-0.4, 0.2)
    }
    
    # Convert to dataframe
    sentiment_df = pd.DataFrame([
        {'Category': cat, 'Sentiment': val} 
        for cat, val in sentiment_data.items()
    ])
    
    # Create sentiment visualization
    fig = px.bar(
        sentiment_df,
        y='Category',
        x='Sentiment',
        color='Sentiment',
        color_continuous_scale=['red', 'gray', 'green'],
        color_continuous_midpoint=0,
        title='Real-time Market Sentiment',
        height=300
    )
    
    fig.update_layout(
        xaxis_title="Sentiment Score (Negative to Positive)",
        yaxis_title=""
    )
    
    # Add zero line
    fig.add_vline(x=0, line_dash="dash", line_color="gray")
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Add sentiment explanation
    with st.expander("Sentiment Analysis Explanation"):
        st.write("""
        Our real-time sentiment analysis aggregates data from news sources, social media, 
        and analyst reports to measure market sentiment. Scores range from -1 (extremely negative) 
        to +1 (extremely positive), with 0 representing neutral sentiment.
        """)

def display_stock_chart(symbol, period='1mo'):
    """Display interactive stock chart with technical indicators"""
    try:
        # Get stock data
        df = yf.download(symbol, period=period)
        
        if df.empty:
            st.error(f"No data available for {symbol}")
            return
        
        # Create subplots: price and volume
        fig = make_subplots(
            rows=2, 
            cols=1, 
            shared_xaxes=True,
            row_heights=[0.7, 0.3],
            vertical_spacing=0.1,
            subplot_titles=(f"{symbol} Price", "Volume")
        )
        
        # Add candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'],
                name='Price'
            ),
            row=1, col=1
        )
        
        # Add volume bar chart
        fig.add_trace(
            go.Bar(
                x=df.index,
                y=df['Volume'],
                name='Volume',
                marker_color='rgba(0, 0, 255, 0.5)'
            ),
            row=2, col=1
        )
        
        # Add moving averages
        ma_periods = [20, 50]
        for period in ma_periods:
            df[f'MA_{period}'] = df['Close'].rolling(window=period).mean()
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df[f'MA_{period}'],
                    name=f'{period}-day MA',
                    line=dict(width=1)
                ),
                row=1, col=1
            )
        
        # Update layout
        fig.update_layout(
            title=f"{symbol} Stock Price Chart",
            xaxis_title="Date",
            yaxis_title="Price ($)",
            xaxis_rangeslider_visible=False,
            height=600
        )
        
        # Display chart
        st.plotly_chart(fig, use_container_width=True)
        
        # Show additional metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            # Calculate daily return
            daily_return = (df['Close'].iloc[-1] / df['Close'].iloc[-2] - 1) * 100
            st.metric("Daily Return", f"{daily_return:.2f}%")
        
        with col2:
            # Calculate volatility (standard deviation of returns)
            returns = df['Close'].pct_change().dropna()
            volatility = returns.std() * 100 * np.sqrt(252)  # Annualized
            st.metric("Volatility", f"{volatility:.2f}%")
        
        with col3:
            # Calculate trading volume change
            volume_change = (df['Volume'].iloc[-1] / df['Volume'].iloc[-2] - 1) * 100
            st.metric("Volume Change", f"{volume_change:.2f}%")
        
    except Exception as e:
        st.error(f"Error displaying chart for {symbol}: {str(e)}")

def display_company_news(symbol):
    """Display company news with sentiment analysis"""
    # In a real app, you would fetch actual news from an API
    # Simulate news data for demonstration
    news_items = [
        {
            "title": f"{symbol} Announces Better-Than-Expected Quarterly Earnings",
            "source": "Financial Times",
            "time": "2 hours ago",
            "summary": f"{symbol} reported quarterly earnings that exceeded analyst expectations by 15%, driven by strong product demand and improved profit margins.",
            "sentiment": 0.75
        },
        {
            "title": f"Analysts Upgrade {symbol} to 'Buy' Rating",
            "source": "Wall Street Journal",
            "time": "5 hours ago",
            "summary": f"Multiple research firms have upgraded {symbol}'s stock rating following the company's announcement of expansion into new markets.",
            "sentiment": 0.65
        },
        {
            "title": f"{symbol} Faces Supply Chain Challenges Amid Global Shortages",
            "source": "Bloomberg",
            "time": "1 day ago",
            "summary": f"{symbol} executives acknowledged ongoing supply chain difficulties but reassured investors they have mitigation strategies in place.",
            "sentiment": -0.25
        }
    ]
    
    # Display news items with sentiment analysis
    for item in news_items:
        # Determine sentiment color
        sentiment_color = "green" if item["sentiment"] > 0.3 else ("red" if item["sentiment"] < -0.3 else "gray")
        sentiment_label = "Positive" if item["sentiment"] > 0.3 else ("Negative" if item["sentiment"] < -0.3 else "Neutral")
        
        # Create news card
        st.markdown(f"""
        ### {item["title"]}
        **Source:** {item["source"]} | **Time:** {item["time"]}
        
        {item["summary"]}
        
        **Sentiment:** <span style='color:{sentiment_color}'>{sentiment_label} ({item["sentiment"]:.2f})</span>
        """, unsafe_allow_html=True)
        
        # Add separator
        st.markdown("---")

def display_market_news():
    """Display general market news"""
    # Simulate market news data
    news_items = [
        {
            "title": "Federal Reserve Signals Potential Rate Cut",
            "source": "Reuters",
            "time": "1 hour ago",
            "summary": "Federal Reserve minutes suggest policymakers are considering an interest rate cut in the next meeting, citing inflation control and economic stability.",
            "sentiment": 0.55
        },
        {
            "title": "Global Supply Chain Issues Continue to Impact Markets",
            "source": "CNBC",
            "time": "3 hours ago",
            "summary": "Ongoing supply chain disruptions are affecting multiple sectors, with semiconductor and automotive industries experiencing significant impacts.",
            "sentiment": -0.35
        },
        {
            "title": "Treasury Yields Rise Amid Economic Growth Projections",
            "source": "Bloomberg",
            "time": "6 hours ago",
            "summary": "Treasury yields climbed as new economic data suggested stronger-than-expected GDP growth for the current quarter.",
            "sentiment": 0.25
        }
    ]
    
    # Display news items with sentiment analysis
    for item in news_items:
        # Determine sentiment color
        sentiment_color = "green" if item["sentiment"] > 0.3 else ("red" if item["sentiment"] < -0.3 else "gray")
        sentiment_label = "Positive" if item["sentiment"] > 0.3 else ("Negative" if item["sentiment"] < -0.3 else "Neutral")
        
        # Create news card
        st.markdown(f"""
        ### {item["title"]}
        **Source:** {item["source"]} | **Time:** {item["time"]}
        
        {item["summary"]}
        
        **Sentiment:** <span style='color:{sentiment_color}'>{sentiment_label} ({item["sentiment"]:.2f})</span>
        """, unsafe_allow_html=True)
        
        # Add separator
        st.markdown("---")

def fetch_real_time_price(symbol):
    """Fetch real-time price data for a given symbol"""
    try:
        # In a real app, you would use a real-time data API
        # Here we're simulating real-time data with random fluctuations
        
        # Start with a base price
        base_prices = {
            '^GSPC': 5000.0,  # S&P 500
            '^DJI': 38000.0,  # Dow Jones
            '^IXIC': 16000.0,  # NASDAQ
            'AAPL': 180.0,
            'MSFT': 400.0,
            'GOOGL': 160.0,
            'AMZN': 180.0,
            'META': 500.0,
            'TSLA': 180.0,
            'NVDA': 800.0,
            'NFLX': 600.0,
            'XLK': 170.0,  # Technology sector ETF
            'XLV': 135.0,  # Healthcare sector ETF
            'XLF': 38.0,   # Financial sector ETF
            'XLE': 85.0,   # Energy sector ETF
            'XLY': 180.0,  # Consumer Discretionary ETF
            'XLP': 72.0,   # Consumer Staples ETF
            'XLU': 65.0,   # Utilities ETF
            'XLRE': 42.0,  # Real Estate ETF
            'XLB': 88.0,   # Materials ETF
            'XLI': 115.0,  # Industrials ETF
            'XLC': 75.0    # Communication Services ETF
        }
        
        # Use default price if symbol not in our base prices
        base_price = base_prices.get(symbol, 100.0)
        
        # Add random fluctuation (±2%)
        current_price = base_price * (1 + np.random.uniform(-0.02, 0.02))
        
        # Calculate daily change percentage
        change_pct = np.random.uniform(-3.0, 3.0)
        
        return current_price, change_pct
    except Exception as e:
        # Return default values in case of error
        return 0.0, 0.0

def fetch_real_time_volume(symbol):
    """Fetch real-time trading volume for a given symbol"""
    try:
        # In a real app, you would use a real-time data API
        # Here we're simulating volume data
        
        # Base volume varies by symbol type
        if symbol.startswith('^'):  # Index
            base_volume = np.random.normal(5000000, 1000000)
        elif symbol.startswith('X'):  # Sector ETF
            base_volume = np.random.normal(2000000, 500000)
        else:  # Individual stock
            base_volume = np.random.normal(10000000, 3000000)
        
        return int(max(100000, base_volume))
    except:
        return 0

def fetch_market_alerts():
    """Fetch real-time market alerts"""
    # Simulate market alerts
    current_time = datetime.now().strftime("%H:%M")
    
    alerts = [
        {
            "time": current_time,
            "message": "High trading volume detected in technology sector, 34% above 30-day average."
        },
        {
            "time": (datetime.now() - timedelta(minutes=15)).strftime("%H:%M"),
            "message": "BREAKING: Federal Reserve announces unchanged interest rates, markets responding positively."
        },
        {
            "time": (datetime.now() - timedelta(minutes=45)).strftime("%H:%M"),
            "message": "Unusual options activity detected in multiple financial stocks ahead of earnings reports."
        },
        {
            "time": (datetime.now() - timedelta(hours=2)).strftime("%H:%M"),
            "message": "Energy sector experiencing volatility following release of new global supply forecasts."
        }
    ]
    
    return alerts

# Login functionality
def show_login():
    st.title("Stock Market Dashboard - Login")
    
    # Create login form
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        
        # Add user type selection for demonstration
        user_type = st.selectbox("Login as", ["Investor", "Company", "Admin"])
        
        submit = st.form_submit_button("Login")
        
        if submit:
            if not username or not password:
                st.error("Please enter both username and password")
            else:
                # In a real app, you would validate credentials against a database
                # For demonstration, we'll accept any non-empty credentials
                st.session_state.logged_in = True
                st.session_state.username = username
                st.session_state.user_role = user_type.lower()
                st.success(f"Logged in as {username} ({user_type})")
                st.rerun()
    
    # Add demo accounts information
    with st.expander("Demo Accounts"):
        st.write("""
        For demonstration purposes, you can use any username and password, and select the user type to see different dashboard views:
        
        1. **Investor** - View portfolio, available stocks, watchlist, and market data
        2. **Company** - Manage stock listings and view market data
        3. **Admin** - Approve/reject stock listings and view comprehensive market data
        """)

# Main app entry point
def main():
    # Check if user is logged in
    if not st.session_state.get('logged_in', False):
        show_login()
    else:
        show()

if __name__ == "__main__":
    main()