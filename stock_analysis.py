import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import ta
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

def show():
    st.title("Stock Analysis")
    
    # Define stock categories
    stock_categories = {
        'Technology': ['AAPL', 'MSFT', 'GOOGL', 'META', 'NVDA'],
        'Healthcare': ['JNJ', 'PFE', 'UNH', 'ABBV', 'MRK'],
        'Finance': ['JPM', 'BAC', 'WFC', 'GS', 'MS'],
        'Consumer': ['AMZN', 'WMT', 'PG', 'KO', 'PEP'],
        'Energy': ['XOM', 'CVX', 'COP', 'SLB', 'EOG']
    }
    
    # Create tabs for selection methods
    tab1, tab2 = st.tabs(["Select from Categories", "Custom Ticker"])
    
    with tab1:
        # Category selection
        selected_category = st.selectbox("Select Sector", options=list(stock_categories.keys()))
        
        # Stock selection from category
        selected_stock = st.selectbox(
            f"Select {selected_category} Stock", 
            options=stock_categories[selected_category]
        )
        
        # Added unique key "category_period" to this select_slider
        period = st.select_slider(
            "Time Period", 
            options=["1m", "3m", "6m", "1y", "5y"], 
            value="1y",
            key="category_period"
        )
        
        if st.button("Analyze Selected Stock"):
            with st.spinner(f"Analyzing {selected_stock}..."):
                analyze_stock(selected_stock, period)
    
    with tab2:
        # Custom ticker input
        custom_symbol = st.text_input("Enter Custom Stock Symbol", "TSLA")
        
        # Added unique key "custom_period" to this select_slider
        custom_period = st.select_slider(
            "Time Period", 
            options=["1m", "3m", "6m", "1y", "5y"], 
            value="1y",
            key="custom_period"
        )
        
        if st.button("Analyze Custom Stock"):
            with st.spinner(f"Analyzing {custom_symbol}..."):
                analyze_stock(custom_symbol, custom_period)

def analyze_stock(symbol, period):
    try:
        # Fetch data
        data = yf.Ticker(symbol).history(period=period)
        
        if data.empty:
            st.error(f"No data found for {symbol}")
            return
        
        # Stock info
        info = yf.Ticker(symbol).info
        company_name = info.get('longName', symbol)
        
        # Display company info
        st.header(f"{company_name} ({symbol})")
        
        # Current price and stats
        current_price = data['Close'].iloc[-1]
        prev_close = data['Close'].iloc[-2]
        price_change = current_price - prev_close
        price_change_pct = (price_change / prev_close) * 100
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Current Price", f"${current_price:.2f}", f"{price_change_pct:.2f}%")
        with col2:
            st.metric("52 Week High", f"${data['High'].max():.2f}")
        with col3:
            st.metric("52 Week Low", f"${data['Low'].min():.2f}")
        
        # Train LSTM model
        future_prices = predict_future_prices(data)
        
        # Plot actual and predicted prices
        st.subheader("LSTM Price Prediction (Next 4 Days)")
        future_dates = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=4)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data.index, y=data['Close'], name='Historical Price', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=future_dates, y=future_prices, name='Predicted Price', line=dict(color='red', dash='dash')))
        
        fig.update_layout(title='LSTM Price Prediction (Next 4 Days)', xaxis_title='Date', yaxis_title='Price')
        st.plotly_chart(fig, use_container_width=True)
        
        # Add technical indicators visualization
        st.subheader("Technical Indicators")
        
        # Calculate technical indicators
        data['SMA_20'] = ta.trend.sma_indicator(data['Close'], window=20)
        data['SMA_50'] = ta.trend.sma_indicator(data['Close'], window=50)
        data['RSI'] = ta.momentum.rsi(data['Close'], window=14)
        data['MACD'] = ta.trend.macd(data['Close'])
        data['MACD_Signal'] = ta.trend.macd_signal(data['Close'])
        
        # Create tabs for different visualizations
        indicator_tab1, indicator_tab2 = st.tabs(["Moving Averages", "RSI & MACD"])
        
        with indicator_tab1:
            # Plot moving averages
            fig_ma = go.Figure()
            fig_ma.add_trace(go.Scatter(x=data.index, y=data['Close'], name='Close Price', line=dict(color='blue')))
            fig_ma.add_trace(go.Scatter(x=data.index, y=data['SMA_20'], name='SMA 20', line=dict(color='orange')))
            fig_ma.add_trace(go.Scatter(x=data.index, y=data['SMA_50'], name='SMA 50', line=dict(color='green')))
            fig_ma.update_layout(title='Moving Averages', xaxis_title='Date', yaxis_title='Price')
            st.plotly_chart(fig_ma, use_container_width=True)
        
        with indicator_tab2:
            # Create a subplot with RSI and MACD
            fig_ind = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                                vertical_spacing=0.1, 
                                subplot_titles=('RSI', 'MACD'))
            
            # RSI
            fig_ind.add_trace(go.Scatter(x=data.index, y=data['RSI'], name='RSI', line=dict(color='purple')), row=1, col=1)
            fig_ind.add_hline(y=70, line_dash="dash", line_color="red", row=1, col=1)
            fig_ind.add_hline(y=30, line_dash="dash", line_color="green", row=1, col=1)
            
            # MACD
            fig_ind.add_trace(go.Scatter(x=data.index, y=data['MACD'], name='MACD', line=dict(color='blue')), row=2, col=1)
            fig_ind.add_trace(go.Scatter(x=data.index, y=data['MACD_Signal'], name='Signal', line=dict(color='red')), row=2, col=1)
            
            fig_ind.update_layout(height=500)
            st.plotly_chart(fig_ind, use_container_width=True)
        
        # Generate Buy/Sell signal based on trends
        trend_signal = generate_trend_signal(data, future_prices)
        signal_color = "green" if "Buy" in trend_signal else ("red" if "Sell" in trend_signal else "orange")
        
        st.subheader("AI Trade Signal")
        st.markdown(f"<div style='padding:10px; background-color:{signal_color}; color:white; border-radius:5px; text-align:center;'><h3>{trend_signal}</h3></div>", unsafe_allow_html=True)
        
        # Add additional metrics
        st.subheader("Key Metrics")
        
        # Calculate additional metrics
        avg_volume = data['Volume'].mean()
        volume_change = (data['Volume'].iloc[-1] / data['Volume'].iloc[-5:].mean() - 1) * 100
        volatility = data['Close'].pct_change().std() * 100 * np.sqrt(252)  # Annualized volatility
        
        metric_col1, metric_col2, metric_col3 = st.columns(3)
        with metric_col1:
            st.metric("Avg Daily Volume", f"{int(avg_volume):,}", f"{volume_change:.2f}%")
        with metric_col2:
            st.metric("Volatility (Annual)", f"{volatility:.2f}%")
        with metric_col3:
            rsi_value = data['RSI'].iloc[-1]
            rsi_status = "Oversold" if rsi_value < 30 else ("Overbought" if rsi_value > 70 else "Neutral")
            st.metric("RSI", f"{rsi_value:.2f}", rsi_status)
        
    except Exception as e:
        st.error(f"Error analyzing {symbol}: {str(e)}")

def predict_future_prices(data):
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(data[['Close']].values)
    
    # Ensure we have enough data
    sequence_length = min(60, len(scaled_data) - 1)
    
    X_train, y_train = [], []
    for i in range(sequence_length, len(scaled_data)):
        X_train.append(scaled_data[i-sequence_length:i, 0])
        y_train.append(scaled_data[i, 0])
    
    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dense(units=25))
    model.add(Dense(units=1))
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=5, batch_size=16, verbose=0)
    
    future_input = scaled_data[-sequence_length:].reshape(1, sequence_length, 1)
    future_prices = []
    
    for _ in range(4):
        predicted_price = model.predict(future_input)[0, 0]
        future_prices.append(predicted_price)
        future_input = np.append(future_input[:, 1:, :], [[[predicted_price]]], axis=1)
    
    return scaler.inverse_transform(np.array(future_prices).reshape(-1, 1)).flatten()

def generate_trend_signal(data, future_prices):
    last_price = data['Close'].iloc[-1]
    predicted_price = future_prices[-1]
    price_trend = (predicted_price - last_price) / last_price * 100
    
    # Get technical signals
    current_rsi = data['RSI'].iloc[-1] if 'RSI' in data.columns else 50
    
    # Combine LSTM prediction with technical indicators
    if price_trend > 3:
        return "Strong Buy" if current_rsi < 70 else "Buy"
    elif price_trend > 1:
        return "Buy" if current_rsi < 60 else "Hold"
    elif price_trend < -3:
        return "Strong Sell" if current_rsi > 30 else "Sell"
    elif price_trend < -1:
        return "Sell" if current_rsi > 40 else "Hold"
    else:
        return "Hold"