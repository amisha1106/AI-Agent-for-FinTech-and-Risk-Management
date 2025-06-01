import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import yfinance as yf

class InvestorProfile:
    def __init__(self):
        self.risk_scores = {}
        self.investor_data = {}
        self.recommendations = {}
        self.cluster_model = None
        self.clusters = None
        
    def show_risk_assessment(self):
        st.header("Investor Risk Assessment")
        
        # Personal information
        st.subheader("Personal Information")
        col1, col2 = st.columns(2)
        with col1:
            name = st.text_input("Full Name", key="name_input")
            age = st.number_input("Age", min_value=18, max_value=100, value=35, key="age_input")
        with col2:
            email = st.text_input("Email", key="email_input")
            income = st.select_slider("Annual Income", 
                                   options=["<$30,000", "$30,000-$60,000", "$60,000-$100,000", "$100,000-$200,000", ">$200,000"],
                                   value="$60,000-$100,000", key="income_input")
        
        # Investment goals
        st.subheader("Investment Goals")
        investment_goal = st.selectbox("Primary Investment Goal", 
                                     ["Retirement", "Short-term Savings", "Income Generation", "Capital Growth", "Financial Security"],
                                     key="goal_input")
        
        time_horizon = st.radio("Investment Time Horizon", 
                               ["Short-term (0-2 years)", "Medium-term (2-5 years)", "Long-term (5+ years)"],
                               key="horizon_input")
        
        # Risk tolerance questionnaire
        st.subheader("Risk Tolerance Assessment")
        
        q1 = st.select_slider(
            "1. How would you react if your portfolio suddenly decreased by 20%?",
            options=["Sell everything immediately", "Sell some investments", "Do nothing", "Buy more at lower prices"],
            value="Do nothing", key="q1_input"
        )
        
        q2 = st.select_slider(
            "2. Which investment scenario do you prefer?",
            options=["Low risk, low return", "Moderate risk, moderate return", "High risk, high return"],
            value="Moderate risk, moderate return", key="q2_input"
        )
        
        q3 = st.select_slider(
            "3. What percentage of your portfolio are you willing to invest in high-risk assets?",
            options=["0-10%", "10-25%", "25-50%", "50-75%", "75-100%"],
            value="25-50%", key="q3_input"
        )
        
        q4 = st.select_slider(
            "4. How much financial knowledge do you have?",
            options=["None", "Basic", "Intermediate", "Advanced", "Professional"],
            value="Intermediate", key="q4_input"
        )
        
        q5 = st.select_slider(
            "5. Which statement best describes your investment approach?",
            options=[
                "I want to preserve my capital above all else",
                "I want stable growth with minimal risk",
                "I want balanced growth with moderate risk",
                "I want substantial growth and am comfortable with volatility",
                "I want maximum growth and can accept significant volatility"
            ],
            value="I want balanced growth with moderate risk", key="q5_input"
        )
        
        # Calculate risk scores if all questions are answered
        submit = st.button("Calculate Risk Profile")
        
        if submit:
            # Calculate risk score (1-10 scale)
            q1_score = {"Sell everything immediately": 1, "Sell some investments": 3, 
                       "Do nothing": 6, "Buy more at lower prices": 10}[q1]
            
            q2_score = {"Low risk, low return": 2, "Moderate risk, moderate return": 6, 
                       "High risk, high return": 10}[q2]
            
            q3_score = {"0-10%": 1, "10-25%": 3, "25-50%": 6, "50-75%": 8, "75-100%": 10}[q3]
            
            q4_score = {"None": 1, "Basic": 3, "Intermediate": 5, "Advanced": 8, 
                       "Professional": 10}[q4]
            
            q5_score = {"I want to preserve my capital above all else": 1,
                       "I want stable growth with minimal risk": 3,
                       "I want balanced growth with moderate risk": 6,
                       "I want substantial growth and am comfortable with volatility": 8,
                       "I want maximum growth and can accept significant volatility": 10}[q5]
            
            # Calculate age factor (younger investors can take more risk)
            age_factor = max(1, 10 - ((age - 20) // 10))
            
            # Calculate time horizon factor
            horizon_factor = {"Short-term (0-2 years)": 2, "Medium-term (2-5 years)": 6,
                            "Long-term (5+ years)": 10}[time_horizon]
            
            # Final risk score calculation
            risk_score = (q1_score * 0.25 + q2_score * 0.2 + q3_score * 0.15 + 
                        q4_score * 0.1 + q5_score * 0.15 + age_factor * 0.05 + 
                        horizon_factor * 0.1)
            
            risk_category = ""
            if risk_score < 3:
                risk_category = "Very Conservative"
            elif risk_score < 5:
                risk_category = "Conservative"
            elif risk_score < 7:
                risk_category = "Moderate"
            elif risk_score < 9:
                risk_category = "Aggressive"
            else:
                risk_category = "Very Aggressive"
            
            # Store risk scores and investor data
            self.risk_scores = {
                "overall": risk_score,
                "market_reaction": q1_score,
                "risk_preference": q2_score,
                "high_risk_allocation": q3_score,
                "knowledge": q4_score,
                "investment_style": q5_score,
                "age_factor": age_factor,
                "time_horizon": horizon_factor,
                "category": risk_category
            }
            
            self.investor_data = {
                "name": name,
                "age": age,
                "email": email,
                "income": income,
                "goal": investment_goal,
                "time_horizon": time_horizon
            }
            
            # Display risk profile
            st.success(f"Risk Assessment Complete!")
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                # Display risk meter
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = risk_score,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': f"Risk Tolerance: {risk_category}"},
                    gauge = {
                        'axis': {'range': [0, 10]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 3], 'color': "green"},
                            {'range': [3, 5], 'color': "lightgreen"},
                            {'range': [5, 7], 'color': "yellow"},
                            {'range': [7, 9], 'color': "orange"},
                            {'range': [9, 10], 'color': "red"}
                        ],
                    }
                ))
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Show risk breakdown radar chart
                categories = ['Market Reaction', 'Risk Preference', 'High-Risk Allocation', 
                            'Knowledge', 'Investment Style']
                values = [q1_score, q2_score, q3_score, q4_score, q5_score]
                
                fig = go.Figure()
                fig.add_trace(go.Scatterpolar(
                    r=values,
                    theta=categories,
                    fill='toself',
                    name='Risk Profile'
                ))
                
                fig.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[0, 10]
                        )
                    ),
                    showlegend=False,
                    title="Risk Profile Breakdown"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Generate investment recommendations based on profile
            self.generate_recommendations()
            
            # Show recommendations
            st.subheader("Personalized Investment Recommendations")
            
            # Asset allocation
            st.write("**Recommended Asset Allocation:**")
            
            allocation_data = pd.DataFrame({
                'Asset': self.recommendations['allocation'].keys(),
                'Percentage': self.recommendations['allocation'].values()
            })
            
            fig = px.pie(allocation_data, values='Percentage', names='Asset', title='Recommended Asset Allocation')
            st.plotly_chart(fig, use_container_width=True)
            
            # Investment suggestions
            st.write("**Suggested Investments:**")
            
            for category, investments in self.recommendations['investments'].items():
                st.write(f"*{category}:*")
                for investment in investments:
                    st.write(f"- {investment}")
            
            # Strategy suggestions
            st.write("**Strategy Recommendations:**")
            for strategy in self.recommendations['strategies']:
                st.write(f"- {strategy}")
                
            # Show personalized alerts
            self.generate_alerts()
    
    def generate_recommendations(self):
        """Generate investment recommendations based on risk profile"""
        
        risk_score = self.risk_scores["overall"]
        risk_category = self.risk_scores["category"]
        goal = self.investor_data["goal"]
        time_horizon = self.investor_data["time_horizon"]
        
        # Asset allocation based on risk score
        allocation = {}
        
        # Very Conservative (0-3)
        if risk_score < 3:
            allocation = {
                "Cash & Cash Equivalents": 20,
                "Bonds": 60,
                "Stocks": 15,
                "Alternative Investments": 5
            }
        # Conservative (3-5)
        elif risk_score < 5:
            allocation = {
                "Cash & Cash Equivalents": 10,
                "Bonds": 50,
                "Stocks": 35,
                "Alternative Investments": 5
            }
        # Moderate (5-7)
        elif risk_score < 7:
            allocation = {
                "Cash & Cash Equivalents": 5,
                "Bonds": 35,
                "Stocks": 55,
                "Alternative Investments": 5
            }
        # Aggressive (7-9)
        elif risk_score < 9:
            allocation = {
                "Cash & Cash Equivalents": 5,
                "Bonds": 15,
                "Stocks": 70,
                "Alternative Investments": 10
            }
        # Very Aggressive (9-10)
        else:
            allocation = {
                "Cash & Cash Equivalents": 0,
                "Bonds": 5,
                "Stocks": 80,
                "Alternative Investments": 15
            }
        
        # Investment suggestions based on risk category and investment goal
        investments = {}
        
        # Cash & Equivalents suggestions
        if allocation["Cash & Cash Equivalents"] > 0:
            investments["Cash & Equivalents"] = [
                "High-yield savings accounts",
                "Money market funds",
                "Short-term CDs"
            ]
        
        # Bond suggestions based on risk profile
        bond_options = []
        if risk_score < 5:  # Conservative investors
            bond_options = [
                "Short-term Treasury bonds",
                "Investment-grade corporate bonds",
                "Municipal bonds"
            ]
        else:  # More aggressive investors
            bond_options = [
                "Corporate bonds",
                "High-yield bonds",
                "International bonds", 
                "Emerging market bonds"
            ]
        
        if allocation["Bonds"] > 0:
            investments["Bonds"] = bond_options
        
        # Stock suggestions based on risk profile
        stock_options = []
        if risk_score < 3:  # Very Conservative
            stock_options = [
                "Large-cap value ETFs",
                "Dividend aristocrat funds",
                "Low-volatility stock funds"
            ]
        elif risk_score < 5:  # Conservative
            stock_options = [
                "Large-cap blend funds",
                "Dividend ETFs",
                "Quality factor ETFs"
            ]
        elif risk_score < 7:  # Moderate
            stock_options = [
                "Total market index funds",
                "Growth and value blend",
                "International developed markets funds"
            ]
        elif risk_score < 9:  # Aggressive
            stock_options = [
                "Small-cap growth funds",
                "Sector-specific ETFs",
                "Emerging market funds",
                "International small-cap funds"
            ]
        else:  # Very Aggressive
            stock_options = [
                "Leveraged ETFs",
                "Small-cap growth stocks",
                "Emerging market funds",
                "Thematic tech funds"
            ]
        
        if allocation["Stocks"] > 0:
            investments["Stocks"] = stock_options
        
        # Alternative investments based on risk profile
        alt_options = []
        if risk_score < 5:  # Conservative
            alt_options = [
                "REITs",
                "Preferred stocks"
            ]
        elif risk_score < 7:  # Moderate
            alt_options = [
                "REITs",
                "Commodity ETFs",
                "Preferred stocks"
            ]
        else:  # Aggressive
            alt_options = [
                "REITs",
                "Commodity ETFs",
                "Cryptocurrency allocation (1-5%)",
                "Private equity funds"
            ]
        
        if allocation["Alternative Investments"] > 0:
            investments["Alternative Investments"] = alt_options
        
        # Strategy recommendations based on goal and risk profile
        strategies = []
        
        # Basic strategies for everyone
        strategies.append("Diversify investments across asset classes to reduce risk")
        
        if "Short-term" in time_horizon:
            strategies.append("Focus on capital preservation and liquidity")
            strategies.append("Use dollar-cost averaging for entry points")
        elif "Medium-term" in time_horizon:
            strategies.append("Balance between growth and capital preservation")
            strategies.append("Consider tax-efficient investment strategies")
        else:  # Long-term
            strategies.append("Focus on long-term growth and compound returns")
            strategies.append("Maximize tax-advantaged accounts (401k, IRA)")
        
        # Goal-specific strategies
        if goal == "Retirement":
            strategies.append("Prioritize tax-advantaged retirement accounts")
            strategies.append("Gradually shift to more conservative allocation as retirement approaches")
        elif goal == "Short-term Savings":
            strategies.append("Focus on high-liquidity, low-volatility investments")
            strategies.append("Consider laddered CDs or short-term Treasury securities")
        elif goal == "Income Generation":
            strategies.append("Focus on dividend-paying stocks and fixed-income investments")
            strategies.append("Consider REITs and preferred stocks for income")
        elif goal == "Capital Growth":
            strategies.append("Emphasize growth-oriented investments like growth stocks and ETFs")
            strategies.append("Consider systematic rebalancing to manage risk")
        elif goal == "Financial Security":
            strategies.append("Build emergency fund before taking on more investment risk")
            strategies.append("Consider appropriate insurance products to protect assets")
        
        # Risk-specific additional strategies
        if risk_score < 5:  # Conservative
            strategies.append("Implement stop-loss orders to protect against significant drawdowns")
        elif risk_score >= 7:  # Aggressive
            strategies.append("Consider tactical allocation to take advantage of market opportunities")
        
        self.recommendations = {
            "allocation": allocation,
            "investments": investments,
            "strategies": strategies
        }
        
        return self.recommendations
    
    def generate_alerts(self):
        """Generate personalized alerts based on investor profile and market conditions"""
        st.subheader("Personalized Alerts")
        
        risk_score = self.risk_scores["overall"]
        risk_category = self.risk_scores["category"]
        
        alerts = []
        
        # Sample market-based alerts (would be dynamic in a real application)
        market_condition = np.random.choice(["bull", "bear", "volatile", "stable"])
        
        if market_condition == "bull" and risk_score < 5:
            alerts.append({
                "type": "opportunity",
                "message": "Market is trending upward. Consider temporarily increasing equity exposure.",
                "priority": "medium"
            })
        elif market_condition == "bear" and risk_score >= 7:
            alerts.append({
                "type": "risk",
                "message": "Market downturn detected. Consider hedging strategies to protect portfolio.",
                "priority": "high"
            })
        elif market_condition == "volatile":
            alerts.append({
                "type": "risk",
                "message": "Increased market volatility detected. Review portfolio diversification.",
                "priority": "medium"
            })
        
        # Risk-based alerts
        if risk_score >= 8:
            alerts.append({
                "type": "risk",
                "message": "Your portfolio has high risk exposure. Consider stress-testing against various scenarios.",
                "priority": "medium"
            })
        
        # Portfolio concentration alerts (simulated)
        sector_concentration = np.random.choice([True, False], p=[0.3, 0.7])
        if sector_concentration:
            alerts.append({
                "type": "risk",
                "message": "High technology sector concentration detected. Consider diversifying across sectors.",
                "priority": "medium"
            })
        
        # Display alerts
        if alerts:
            for alert in alerts:
                color = "red" if alert["type"] == "risk" else "green"
                priority_text = f"Priority: {alert['priority'].upper()}"
                st.markdown(f"<div style='padding:10px; background-color:{color}; color:white; border-radius:5px;'><b>{alert['type'].upper()}:</b> {alert['message']} ({priority_text})</div>", unsafe_allow_html=True)
        else:
            st.info("No alerts at this time. Your portfolio is aligned with your risk profile.")
    
    def show_behavioral_analysis(self):
        st.header("Investor Behavioral Analysis")
        
        # In a real app, this would use actual investor data
        # For demo purposes, we'll generate synthetic data
        
        # Sample data generation for clustering
        st.subheader("Investor Clustering Analysis")
        st.write("This analysis helps identify patterns among investors with similar profiles.")
        
        # Generate sample data
        n_samples = 500
        np.random.seed(42)
        
        # Generate synthetic investor data
        ages = np.random.normal(45, 15, n_samples).clip(18, 85)
        risk_scores = np.random.normal(5, 2, n_samples).clip(1, 10)
        investment_amounts = np.exp(np.random.normal(10, 1, n_samples)).clip(1000, 1000000)
        trading_frequency = np.random.exponential(10, n_samples).clip(0, 100)
        
        # Create DataFrame
        investor_df = pd.DataFrame({
            'Age': ages,
            'Risk_Score': risk_scores,
            'Investment_Amount': investment_amounts,
            'Trading_Frequency': trading_frequency
        })
        
        # Display sample data
        st.write("Sample investor population data:")
        st.dataframe(investor_df.head())
        
        # Scale features for clustering
        from sklearn.preprocessing import StandardScaler
        
        features = ['Age', 'Risk_Score', 'Investment_Amount', 'Trading_Frequency']
        X = investor_df[features].copy()
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Perform clustering
        clustering_method = st.radio("Select Clustering Method", ["K-Means", "DBSCAN"])
        
        if clustering_method == "K-Means":
            n_clusters = st.slider("Number of Clusters", min_value=2, max_value=10, value=4)
            
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            investor_df['Cluster'] = kmeans.fit_predict(X_scaled)
            self.cluster_model = kmeans
            
            # Display cluster centers
            cluster_centers = pd.DataFrame(scaler.inverse_transform(kmeans.cluster_centers_), 
                                         columns=features)
            
            st.write("Cluster Centers (Average Investor in Each Group):")
            st.dataframe(cluster_centers.round(2))
            
            # Visualization
            fig = px.scatter_3d(investor_df, x='Age', y='Risk_Score', z='Investment_Amount',
                              color='Cluster', size='Trading_Frequency',
                              labels={'Cluster': 'Investor Group'})
            
            fig.update_layout(title="Investor Clustering Visualization")
            st.plotly_chart(fig, use_container_width=True)
            
            # Interpret clusters
            st.subheader("Investor Group Profiles")
            
            for i in range(n_clusters):
                cluster_data = investor_df[investor_df['Cluster'] == i]
                
                avg_age = cluster_data['Age'].mean()
                avg_risk = cluster_data['Risk_Score'].mean()
                avg_investment = cluster_data['Investment_Amount'].mean()
                avg_trading = cluster_data['Trading_Frequency'].mean()
                cluster_size = len(cluster_data)
                
                # Risk category interpretation
                risk_category = "Conservative"
                if avg_risk < 3:
                    risk_category = "Very Conservative"
                elif avg_risk < 5:
                    risk_category = "Conservative" 
                elif avg_risk < 7:
                    risk_category = "Moderate"
                elif avg_risk < 9:
                    risk_category = "Aggressive"
                else:
                    risk_category = "Very Aggressive"
                
                # Trading style interpretation
                trading_style = "Active Trader"
                if avg_trading < 5:
                    trading_style = "Buy and Hold"
                elif avg_trading < 20:
                    trading_style = "Occasional Trader"
                else:
                    trading_style = "Active Trader"
                
                st.markdown(f"**Group {i+1} ({cluster_size} investors):**")
                st.markdown(f"- **Age Profile**: {avg_age:.1f} years")
                st.markdown(f"- **Risk Tolerance**: {avg_risk:.1f}/10 ({risk_category})")
                st.markdown(f"- **Average Investment**: ${avg_investment:,.2f}")
                st.markdown(f"- **Trading Style**: {trading_style} ({avg_trading:.1f} trades/month)")
                
                # Recommended strategies
                st.markdown("**Recommended Group Strategies:**")
                
                if risk_category in ["Very Conservative", "Conservative"]:
                    st.markdown("- Focus on capital preservation and income generation")
                    st.markdown("- Emphasize dividend-paying stocks and bonds")
                    st.markdown("- Consider automatic rebalancing to maintain target allocations")
                elif risk_category == "Moderate":
                    st.markdown("- Balance between growth and income investments")
                    st.markdown("- Consider a core-satellite investment approach")
                    st.markdown("- Regular portfolio reviews and tactical adjustments")
                else:
                    st.markdown("- Emphasize growth-oriented investments")
                    st.markdown("- Consider increasing international and emerging market exposure")
                    st.markdown("- Implement risk management techniques for high-volatility assets")
                
                st.markdown("---")
            
        else:  # DBSCAN
            eps = st.slider("Maximum distance between samples (eps)", 
                          min_value=0.1, max_value=2.0, value=0.5, step=0.1)
            min_samples = st.slider("Minimum samples per cluster", 
                                  min_value=5, max_value=50, value=15)
            
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            investor_df['Cluster'] = dbscan.fit_predict(X_scaled)
            self.cluster_model = dbscan
            
            # Count clusters and noise points
            n_clusters = len(set(investor_df['Cluster'])) - (1 if -1 in investor_df['Cluster'] else 0)
            n_noise = list(investor_df['Cluster']).count(-1)
            
            st.write(f"Number of clusters: {n_clusters}")
            st.write(f"Number of outliers: {n_noise}")
            
            # Visualization
            fig = px.scatter_3d(investor_df, x='Age', y='Risk_Score', z='Investment_Amount',
                              color='Cluster', size='Trading_Frequency',
                              labels={'Cluster': 'Investor Group'})
            
            fig.update_layout(title="Investor Clustering Visualization")
            st.plotly_chart(fig, use_container_width=True)
            
            # Display cluster statistics excluding noise
            if n_clusters > 0:
                st.subheader("Investor Group Profiles")
                
                for i in sorted(set(investor_df['Cluster'])):
                    if i == -1:  # Skip noise points
                        continue
                        
                    cluster_data = investor_df[investor_df['Cluster'] == i]
                    
                    avg_age = cluster_data['Age'].mean()
                    avg_risk = cluster_data['Risk_Score'].mean()
                    avg_investment = cluster_data['Investment_Amount'].mean()
                    avg_trading = cluster_data['Trading_Frequency'].mean()
                    cluster_size = len(cluster_data)
                    
                    # Risk category interpretation
                    risk_category = "Conservative"
                    if avg_risk < 3:
                        risk_category = "Very Conservative"
                    elif avg_risk < 5:
                        risk_category = "Conservative" 
                    elif avg_risk < 7:
                        risk_category = "Moderate"
                    elif avg_risk < 9:
                        risk_category = "Aggressive"
                    else:
                        risk_category = "Very Aggressive"
                    
                    st.markdown(f"**Group {i+1} ({cluster_size} investors):**")
                    st.markdown(f"- **Age Profile**: {avg_age:.1f} years")
                    st.markdown(f"- **Risk Tolerance**: {avg_risk:.1f}/10 ({risk_category})")
                    st.markdown(f"- **Average Investment**: ${avg_investment:,.2f}")
                    st.markdown(f"- **Trading Frequency**: {avg_trading:.1f} trades/month")
                    st.markdown("---")
        
        # Behavioral insights based on clustering
        st.subheader("Behavioral Finance Insights")
        
        behavioral_biases = {
            "Loss Aversion": "Investors feel losses more deeply than equivalent gains",
            "Overconfidence": "Overestimating investment knowledge or ability to time markets",
            "Herding": "Following the crowd rather than independent analysis",
            "Recency Bias": "Overemphasizing recent events in decision-making",
            "Anchoring": "Relying too heavily on the first piece of information encountered"
        }
        
        # Display explanations of common biases
        for bias, description in behavioral_biases.items():
            with st.expander(f"{bias}"):
                st.write(description)
                
                # Randomly determine if this bias is likely for current investor
                has_bias = np.random.choice([True, False], p=[0.4, 0.6])
                
                if has_bias:
                    st.warning(f"Risk Assessment: You may exhibit {bias.lower()} in your trading patterns.")
                    
                    # Mitigation strategies
                    st.markdown("**Mitigation Strategies:**")
                    if bias == "Loss Aversion":
                        st.markdown("- Set predefined exit strategies before investing")
                        st.markdown("- Use automated stop-loss orders to remove emotion")
                        st.markdown("- Focus on long-term performance rather than short-term fluctuations")
                    elif bias == "Overconfidence":
                        st.markdown("- Track all trades and review performance regularly")
                        st.markdown("- Consider a portion of your portfolio in index funds")
                        st.markdown("- Seek contrarian opinions before making large investments")
                    elif bias == "Herding":
                        st.markdown("- Develop a personalized investment strategy and stick to it")
                        st.markdown("- Research investments independently before following trends")
                        st.markdown("- Set thresholds for when to review but not necessarily change strategy")
                    elif bias == "Recency Bias":
                        st.markdown("- Study longer-term market history and cycles")
                        st.markdown("- Use dollar-cost averaging to avoid timing decisions")
                        st.markdown("- Maintain consistent asset allocation through market cycles")
                    elif bias == "Anchoring":
                        st.markdown("- Use multiple valuation methods for investment decisions")
                        st.markdown("- Consider different time periods for performance analysis")
                        st.markdown("- Deliberately seek information that challenges your initial views")

def show():
    """Main function to display the investor profile and recommendations"""
    
    st.title("Investor Profiles & Recommendations")
    
    investor = InvestorProfile()
    
    # Create tabs for different sections
    tabs = st.tabs(["Risk Assessment", "Behavioral Analysis"])
    
    with tabs[0]:
        investor.show_risk_assessment()
    
    with tabs[1]:
        investor.show_behavioral_analysis()