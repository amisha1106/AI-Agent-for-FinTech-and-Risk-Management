import streamlit as st
import dashboard, portfolio, stock_analysis, investor, company
import json
import os

# Page configuration
st.set_page_config(page_title="Stock Trading Platform", layout="wide")

# Function to load user credentials from a file
def load_credentials():
    if os.path.exists("user_credentials.json"):
        with open("user_credentials.json", "r") as file:
            return json.load(file)
    else:
        # Default credentials
        default_credentials = {
            "admin": {"password": "admin123", "role": "Admin"},
            "investor": {"password": "investor123", "role": "Investor"},
            "company": {"password": "company123", "role": "Company"},
        }
        # Save default credentials
        with open("user_credentials.json", "w") as file:
            json.dump(default_credentials, file)
        return default_credentials

# Function to save user credentials to a file
def save_credentials(credentials):
    with open("user_credentials.json", "w") as file:
        json.dump(credentials, file)

# Load user credentials
if "user_credentials" not in st.session_state:
    st.session_state["user_credentials"] = load_credentials()

# Session state initialization
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

if "user_role" not in st.session_state:
    st.session_state["user_role"] = "Guest"  # Default role is Guest

if "show_login" not in st.session_state:
    st.session_state["show_login"] = True

if "show_register" not in st.session_state:
    st.session_state["show_register"] = False

# Registration Page
def register():
    st.sidebar.title("Register")
    
    new_username = st.sidebar.text_input("Choose Username", key="new_username")
    new_password = st.sidebar.text_input("Choose Password", type="password", key="new_password")
    confirm_password = st.sidebar.text_input("Confirm Password", type="password", key="confirm_password")
    role = st.sidebar.selectbox("Register as", ["Investor", "Company"])
    
    if st.sidebar.button("Register"):
        if new_username and new_password and confirm_password:
            if new_username in st.session_state["user_credentials"]:
                st.sidebar.error("Username already exists!")
            elif new_password != confirm_password:
                st.sidebar.error("Passwords do not match!")
            else:
                # Add new user
                st.session_state["user_credentials"][new_username] = {
                    "password": new_password,
                    "role": role
                }
                # Save updated credentials
                save_credentials(st.session_state["user_credentials"])
                st.sidebar.success("Registration successful! Please login.")
                st.session_state["show_register"] = False
                st.session_state["show_login"] = True
                st.rerun()
        else:
            st.sidebar.error("Please fill all fields!")
    
    if st.sidebar.button("Back to Login"):
        st.session_state["show_register"] = False
        st.session_state["show_login"] = True
        st.rerun()

# Authentication Page
def login():
    st.sidebar.image("https://your-logo-url.com", width=100)
    st.sidebar.title("Login")

    username = st.sidebar.text_input("Username", key="username")
    password = st.sidebar.text_input("Password", type="password", key="password")

    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        if st.button("Login"):
            if username in st.session_state["user_credentials"] and st.session_state["user_credentials"][username]["password"] == password:
                st.session_state["authenticated"] = True
                st.session_state["user_role"] = st.session_state["user_credentials"][username]["role"]
                st.sidebar.success(f"Logged in as {st.session_state['user_role']}")
                st.rerun()
            else:
                st.sidebar.error("Invalid Username or Password!")
    
    with col2:
        if st.button("Register"):
            st.session_state["show_login"] = False
            st.session_state["show_register"] = True
            st.rerun()
    
    # Continue as guest option
    if st.sidebar.button("Continue as Guest"):
        st.session_state["user_role"] = "Guest"
        st.rerun()

# Logout function
def logout():
    st.session_state["authenticated"] = False
    st.session_state["user_role"] = "Guest"  # Reset to Guest instead of None
    st.session_state["show_login"] = True
    st.session_state["show_register"] = False
    st.rerun()

# Main Application
st.sidebar.title("Stock Trading Platform")

# Authentication flow
if not st.session_state["authenticated"]:
    st.sidebar.subheader(f"Welcome, Guest")
    if st.session_state["show_login"]:
        login()
    elif st.session_state["show_register"]:
        register()
else:
    st.sidebar.subheader(f"Welcome, {st.session_state['user_role']}")
    # Logout Button
    if st.sidebar.button("Logout"):
        logout()

# Navigation based on user role
if st.session_state["user_role"] == "Admin":
    selected = st.sidebar.radio("Navigate to:", ["Dashboard", "Portfolio", "Stock Analysis", "Investor", "Company"])
elif st.session_state["user_role"] == "Investor":
    selected = st.sidebar.radio("Navigate to:", ["Dashboard", "Portfolio", "Stock Analysis", "Investor"])
elif st.session_state["user_role"] == "Company":
    selected = st.sidebar.radio("Navigate to:", ["Dashboard", "Company"])
else:  # Guest user
    selected = "Stock Analysis"  # Guest can only access Stock Analysis
    st.sidebar.info("Please login or register to access more features")

# Page Redirection
if selected == "Dashboard":
    dashboard.show()
elif selected == "Portfolio":
    portfolio.show()
elif selected == "Stock Analysis":
    stock_analysis.show()
elif selected == "Investor":
    investor.show()
elif selected == "Company":
    company.show()