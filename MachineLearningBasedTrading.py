import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

def update_dataframe_columns():
    """Update the dataframe columns based on user selection."""
    selected_columns = st.session_state.selected_columns
    if selected_columns:
        st.session_state.df = st.session_state['df'][selected_columns]
    else:
        st.session_state.df = pd.DataFrame()

# initialization
df = pd.DataFrame()
if 'df' not in st.session_state:
    st.session_state.df = df

st.title("Experimental Machine Learning Based Trading Strategy")

st.header("Stock Price Data")
st.write("This app fetches data from Yahoo Finance. You can enter the stock ticker and the date range to fetch the data.")

col1, col2, col3, col4 = st.columns(4, vertical_alignment="bottom")
with col1:
    ticker = st.text_input("Ticker:", "AAPL")
with col2:
    start_date = st.date_input("Start date:", pd.to_datetime("2020-01-01"))
with col3:
    end_date = st.date_input("End date:", pd.to_datetime("2023-01-01")) 
with col4:
    if st.button("Fetch Data"):
        with st.spinner("Fetching data..."):
            st.session_state.df = yf.download(ticker, start=start_date, end=end_date, multi_level_index=False, auto_adjust=False)
            

if not st.session_state.df.empty:
    st.write(f"Data for {ticker} from {start_date} to {end_date}:")
    st.session_state.df
    # Plotting the closing price
    st.write("### Closing Price")
    plt.figure(figsize=(10, 5))
    plt.plot(st.session_state.df['Close'], label='Close Price')
    plt.title(f'{ticker} Closing Price')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    st.pyplot(plt)
else:
    st.write("No data available for the selected ticker and date range. Please check the ticker symbol and date range.")

st.header("Exploratory Data Analysis")
st.write("This section provides some basic exploratory data analysis on the fetched stock data.")
if not st.session_state.df.empty:
    if 'df_selected' not in st.session_state:
        st.session_state.df_selected = st.session_state.df.copy()
    st.subheader("Columns chooser")
    columns = st.session_state.df.columns.tolist()
    selected_columns = st.multiselect("Select columns to process", columns, default=columns, on_change=update_dataframe_columns)
    if 'selected_columns' not in st.session_state:
        st.session_state.selected_columns = selected_columns
    st.write("Selected columns:", selected_columns)
    st.session_state.df_selected = st.session_state.df[selected_columns]
    st.subheader("Basic Statistics")
    st.write(st.session_state.df_selected.describe())

    st.subheader("Correlation Matrix")
    correlation_matrix = st.session_state.df_selected.corr()
    st.write(correlation_matrix)

    plt.figure(figsize=(10, 5))
    plt.imshow(correlation_matrix, cmap='coolwarm', interpolation='none')
    plt.colorbar()
    plt.title('Correlation Matrix')
    st.pyplot(plt)

    # daily returns distribution
    st.subheader("Daily Returns Distribution")
    daily_returns = st.session_state.df['Close'].pct_change().dropna()
    plt.figure(figsize=(10, 5))
    plt.hist(daily_returns, bins=50, alpha=0.7, color='blue')
    plt.xlabel('Daily Returns') 
    plt.ylabel('Frequency')
    st.pyplot(plt)

st.header("Feature Engineering")
st.write("This section allows you to create new features from the stock data.")
sma_option = st.checkbox("Simple Moving Average (SMA)", value=False)
if sma_option:
    window = st.number_input("SMA Window Size", min_value=1, value=20, step=1)
    st.session_state.df['SMA'] = st.session_state.df['Close'].rolling(window=window).mean()
    st.write("SMA added to the dataframe.")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(st.session_state.df['Close'], label='Close Price', color='blue')
    ax.plot(st.session_state.df['SMA'], label=f'SMA {window}', color='red')
    ax.set_title(f'{ticker} Closing Price with SMA')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.legend()
    st.pyplot(fig)

rsi_option = st.checkbox("Relative Strength Index (RSI)", value=False)
if rsi_option:
    window = st.number_input("RSI Window Size", min_value=1, value=14, step=1)
    delta = st.session_state.df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    st.session_state.df['RSI'] = 100 - (100 / (1 + rs))
    st.write("RSI added to the dataframe.")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Panel 1: Close Price
    ax1.plot(st.session_state.df['Close'], label='Close Price', color='blue')
    ax1.set_title(f'{ticker} Closing Price')
    ax1.set_ylabel('Price')
    ax1.legend()

    # Panel 2: RSI
    ax2.plot(st.session_state.df['RSI'], label='RSI', color='blue')
    ax2.axhline(70, color='red', linestyle='--', label='Overbought (70)')
    ax2.axhline(30, color='green', linestyle='--', label='Oversold (30)')
    ax2.set_title('Relative Strength Index (RSI)')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('RSI')
    ax2.legend()

    plt.tight_layout()
    st.pyplot(fig)