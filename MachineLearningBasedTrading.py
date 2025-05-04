import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier

# counter for the number of yfinance API calls
if 'counter' not in st.session_state:
    st.session_state.counter = 0

# dialog for revision history
@st.dialog("Revision History")
def revision_history():
    st.write("Version 1.0 - May 2025")
    st.write("Initial release with basic features and functionalities.")
    st.write("Features:")
    st.write("- Fetch stock data from Yahoo Finance")
    st.write("- Basic exploratory data analysis (EDA)")
    st.write("- Data labelling (Fixed time horizon, Triple barriers)")
    st.write("- Features engineering (SMA, RSI, MACD, OBV, ATR)")
    st.write("- Machine Learning Algorithms selection (KNN, Decision Tree, Random Forest, XGBoost, LightGBM) and Parameters")
    st.write("- Model training and testing")
    st.write("- Classification report and confusion matrix")

# Dataframes initialization
# Original dataframe from Yahoo Finance
df = pd.DataFrame()
if 'df' not in st.session_state:
    st.session_state.df = df

# dataframe with label
df_label = pd.DataFrame()
if 'df_label' not in st.session_state:
    st.session_state.df_label = df_label

# Selected dataframe for EDA
df_selected = pd.DataFrame()
if 'df_selected' not in st.session_state:
    st.session_state.df_selected = df_selected

# dataframe with features
df_features = pd.DataFrame()
if 'df_features' not in st.session_state:
    st.session_state.df_features = df_features

# dataframe with selected features for ML
df_selected_features = pd.DataFrame()
if 'df_selected_features' not in st.session_state:
    st.session_state.df_selected_features = df_selected_features

st.title("Machine Learning-based Trading Strategy")
st.write("Created by **Yuan Lukito** (yuanlukito@ti.ukdw.ac.id) and **Nugroho Agus Haryono** (nugroho@staff.ukdw.ac.id)")
st.image("ukdw.png")
st.write("**Program Studi Informatika, Fakultas Teknologi Informasi**")
st.write("**Universitas Kristen Duta Wacana**")


st.write("Version 1.0 - May 2025")
if st.button("Revision History"):
    revision_history()
st.divider()

################################################################################
# STOCK PRICE DATA
# This section fetches stock data from Yahoo Finance based on user input.
################################################################################

st.header("Stock Price Data")
st.write("This app fetches data from Yahoo Finance. You can enter the stock ticker and the date range to fetch the data.")

col1, col2, col3, col4 = st.columns(4, vertical_alignment="bottom")
with col1:
    ticker = st.text_input("Ticker:", "AAPL")
with col2:
    start_date = st.date_input("Start date:", pd.to_datetime("2020-01-01"))
with col3:
    end_date = st.date_input("End date:", pd.to_datetime("2023-12-31"))
with col4:
    if st.button("Fetch Data"):
        with st.spinner("Fetching data..."):
            st.session_state.df = yf.download(ticker, start=start_date, end=end_date, multi_level_index=False, auto_adjust=False)
            st.session_state.counter += 1

if st.session_state.df.empty and st.session_state.counter > 0:
    st.error("No data available. Please check the ticker symbol and date range.")       
            

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

st.divider()

################################################################################
# EXPLORATORY DATA ANALYSIS
# This section provides some basic exploratory data analysis on the 
# fetched stock data.
################################################################################

if not st.session_state.df.empty:
    st.header("Exploratory Data Analysis")
    st.write("This section provides some basic exploratory data analysis on the fetched stock data.")

    if st.session_state.df_selected.empty:
        st.session_state.df_selected = st.session_state.df.copy()
    
    st.subheader("Basic Statistics")
    st.write(st.session_state.df_selected.describe())

    st.subheader("Correlation Matrix")
    correlation_matrix = st.session_state.df_selected.corr()
    st.write(correlation_matrix)

    plt.figure(figsize=(10, 5))
    plt.imshow(correlation_matrix, cmap='coolwarm', interpolation='none')
    plt.colorbar()
    plt.title('Correlation Matrix')
    plt.xticks(ticks=np.arange(len(correlation_matrix.columns)), labels=correlation_matrix.columns, rotation=90)
    plt.yticks(ticks=np.arange(len(correlation_matrix.index)), labels=correlation_matrix.index)

    # Adding annotations
    for i in range(len(correlation_matrix)):
        for j in range(len(correlation_matrix.columns)):
            plt.text(j, i, f'{correlation_matrix.iloc[i, j]:.2f}', 
                    ha='center', va='center', color='black')

    st.pyplot(plt)

    # daily returns distribution
    st.subheader("Daily Returns Distribution")
    daily_returns = st.session_state.df['Close'].pct_change().dropna()
    plt.figure(figsize=(10, 5))
    plt.hist(daily_returns, bins=50, alpha=0.7, color='blue')
    plt.xlabel('Daily Returns') 
    plt.ylabel('Frequency')
    st.pyplot(plt)
    st.divider()

################################################################################
# DATA LABELLING
# This section allows you to label the data for machine learning.
################################################################################

if not st.session_state.df.empty:
    st.header("Data Labelling")
    st.write("This section allows you to label the data for machine learning.")
    
    st.session_state.df_label = st.session_state.df.copy()
    
    labelling_option = st.selectbox("Select labelling method", ["Fixed time horizon", "Triple barriers"])
    if labelling_option == "Fixed time horizon":
        st.write("This method labels the data based on a fixed time horizon. It compares current Close with current+horizon Close")
        horizon = st.number_input("Horizon (in days)", min_value=1, value=5, step=1)
        st.session_state.df_label['Label'] = np.where(st.session_state.df['Close'].shift(-horizon) > st.session_state.df['Close'], "Buy", "Hold")
        st.session_state.df_label['Label'] = np.where(st.session_state.df['Close'].shift(-horizon) < st.session_state.df['Close'], "Sell", st.session_state.df_label['Label'])
        st.write("Labels added to the dataframe.")

    elif labelling_option == "Triple barriers":
        st.write("This method labels the data based on triple barriers.")
        collabel1, collabel2, collabel3 = st.columns(3, vertical_alignment="bottom")
        with collabel1:
            barrier1 = st.number_input("Upper Barrier (%)", min_value=0.0, value=5.0, step=0.1) / 100
        with collabel2:
            barrier2 = st.number_input("Lower Barrier (%)", min_value=-20.0, value=-2.5, step=0.1) / 100
        with collabel3:
            time_horizon = st.number_input("Fixed Time (days)", min_value=1, value=10, step=1)

        # triple barriers logic
        labels = []
        for i in range(len(st.session_state.df_label)):
            price = st.session_state.df_label.Close.iloc[i]
            upper = price * (1 + barrier1)
            lower = price * (1 + barrier2)
            for t in range(1, time_horizon+1):
                if i + t >= len(st.session_state.df_label):
                    labels.append('Hold')
                    break
                future_price = st.session_state.df_label.Close.iloc[i + t]
                if future_price >= upper:
                    labels.append('Buy')
                    break
                if future_price <= lower:
                    labels.append('Sell')
                    break
            else:
                labels.append('Hold')
        st.session_state.df_label['Label'] = labels
        st.write("Labels added to the dataframe.")
        st.session_state.df_label
    
    if not st.session_state.df_label.empty:
        label_values = st.session_state.df_label['Label'].value_counts()
        st.write("Label counts:")
        ordered_labels = ["Buy", "Hold", "Sell"]
        label_values = label_values.reindex(ordered_labels, fill_value=0)
        st.write(label_values)
        # Ensure the order of labels is "Buy", "Hold", "Sell"
        ordered_labels = ["Buy", "Hold", "Sell"]
        label_values = label_values.reindex(ordered_labels, fill_value=0)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(label_values.index, label_values.values, color=['green', 'yellow', 'red'])
        ax.set_title('Label Counts')
        ax.set_xlabel('Labels')
        ax.set_ylabel('Counts')
        st.pyplot(fig)
    st.divider()

################################################################################
# FEATURES ENGINEERING
# This section allows you to create new features using technical indicators 
# derived from the stock data.
################################################################################

if not st.session_state.df_label.empty:
    st.header("Features Engineering")
    st.write("This section allows you to create new features from the stock data.")
    st.session_state.df_features = st.session_state.df_label.copy()
    
    st.write("### Trend Following Indicators")

    # Simple Moving Average
    sma_option = st.checkbox("Simple Moving Average (SMA)", value=False)
    if sma_option:
        window = st.number_input("SMA Window Size", min_value=1, value=20, step=1)
        st.session_state.df_features['SMA'] = st.session_state.df_features['Close'].rolling(window=window).mean()
        st.write("SMA added to the dataframe.")
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(st.session_state.df_features['Close'], label='Close Price', color='blue')
        ax.plot(st.session_state.df_features['SMA'], label=f'SMA {window}', color='red')
        ax.set_title(f'{ticker} Closing Price with SMA')
        ax.set_xlabel('Date')
        ax.set_ylabel('Price')
        ax.legend()
        st.pyplot(fig)

    macd_option = st.checkbox("Moving Average Convergence Divergence (MACD)", value=False)
    if macd_option:
        short_window = st.number_input("Short EMA Window Size", min_value=1, value=12, step=1)
        long_window = st.number_input("Long EMA Window Size", min_value=1, value=26, step=1)
        signal_window = st.number_input("Signal Line Window Size", min_value=1, value=9, step=1)

        # Calculate MACD
        short_ema = st.session_state.df_label['Close'].ewm(span=short_window, adjust=False).mean()
        long_ema = st.session_state.df_label['Close'].ewm(span=long_window, adjust=False).mean()
        macd = short_ema - long_ema
        signal_line = macd.ewm(span=signal_window, adjust=False).mean()
        macd_histogram = macd - signal_line

        st.session_state.df_features['MACD Histogram'] = macd_histogram
        st.session_state.df_features['MACD'] = macd
        st.session_state.df_features['Signal Line'] = signal_line

        st.write("MACD and Signal Line added to the dataframe.")
        fig = plt.figure(figsize=(10, 10))

        # Panel 1: Close Price (2/3 of the figure height)
        ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
        ax1.plot(st.session_state.df_features['Close'], label='Close Price', color='blue')
        ax1.set_title(f'{ticker} Closing Price')
        ax1.set_ylabel('Price')
        ax1.legend()

        # Panel 2: MACD and Signal Line (1/3 of the figure height)
        ax2 = plt.subplot2grid((3, 1), (2, 0), rowspan=1)
        ax2.plot(st.session_state.df_features['MACD'], label='MACD', color='blue')
        ax2.plot(st.session_state.df_features['Signal Line'], label='Signal Line', color='red')
        ax2.bar(st.session_state.df_features.index, st.session_state.df_features['MACD Histogram'], label='MACD Histogram', color='black', alpha=0.9)
        ax2.set_title('MACD and Signal Line')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Value')
        ax2.legend()

        plt.tight_layout()
        st.pyplot(fig)

    st.write("### Momentum Indicators")

    # RSI
    rsi_option = st.checkbox("Relative Strength Index (RSI)", value=False)
    if rsi_option:
        window = st.number_input("RSI Window Size", min_value=1, value=14, step=1)
        delta = st.session_state.df_features['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        st.session_state.df_features['RSI'] = 100 - (100 / (1 + rs))
        st.write("RSI added to the dataframe.")
        fig = plt.figure(figsize=(10, 8))

        # Panel 1: Close Price (2/3 of the figure height)
        ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
        ax1.plot(st.session_state.df_features['Close'], label='Close Price', color='blue')
        ax1.set_title(f'{ticker} Closing Price')
        ax1.set_ylabel('Price')
        ax1.legend()

        # Panel 2: RSI (1/3 of the figure height)
        ax2 = plt.subplot2grid((3, 1), (2, 0), rowspan=1)
        ax2.plot(st.session_state.df_features['RSI'], label='RSI', color='black')
        ax2.axhline(70, color='red', linestyle='--', label='Overbought (70)')
        ax2.axhline(30, color='green', linestyle='--', label='Oversold (30)')
        ax2.set_title('Relative Strength Index (RSI)')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('RSI')
        ax2.legend()

        plt.tight_layout()
        st.pyplot(fig)
    
    stochastic_option = st.checkbox("Stochastic Oscillator", value=False)
    if stochastic_option:
        window = st.number_input("Stochastic Window Size", min_value=1, value=14, step=1)
        k_window = st.number_input("K Window Size", min_value=1, value=3, step=1)
        d_window = st.number_input("D Window Size", min_value=1, value=3, step=1)

        low_min = st.session_state.df_features['Low'].rolling(window=window).min()
        high_max = st.session_state.df_features['High'].rolling(window=window).max()
        stoch_k = 100 * ((st.session_state.df_features['Close'] - low_min) / (high_max - low_min))
        stoch_d = stoch_k.rolling(window=d_window).mean()

        st.session_state.df_features['Stochastic K'] = stoch_k
        st.session_state.df_features['Stochastic D'] = stoch_d
        st.write("Stochastic Oscillator added to the dataframe.")
        fig = plt.figure(figsize=(10, 8))

        # Top panel: Close Price (2/3 of the figure height)
        ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
        ax1.plot(st.session_state.df_features['Close'], label='Close Price', color='blue')
        ax1.set_title('Close Price')
        ax1.set_ylabel('Price')
        ax1.legend()

        # Bottom panel: Stochastic Oscillator (1/3 of the figure height)
        ax2 = plt.subplot2grid((3, 1), (2, 0), rowspan=1)
        ax2.plot(st.session_state.df_features['Stochastic K'], label='Stochastic K', color='black')
        ax2.plot(st.session_state.df_features['Stochastic D'], label='Stochastic D', color='red')
        ax2.axhline(80, color='purple', linestyle='--', label='Overbought (80)')
        ax2.axhline(20, color='green', linestyle='--', label='Oversold (20)')
        ax2.set_title('Stochastic Oscillator')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Value')
        ax2.legend()

        plt.tight_layout()
        st.pyplot(fig)

    st.write("### Volatility Indicators")

    bollinger_bands_option = st.checkbox("Bollinger Bands", value=False)
    if bollinger_bands_option:
        window = st.number_input("Bollinger Bands Window Size", min_value=1, value=20, step=1)
        num_std = st.number_input("Number of Standard Deviations", min_value=0.0, value=2.0, step=0.1)
        rolling_mean = st.session_state.df['Close'].rolling(window=window).mean()
        rolling_std = st.session_state.df['Close'].rolling(window=window).std()
        upper_band = rolling_mean + (rolling_std * num_std)
        lower_band = rolling_mean - (rolling_std * num_std)
        st.session_state.df_features['Upper Band'] = upper_band
        st.session_state.df_features['Lower Band'] = lower_band
        st.write("Bollinger Bands added to the dataframe.")
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(st.session_state.df['Close'], label='Close Price', color='blue')
        ax.plot(upper_band, label='Upper Band', color='red')
        ax.plot(lower_band, label='Lower Band', color='green')
        ax.set_title('Bollinger Bands')
        ax.set_xlabel('Date')
        ax.set_ylabel('Price')
        ax.legend()
        st.pyplot(fig)

    atr_option = st.checkbox("Average True Range (ATR)", value=False)
    if atr_option:
        window = st.number_input("ATR Window Size", min_value=1, value=14, step=1)
        high_low = st.session_state.df['High'] - st.session_state.df['Low']
        high_close = np.abs(st.session_state.df['High'] - st.session_state.df['Close'].shift())
        low_close = np.abs(st.session_state.df['Low'] - st.session_state.df['Close'].shift())
        tr = pd.DataFrame({'high_low': high_low, 'high_close': high_close, 'low_close': low_close}).max(axis=1)
        atr = tr.rolling(window=window).mean()
        st.session_state.df_features['ATR'] = atr
        st.write("Average True Range (ATR) added to the dataframe.")
        fig, ax = plt.subplots(figsize=(10, 10))

        # Panel 1: Close Price (2/3 of the figure height)
        ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
        ax1.plot(st.session_state.df['Close'], label='Close Price', color='blue')
        ax1.set_title('Close Price')
        ax1.set_ylabel('Price')
        ax1.legend()

        # Panel 2: ATR (1/3 of the figure height)
        ax = plt.subplot2grid((3, 1), (2, 0), rowspan=1)    
        ax.plot(st.session_state.df_features['ATR'], label='ATR', color='red')
        ax.set_title('Average True Range (ATR)')
        ax.set_xlabel('Date')
        ax.set_ylabel('ATR')
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig)

    st.write("### Volume Indicators")

    obv_option = st.checkbox("On-Balance Volume (OBV)", value=False)
    if obv_option:
        obv = (np.sign(st.session_state.df['Close'].diff()) * st.session_state.df['Volume']).fillna(0).cumsum()
        st.session_state.df_features['OBV'] = obv
        st.write("On-Balance Volume (OBV) added to the dataframe.")
        fig = plt.figure(figsize=(10, 10))

        # Panel 1: Close Price (2/3 of the figure height)
        ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
        ax1.plot(st.session_state.df['Close'], label='Close Price', color='blue')
        ax1.set_title('Close Price')
        ax1.set_ylabel('Price')
        ax1.legend()

        # Panel 2: OBV (1/3 of the figure height)
        ax2 = plt.subplot2grid((3, 1), (2, 0), rowspan=1)
        ax2.plot(st.session_state.df_features['OBV'], label='OBV', color='red')
        ax2.set_title('On-Balance Volume (OBV)')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('OBV')
        ax2.legend()

        plt.tight_layout()
        st.pyplot(fig)
    st.divider()

################################################################################
# SELECTED FEATURES FOR ML
# This section allows you to select the features to be used for machine learning.
################################################################################

if not st.session_state.df_features.empty:
    st.header("Dataframe with Features")
    st.write("This is the dataframe with the features added.")
    
    st.subheader("Select Columns for Next Process")
    available_columns = st.session_state.df_features.columns.tolist()
    selected_columns = st.multiselect("Select columns to use:", available_columns, default=available_columns, accept_new_options=False)

    if 'Label' not in selected_columns:
        st.error("The 'Label' column must be selected for the next process.")
        selected_columns.append('Label')
    
    if selected_columns:
        st.session_state.df_selected_features = st.session_state.df_features[selected_columns].copy().dropna()
        st.write("Selected columns added to the dataframe for the next process.")
    else:
        st.session_state.df_selected_features = pd.DataFrame()
        st.error("No columns selected. Please select at least one column.")

    st.session_state.df_selected_features
    columns = st.session_state.df_selected_features.columns.tolist()
    if 'Label' in columns:
        columns.remove('Label')
    
    # display correlation matrix for selected features
    st.subheader("Correlation Matrix for Selected Features")
    correlation_matrix = st.session_state.df_selected_features[columns].corr()
    plt.figure(figsize=(10, 5))
    plt.imshow(correlation_matrix, cmap='coolwarm', interpolation='none')
    plt.colorbar()
    plt.title('Correlation Matrix')
    plt.xticks(ticks=np.arange(len(correlation_matrix.columns)), labels=correlation_matrix.columns, rotation=90)
    plt.yticks(ticks=np.arange(len(correlation_matrix.index)), labels=correlation_matrix.index)

    # Adding annotations
    for i in range(len(correlation_matrix)):
        for j in range(len(correlation_matrix.columns)):
            plt.text(j, i, f'{correlation_matrix.iloc[i, j]:.2f}', 
                    ha='center', va='center', color='black')

    st.pyplot(plt)

    st.divider()

#################################################################################
# MACHINE LEARNING
# This section allows you to select the algorithm and parameters 
# for model training and testing.
#################################################################################

if not st.session_state.df_selected_features.empty:
    st.header("Machine Learning Model")
    st.write("This section allows you to select the algorithm and parameters for model training and testing.")

    st.subheader("Select Algorithm and Parameters")
    # Select algorithm
    algorithm = st.selectbox("Select Algorithm", ["K-Nearest Neighbors", "Decision Tree", "Random Forest", "XGBoost", "LightGBM", "Support Vector Machine", "Multi-Layer Perceptron"], index=0, accept_new_options=False)
    

    if algorithm == "K-Nearest Neighbors":
        st.write("K-Nearest Neighbors selected.")
        col1, col2 = st.columns(2, vertical_alignment="bottom")
        with col1:
            n_neighbors = st.number_input("Number of Neighbors", min_value=1, value=3, step=1)
        with col2:
            weights = st.selectbox("Weights", ["uniform", "distance"], index=0, accept_new_options=False)

    # Parameters for Decision Tree
    if algorithm == "Decision Tree":
        st.write("Decision Tree selected.") 
        col1, col2, col3 = st.columns(3, vertical_alignment="bottom")
        with col1:
            max_depth = st.number_input("Max Depth", min_value=1, value=10, step=1)
        with col2:
            min_samples_split = st.number_input("Min Samples Split", min_value=2, value=2, step=1)
        with col3:
            min_samples_leaf = st.number_input("Min Samples Leaf", min_value=1, value=1, step=1)

    # Parameters for Random Forest
    if algorithm == "Random Forest":
        col1, col2, col3, col4 = st.columns(4, vertical_alignment="bottom")
        with col1:
            n_estimators = st.number_input("Number of Trees", min_value=1, value=100, step=1)
        with col2:
            max_depth = st.number_input("Max Depth", min_value=1, value=10, step=1)
        with col3:
            min_samples_split = st.number_input("Min Samples Split", min_value=2, value=2, step=1)
        with col4:
            min_samples_leaf = st.number_input("Min Samples Leaf", min_value=1, value=1, step=1)

    # Parameters for XGBoost
    elif algorithm == "XGBoost":
        col1, col2, col3 = st.columns(3, vertical_alignment="bottom")
        with col1:
            n_estimators = st.number_input("Number of Trees", min_value=1, value=100, step=1)
        with col2:
            max_depth = st.number_input("Max Depth", min_value=1, value=10, step=1)
        with col3:
            learning_rate = st.number_input("Learning Rate", min_value=0.01, value=0.3, step=0.01)

    # Parameters for LightGBM
    elif algorithm == "LightGBM":
        col1, col2, col3 = st.columns(3, vertical_alignment="bottom")
        with col1:
            n_estimators = st.number_input("Number of Trees", min_value=1, value=100, step=1)
        with col2:
            max_depth = st.number_input("Max Depth", min_value=-1, value=-1)  # -1 means no limit
        with col3:
            learning_rate = st.number_input("Learning Rate", min_value=0.01, value=0.3, step=0.01)

    if algorithm == "Support Vector Machine":
        st.write("Support Vector Machine selected.")
        col1, col2 = st.columns(2, vertical_alignment="bottom")
        with col1:
            kernel = st.selectbox("Kernel", ["linear", "poly", "rbf", "sigmoid"], index=0, accept_new_options=False)
        with col2:
            C = st.number_input("Regularization Parameter (C)", min_value=0.01, value=1.0, step=0.01)

    if algorithm == "Multi-Layer Perceptron":
        st.write("Multi-Layer Perceptron selected.")
        col1, col2 = st.columns(2, vertical_alignment="bottom")
        with col1:
            hidden_layer_sizes = st.number_input("Hidden Layer Sizes", min_value=1, value=100, step=1)
        with col2:
            activation = st.selectbox("Activation Function", ["identity", "logistic", "tanh", "relu"], index=0, accept_new_options=False)

    st.subheader("Train-Test Split")
    # train-test split
    col1, col2 = st.columns(2, vertical_alignment="bottom")
    with col1:
        test_size = st.number_input("Test Size (%)", min_value=1, max_value=100, value=20, step=1) / 100
    with col2:
        random_state = st.number_input("Random State", min_value=0, value=42, step=1)

    st.subheader("Model Training and Testing")
    if st.button(f"Train and test Model with {algorithm} algorithm"):
        with st.spinner("Training and testing model..."):
            # K-Nearest Neighbors
            if algorithm == "K-Nearest Neighbors":
                st.subheader("K-Nearest Neighbors Model")
                X = st.session_state.df_selected_features.drop(columns=['Label'])
                y = st.session_state.df_selected_features['Label']

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

                # Standardize the features
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)

                # Train and predict using K-Nearest Neighbors
                knn = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights)
                knn.fit(X_train, y_train)
                y_pred = knn.predict(X_test)

                st.write("### Classification Report")
                
                model_params = {
                    "n_neighbors": n_neighbors,
                    "weights": weights
                }
                st.write("#### K-Nearest Neighbors Model Parameters")
                st.table(pd.DataFrame(model_params, index=["Value"]))

                report = classification_report(y_test, y_pred, output_dict=True)
                report_df = pd.DataFrame(report).transpose()
                st.dataframe(report_df)

                st.write("### Confusion Matrix")
                cm = confusion_matrix(y_test, y_pred, labels=["Buy", "Hold", "Sell"])
                fig, ax = plt.subplots(figsize=(5, 5))
                cax = ax.matshow(cm, cmap='coolwarm', alpha=0.7)
                fig.colorbar(cax)
                for (i, j), z in np.ndenumerate(cm):
                    ax.text(j, i, f'{z}', ha='center', va='center')
                ax.set_xticks(range(len(["Buy", "Hold", "Sell"])))
                ax.set_yticks(range(len(["Buy", "Hold", "Sell"])))
                ax.set_xticklabels(["Buy", "Hold", "Sell"])
                ax.set_yticklabels(["Buy", "Hold", "Sell"])
                plt.xlabel('Predicted')
                plt.ylabel('True')
                plt.title('Confusion Matrix')
                st.pyplot(fig)
                st.write("### Accuracy Score")
                st.text(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
                st.divider()

            elif algorithm == "Decision Tree":
                st.subheader("Decision Tree Model")
                X = st.session_state.df_selected_features.drop(columns=['Label'])
                y = st.session_state.df_selected_features['Label']

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

                # Train and predict using Decision Tree
                dt = DecisionTreeClassifier(max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, random_state=random_state)
                dt.fit(X_train, y_train)
                y_pred = dt.predict(X_test)

                st.write("### Classification Report")
                model_params = {
                    "max_depth": max_depth,
                    "min_samples_split": min_samples_split,
                    "min_samples_leaf": min_samples_leaf
                }
                st.write("#### Decision Tree Model Parameters")
                st.table(pd.DataFrame(model_params, index=["Value"]))

                report = classification_report(y_test, y_pred, output_dict=True)
                report_df = pd.DataFrame(report).transpose()
                st.dataframe(report_df)

                st.write("### Confusion Matrix")
                cm = confusion_matrix(y_test, y_pred, labels=["Buy", "Hold", "Sell"])
                fig, ax = plt.subplots(figsize=(5, 5))
                cax = ax.matshow(cm, cmap='coolwarm', alpha=0.7)
                fig.colorbar(cax)
                for (i, j), z in np.ndenumerate(cm):
                    ax.text(j, i, f'{z}', ha='center', va='center')
                ax.set_xticks(range(len(["Buy", "Hold", "Sell"])))
                ax.set_yticks(range(len(["Buy", "Hold", "Sell"])))
                ax.set_xticklabels(["Buy", "Hold", "Sell"])
                ax.set_yticklabels(["Buy", "Hold", "Sell"])
                plt.xlabel('Predicted')
                plt.ylabel('True')
                plt.title('Confusion Matrix')
                st.pyplot(fig)
                st.write("### Accuracy Score")
                st.text(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
                st.divider()
            
            elif algorithm == "Random Forest":
                st.subheader("Random Forest Model")
                X = st.session_state.df_selected_features.drop(columns=['Label'])
                y = st.session_state.df_selected_features['Label']

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

                # Train and predict using Random Forest
                rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, random_state=random_state)
                rf.fit(X_train, y_train)
                y_pred = rf.predict(X_test)

                st.write("### Classification Report")

                model_params = {
                    "n_estimators": n_estimators,
                    "max_depth": max_depth,
                    "min_samples_split": min_samples_split,
                    "min_samples_leaf": min_samples_leaf
                }
                st.write("#### Random Forest Model Parameters")
                st.table(pd.DataFrame(model_params, index=["Value"]))

                report = classification_report(y_test, y_pred, output_dict=True)
                report_df = pd.DataFrame(report).transpose()
                st.dataframe(report_df)

                st.write("### Confusion Matrix")
                cm = confusion_matrix(y_test, y_pred, labels=["Buy", "Hold", "Sell"])
                fig, ax = plt.subplots(figsize=(5, 5))
                cax = ax.matshow(cm, cmap='coolwarm', alpha=0.7)
                fig.colorbar(cax)
                for (i, j), z in np.ndenumerate(cm):
                    ax.text(j, i, f'{z}', ha='center', va='center')
                ax.set_xticks(range(len(["Buy", "Hold", "Sell"])))
                ax.set_yticks(range(len(["Buy", "Hold", "Sell"])))
                ax.set_xticklabels(["Buy", "Hold", "Sell"])
                ax.set_yticklabels(["Buy", "Hold", "Sell"])
                plt.xlabel('Predicted')
                plt.ylabel('True')
                plt.title('Confusion Matrix')
                st.pyplot(fig)
                st.write("### Accuracy Score")
                st.text(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
                st.divider()
            
            elif algorithm == "XGBoost":
                st.subheader("XGBoost Model")
                X = st.session_state.df_selected_features.drop(columns=['Label'])
                label_mapping = {"Buy": 1, "Hold": 0, "Sell": 2}
                y = st.session_state.df_selected_features['Label'].map(label_mapping)

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

                # Train and predict using XGBoost
                xgb = XGBClassifier(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate, random_state=random_state)
                xgb.fit(X_train, y_train)
                y_pred = xgb.predict(X_test)

                st.write("### Classification Report")

                model_params = {
                    "n_estimators": n_estimators,
                    "max_depth": max_depth,
                    "learning_rate": learning_rate
                }
                st.write("#### XGBoost Model Parameters")
                st.table(pd.DataFrame(model_params, index=["Value"]))

                report = classification_report(y_test, y_pred, labels=[1, 0, 2], target_names=["Buy", "Hold", "Sell"], output_dict=True)
                report_df = pd.DataFrame(report).transpose()
                st.dataframe(report_df)

                st.write("### Confusion Matrix")
                cm = confusion_matrix(y_test.map({v: k for k, v in label_mapping.items()}), 
                                      pd.Series(y_pred).map({v: k for k, v in label_mapping.items()}), 
                                      labels=["Buy", "Hold", "Sell"])
                fig, ax = plt.subplots(figsize=(5, 5))
                cax = ax.matshow(cm, cmap='coolwarm', alpha=0.7)
                fig.colorbar(cax)
                for (i, j), z in np.ndenumerate(cm):
                    ax.text(j, i, f'{z}', ha='center', va='center')
                ax.set_xticks(range(len(["Buy", "Hold", "Sell"])))
                ax.set_yticks(range(len(["Buy", "Hold", "Sell"])))
                ax.set_xticklabels(["Buy", "Hold", "Sell"])
                ax.set_yticklabels(["Buy", "Hold", "Sell"])
                plt.xlabel('Predicted')
                plt.ylabel('True')
                plt.title('Confusion Matrix')
                st.pyplot(fig)
                st.write("### Accuracy Score")
                st.text(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
                st.divider()
            
            elif algorithm == "LightGBM":
                st.subheader("LightGBM Model")
                X = st.session_state.df_selected_features.drop(columns=['Label'])
                y = st.session_state.df_selected_features['Label']

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

                # Train and predict using LightGBM
                lgbm = LGBMClassifier(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate, random_state=random_state)
                lgbm.fit(X_train, y_train)
                y_pred = lgbm.predict(X_test)

                st.write("### Classification Report")
                model_params = {
                    "n_estimators": n_estimators,
                    "max_depth": max_depth,
                    "learning_rate": learning_rate
                }
                st.write("#### LightGBM Model Parameters")
                st.table(pd.DataFrame(model_params, index=["Value"]))

                report = classification_report(y_test, y_pred, output_dict=True)
                report_df = pd.DataFrame(report).transpose()
                st.dataframe(report_df)

                st.write("### Confusion Matrix")
                cm = confusion_matrix(y_test, y_pred, labels=["Buy", "Hold", "Sell"])
                fig, ax = plt.subplots(figsize=(5, 5))
                cax = ax.matshow(cm, cmap='coolwarm', alpha=0.7)
                fig.colorbar(cax)
                for (i, j), z in np.ndenumerate(cm):
                    ax.text(j, i, f'{z}', ha='center', va='center')
                ax.set_xticks(range(len(["Buy", "Hold", "Sell"])))
                ax.set_yticks(range(len(["Buy", "Hold", "Sell"])))
                ax.set_xticklabels(["Buy", "Hold", "Sell"])
                ax.set_yticklabels(["Buy", "Hold", "Sell"])
                plt.xlabel('Predicted')
                plt.ylabel('True')
                plt.title('Confusion Matrix')
                st.pyplot(fig)
                st.write("### Accuracy Score")
                st.text(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
                st.divider()
            
            elif algorithm == "Support Vector Machine":
                st.subheader("Support Vector Machine Model")
                X = st.session_state.df_selected_features.drop(columns=['Label'])
                y = st.session_state.df_selected_features['Label']

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

                # Standardize the features
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)

                # Train and predict using Support Vector Machine
                
                svm = SVC(kernel=kernel, C=C, random_state=random_state)
                svm.fit(X_train, y_train)
                y_pred = svm.predict(X_test)

                st.write("### Classification Report")
                model_params = {
                    "kernel": kernel,
                    "C": C
                }
                st.write("#### Support Vector Machine Model Parameters")
                st.table(pd.DataFrame(model_params, index=["Value"]))

                report = classification_report(y_test, y_pred, output_dict=True)
                report_df = pd.DataFrame(report).transpose()
                st.dataframe(report_df)

                st.write("### Confusion Matrix")
                cm = confusion_matrix(y_test, y_pred, labels=["Buy", "Hold", "Sell"])
                fig, ax = plt.subplots(figsize=(5, 5))
                cax = ax.matshow(cm, cmap='coolwarm', alpha=0.7)
                fig.colorbar(cax)
                for (i, j), z in np.ndenumerate(cm):
                    ax.text(j, i, f'{z}', ha='center', va='center')
                ax.set_xticks(range(len(["Buy", "Hold", "Sell"])))
                ax.set_yticks(range(len(["Buy", "Hold", "Sell"])))
                ax.set_xticklabels(["Buy", "Hold", "Sell"])
                ax.set_yticklabels(["Buy", "Hold", "Sell"])
                plt.xlabel('Predicted')
                plt.ylabel('True')
                plt.title('Confusion Matrix')
                st.pyplot(fig)
                st.write("### Accuracy Score")
                st.text(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
                st.divider()
            
            elif algorithm == "Multi-Layer Perceptron":
                st.subheader("Multi-Layer Perceptron Model")
                X = st.session_state.df_selected_features.drop(columns=['Label'])
                y = st.session_state.df_selected_features['Label']

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

                # Standardize the features
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)

                # Train and predict using Multi-Layer Perceptron
                mlp = MLPClassifier(hidden_layer_sizes=(hidden_layer_sizes,), activation=activation, random_state=random_state)
                mlp.fit(X_train, y_train)
                y_pred = mlp.predict(X_test)

                st.write("### Classification Report")
                
                model_params = {
                    "hidden_layer_sizes": hidden_layer_sizes,
                    "activation": activation
                }
                st.write("#### Multi-Layer Perceptron Model Parameters")
                st.table(pd.DataFrame(model_params, index=["Value"]))

                report = classification_report(y_test, y_pred, output_dict=True)
                report_df = pd.DataFrame(report).transpose()
                st.dataframe(report_df)

                st.write("### Confusion Matrix")
                cm = confusion_matrix(y_test, y_pred, labels=["Buy", "Hold", "Sell"])
                fig, ax = plt.subplots(figsize=(5, 5))
                cax = ax.matshow(cm, cmap='coolwarm', alpha=0.7)
                fig.colorbar(cax)
                for (i, j), z in np.ndenumerate(cm):
                    ax.text(j, i, f'{z}', ha='center', va='center')
                ax.set_xticks(range(len(["Buy", "Hold", "Sell"])))
                ax.set_yticks(range(len(["Buy", "Hold", "Sell"])))
                ax.set_xticklabels(["Buy", "Hold", "Sell"])
                ax.set_yticklabels(["Buy", "Hold", "Sell"])
                plt.xlabel('Predicted')
                plt.ylabel('True')
                plt.title('Confusion Matrix')
                st.pyplot(fig)
                st.write("### Accuracy Score")
                st.text(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
                st.divider()
