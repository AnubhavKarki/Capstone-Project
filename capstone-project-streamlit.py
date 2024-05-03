import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import xgboost as xgb
from sklearn.model_selection import train_test_split
import joblib
import matplotlib.dates as mdates

# Setting the title
st.title("Netflix Stock Price Prediction")

# Disabling warning:
st.set_option('deprecation.showPyplotGlobalUse', False)

# Load the dataset
df = pd.read_csv('NFLX.csv')

# Assuming X and y represent your features and target variable, respectively
X = df.drop(columns=['Close', 'Date', 'Low', 'Adj Close', 'Volume'])  # Assuming 'Close' is the target variable
y = df['Close']

# Define XGBoost Regressor model
model = xgb.XGBRegressor()

# Train the model on 100% of the available data
model.fit(X, y)

# Save the trained model as a serialized file (Pickling)
joblib.dump(model, 'xgboost_model.pkl')

# Load the trained model (Unpickling)
model = joblib.load('xgboost_model.pkl')

# Streamlit app
def main():
    st.title("Stock Price Prediction")

    # Input form
    st.sidebar.subheader("Enter the required parameters:")
    # Example input fields
    open_value = st.sidebar.slider('Open', min_value=df['Open'].min(), max_value=df['Open'].max(), value=df['Open'].mean())
    high_value = st.sidebar.slider('High', min_value=df['High'].min(), max_value=df['High'].max(), value=df['High'].mean())

    # Predict button
    if st.sidebar.button('Predict'):
        # Make prediction
        prediction = predict_stock_price(open_value, high_value)
        
        # Display prediction
        st.subheader("Predicted Stock Closing Price:")
        st.write(prediction)
        
        # Plot the predicted stock prices

        plot_stock_trend(prediction, df)

# Function to make predictions
def predict_stock_price(open_value, high_value):
    # Make predictions
    prediction = model.predict(np.array([[open_value, high_value]]))
    return prediction

# Function to plot the predicted stock prices

# Function to plot the trend of the last 300 samples
def plot_stock_trend(prediction, data):
    # Convert date column to datetime
    data = data.iloc[-300:]
    data['Date'] = pd.to_datetime(data['Date'])
    
    # Extract dates and actual close prices
    dates = data['Date']
    actual_prices = data['Close']
    
    # Create figure and axis objects
    fig, ax = plt.subplots()
    
    # Plot actual stock prices
    ax.plot(dates, actual_prices, label='Actual Close Price', marker='o')
    
    # Plot predicted stock price
    ax.axhline(y=prediction, color='r', linestyle='--', label='Predicted Close Price')
    
    # Set labels and title
    ax.set_xlabel('Date')
    ax.set_ylabel('Close Price')
    ax.set_title('Actual vs Predicted Stock Prices (Last 500 Samples)')
    
    # Set x-axis tick frequency to monthly intervals
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    plt.xticks(rotation=45)
    
    # Add legend
    ax.legend()
    
    # Show plot
    st.pyplot(fig)

main()