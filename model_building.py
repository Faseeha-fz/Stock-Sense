import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import tensorflow as tf

# Function to preprocess data and prepare for model training
def preprocess_data(df):
    # Handle the case when 'Date' is not in index
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
   
    # Normalize the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df[['Close']])
   
    return scaled_data, scaler

# Function to create and train the model
def create_model(data):
    # Check if 'Stock' column exists and group by it
    if 'Stock' in data.columns:
        # Use the first stock for prediction (assuming we're using one stock at a time)
        stock_name = data['Stock'].unique()[0]
        filtered_data = data[data['Stock'] == stock_name].copy()
    else:
        filtered_data = data.copy()
    
    # Make sure Date is properly handled
    if 'Date' in filtered_data.columns:
        filtered_data['Date'] = pd.to_datetime(filtered_data['Date'])
        filtered_data_with_date = filtered_data.copy()
        filtered_data.set_index('Date', inplace=True)
    else:
        filtered_data_with_date = filtered_data.copy()
        filtered_data_with_date['Date'] = filtered_data.index
    
    # Preprocess the data
    scaled_data, scaler = preprocess_data(filtered_data)
   
    # Create training and testing data
    train_data, test_data = train_test_split(scaled_data, test_size=0.2, shuffle=False)
   
    # Prepare the data for model training (lookback for time series)
    look_back = 60
    X_train, y_train = [], []
    for i in range(look_back, len(train_data)):
        X_train.append(train_data[i-look_back:i, 0])
        y_train.append(train_data[i, 0])
   
    X_train, y_train = np.array(X_train), np.array(y_train)
   
    # Reshape the data for the LSTM model
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
   
    # Create LSTM model
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
        tf.keras.layers.LSTM(units=50, return_sequences=False),
        tf.keras.layers.Dense(units=1)
    ])
   
    # Compile and fit the model
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=0)
   
    # Prepare test data
    X_test, y_test = [], []
    for i in range(look_back, len(test_data)):
        X_test.append(test_data[i-look_back:i, 0])
        y_test.append(test_data[i, 0])
    
    X_test, y_test = np.array(X_test), np.array(y_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    
    # Predicting on test data
    test_predictions = model.predict(X_test)
    
    # Convert the predictions back to original scale
    test_predictions = scaler.inverse_transform(test_predictions).flatten()  # Flatten to 1D
    y_test = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()  # Flatten to 1D
    
    # Get dates for the test predictions
    if len(filtered_data_with_date) > len(test_predictions):
        # Use the last portion of dates that matches the length of predictions
        test_dates = filtered_data_with_date['Date'].values[-len(test_predictions):]
    else:
        # In case we don't have enough dates, create some
        last_date = filtered_data_with_date['Date'].iloc[-1]
        if isinstance(last_date, str):
            last_date = pd.to_datetime(last_date)
        test_dates = pd.date_range(start=last_date, periods=len(test_predictions))
   
    # Create DataFrame for plotting
    plotdf = pd.DataFrame({
        'Date': test_dates,
        'test_predicted_close': test_predictions
    })
   
    # Future prediction (next 30 days)
    future_input = scaled_data[-look_back:]
    future_input = np.reshape(future_input, (1, look_back, 1))
    
    future_predictions = []
    for _ in range(30):
        prediction = model.predict(future_input)
        future_predictions.append(prediction[0, 0])
        # Update input for next prediction
        future_input = np.append(future_input[:, 1:, :], [[[prediction[0, 0]]]], axis=1)
    
    future_predictions = np.array(future_predictions).reshape(-1, 1)
    future_predictions = scaler.inverse_transform(future_predictions)
   
    return plotdf, future_predictions

# Function to plot results (for standalone testing)
def plot_results(df):
    # Create model and get predicted data
    plotdf, future_predicted_values = create_model(df)
   
    # Plotting the results
    plt.figure(figsize=(10,6))
    plt.plot(plotdf['Date'], plotdf['test_predicted_close'], label='Predicted Close Price')
    plt.legend()
    plt.show()
   
    # Future prediction visualization
    future_dates = pd.date_range(start=plotdf['Date'].iloc[-1], periods=30 + 1)[1:]
    plt.figure(figsize=(10,6))
    plt.plot(future_dates, future_predicted_values, label='Future Predicted Close Price')
    plt.legend()
    plt.show()

# If running as main script, run the plot_results function
if __name__ == "__main__":
    # Example data (replace with actual stock data)
    data = pd.read_csv('your_stock_data.csv')  # Assuming you have your stock data in CSV
    plot_results(data)