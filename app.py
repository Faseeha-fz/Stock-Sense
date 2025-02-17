# app.py
import streamlit as st
import pandas as pd
import yfinance as yf
import datetime
import model_building as m
import matplotlib.pyplot as plt
import numpy as np

# Custom CSS for styling
st.markdown("""
<style>
body { background-color: #f2f5ee; color: #333; font-family: serif; }
.sidebar .sidebar-content { background-color: #c7e7f1; }
h1, h2, h3, h4 { color: #1f71c3; }
.stButton>button { background-color: #1f71c3; color: white; }
.stTextInput>div>input { background-color: #ecf0f1; color: #2c3e50; }
</style>
""", unsafe_allow_html=True)

# Sidebar for user input
st.sidebar.markdown("# STOCK SENSE")
available_stocks = ['RELIANCE.NS', 'TCS.NS', 'INFY.NS', 'HDFCBANK.NS', 'ICICIBANK.NS', 'SBIN.NS', 'WIPRO.NS']
user_input = st.sidebar.multiselect('Please select the stock', available_stocks, ['RELIANCE.NS'])
st.sidebar.markdown("### Choose Date for your analysis")
START = st.sidebar.date_input("From", datetime.date(2015, 1, 1))
END = st.sidebar.date_input("To", datetime.date(2024, 2, 29))

bt = st.sidebar.button('Submit')
investment_amount = st.sidebar.number_input("Enter the amount invested", min_value=0.0, value=1000.0, step=100.0)
investment_date = st.sidebar.date_input("Investment Date", datetime.date(2020, 1, 1))
withdrawal_date = st.sidebar.date_input("Withdrawal Date", datetime.date(2024, 2, 29))

if bt:
    all_data = []
    for stock in user_input:
        df = yf.download(stock, start=START, end=END)
        df.reset_index(inplace=True)
        df['Stock'] = stock
        all_data.append(df)

    combined_df = pd.concat(all_data, ignore_index=True)
    

    # Create 'reliance' DataFrame if 'combined_df' is non-empty
    if not combined_df.empty:
        if "Adj Close" in combined_df.columns:
            reliance = combined_df.drop(["Adj Close"], axis=1).dropna().reset_index(drop=True)
        else:
            reliance = combined_df.copy().dropna().reset_index(drop=True)
        reliance['Date'] = pd.to_datetime(reliance['Date'], format='%Y-%m-%d')
        reliance = reliance.set_index('Date')

        plotdf, future_predicted_values = m.create_model(combined_df)

        st.title('Stock Market Prediction')
        st.header("Data We collected from the source")
        st.write(combined_df)

        # Exploratory Data Analysis (EDA)
        st.title('Exploratory Data Analysis (EDA)')
        st.write(reliance)

        # Visualizations
        st.title('Visualizations')
        st.header("Graphs")

        plt.figure(figsize=(20, 10))

        # Plot Open Prices
        plt.subplot(2, 2, 1)
        plt.plot(reliance['Open'], color='green')
        plt.xlabel('Date')
        plt.ylabel('Open Price')
        plt.title('Open')

        # Plot Close Prices
        plt.subplot(2, 2, 2)
        plt.plot(reliance['Close'], color='red')
        plt.xlabel('Date')
        plt.ylabel('Close Price')
        plt.title('Close')

        # Plot High Prices
        plt.subplot(2, 2, 3)
        plt.plot(reliance['High'], color='green')
        plt.xlabel('Date')
        plt.ylabel('High Price')
        plt.title('High')

        # Plot Low Prices
        plt.subplot(2, 2, 4)
        plt.plot(reliance['Low'], color='red')
        plt.xlabel('Date')
        plt.ylabel('Low Price')
        plt.title('Low')

        st.pyplot(plt.gcf()) 

        # Box Plots
        st.header("Box Plots")
        
        plt.figure(figsize=(20, 10))

        plt.subplot(2, 2, 1)
        plt.boxplot(reliance['Open'])
        plt.xlabel('Open Price')
        plt.title('Open')

        plt.subplot(2, 2, 2)
        plt.boxplot(reliance['Close'])
        plt.xlabel('Close Price')
        plt.title('Close')

        plt.subplot(2, 2, 3)
        plt.boxplot(reliance['High'])
        plt.xlabel('High Price')
        plt.title('High')

        plt.subplot(2, 2, 4)
        plt.boxplot(reliance['Low'])
        plt.xlabel('Low Price')
        plt.title('Low')

        st.pyplot(plt.gcf()) 

        # Histograms
        st.header("Histogram")
        
        plt.figure(figsize=(20, 10))

        plt.subplot(2, 2, 1)
        plt.hist(reliance['Open'], bins=50, color='green')
        plt.xlabel("Open Price")
        plt.ylabel("Frequency")
        plt.title('Open')

        plt.subplot(2, 2, 2)
        plt.hist(reliance['Close'], bins=50, color='red')
        plt.xlabel("Close Price")
        plt.ylabel("Frequency")
        plt.title('Close')

        plt.subplot(2, 2, 3)
        plt.hist(reliance['High'], bins=50, color='green')
        plt.xlabel("High Price")
        plt.ylabel("Frequency")
        plt.title('High')

        plt.subplot(2, 2, 4)
        plt.hist(reliance['Low'], bins=50, color='red')
        plt.xlabel("Low Price")
        plt.ylabel("Frequency")
        plt.title('Low')

        st.pyplot(plt.gcf()) 

        # Long-term and short-term trends
        st.title('Finding long-term and short-term trends')

        reliance_ma = reliance.copy()
        reliance_ma['30-day MA'] = reliance['Close'].rolling(window=30).mean()
        reliance_ma['200-day MA'] = reliance['Close'].rolling(window=200).mean()

        st.write(reliance_ma)

        st.subheader('Stock Price vs 30-day Moving Average')
        plt.plot(reliance_ma['Close'], label='Original data')
        plt.plot(reliance_ma['30-day MA'], label='30-MA')
        plt.legend()
        plt.title('Stock Price vs 30-day Moving Average')
        plt.xlabel('Date')
        plt.ylabel('Price')

        st.pyplot(plt.gcf()) 

        st.subheader('Stock Price vs 200-day Moving Average')
        plt.plot(reliance_ma['Close'], label='Original data')
        plt.plot(reliance_ma['200-day MA'], label='200-MA')
        plt.legend()
        plt.title('Stock Price vs 200-day Moving Average')
        plt.xlabel('Date')
        plt.ylabel('Price')

        st.pyplot(plt.gcf()) 

        df1 = pd.DataFrame(future_predicted_values)
        df1.rename(columns={0: "Predicted Prices"}, inplace=True)
        st.markdown("### Next 30 days forecast")
        st.write(df1)

        # Original vs Predicted Close Price
        st.markdown("### Original vs Predicted Close Price")
        if 'Date' in plotdf.columns and 'test_predicted_close' in plotdf.columns:
            # Convert dates to datetime if needed
            plotdf['Date'] = pd.to_datetime(plotdf['Date'])
    
            # Get predicted prices from plotdf
            predicted_dates = plotdf['Date'].to_numpy()
            predicted_prices = plotdf['test_predicted_close'].to_numpy()
    
            # Get the actual close prices that match with the prediction dates
            actual_prices = []
            valid_indices = []
    
            for i, date in enumerate(predicted_dates):
                # Convert to Timestamp for index lookup if needed
                date_ts = pd.Timestamp(date)
                if date_ts in reliance.index:
                    actual_prices.append(reliance.loc[date_ts, 'Close'])
                    valid_indices.append(i)
    
            # Filter predicted_dates and predicted_prices to match valid actual_prices
            valid_dates = predicted_dates[valid_indices]
            valid_predicted = predicted_prices[valid_indices]
            actual_prices = np.array(actual_prices)
    
            if len(valid_dates) > 0 and len(actual_prices) > 0 and len(valid_predicted) > 0:
                # Make sure the arrays are the same length
                min_length = min(len(valid_predicted), len(actual_prices))
                valid_predicted = valid_predicted[:min_length]
                actual_prices = actual_prices[:min_length]
                valid_dates = valid_dates[:min_length]
        
                # Create the figure
                fig = plt.figure(figsize=(20, 10))
                plt.plot(valid_dates, actual_prices, label='Actual Close Price', color='blue', linewidth=2)
                plt.plot(valid_dates, valid_predicted, label='Predicted Close Price', color='red', linewidth=2)
                plt.title('Original vs Predicted Close Price', fontsize=16)
                plt.xlabel('Date', fontsize=12)
                plt.ylabel('Price', fontsize=12)
                plt.legend(fontsize=12)
                plt.grid(True, alpha=0.3)
        
                # Format x-axis dates
                plt.gcf().autofmt_xdate()
        
                # Calculate and display RMSE
                rmse = np.sqrt(np.mean(np.square(actual_prices - valid_predicted)))
                plt.annotate(f'RMSE: {rmse:.2f}', 
                            xy=(0.05, 0.95), 
                            xycoords='axes fraction',
                            fontsize=12,
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
                st.pyplot(fig)
        
                # Fix correlation calculation
                try:
                    # Make sure inputs are 1D arrays with same shape
                    if len(actual_prices) == len(valid_predicted) and len(actual_prices) > 1:
                        from scipy.stats import pearsonr
                        corr, p_value = pearsonr(actual_prices, valid_predicted)
                        st.write(f"Correlation between actual and predicted prices: {corr:.4f} (p-value: {p_value:.4f})")
                    else:
                        st.warning("Cannot calculate correlation: arrays have different lengths or insufficient data points")
                except Exception as e:
                    st.warning(f"Could not calculate correlation: {str(e)}")
            else:
                st.warning("Unable to match actual prices with prediction dates. Check that prediction dates exist in the original data.")
        else:
            st.error("Error: Required columns not found in model output. Need 'Date' and 'test_predicted_close'.")
        
        st.title('Investment Gain or Loss Calculation')


        # Function to find the next available date in the DataFrame
        def get_next_available_date(date, df):
            available_dates = df.index
            next_dates = available_dates[available_dates >= pd.Timestamp(date)]
            return next_dates[0] if not next_dates.empty else None

        try:
            # Find the next available investment and withdrawal dates
            next_investment_date = get_next_available_date(investment_date, reliance)
            next_withdrawal_date = get_next_available_date(withdrawal_date, reliance)

            if next_investment_date is None or next_withdrawal_date is None:
                st.error("Selected dates are outside the available data range")
            elif next_investment_date >= next_withdrawal_date:
                st.error("Investment date must be before withdrawal date")
            else:
                # Calculate investment results
                withdrawal_price = float(reliance.loc[next_withdrawal_date]['Close'])
                investment_price = float(reliance.loc[next_investment_date]['Close'])
                shares_bought = investment_amount / investment_price
                current_value = shares_bought * withdrawal_price
                gain_loss = current_value - investment_amount
                gain_loss_percentage = (gain_loss / investment_amount) * 100

                # Display results in a more organized way
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("### Investment Details")
                    st.write(f"Amount Invested: ₹{float(investment_amount):.2f}")
                    st.write(f"Shares Bought: {shares_bought:.2f}")
                    st.write(f"Purchase Date: {next_investment_date.date()}")
                    st.write(f"Purchase Price: ₹{investment_price:.2f}")
                
                with col2:
                    st.write("### Returns Details")
                    st.write(f"Current Value: ₹{current_value:.2f}")
                    st.write(f"Withdrawal Date: {next_withdrawal_date.date()}")
                    st.write(f"Selling Price: ₹{withdrawal_price:.2f}")
                    st.write(f"Total {'Profit' if gain_loss >= 0 else 'Loss'}: ₹{abs(gain_loss):.2f} ({gain_loss_percentage:.2f}%)")

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")