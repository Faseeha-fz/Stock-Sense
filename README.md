# Stock Market Price Prediction

This project is a machine learning-based tool for predicting stock price movements using various models. Users can select stocks from the **yfinance** library and analyze their future trends.

## Features

- Predicts stock price movement (gain or loss) using machine learning models.
- Supports multiple stock selections from **yfinance**.
- Implements models like Linear Regression, Decision Trees, Random Forests, XGBoost, and LSTM.
- After comparing several machine learning models, LSTM was selected for calculations.
- Provides an interactive interface using Streamlit.
- Offers empirical analysis of model performance.

## Tech Stack

- **Programming Language:** Python
- **Libraries:** NumPy, Pandas, Matplotlib, Scikit-Learn, XGBoost, TensorFlow/Keras, yfinance
- **Framework:** Streamlit

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Faseeha-fz/Stock-Sense.git
   cd Stock-Sense
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the application:
   ```bash
   streamlit run app.py
   ```

## Usage

- Choose a stock from **yfinance**.
- Select a machine learning model for prediction.
- View the predicted stock price trend and analysis.


