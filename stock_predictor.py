import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import yfinance as yf
import json
from datetime import datetime, timedelta

def fetch_stock_data(symbol, start_date, end_date):
    stock = yf.Ticker(symbol)
    data = stock.history(start=start_date, end=end_date)
    return data['Close']

def train_model(data):
    X = np.array(range(len(data))).reshape(-1, 1)
    y = data.values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    
    return model, mse

def predict_next_week(model, data):
    last_day = len(data)
    next_week = np.array(range(last_day, last_day + 7)).reshape(-1, 1)
    predictions = model.predict(next_week)
    return predictions.tolist()

def main():
    symbol = 'AAPL'  # Apple Inc. as an example
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
    
    data = fetch_stock_data(symbol, start_date, end_date)
    model, mse = train_model(data)
    next_week_predictions = predict_next_week(model, data)
    
    results = {
        'symbol': symbol,
        'last_price': data.iloc[-1],
        'mse': mse,
        'next_week_predictions': next_week_predictions,
        'last_updated': end_date
    }
    
    with open('prediction_results.json', 'w') as f:
        json.dump(results, f)

if __name__ == "__main__":
    main()
