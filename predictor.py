import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from typing import List, Tuple, Optional
from datetime import datetime, timedelta


class KronosMiniPredictor:
    def __init__(self):
        self.scaler = MinMaxScaler()
        self.model = LinearRegression()
        
    def prepare_features(self, data: pd.DataFrame, window_size: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare features for training."""
        # Use closing prices for prediction
        prices = data['Close'].values.reshape(-1, 1)
        
        # Scale the data
        scaled_prices = self.scaler.fit_transform(prices)
        
        # Create features and targets
        X, y = [], []
        for i in range(window_size, len(scaled_prices)):
            X.append(scaled_prices[i-window_size:i, 0])
            y.append(scaled_prices[i, 0])
            
        return np.array(X), np.array(y)
    
    def train(self, data: pd.DataFrame) -> None:
        """Train the prediction model."""
        if len(data) < 10:  # Minimum data requirement
            raise ValueError("Insufficient data for training")
            
        X, y = self.prepare_features(data)
        
        # Train the model
        self.model.fit(X, y)
    
    def predict_next_days(self, data: pd.DataFrame, days: int = 5) -> List[float]:
        """Predict prices for the next N days."""
        if len(data) < 5:  # Minimum data requirement
            raise ValueError("Insufficient data for prediction")
            
        # Get the last 5 closing prices
        last_prices = data['Close'].tail(5).values.reshape(-1, 1)
        scaled_last_prices = self.scaler.transform(last_prices).flatten()
        
        predictions = []
        current_window = scaled_last_prices.tolist()
        
        # Predict for the specified number of days
        for _ in range(days):
            # Reshape for prediction
            X_pred = np.array(current_window).reshape(1, -1)
            
            # Predict next value
            next_scaled = self.model.predict(X_pred)[0]
            predictions.append(next_scaled)
            
            # Update window for next prediction
            current_window = current_window[1:] + [next_scaled]
        
        # Inverse transform to get actual prices
        predictions_array = np.array(predictions).reshape(-1, 1)
        actual_predictions = self.scaler.inverse_transform(predictions_array).flatten()
        
        return actual_predictions.tolist()