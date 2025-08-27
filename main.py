#!/usr/bin/env python3

import argparse
import sys
from typing import Optional
from database import StockDatabase
from data_fetcher import DataFetcher
from predictor import KronosMiniPredictor
import pandas as pd
from datetime import datetime, timedelta


def main():
    parser = argparse.ArgumentParser(description="Local Stock Price Predictor")
    parser.add_argument("symbol", help="Stock symbol (e.g., AAPL, 000001, BTC)")
    parser.add_argument("market", choices=["US", "HK", "CN", "crypto"], 
                       help="Market type")
    parser.add_argument("--days", type=int, default=5, 
                       help="Number of days to predict (default: 5)")
    
    args = parser.parse_args()
    
    # Initialize components
    db = StockDatabase()
    fetcher = DataFetcher(db)
    predictor = KronosMiniPredictor()
    
    print(f"Fetching data for {args.symbol} in {args.market} market...")
    
    # Fetch data
    data = fetcher.fetch_data(args.symbol, args.market)
    if data is None or data.empty:
        print(f"Error: Could not fetch data for {args.symbol}")
        sys.exit(1)
    
    print(f"Successfully fetched {len(data)} days of historical data")
    
    # Get real-time price
    real_time_price = fetcher.get_real_time_price(args.symbol, args.market)
    if real_time_price:
        print(f"Current price: ${real_time_price:.2f}")
    
    # Train predictor
    try:
        predictor.train(data)
    except ValueError as e:
        print(f"Error training model: {e}")
        sys.exit(1)
    
    # Predict future prices
    try:
        predictions = predictor.predict_next_days(data, args.days)
    except ValueError as e:
        print(f"Error making predictions: {e}")
        sys.exit(1)
    
    # Display predictions
    print(f"\nPredictions for next {args.days} days:")
    print("-" * 30)
    current_date = datetime.now()
    for i, price in enumerate(predictions, 1):
        prediction_date = current_date + timedelta(days=i)
        print(f"Day {i} ({prediction_date.strftime('%Y-%m-%d')}): ${price:.2f}")
    
    # Show trend
    if len(predictions) > 1:
        trend = "↑" if predictions[-1] > predictions[0] else "↓"
        change = ((predictions[-1] - predictions[0]) / predictions[0]) * 100
        print(f"\nOverall trend: {trend} ({change:+.2f}%)")


if __name__ == "__main__":
    main()