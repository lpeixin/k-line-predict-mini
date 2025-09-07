#!/usr/bin/env python3

import argparse
import sys
from typing import Optional
from database import StockDatabase
from data_fetcher import DataFetcher
from predictor import (
    KronosMiniPredictor,
    KronosModelPredictor,
    KronosNotAvailableError,
    VARIANT_PRESETS,
)
import pandas as pd
from datetime import datetime, timedelta


def main():
    parser = argparse.ArgumentParser(description="Local Stock Price Predictor")
    parser.add_argument("symbol", help="Stock symbol (e.g., AAPL, 000001, BTC)")
    parser.add_argument(
        "market", choices=["US", "HK", "CN", "crypto"], help="Market type"
    )
    parser.add_argument(
        "--days",
        type=int,
        default=5,
        help="Number of days to predict (default: 5)",
    )
    parser.add_argument(
        "--model-variant",
        choices=list(VARIANT_PRESETS.keys()),
        default=None,
        help="Kronos model variant preset (mini/small/base). Overrides env if set.",
    )
    parser.add_argument(
        "--model-id",
        default=None,
        help="Explicit Kronos model HF repo id (e.g. NeoQuasar/Kronos-small). Overrides variant preset.",
    )
    parser.add_argument(
        "--tokenizer-id",
        default=None,
        help="Explicit Kronos tokenizer HF repo id (e.g. NeoQuasar/Kronos-Tokenizer-base).",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device for inference (e.g., cuda:0 or cpu). Defaults to GPU if available.",
    )
    parser.add_argument(
        "--max-context",
        type=int,
        default=None,
        help="Override max context length (will be clamped for small/base).",
    )
    
    args = parser.parse_args()
    
    # Initialize components
    db = StockDatabase()
    fetcher = DataFetcher(db)
    predictor = KronosModelPredictor(
        model_variant=args.model_variant,
        model_id=args.model_id,
        tokenizer_id=args.tokenizer_id,
        device=args.device,
        max_context=args.max_context,
    )
    
    print(f"Fetching data for {args.symbol} in {args.market} market...")
    
    # Fetch data
    data = fetcher.fetch_data(args.symbol, args.market)
    if data is None or data.empty:
        print(f"Error: Could not fetch data for {args.symbol}")
        sys.exit(1)
    
    print(
        f"Successfully fetched {len(data)} days of historical data | Model: {predictor.model_id} | Tokenizer: {predictor.tokenizer_id} | Context: {predictor.max_context} | Device: {predictor.device}"
    )
    
    # Get real-time price
    real_time_price = fetcher.get_real_time_price(args.symbol, args.market)
    if real_time_price:
        print(f"Current price: ${real_time_price:.2f}")
    
    # Predict future prices (Kronos is pre-trained; no local training step)
    try:
        predictions = predictor.predict_next_days(data, args.days)
    except KronosNotAvailableError as e:
        print(str(e))
        sys.exit(2)
    except ValueError as e:
        print(f"Prediction error: {e}")
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