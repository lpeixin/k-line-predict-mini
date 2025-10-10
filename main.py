#!/usr/bin/env python3

import argparse
import sys
from datetime import datetime, timedelta
from data_fetcher import DataFetcher
from predictor import (
    KronosModelPredictor,
    KronosNotAvailableError,
    VARIANT_PRESETS,
)
import pandas as pd


def download_command(args):
    """Handle data download command."""
    fetcher = DataFetcher()

    print(f"Downloading {args.market} data for {args.symbol} from {args.start_date} to {args.end_date}...")

    data = fetcher.fetch_data(args.symbol, args.market, args.start_date, args.end_date)
    if data is None or data.empty:
        print(f"Error: Could not fetch data for {args.symbol}")
        sys.exit(1)

    csv_path = fetcher.save_to_csv(data, args.symbol, args.market)
    print(f"Successfully downloaded {len(data)} days of data")
    print(f"Saved to: {csv_path}")


def predict_command(args):
    """Handle prediction command."""
    # Load data from CSV
    try:
        data = KronosModelPredictor.load_data_from_csv(args.csv_file)
    except Exception as e:
        print(f"Error loading CSV file {args.csv_file}: {e}")
        sys.exit(1)

    if data.empty:
        print("Error: CSV file is empty")
        sys.exit(1)

    print(f"Loaded {len(data)} days of historical data from {args.csv_file}")

    # Initialize predictor
    predictor = KronosModelPredictor(
        model_variant=args.model_variant,
        model_id=args.model_id,
        tokenizer_id=args.tokenizer_id,
        device=args.device,
        max_context=args.max_context,
    )

    print(f"Using model: {predictor.model_id} | Tokenizer: {predictor.tokenizer_id} | Context: {predictor.max_context} | Device: {predictor.device}")

    # Predict future prices
    try:
        predictions = predictor.predict_next_days(data, args.days, args.granularity)
    except KronosNotAvailableError as e:
        print(str(e))
        sys.exit(2)
    except ValueError as e:
        print(f"Prediction error: {e}")
        sys.exit(1)

    # Display predictions
    print(f"\nPredictions for next {args.days} {args.granularity}s:")
    print("-" * 40)
    current_date = data.index[-1] if isinstance(data.index, pd.DatetimeIndex) else datetime.now()
    for i, price in enumerate(predictions, 1):
        prediction_date = current_date + timedelta(days=i)
        print(f"{args.granularity.capitalize()} {i} ({prediction_date.strftime('%Y-%m-%d')}): ${price:.4f}")

    # Show trend
    if len(predictions) > 1:
        trend = "↑" if predictions[-1] > predictions[0] else "↓"
        change = ((predictions[-1] - predictions[0]) / predictions[0]) * 100
        print(f"\nOverall trend: {trend} ({change:+.2f}%)")


def main():
    parser = argparse.ArgumentParser(
        description="K-Line Predictor: Download stock data and predict future prices using Kronos models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download AAPL data for 2023
  python main.py download AAPL US --start-date 2023-01-01 --end-date 2023-12-31

  # Predict next 5 days using mini model
  python main.py predict data/AAPL_US.csv --model-variant mini --days 5

  # Predict using custom model
  python main.py predict data/AAPL_US.csv --model-id NeoQuasar/Kronos-base --days 10
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Download subcommand
    download_parser = subparsers.add_parser('download', help='Download historical stock data')
    download_parser.add_argument('symbol', help='Stock symbol (e.g., AAPL, 000001, BTC)')
    download_parser.add_argument(
        'market',
        choices=['US', 'HK', 'CN', 'crypto'],
        help='Market type (US: yfinance, HK: yfinance, CN: AKShare, crypto: yfinance)'
    )
    download_parser.add_argument(
        '--start-date',
        required=True,
        help='Start date in YYYY-MM-DD format'
    )
    download_parser.add_argument(
        '--end-date',
        required=True,
        help='End date in YYYY-MM-DD format'
    )

    # Predict subcommand
    predict_parser = subparsers.add_parser('predict', help='Predict future prices using Kronos model')
    predict_parser.add_argument('csv_file', help='Path to CSV file with historical data')
    predict_parser.add_argument(
        '--days',
        type=int,
        default=5,
        help='Number of future days to predict (default: 5)'
    )
    predict_parser.add_argument(
        '--granularity',
        choices=['day'],
        default='day',
        help='Prediction granularity (default: day, only day supported currently)'
    )
    predict_parser.add_argument(
        '--model-variant',
        choices=list(VARIANT_PRESETS.keys()),
        default=None,
        help='Kronos model variant preset (mini/small/base). Overrides env if set.'
    )
    predict_parser.add_argument(
        '--model-id',
        default=None,
        help='Explicit Kronos model HF repo id (e.g. NeoQuasar/Kronos-small). Overrides variant preset.'
    )
    predict_parser.add_argument(
        '--tokenizer-id',
        default=None,
        help='Explicit Kronos tokenizer HF repo id (e.g. NeoQuasar/Kronos-Tokenizer-base).'
    )
    predict_parser.add_argument(
        '--device',
        default=None,
        help='Device for inference (e.g., cuda:0 or cpu). Defaults to GPU if available.'
    )
    predict_parser.add_argument(
        '--max-context',
        type=int,
        default=None,
        help='Override max context length (will be clamped for small/base).'
    )

    args = parser.parse_args()

    if args.command == 'download':
        download_command(args)
    elif args.command == 'predict':
        predict_command(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()