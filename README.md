# K-Line Predictor (Kronos-Powered)

A command-line tool for downloading historical stock data and predicting future K-line prices using the open-source Kronos foundation model.

## Features

- **Data Download**: Download historical OHLCV data from multiple sources
  - US stocks: Yahoo Finance (yfinance)
  - Hong Kong stocks: Yahoo Finance (yfinance)
  - Chinese A-shares: AKShare
  - Cryptocurrencies: Yahoo Finance (yfinance)
- **Price Prediction**: Use Kronos models for future price forecasting
- **Flexible Configuration**: Support for different Kronos model variants (mini/small/base)
- **CSV Storage**: Data saved in standardized CSV format

## Installation

1. Clone the repository:
```bash
git clone <your-repo-url>
cd k-line-predict-mini
```

2. Install Python dependencies:
```bash
pip install -r requirements.txt
```

3. Install PyTorch (choose based on your system):
```bash
# CPU version
pip install torch --index-url https://download.pytorch.org/whl/cpu

# GPU version (if you have CUDA)
pip install torch
```

4. Clone and setup Kronos repository:
```bash
git clone https://github.com/shiyu-coder/Kronos.git kronos_repo
export KRONOS_REPO_PATH=$(pwd)/kronos_repo
pip install -r kronos_repo/requirements.txt
```

## Usage

The tool provides two main commands: `download` for fetching historical data and `predict` for forecasting future prices.

### Download Historical Data

```bash
python main.py download <SYMBOL> <MARKET> --start-date YYYY-MM-DD --end-date YYYY-MM-DD
```

**Examples:**
```bash
# Download Apple stock data for 2023
python main.py download AAPL US --start-date 2023-01-01 --end-date 2023-12-31

# Download Chinese stock data
python main.py download 000001 CN --start-date 2023-01-01 --end-date 2023-12-31

# Download Bitcoin data
python main.py download BTC crypto --start-date 2023-01-01 --end-date 2023-12-31
```

**Parameters:**
- `SYMBOL`: Stock symbol (e.g., AAPL, 000001, BTC)
- `MARKET`: Market type (US, HK, CN, crypto)
- `--start-date`: Start date in YYYY-MM-DD format
- `--end-date`: End date in YYYY-MM-DD format

Data is saved to `data/<SYMBOL>_<MARKET>.csv` with columns: `open`, `high`, `low`, `close`, `volume`, `amount`.

### Predict Future Prices

```bash
python main.py predict <CSV_FILE> [--model-variant VARIANT] [--days N] [other options]
```

**Examples:**
```bash
# Predict next 5 days using mini model (default)
python main.py predict data/AAPL_US.csv

# Predict next 10 days using small model
python main.py predict data/AAPL_US.csv --model-variant small --days 10

# Use only last 30 days of historical data for prediction
python main.py predict data/AAPL_US.csv --lookback-days 30 --days 5

# Use custom model and tokenizer
python main.py predict data/AAPL_US.csv \
  --model-id NeoQuasar/Kronos-base \
  --tokenizer-id NeoQuasar/Kronos-Tokenizer-base \
  --days 7
```

**Parameters:**
- `CSV_FILE`: Path to CSV file with historical data
- `--days`: Number of future days to predict (default: 5)
- `--lookback-days`: Number of historical days to use for prediction (default: use all available data within model context limit)
- `--granularity`: Prediction granularity (default: day, currently only 'day' supported)
- `--model-variant`: Kronos model variant (mini/small/base)
- `--model-id`: Custom Hugging Face model repo ID
- `--tokenizer-id`: Custom Hugging Face tokenizer repo ID
- `--device`: Device for inference (cuda:0, cpu, etc.)
- `--max-context`: Override maximum context length

## Kronos Model Variants

| Variant | Model ID                     | Tokenizer ID                      | Max Context | Description |
|---------|------------------------------|------------------------------------|-------------|-------------|
| mini    | NeoQuasar/Kronos-mini        | NeoQuasar/Kronos-Tokenizer-2k      | 2048        | Fastest, good for quick predictions |
| small   | NeoQuasar/Kronos-small       | NeoQuasar/Kronos-Tokenizer-base    | 512         | Balanced performance |
| base    | NeoQuasar/Kronos-base        | NeoQuasar/Kronos-Tokenizer-base    | 512         | Most accurate, slower |

## Configuration

You can configure the tool using environment variables:

```bash
export KRONOS_MODEL_VARIANT=mini        # or small / base
export KRONOS_MODEL_ID=NeoQuasar/Kronos-mini
export KRONOS_TOKENIZER_ID=NeoQuasar/Kronos-Tokenizer-2k
export KRONOS_MAX_CONTEXT=2048
export KRONOS_DEVICE=cuda:0
export KRONOS_REPO_PATH=/path/to/kronos_repo
```

Configuration precedence: CLI arguments > Environment variables > Variant defaults.

## CSV Data Format

Downloaded data is saved in CSV format with the following columns:
- `open`: Opening price
- `high`: Highest price
- `low`: Lowest price
- `close`: Closing price
- `volume`: Trading volume
- `amount`: Trading amount (volume * close for approximation)

The date is used as the index.

## Requirements

- Python 3.8+
- PyTorch
- yfinance
- akshare
- pandas
- numpy
- scikit-learn

## Troubleshooting

### Kronos Import Error
If you get an error about importing Kronos:
1. Ensure you've cloned the Kronos repository
2. Set the `KRONOS_REPO_PATH` environment variable
3. Make sure Kronos dependencies are installed

### Data Download Issues
- US/HK/Crypto: Check internet connection and symbol validity
- CN: Ensure AKShare is properly installed and configured

### GPU Issues
- If CUDA is not available, the tool will automatically fall back to CPU
- You can explicitly set `--device cpu` to force CPU usage

## Disclaimer

This tool is for educational and research purposes only. Financial predictions are inherently uncertain, and past performance does not guarantee future results. Always do your own research and consult with financial professionals before making investment decisions.