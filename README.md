# Local Stock Price Predictor

A local stock price prediction tool using Kronos-mini model with support for US, HK, CN stocks and cryptocurrencies.

## Features

- Predicts stock prices for the next 5 days
- Supports US, Hong Kong, Chinese A-shares, and cryptocurrencies
- Uses local SQLite database to cache historical data
- Real-time price fetching
- Simple command-line interface

## Installation

### Method 1: Using the installation script (Recommended)

On macOS/Linux:
```bash
./install.sh
source venv/bin/activate
```

On Windows:
```cmd
install.bat
venv\Scripts\activate.bat
```

### Method 2: Manual installation

1. Create a virtual environment:
   ```bash
   python3 -m venv venv
   ```

2. Activate the virtual environment:
   
   On macOS/Linux:
   ```bash
   source venv/bin/activate
   ```
   
   On Windows:
   ```cmd
   venv\Scripts\activate.bat
   ```

3. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

### Method 3: Install as a package

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate.bat
pip install -e .
```

## Usage

```
python main.py <SYMBOL> <MARKET> [--days N]
```

Or if installed as a package:
```
stock-predict <SYMBOL> <MARKET> [--days N]
```

### Examples:

```bash
# Predict Apple stock (US market)
python main.py AAPL US

# Predict Tencent stock (HK market)
python main.py 0700 HK

# Predict Ping An Bank (CN market)
python main.py 000001 CN

# Predict Bitcoin price
python main.py BTC crypto

# Predict for 10 days instead of default 5
python main.py AAPL US --days 10
```

## Supported Markets

- `US`: US stocks (e.g., AAPL, GOOGL)
- `HK`: Hong Kong stocks (e.g., 0700, 9988)
- `CN`: Chinese A-shares (e.g., 000001, 600000)
- `crypto`: Cryptocurrencies (e.g., BTC, ETH)

## How It Works

1. Fetches historical data using appropriate libraries:
   - `yfinance` for US/HK stocks and cryptocurrencies
   - `akshare` for Chinese A-shares
2. Caches data in local SQLite database
3. Uses a simple linear regression model (Kronos-mini) for price prediction
4. Provides real-time price and future predictions

## Note

This is a simplified prediction model for educational purposes. Stock market prediction is inherently uncertain, and past performance does not guarantee future results. Use predictions at your own risk.