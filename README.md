# Local Stock Price Predictor (Kronos-mini Powered)

A local stock & crypto K-line forecasting tool integrating the open-source Kronos-mini foundation model ("NeoQuasar/Kronos-mini"). Supports US, HK, CN stocks and major cryptocurrencies.

## Features

- Predicts future daily close prices (default 5 days) using Kronos-mini
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
python main.py <SYMBOL> <MARKET> [--days N] [--model-variant mini|small|base] \
   [--model-id HF_MODEL_ID] [--tokenizer-id HF_TOKENIZER_ID] [--device DEV] [--max-context N]
```

If installed as a package:
```
stock-predict <SYMBOL> <MARKET> [...same options]
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

# Use the small preset variant
python main.py AAPL US --model-variant small

# Explicit model & tokenizer ids
python main.py AAPL US --model-id NeoQuasar/Kronos-base \
   --tokenizer-id NeoQuasar/Kronos-Tokenizer-base

# Override context length (clamped if exceeds variant limit)
python main.py AAPL US --max-context 400
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
3. Uses the pre-trained Kronos-mini foundation model (no local training step)
4. Provides real-time price and future predictions

## Kronos Integration & Configuration

This project wraps the official Kronos repository instead of copying its code. You must clone the Kronos repo (once) so that the `model` package is importable:

```bash
git clone https://github.com/shiyu-coder/Kronos.git kronos_repo
export KRONOS_REPO_PATH=$(pwd)/kronos_repo
```

Or add the path to `PYTHONPATH` manually. At runtime the wrapper attempts:

1. Native import (`import model`)
2. `KRONOS_REPO_PATH`
3. Local folders `./kronos_repo` or `./Kronos`

If not found, an actionable error is shown.

### Configuration Precedence

CLI arguments > Environment variables > Variant preset defaults.

Environment variables (used when corresponding CLI args omitted):

```bash
export KRONOS_MODEL_VARIANT=mini        # or small / base
export KRONOS_MODEL_ID=NeoQuasar/Kronos-mini
export KRONOS_TOKENIZER_ID=NeoQuasar/Kronos-Tokenizer-2k
export KRONOS_MAX_CONTEXT=2048
export KRONOS_DEVICE=cuda:0
```

### Variant Presets

| Variant | Model ID                     | Tokenizer ID                      | Max Context |
|---------|------------------------------|------------------------------------|-------------|
| mini    | NeoQuasar/Kronos-mini        | NeoQuasar/Kronos-Tokenizer-2k      | 2048        |
| small   | NeoQuasar/Kronos-small       | NeoQuasar/Kronos-Tokenizer-base    | 512         |
| base    | NeoQuasar/Kronos-base        | NeoQuasar/Kronos-Tokenizer-base    | 512         |

If you pass `--model-id` it overrides the preset. Same for `--tokenizer-id`.

### Extra Dependencies

Install torch etc. (CPU example):

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Then install Kronos dependencies (in the cloned repo):

```bash
pip install -r kronos_repo/requirements.txt
```

## Disclaimer

Financial forecasts are uncertain; this tool is for research/education only. No investment advice.