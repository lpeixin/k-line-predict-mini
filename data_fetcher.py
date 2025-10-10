import os
import yfinance as yf
import akshare as ak
import pandas as pd
from typing import Optional
from datetime import datetime, timedelta


class DataFetcher:
    def __init__(self):
        pass

    def fetch_us_stock(self, symbol: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """Fetch US stock data using yfinance for specified date range."""
        try:
            ticker = yf.Ticker(symbol)
            # yfinance end date is exclusive, so add one day to include the end_date
            end_dt = datetime.strptime(end_date, '%Y-%m-%d') + timedelta(days=1)
            end_date_inclusive = end_dt.strftime('%Y-%m-%d')
            data = ticker.history(start=start_date, end=end_date_inclusive)
            if data.empty:
                return None
            # Standardize column names
            data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
            data['amount'] = data['Volume'] * data['Close']  # Approximate amount
            data = data[['Open', 'High', 'Low', 'Close', 'Volume', 'amount']]
            data.columns = ['open', 'high', 'low', 'close', 'volume', 'amount']
            return data
        except Exception as e:
            print(f"Error fetching US stock data for {symbol}: {e}")
            return None

    def fetch_hk_stock(self, symbol: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """Fetch Hong Kong stock data using yfinance for specified date range."""
        try:
            if not symbol.endswith('.HK'):
                symbol = f"{symbol}.HK"
            ticker = yf.Ticker(symbol)
            # yfinance end date is exclusive, so add one day to include the end_date
            end_dt = datetime.strptime(end_date, '%Y-%m-%d') + timedelta(days=1)
            end_date_inclusive = end_dt.strftime('%Y-%m-%d')
            data = ticker.history(start=start_date, end=end_date_inclusive)
            if data.empty:
                return None
            data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
            data['amount'] = data['Volume'] * data['Close']
            data = data[['Open', 'High', 'Low', 'Close', 'Volume', 'amount']]
            data.columns = ['open', 'high', 'low', 'close', 'volume', 'amount']
            return data
        except Exception as e:
            print(f"Error fetching HK stock data for {symbol}: {e}")
            return None

    def fetch_cn_stock(self, symbol: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """Fetch Chinese A-share stock data using AKShare for specified date range."""
        try:
            clean_symbol = symbol.split('.')[0]
            start_dt = datetime.strptime(start_date, '%Y-%m-%d')
            end_dt = datetime.strptime(end_date, '%Y-%m-%d')

            data = ak.stock_zh_a_hist(symbol=clean_symbol, start_date=start_date, end_date=end_date, adjust="qfq")
            if data.empty:
                return None

            data['date'] = pd.to_datetime(data['日期'])
            data.set_index('date', inplace=True)
            data.rename(columns={
                '开盘': 'open',
                '最高': 'high',
                '最低': 'low',
                '收盘': 'close',
                '成交量': 'volume'
            }, inplace=True)
            data['amount'] = data['volume'] * data['close']
            data = data[['open', 'high', 'low', 'close', 'volume', 'amount']]
            return data
        except Exception as e:
            print(f"Error fetching CN stock data for {symbol}: {e}")
            return None

    def fetch_crypto(self, symbol: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """Fetch cryptocurrency data using yfinance for specified date range."""
        try:
            if not (symbol.endswith('-USD') or symbol.endswith('-USDT')):
                symbol = f"{symbol}-USD"
            ticker = yf.Ticker(symbol)
            # yfinance end date is exclusive, so add one day to include the end_date
            end_dt = datetime.strptime(end_date, '%Y-%m-%d') + timedelta(days=1)
            end_date_inclusive = end_dt.strftime('%Y-%m-%d')
            data = ticker.history(start=start_date, end=end_date_inclusive)
            if data.empty:
                return None
            data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
            data['amount'] = data['Volume'] * data['Close']
            data = data[['Open', 'High', 'Low', 'Close', 'Volume', 'amount']]
            data.columns = ['open', 'high', 'low', 'close', 'volume', 'amount']
            return data
        except Exception as e:
            print(f"Error fetching crypto data for {symbol}: {e}")
            return None

    def fetch_data(self, symbol: str, market: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """Fetch data based on market type and date range."""
        if market == "US":
            return self.fetch_us_stock(symbol, start_date, end_date)
        elif market == "HK":
            return self.fetch_hk_stock(symbol, start_date, end_date)
        elif market == "CN":
            return self.fetch_cn_stock(symbol, start_date, end_date)
        elif market == "crypto":
            return self.fetch_crypto(symbol, start_date, end_date)
        else:
            raise ValueError(f"Unsupported market: {market}")

    def save_to_csv(self, data: pd.DataFrame, symbol: str, market: str) -> str:
        """Save data to CSV in data folder."""
        os.makedirs('data', exist_ok=True)
        filename = f"data/{symbol}_{market}.csv"
        data.to_csv(filename)
        return filename