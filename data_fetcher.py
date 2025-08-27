import yfinance as yf
import akshare as ak
import pandas as pd
from typing import Optional, Tuple
from datetime import datetime, timedelta


class DataFetcher:
    def __init__(self, db):
        self.db = db

    def fetch_us_stock(self, symbol: str) -> Optional[pd.DataFrame]:
        """Fetch US stock data using yfinance."""
        try:
            ticker = yf.Ticker(symbol)
            # Get max historical data
            data = ticker.history(period="max")
            return data if not data.empty else None
        except Exception as e:
            print(f"Error fetching US stock data for {symbol}: {e}")
            return None

    def fetch_hk_stock(self, symbol: str) -> Optional[pd.DataFrame]:
        """Fetch Hong Kong stock data using yfinance."""
        try:
            # HK stocks in Yahoo Finance usually have .HK suffix
            if not symbol.endswith('.HK'):
                symbol = f"{symbol}.HK"
            ticker = yf.Ticker(symbol)
            data = ticker.history(period="max")
            return data if not data.empty else None
        except Exception as e:
            print(f"Error fetching HK stock data for {symbol}: {e}")
            return None

    def fetch_cn_stock(self, symbol: str) -> Optional[pd.DataFrame]:
        """Fetch Chinese A-share stock data using AKShare."""
        try:
            # Remove any suffix for AKShare
            clean_symbol = symbol.split('.')[0]
            
            # For A-shares, we need to format the symbol properly
            if clean_symbol.startswith('6'):
                formatted_symbol = f"sh{clean_symbol}"
            else:
                formatted_symbol = f"sz{clean_symbol}"
                
            # Get historical data
            data = ak.stock_zh_a_hist(symbol=clean_symbol, period="daily", adjust="qfq")
            if data.empty:
                return None
                
            # Format data to match expected structure
            data['date'] = pd.to_datetime(data['日期'])
            data.set_index('date', inplace=True)
            data.rename(columns={
                '开盘': 'Open',
                '最高': 'High',
                '最低': 'Low',
                '收盘': 'Close',
                '成交量': 'Volume'
            }, inplace=True)
            
            return data[['Open', 'High', 'Low', 'Close', 'Volume']]
        except Exception as e:
            print(f"Error fetching CN stock data for {symbol}: {e}")
            return None

    def fetch_crypto(self, symbol: str) -> Optional[pd.DataFrame]:
        """Fetch cryptocurrency data using yfinance."""
        try:
            # Crypto symbols in Yahoo Finance
            if not (symbol.endswith('-USD') or symbol.endswith('-USDT')):
                symbol = f"{symbol}-USD"
            ticker = yf.Ticker(symbol)
            data = ticker.history(period="max")
            return data if not data.empty else None
        except Exception as e:
            print(f"Error fetching crypto data for {symbol}: {e}")
            return None

    def get_real_time_price(self, symbol: str, market: str) -> Optional[float]:
        """Get real-time price for a symbol."""
        try:
            if market == "US":
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period="1d")
                return hist['Close'][-1] if not hist.empty else None
            elif market == "HK":
                formatted_symbol = f"{symbol}.HK" if not symbol.endswith('.HK') else symbol
                ticker = yf.Ticker(formatted_symbol)
                hist = ticker.history(period="1d")
                return hist['Close'][-1] if not hist.empty else None
            elif market == "CN":
                # For CN stocks, we'll get the latest close price from database
                data = self.db.get_stock_data(symbol, market, 1)
                return data['Close'][-1] if data is not None and not data.empty else None
            elif market == "crypto":
                formatted_symbol = symbol
                if not (symbol.endswith('-USD') or symbol.endswith('-USDT')):
                    formatted_symbol = f"{symbol}-USD"
                ticker = yf.Ticker(formatted_symbol)
                hist = ticker.history(period="1d")
                return hist['Close'][-1] if not hist.empty else None
            return None
        except Exception as e:
            print(f"Error getting real-time price for {symbol}: {e}")
            return None

    def fetch_data(self, symbol: str, market: str) -> Optional[pd.DataFrame]:
        """Fetch data based on market type."""
        # First try to get from database
        data = self.db.get_stock_data(symbol, market)
        
        # If not in database or data is old, fetch new data
        if data is None or len(data) < 30:  # Simple check for sufficient data
            if market == "US":
                data = self.fetch_us_stock(symbol)
            elif market == "HK":
                data = self.fetch_hk_stock(symbol)
            elif market == "CN":
                data = self.fetch_cn_stock(symbol)
            elif market == "crypto":
                data = self.fetch_crypto(symbol)
            
            # Save to database if fetched successfully
            if data is not None:
                self.db.save_stock_data(symbol, market, data)
                return data
            elif data is None:
                # Try to get from database as fallback
                data = self.db.get_stock_data(symbol, market)
        
        return data