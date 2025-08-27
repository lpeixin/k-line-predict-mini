import sqlite3
from typing import Optional, Tuple, List
import pandas as pd
from datetime import datetime


class StockDatabase:
    def __init__(self, db_path: str = "stock_data.db"):
        self.db_path = db_path
        self.init_database()

    def init_database(self) -> None:
        """Initialize the database with required tables."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS stock_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                market TEXT NOT NULL,
                date TEXT NOT NULL,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume INTEGER,
                UNIQUE(symbol, market, date)
            )
        """)
        
        conn.commit()
        conn.close()

    def save_stock_data(self, symbol: str, market: str, data: pd.DataFrame) -> None:
        """Save stock data to database."""
        conn = sqlite3.connect(self.db_path)
        
        for date, row in data.iterrows():
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO stock_data 
                (symbol, market, date, open, high, low, close, volume)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                symbol,
                market,
                date.strftime('%Y-%m-%d'),
                float(row['Open']),
                float(row['High']),
                float(row['Low']),
                float(row['Close']),
                int(row['Volume'])
            ))
        
        conn.commit()
        conn.close()

    def get_stock_data(self, symbol: str, market: str, days: int = 365) -> Optional[pd.DataFrame]:
        """Retrieve stock data from database."""
        conn = sqlite3.connect(self.db_path)
        
        query = """
            SELECT date, open, high, low, close, volume
            FROM stock_data
            WHERE symbol = ? AND market = ?
            ORDER BY date DESC
            LIMIT ?
        """
        
        df = pd.read_sql_query(query, conn, params=(symbol, market, days))
        conn.close()
        
        if df.empty:
            return None
            
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        df.sort_index(inplace=True)
        
        # Rename columns to match expected format (capitalize first letter)
        df.rename(columns={
            'open': 'Open',
            'high': 'High', 
            'low': 'Low',
            'close': 'Close',
            'volume': 'Volume'
        }, inplace=True)
        
        return df