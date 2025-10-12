"""Kronos model integration layer.

Supports dynamic selection of Kronos variants (mini / small / base) and custom
tokenizer or model ids via:
1. Constructor parameters
2. Environment variables
3. (Fallback defaults)

Environment variables (if constructor args omitted):
    KRONOS_MODEL_VARIANT   -> one of: mini, small, base
    KRONOS_MODEL_ID        -> full HF repo id e.g. NeoQuasar/Kronos-mini
    KRONOS_TOKENIZER_ID    -> full HF tokenizer repo id
    KRONOS_MAX_CONTEXT     -> int overriding default context length
    KRONOS_DEVICE          -> e.g. cuda:0 / cpu

Variant defaults (from Kronos model card):
    mini  -> model: NeoQuasar/Kronos-mini,  tokenizer: NeoQuasar/Kronos-Tokenizer-2k,   max_context=2048
    small -> model: NeoQuasar/Kronos-small, tokenizer: NeoQuasar/Kronos-Tokenizer-base, max_context=512
    base  -> model: NeoQuasar/Kronos-base,  tokenizer: NeoQuasar/Kronos-Tokenizer-base, max_context=512

Backward compatibility:
    The older class name KronosMiniPredictor is kept as an alias pointing to
    KronosModelPredictor (with default variant 'mini').
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta
from typing import List

import pandas as pd


class KronosNotAvailableError(ImportError):
    pass


def _ensure_kronos_on_path() -> None:
    """Try to ensure Kronos repo is importable.

    Looks for:
    - Already importable 'model' package with Kronos classes.
    - Environment variable KRONOS_REPO_PATH.
    - Local folder names: 'kronos_repo', 'Kronos'.
    Does not perform network operations (keeps this lightweight & offline friendly).
    """
    try:
        import model  # noqa: F401  # type: ignore
        return
    except Exception:  # pragma: no cover - fall through to path injection
        pass

    candidate_dirs = []
    env_path = os.getenv("KRONOS_REPO_PATH")
    if env_path:
        candidate_dirs.append(env_path)
    # common local clone names
    candidate_dirs.extend([
        os.path.join(os.getcwd(), "kronos_repo"),
        os.path.join(os.getcwd(), "Kronos"),
    ])

    for d in candidate_dirs:
        if d and os.path.isdir(d) and os.path.isdir(os.path.join(d, "model")):
            if d not in sys.path:
                sys.path.insert(0, d)
            try:
                import model  # noqa: F401  # type: ignore
                return
            except Exception:  # pragma: no cover
                continue

    raise KronosNotAvailableError(
        "Unable to import Kronos. Please clone the repo: \n"
        "  git clone https://github.com/shiyu-coder/Kronos.git kronos_repo\n"
        "Then export KRONOS_REPO_PATH=./kronos_repo (or add it to PYTHONPATH)."
    )


VARIANT_PRESETS = {
    "mini": {
        "model_id": "NeoQuasar/Kronos-mini",
        "tokenizer_id": "NeoQuasar/Kronos-Tokenizer-2k",
        "max_context": 2048,
    },
    "small": {
        "model_id": "NeoQuasar/Kronos-small",
        "tokenizer_id": "NeoQuasar/Kronos-Tokenizer-base",
        "max_context": 512,
    },
    "base": {
        "model_id": "NeoQuasar/Kronos-base",
        "tokenizer_id": "NeoQuasar/Kronos-Tokenizer-base",
        "max_context": 512,
    },
}


class KronosModelPredictor:
    """Wrapper for configurable Kronos model forecasting daily closes."""

    def __init__(
        self,
        model_variant: str | None = None,
        model_id: str | None = None,
        tokenizer_id: str | None = None,
        device: str | None = None,
        max_context: int | None = None,
    ) -> None:
        _ensure_kronos_on_path()

        # Resolve configuration precedence: constructor > env > preset default
        env_variant = os.getenv("KRONOS_MODEL_VARIANT")
        variant = (model_variant or env_variant or "mini").lower()
        if variant not in VARIANT_PRESETS:
            raise ValueError(f"Unknown Kronos model variant '{variant}'. Valid: {list(VARIANT_PRESETS)}")

        preset = VARIANT_PRESETS[variant]
        resolved_model_id = model_id or os.getenv("KRONOS_MODEL_ID") or preset["model_id"]
        resolved_tokenizer_id = tokenizer_id or os.getenv("KRONOS_TOKENIZER_ID") or preset["tokenizer_id"]
        resolved_max_context = (
            max_context
            or (int(os.getenv("KRONOS_MAX_CONTEXT")) if os.getenv("KRONOS_MAX_CONTEXT") else None)
            or preset["max_context"]
        )

        self.device = device or os.getenv("KRONOS_DEVICE") or ("cuda:0" if self._gpu_available() else "cpu")
        self.max_context = resolved_max_context
        self.model_variant = variant
        self.model_id = resolved_model_id
        self.tokenizer_id = resolved_tokenizer_id

        # Safeguard: if variant small/base but context >512, clamp
        if variant in ("small", "base") and self.max_context > 512:
            self.max_context = 512

        # Delayed imports after path adjustment
        from model import Kronos, KronosTokenizer, KronosPredictor  # type: ignore

        self.tokenizer = KronosTokenizer.from_pretrained(self.tokenizer_id)
        self.model = Kronos.from_pretrained(self.model_id)
        self.predictor = KronosPredictor(
            self.model,
            self.tokenizer,
            device=self.device,
            max_context=self.max_context,
        )

    @staticmethod
    def _gpu_available() -> bool:  # pragma: no cover - runtime environment dependent
        try:
            import torch

            return torch.cuda.is_available()
        except Exception:
            return False

    def predict_next_days(self, data: pd.DataFrame, days: int = 5, granularity: str = "day", lookback_days: int = None) -> List[float]:
        """Predict future prices for the next N days.

        Args:
            data: Historical OHLCV data with columns: open, high, low, close, volume, amount
            days: Number of future periods to predict
            granularity: Prediction granularity ('day' only for now, extensible)
            lookback_days: Number of historical days to use for prediction. If None, uses all available data within model context limit.

        Returns:
            List of predicted close prices
        """
        if data is None or data.empty:
            raise ValueError("No data provided for prediction")
        if days <= 0:
            raise ValueError("days must be > 0")
        if granularity != "day":
            raise ValueError("Only 'day' granularity is supported currently")

        # Ensure data has required columns in lowercase
        required_cols = ['open', 'high', 'low', 'close', 'volume', 'amount']
        if not all(col in data.columns for col in required_cols):
            raise ValueError(f"Data must contain columns: {required_cols}")

        df = data.copy()
        df.index = pd.to_datetime(df.index) if not isinstance(df.index, pd.DatetimeIndex) else df.index

        # Ensure timestamps are timezone-naive (remove timezone info if present)
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)

        # Add missing optional columns if needed
        if "amount" not in df.columns:
            df["amount"] = 0.0

        # Determine lookback window
        if lookback_days is not None:
            # User specified lookback days, but still respect model context limit
            lookback = min(len(df), lookback_days, self.max_context)
        else:
            # Default behavior: use all available data within model context limit
            lookback = min(len(df), self.max_context)
        
        x_df = df.tail(lookback)

        # Historical timestamps - ensure they are timezone-naive
        x_timestamp = pd.Series(x_df.index)
        if x_timestamp.dt.tz is not None:
            x_timestamp = x_timestamp.dt.tz_localize(None)

        last_ts = x_timestamp.iloc[-1]

        # Generate future daily timestamps
        future_ts = []
        cursor = last_ts
        while len(future_ts) < days:
            cursor = cursor + timedelta(days=1)
            future_ts.append(cursor)
        y_timestamp = pd.Series(future_ts)

        # Debug: ensure y_timestamp is also timezone-naive
        if hasattr(y_timestamp, 'dt') and y_timestamp.dt.tz is not None:
            y_timestamp = y_timestamp.dt.tz_localize(None)

        # Debug: print timestamp info
        print(f"DEBUG: x_timestamp tz: {x_timestamp.dt.tz}")
        print(f"DEBUG: y_timestamp tz: {y_timestamp.dt.tz if hasattr(y_timestamp, 'dt') else 'No dt attr'}")
        print(f"DEBUG: x_timestamp sample: {x_timestamp.head()}")
        print(f"DEBUG: y_timestamp sample: {y_timestamp.head()}")

        # Run Kronos forecasting
        pred_df = self.predictor.predict(
            df=x_df[required_cols],
            x_timestamp=x_timestamp,
            y_timestamp=y_timestamp,
            pred_len=days,
            T=1.0,
            top_p=0.9,
            sample_count=1,
        )

        # Return list of close prices
        return pred_df["close"].tolist()

    @staticmethod
    def load_data_from_csv(csv_path):
        """Load stock data from CSV file and prepare for Kronos prediction."""
        df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        
        # Convert index to DatetimeIndex and handle timezone
        df.index = pd.to_datetime(df.index, utc=True)
        df.index = df.index.tz_localize(None)  # Remove timezone info
        
        # Ensure required columns exist
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"CSV must contain columns: {required_cols}")
        
        # Rename columns to match Kronos expectations if needed
        # Note: CSV has both 'volume' and 'amount' columns, we use 'volume'
        
        return df


# Backward compatible alias
class KronosMiniPredictor(KronosModelPredictor):
    def __init__(self, **kwargs):  # type: ignore[override]
        if "model_variant" not in kwargs and "model_id" not in kwargs:
            kwargs["model_variant"] = "mini"
        super().__init__(**kwargs)


__all__ = [
    "KronosModelPredictor",
    "KronosMiniPredictor",
    "KronosNotAvailableError",
    "VARIANT_PRESETS",
]