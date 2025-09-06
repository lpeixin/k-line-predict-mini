"""Kronos-mini integration layer.

This module integrates the open-source Kronos-mini foundation model for
K-line (OHLCV) forecasting. If the Kronos repository (https://github.com/shiyu-coder/Kronos)
is not available in PYTHONPATH, a clear error is raised explaining how to
install it. The previous local linear regression implementation has been
removed in favour of the real Kronos-mini model.

Usage expectation:
1. Clone Kronos (one-time):
   git clone https://github.com/shiyu-coder/Kronos.git kronos_repo
2. Add the repo root to PYTHONPATH or set env KRONOS_REPO_PATH.
3. Ensure dependencies (torch >= 2.0 etc.) are installed.

We intentionally do NOT vendor Kronos code here to keep licensing & updates clean.
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


class KronosMiniPredictor:
    """Wrapper around Kronos-mini for next-N-day close price forecasting."""

    def __init__(self, device: str | None = None, max_context: int = 2048):
        _ensure_kronos_on_path()
        # Delayed imports until path is ensured
        from model import Kronos, KronosTokenizer, KronosPredictor  # type: ignore

        self.device = device or ("cuda:0" if self._gpu_available() else "cpu")
        self.max_context = max_context

        # Load tokenizer & model from Hugging Face Hub (mini + 2k tokenizer)
        self.tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-2k")
        self.model = Kronos.from_pretrained("NeoQuasar/Kronos-mini")
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

    def predict_next_days(self, data: pd.DataFrame, days: int = 5) -> List[float]:
        if data is None or data.empty:
            raise ValueError("No data provided for prediction")
        if days <= 0:
            raise ValueError("days must be > 0")

        # Kronos expects lower-case column names: open, high, low, close (volume optional)
        df = data.copy().rename(
            columns={
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume",
            }
        )

        # Add missing optional columns
        if "amount" not in df.columns:
            df["amount"] = 0.0

        # Determine lookback window (respect Kronos context limit)
        lookback = min(len(df), self.max_context)
        x_df = df.tail(lookback)

        # Historical timestamps
        x_timestamp = pd.Series(x_df.index)
        last_ts = x_timestamp.iloc[-1]

        # Generate future daily timestamps (simple calendar days; skipping weekends optional)
        future_ts = []
        cursor = last_ts
        while len(future_ts) < days:
            cursor = cursor + timedelta(days=1)
            future_ts.append(cursor)
        y_timestamp = pd.Series(future_ts)

        # Run Kronos forecasting
        pred_df = self.predictor.predict(
            df=x_df[["open", "high", "low", "close", "volume", "amount"]],
            x_timestamp=x_timestamp,
            y_timestamp=y_timestamp,
            pred_len=days,
            T=1.0,
            top_p=0.9,
            sample_count=1,
        )

        # Return list of close prices
        return pred_df["close"].tolist()


__all__ = ["KronosMiniPredictor", "KronosNotAvailableError"]