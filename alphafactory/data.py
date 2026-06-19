"""Data loading, cleaning and resampling.

Reads the MetaTrader-style M1 tab-separated exports, cleans them, resamples to a
working timeframe (H1 by default) and caches the result to disk so the heavy CSV
parse only happens once.

Cleaning is explicit (the assignment requires it):
  * duplicate timestamps are dropped (keep first);
  * rows are sorted chronologically;
  * rows with non-positive prices or high < low are discarded as anomalies;
  * after resampling, bars covering hours with no underlying ticks are dropped
    rather than forward-filled, so we never invent flat (fake) returns across
    weekends / market closures.  Returns are therefore computed across the
    *available* bars, which keeps real gap risk in the series.

The <SPREAD> column is the realized broker spread in MetaTrader *points*.  It is
converted to a fraction of price using the per-instrument tick size and stored on
each H1 bar (mean spread over the hour) so trades can be charged the realized,
time-varying spread rather than a flat assumption.
"""
from __future__ import annotations

import glob
import hashlib
import os
import pickle
from dataclasses import dataclass

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Instrument metadata.  Tick size from the assignment:
#   SPXUSD = 0.01 ; USDJPY, XAUUSD, ETHUSD = 0.001
# `asset_class` is used only for reporting / the flat-cost fallback table.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Instrument:
    symbol: str
    tick_size: float
    asset_class: str
    file_glob: str


INSTRUMENTS: dict[str, Instrument] = {
    "SPXUSD": Instrument("SPXUSD", 0.01, "index", "SPXUSD_*.csv"),
    "USDJPY": Instrument("USDJPY", 0.001, "fx", "USDJPY*_*.csv"),
    "XAUUSD": Instrument("XAUUSD", 0.001, "gold", "XAUUSD*_*.csv"),
    "ETHUSD": Instrument("ETHUSD", 0.001, "crypto", "ETHUSD_*.csv"),
}

_COLS = ["<DATE>", "<TIME>", "<OPEN>", "<HIGH>", "<LOW>", "<CLOSE>",
         "<TICKVOL>", "<VOL>", "<SPREAD>"]


def _find_csv(data_dir: str, instr: Instrument) -> str:
    matches = sorted(glob.glob(os.path.join(data_dir, instr.file_glob)))
    if not matches:
        raise FileNotFoundError(
            f"No CSV for {instr.symbol} matching {instr.file_glob} in {data_dir}")
    return matches[0]


def _read_m1(path: str) -> pd.DataFrame:
    """Read one raw M1 CSV into a tidy DataFrame indexed by timestamp.

    Memory-conscious: only the columns we need are read, with compact dtypes,
    and the two string date/time columns are dropped immediately after the
    timestamp is built.
    """
    df = pd.read_csv(
        path,
        sep="\t",
        usecols=["<DATE>", "<TIME>", "<OPEN>", "<HIGH>", "<LOW>", "<CLOSE>",
                 "<TICKVOL>", "<SPREAD>"],
        dtype={
            "<DATE>": "string", "<TIME>": "string",
            "<OPEN>": "float64", "<HIGH>": "float64",
            "<LOW>": "float64", "<CLOSE>": "float64",
            "<TICKVOL>": "int64", "<SPREAD>": "int64",
        },
    )
    dt = pd.to_datetime(df["<DATE>"] + " " + df["<TIME>"],
                        format="%Y.%m.%d %H:%M:%S")
    out = pd.DataFrame({
        "open": df["<OPEN>"].to_numpy(),
        "high": df["<HIGH>"].to_numpy(),
        "low": df["<LOW>"].to_numpy(),
        "close": df["<CLOSE>"].to_numpy(),
        "tickvol": df["<TICKVOL>"].to_numpy(),
        "spread": df["<SPREAD>"].to_numpy(),
    }, index=pd.DatetimeIndex(dt, name="dt"))
    del df
    return out


def _clean_m1(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Explicit cleaning. Returns the cleaned frame plus a diagnostics dict."""
    n0 = len(df)
    df = df.sort_index()
    dup = df.index.duplicated(keep="first")
    n_dup = int(dup.sum())
    df = df[~dup]
    bad = (df[["open", "high", "low", "close"]] <= 0).any(axis=1) | (df["high"] < df["low"])
    n_bad = int(bad.sum())
    df = df[~bad]
    diag = {"raw_rows": n0, "dropped_duplicates": n_dup, "dropped_anomalies": n_bad,
            "clean_rows": len(df)}
    return df, diag


def _resample_h1(df: pd.DataFrame, instr: Instrument) -> pd.DataFrame:
    """M1 -> H1 OHLC, dropping hours with no ticks. Adds spread_frac per bar."""
    agg = df.resample("1h").agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        spread_pts=("spread", "mean"),
        ticks=("tickvol", "sum"),
    )
    agg = agg.dropna(subset=["close"])          # drop empty (no-trade) hours
    agg = agg[agg["ticks"] > 0]                  # belt and braces
    # realized spread as a fraction of price
    agg["spread_frac"] = (agg["spread_pts"] * instr.tick_size / agg["close"]).clip(lower=0.0)
    agg["ret"] = agg["close"].pct_change()       # close-to-close simple return
    agg = agg.dropna(subset=["ret"])
    agg.attrs["symbol"] = instr.symbol
    agg.attrs["tick_size"] = instr.tick_size
    agg.attrs["asset_class"] = instr.asset_class
    return agg


def _cache_key(path: str, timeframe: str) -> str:
    st = os.stat(path)
    raw = f"{path}|{st.st_size}|{int(st.st_mtime)}|{timeframe}"
    return hashlib.md5(raw.encode()).hexdigest()[:16]


def load_symbol(symbol: str, data_dir: str, cache_dir: str,
                timeframe: str = "1h", use_cache: bool = True) -> pd.DataFrame:
    """Load one symbol resampled to `timeframe`, using the disk cache if fresh."""
    instr = INSTRUMENTS[symbol]
    path = _find_csv(data_dir, instr)
    key = _cache_key(path, timeframe)
    cache_file = os.path.join(cache_dir, f"{symbol}_{timeframe}_{key}.pkl")
    if use_cache and os.path.exists(cache_file):
        with open(cache_file, "rb") as fh:
            return pickle.load(fh)

    raw = _read_m1(path)
    raw, diag = _clean_m1(raw)
    h = _resample_h1(raw, instr)
    h.attrs["diagnostics"] = diag
    span_days = (h.index[-1] - h.index[0]).days or 1
    h.attrs["bars_per_year"] = len(h) / (span_days / 365.25)
    os.makedirs(cache_dir, exist_ok=True)
    with open(cache_file, "wb") as fh:
        pickle.dump(h, fh, protocol=pickle.HIGHEST_PROTOCOL)
    return h


def load_all(data_dir: str, cache_dir: str, timeframe: str = "1h",
             symbols: list[str] | None = None,
             use_cache: bool = True) -> dict[str, pd.DataFrame]:
    symbols = symbols or list(INSTRUMENTS.keys())
    return {s: load_symbol(s, data_dir, cache_dir, timeframe, use_cache) for s in symbols}


def bars_per_year(df: pd.DataFrame) -> float:
    return float(df.attrs.get("bars_per_year", 252 * 24))
