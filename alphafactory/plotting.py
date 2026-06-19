"""Figures for the research report.  All output is written to results/figures.

Kept deliberately small and dependency-light (matplotlib only).  Every figure
is regenerated from the same arrays the gate consumes, so nothing here can
drift from the reported numbers.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from . import stats as stats_mod  # noqa: E402
from .gate import common  # noqa: E402

plt.rcParams.update({
    "figure.dpi": 130,
    "font.size": 9,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

C_FAM_A = "#1f6feb"
C_FAM_B = "#d1495b"
C_FAM_C = "#2e8b57"
_FAM_COLORS = {"voltarget_mom": C_FAM_A, "bollinger_revert": C_FAM_B}
C_PASS = "#2e7d32"
C_FAIL = "#9e9e9e"


def _ensure(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)


def plot_funnel(waterfall: pd.DataFrame, n_total: int, path: str) -> str:
    _ensure(path)
    fig, ax = plt.subplots(figsize=(7.0, 3.2))
    stages = ["Population"] + list(waterfall["layer"])
    counts = [n_total] + list(waterfall["out"])
    bars = ax.bar(stages, counts, color=["#444"] + [C_FAM_A] * len(waterfall))
    for b, c in zip(bars, counts):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.4,
                str(c), ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("candidates surviving")
    ax.set_title("Survival funnel (intersection cascade)")
    ax.set_ylim(0, max(counts) * 1.18 + 1)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_oos_distribution(layer1: pd.DataFrame, cfg: dict, path: str) -> str:
    _ensure(path)
    c = cfg["layer1"]
    fig, ax = plt.subplots(figsize=(7.0, 3.2))
    sh = layer1["oos_sharpe"].to_numpy(float)
    ax.hist(sh, bins=18, color=C_FAM_A, alpha=0.8, edgecolor="white")
    ax.axvline(c["min_oos_sharpe"], color=C_PASS, lw=1.8, ls="--",
               label=f"L1 OoS Sharpe gate = {c['min_oos_sharpe']}")
    ax.axvline(0.0, color="#888", lw=1.0)
    ax.set_xlabel("annualized out-of-sample Sharpe")
    ax.set_ylabel("number of instances")
    ax.set_title("Layer 1: out-of-sample Sharpe across the full population")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_dsr_scatter(layer4: pd.DataFrame, cfg: dict, path: str) -> str:
    _ensure(path)
    c = cfg["layer4"]
    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    passed = layer4["passed"].to_numpy(bool)
    x = layer4["sharpe"].to_numpy(float)
    y = layer4["dsr"].to_numpy(float)
    ax.scatter(x[~passed], y[~passed], s=28, c=C_FAIL, alpha=0.8,
               label="rejected", edgecolor="white", linewidth=0.4)
    ax.scatter(x[passed], y[passed], s=42, c=C_PASS, alpha=0.95,
               label="survived L4", edgecolor="white", linewidth=0.5)
    ax.axhline(c["min_dsr"], color=C_PASS, lw=1.8, ls="--",
               label=f"DSR gate = {c['min_dsr']}")
    ax.set_xlabel("annualized Sharpe (full sample)")
    ax.set_ylabel("Deflated Sharpe Ratio")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(f"Layer 4: Deflated Sharpe deflates {len(layer4)} parallel trials")
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_equity_curves(pop, data: dict, ids: list[str], path: str,
                       cost_model: str = "realized_spread") -> str:
    _ensure(path)
    by_id = {inst.id: inst for inst in pop}
    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    for i, iid in enumerate(ids):
        inst = by_id.get(iid)
        if inst is None:
            continue
        df = data[inst.symbol]
        net = common.full_net_returns(df, inst.family, inst.params, cost_model)
        eq = np.cumprod(1.0 + net)
        color = _FAM_COLORS.get(inst.family, "#888888")
        ax.plot(df.index, eq, lw=1.1, alpha=0.9,
                color=color, label=iid.replace("|", "  "))
    ax.axhline(1.0, color="#888", lw=0.8)
    ax.set_yscale("log")
    ax.set_ylabel("growth of 1 unit (log, net of costs)")
    ax.set_title("Representative equity curves (full sample, after costs)")
    ax.legend(frameon=False, fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_per_layer_bars(per_layer: pd.DataFrame, n_total: int, path: str) -> str:
    _ensure(path)
    fig, ax = plt.subplots(figsize=(7.0, 3.2))
    layers = list(per_layer["layer"])
    passed = list(per_layer["passed"])
    failed = list(per_layer["failed"])
    ax.bar(layers, passed, color=C_PASS, label="pass (independent)")
    ax.bar(layers, failed, bottom=passed, color=C_FAIL, label="fail")
    for i, p in enumerate(passed):
        ax.text(i, n_total + 0.4, f"{p}/{n_total}", ha="center",
                va="bottom", fontsize=8)
    ax.set_ylabel("candidates")
    ax.set_title("Per-layer pass counts (each layer judged on full population)")
    ax.set_ylim(0, n_total * 1.12 + 1)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path
