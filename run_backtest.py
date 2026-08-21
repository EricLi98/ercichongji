"""回测运行脚本。

用法：
    python run_backtest.py explore     # 四腿对照：策略(exit_below_ma=False) vs 对照(True) vs 三基准
    python run_backtest.py sweep       # 参数平原扫描（TODO）
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from market_adapter import load_market
from pullback_momentum import Config, run, build_panel, generate_signals, backtest, performance


# ============================================================
# 基准
# ============================================================

def benchmark_strong_no_pullback(df: pd.DataFrame, cfg: Config) -> dict:
    """基准A：等权持有全部强势股（不等回调直接买）。"""
    cfg2 = Config(**{**cfg.__dict__, "exit_below_ma": False})
    px = build_panel(df, cfg2)
    close, high, low, vol = px["close"], px["high"], px["low"], px["volume"]
    open_ = px["open"]

    ma_s = close.rolling(cfg2.ma_support).mean()
    ma_l = close.rolling(cfg2.mom_window).mean()
    v5, v20 = vol.rolling(5).mean(), vol.rolling(20).mean()

    mom = close / close.shift(cfg2.mom_window) - 1
    mom_rank = mom.rank(axis=1, pct=True)
    roll_high = high.rolling(cfg2.mom_window).max()
    made_new_high = (high >= roll_high).rolling(cfg2.high_lookback).max().fillna(0).astype(bool)

    strong = (
        (mom_rank >= 1 - cfg2.top_pct)
        & (close > ma_l)
        & (ma_s > ma_l)
        & made_new_high
    )

    trigger = (
        (close > open_) & (close > close.shift(1))
        & ((vol > v5.shift(1) * cfg2.vol_up) | (close > high.shift(1)))
    )

    signal = (strong & trigger & px["tradable"] & ~px["limit_up"].fillna(False))
    signal = signal.fillna(False)

    eq, trades = backtest(px, signal, mom_rank, cfg2)
    return dict(
        equity=eq, trades=trades,
        stats=performance(eq, trades),
        label="基准A: 强势不等回调",
    )


def benchmark_random(df: pd.DataFrame, cfg: Config, seed: int = 42) -> dict:
    """基准C：同池随机选股。"""
    rng = np.random.RandomState(seed)
    px = build_panel(df, cfg)
    close = px["close"]
    tradable = px["tradable"]
    signal = pd.DataFrame(False, index=close.index, columns=close.columns)
    for t in range(len(close)):
        pool = tradable.iloc[t]
        avail = pool[pool].index.tolist()
        if len(avail) > cfg.max_positions:
            picks = rng.choice(avail, size=cfg.max_positions, replace=False)
        else:
            picks = avail
        signal.iloc[t][picks] = True
    score = pd.DataFrame(rng.rand(*close.shape), index=close.index, columns=close.columns)
    eq, trades = backtest(px, signal, score, cfg)
    return dict(
        equity=eq, trades=trades,
        stats=performance(eq, trades),
        label="基准C: 随机选股",
    )


# ============================================================
# explore 模式
# ============================================================

def explore(df: pd.DataFrame):
    """四腿对照。"""
    cfg_no_ma = Config(exit_below_ma=False)
    cfg_with_ma = Config(exit_below_ma=True)

    print("=" * 60)
    print("策略 (exit_below_ma=False)")
    print("=" * 60)
    res_no = run(df, cfg_no_ma)
    print(pd.Series(res_no["stats"]))
    print("\n出场拆解:")
    print(res_no["exits"])
    print(f"\n日均信号数: {res_no['signals_per_day'].mean():.1f}")

    print("\n" + "=" * 60)
    print("对照 (exit_below_ma=True)")
    print("=" * 60)
    res_yes = run(df, cfg_with_ma)
    print(pd.Series(res_yes["stats"]))
    print("\n出场拆解:")
    print(res_yes["exits"])
    print(f"\n日均信号数: {res_yes['signals_per_day'].mean():.1f}")

    print("\n" + "=" * 60)
    print("基准A: 强势不等回调")
    print("=" * 60)
    bench_a = benchmark_strong_no_pullback(df, cfg_no_ma)
    print(pd.Series(bench_a["stats"]))

    print("\n" + "=" * 60)
    print("基准C: 随机选股")
    print("=" * 60)
    bench_c = benchmark_random(df, cfg_no_ma)
    print(pd.Series(bench_c["stats"]))

    print("\n" + "=" * 60)
    print("四腿对照表")
    print("=" * 60)
    rows = [
        ("策略(exit_below_ma=False)", res_no["stats"]),
        ("对照(exit_below_ma=True)", res_yes["stats"]),
        ("基准A: 强势不等回调", bench_a["stats"]),
        ("基准C: 随机选股", bench_c["stats"]),
    ]
    summary = pd.DataFrame({label: stats for label, stats in rows}).T
    print(summary.to_string())


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "explore"
    cache = Path("data/cache/panel_2021_2026.parquet")
    df = load_market("20210101", "20260814", cache=str(cache))

    if mode == "explore":
        explore(df)
    else:
        print(f"未知模式: {mode}")


if __name__ == "__main__":
    main()
