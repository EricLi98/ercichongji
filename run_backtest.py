"""回测运行脚本。

用法：
    python run_backtest.py explore     # 三基准对照（仅 explore 区间 2021-2023）
    python run_backtest.py oos         # 样本外验证（2024-2026）
    python run_backtest.py full        # 全量（仅最终报告，不调参）
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
# 日期分界 — explore 只用前段，后段留给 OOS，不可逆
# ============================================================
EXPLORE_START, EXPLORE_END = "20210101", "20231231"
OOS_START, OOS_END         = "20240101", "20260814"
FULL_START, FULL_END       = "20210101", "20260814"


# ============================================================
# 基准（全部通过 generate_signals(layers=True) 取掩码，不重复定义）
# ============================================================

def benchmark_A_strong_no_pullback(df: pd.DataFrame, cfg: Config) -> dict:
    """基准A：强势 + 触发，去掉回调层。验证「等回调」有没有超额。"""
    px = build_panel(df, cfg)
    sig_full, mom_rank, layers = generate_signals(px, cfg, layers=True)
    # 去掉 pullback：strong & trigger & base
    signal = (layers["strong"] & layers["trigger"] & layers["base"]).fillna(False)
    eq, trades = backtest(px, signal, mom_rank, cfg)
    return dict(equity=eq, trades=trades, stats=performance(eq, trades),
                label="基准A: 强势不等回调")


def benchmark_B_pullback_no_strong(df: pd.DataFrame, cfg: Config) -> dict:
    """基准B：回调 + 触发，去掉趋势层。验证「强势」有没有贡献。"""
    px = build_panel(df, cfg)
    sig_full, mom_rank, layers = generate_signals(px, cfg, layers=True)
    # 去掉 strong：pullback & trigger & base
    signal = (layers["pullback"] & layers["trigger"] & layers["base"]).fillna(False)
    eq, trades = backtest(px, signal, mom_rank, cfg)
    return dict(equity=eq, trades=trades, stats=performance(eq, trades),
                label="基准B: 回调不要求强势")


def benchmark_C_random(df: pd.DataFrame, cfg: Config, seed: int = 42) -> dict:
    """基准C：同池随机选股，密度与策略对齐。"""
    rng = np.random.default_rng(seed)
    px = build_panel(df, cfg)
    base = px["tradable"].fillna(False)
    # 信号密度与策略对齐
    sig_ref, _ = generate_signals(px, cfg)
    dens = sig_ref.values.sum() / max(base.values.sum(), 1)
    signal = pd.DataFrame(rng.random(base.shape) < dens,
                          index=base.index, columns=base.columns) & base
    score = pd.DataFrame(rng.random(base.shape), index=base.index, columns=base.columns)
    eq, trades = backtest(px, signal, score, cfg)
    return dict(equity=eq, trades=trades, stats=performance(eq, trades),
                label="基准C: 同池随机")


# ============================================================
# explore 模式
# ============================================================

def explore(df: pd.DataFrame):
    """三腿对照 — 只回答一个问题：各层有没有贡献。"""
    cfg = Config()  # exit_below_ma 用默认值，True/False 留给 sweep

    print("=" * 60)
    print("完整策略")
    print("=" * 60)
    res = run(df, cfg)
    print(pd.Series(res["stats"]))
    print("\n出场拆解:")
    print(res["exits"])
    print(f"\n日均信号数: {res['signals_per_day'].mean():.1f}")

    print("\n" + "=" * 60)
    print("基准A: 强势不等回调")
    print("=" * 60)
    bench_a = benchmark_A_strong_no_pullback(df, cfg)
    print(pd.Series(bench_a["stats"]))

    print("\n" + "=" * 60)
    print("基准B: 回调不要求强势")
    print("=" * 60)
    bench_b = benchmark_B_pullback_no_strong(df, cfg)
    print(pd.Series(bench_b["stats"]))

    print("\n" + "=" * 60)
    print("基准C: 同池随机")
    print("=" * 60)
    bench_c = benchmark_C_random(df, cfg)
    print(pd.Series(bench_c["stats"]))

    print("\n" + "=" * 60)
    print("对照表")
    print("=" * 60)
    rows = [
        ("完整策略", res["stats"]),
        ("基准A: 强势不等回调", bench_a["stats"]),
        ("基准B: 回调不要求强势", bench_b["stats"]),
        ("基准C: 同池随机", bench_c["stats"]),
    ]
    summary = pd.DataFrame({label: s for label, s in rows}).T
    print(summary.to_string())


# ============================================================
# main
# ============================================================

def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "explore"

    # 全量缓存只加载一次
    cache = Path("data/cache/panel_2021_2026.parquet")
    df_all = load_market(FULL_START, FULL_END, cache=str(cache))

    if mode == "explore":
        df = df_all[(df_all["date"] >= EXPLORE_START) & (df_all["date"] <= EXPLORE_END)]
        print(f"[explore] {df['date'].min():%Y-%m-%d} → {df['date'].max():%Y-%m-%d}  "
              f"{len(df):,} 行 × {df['code'].nunique()} 只\n")
        explore(df)

    elif mode == "oos":
        df = df_all[(df_all["date"] >= OOS_START) & (df_all["date"] <= OOS_END)]
        print(f"[oos] {df['date'].min():%Y-%m-%d} → {df['date'].max():%Y-%m-%d}  "
              f"{len(df):,} 行 × {df['code'].nunique()} 只\n")
        explore(df)  # 同样的对照结构

    elif mode == "full":
        print(f"[full] {df_all['date'].min():%Y-%m-%d} → {df_all['date'].max():%Y-%m-%d}  "
              f"{len(df_all):,} 行 × {df_all['code'].nunique()} 只\n")
        explore(df_all)

    else:
        print(f"未知模式: {mode}  (可选: explore / oos / full / sweep)")


if __name__ == "__main__":
    main()
