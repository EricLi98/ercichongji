"""回测运行脚本。

用法：
    python run_backtest.py explore     # 三基准对照（仅 explore 区间 2021-2023）
    python run_backtest.py oos         # 样本外验证（2024-2026）
    python run_backtest.py full        # 全量（仅最终报告，不调参）
    python run_backtest.py sweep       # 参数平原扫描（只在 explore 区间跑）
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

# ============================================================
# 参数平原
# ============================================================

_PANEL_PARAMS = {"min_amount", "min_list_days", "exclude_st"}   # 影响 build_panel 的参数

_SWEEP_GRID = [
    # 风控层优先：上一轮 explore 显示成本拖累 12.25%/年，
    # 止损宽度与持有周期决定换手，是当前的主要矛盾。
    ("k_stop",        [1.5, 2.0, 2.5, 3.0, 3.5, 4.0]),
    ("k_trail",       [3.0, 4.0, 5.0, 6.0, 8.0]),
    ("max_hold",      [15, 20, 30, 45, 60]),
    ("exit_below_ma", [True, False]),
    # 形态层
    ("pb_max_atr",    [2.0, 3.0, 4.0, 5.0, 6.0]),
    ("pb_days_max",   [4, 5, 6, 7, 8, 10]),
]


def sweep(df: pd.DataFrame, base_cfg: Config, param: str, values: list) -> pd.DataFrame:
    """
    单参数扫描。看的不是最高点，是**平原** ——
    相邻取值应连续变化；某一点突然跳高是噪音，不是参数。
    额外输出每槽年周转与纯成本拖累，用来判断收益是否被换手吃掉。
    """
    rebuild = param in _PANEL_PARAMS
    px0 = None if rebuild else build_panel(df, base_cfg)
    years = max((df["date"].max() - df["date"].min()).days / 365.25, 1e-9)
    rt = base_cfg.commission * 2 + base_cfg.stamp_tax + base_cfg.slippage * 2

    rows = []
    for v in values:
        cfg = Config(**{**base_cfg.__dict__, param: v})
        px = build_panel(df, cfg) if rebuild else px0
        sig, score = generate_signals(px, cfg)
        eq, tr = backtest(px, sig, score, cfg)
        st = performance(eq, tr)
        tpy = len(tr) / cfg.max_positions / years        # 每槽年周转
        rows.append({param: v, "年化": st["年化"], "最大回撤": st["最大回撤"],
                     "夏普": st["夏普"], "胜率": st["胜率"], "交易数": st["交易次数"],
                     "周转/年": f"{tpy:.1f}", "成本拖累": f"{-(1 - (1 - rt) ** tpy):.2%}"})
    return pd.DataFrame(rows)


def run_sweep(df: pd.DataFrame, cfg: Config | None = None) -> None:
    cfg = cfg or Config()
    for param, vals in _SWEEP_GRID:
        print(f"\n===== {param} =====")
        print(sweep(df, cfg, param, vals).to_string(index=False))
    print("\n判据：结果应随参数连续变化。单点跳高 = 噪音，不是参数。")


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

    elif mode == "sweep":
        df = df_all[(df_all["date"] >= EXPLORE_START) & (df_all["date"] <= EXPLORE_END)]
        print(f"[sweep] {df['date'].min():%Y-%m-%d} → {df['date'].max():%Y-%m-%d}  "
              f"仅在 explore 区间扫描\n")
        run_sweep(df)

    elif mode == "full":
        print(f"[full] {df_all['date'].min():%Y-%m-%d} → {df_all['date'].max():%Y-%m-%d}  "
              f"{len(df_all):,} 行 × {df_all['code'].nunique()} 只\n")
        explore(df_all)

    else:
        print(f"未知模式: {mode}  (可选: explore / oos / full / sweep)")


if __name__ == "__main__":
    main()
