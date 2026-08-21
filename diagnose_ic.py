# -*- coding: utf-8 -*-
"""
IC 悖论诊断
===========
现象：Alpha158 的 RankIC = +0.0315（正），组合却跑输同池随机 23.7pp/年。

正 IC 配负组合收益，通常只有三个来源。三项检验都在 explore 区间内完成，
不碰 OOS：

  D1  期限衰减   —— IC 衡量 1 日预测力（qlib 默认 label 是 T+1→T+2 收益），
                    而实际平均持仓 12.9 天。若信号 3-5 天就衰减完，后面就是裸奔。
  D2  分组单调性 —— IC 是全截面平均相关性；只取 top 5% 的极端尾部，
                    截面上的正相关在尾部可能反号。
  D3  一字板逆选 —— 得分最高的股票次日常一字涨停买不进，
                    实际只成交了没涨起来的那批，系统性拉低 top 组收益。

用法：
    python diagnose_ic.py --score data/cache/qlib_score.parquet
    python diagnose_ic.py --score ... --panel data/cache/panel_2021_2026.parquet
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from pullback_momentum import Config, build_panel

HORIZONS = [1, 2, 3, 5, 10, 20, 30]
N_GROUPS = 10


# ============================================================
# 载入
# ============================================================

def load_panel(panel_path: Path | None, score: pd.DataFrame, cfg: Config) -> dict:
    if panel_path and panel_path.exists():
        df = pd.read_parquet(panel_path)
    else:
        from market_adapter import load_market
        df = load_market("20210101", "20260814",
                         cache="data/cache/panel_2021_2026.parquet")
    s0, s1 = score.index.min(), score.index.max()
    df = df[(df["date"] >= s0) & (df["date"] <= s1)]
    print(f"面板 {s0:%Y-%m-%d} → {s1:%Y-%m-%d}   {len(df):,} 行")
    return build_panel(df, cfg)


def _fwd_ret(close: pd.DataFrame, open_: pd.DataFrame, h: int) -> pd.DataFrame:
    """
    T 日信号 → T+1 开盘买入 → T+1+h 开盘卖出的前瞻收益。
    刻意用开盘价而非收盘价，与回测引擎的成交口径一致；
    用收盘口径算出来的 IC 会比实际可实现的高。
    """
    return open_.shift(-(1 + h)) / open_.shift(-1) - 1


# ============================================================
# D1 期限衰减
# ============================================================

def d1_decay(score: pd.DataFrame, px: dict) -> pd.DataFrame:
    close, open_ = px["close"], px["open"]
    tradable = px["tradable"]
    sc = score.where(tradable)

    rows = []
    for h in HORIZONS:
        fwd = _fwd_ret(close, open_, h).where(tradable)
        ic, ric = [], []
        for d in sc.index:
            a, b = sc.loc[d], fwd.loc[d]
            m = a.notna() & b.notna()
            if m.sum() < 50:
                continue
            ic.append(a[m].corr(b[m]))
            ric.append(a[m].corr(b[m], method="spearman"))
        ic, ric = np.array(ic), np.array(ric)
        if len(ric) == 0:
            continue
        rows.append({
            "持有日": h,
            "IC": f"{ic.mean():+.4f}",
            "RankIC": f"{ric.mean():+.4f}",
            "ICIR": f"{ric.mean() / ric.std():+.2f}" if ric.std() else "—",
            "IC>0占比": f"{(ric > 0).mean():.1%}",
            "样本日": len(ric),
        })
    return pd.DataFrame(rows)


# ============================================================
# D2 分组单调性
# ============================================================

def d2_groups(score: pd.DataFrame, px: dict, h: int = 10) -> pd.DataFrame:
    close, open_ = px["close"], px["open"]
    tradable = px["tradable"]
    sc = score.where(tradable)
    fwd = _fwd_ret(close, open_, h).where(tradable)

    q = sc.rank(axis=1, pct=True)
    rows = []
    for g in range(N_GROUPS):
        lo, hi = g / N_GROUPS, (g + 1) / N_GROUPS
        m = (q > lo) & (q <= hi) if g else (q >= 0) & (q <= hi)
        r = fwd.where(m)
        daily = r.mean(axis=1).dropna()
        rows.append({
            "分组": f"D{g + 1}" + ("(最低)" if g == 0 else "(最高)" if g == N_GROUPS - 1 else ""),
            f"{h}日均收益": f"{daily.mean():+.3%}",
            "胜率": f"{(daily > 0).mean():.1%}",
            "年化": f"{(1 + daily.mean()) ** (242 / h) - 1:+.2%}",
        })
    return pd.DataFrame(rows)


# ============================================================
# D3 一字板逆向选择
# ============================================================

def d3_limit_selection(score: pd.DataFrame, px: dict,
                       top_pct: float = 0.05, h: int = 10) -> None:
    close, open_ = px["close"], px["open"]
    tradable, can_buy = px["tradable"], px["can_buy"]
    fwd = _fwd_ret(close, open_, h)

    top = (score.where(tradable).rank(axis=1, pct=True) >= 1 - top_pct)
    # T 日选中 → T+1 能否买入
    buyable = can_buy.shift(-1).fillna(False)

    filled = top & buyable
    blocked = top & ~buyable

    n_f, n_b = int(filled.values.sum()), int(blocked.values.sum())
    r_f = fwd.where(filled).stack().mean()
    r_b = fwd.where(blocked).stack().mean()

    print(f"  top{top_pct:.0%} 信号总数     {n_f + n_b:,}")
    print(f"  次日可成交            {n_f:,} ({n_f / max(n_f + n_b, 1):.1%})   "
          f"{h}日收益 {r_f:+.3%}")
    print(f"  次日被挡（一字/停牌） {n_b:,} ({n_b / max(n_f + n_b, 1):.1%})   "
          f"{h}日收益 {r_b:+.3%}")
    if n_b and np.isfinite(r_b) and np.isfinite(r_f):
        gap = r_b - r_f
        print(f"  逆向选择损失          {gap:+.3%} / {h}日"
              f"  → 年化 {(1 + gap) ** (242 / h) - 1:+.2%}")
        if gap > 0.005:
            print("  ⚠️  被挡掉的信号显著更赚 —— 模型的预测力有相当部分"
                  "落在买不进的股票上，理论 IC 无法兑现。")
        else:
            print("  ✅ 成交与被挡两组收益接近，逆向选择不是主因。")


# ============================================================
# D5 波动率分层 —— alpha 是否寄生在高波动上
# ============================================================

VOL_BUCKETS = [(0.0, 1 / 3, "低波动"), (1 / 3, 2 / 3, "中波动"), (2 / 3, 1.0, "高波动")]


def _atr_pct(px: dict, n: int = 14) -> pd.DataFrame:
    """ATR / 收盘价，与 backtest 里 k_stop × ATR 的口径一致。"""
    from pullback_momentum import _atr
    return _atr(px["high"], px["low"], px["close"], n) / px["close"]


def d5_vol_conditional(score: pd.DataFrame, px: dict, h: int = 10,
                       n_groups: int = 5) -> None:
    """
    D4 反推出模型选中标的的 ATR/价格 ≈ 5.17%，约为市场 1.8 倍。
    直接加波动率上限有个陷阱：2021-2023 的高动量 ≡ 赛道股 ≡ 高波动，
    过滤高波动可能把 alpha 一起过滤掉。

    在每个波动率桶【内部】重跑分组，看 alpha 是否依赖高波动。

    ⚠️ 判据必须用 RankIC，不能用 Q5-Q1 原始收益差：高波动票的收益离散度
    天然更大，即使预测能力完全相同，原始价差也必然更宽 —— 那个指标
    结构性地偏向「高波动更有 alpha」。已用合成数据验证过这一点。
    表里同时给出 Q5-Q1/σ（用桶内收益标准差归一）作为参照。
    """
    close, open_ = px["close"], px["open"]
    tradable = px["tradable"]
    ap = _atr_pct(px).where(tradable)
    sc = score.where(tradable)
    fwd = _fwd_ret(close, open_, h).where(tradable)

    vq = ap.rank(axis=1, pct=True)          # 每日截面上的波动率分位

    print(f"  全市场 ATR/价格 中位数 {ap.stack().median():.2%}")
    rows, rics = [], []
    for lo, hi, name in VOL_BUCKETS:
        inb = (vq > lo) & (vq <= hi) if lo else (vq >= 0) & (vq <= hi)
        # 桶内重新做截面分位，否则高分组会被波动率本身主导
        q = sc.where(inb).rank(axis=1, pct=True)

        vals = []
        for g in range(n_groups):
            a, b = g / n_groups, (g + 1) / n_groups
            m = ((q > a) & (q <= b)) if g else ((q >= 0) & (q <= b))
            vals.append(fwd.where(m).mean(axis=1).dropna().mean())

        ric = []
        for d in sc.index:
            aa, bb = sc.loc[d].where(inb.loc[d]), fwd.loc[d].where(inb.loc[d])
            m2 = aa.notna() & bb.notna()
            if m2.sum() >= 30:
                ric.append(aa[m2].corr(bb[m2], method="spearman"))
        ric_m = float(np.mean(ric)) if ric else np.nan
        rics.append(ric_m)

        sd = float(fwd.where(inb).stack().std())
        spread = vals[-1] - vals[0]
        row = {"波动桶": name, "ATR/价格": f"{ap.where(inb).stack().median():.2%}"}
        for g, v in enumerate(vals):
            row[f"Q{g + 1}"] = f"{v:+.2%}"
        row["Q5-Q1"] = f"{spread:+.2%}"
        row["Q5-Q1/σ"] = f"{spread / sd:+.3f}" if sd else "—"
        row["RankIC"] = f"{ric_m:+.4f}" if np.isfinite(ric_m) else "—"
        row["单调"] = "✓" if all(vals[i] <= vals[i + 1] + 1e-5
                                for i in range(len(vals) - 1)) else "✗"
        rows.append(row)

    print("\n" + pd.DataFrame(rows).to_string(index=False))
    print(f"\n  Q1=桶内最低分，Q{n_groups}=桶内最高分；{h} 日持有，毛收益等权。")
    print("  ⚠️ 跨桶比较看 RankIC 与 Q5-Q1/σ，不要看 Q5-Q1 原始值。")

    lo_ic, mid_ic, hi_ic = rics
    print("\n  " + "-" * 58)
    if not np.isfinite(lo_ic) or not np.isfinite(hi_ic):
        print("  → 样本不足，无法判定。")
    elif lo_ic > 0.01 and lo_ic >= hi_ic * 0.5:
        print(f"  → 低波动桶 RankIC {lo_ic:+.4f} ≈ 高波动桶 {hi_ic:+.4f}，")
        print("     alpha 不依赖高波动。可以加波动率上限（如 ATR/价格 ≤ 中位数×1.3），")
        print("     预期只削掉尾部亏损、不伤 alpha。")
    elif lo_ic <= 0.01 or hi_ic > lo_ic * 3:
        print(f"  → 低波动桶 RankIC {lo_ic:+.4f} 远弱于高波动桶 {hi_ic:+.4f}，")
        print("     alpha 主要寄生在高波动上。直接过滤会把收益一起过滤掉。")
        print("     改走波动率加权仓位：仓位 ∝ 1/ATR，等化风险预算而非剔除标的。")
    else:
        print(f"  → 各桶 RankIC {lo_ic:+.4f} / {mid_ic:+.4f} / {hi_ic:+.4f} 梯度温和，")
        print("     波动率是部分可分离的维度，过滤有代价但不致命。")
        print("     建议先做温和上限（剔除最高 1/6），再看 alpha 损失多少。")
    print("  " + "-" * 58)


# ============================================================
# D4 止损法证 —— 设计宽度 vs 实际成交
# ============================================================

def d4_stop_forensics(trades: pd.DataFrame) -> None:
    """
    回答一个决定下一步方向的问题：止损平均亏 -18.6%，是
      (a) 跌停顺延导致跳空穿透  → 病在选股，要在入场端排除崩塌风险票
      (b) k_stop × ATR 本身就宽 → 病在参数，给 stop_pct 加绝对上限即可
    两者修法完全不同，不能靠猜。
    """
    if not len(trades):
        print("  无成交记录。")
        return
    st = trades[trades["reason"] == "stop_loss"]
    if not len(st):
        print("  无止损出场记录。")
        return

    des = st["stop_pct"]                  # 设计止损宽度（正数）
    act = -st["ret"]                      # 实际亏损（转成正数）
    pierce = act - des                    # 穿透幅度：>0 表示亏得比设计更多

    print(f"  止损单 {len(st)} 笔 / 总成交 {len(trades)} 笔 "
          f"({len(st) / len(trades):.1%})")
    q = [.1, .5, .9]
    tbl = pd.DataFrame({
        "设计宽度": des.quantile(q).values,
        "实际亏损": act.quantile(q).values,
        "穿透幅度": pierce.quantile(q).values,
    }, index=[f"P{int(x * 100)}" for x in q])
    print("\n" + tbl.map("{:.2%}".format).to_string())
    print(f"\n  均值：设计 {des.mean():.2%} / 实际 {act.mean():.2%} / "
          f"穿透 {pierce.mean():+.2%}")

    # ---- 顺延分布：跌停不可卖会把成交推后 ----
    if "defer_days" in st.columns:
        dd = st["defer_days"]
        print(f"\n  卖出顺延天数（0 = 次日正常成交）")
        for k in [0, 1, 2, 3]:
            m = (dd == k) if k < 3 else (dd >= 3)
            lbl = f"{k}天" if k < 3 else "≥3天"
            if m.sum():
                print(f"    {lbl:<5} {m.sum():>4} 笔 ({m.mean():>5.1%})   "
                      f"实际亏损 {act[m].mean():.2%}   穿透 {pierce[m].mean():+.2%}")
        n_def = int((dd > 0).sum())
        print(f"    顺延占比 {n_def / len(st):.1%}")

    if "slip_to_fill" in st.columns:
        sl = st["slip_to_fill"].dropna()
        if len(sl):
            print(f"\n  信号日收盘 → 成交价 滑移：均值 {sl.mean():+.2%}，"
                  f"P10 {sl.quantile(.1):+.2%}")

    # ---- 判定 ----
    print("\n  " + "-" * 58)
    if des.median() > 0.13:
        print("  → (b) 设计宽度本身就宽。病在参数，不在选股。")
        print("     修法：给止损加绝对上限，如 min(k_stop×ATR, 10%)；")
        print("     或直接下调 k_stop。先跑 sweep k_stop 看平原。")
    elif pierce.mean() > 0.05:
        print("  → (a) 确认跳空穿透。设计宽度合理，但实际亏损远超设计。")
        if "defer_days" in st.columns and (st["defer_days"] > 0).mean() > 0.2:
            print("     顺延占比高 ⇒ 连续跌停不可卖是主因。")
        print("     这不是执行层缺陷 —— 回测正确模拟了现实，跌停就是卖不掉。")
        print("     修法在【入场端】排除崩塌风险票：波动率上限、")
        print("     剔除近 N 日出现过跌停的、提高 amount 流动性门槛。")
    else:
        print("  → 设计与实际接近，止损不是主导项。回头看 timeout 段。")
    print("  " + "-" * 58)


def run_with_backtest(score_path: Path, panel_path: Path, top_pct: float) -> None:
    from run_backtest import score_to_signal_backtest
    from market_adapter import load_market

    df = (pd.read_parquet(panel_path) if panel_path.exists()
          else load_market("20210101", "20260814", cache=str(panel_path)))
    res = score_to_signal_backtest(df, score_path, top_pct=top_pct)
    print("\n" + pd.Series(res["stats"]).to_string())

    tr = res["trades"]
    print("\n--- 出场拆解 ---")
    print(tr.groupby("reason")["ret"].agg(
        次数="count", 平均收益="mean", 胜率=lambda x: (x > 0).mean()
    ).sort_values("次数", ascending=False).to_string())

    print("\n" + "=" * 66)
    print("D4  止损法证 —— 设计宽度 vs 实际成交")
    print("=" * 66)
    d4_stop_forensics(tr)


# ============================================================

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--score", type=Path, default=Path("data/cache/qlib_score.parquet"))
    ap.add_argument("--panel", type=Path, default=Path("data/cache/panel_2021_2026.parquet"))
    ap.add_argument("--hold", type=int, default=10, help="D2/D3 使用的持有期")
    ap.add_argument("--top-pct", type=float, default=0.05)
    ap.add_argument("--stops-only", action="store_true",
                    help="跳过 D1-D3，只跑 D4 止损法证（需要回测）")
    ap.add_argument("--vol-only", action="store_true",
                    help="只跑 D5 波动率分层，不做回测")
    args = ap.parse_args()

    if not args.score.exists():
        print(f"❌ 找不到 {args.score}（先跑 qlib_train.py --train）", file=sys.stderr)
        return 1

    if args.stops_only:
        run_with_backtest(args.score, args.panel, args.top_pct)
        return 0

    cfg = Config()
    score = pd.read_parquet(args.score)
    score.index = pd.to_datetime(score.index)
    px = load_panel(args.panel, score, cfg)
    score = score.reindex(index=px["close"].index, columns=px["close"].columns)

    if args.vol_only:
        print("=" * 66)
        print(f"D5  波动率分层（{args.hold} 日持有）")
        print("=" * 66)
        d5_vol_conditional(score, px, args.hold)
        return 0

    print("\n" + "=" * 66)
    print("D1  IC 期限衰减  —— 决定持仓周期该多长")
    print("=" * 66)
    d1 = d1_decay(score, px)
    print(d1.to_string(index=False))
    print("\n判据：RankIC 若在 3-5 日内掉到 0.01 以下，说明 12.9 日的持仓")
    print("      大部分时间在裸奔 → 压 max_hold，或改用长周期 label 重训。")

    print("\n" + "=" * 66)
    print(f"D2  分组单调性（{args.hold} 日持有） —— top 组是否尾部反号")
    print("=" * 66)
    print(d2_groups(score, px, args.hold).to_string(index=False))
    print("\n判据：D1→D10 应单调递增。若 D10 低于 D8/D9，说明极端高分组")
    print("      是高波动/崩塌风险名字，正 IC 在尾部失效 → 别取 top5%，改中高分位。")

    print("\n" + "=" * 66)
    print(f"D3  一字板逆向选择（{args.hold} 日持有）")
    print("=" * 66)
    d3_limit_selection(score, px, args.top_pct, args.hold)

    print("\n" + "=" * 66)
    print(f"D5  波动率分层（{args.hold} 日持有） —— alpha 是否寄生在高波动上")
    print("=" * 66)
    d5_vol_conditional(score, px, args.hold)

    run_with_backtest(args.score, args.panel, args.top_pct)
    return 0


if __name__ == "__main__":
    sys.exit(main())
