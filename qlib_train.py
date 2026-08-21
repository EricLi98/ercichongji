# -*- coding: utf-8 -*-
"""
Qlib Alpha158 + LightGBM —— 只产出打分，不做回测
================================================
架构决策：**Qlib 出 score，回测留在 pullback_momentum.backtest()**。

理由是 Qlib 回测引擎的执行约束不够：
  · limit_threshold 默认 None —— 涨跌停完全不限制；
    即便设成 0.095 也只是单一全局阈值，处理不了 20cm 板块、ST 的 5%、一字板不可买入
  · 没有 ST 标记
成本倒是一致（Qlib 默认 open 0.0015 + close 0.0025 = 往返 0.4%）。

所以本脚本止步于导出 date × code 的打分矩阵，后续交给已经验证过的回测引擎。

前置：
    pip install pyqlib lightgbm
    python make_vwap.py --qlib-dir <QLIB_DIR>     # Alpha158 需要 $vwap，必须先补

用法：
    python qlib_train.py --check                  # 只验数据可读、$vwap 是否就位
    python qlib_train.py --train
    python qlib_train.py --train --test-end 20231231   # 复核 explore 区间

⚠️ OOS（2024-01 → 2026-08）默认不在任何 segment 里。想跑必须显式 --unlock-oos，
   且只有一次机会 —— 跑完不得回头调参再跑。
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------- 区间
# ML 需要远比规则策略更长的训练窗口：158 维特征 + 5 年数据必然过拟合。
# Qlib 数据覆盖 2005-01-04 → 2026-08-14，把窗口拉长。
# 2005–2009 不用：股权分置改革期复权与涨跌停规则均不同。
TRAIN = ("2010-01-01", "2019-12-31")
VALID = ("2020-01-01", "2020-12-31")
TEST = ("2021-01-01", "2023-12-31")     # 与规则策略的 explore 同区间，可直接对比
OOS = ("2024-01-01", "2026-08-14")      # 冻结

QLIB_DIR = Path(os.environ.get(
    "QLIB_DIR", Path.home() / "astock" / "data" / "qlib"))
OUT_DIR = Path("data/cache")


# ============================================================
# 数据体检
# ============================================================

def check(qlib_dir: Path) -> int:
    import qlib
    from qlib.constant import REG_CN
    from qlib.data import D

    print(f"provider_uri = {qlib_dir}")
    qlib.init(provider_uri=str(qlib_dir), region=REG_CN)

    cal = D.calendar(start_time=TRAIN[0], end_time=OOS[1])
    print(f"交易日        {cal[0]:%Y-%m-%d} → {cal[-1]:%Y-%m-%d}   {len(cal)} 天")

    insts = D.instruments("all")
    codes = D.list_instruments(insts, as_list=True)
    print(f"股票池        {len(codes)} 只（含退市 → 无幸存者偏差）")

    probe = codes[:5]
    need = ["$open", "$high", "$low", "$close", "$volume", "$vwap"]
    try:
        df = D.features(probe, need, start_time="2023-01-01", end_time="2023-01-31")
    except Exception as e:
        print(f"\n❌ 读取失败：{e}")
        print("   若报错涉及 $vwap，先跑：python make_vwap.py --qlib-dir <QLIB_DIR>")
        return 2

    print(f"\n字段可读性（样本 {len(probe)} 只 / 2023-01）")
    for c in need:
        s = df[c]
        print(f"  {c:<9} 非空 {s.notna().mean():>6.1%}   中位 {s.median():>10.4f}")

    if df["$vwap"].notna().mean() < 0.5:
        print("\n⚠️  $vwap 覆盖率过低，Alpha158 的 VWAP0 特征会大量缺失。")
        return 2

    lo, hi, vw = df["$low"], df["$high"], df["$vwap"]
    ok = lo.notna() & hi.notna() & vw.notna()
    inside = ((vw[ok] >= lo[ok] * 0.999) & (vw[ok] <= hi[ok] * 1.001)).mean()
    print(f"\n$vwap ∈ [$low, $high] 命中率 = {inside:.2%}"
          f"{'   ✅' if inside > 0.95 else '   ⚠️ 单位或复权基准可能不一致'}")
    print("\n✅ 数据体检通过。")
    return 0


# ============================================================
# 训练
# ============================================================

def train(qlib_dir: Path, test_seg: tuple[str, str], out: Path) -> int:
    import qlib
    from qlib.constant import REG_CN
    from qlib.contrib.data.handler import Alpha158
    from qlib.contrib.model.gbdt import LGBModel
    from qlib.data.dataset import DatasetH

    qlib.init(provider_uri=str(qlib_dir), region=REG_CN)

    print(f"train {TRAIN[0]} → {TRAIN[1]}")
    print(f"valid {VALID[0]} → {VALID[1]}")
    print(f"test  {test_seg[0]} → {test_seg[1]}")

    # fit_start/fit_end 必须只覆盖 train：标准化参数若用到 valid/test 即为未来函数
    handler = Alpha158(
        instruments="all",
        start_time=TRAIN[0], end_time=test_seg[1],
        fit_start_time=TRAIN[0], fit_end_time=TRAIN[1],
    )
    dataset = DatasetH(handler, segments={
        "train": TRAIN, "valid": VALID, "test": test_seg,
    })

    model = LGBModel(
        loss="mse", learning_rate=0.02, num_leaves=64,
        max_depth=8, feature_fraction=0.7, bagging_fraction=0.8,
        bagging_freq=5, min_data_in_leaf=200, lambda_l1=10, lambda_l2=100,
        num_threads=os.cpu_count() or 4,
    )
    print("\n训练中 ...")
    model.fit(dataset)

    pred = model.predict(dataset, segment="test")
    if isinstance(pred, pd.DataFrame):
        pred = pred.iloc[:, 0]
    pred.name = "score"

    # ---- IC 诊断：先看这个，再看回测 ----
    label = dataset.prepare("test", col_set="label",
                            data_key="raw").iloc[:, 0].rename("label")
    j = pd.concat([pred, label], axis=1).dropna()
    ic = j.groupby(level=0).apply(lambda g: g["score"].corr(g["label"]))
    ric = j.groupby(level=0).apply(
        lambda g: g["score"].corr(g["label"], method="spearman"))
    print(f"\nIC     {ic.mean():+.4f}   ICIR {ic.mean() / ic.std():+.2f}")
    print(f"RankIC {ric.mean():+.4f}   IR   {ric.mean() / ric.std():+.2f}")
    print("  参考：A股日频 Alpha158+LGBM 的 RankIC 通常在 0.03~0.05；"
          "低于 0.02 基本没有可交易的信息量。")

    # ---- 导出 date × code 打分矩阵 ----
    score = pred.unstack(level="instrument")
    score.columns = [_to_tushare(c) for c in score.columns]
    score.index = pd.to_datetime(score.index)
    score = score.sort_index()

    out.parent.mkdir(parents=True, exist_ok=True)
    score.to_parquet(out)
    print(f"\n已导出 {out}   {score.shape[0]} 日 × {score.shape[1]} 只")
    print("\n接入自有回测引擎：")
    print("    from run_backtest import score_to_signal_backtest")
    print(f"    score_to_signal_backtest(df, '{out}')")
    return 0


def _to_tushare(code: str) -> str:
    """qlib instrument id → Tushare 格式 000001.SZ。两种常见写法都兼容。"""
    c = str(code).strip()
    if "." in c:                      # 已是 000001.SZ / 000001.sz
        sym, ex = c.split(".", 1)
        return f"{sym}.{ex.upper()}"
    if len(c) > 2 and c[:2].upper() in ("SH", "SZ", "BJ"):   # SH600519
        return f"{c[2:]}.{c[:2].upper()}"
    return c


# ============================================================

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--qlib-dir", type=Path, default=QLIB_DIR)
    ap.add_argument("--check", action="store_true", help="只做数据体检")
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--test-end", default=TEST[1], help="test 段结束日 YYYY-MM-DD")
    ap.add_argument("--unlock-oos", action="store_true",
                    help="把 test 段延到 OOS。只有一次机会，慎用")
    ap.add_argument("--out", type=Path, default=OUT_DIR / "qlib_score.parquet")
    args = ap.parse_args()

    if not args.qlib_dir.is_dir():
        print(f"❌ 找不到 {args.qlib_dir}（可用 QLIB_DIR 环境变量指定）", file=sys.stderr)
        return 1

    if args.check:
        return check(args.qlib_dir)

    if args.train:
        seg = TEST
        if args.unlock_oos:
            print("\n" + "!" * 62)
            print("!! 你正在解锁 OOS（2024-01 → 2026-08）。")
            print("!! 跑完不得回头调参再跑 —— 那等于把它变成训练集。")
            print("!" * 62 + "\n")
            seg = (TEST[0], OOS[1])
        else:
            seg = (TEST[0], args.test_end)
        return train(args.qlib_dir, seg, args.out)

    ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
