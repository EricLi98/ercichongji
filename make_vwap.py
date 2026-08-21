#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
为 Qlib bin 数据补齐 $vwap
==========================
Alpha158 的 price 特征默认包含 VWAP，缺失会导致表达式解析失败。
本脚本由 amount / volume 推导 vwap，并写入 vwap.day.bin。

难点在于两个未知量：
  1. 单位   —— Tushare 原始单位是 vol=手、amount=千元，dump 时可能已换算也可能没有
  2. 复权基准 —— close.day.bin 可能存原始价，也可能存复权价

脚本用一个自洽判据同时解决两者：**vwap 必须落在 [low, high] 区间内**。
逐个试候选缩放系数，取命中率最高的那个；命中率过低则拒绝写入并报告。

用法：
    python make_vwap.py --qlib-dir /path/to/qlib [--dry-run] [--limit 50]
"""
from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

import numpy as np

# amount/volume 的可能单位组合导致的缩放系数
SCALE_CANDIDATES = [1.0, 10.0, 0.1, 1000.0, 0.001, 100.0, 0.01]
MIN_HIT_RATE = 0.95          # vwap 落在 [low, high] 的最低比例
SAMPLE_FOR_DETECT = 300      # 用多少只股票探测缩放系数


def read_bin(path: Path) -> tuple[int, np.ndarray]:
    """返回 (start_index, values)。格式: [int32 start][float32 × N]"""
    raw = path.read_bytes()
    start = struct.unpack("<i", raw[:4])[0]
    vals = np.frombuffer(raw[4:], dtype="<f4")
    return start, vals


def write_bin(path: Path, start: int, vals: np.ndarray) -> None:
    with open(path, "wb") as f:
        f.write(struct.pack("<i", int(start)))
        f.write(np.asarray(vals, dtype="<f4").tobytes())


def _hit_rate(vwap: np.ndarray, low: np.ndarray, high: np.ndarray) -> float:
    """vwap 落在 [low, high] 内的比例（只统计三者都有效的 bar）。"""
    ok = np.isfinite(vwap) & np.isfinite(low) & np.isfinite(high) & (high > 0)
    if ok.sum() == 0:
        return 0.0
    tol = 1e-6
    inside = (vwap[ok] >= low[ok] * (1 - tol)) & (vwap[ok] <= high[ok] * (1 + tol))
    return float(inside.mean())


def load_stock(d: Path) -> dict | None:
    need = ["amount", "volume", "low", "high"]
    if not all((d / f"{n}.day.bin").exists() for n in need):
        return None
    out = {}
    for n in need:
        s, v = read_bin(d / f"{n}.day.bin")
        out[n] = (s, v)
    # 对齐到共同的 start_index
    starts = [out[n][0] for n in need]
    lens = [out[n][0] + len(out[n][1]) for n in need]
    s0, e0 = max(starts), min(lens)
    if e0 <= s0:
        return None
    return {n: out[n][1][s0 - out[n][0]: e0 - out[n][0]] for n in need} | {"start": s0}


def detect_scale(dirs: list[Path]) -> tuple[float, float]:
    """在样本股票上探测缩放系数，返回 (scale, 命中率)。"""
    best = (1.0, 0.0)
    per_scale: dict[float, list[float]] = {s: [] for s in SCALE_CANDIDATES}

    for d in dirs:
        st = load_stock(d)
        if st is None:
            continue
        vol = np.where(st["volume"] > 0, st["volume"], np.nan)
        for sc in SCALE_CANDIDATES:
            vwap = st["amount"] * sc / vol
            per_scale[sc].append(_hit_rate(vwap, st["low"], st["high"]))

    for sc, rates in per_scale.items():
        if not rates:
            continue
        m = float(np.mean(rates))
        if m > best[1]:
            best = (sc, m)
    return best


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--qlib-dir", required=True, type=Path)
    ap.add_argument("--dry-run", action="store_true", help="只探测不写入")
    ap.add_argument("--limit", type=int, default=0, help="只处理前 N 只（调试用）")
    ap.add_argument("--overwrite", action="store_true", help="覆盖已存在的 vwap.day.bin")
    args = ap.parse_args()

    feat = args.qlib_dir / "features"
    if not feat.is_dir():
        print(f"❌ 找不到 {feat}", file=sys.stderr)
        return 1

    dirs = sorted(p for p in feat.iterdir() if p.is_dir())
    print(f"发现 {len(dirs)} 只股票")

    # ---- 1) 探测缩放系数 ----
    print(f"\n[1/2] 在 {min(SAMPLE_FOR_DETECT, len(dirs))} 只样本上探测单位/复权基准 ...")
    scale, rate = detect_scale(dirs[:SAMPLE_FOR_DETECT])
    print(f"  最佳缩放系数 = {scale}   vwap∈[low,high] 命中率 = {rate:.2%}")

    if rate < MIN_HIT_RATE:
        print(f"\n❌ 命中率低于 {MIN_HIT_RATE:.0%}，拒绝写入。可能原因：")
        print("   · close/low/high 存的是复权价，而 amount/volume 是原始值")
        print("     → 需要先用 factor.day.bin 把 amount 调整到同一基准")
        print("   · amount 或 volume 本身有质量问题")
        print("   排查建议：手工取一只股票，比对 amount/volume 与当日 low/high。")
        return 2

    if scale != 1.0:
        print(f"  ⚠️  非 1.0 说明 amount 与 volume 单位不一致"
              f"（Tushare 原始为 vol=手、amount=千元 → 系数 10）")

    if args.dry_run:
        print("\n[dry-run] 未写入任何文件。")
        return 0

    # ---- 2) 写入 ----
    todo = dirs[: args.limit] if args.limit else dirs
    print(f"\n[2/2] 写入 vwap.day.bin ({len(todo)} 只) ...")
    written = skipped = failed = 0
    for i, d in enumerate(todo, 1):
        out = d / "vwap.day.bin"
        if out.exists() and not args.overwrite:
            skipped += 1
            continue
        st = load_stock(d)
        if st is None:
            failed += 1
            continue
        vol = np.where(st["volume"] > 0, st["volume"], np.nan)
        vwap = st["amount"] * scale / vol
        # 停牌/异常 bar 置 NaN，交给 qlib 的 NaN 处理
        vwap = np.where(np.isfinite(vwap) & (vwap > 0), vwap, np.nan)
        write_bin(out, st["start"], vwap)
        written += 1
        if i % 500 == 0:
            print(f"  {i}/{len(todo)} ...")

    print(f"\n完成：写入 {written}，跳过 {skipped}，失败 {failed}")
    print("验证：qlib.init 后跑 D.features(['SH600519'], ['$vwap'], '2024-01-01', '2024-01-10')")
    return 0


if __name__ == "__main__":
    sys.exit(main())
