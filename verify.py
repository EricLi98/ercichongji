# -*- coding: utf-8 -*-
"""
补充验证 —— 堵住 diagnose() 的两个漏洞
=====================================
V1  adj_factor 是否有重复主键（决定 JOIN 会不会把行情翻倍）—— 🔴 致命项
V2  daily 里是否真的有退市股的历史 K 线（GATE 1 的真正判据）
V3  stk_limit 覆盖起点（决定回测区间下限）
V4  JOIN 后行数是否守恒（对 V1 的端到端复核）

用法：python verify.py
"""
from __future__ import annotations

import pandas as pd

import data_loader
from market_adapter import _date_sql, _to_dt

SEP = "=" * 64


def main() -> None:
    con = data_loader.connect()
    dd = _date_sql(con, "daily")
    da = _date_sql(con, "adj_factor")
    dd_j = _date_sql(con, "daily", alias="d")
    da_j = _date_sql(con, "adj_factor", alias="a")

    # ---------------------------------------------------------- V1
    print(SEP)
    print("V1  adj_factor 主键唯一性  [🔴 致命项]")
    print(SEP)
    dup = con.execute(f"""
        SELECT COUNT(*) FROM (
            SELECT ts_code, {da} AS d, COUNT(*) n
            FROM adj_factor GROUP BY 1, 2 HAVING n > 1
        )
    """).fetchone()[0]

    if dup == 0:
        print("  ✅ (ts_code, trade_date) 唯一 → JOIN 安全。")
        print("     100.3% 的来源是停牌股（有复权因子、无成交），属正常。")
    else:
        print(f"  ❌ 存在 {dup:,} 组重复主键 → LEFT JOIN 会放大行情行数！")
        print("     修复：在 market_adapter._load_daily 的 SQL 中改用去重子查询：")
        print("       LEFT JOIN (SELECT ts_code, trade_date, ANY_VALUE(adj_factor) AS adj_factor")
        print("                  FROM adj_factor GROUP BY 1,2) a  ON ...")
        sample = con.execute(f"""
            SELECT ts_code, {da} AS d, COUNT(*) n, COUNT(DISTINCT adj_factor) n_val
            FROM adj_factor GROUP BY 1,2 HAVING n > 1 ORDER BY n DESC LIMIT 5
        """).fetchdf()
        print("\n  重复样例（n_val>1 说明因子值本身也冲突，需人工裁决）：")
        print(sample.to_string(index=False))

    # ---------------------------------------------------------- V2
    print("\n" + SEP)
    print("V2  退市股是否真的有历史行情  [GATE 1 的真正判据]")
    print(SEP)
    try:
        delisted = con.execute(
            "SELECT ts_code FROM stock_basic WHERE list_status = 'D'"
        ).fetchdf()["ts_code"].tolist()
        print(f"  stock_basic 中退市股：{len(delisted)} 只")

        have = con.execute(f"""
            SELECT COUNT(DISTINCT ts_code) FROM daily
            WHERE ts_code IN (SELECT ts_code FROM stock_basic WHERE list_status='D')
        """).fetchone()[0]
        cov = have / max(len(delisted), 1)
        print(f"  daily 中有 K 线的：{have} 只 ({cov:.1%})")

        if cov >= 0.95:
            print("  ✅ GATE 1 真实通过，退市股的价格历史确实在库里。")
        else:
            print(f"  ❌ {len(delisted) - have} 只退市股仅有名录、无行情 → 幸存者偏差仍然存在。")
            print("     强势股策略会漏掉「退市前爆炒→崩塌」这类样本，收益系统性上偏。")

        bars = con.execute(f"""
            SELECT MEDIAN(n) FROM (
                SELECT ts_code, COUNT(*) n FROM daily
                WHERE ts_code IN (SELECT ts_code FROM stock_basic WHERE list_status='D')
                GROUP BY 1)
        """).fetchone()[0]
        print(f"  退市股 K 线数中位数：{bars:.0f} 根"
              f"{'   ⚠️ 偏少，可能只保留了尾部片段' if bars and bars < 250 else ''}")
    except Exception as e:
        print(f"  查询失败：{e}")

    # ---------------------------------------------------------- V4
    print("\n" + SEP)
    print("V4  JOIN 行数守恒（对 V1 的端到端复核）")
    print(SEP)
    win = ("20240101", "20241231")
    n_raw = con.execute(
        f"SELECT COUNT(*) FROM daily WHERE {dd} BETWEEN ? AND ?", list(win)
    ).fetchone()[0]
    n_join = con.execute(f"""
        SELECT COUNT(*) FROM daily d
        LEFT JOIN adj_factor a ON a.ts_code = d.ts_code AND {da_j} = {dd_j}
        WHERE {dd_j} BETWEEN ? AND ?
    """, list(win)).fetchone()[0]
    print(f"  2024 全年  daily {n_raw:,} 行 → JOIN 后 {n_join:,} 行")
    if n_join == n_raw:
        print("  ✅ 行数守恒，_load_daily 安全。")
    else:
        print(f"  ❌ 膨胀 {n_join - n_raw:,} 行（+{n_join / n_raw - 1:.2%}）→ 必须先修 V1。")

    con.close()

    # ---------------------------------------------------------- V3
    print("\n" + SEP)
    print("V3  stk_limit 覆盖起点（决定回测区间下限）")
    print(SEP)
    try:
        lim = data_loader.load_stk_limit()
        d = _to_dt(lim["trade_date"]).dropna()
        print(f"  {d.min():%Y-%m-%d} → {d.max():%Y-%m-%d}   {len(lim):,} 行")
        print(f"  建议：{d.min():%Y%m%d} 之后为「真实涨跌停价」区间，之前退化为板块推断。")
        print("        样本外区间务必落在真实区间内。")
    except Exception as e:
        print(f"  ❌ 不可用：{e}")

    nc = data_loader.load_namechange()
    if len(nc):
        s = _to_dt(nc["start_date"]).dropna()
        n_st = nc["name"].astype(str).str.upper().str.contains("ST|PT").sum()
        print(f"\n  namechange  {len(nc):,} 行，含 ST/PT {n_st} 条，起于 {s.min():%Y-%m-%d}")
        print("  ✅ 可还原历史 ST 状态。")
    else:
        print("\n  ⚠️ namechange 缺失 → is_st 恒为 False，只能靠 list_status 事后过滤（有前视偏差）。")

    print("\n" + SEP)


if __name__ == "__main__":
    main()
