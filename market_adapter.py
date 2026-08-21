# -*- coding: utf-8 -*-
"""
离线库 → 策略长表 适配层
========================
把 data_loader 暴露的「按股票切片、不复权、Tushare 原始单位」的接口，
转成 pullback_momentum.py 需要的「全市场截面、后复权、SI 单位」长表。

处理的四件事：
  1. 全市场区间加载   —— 单条 SQL 取代 5000 次 _query_stock
  2. 后复权           —— join adj_factor 并按窗口首日归一
  3. 单位换算         —— vol 手→股(×100)，amount 千元→元(×1000)
  4. 状态标记         —— stk_limit 真实涨跌停价、namechange 还原历史 ST、上市交易日数

用法：
    python market_adapter.py            # 先跑诊断，确认数据可用
    df = load_market('20180101', '20260630')
    from pullback_momentum import run
    res = run(df)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import config
import data_loader

FAR_FUTURE = pd.Timestamp("2099-12-31")
ST_TOKENS = ("ST", "PT")          # 覆盖 ST / *ST / SST / S*ST / PT


# ============================================================
# 0. 通用工具
# ============================================================

def _to_dt(s: pd.Series) -> pd.Series:
    """兼容 'YYYYMMDD' / 'YYYY-MM-DD' / DATE / TIMESTAMP 四种存法。"""
    if pd.api.types.is_datetime64_any_dtype(s):
        return s.dt.normalize()
    txt = s.astype(str).str.replace("-", "", regex=False).str.slice(0, 8)
    return pd.to_datetime(txt, format="%Y%m%d", errors="coerce")


def _date_sql(con, table: str, col: str = "trade_date", alias: str = "") -> str:
    """返回把日期列统一成 'YYYYMMDD' 字符串的 SQL 表达式（用于区间过滤）。"""
    prefixed = f"{alias}.{col}" if alias else col
    try:
        t = con.execute(f"SELECT typeof({col}) FROM {table} LIMIT 1").fetchone()[0].upper()
    except Exception:
        t = "VARCHAR"
    if "DATE" in t or "TIMESTAMP" in t:
        return f"strftime({prefixed}, '%Y%m%d')"
    return f"REPLACE(CAST({prefixed} AS VARCHAR), '-', '')"


def _infer_board(ts_code: str, exchange: str = "") -> str:
    s = str(ts_code)
    if s.startswith("688") or s.startswith("689"):
        return "STAR"
    if s.startswith("300") or s.startswith("301"):
        return "GEM"
    if str(exchange).upper() == "BSE" or s.startswith(("4", "8", "92")):
        return "BJ"
    return "MAIN"


# ============================================================
# 1. 诊断 —— 跑策略前必须先过这一关
# ============================================================

def diagnose(sample_date: str | None = None) -> None:
    con = data_loader.connect()
    print("=" * 62)
    print("离线库诊断")
    print("=" * 62)

    # ---- 1) 幸存者偏差：这是最要命的一项 ----
    try:
        st = con.execute(
            "SELECT list_status, COUNT(*) n FROM stock_basic GROUP BY 1 ORDER BY 2 DESC"
        ).fetchdf()
        print("\n[1] stock_basic.list_status 分布")
        print(st.to_string(index=False))
        has_d = "D" in set(st["list_status"].astype(str))
        if not has_d:
            print("  ⚠️  没有 list_status='D' 的退市股 → 存在幸存者偏差。")
            print("     强势股策略会被系统性高估（退市前的暴跌样本全部缺失）。")
            print("     修复：tushare pro.stock_basic(list_status='D') 补拉后并入。")
        else:
            print(f"  ✅ 含退市股 {int(st.loc[st.list_status == 'D', 'n'].iloc[0])} 只。")
    except Exception as e:
        print(f"\n[1] stock_basic 查询失败：{e}")

    # ---- 2) 各表时间覆盖 ----
    print("\n[2] 各表日期覆盖")
    for tbl in ["daily", "adj_factor", "daily_basic", "index_daily"]:
        try:
            d = _date_sql(con, tbl)
            r = con.execute(f"SELECT MIN({d}), MAX({d}), COUNT(*) FROM {tbl}").fetchone()
            print(f"  {tbl:<12} {r[0]} → {r[1]}   {r[2]:,} 行")
        except Exception as e:
            print(f"  {tbl:<12} 不可用：{e}")

    # ---- 3) 单位校验：amount ?= close × vol × 100 / 1000 ----
    print("\n[3] 单位校验（Tushare 约定：vol=手, amount=千元）")
    try:
        d = _date_sql(con, "daily")
        day = sample_date or con.execute(f"SELECT MAX({d}) FROM daily").fetchone()[0]
        s = con.execute(
            f"SELECT close, vol, amount FROM daily WHERE {d} = ? AND vol > 0 LIMIT 200", [day]
        ).fetchdf()
        ratio = (s["amount"] * 1000) / (s["close"] * s["vol"] * 100)
        med = float(ratio.median())
        print(f"  样本日 {day}，成交额/(收盘×量) 中位数 = {med:.3f}")
        if 0.9 < med < 1.1:
            print("  ✅ 符合 手/千元 约定，适配层的 ×100 / ×1000 换算正确。")
        else:
            print(f"  ⚠️  偏离 1.0，你的库单位可能不是 手/千元。请核对后改 _UNIT_* 常量。")
    except Exception as e:
        print(f"  校验失败：{e}")

    # ---- 4) 复权因子覆盖率 ----
    print("\n[4] adj_factor 覆盖率")
    try:
        dd, da = _date_sql(con, "daily"), _date_sql(con, "adj_factor")
        day = sample_date or con.execute(f"SELECT MAX({dd}) FROM daily").fetchone()[0]
        n1 = con.execute(f"SELECT COUNT(*) FROM daily WHERE {dd}=?", [day]).fetchone()[0]
        n2 = con.execute(f"SELECT COUNT(*) FROM adj_factor WHERE {da}=?", [day]).fetchone()[0]
        print(f"  {day}: daily {n1} 只 / adj_factor {n2} 只 = {n2 / max(n1, 1):.1%}")
        if n2 < n1 * 0.99:
            print("  ⚠️  复权因子缺口会导致这些股票除权日被误判为暴跌。")
    except Exception as e:
        print(f"  查询失败：{e}")

    con.close()

    # ---- 5) 缓存表 ----
    print("\n[5] 缓存表")
    try:
        lim = data_loader.load_stk_limit()
        ld = _to_dt(lim["trade_date"])
        print(f"  stk_limit   {ld.min():%Y%m%d} → {ld.max():%Y%m%d}   {len(lim):,} 行")
        print("     ⚠️  回测起点早于该起始日的区间将无真实涨跌停价（退化为板块推断）。")
    except Exception as e:
        print(f"  stk_limit   缺失：{e}")
    nc = data_loader.load_namechange()
    if len(nc):
        n_st = nc["name"].astype(str).str.upper().str.contains("|".join(ST_TOKENS)).sum()
        print(f"  namechange  {len(nc):,} 行，其中含 ST/PT 名称 {n_st} 条")
    else:
        print("  namechange  缺失 → 无法还原历史 ST，只能剔除当前 ST（有前视偏差）")
    print("\n" + "=" * 62)


# ============================================================
# 2. 全市场加载
# ============================================================

_UNIT_VOL = 100.0     # 手 → 股
_UNIT_AMT = 1000.0    # 千元 → 元


def _load_daily(start: str, end: str) -> pd.DataFrame:
    """单条 SQL 取全市场行情 + 复权因子，避免 5000 次连接。"""
    con = data_loader.connect()
    dd, da = _date_sql(con, "daily", "trade_date", "d"), _date_sql(con, "adj_factor", "trade_date", "a")
    sql = f"""
        SELECT d.ts_code,
               {dd}            AS trade_date,
               d.open, d.high, d.low, d.close, d.pre_close,
               d.vol, d.amount,
               a.adj_factor
        FROM daily d
        LEFT JOIN adj_factor a
               ON a.ts_code = d.ts_code AND {da} = {dd}
        WHERE {dd} BETWEEN ? AND ?
        ORDER BY d.ts_code, {dd}
    """
    df = con.execute(sql, [start, end]).fetchdf()
    con.close()
    return df


def load_market(
    start: str,
    end: str,
    adjust: str = "hfq",
    cache: str | Path | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    返回 pullback_momentum 可直接消费的长表。

    参数
    ----
    start, end : 'YYYYMMDD'
    adjust     : 'hfq' 后复权（默认，回测应使用）/ 'none' 不复权
    cache      : 落盘路径，命中则直接读 parquet

    输出列
    ------
    date, code, open, high, low, close, volume(股), amount(元),
    limit_up, limit_down, one_word_up, paused, is_st, list_days, board
    """
    cache = Path(cache) if cache else None
    if cache and cache.exists():
        if verbose:
            print(f"[cache] {cache}")
        return pd.read_parquet(cache)

    if verbose:
        print(f"[1/5] 加载行情 {start} → {end} ...")
    df = _load_daily(start, end)
    if df.empty:
        raise ValueError(f"区间 {start}~{end} 无数据，请检查 daily 表覆盖范围。")

    df["date"] = pd.to_datetime(df["trade_date"], format="%Y%m%d")
    df = df.rename(columns={"ts_code": "code"}).drop(columns=["trade_date"])
    df = df.sort_values(["code", "date"], ignore_index=True)

    # ---- 涨跌停：必须在复权【之前】用原始价判断 ----
    if verbose:
        print("[2/5] 标记涨跌停 ...")
    df = _attach_limits(df)

    # ---- 后复权 ----
    if verbose:
        print(f"[3/5] 复权 ({adjust}) ...")
    if adjust == "hfq":
        g = df.groupby("code", sort=False)["adj_factor"]
        f = g.ffill().bfill()
        base = f.groupby(df["code"], sort=False).transform("first")   # 归一到窗口首日
        mult = (f / base).fillna(1.0)
        for c in ["open", "high", "low", "close", "pre_close"]:
            df[c] = df[c] * mult
    df = df.drop(columns=["adj_factor", "pre_close"], errors="ignore")

    # ---- 单位换算 ----
    df["volume"] = df.pop("vol") * _UNIT_VOL
    df["amount"] = df["amount"] * _UNIT_AMT
    df["paused"] = df["volume"].fillna(0) <= 0

    # ---- ST / 上市天数 / 板块 ----
    if verbose:
        print("[4/5] 附加 ST / 上市交易日数 / 板块 ...")
    df = _attach_st(df)
    df = _attach_basic(df, start, end)

    cols = ["date", "code", "open", "high", "low", "close", "volume", "amount",
            "limit_up", "limit_down", "one_word_up", "paused", "is_st",
            "list_days", "board"]
    df = df[[c for c in cols if c in df.columns]]

    for c in ["open", "high", "low", "close", "volume", "amount"]:
        df[c] = df[c].astype("float32")
    df["code"] = df["code"].astype("category")

    if verbose:
        print(f"[5/5] 完成：{len(df):,} 行 × {df['code'].nunique()} 只 "
              f"({df.memory_usage(deep=True).sum() / 1e6:.0f} MB)")
    if cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache, index=False)
    return df


# ============================================================
# 3. 各类标记
# ============================================================

def _attach_limits(df: pd.DataFrame) -> pd.DataFrame:
    """用 stk_limit 的真实涨跌停价打标；缺失区间退化为板块涨跌幅推断。"""
    try:
        lim = data_loader.load_stk_limit()
    except Exception:
        lim = pd.DataFrame()

    if len(lim):
        lim = lim.rename(columns={"ts_code": "code"}).copy()
        lim["date"] = _to_dt(lim["trade_date"])
        lim = lim[["code", "date", "up_limit", "down_limit"]]
        df = df.merge(lim, on=["code", "date"], how="left")
    else:
        df["up_limit"] = np.nan
        df["down_limit"] = np.nan

    eps = 1e-3
    df["limit_up"] = df["close"] >= df["up_limit"] - eps
    df["limit_down"] = df["close"] <= df["down_limit"] + eps
    # 一字涨停：开盘即封板，次日无法买入
    df["one_word_up"] = (df["open"] >= df["up_limit"] - eps) & (df["high"] <= df["low"] + eps)

    # stk_limit 未覆盖的日期 → 按涨跌幅推断
    miss = df["up_limit"].isna()
    if miss.any():
        board = df["code"].astype(str).map(lambda c: _infer_board(c))
        cap = board.map({"MAIN": 0.10, "GEM": 0.20, "STAR": 0.20, "BJ": 0.30}).fillna(0.10)
        pc = df["pre_close"].replace(0, np.nan)
        ret = df["close"] / pc - 1
        df.loc[miss, "limit_up"] = (ret >= cap - 1e-4)[miss]
        df.loc[miss, "limit_down"] = (ret <= -cap + 1e-4)[miss]
        df.loc[miss, "one_word_up"] = (
            (df["open"] / pc - 1 >= cap - 1e-4) & (df["high"] <= df["low"] + eps)
        )[miss]

    for c in ["limit_up", "limit_down", "one_word_up"]:
        df[c] = df[c].fillna(False).astype(bool)
    return df.drop(columns=["up_limit", "down_limit"], errors="ignore")


def _attach_st(df: pd.DataFrame) -> pd.DataFrame:
    """用 namechange 的区间还原历史 ST 状态（避免用当前状态回溯，那是前视偏差）。"""
    nc = data_loader.load_namechange()
    if not len(nc):
        df["is_st"] = False
        return df

    nc = nc.rename(columns={"ts_code": "code"}).copy()
    up = nc["name"].astype(str).str.upper()
    nc = nc[up.str.contains("|".join(ST_TOKENS), na=False)]
    if not len(nc):
        df["is_st"] = False
        return df

    nc["start"] = _to_dt(nc["start_date"])
    nc["end"] = _to_dt(nc["end_date"]).fillna(FAR_FUTURE)
    nc = nc.dropna(subset=["start"]).sort_values("start")[["code", "start", "end"]]

    left = df[["date", "code"]].copy()
    left["_i"] = np.arange(len(left))
    left = left.sort_values("date")
    m = pd.merge_asof(
        left, nc, left_on="date", right_on="start",
        left_by="code", right_by="code", direction="backward",
    )
    flag = (m["end"] >= m["date"]).fillna(False).to_numpy()
    out = np.zeros(len(df), dtype=bool)
    out[m["_i"].to_numpy()] = flag
    df["is_st"] = out
    return df


def _attach_basic(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    """上市交易日数（不是自然日）+ 板块。"""
    basic = data_loader.load_stock_basic().rename(columns={"ts_code": "code"})
    basic["list_dt"] = _to_dt(basic["list_date"])

    tds = pd.to_datetime(
        pd.Series(data_loader.trading_days()).astype(str).str.replace("-", "", regex=False),
        format="%Y%m%d", errors="coerce",
    ).dropna().sort_values().to_numpy()

    listed = basic.dropna(subset=["list_dt"]).copy()
    listed["_ord0"] = np.searchsorted(tds, listed["list_dt"].to_numpy(), side="left")
    ord0 = listed.set_index("code")["_ord0"]

    cur = np.searchsorted(tds, df["date"].to_numpy(), side="left")
    df["list_days"] = cur - df["code"].astype(str).map(ord0).to_numpy()
    df["list_days"] = df["list_days"].fillna(9999).astype("int32")

    ex = (basic.set_index("code")["exchange"]
          if "exchange" in basic.columns else pd.Series(dtype=object))
    mapping = {c: _infer_board(c, ex.get(c, "")) for c in df["code"].astype(str).unique()}
    df["board"] = df["code"].astype(str).map(mapping).astype("category")
    return df


if __name__ == "__main__":
    diagnose()
