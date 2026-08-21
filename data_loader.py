"""离线数据只读封装。

只从 astock-lab/data 读取，绝不写入。stk_limit / namechange / 指数等补充表
从本项目 data/cache/ 读取（由 src.fetch_index.py 等一次性补拉生成）。
"""
from __future__ import annotations

import duckdb
import pandas as pd

import config

# 数据库表 → (主键, 日期列) 元信息
_TABLES = {
    "daily": ("ts_code", "trade_date"),
    "daily_basic": ("ts_code", "trade_date"),
    "adj_factor": ("ts_code", "trade_date"),
    "stock_basic": ("ts_code", None),
    "trade_cal": ("exchange", "cal_date"),
    "index_daily": ("ts_code", "trade_date"),
}


def connect() -> duckdb.DuckDBPyConnection:
    """以只读方式连接离线库。任何写入尝试都会抛错。"""
    return duckdb.connect(str(config.OFFLINE_DB_PATH), read_only=True)


def _read_cache(fname: str, required: bool = True) -> pd.DataFrame:
    p = config.CACHE_DIR / fname
    if not p.exists():
        if required:
            raise FileNotFoundError(f"缓存缺失：{p}（先运行 src.fetch_index.py 等补拉脚本）")
        return pd.DataFrame()
    return pd.read_parquet(p)


# ---------------------------------------------------------------- 全表加载（按需使用）

def load_stock_basic() -> pd.DataFrame:
    with connect() as con:
        return con.execute("SELECT * FROM stock_basic").fetchdf()


def load_trade_cal() -> pd.DataFrame:
    """全市场交易日历，含 is_open 标记。"""
    with connect() as con:
        df = con.execute("SELECT * FROM trade_cal").fetchdf()
    return df


def load_stk_limit() -> pd.DataFrame:
    """涨跌停价（tushare 补下载缓存，全市场）。"""
    return _read_cache("stk_limit.parquet")


def load_namechange() -> pd.DataFrame:
    """历史曾用名（tushare 补下载缓存）。"""
    return _read_cache("namechange.parquet", required=False)


def load_index_daily() -> pd.DataFrame:
    """离线指数日线（仅 000300.SH）。"""
    with connect() as con:
        return con.execute("SELECT * FROM index_daily").fetchdf()


def load_fetched_index() -> pd.DataFrame:
    """tushare 补拉的指数日线（上证/创业板指），由 src.fetch_index 生成。"""
    return _read_cache("index_daily_ts.parquet", required=False)


# ---------------------------------------------------------------- 按股票切片（内存友好）

def _query_stock(table: str, ts_code: str, cols: str = "*") -> pd.DataFrame:
    with connect() as con:
        return con.execute(
            f"SELECT {cols} FROM {table} WHERE ts_code = ? ORDER BY trade_date", [ts_code]
        ).fetchdf()


def load_stock_daily(ts_code: str) -> pd.DataFrame:
    return _query_stock("daily", ts_code)


def load_stock_daily_basic(ts_code: str) -> pd.DataFrame:
    return _query_stock("daily_basic", ts_code)


def load_stock_adj(ts_code: str) -> pd.DataFrame:
    return _query_stock("adj_factor", ts_code)


def all_ts_codes(market: str | None = None) -> list[str]:
    """全部股票代码；market: SSE/SZSE/BSE。"""
    with connect() as con:
        if market:
            return [
                r[0]
                for r in con.execute(
                    "SELECT DISTINCT ts_code FROM stock_basic WHERE exchange = ?", [market]
                ).fetchall()
            ]
        return [r[0] for r in con.execute("SELECT DISTINCT ts_code FROM stock_basic").fetchall()]


def trading_days(start: str | None = None, end: str | None = None) -> list[str]:
    """按 trade_cal 返回 is_open==1 的交易日（升序）。"""
    with connect() as con:
        sql = "SELECT cal_date FROM trade_cal WHERE is_open = 1"
        if start:
            sql += f" AND cal_date >= '{start}'"
        if end:
            sql += f" AND cal_date <= '{end}'"
        sql += " ORDER BY cal_date"
        return [r[0] for r in con.execute(sql).fetchall()]
