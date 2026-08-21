"""项目配置 — 路径指向 astock-lab 的离线数据。

优先读环境变量，兜底到同级 ../astock/data/。
"""
import os
from pathlib import Path

_HERE = Path(__file__).resolve().parent

# DuckDB 离线库（daily / adj_factor / stock_basic / trade_cal / index_daily）
OFFLINE_DB_PATH = Path(os.environ.get(
    "ASTOCK_DB", _HERE.parent / "astock" / "data" / "astock.duckdb"
))

# 补充表缓存（stk_limit / namechange / index_daily_ts）
CACHE_DIR = Path(os.environ.get(
    "ASTOCK_CACHE", _HERE.parent / "astock" / "data" / "parquet"
))
