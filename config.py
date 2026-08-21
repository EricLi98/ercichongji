"""项目配置 — 路径指向 astock-lab 的离线数据。"""
from pathlib import Path

# DuckDB 离线库（daily / adj_factor / stock_basic / trade_cal / index_daily）
OFFLINE_DB_PATH = Path("/Users/ericli/Documents/projs/astock/data/astock.duckdb")

# 补充表缓存（stk_limit / namechange / index_daily_ts）
# 由 src.fetch_index.py 等脚本一次性生成
CACHE_DIR = Path("/Users/ericli/Documents/projs/astock/data/parquet")
