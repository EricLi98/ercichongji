# Qlib 数据目录描述

路径：`/Users/ericli/Documents/projs/astock/data/qlib/`

## 总览

| 项目 | 值 |
|---|---|
| 格式 | Qlib 标准二进制（`.day.bin`） |
| 股票数 | 5858（SH 2449 / SZ 3068 / BJ 340） |
| 交易日 | 5250 天（2005-01-04 → 2026-08-14） |
| 特征数 | 7 只 / 股（open, high, low, close, volume, amount, factor） |
| 总大小 | 540 MB（全部在 `features/`） |

## 目录结构

```
qlib/
├── calendars/
│   └── day.txt                 # 5250 行，每行一个交易日 YYYY-MM-DD
├── instruments/
│   ├── all.txt                 # 5857 行，格式: CODE  START_DATE  END_DATE
│   └── csi300.txt              # 300 行，沪深300成分股
└── features/
    ├── 000001.sz/              # 平安银行
    │   ├── open.day.bin
    │   ├── high.day.bin
    │   ├── low.day.bin
    │   ├── close.day.bin
    │   ├── volume.day.bin
    │   ├── amount.day.bin
    │   └── factor.day.bin      # 复权因子
    ├── 000002.sz/              # 万科A
    │   └── ...
    └── ... (5858 个子目录)
```

## 二进制格式（`.day.bin`）

每个文件结构：

```
[int32 start_index] [float32 × N]
```

- `start_index`：数据起始位置对应 `calendars/day.txt` 的行索引（0-based）
- `N = (file_size_bytes - 4) / 4`：float32 数据点数
- 缺失值为 `NaN`（典型覆盖率 ~97%，停牌/未上市日为 NaN）

以 `000001.sz/close.day.bin` 为例：

| 字段 | 值 |
|---|---|
| 文件大小 | 21,004 bytes |
| start_index | 0 |
| 数据点数 | 5250 |
| 首值 | 6.52（2005-01-04） |
| 末值 | 11.11（2026-08-14） |
| NaN 比例 | 2.8%（148/5250） |

## 代码命名

| 后缀 | 交易所 | 示例 |
|---|---|---|
| `.sz` | 深交所 | `000001.sz`, `300750.sz` |
| `.sh` | 上交所 | `600519.sh`, `688981.sh` |
| `.bj` | 北交所 | `430047.bj`, `830799.bj` |

与 Tushare 的 `.SZ`/`.SH`/`.BJ` 对应关系：小写 + 点号前缀。

## instruments/all.txt 格式

```
CODE        START_DATE  END_DATE
000001.SZ   2005-01-04  2026-08-14
000002.SZ   2005-01-04  2026-08-14
000004.SZ   2005-01-04  2026-07-13    # 已退市
```

日期范围反映每只股票的实际可交易区间，退市股的 END_DATE 早于最新交易日。

## 与 ercichongji 策略的关系

此目录为 Qlib 框架的标准数据格式，可直接被 `qlib.init(provider_uri=...)` 加载。
ercichongji 项目使用 DuckDB 离线库而非 Qlib，两者数据来源相同（Tushare），但格式不同：

| | Qlib bin | DuckDB + parquet |
|---|---|---|
| 存储 | 每股每特征一个文件 | 列式表（daily / adj_factor） |
| 复权 | factor.day.bin 内置 | adj_factor 表，加载时计算 |
| 涨跌停 | 无 | stk_limit 补充表 |
| ST 标记 | 无 | namechange 还原 |
