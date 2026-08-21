# A股「强势股回调」策略 — Agent 运行手册

> 面向自动化 agent。人类读者请直接看 §3 和 §6。
> **本项目是研究工具，不构成投资建议。任何回测结果都不可直接用于实盘决策。**

---

## 0. TL;DR

```bash
pip install duckdb pandas numpy pyarrow
python market_adapter.py          # 必须先跑，且必须人工确认输出
```

```python
from market_adapter import load_market
from pullback_momentum import run

df  = load_market('20180101', '20260630', cache='data/cache/panel.parquet')
res = run(df)

print(pd.Series(res['stats']))
print(res['exits'])
```

**在 §2 的三道 GATE 全部通过之前，不要报告任何绩效数字。**

---

## 1. 项目结构

| 文件 | 职责 | 可否修改 |
|---|---|---|
| `data_loader.py` | 离线库只读封装（DuckDB + parquet 缓存） | ❌ 只读层，不要动 |
| `market_adapter.py` | 离线库 → 策略长表：全市场加载 / 后复权 / 单位换算 / 状态标记 | ⚠️ 仅在 GATE 失败时按提示改常量 |
| `pullback_momentum.py` | 信号生成 + 事件驱动回测 + 绩效 | ✅ 参数与规则可调，见 §6 边界 |
| `config.py` | 需提供 `OFFLINE_DB_PATH`、`CACHE_DIR` | 环境相关 |

数据流：

```
DuckDB(daily, adj_factor, stock_basic, trade_cal)
  + parquet(stk_limit, namechange)
        │
        ▼  market_adapter.load_market()
   长表 DataFrame  ← 契约见 §3
        │
        ▼  pullback_momentum.build_panel()
   宽表面板 dict[str, DataFrame(date × code)]
        │
        ▼  generate_signals() → backtest() → performance()
```

策略逻辑分四层：**趋势层**（选强势）→ **形态层**（等回调）→ **信号层**（等企稳）→ **风控层**（管出场）。
执行约定：T 日收盘产生信号，**T+1 开盘成交**，无未来函数。

---

## 2. 前置检查（阻断式）

运行 `python market_adapter.py`，逐项核对。**任一 GATE 失败必须先解决或向人类报告，不得跳过。**

### GATE 1 — 幸存者偏差 🔴 最高优先级

检查 `[1] stock_basic.list_status 分布`。

| 结果 | 动作 |
|---|---|
| 含 `list_status='D'` | ✅ 通过 |
| 只有 `'L'` | ❌ **停止**。股票池只含在市公司，退市股（往往在退市前有过爆炒+崩塌）全部缺失，强势股策略收益会被系统性高估。需补拉 `pro.stock_basic(list_status='D')` 并入 `stock_basic`。若无法补拉，向人类报告并在所有结论中标注「存在幸存者偏差，收益上偏」。 |

### GATE 2 — 单位一致性

检查 `[3] 单位校验` 输出的中位数。

| 结果 | 动作 |
|---|---|
| 0.9 ~ 1.1 | ✅ 通过。库符合 Tushare 约定（`vol`=手，`amount`=千元） |
| 显著偏离 1.0 | ❌ 修改 `market_adapter.py` 顶部的 `_UNIT_VOL` / `_UNIT_AMT`，使换算后 `volume` 单位为**股**、`amount` 单位为**元**。改完重跑诊断确认。 |

> 为什么重要：`Config.min_amount = 5e7`（5000万元日均成交额）是以**元**为单位的流动性过滤。单位错一个量级，要么全市场被清空，要么过滤完全失效。

### GATE 3 — 覆盖区间

检查 `[2]` 和 `[5]`，据此确定合法回测区间：

- `start` 不得早于 `daily` 与 `adj_factor` 的共同起点
- `start` **建议**不早于 `stk_limit` 起始日。更早的区间会退化为按板块涨跌幅推断涨跌停，除权日可能误判
- `[4] adj_factor 覆盖率` < 99% 时，缺口股票的除权日会被识别成暴跌，可能误触发回调信号
- `[5] namechange` 缺失时 `is_st` 恒为 `False`，无法还原历史 ST 状态

---

## 3. 数据契约

`load_market()` 输出的长表列定义。**修改 adapter 时必须维持此契约**，否则 `build_panel()` 会静默降级。

| 列 | 类型 | 单位/语义 |
|---|---|---|
| `date` | datetime64 | 交易日 |
| `code` | category | `'000001.SZ'` |
| `open/high/low/close` | float32 | **后复权**，归一到窗口首日 |
| `volume` | float32 | **股**（已 ×100） |
| `amount` | float32 | **元**（已 ×1000） |
| `limit_up` / `limit_down` | bool | **收盘**封板。注意：盘中仍可能成交 |
| `one_word_up` | bool | 一字涨停（开盘即封板）→ 次日无法以开盘价买入 |
| `paused` | bool | 停牌 |
| `is_st` | bool | 由 `namechange` 区间还原的**历史**状态 |
| `list_days` | int32 | 上市**交易日**数（非自然日） |
| `board` | category | `MAIN` / `GEM` / `STAR` / `BJ` |

### 三个易错语义

1. **`limit_up` ≠ 不可买入。** 收盘封板的票很多是盘中拉起的，开盘价可正常成交。能否以开盘价买入只取决于 `paused` 和 `one_word_up`。「不追涨停」的逻辑在信号层（`generate_signals` 排除信号日涨停），不在执行层。
2. **涨跌停必须用原始价判断。** adapter 已在复权**之前**完成 `limit_*` 标记。若重构此处，务必保持顺序，否则后复权价与 `stk_limit` 的原始价不可比。
3. **后复权锚定窗口首日。** 同一区间重跑结果稳定可复现。不要改成锚定最新日（前复权），那会导致每次数据更新后历史价格全部漂移。

---

## 4. 参数

全部在 `pullback_momentum.Config`。默认值是起点，不是结论。

```python
cfg = Config(
    mom_window=60, top_pct=0.20, high_lookback=20,      # 趋势层
    use_atr=True, pb_min_atr=1.0, pb_max_atr=4.0,        # 形态层（ATR 归一，兼容 10cm/20cm）
    pb_days_min=2, pb_days_max=7, shrink_ratio=0.90,
    vol_up=1.20,                                          # 信号层
    max_positions=10, stop_loss=0.07, trail_stop=0.12,   # 风控层
    max_hold=15, exit_below_ma=True,
)
res = run(df, cfg)
```

**`use_atr=True` 是推荐设置。** 固定百分比阈值会让 20cm 板块（创业板/科创板）样本严重失真——同样的 5% 回调，在 10cm 票是深调，在 20cm 票只是日常波动。

---

## 5. 结果解读

### 必看的三个输出

```python
res['stats']            # 绩效汇总
res['exits']            # 出场原因拆解 ← 诊断策略问题最快的入口
res['signals_per_day']  # 日均信号数，判断信号松紧
```

### `exits` 的读法

| 主导出场原因 | 含义 | 调整方向 |
|---|---|---|
| `stop_loss` 占比高 | 「洗盘 vs 破位」没区分开 | 收紧 `pb_max_atr`，或降低 `shrink_ratio`（要求更明显的缩量） |
| `break_ma` 占比高**且平均亏损** | MA 止损对本场景太紧，赢家在启动前被砍 | 见下方「已知问题」 |
| `timeout` 占比高**且收益≈0** | 信号只是噪音，没有真实边缘 | 重新审视趋势层筛选 |
| `trail_stop` 贡献主要利润 | ✅ 健康形态 |

### 红线指标 — 出现即怀疑回测有问题

| 现象 | 最可能的原因 |
|---|---|
| 胜率 > 65% **且** 盈亏比 > 2 | 几乎必然存在未来函数 |
| 年化 > 60% **且** 最大回撤 < 15% | 检查交易成本、涨跌停、停牌是否真的生效 |
| 交易次数 < 50 | 样本不足，**不得下任何结论** |
| 单一年份贡献 > 60% 总收益 | 这不是策略，是一次运气 |
| `signals_per_day` 全市场 > 50 | 信号过松，等同于宽基动量组合 |

### 必须同时报告的基准

绝对收益单独看**没有意义**。此策略天然暴露于「小盘 + 高波动 + 动量」，A股在 2019–2021 这类环境下任何动量策略都能跑出漂亮曲线。至少并列三条基准：

1. 中证 1000 / 中证 2000 同期
2. 等权持有全部「强势股」（**不等回调直接买**）
3. 同池随机选股

**只有跑赢第 2 条，「等回调」这个动作才有超额。** 这是本策略唯一需要证明的命题。

---

## 6. Agent 禁止事项

以下行为会产出看起来很美、实则无效的结果。**违反任一条，结果作废。**

### 关于回测诚实性

- ❌ 降低或删除 `commission` / `stamp_tax` / `slippage`
- ❌ 修改 T+1 约束、允许当日买卖
- ❌ 允许在 `one_word_up` 或 `paused` 日成交
- ❌ 在 `generate_signals()` 中引入任何 `shift(-n)`、全样本 `mean/std/rank`，或按 `code` 分组后的全历史统计量
- ❌ 用当前 ST / 退市状态回溯过滤历史样本
- ❌ 把 `min_amount` 调到极低值以增加样本——选出的股票实盘无法成交

### 关于参数

- ❌ 网格搜索到单点最优就上报。**必须验证参数平原**：邻近取值结果应连续变化。若 `pb_days_max` 从 7 改到 6 让年化从 18% 跳到 31%，那是噪音不是参数。
- ❌ 用全区间调参后用同一区间报告绩效。样本外区间**建议直接用 2022 年至今**（而非随机切分）——2021 年后量化生态密度上升、短周期动量信号拥挤度提高，这段能站住才有实盘意义。
- ❌ 同时调整 3 个以上参数。一次一个，记录每次的 `exits` 变化。

### 关于结论

- ❌ 在 GATE 1 未通过时报告绝对收益而不加标注
- ❌ 把回测年化直接表述为「预期收益」
- ❌ 省略 §5 的基准对比

---

## 7. 已知问题

### 7.1 `break_ma` 过早止损（未解决，欢迎实验）

合成数据冒烟测试中，`exit_below_ma=True` 触发了 67% 的出场且平均亏损，而熬到 `timeout` 的持仓平均 +17%、胜率 98%。

原因是结构性的：本策略买的就是**刚回调过**的票，成本价天然贴近 MA20，稍有反复即被扫出，赢家在真正启动前被砍掉。真实数据上大概率重现。

可实验方向（逐个试，别一起改）：
- `exit_below_ma=False`，只留硬止损
- 改成「连续 2 日收盘破 MA20」才出场
- 持仓前 3 日豁免此规则

> 注：上述数字来自**人工注入形态的合成数据**，收益量级无参考价值；但出场分布的结构性倾斜与数据真假无关。

### 7.2 其他

- `load_stk_limit()` 一次性读入全市场全历史 parquet，长区间需注意内存。全市场 2018–2026 约 1000 万行，长表约 400–600 MB
- `data_loader._TABLES` 是死代码，未被引用
- `data_loader.trading_days()` 用 f-string 拼日期，与其他函数的参数绑定风格不一致
- 停牌日在 `daily` 表中直接缺行，`build_panel()` pivot 后为 NaN，回测中按成本价估值

---

## 8. 扩展点

| 需求 | 入口 |
|---|---|
| 换趋势定义（如相对指数 RS） | `generate_signals()` 的 `strong` 块 |
| 换回调形态（如平台整理） | `generate_signals()` 的 `pullback` 块 |
| 换持仓排序（如小市值优先） | `generate_signals()` 返回的 `score`，接 `daily_basic.circ_mv` |
| 加行业中性 / 市值中性 | 在 `backtest()` 的 §E 入场排序处加约束 |
| 加指数择时开关 | `index_daily`（000300.SH）或 `load_fetched_index()` |

---

## 9. 免责

本项目仅用于量化研究与回测方法学验证。回测表现与实盘之间隔着资金容量、涨停板成交概率、冲击成本估计误差与执行纪律等多重差距，历史规律亦可能失效。**不构成任何投资建议。**
