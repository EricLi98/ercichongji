# -*- coding: utf-8 -*-
"""
A股「强势股回调」策略框架
=========================
逻辑：趋势层筛强势 → 形态层等回调 → 信号层等企稳 → 风控层管出场
执行：T 日收盘产生信号，T+1 开盘成交（无未来函数）

数据接口约定
------------
输入长表 DataFrame，必需列：
    date      datetime64   交易日
    code      str          证券代码，如 '000001.SZ'
    open/high/low/close     **后复权**价格（务必复权，否则除权日会误判为暴跌）
    volume    float        成交量(股)
    amount    float        成交额(元)

可选列（强烈建议提供，缺失时退化为近似判断，回测会偏乐观）：
    limit_up    bool  当日涨停
    limit_down  bool  当日跌停
    paused      bool  当日停牌
    is_st       bool  当日 ST/*ST
    list_days   int   上市天数
    board       str   'MAIN' / 'GEM'(创业板) / 'STAR'(科创板) / 'BJ'(北交所)

⚠️ 股票池必须包含**已退市**股票，否则存在幸存者偏差。
"""

from dataclasses import dataclass, field
import numpy as np
import pandas as pd


# ============================================================
# 参数
# ============================================================

@dataclass
class Config:
    # ---- 趋势层：什么叫「强势」 ----
    mom_window: int = 60        # 动量回看窗口（交易日）
    top_pct: float = 0.20       # 取动量全市场前 20%
    high_lookback: int = 20     # 近 N 日内必须创过 mom_window 日新高

    # ---- 形态层：什么叫「短暂调整」 ----
    pb_window: int = 10         # 阶段高点的回看窗口
    use_atr: bool = True        # True: 用 ATR 归一化回撤（推荐，兼容 10cm/20cm）
    pb_min_atr: float = 1.0     # 最浅回撤（ATR 倍数）
    pb_max_atr: float = 4.0     # 最深回撤（ATR 倍数）
    pb_min_pct: float = 0.03    # use_atr=False 时启用
    pb_max_pct: float = 0.15
    pb_days_min: int = 2        # 回调至少持续 2 日（排除单日插针）
    pb_days_max: int = 7        # 超过 7 日不算「短暂」
    shrink_ratio: float = 0.90  # VOL5/VOL20 阈值，缩量=洗盘，放量=出货
    ma_support: int = 20        # 回调不得跌破的均线

    # ---- 信号层：企稳买点 ----
    vol_up: float = 1.20        # 触发日成交量 / 前5日均量

    # ---- 过滤 ----
    min_amount: float = 5e7     # 20日均成交额下限（元），保证可容纳资金
    min_list_days: int = 250    # 剔除次新股
    exclude_st: bool = True

    # ---- 组合与风控 ----
    init_cash: float = 1_000_000
    max_positions: int = 10
    stop_loss: float = 0.07     # 相对成本价硬止损
    trail_stop: float = 0.12    # 自持仓最高收盘价回撤止盈
    max_hold: int = 15          # 最长持有交易日
    exit_below_ma: bool = True  # 收盘跌破 MA20 离场

    # ---- 交易成本 ----
    commission: float = 0.00025  # 双边佣金
    stamp_tax: float = 0.0005    # 卖出印花税（2023-08 后为 0.05%）
    slippage: float = 0.0015     # 冲击成本，小盘股建议调高

    limit_pct: dict = field(default_factory=lambda: {
        'MAIN': 0.10, 'GEM': 0.20, 'STAR': 0.20, 'BJ': 0.30})


# ============================================================
# 1. 长表 → 宽表面板
# ============================================================

def build_panel(df: pd.DataFrame, cfg: Config) -> dict:
    """转成 dict[str, DataFrame(index=date, columns=code)]，并补齐涨跌停/可交易标记。"""
    df = df.sort_values(['date', 'code'])
    px = {}
    for c in ['open', 'high', 'low', 'close', 'volume', 'amount']:
        px[c] = df.pivot(index='date', columns='code', values=c).sort_index()

    close, open_, high, low = px['close'], px['open'], px['high'], px['low']
    prev_close = close.shift(1)

    # --- 涨跌停 ---
    if 'limit_up' in df.columns:
        px['limit_up'] = df.pivot(index='date', columns='code',
                                  values='limit_up').reindex_like(close).fillna(False)
        px['limit_down'] = df.pivot(index='date', columns='code',
                                    values='limit_down').reindex_like(close).fillna(False)
    else:
        # 近似：按板块涨跌幅上限判断。后复权价在除权日会有误差，仅作降级方案。
        lim = _board_limit(df, close, cfg)
        ret = close / prev_close - 1
        px['limit_up'] = ret >= lim - 1e-4
        px['limit_down'] = ret <= -lim + 1e-4
        px['_limit'] = lim

    # --- 停牌 / 可交易 ---
    paused = (px['volume'].fillna(0) <= 0) | close.isna()
    if 'paused' in df.columns:
        p2 = df.pivot(index='date', columns='code', values='paused')
        paused = paused | p2.reindex_like(close).fillna(True)

    # 一字涨停：开盘即封板，买不进
    if 'one_word_up' in df.columns:                       # 适配层已算好（基于真实涨停价）
        one_word_up = df.pivot(index='date', columns='code',
                               values='one_word_up').reindex_like(close).fillna(False)
    else:
        lim = px.get('_limit', _board_limit(df, close, cfg))
        one_word_up = (open_ >= prev_close * (1 + lim) - 1e-4) & (high <= low + 1e-6)

    # 注意：limit_up 指「收盘封板」，当日盘中仍可能成交，不能据此禁止开盘买入。
    # 能否以开盘价成交，只取决于停牌与一字板。
    px['can_buy'] = ~paused & ~one_word_up.astype(bool)
    px['can_sell'] = ~paused & ~px['limit_down'].fillna(False)

    # --- 基础过滤器 ---
    ok = px['amount'].rolling(20).mean() > cfg.min_amount
    if 'list_days' in df.columns:
        ld = df.pivot(index='date', columns='code', values='list_days').reindex_like(close)
        ok &= ld >= cfg.min_list_days
    if cfg.exclude_st and 'is_st' in df.columns:
        st = df.pivot(index='date', columns='code', values='is_st').reindex_like(close).fillna(False)
        ok &= ~st.astype(bool)
    px['tradable'] = ok.fillna(False)

    return px


def _board_limit(df, close, cfg):
    """返回与 close 同形的涨跌幅上限矩阵。"""
    if 'board' in df.columns:
        b = df.groupby('code')['board'].last()
    else:  # 用代码前缀推断
        b = pd.Series(index=close.columns, dtype=object)
        for c in close.columns:
            s = str(c)
            b[c] = ('STAR' if s.startswith('688') else
                    'GEM' if s.startswith('30') else
                    'BJ' if s.startswith(('8', '43', '92')) else 'MAIN')
    vals = b.reindex(close.columns).map(cfg.limit_pct).fillna(0.10).astype(float)
    return pd.DataFrame(np.tile(vals.values, (len(close), 1)),
                        index=close.index, columns=close.columns)


# ============================================================
# 2. 信号生成（全向量化）
# ============================================================

def generate_signals(px: dict, cfg: Config, layers: bool = False):
    """
    返回 (signal, score)；layers=True 时额外返回各层掩码，供对照实验拆解。
    signal 为买入信号布尔矩阵，score 用于名额不足时排序。
    """
    close, high, low, open_, vol = (px['close'], px['high'], px['low'],
                                    px['open'], px['volume'])

    ma_s = close.rolling(cfg.ma_support).mean()
    ma_l = close.rolling(cfg.mom_window).mean()
    v5, v20 = vol.rolling(5).mean(), vol.rolling(20).mean()

    # ---------- 趋势层：强势 ----------
    mom = close / close.shift(cfg.mom_window) - 1
    mom_rank = mom.rank(axis=1, pct=True)           # 截面排名，无未来信息

    roll_high = high.rolling(cfg.mom_window).max()
    made_new_high = (high >= roll_high).rolling(cfg.high_lookback).max().fillna(0).astype(bool)

    strong = (
        (mom_rank >= 1 - cfg.top_pct)
        & (close > ma_l)
        & (ma_s > ma_l)                              # 均线多头排列
        & made_new_high
    )

    # ---------- 形态层：短暂调整 ----------
    swing_high = high.rolling(cfg.pb_window).max()
    drop = swing_high - close                        # 绝对回撤金额

    if cfg.use_atr:
        atr = _atr(high, low, close, 14)
        depth_ok = (drop >= cfg.pb_min_atr * atr) & (drop <= cfg.pb_max_atr * atr)
    else:
        dd = close / swing_high - 1
        depth_ok = (dd <= -cfg.pb_min_pct) & (dd >= -cfg.pb_max_pct)

    days_since_high = _days_since(high >= swing_high, close)
    days_ok = (days_since_high >= cfg.pb_days_min) & (days_since_high <= cfg.pb_days_max)

    shrink = (v5 / v20) < cfg.shrink_ratio           # 缩量 = 洗盘；放量回调多为出货
    not_broken = close > ma_s                        # 未破位

    pullback = depth_ok & days_ok & shrink & not_broken

    # ---------- 信号层：企稳启动 ----------
    trigger = (
        (close > open_) & (close > close.shift(1))    # 阳线且收高
        & ((vol > v5.shift(1) * cfg.vol_up) | (close > high.shift(1)))
    )

    # ---------- 合成 ----------
    base = px['tradable'] & ~px['limit_up'].fillna(False)   # 可交易 且 不追涨停
    signal = (strong & pullback & trigger & base).fillna(False)

    if layers:
        return signal, mom_rank, {
            'strong': strong.fillna(False),
            'pullback': pullback.fillna(False),
            'trigger': trigger.fillna(False),
            'base': base.fillna(False),
        }
    return signal, mom_rank


def _atr(high, low, close, n=14):
    pc = close.shift(1)
    tr = pd.concat([(high - low).stack(),
                    (high - pc).abs().stack(),
                    (low - pc).abs().stack()], axis=1).max(axis=1).unstack()
    return tr.rolling(n).mean()


def _days_since(cond: pd.DataFrame, ref: pd.DataFrame) -> pd.DataFrame:
    """距离上一次 cond 为 True 的交易日数（同日为 0）。"""
    pos = pd.DataFrame(np.arange(len(ref))[:, None].repeat(ref.shape[1], axis=1),
                       index=ref.index, columns=ref.columns)
    last = pos.where(cond.fillna(False)).ffill()
    return pos - last


# ============================================================
# 3. 事件驱动回测（T+1、涨跌停、成本全部落地）
# ============================================================

def backtest(px: dict, signal: pd.DataFrame, score: pd.DataFrame, cfg: Config):
    close, open_ = px['close'], px['open']
    ma_s = close.rolling(cfg.ma_support).mean()

    dates, codes = close.index, close.columns
    C, O = close.values, open_.values
    CB, CS = px['can_buy'].values, px['can_sell'].values
    MS, SIG, SC = ma_s.values, signal.values, score.values

    cash = cfg.init_cash
    book = {}                      # col_idx -> dict
    pend_buy, pend_sell = [], []
    equity = np.zeros(len(dates))
    trades = []

    for t in range(len(dates)):
        # ---- A. 执行卖出（T+1 开盘） ----
        for j in list(pend_sell):
            if j not in book or not CS[t, j] or np.isnan(O[t, j]):
                continue                                   # 跌停/停牌，顺延
            p = O[t, j] * (1 - cfg.slippage)
            b = book.pop(j)
            gross = p * b['shares']
            cash += gross * (1 - cfg.commission - cfg.stamp_tax)
            trades.append(dict(code=codes[j], entry=b['entry'], exit=p,
                               open_date=b['date'], close_date=dates[t],
                               days=b['days'], ret=p / b['entry'] - 1,
                               reason=b['reason']))
        pend_sell = [j for j in pend_sell if j in book]

        # ---- B. 执行买入（T+1 开盘） ----
        equity_est = cash + sum(C[t, j] * b['shares'] for j, b in book.items()
                                if not np.isnan(C[t, j]))
        for j in pend_buy:
            if len(book) >= cfg.max_positions or j in book:
                continue
            if not CB[t, j] or np.isnan(O[t, j]):
                continue                                   # 一字板/停牌，放弃
            p = O[t, j] * (1 + cfg.slippage)
            target = min(cash / (1 + cfg.commission), equity_est / cfg.max_positions)
            shares = int(target / p // 100) * 100           # 整手
            if shares <= 0:
                continue
            cost = p * shares * (1 + cfg.commission)
            if cost > cash:
                continue
            cash -= cost
            book[j] = dict(shares=shares, entry=p, peak=C[t, j],
                           date=dates[t], days=0, reason='')
        pend_buy = []

        # ---- C. 盯市 ----
        mv = 0.0
        for j, b in book.items():
            b['days'] += 1
            px_now = C[t, j]
            if np.isnan(px_now):
                px_now = b['entry']                        # 停牌按成本估值
            b['peak'] = max(b['peak'], px_now)
            mv += px_now * b['shares']
        equity[t] = cash + mv

        if t == len(dates) - 1:
            break

        # ---- D. 出场判断（T 日收盘） ----
        for j, b in book.items():
            c = C[t, j]
            if np.isnan(c):
                continue
            r = None
            if c <= b['entry'] * (1 - cfg.stop_loss):
                r = 'stop_loss'
            elif c <= b['peak'] * (1 - cfg.trail_stop):
                r = 'trail_stop'
            elif cfg.exit_below_ma and not np.isnan(MS[t, j]) and c < MS[t, j]:
                r = 'break_ma'
            elif b['days'] >= cfg.max_hold:
                r = 'timeout'
            if r and b['days'] >= 1:                       # T+1 限制
                b['reason'] = r
                pend_sell.append(j)

        # ---- E. 入场排序（T 日收盘） ----
        slots = cfg.max_positions - len(book) + len(pend_sell)
        if slots > 0:
            cand = np.where(SIG[t] & ~np.isin(np.arange(len(codes)), list(book.keys())))[0]
            if len(cand):
                cand = cand[np.argsort(-np.nan_to_num(SC[t, cand]))][:slots]
                pend_buy = list(cand)

    eq = pd.Series(equity, index=dates)
    return eq, pd.DataFrame(trades)


# ============================================================
# 4. 绩效
# ============================================================

def performance(equity: pd.Series, trades: pd.DataFrame, freq: int = 242) -> dict:
    eq = equity[equity > 0]
    ret = eq.pct_change().dropna()
    years = len(eq) / freq
    dd = eq / eq.cummax() - 1
    win = trades['ret'] > 0 if len(trades) else pd.Series(dtype=bool)

    return {
        '总收益':   f"{eq.iloc[-1] / eq.iloc[0] - 1:.2%}",
        '年化':     f"{(eq.iloc[-1] / eq.iloc[0]) ** (1 / years) - 1:.2%}",
        '年化波动': f"{ret.std() * np.sqrt(freq):.2%}",
        '夏普':     f"{(ret.mean() * freq - 0.02) / (ret.std() * np.sqrt(freq)):.2f}",
        '最大回撤': f"{dd.min():.2%}",
        'Calmar':   f"{((eq.iloc[-1] / eq.iloc[0]) ** (1 / years) - 1) / abs(dd.min()):.2f}",
        '交易次数': len(trades),
        '胜率':     f"{win.mean():.2%}" if len(trades) else '-',
        '盈亏比':   (f"{trades.loc[win, 'ret'].mean() / abs(trades.loc[~win, 'ret'].mean()):.2f}"
                     if len(trades) and (~win).any() else '-'),
        '平均持仓': f"{trades['days'].mean():.1f}日" if len(trades) else '-',
    }


def exit_breakdown(trades: pd.DataFrame) -> pd.DataFrame:
    """按出场原因拆解 —— 诊断策略问题最快的入口。"""
    if not len(trades):
        return pd.DataFrame()
    return trades.groupby('reason')['ret'].agg(
        次数='count', 平均收益='mean', 胜率=lambda s: (s > 0).mean()
    ).sort_values('次数', ascending=False)


# ============================================================
# 5. 入口
# ============================================================

def run(df: pd.DataFrame, cfg: Config = None):
    cfg = cfg or Config()
    px = build_panel(df, cfg)
    sig, score = generate_signals(px, cfg)
    eq, trades = backtest(px, sig, score, cfg)
    return dict(equity=eq, trades=trades,
                stats=performance(eq, trades),
                exits=exit_breakdown(trades),
                signals_per_day=sig.sum(axis=1))


if __name__ == '__main__':
    # df = pd.read_parquet('astock_daily.parquet')
    # res = run(df)
    # print(pd.Series(res['stats']))
    # print(res['exits'])
    pass
