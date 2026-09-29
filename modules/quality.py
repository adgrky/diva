"""データ品質のチェック。

yfinance には壊れた銘柄が混ざる。実測した例:
    8303.T（旧新生銀行）… 2023年に比率 5e-08 の「分割」が記録されており、
    週足終値が 553億円 に跳ねていた。この1銘柄のせいで検証のトータルリターン
    平均が +480,000% になり、指標の良し悪しがまったく読めなくなった。

順位相関は外れ値に強いので影響を受けないが、平均・分位の集計は壊れる。
取り込みの時点で弾き、集計の時点でも念のため落とす。
"""
from __future__ import annotations

import pandas as pd

# 株価が1週間で何倍まで動きうるか。分割調整済みの終値なので、
# これを超える段差はデータ破損とみなす（ストップ高連続でも20倍は動かない）。
MAX_WEEKLY_JUMP = 20.0
# 分割比率として妥当な範囲。1/1000（1000株併合）〜1000倍分割まで。
MIN_SPLIT_RATIO = 1e-3
MAX_SPLIT_RATIO = 1e3
def split_is_sane(ratio: float) -> bool:
    return MIN_SPLIT_RATIO <= float(ratio) <= MAX_SPLIT_RATIO


# ─── 止まった株価（2026-09-26 に発見）
# yfinance は売買が成立しなかった日・データの無い日（上場廃止中など）に、
# 「始値=高値=安値=終値=直前の終値、出来高0」の埋め草の行を返す。Close が NaN では
# ないので、そのまま週足にすると同じ終値が何年も並ぶ。
# 実測: 7564.T が 2010-04〜2018-09 に 286.25 のまま（その後 ~3,515 に跳ねる）、
# 9204.T（スカイマーク、破綻→再上場）が 2015〜2022 に 26.00 のまま。
# この区間をまたいで ret_1y を取ると +1546% のような偽のリターンになる。
#
# 同じ終値がこの週数以上続いたら、出来高があっても止まっているとみなす。
# 埋め草を落とした後も、ごく僅かな出来高を伴って 1.00 円が10年続く銘柄（7872.T）がある。
# 13週や26週にしてはいけない。低位株は呼び値1円で本当に動かないことがあり、
# 毎週売買がありながら同じ終値が13〜24週続く区間が23か所ある（実測: 8107.T が
# 18円・30円・90円で何度も）。1年動かないものだけを止まっているとみなす。
FROZEN_WEEKS = 52
# 「その日の株価」として何日前の終値までなら使ってよいか。週足なので祝日の週を
# 挟んでも3週間あれば届く。これより古い値を使うと、欠けた区間の外の値を
# 黙って流用することになる（欠損を落としても「直前の値」を拾えば同じ事故が起きる）。
MAX_STALE_DAYS = 31


def filler_bars(daily: pd.DataFrame) -> pd.Series:
    """日足のうち yfinance の埋め草（出来高0で四本値がすべて同じ）の行。"""
    if daily is None or daily.empty or not {"Open", "High", "Low", "Close", "Volume"} <= set(daily):
        return pd.Series(False, index=getattr(daily, "index", None), dtype=bool)
    c = daily["Close"]
    return (daily["Volume"].fillna(0).eq(0) & daily["Open"].eq(c)
            & daily["High"].eq(c) & daily["Low"].eq(c))


def frozen_mask(prices: pd.DataFrame, min_weeks: int = FROZEN_WEEKS) -> pd.Series:
    """縦持ちの週足（ticker, date, close）で、止まった区間の2行目以降を True にする。

    区間の最初の1行は本物の最後の取引なので残す。
    """
    if prices is None or prices.empty:
        return pd.Series(False, index=getattr(prices, "index", None), dtype=bool)
    p = prices.sort_values(["ticker", "date"])
    same = p.groupby("ticker", sort=False)["close"].diff().eq(0)
    rid = (~same).cumsum()
    n = rid.map(rid.value_counts())
    return (same & (n >= min_weeks)).reindex(prices.index)


def frozen_runs(prices: pd.DataFrame, min_weeks: int = FROZEN_WEEKS) -> pd.DataFrame:
    """止まった区間の一覧（ticker, start, end, weeks, close）。監査用。"""
    cols = ["ticker", "start", "end", "weeks", "close"]
    if prices is None or prices.empty:
        return pd.DataFrame(columns=cols)
    p = prices.sort_values(["ticker", "date"]).reset_index(drop=True)
    same = p.groupby("ticker", sort=False)["close"].diff().eq(0)
    p["rid"] = (~same).cumsum()
    r = p.groupby("rid").agg(ticker=("ticker", "first"), start=("date", "first"),
                             end=("date", "last"), weeks=("close", "size"),
                             close=("close", "first"))
    return r[r["weeks"] >= min_weeks][cols].reset_index(drop=True)


def price_asof(s: pd.Series, when: pd.Timestamp,
               max_stale_days: int = MAX_STALE_DAYS) -> float:
    """when 以前で最後の終値。ただし古すぎれば NaN（欠けた区間を流用しない）。

    s は日付インデックスで昇順に並んだ終値。
    """
    i = s.index.searchsorted(when, side="right")
    if i == 0:
        return float("nan")
    if (when - s.index[i - 1]).days > max_stale_days:
        return float("nan")
    return float(s.iloc[i - 1])
def winsorize(s: pd.Series, lower: float = 0.01, upper: float = 0.99) -> pd.Series:
    """裾を刈って平均を使えるようにする。中央値を見るなら不要。"""
    v = s.dropna()
    if v.empty:
        return s
    lo, hi = v.quantile(lower), v.quantile(upper)
    return s.clip(lo, hi)


def last_bad_jump(close: pd.Series) -> pd.Timestamp | None:
    """最後に起きた異常な段差の日付。無ければ None。"""
    c = close.dropna()
    if len(c) < 2:
        return None
    ratio = (c / c.shift(1)).dropna()
    bad = ratio[(ratio > MAX_WEEKLY_JUMP) | (ratio < 1 / MAX_WEEKLY_JUMP)]
    return bad.index.max() if len(bad) else None
def trim_frame(prices: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """縦持ちの価格テーブルから破損区間を落とす。

    Returns
    -------
    (トリム後のテーブル, 何をどれだけ落としたかの一覧)
    """
    if prices is None or prices.empty:
        return prices, pd.DataFrame()
    keep, notes = [], []
    for ticker, g in prices.groupby("ticker", sort=False):
        g = g.copy()
        idx = pd.to_datetime(g["date"])
        s = pd.Series(g["close"].values, index=idx)
        at = last_bad_jump(s)
        if at is None:
            keep.append(g)
            continue
        mask = (idx > at).values
        notes.append({"ticker": ticker, "破損日": at.strftime("%Y-%m-%d"),
                      "落とした行数": int((~mask).sum()), "残した行数": int(mask.sum())})
        if mask.any():
            keep.append(g[mask])
    out = pd.concat(keep, ignore_index=True) if keep else prices.iloc[0:0]
    return out, pd.DataFrame(notes)
