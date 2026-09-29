"""yfinance の一括ダウンロードで全市場の株価・配当・分割を取る。

実測（2026-09-15）: 10銘柄×10年で 2.1 秒。100銘柄ずつ投げれば全市場3,700銘柄が
数分で終わる。個別に yf.Ticker を叩くと1銘柄あたり数リクエストかかって現実的な
時間に収まらないので、ここは必ず一括で取る。

auto_adjust=False でも Dividends 列と Close 列はどちらも株式分割調整済みで返る
（NTT 9432 で検証: 2016年 1.2円/回 → 2025年 2.65円/回。100:1・2:1・2:1・25:1 の
分割をまたいでも利回り DPS/Close が時系列で一貫する）。

**ただし例外がひとつある。権利落ち日と分割日がぴったり同じ日の配当だけ、
分割が当たらないまま返ってくる。** 調整されるのは「その配当より後に起きた分割」
なので、同じ日の分割は「後」に入らないため。

実測（2026-09-23）: 日本製鉄 5401 は 2025-09-29 に 5:1 の分割をしていて、同じ日の
配当が 60円 のまま入っていた（前後の配当は 16円・12円）。年間配当が 24円 のところ
72円 と3倍で積まれ、配当性向が 2193%（実際は数十%）になっていた。
全市場では 149件・144銘柄が同じ状態だった。
"""
from __future__ import annotations

import time
import warnings
from datetime import datetime, timedelta
from typing import Callable, Iterable

import pandas as pd
import yfinance as yf

from modules.quality import (FROZEN_WEEKS, filler_bars, frozen_mask, last_bad_jump,
                             split_is_sane)

warnings.filterwarnings("ignore", category=FutureWarning)

ProgressFn = Callable[[float, str], None]

# 東証の大引けは 15:30（クロージングオークション込み）。それ以前に取得すると
# 当日の未確定バーが混ざるので落とす。
_MARKET_CLOSE_HOUR = 15
_MARKET_CLOSE_MINUTE = 30


def _now_jst() -> datetime:
    return datetime.utcnow() + timedelta(hours=9)


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    """未確定バーと Close 欠損行を落とす。

    yfinance は日本時間の深夜に Open/High/Low だけ入って Close が NaN の行を返す
    ことがある。これを放置すると全指標が NaN 化して「候補ゼロ」になる事故が起きる。
    """
    if df is None or df.empty:
        return pd.DataFrame()
    df = df[~df.index.duplicated(keep="last")].sort_index()

    # 先頭と末尾の Close が NaN の行を落とす。
    # 複数銘柄を一括ダウンロードすると全銘柄が共通の日付インデックスになり、
    # まだ上場していなかった期間が NaN で埋められて返る。これを落とさないと
    # 全銘柄の「上場日」がバッチ内の最古の日付になってしまい、上場年数の
    # ゲートが機能しなくなる（実測: 3,707銘柄すべてが 2000-01-04 上場になった）。
    close = df["Close"]
    first_valid, last_valid = close.first_valid_index(), close.last_valid_index()
    if last_valid is None:
        return pd.DataFrame()
    df = df.loc[first_valid:last_valid]

    # 大引け前の当日バーを落とす
    now = _now_jst()
    before_close = (now.hour, now.minute) < (_MARKET_CLOSE_HOUR, _MARKET_CLOSE_MINUTE)
    if before_close and len(df) and df.index[-1].date() == now.date():
        df = df.iloc[:-1]
    return df


def _weekly(df: pd.DataFrame) -> pd.DataFrame:
    """週足に落とす。全市場の日足を持つと2,000万行を超えるため。

    売買の無かった日の埋め草（出来高0・四本値が直前の終値）は終値として使わない。
    1週間まるごと埋め草なら、その週は行ごと無し（=欠損）にする。
    日足そのものからは消さない。配当・分割がその日に付いていることがあるため。
    """
    close = df["Close"].mask(filler_bars(df))
    w = pd.DataFrame({
        "close": close.resample("W-FRI").last(),
        "volume": df["Volume"].resample("W-FRI").sum(),
    }).dropna(subset=["close"])
    if len(w) >= FROZEN_WEEKS:
        frozen = frozen_mask(pd.DataFrame({"ticker": "_", "date": w.index,
                                           "close": w["close"].values}))
        w = w[~frozen.to_numpy()]
    return w


def _quote(ticker: str, df: pd.DataFrame) -> dict:
    """日足から最新のスナップショット指標を作る。"""
    close = df["Close"].dropna()
    if close.empty:
        return {}
    d1y = df.tail(252)
    c1y = d1y["Close"].dropna()
    turnover = (df["Close"] * df["Volume"]).tail(20).mean()
    hi = float(c1y.max()) if len(c1y) else None
    lo = float(c1y.min()) if len(c1y) else None
    last = float(close.iloc[-1])
    pos = (last - lo) / (hi - lo) if hi is not None and lo is not None and hi > lo else None
    ret_1y = last / float(c1y.iloc[0]) - 1 if len(c1y) > 200 else None
    return {
        "ticker": ticker,
        "asof": close.index[-1].strftime("%Y-%m-%d"),
        "last_close": last,
        "high_52w": hi,
        "low_52w": lo,
        "pos_52w": pos,
        "avg_turnover": float(turnover) if pd.notna(turnover) else None,
        "ret_1y": ret_1y,
        "listing_start": df.index[0].strftime("%Y-%m-%d"),
        "n_bars": int(len(df)),
    }


def _events(ticker: str, df: pd.DataFrame, column: str, value_name: str) -> pd.DataFrame:
    if column not in df.columns:
        return pd.DataFrame(columns=["ticker", "date", value_name])
    s = df[column]
    s = s[s > 0]
    if s.empty:
        return pd.DataFrame(columns=["ticker", "date", value_name])
    return pd.DataFrame({
        "ticker": ticker,
        "date": s.index.strftime("%Y-%m-%d"),
        value_name: s.values.astype(float),
    })


def fix_same_day_split_dividends(div: pd.DataFrame, splits: pd.DataFrame
                                 ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """権利落ち日と分割日が同じ日の配当に、分割を当て直す。

    yfinance が調整してくれるのは「その配当より**後**に起きた分割」だけなので、
    同じ日の分割だけが漏れる（モジュール冒頭の説明を参照）。

    直すのは、次の両方を満たすものだけ。
      ・権利落ち日が分割日とぴったり同じ
      ・前後2回ずつの配当と比べて、ちょうど分割の比率ぶん大きい

    2つ目を必ず重ねる。同じ日に記念配当が乗っていた場合に、本物の増配を
    分割のせいだと決めつけて割ってしまうのを防ぐため。基準を「全期間の中央値」に
    すると、過去に何度も分割した銘柄（KDDIなど）で昔の配当が小さく調整されている
    せいで誤検出する。実測で KDDI 2025-03-28 が誤って引っかかった。

    返り値は (直した配当, 直した明細)。
    """
    empty = pd.DataFrame(columns=["ticker", "date", "amount", "ratio", "before", "after"])
    if div is None or div.empty or splits is None or splits.empty:
        return div, empty

    sp = splits[splits["ratio"] > 1.2][["ticker", "date", "ratio"]]
    if sp.empty:
        return div, empty

    out = div.copy()
    key = out["ticker"].astype(str) + "|" + out["date"].astype(str)
    sp_key = dict(zip(sp["ticker"].astype(str) + "|" + sp["date"].astype(str), sp["ratio"]))

    changed = []
    for ticker, g in out.groupby("ticker", sort=False):
        g = g.sort_values("date")
        idx = list(g.index)
        if len(idx) < 4:
            continue
        for pos, i in enumerate(idx):
            ratio = sp_key.get(f"{ticker}|{out.at[i, 'date']}")
            if not ratio:
                continue
            nb = [out.at[j, "amount"] for j in idx[max(0, pos - 2):pos] + idx[pos + 1:pos + 3]]
            if len(nb) < 2:
                continue
            base = float(pd.Series(nb).median())
            v = float(out.at[i, "amount"])
            if base <= 0 or v / base <= 1.5:
                continue
            if not (0.7 <= (v / ratio) / base <= 1.45):
                continue
            out.at[i, "amount"] = v / ratio
            changed.append({"ticker": ticker, "date": out.at[i, "date"], "amount": v / ratio,
                            "ratio": ratio, "before": v, "after": v / ratio})

    return out, (pd.DataFrame(changed) if changed else empty)


def download_batch(tickers: list[str], period: str = "max",
                   retries: int = 2) -> dict[str, pd.DataFrame]:
    """1バッチ分を一括ダウンロードして、銘柄ごとの日足に分解する。"""
    if not tickers:
        return {}
    last_err: Exception | None = None
    for attempt in range(retries + 1):
        try:
            raw = yf.download(
                tickers, period=period, interval="1d", actions=True,
                group_by="ticker", auto_adjust=False, threads=True, progress=False,
            )
            break
        except Exception as exc:  # ネットワーク断・レート制限
            last_err = exc
            if attempt == retries:
                raise
            time.sleep(3 * (attempt + 1))
    else:  # pragma: no cover
        raise last_err  # type: ignore[misc]

    out: dict[str, pd.DataFrame] = {}
    if raw is None or raw.empty:
        return out
    for t in tickers:
        try:
            sub = raw[t] if isinstance(raw.columns, pd.MultiIndex) else raw
        except KeyError:
            continue
        cleaned = _clean(sub)
        if not cleaned.empty:
            out[t] = cleaned
    return out


def scan(tickers: Iterable[str], config: dict,
         progress: ProgressFn | None = None) -> dict[str, pd.DataFrame]:
    """全銘柄をバッチで取得し、DB に入れる形の DataFrame 群にまとめる。

    Returns
    -------
    dict
        "prices"（週足）/ "dividends" / "splits" / "quotes" の4つの DataFrame。
    """
    tickers = list(tickers)
    batch_size = int(config["universe"]["bulk_batch_size"])
    period = config["universe"]["price_period"]

    weekly_parts, div_parts, split_parts, quotes = [], [], [], []
    broken: list[dict] = []
    n_batches = (len(tickers) + batch_size - 1) // batch_size

    for i in range(n_batches):
        chunk = tickers[i * batch_size:(i + 1) * batch_size]
        if progress:
            progress(i / n_batches, f"株価取得 {i * batch_size + 1}〜{i * batch_size + len(chunk)} / {len(tickers)}")
        try:
            frames = download_batch(chunk, period=period)
        except Exception as exc:
            print(f"⚠️  バッチ {i + 1}/{n_batches} 取得失敗: {exc}")
            continue

        for t, df in frames.items():
            w = _weekly(df)
            # 古い株式分割が未調整のまま残っている銘柄がある（実測: 3,707銘柄中39銘柄。
            # 8303.T は2023年に比率 5e-08 の分割が入り終値が553億円に跳ねていた）。
            # 銘柄ごと捨てず、破損箇所より後だけ残す。
            at = last_bad_jump(w["close"]) if not w.empty else None
            if at is not None:
                broken.append({"ticker": t, "破損日": at.strftime("%Y-%m-%d"),
                               "残した行数": int((w.index > at).sum())})
                w = w[w.index > at]
                df = df[df.index > at]
            if not w.empty:
                weekly_parts.append(pd.DataFrame({
                    "ticker": t,
                    "date": w.index.strftime("%Y-%m-%d"),
                    "close": w["close"].values,
                    "volume": w["volume"].values,
                }))
            div_parts.append(_events(t, df, "Dividends", "amount"))
            sp = _events(t, df, "Stock Splits", "ratio")
            if not sp.empty:
                sp = sp[sp["ratio"].map(split_is_sane)]
            split_parts.append(sp)
            q = _quote(t, df)
            if q:
                quotes.append(q)

    if progress:
        progress(1.0, f"株価取得 完了（{len(quotes)} 銘柄"
                      + (f" / データ破損を一部除外 {len(broken)} 銘柄）" if broken else "）"))
    for b in broken:
        print(f"  {b['ticker']}: {b['破損日']} 以前を破損として除外（残り {b['残した行数']} 行）")

    def _concat(parts, cols):
        parts = [p for p in parts if p is not None and not p.empty]
        return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=cols)

    dividends = _concat(div_parts, ["ticker", "date", "amount"])
    splits = _concat(split_parts, ["ticker", "date", "ratio"])
    dividends, fixed = fix_same_day_split_dividends(dividends, splits)
    if not fixed.empty:
        print(f"  権利落ち日と分割日が重なって分割が漏れていた配当を直した: "
              f"{len(fixed)} 件 / {fixed['ticker'].nunique()} 銘柄")

    return {
        "prices": _concat(weekly_parts, ["ticker", "date", "close", "volume"]),
        "dividends": dividends,
        "splits": splits,
        "quotes": pd.DataFrame(quotes),
        "broken": pd.DataFrame(broken),
        "div_fixed": fixed,
    }
