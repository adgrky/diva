"""配当履歴から「増配の実績」を測る。

このアプリで最も信頼できるデータがここ。yfinance の配当履歴は 2000年前後まで
遡れて（NTT で53件・26年分）、しかも株式分割調整済みで返る。財務諸表は5期分しか
取れないので、Phase 3 で EDINET を入れるまでは、増配実績の判定はこのモジュールが
一手に引き受ける。

【年度の切り方】
日本企業は中間（9月）と期末（3月）に分けて払うため、暦年で合算すると
「FY2025 = 2025年9月の中間 + 2026年3月の期末」がバラバラの年に散る。
決算期を推定する代わりに、**直近の権利落ち日を起点に1年ずつ遡ったバケット**で
合算する（pd.DateOffset(years=n) を使うので日付のズレが蓄積しない）。
年1回・年2回・年4回のどれでも同じロジックで正しく揃う。
このバケットは厳密には実質TTM（直近12ヶ月）であり、会計年度とは一致しない。
正確な会計年度ベースの DPS は Phase 3 の EDINET で入れる。

【記念配当・特別配当】
yfinance は普通配当と記念配当を区別しない。区別せずに扱うと、記念配当の翌年に
必ず「減配」が立ち、増配銘柄を誤って落としてしまう。ここでは
「一時的に跳ねて翌年に戻った年」をスパイクとして検出し、連続増配・減配回数の
判定からは除外する（値は消さず、フラグとして残す）。
"""
from __future__ import annotations

from datetime import date

import math
from dataclasses import dataclass, field, asdict

import pandas as pd

# スパイク判定：前年比でこの倍率を超え、かつ翌年に前年並みへ戻ったら記念配当を疑う
_SPIKE_UP = 1.40
_SPIKE_REVERT = 1.15   # 翌年が「スパイク前 × この倍率」以下なら戻ったとみなす
# 減配とみなす下落幅。1円未満の端数調整で減配判定が立たないようにする
_CUT_TOLERANCE = 0.995


_FY_TOLERANCE_DAYS = 30


def annual_dps(div: pd.DataFrame, fy_month: int | None = None) -> pd.DataFrame:
    """1銘柄の配当明細 -> 年度別 DPS。

    【年度の切り方 — 以前は直近の権利落ち日から1年ずつ遡っていた】
    権利落ち日は年によって数日ずれる（3月30日 → 3月28日 → 3月28日）。
    遡るうちにズレが積み上がり、**同じ年度に3回ぶん入る年**ができる。その翌年は
    2回ぶんしかないので、増減が無くても減配に見える。

        実測（1332 日本水産・2025年6月までで切った場合）
            2023-03-30 10円 ┐
            2023-09-28 10円 ├→ 全部「2024年度」に入って 34円
            2024-03-28 14円 ┘
            2024-09-27 12円 ┐→「2025年度」は 28円 ＝ 6円の減配に見える
            2025-03-28 16円 ┘

    3月期の会社の 84% がこれで誤って減配と判定されていた。

    そこで**決算月そのもの**で切る。直近の権利落ち日の月日を年度末とみなし、
    そこから30日以内の遅れは同じ年度に入れる（権利落ち日のズレを吸収する）。
    """
    if div is None or div.empty:
        return pd.DataFrame(columns=["dps", "n_payments", "period_end"])
    d = div.copy()
    d["date"] = pd.to_datetime(d["date"])
    d = d[d["amount"] > 0].sort_values("date")
    if d.empty:
        return pd.DataFrame(columns=["dps", "n_payments", "period_end"])

    # 年度末の月は、**直近の権利落ち日から取ってはいけない**。データを途中で
    # 切ったとき、最後の1回が中間配当だと、その月を年度末と勘違いして年度の
    # 区切りが半年ずれる（実測: 3月期の明細を9月で切ると年度が全部ずれた）。
    # 支払月の出現回数から決めれば、どこで切っても同じ答えになる。
    #
    # 【決算月が分かるときは、それを使う（2026-10-02）】
    # 年2回払う会社は中間と期末の回数が同じなので、支払月からは決算月が決まらない。
    # 同数のときは3月を優先していたため、9月決算の会社は半年ずれた年度で合算され、
    # 増配なのに減配と判定されていた。
    #     東陽テクニカ（9月決算）本当は 68円 → 69円 の増配
    #                            3月区切りだと 73円 → 69円 の「減配」
    # 財務（fundamentals.fiscal_end）から決算月が取れる銘柄は、そちらで切る。
    counts = d["date"].dt.month.value_counts()
    if fy_month:
        month = int(fy_month)
    else:
        top = counts.max()
        tied = [m for m in counts.index if counts[m] == top]
        # 同数のときの優先順（日本の決算期末の多い順）。ここを固定しておかないと、
        # データが1件増えるたびに年度の区切りが動いてしまう。
        order = [3, 12, 9, 6, 2, 5, 8, 11, 1, 4, 7, 10]
        month = min(tied, key=lambda m: order.index(m) if m in order else 99)
    days = d.loc[d["date"].dt.month == month, "date"].dt.day
    day = int(min(days.median(), 28)) if len(days) else 28

    def label(ts: pd.Timestamp) -> int:
        """その配当がどの年度のものか。年度末は anchor と同じ月日。"""
        fy_end = pd.Timestamp(year=ts.year, month=month, day=day)
        return ts.year if (ts - fy_end).days <= _FY_TOLERANCE_DAYS else ts.year + 1

    d["year"] = [label(ts) for ts in d["date"]]
    g = d.groupby("year")
    out = pd.DataFrame({
        "dps": g["amount"].sum(),
        "n_payments": g["amount"].size(),
        "period_end": [pd.Timestamp(year=int(y), month=month, day=day).strftime("%Y-%m-%d")
                       for y in g.groups],
    }).sort_index()
    out.index.name = "year"

    # 端の年が「途中まで」のときは落とす。
    # 中間配当だけ済んでいて期末配当がまだ、という年は回数が足りないので、
    # 残したままだと**毎年9月から3月までのあいだ、全社が減配したように見える**。
    # 同じことがデータを途中で切ったときにも起きる（検証で先を見ないために切る）。
    # 「いつもの回数」は**直近の完結した3年**で決める。全期間の最頻値にすると、
    # 昔は年1回だった会社（フェローテック・和田興産など）で「1回」が基準になり、
    # 中間配当だけ済んだ今年が残って減配に見えていた。
    if len(out) >= 3:
        usual = int(out["n_payments"].iloc[-4:-1].mode().max()) \
            if len(out) >= 4 else int(out["n_payments"].median())
        if out["n_payments"].iloc[-1] < usual:
            out = out.iloc[:-1]
        if len(out) >= 3 and out["n_payments"].iloc[0] < usual:
            out = out.iloc[1:]
    return out


def flag_spikes(dps: pd.Series) -> pd.Series:
    """記念配当・特別配当と思われる年に True を立てる。"""
    flags = pd.Series(False, index=dps.index)
    vals = dps.values
    for i in range(1, len(vals) - 1):
        prev, cur, nxt = vals[i - 1], vals[i], vals[i + 1]
        if prev <= 0:
            continue
        if cur >= prev * _SPIKE_UP and nxt <= prev * _SPIKE_REVERT:
            flags.iloc[i] = True
    return flags


@dataclass
class DividendProfile:
    """1銘柄の配当プロフィール。スコア計算はこれを入力に取る。"""
    ticker: str
    years_paying: int = 0
    dps_latest: float = 0.0
    dps_prev: float = 0.0
    streak: int = 0                 # 連続増配年数（据え置きは途切れないが伸びない）
    streak_no_cut: int = 0          # 減配なしで継続している年数
    cuts_10y: int = 0
    cuts_all: int = 0
    cagr_5y: float | None = None
    cagr_10y: float | None = None
    growth_latest: float | None = None
    payments_per_year: int = 0
    has_spike: bool = False
    series: dict = field(default_factory=dict)   # {年: DPS}

    def to_dict(self) -> dict:
        return asdict(self)


def _cagr(series: pd.Series, years: int) -> float | None:
    """n年の年率成長率。始点が0または欠損なら None。"""
    if len(series) < years + 1:
        return None
    end = series.iloc[-1]
    start = series.iloc[-(years + 1)]
    if start <= 0 or end <= 0:
        return None
    return float((end / start) ** (1 / years) - 1)


def build_profile(ticker: str, div: pd.DataFrame,
                  fy_month: int | None = None) -> DividendProfile:
    table = annual_dps(div, fy_month)
    if table.empty:
        return DividendProfile(ticker=ticker)

    dps = table["dps"]
    spikes = flag_spikes(dps)

    # 連続増配・減配回数はスパイク（記念配当）の年を挟んでも途切れないよう、
    # スパイク年を取り除いた系列で判定する
    clean = dps[~spikes]

    streak = 0
    streak_no_cut = 0
    vals = clean.values
    for i in range(len(vals) - 1, 0, -1):
        if vals[i] > vals[i - 1]:
            streak += 1
        else:
            break
    for i in range(len(vals) - 1, 0, -1):
        if vals[i] >= vals[i - 1] * _CUT_TOLERANCE:
            streak_no_cut += 1
        else:
            break

    cuts_all = int((clean.diff() < 0).sum()) if len(clean) > 1 else 0
    recent = clean[clean.index >= clean.index.max() - 10]
    cuts_10y = int((recent.diff() < 0).sum()) if len(recent) > 1 else 0

    growth = None
    if len(dps) >= 2 and dps.iloc[-2] > 0:
        growth = float(dps.iloc[-1] / dps.iloc[-2] - 1)

    return DividendProfile(
        ticker=ticker,
        years_paying=int(len(dps)),
        dps_latest=float(dps.iloc[-1]),
        dps_prev=float(dps.iloc[-2]) if len(dps) >= 2 else 0.0,
        streak=streak,
        streak_no_cut=streak_no_cut,
        cuts_10y=cuts_10y,
        cuts_all=cuts_all,
        cagr_5y=_cagr(dps, 5),
        cagr_10y=_cagr(dps, 10),
        growth_latest=growth,
        payments_per_year=int(table["n_payments"].iloc[-1]),
        has_spike=bool(spikes.any()),
        series={int(k): float(v) for k, v in dps.items()},
    )


def payout_months(dividends: pd.DataFrame, years: int = 3) -> pd.Series:
    """銘柄ごとの「配当がある月」を出す。

    yfinance の exDividendDate は直近の1回しか返さないので、3月決算の会社は
    9月（中間）しか出ず、期末の3月が見えない（実測で9月が479銘柄に偏った）。
    受け取り月を平準化したいときに使うのだから、年間のパターンが要る。

    Returns
    -------
    Series  index=ticker, value=月の集合（例 {3, 9}）
    """
    if dividends is None or dividends.empty:
        return pd.Series(dtype=object)
    d = dividends.copy()
    d["date"] = pd.to_datetime(d["date"])
    d = d[d["amount"] > 0]
    if d.empty:
        return pd.Series(dtype=object)
    cutoff = d["date"].max() - pd.DateOffset(years=years)
    d = d[d["date"] >= cutoff]
    return d.groupby("ticker")["date"].apply(lambda s: sorted(set(s.dt.month)))


def fiscal_months() -> dict[str, int]:
    """銘柄ごとの決算月（直近の決算期末の月）。財務が取れている銘柄だけ。"""
    from modules.store import read_df
    try:
        f = read_df("SELECT ticker, MAX(fiscal_end) AS fe FROM fundamentals GROUP BY ticker")
    except Exception:
        return {}
    out = {}
    for t, fe in f.itertuples(index=False):
        try:
            out[str(t)] = int(str(fe)[5:7])
        except (TypeError, ValueError):
            continue
    return out


def build_profiles(dividends: pd.DataFrame,
                   fy_months: dict[str, int] | None = None) -> dict[str, DividendProfile]:
    """全銘柄分をまとめて作る。dividends は ticker/date/amount の縦持ち。

    fy_months（銘柄→決算月）を渡すと、配当年度を決算期で区切る。
    渡さないと支払月から推定する（過去検証など、財務を使えない場面用）。
    """
    out: dict[str, DividendProfile] = {}
    if dividends is None or dividends.empty:
        return out
    fy_months = fy_months or {}
    for ticker, grp in dividends.groupby("ticker", sort=False):
        out[str(ticker)] = build_profile(str(ticker), grp, fy_months.get(str(ticker)))
    return out


def profiles_to_frame(profiles: dict[str, DividendProfile]) -> pd.DataFrame:
    if not profiles:
        # 列だけは揃えて返す。空の DataFrame を返すと、配当履歴が1件も無い保有
        # （無配・新規上場）だけのときに呼び出し側が df["dps_latest"] で落ちる。
        cols = [k for k in DividendProfile("").to_dict() if k not in ("ticker", "series")]
        return pd.DataFrame(columns=cols, index=pd.Index([], name="ticker"))
    rows = []
    for p in profiles.values():
        d = p.to_dict()
        d.pop("series", None)
        rows.append(d)
    return pd.DataFrame(rows).set_index("ticker")


def next_ex_dates(tickers: list[str] | None = None, lookback_years: int = 3,
                  horizon_days: int = 400) -> pd.DataFrame:
    """次の権利落ち日を、配当履歴から予測する。

    【なぜ予測するか】
    yfinance の ex_dividend_date は実測で **1,272社のうち5社しか入っていない**
    （保有86銘柄では0件）。そのままでは使えない。

    【どう置くか】
    日本株の権利落ち日は決算期末に固定されていて、実測では 2021〜2026年の
    25,085件のうち **23,643件（94.2%）が「その月の最終営業日の1営業日前」**。
    年をまたいでも比率は変わらない。

    そこで、その銘柄自身の過去の権利落ち日が「月の最終営業日の何営業日前か」を
    測り、同じ位置に置く。月末に張り付いていない銘柄（月の途中で落ちるもの）は
    同じ日付を使い、休業日なら手前の営業日にずらす。

    **月末を土日だけで計算してはいけない。** 祝日をまたぐ月でずれ、しかも
    ずれる方向が「実際より後ろ」になる。「その日までに買えば間に合う」と出した
    日にはもう権利が落ちていて、配当を1回取り逃がす。東証の休業日は
    modules/jp_calendar.py で持つ。

    **予測であって会社の発表ではない**ので、画面では必ずそう書くこと。
    """
    from modules.jp_calendar import (last_trading_day, prev_trading_day,
                                     shift_trading_days, trading_days_before_month_end)
    from modules.store import read_df
    if tickers:
        ph = ",".join("?" * len(tickers))
        div = read_df(f"SELECT ticker, date, amount FROM dividends WHERE ticker IN ({ph})",
                      tuple(tickers))
    else:
        div = read_df("SELECT ticker, date, amount FROM dividends")
    cols = ["ticker", "次の権利落ち日", "あと何日", "1株配当の目安", "根拠"]
    if div.empty:
        return pd.DataFrame(columns=cols)

    div["date"] = pd.to_datetime(div["date"])
    today = pd.Timestamp.today().normalize()
    recent = div[div["date"] >= today - pd.DateOffset(years=lookback_years)]
    if recent.empty:
        return pd.DataFrame(columns=cols)

    rows = []
    for ticker, g in recent.groupby("ticker"):
        g = g.sort_values("date")
        best = None
        for month, mg in g.groupby(g["date"].dt.month):
            last = mg.iloc[-1]
            hist = last["date"].date()
            # 月末からの位置（営業日単位）。0〜3営業日前なら「月末張り付き」とみなす
            offset = trading_days_before_month_end(hist)
            at_month_end = offset <= 3
            note = (f"前年の{int(month)}月は月末の{offset}営業日前"
                    if at_month_end else f"前年の{int(month)}月{hist.day}日")

            for add_year in (0, 1):
                year = today.year + add_year
                try:
                    if at_month_end:
                        cand = shift_trading_days(
                            last_trading_day(year, int(month)), -offset) if offset                             else last_trading_day(year, int(month))
                    else:
                        cand = date(year, int(month), hist.day)
                        while not _is_td(cand):
                            cand = prev_trading_day(cand)
                except ValueError:
                    continue
                ts = pd.Timestamp(cand)
                if ts <= today or (ts - today).days > horizon_days:
                    continue
                if best is None or ts < best[0]:
                    best = (ts, float(last["amount"]), note)
                break
        if best:
            rows.append({
                "ticker": ticker,
                "次の権利落ち日": best[0].date(),
                "あと何日": int((best[0] - today).days),
                "1株配当の目安": best[1],
                "根拠": best[2],
            })
    if not rows:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(rows).sort_values("次の権利落ち日").reset_index(drop=True)


def _is_td(d) -> bool:
    from modules.jp_calendar import is_trading_day
    return is_trading_day(d)
