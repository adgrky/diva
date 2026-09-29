"""スコアの予測力を過去データで確かめる。

【なぜ必ずやるか】
別プロジェクト（stock-recommender）で、13個の銘柄選別指標を測ったら相関が
すべて |r| < 0.1 で、「銘柄選別に上乗せできる情報は無い」という結果が出ている。
配当成長は時間軸が5〜10年と違うので同じ結論とは限らないが、確かめずに信じたら
同じ失敗になる。きれいなスコアが並ぶだけのアプリにしないために、
**重みを決める前にここを通す**。

【何を検証できて、何ができないか】
過去時点の特徴量を先読みなしで作れるのは、株価と配当履歴から計算できるものだけ。
yfinance の財務諸表は4〜5期分しか取れないため、2016年時点の配当性向や営業CFは
再現できない。したがって:

    検証できる    B 増配意思 / E 割安 / D 見過ごされ度（売買代金の薄さのみ）
    検証できない  A 増配余力 / C 原資成長 → Phase 3 で EDINET を入れてから

検証できない層を「効くはず」で重み付けするのは、やってはいけないことの見本なので、
そうした層は等ウェイトのまま据え置き、検証済みの層だけ重みを動かす。

【学習と検証の分け方】
前半のコホート（2014・2016年時点）で重みを決め、後半（2018・2020年時点）で
答え合わせする。全期間の最良値でチューニングすると、後から何の意味もない
数字が出てくる。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from modules.dividend_history import build_profile
from modules.quality import price_asof, trim_frame, winsorize

# 検証できる指標だけを列挙する。名前は「どの層に属するか」を接頭辞で示す。
TESTABLE_FACTORS = {
    "B_連続増配年数": "streak",
    "B_減配なし継続年数": "streak_no_cut",
    "B_DPS5年成長": "cagr_5y",
    "B_DPS10年成長": "cagr_10y",
    "B_10年減配回数": "cuts_10y_neg",       # 少ないほど良いので符号反転
    "D_売買代金の薄さ": "turnover_neg",
    "E_自己利回りパーセンタイル": "yield_percentile",
    "E_絶対利回り": "dividend_yield",
}

OUTCOMES = {
    "DPS成長（5年）": "fwd_dps_growth",
    "トータルリターン（5年）": "fwd_total_return",
    "減配の発生": "fwd_had_cut",
}


@dataclass
class Cohort:
    asof: pd.Timestamp
    horizon_years: int = 5


def _weekly_series(prices: pd.DataFrame) -> pd.DataFrame:
    df = prices.copy()
    df.index = pd.to_datetime(df["date"])
    return df[["close", "volume"]].sort_index()


def build_asof_features(ticker: str, bars: pd.DataFrame, div: pd.DataFrame,
                        asof: pd.Timestamp, window_years: int = 7) -> dict | None:
    """asof 時点で判明していた情報だけで特徴量を作る（先読み厳禁）。"""
    bars = bars[bars.index <= asof]
    px = bars["close"]
    dv = div[div["date"] <= asof]
    if len(px) < 52 * 3 or dv.empty:
        return None

    prof = build_profile(ticker, dv)
    if prof.years_paying < 5 or prof.dps_latest <= 0:
        return None

    price_now = price_asof(px, asof)   # 欠けた区間の古い値は使わない（quality 参照）
    if not price_now > 0:
        return None

    # 自己ヒストリカル利回りパーセンタイル（asof 時点までの分布で）
    cum = dv.set_index("date")["amount"].cumsum()
    idx = px.index
    cum_at = cum.reindex(cum.index.union(idx)).ffill().reindex(idx).fillna(0.0)
    prev_idx = pd.DatetimeIndex(idx - pd.DateOffset(years=1))
    cum_prev = cum.reindex(cum.index.union(prev_idx)).ffill().reindex(prev_idx).fillna(0.0)
    ttm = pd.Series(cum_at.values - cum_prev.values, index=idx).clip(lower=0)
    y = (ttm / px).replace([np.inf, -np.inf], np.nan).dropna()
    hist = y[(y.index >= asof - pd.DateOffset(years=window_years)) & (y > 0)]
    cur_y = float(y.iloc[-1]) if len(y) else np.nan
    pctile = float((hist < cur_y).mean()) if len(hist) >= 52 and cur_y > 0 else np.nan

    # 売買代金＝終値 × 出来高。ここを終値だけで代用すると「低位株ほど良い」という
    # 別の効果を測ってしまい、見過ごされ度の検証にならない。
    turnover = float((bars["close"] * bars["volume"]).tail(20).mean())

    return {
        "ticker": ticker,
        "streak": prof.streak,
        "streak_no_cut": prof.streak_no_cut,
        "cagr_5y": prof.cagr_5y,
        "cagr_10y": prof.cagr_10y,
        "cuts_10y_neg": -prof.cuts_10y,
        "turnover_neg": -turnover,
        "yield_percentile": pctile,
        "dividend_yield": cur_y,
        "_price": price_now,
        "_dps": prof.dps_latest,
    }


def build_outcomes(ticker: str, bars: pd.DataFrame, div: pd.DataFrame,
                   asof: pd.Timestamp, feat: dict, horizon_years: int = 5) -> dict | None:
    """asof から horizon 年後の実績。"""
    end = asof + pd.DateOffset(years=horizon_years)
    prices = bars["close"]
    px_fwd = prices[(prices.index > asof) & (prices.index <= end)]
    if px_fwd.empty or prices.index.max() < end - pd.Timedelta(days=45):
        return None  # 期間が満了していない銘柄は検証に入れない（上場廃止も含む）

    price_end = price_asof(px_fwd, end)
    if not price_end > 0:
        return None  # 満期日の近くに株価が無い（売買停止・欠けた区間）

    dv_fwd = div[(div["date"] > asof) & (div["date"] <= end)]
    divs_received = float(dv_fwd["amount"].sum())

    prof_end = build_profile(ticker, div[div["date"] <= end])
    dps_end = prof_end.dps_latest
    dps_start = feat["_dps"]

    # 期間中に減配があったか（記念配当のスパイクは除外済みの系列で見る）
    series = pd.Series(prof_end.series).sort_index()
    in_window = series[(series.index > asof.year) & (series.index <= end.year)]
    had_cut = bool((in_window.diff().dropna() < 0).any())

    return {
        "fwd_dps_growth": (dps_end / dps_start - 1) if dps_start > 0 else np.nan,
        "fwd_total_return": (price_end + divs_received) / feat["_price"] - 1,
        "fwd_had_cut": float(had_cut),
    }


def run_cohort(prices_all: pd.DataFrame, div_all: pd.DataFrame, cohort: Cohort,
               min_names: int = 100) -> pd.DataFrame:
    """1つの起点日について、特徴量と5年後の実績を突き合わせる。"""
    # 破損区間を落としてから測る。順位相関は外れ値に強いが、分位別の平均や
    # ベンチマークは1銘柄で壊れる（実測: 8303.T のせいで平均リターンが +480,000% になった）。
    prices_all, _ = trim_frame(prices_all)
    div_all = div_all.copy()
    div_all["date"] = pd.to_datetime(div_all["date"])
    div_by = {t: g for t, g in div_all.groupby("ticker", sort=False)}

    rows = []
    for ticker, g in prices_all.groupby("ticker", sort=False):
        dv = div_by.get(ticker)
        if dv is None or dv.empty:
            continue
        bars = _weekly_series(g)
        feat = build_asof_features(ticker, bars, dv, cohort.asof)
        if feat is None:
            continue
        out = build_outcomes(ticker, bars, dv, cohort.asof, feat, cohort.horizon_years)
        if out is None:
            continue
        rows.append({**feat, **out, "asof": cohort.asof.strftime("%Y-%m-%d")})

    df = pd.DataFrame(rows)
    if len(df) < min_names:
        return pd.DataFrame()
    return df


def factor_power(df: pd.DataFrame) -> pd.DataFrame:
    """各指標が将来の実績をどれだけ説明するか（順位相関）。

    ピアソンではなくスピアマンを使う。配当成長も株価リターンも分布が歪んでおり、
    外れ値1銘柄で相関が決まってしまうため。
    """
    rows = []
    for label, col in TESTABLE_FACTORS.items():
        if col not in df.columns:
            continue
        for oname, ocol in OUTCOMES.items():
            sub = df[[col, ocol]].dropna()
            if len(sub) < 50:
                rows.append({"指標": label, "実績": oname, "相関": np.nan, "n": len(sub)})
                continue
            r = sub[col].corr(sub[ocol], method="spearman")
            rows.append({"指標": label, "実績": oname, "相関": float(r), "n": len(sub)})
    return pd.DataFrame(rows)


def quintile_table(df: pd.DataFrame, factor_col: str, outcome_col: str,
                   q: int = 5) -> pd.DataFrame:
    """分位ごとの実績。相関が小さくても上位分位だけ効いていることがある。"""
    sub = df[[factor_col, outcome_col]].dropna()
    if len(sub) < q * 10:
        return pd.DataFrame()
    try:
        sub["分位"] = pd.qcut(sub[factor_col].rank(method="first"), q,
                             labels=[f"Q{i + 1}" for i in range(q)])
    except ValueError:
        return pd.DataFrame()
    # 平均は裾を刈ってから取る。リターンの分布は右に極端に長く、
    # 上位1%の数銘柄で分位の平均が決まってしまう。
    sub["_w"] = winsorize(sub[outcome_col])
    g = sub.groupby("分位", observed=True).agg(
        平均=("_w", "mean"), 中央値=(outcome_col, "median"), 銘柄数=(outcome_col, "count"))
    return g.reset_index()


def benchmark(df: pd.DataFrame, top_n: int = 30) -> pd.DataFrame:
    """比較対象。これに勝てないスコアは採用しない。

    リターンは中央値で比べる。平均は上位数銘柄で決まってしまい、
    「実際に持ったらどうだったか」の感覚と合わない。
    """
    df = df.copy()
    df["_w"] = winsorize(df["fwd_total_return"])

    def row(label: str, sub: pd.DataFrame) -> dict:
        return {"戦略": label, "銘柄数": len(sub),
                "リターン中央値": sub["fwd_total_return"].median(),
                "リターン平均（裾を刈る）": sub["_w"].mean(),
                "DPS成長 中央値": sub["fwd_dps_growth"].median(),
                "減配発生率": sub["fwd_had_cut"].mean()}

    # コホートごとに上位N銘柄を取る。全コホートを混ぜて上位を取ると、
    # たまたま一番良かった年のものばかり選ばれる。
    def top_by(col: str) -> pd.DataFrame:
        return pd.concat([g.nlargest(top_n, col) for _, g in df.groupby("asof")])

    return pd.DataFrame([
        row("全銘柄を等分で持つ", df),
        row(f"単純に利回り上位{top_n}銘柄", top_by("dividend_yield")),
        row(f"連続増配年数 上位{top_n}銘柄", top_by("streak")),
        row(f"自己利回りパーセンタイル上位{top_n}銘柄", top_by("yield_percentile")),
    ])


def composite_test(train: pd.DataFrame, test: pd.DataFrame,
                   outcome: str = "fwd_total_return", top_n: int = 30) -> pd.DataFrame:
    """学習コホートで効いた指標だけを等ウェイト合成し、検証コホートで確かめる。

    重みを細かく最適化しない。相関の順位だけを使って「効いた指標を等分で足す」に
    留める。少ないコホートで重みまで当てにいくと、確実に過剰適合する。
    """
    power = factor_power(train)
    keep = power[(power["実績"] == OUTCOMES_INV[outcome]) & (power["相関"] > 0.05)]["指標"].tolist()
    if not keep:
        return pd.DataFrame([{"結果": "学習側で相関0.05を超える指標が無かった。合成しても意味がない。"}])

    cols = [TESTABLE_FACTORS[k] for k in keep]
    scored = test.assign(_score=test[cols].rank(pct=True).mean(axis=1))
    # コホートごとに上位を取る（混ぜて取ると当たり年に偏る）
    sel = pd.concat([g.nlargest(top_n, "_score") for _, g in scored.groupby("asof")])

    return pd.DataFrame([{
        "採用した指標": "、".join(keep),
        "検証コホートの上位銘柄数": len(sel),
        "上位の中央値": sel[outcome].median(),
        "全体の中央値": test[outcome].median(),
        "差": sel[outcome].median() - test[outcome].median(),
        "減配発生率（上位）": sel["fwd_had_cut"].mean(),
        "減配発生率（全体）": test["fwd_had_cut"].mean(),
    }])


OUTCOMES_INV = {v: k for k, v in OUTCOMES.items()}
