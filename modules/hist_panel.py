"""有報の過去データを「比べられる形」に整えて、1枚のパネルにする。

【なぜこの層が必要か — 実測で見つけた落とし穴】

有価証券報告書の「主要な経営指標等の推移」は5年ぶんの数字を並べているが、
**1株当たり配当額だけは株式分割の調整が入っていない**。1株当たり当期純利益は
提出時点にそろえて遡及修正されているのに、配当は「その年に実際に出した額」の
まま載る。

    NTT（9432）2026年提出ぶん
        年度   EPS     DPS
        2022   13.17   115.0   ← 2023年の25分割が EPS だけに反映されている
        2024   15.09     5.1
    そのまま割ると 配当性向 873%。正しくは 115÷25÷13.17 = 35%。

2013〜2019年に株式分割をした会社は **1,530社**（全上場の約4割）。
つまり素朴に dps ÷ eps を計算すると、4割の会社の配当性向が壊れる。

【そろえ方】
すべての1株あたりの数字を「今日の株数」に直してから比べる。

    EPS は提出日時点の株数 → 提出日より後の分割で割る
    DPS はその年度末時点の株数 → 年度末より後の分割で割る

比率（ROE・自己資本比率）と金額（売上・純利益・営業CF）は分割の影響を受けない
のでそのまま使う。

【年度末と提出日をどこから取るか】
edinet_summary には提出日が入っていないので、edinet_doc_index / edinet_index の
period_end・submit_date から割り当て直す。1つの有報は5年ぶんを載せるので、
年度 y の行は「y を含む最初に取り込んだ有報」から来たものとみなす
（fetch_edinet_history は既にある行を上書きしないため）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from modules.quality import price_asof, trim_frame
from modules.store import read_df

_TODAY = pd.Timestamp.today().normalize()
_COVERS = 5  # 1つの有報が載せる年数


def _load_docs() -> pd.DataFrame:
    """銘柄ごとの有報の一覧（期末日・提出日）。"""
    uni = read_df("SELECT code, ticker FROM universe")
    parts = []
    for sql in ("SELECT code, period_end, submit_date FROM edinet_doc_index",
                "SELECT code, period_end, submit_date FROM edinet_index"):
        d = read_df(sql)
        if not d.empty:
            parts.append(d)
    if not parts:
        return pd.DataFrame(columns=["ticker", "period_end", "submit_date", "fy_doc"])
    docs = pd.concat(parts, ignore_index=True)
    docs["code"] = docs["code"].astype(str)
    uni["code"] = uni["code"].astype(str)
    docs = docs.merge(uni, on="code", how="inner")
    docs["period_end"] = pd.to_datetime(docs["period_end"], errors="coerce")
    docs["submit_date"] = pd.to_datetime(docs["submit_date"], errors="coerce")
    docs = docs.dropna(subset=["period_end", "submit_date"])
    docs["fy_doc"] = docs["period_end"].dt.year
    return (docs.sort_values(["ticker", "submit_date"])
                .drop_duplicates(["ticker", "fy_doc"], keep="first")
                .reset_index(drop=True))


def _split_index() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """銘柄ごとに（分割日の配列, 比率の累積積の配列）を作る。

    ある期間の分割倍率は、累積積の比で一発で出せる。
    """
    sp = read_df("SELECT ticker, date, ratio FROM splits")
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    if sp.empty:
        return out
    sp["date"] = pd.to_datetime(sp["date"], errors="coerce")
    sp = sp.dropna(subset=["date"])
    sp = sp[(sp["ratio"] > 0) & np.isfinite(sp["ratio"])]
    for t, g in sp.sort_values("date").groupby("ticker"):
        out[str(t)] = (g["date"].values.astype("datetime64[ns]"),
                       np.cumprod(g["ratio"].to_numpy(float)))
    return out


def _cum_after(idx: dict, ticker: str, start: pd.Timestamp,
               end: pd.Timestamp = _TODAY) -> float:
    """start より後 end 以下に起きた分割の倍率の積。無ければ 1.0。"""
    ent = idx.get(ticker)
    if ent is None or pd.isna(start):
        return 1.0
    dates, cum = ent
    i = np.searchsorted(dates, np.datetime64(start), side="right")
    j = np.searchsorted(dates, np.datetime64(end), side="right")
    if j <= i:
        return 1.0
    before = cum[i - 1] if i > 0 else 1.0
    return float(cum[j - 1] / before)


def _assign_filing(summary: pd.DataFrame, docs: pd.DataFrame,
                   idx: dict) -> pd.DataFrame:
    """各行が、どの有報から来たものかを突き止める。

    1つの年度を2つの有報が載せていることがある（例: FY2018 は2018年提出ぶんにも
    2022年提出ぶんにも載る）。取り込んだ順は提出日の順ではないので、日付では
    決められない。そこで **株数の連続性** で決める。

        株数 = 純利益 ÷ 1株利益

    これを今日の株数にそろえたとき、同じ会社なら年をまたいでもほぼ変わらない
    （自社株買いや増資で数%動くだけ）。分割の候補は2倍・25倍と桁で違うので、
    銘柄の代表値にいちばん近い候補を選べば取り違えない。
    """
    by_ticker = {t: g.sort_values("submit_date") for t, g in docs.groupby("ticker")}
    cands: list[list[tuple[pd.Timestamp, pd.Timestamp]]] = []
    for t, fy in zip(summary["ticker"], summary["fiscal_year"]):
        g = by_ticker.get(t)
        if g is None:
            cands.append([])
            continue
        ok = g[(g["fy_doc"] >= fy) & (g["fy_doc"] - (_COVERS - 1) <= fy)]
        cands.append(list(zip(ok["period_end"], ok["submit_date"])))

    ni = summary["net_income"].to_numpy(float)
    eps = summary["eps"].to_numpy(float)
    tickers = summary["ticker"].to_numpy()

    def shares_today(i: int, submit: pd.Timestamp) -> float:
        if not np.isfinite(ni[i]) or not np.isfinite(eps[i]) or abs(eps[i]) < 1e-9:
            return np.nan
        return ni[i] / (eps[i] / _cum_after(idx, tickers[i], submit))

    # まず候補が1つしかない行で、銘柄ごとの代表的な株数を決める
    ref: dict[str, list[float]] = {}
    for i, c in enumerate(cands):
        if len(c) == 1:
            v = shares_today(i, c[0][1])
            if np.isfinite(v) and v > 0:
                ref.setdefault(str(tickers[i]), []).append(v)
    med = {t: float(np.median(v)) for t, v in ref.items()}

    pe, sd = [], []
    for i, c in enumerate(cands):
        if not c:
            pe.append(pd.NaT); sd.append(pd.NaT); continue
        if len(c) == 1:
            pe.append(c[0][0]); sd.append(c[0][1]); continue
        r = med.get(str(tickers[i]))
        best = c[-1]  # 代表値が取れないときは最新の有報
        if r and r > 0:
            score = []
            for period_end, submit in c:
                v = shares_today(i, submit)
                score.append(abs(np.log(v / r)) if np.isfinite(v) and v > 0 else np.inf)
            if np.isfinite(min(score)):
                best = c[int(np.argmin(score))]
        pe.append(best[0]); sd.append(best[1])

    summary = summary.copy()
    summary["doc_period_end"] = pd.to_datetime(pd.Series(pe, index=summary.index))
    summary["doc_submit_date"] = pd.to_datetime(pd.Series(sd, index=summary.index))
    md = summary["doc_period_end"]
    summary["fy_end"] = [
        (pd.Timestamp(year=int(fy), month=d.month, day=min(d.day, 28))
         if pd.notna(d) else pd.Timestamp(year=int(fy), month=3, day=31))
        for fy, d in zip(summary["fiscal_year"], md)
    ]
    summary["doc_submit_date"] = summary["doc_submit_date"].fillna(
        summary["fy_end"] + pd.DateOffset(months=3))
    return summary


def _repair_eps_frame(s: pd.DataFrame, idx: dict) -> pd.DataFrame:
    """1株利益の「株数の枠」を、株数の連続性で直す。

    提出日から機械的に決めるだけでは足りない。有報が過去年の1株利益を分割に
    合わせて遡及修正しているかは **会社によって違う**（実測: NTT は修正あり、
    小林製薬は修正なし）。修正の有無を外から知る術はないので、データ自身に
    語らせる。

        株数 = 純利益 ÷ 1株利益

    この株数は自社株買い・増資で年に数%しか動かない。一方、分割は2倍・25倍と
    桁で動く。だから最新年を基準に1年ずつ遡り、「その年の分割候補のうち、翌年の
    株数にいちばん近くなる倍率」を選べば、枠のズレだけを拾える。

    ただし直すのは **明らかに良くなるときだけ**（3割以上ズレていて、かつ候補の
    ほうが3倍以上まし）。本物の大型増資を分割と取り違えないための歯止め。
    """
    out = s["split_eps"].to_numpy(float).copy()
    ni = s["net_income"].to_numpy(float)
    eps = s["eps"].to_numpy(float)
    pos = {t: g for t, g in s.groupby("ticker", sort=False).indices.items()}
    for t, ii in pos.items():
        ii = np.sort(ii)
        if len(ii) < 2:
            continue
        ent = idx.get(str(t))
        if ent is None:
            continue
        dates, cum = ent
        sh = np.full(len(ii), np.nan)
        for k, i in enumerate(ii):
            if np.isfinite(ni[i]) and np.isfinite(eps[i]) and abs(eps[i]) > 1e-9:
                sh[k] = ni[i] / (eps[i] / out[i])
        for k in range(len(ii) - 2, -1, -1):
            i = ii[k]
            tgt = sh[k + 1]
            if not np.isfinite(tgt) or tgt <= 0 or not np.isfinite(sh[k]) or sh[k] <= 0:
                continue
            base = abs(np.log(sh[k] / tgt))
            if base < np.log(1.30):
                continue
            fy_end = s["fy_end"].iloc[i]
            after = np.searchsorted(dates, np.datetime64(fy_end), side="right")
            if after >= len(dates):
                continue
            prev = cum[after - 1] if after > 0 else 1.0
            cands = [1.0] + [float(cum[-1] / c) for c in cum[after - 1:]] \
                + [float(cum[-1] / prev)]
            best_f, best_s = out[i], base
            for f in dict.fromkeys(cands):
                if f <= 0:
                    continue
                cand_sh = ni[i] / (eps[i] / f)
                sc = abs(np.log(cand_sh / tgt))
                if sc < best_s:
                    best_f, best_s = f, sc
            if best_s * 3 < base:
                out[i] = best_f
                sh[k] = ni[i] / (eps[i] / best_f)
    s = s.copy()
    s["split_eps"] = out
    return s


def _attach_actual_dps(s: pd.DataFrame) -> pd.DataFrame:
    """実際に払われた配当（株価側・分割調整済み）を年度に貼り付ける。

    有報の1株配当は 95% は一致するが、5% は分割の枠がずれる・記念配当の扱いが
    違う・年度の切り方がずれる。**アプリの他の画面はすべて株価側の配当を使って
    いる**ので、こちらを本命にして、取れない年だけ有報の値で埋める。
    """
    dv = read_df("SELECT ticker, date, amount FROM dividends")
    if dv.empty:
        s["dps_actual"] = np.nan
        s["dps_use"] = s["dps_adj"]
        s["dps_source"] = "有報"
        return s
    dv["date"] = pd.to_datetime(dv["date"], errors="coerce")
    dv = dv.dropna(subset=["date"])
    by = {t: g.set_index("date")["amount"].sort_index() for t, g in dv.groupby("ticker")}
    cover = {t: (g.index.min(), g.index.max()) for t, g in by.items()}

    vals = []
    for t, fe in zip(s["ticker"], s["fy_end"]):
        g = by.get(t)
        if g is None or pd.isna(fe):
            vals.append(np.nan)
            continue
        lo, hi = cover[t]
        # 配当の記録がその年度まで届いていない銘柄で 0 を「無配」と誤らせない
        end = fe + pd.Timedelta(days=7)
        if end > hi + pd.Timedelta(days=200) or end - pd.DateOffset(years=1) < lo:
            vals.append(np.nan)
            continue
        vals.append(float(g[(g.index > end - pd.DateOffset(years=1)) & (g.index <= end)].sum()))
    s = s.copy()
    s["dps_actual"] = vals
    s["dps_use"] = s["dps_actual"].where(s["dps_actual"].notna(), s["dps_adj"])
    s["dps_source"] = np.where(s["dps_actual"].notna(), "株価", "有報")

    with np.errstate(divide="ignore", invalid="ignore"):
        s["div_total"] = s["dps_use"] * s["shares"]
        s["payout"] = np.where(s["eps_adj"] > 0, s["dps_use"] / s["eps_adj"], np.nan)
        s["ocf_cover"] = np.where(s["div_total"] > 0,
                                  s["operating_cf"] / s["div_total"], np.nan)
    s["payout"] = s["payout"].where(s["payout"].between(-5, 20))
    s["ocf_cover"] = s["ocf_cover"].where(s["ocf_cover"].between(-100, 500))
    return s


def build_panel(min_fy: int = 2012) -> pd.DataFrame:
    """銘柄 × 年度のパネル。すべて「今日の株数」にそろえてある。"""
    s = read_df("SELECT ticker, fiscal_year, basis, sales, ordinary_income, net_income, "
                "eps, dps, roe, equity_ratio, net_assets, total_assets, operating_cf, "
                "employees FROM edinet_summary")
    s = s[s["fiscal_year"] >= min_fy].copy()
    s["ticker"] = s["ticker"].astype(str)
    idx = _split_index()
    s = _assign_filing(s, _load_docs(), idx)

    f_eps = [_cum_after(idx, t, d) for t, d in zip(s["ticker"], s["doc_submit_date"])]
    f_dps = [_cum_after(idx, t, d) for t, d in zip(s["ticker"], s["fy_end"])]
    s["split_eps"] = f_eps
    s["split_dps"] = f_dps
    s = _repair_eps_frame(s, idx)
    s["eps_adj"] = s["eps"] / s["split_eps"]
    s["dps_adj"] = s["dps"] / s["split_dps"]

    # 株数は金額 ÷ 1株あたり。どちらも同じ枠にそろっているので割れる。
    with np.errstate(divide="ignore", invalid="ignore"):
        s["shares"] = np.where(s["eps_adj"].abs() > 1e-9,
                               s["net_income"] / s["eps_adj"], np.nan)
        s["div_total"] = s["dps_adj"] * s["shares"]
        s["payout"] = np.where(s["eps_adj"] > 0, s["dps_adj"] / s["eps_adj"], np.nan)
        # ここでの payout / ocf_cover は仮。実際に払われた配当（_attach_actual_dps）
        # に差し替えたあと、_recompute で計算し直す。
        s["ocf_cover"] = np.where(s["div_total"] > 0,
                                  s["operating_cf"] / s["div_total"], np.nan)
        s["margin"] = np.where(s["sales"] > 0, s["ordinary_income"] / s["sales"], np.nan)
        # 経常利益に対して純利益が大きい＝特別利益が乗っている疑い
        s["ni_over_op"] = np.where(s["ordinary_income"] > 0,
                                   s["net_income"] / s["ordinary_income"], np.nan)
        s["ocf_over_ni"] = np.where(s["net_income"] > 0,
                                    s["operating_cf"] / s["net_income"], np.nan)
    s["payout"] = s["payout"].where(s["payout"].between(-5, 20))
    s["ocf_cover"] = s["ocf_cover"].where(s["ocf_cover"].between(-100, 500))

    s = s.sort_values(["ticker", "fiscal_year"]).reset_index(drop=True)
    s = _attach_actual_dps(s)
    g = s.groupby("ticker", sort=False)

    # 年度が飛んでいる（取り込めていない年がある）行では前年比を作らない
    s["gap"] = g["fiscal_year"].diff()
    one = s["gap"] == 1
    for col in ("eps_adj", "dps_use", "net_income", "operating_cf", "sales",
                "equity_ratio", "roe", "margin", "payout"):
        s[f"d_{col}"] = g[col].diff().where(one)
        prev = g[col].shift(1).where(one)
        s[f"g_{col}"] = np.where(prev.abs() > 1e-9, s[col] / prev - 1, np.nan)
    s["prev_dps"] = g["dps_use"].shift(1).where(one)
    s["prev_ni"] = g["net_income"].shift(1).where(one)

    # 減益の年に配当をどうしたか（増配意思の代理変数のもと）
    # 「判定できない」を持てる真偽値にしておく。object 型のままだと
    # fillna(False) のたびに pandas が警告を出し、将来の版で挙動が変わる。
    s["is_down_year"] = (s["net_income"] < s["prev_ni"]).where(
        s["prev_ni"].notna()).astype("boolean")
    known = s["prev_dps"].notna() & (s["prev_dps"] > 0)
    s["dps_held"] = (s["dps_use"] >= s["prev_dps"] * 0.999).where(known).astype("boolean")
    s["dps_cut"] = (s["dps_use"] < s["prev_dps"] * 0.999).where(known).astype("boolean")
    return s.drop(columns=["gap"])


# ─────────────────────────────────────────────────────────────────────
# 先の実績（答え合わせ用）
# ─────────────────────────────────────────────────────────────────────

def _asof(fy_end: pd.Timestamp) -> pd.Timestamp:
    """その年度の数字が世に出ている日。決算期末の3ヶ月後（有報の提出期限）。

    3月期と12月期を同じ「6月末」で揃えると、12月期の会社には半年ぶん先の
    情報が混ざる。期末からの経過月数で揃えるのが正しい。
    """
    return fy_end + pd.DateOffset(months=3)


def attach_outcomes(panel: pd.DataFrame, horizons=(1, 2, 3, 5)) -> pd.DataFrame:
    """各行に「その後どうなったか」を貼る。判断日は年度末の3ヶ月後。"""
    from modules.dividend_history import build_profile

    # 分割が未調整のまま残った区間を落とす（ほかの検証と同じ trim_frame を通す）。
    # 通していなかったため、8303.T の ret_1y が +183,066%（=1,830倍）になり、
    # パネル全体の ret_1y 平均が +3,300% に化けていた。
    px, _ = trim_frame(read_df("SELECT ticker, date, close FROM prices"))
    px["date"] = pd.to_datetime(px["date"])
    px = px[px["close"].notna() & (px["close"] > 0)]
    P = {t: g.set_index("date")["close"].sort_index() for t, g in px.groupby("ticker")}

    dv = read_df("SELECT ticker, date, amount FROM dividends")
    dv["date"] = pd.to_datetime(dv["date"])
    D = {t: g.set_index("date")["amount"].sort_index() for t, g in dv.groupby("ticker")}

    last_px = max((s.index.max() for s in P.values()), default=pd.NaT)
    cut_cache: dict[tuple[str, int], set[int]] = {}

    def cut_years(ticker: str, end: pd.Timestamp) -> set[int]:
        """end までのデータだけで見た「減配した年」の集合。

        期間の外まで見ないこと（先読み防止）と、記念配当を減配と数えないこと
        の両方が要る。判定はアプリ本体と同じ build_profile に任せる。
        """
        key = (ticker, end.year)
        if key in cut_cache:
            return cut_cache[key]
        s = D.get(ticker)
        if s is None:
            cut_cache[key] = set()
            return cut_cache[key]
        w = s[s.index <= end].reset_index()
        w.columns = ["date", "amount"]
        try:
            ser = pd.Series(build_profile(ticker, w).series).sort_index()
        except Exception:
            cut_cache[key] = set()
            return cut_cache[key]
        d = ser.diff()
        cut_cache[key] = {int(y) for y, v in d.items() if pd.notna(v) and v < 0}
        return cut_cache[key]

    rows = []
    for r in panel.itertuples(index=False):
        t = r.ticker
        p, d = P.get(t), D.get(t)
        rec: dict = {}
        asof = _asof(r.fy_end) if pd.notna(r.fy_end) else pd.NaT
        rec["asof"] = asof
        if p is None or pd.isna(asof) or p.empty:
            rows.append(rec)
            continue
        if p.index[0] > asof:
            rows.append(rec)
            continue
        # 判断日・満期日の株価は「その日に近い終値」だけを使う。株価の欠けた区間
        # （売買停止・上場廃止中など）で何年も前の値を拾うと偽のリターンになる。
        # 株価が無くても配当の実績は正しいので、配当側の答えは付ける。
        price0 = price_asof(p, asof)
        rec["price0"] = price0
        d0 = float(d[(d.index > asof - pd.DateOffset(years=1)) & (d.index <= asof)].sum()) \
            if d is not None else 0.0
        rec["ttm_dps"] = d0
        rec["start_yield"] = d0 / price0 if price0 > 0 else np.nan
        for h in horizons:
            end = asof + pd.DateOffset(years=h)
            if pd.isna(last_px) or end > last_px - pd.Timedelta(days=30):
                continue
            recv = float(d[(d.index > asof) & (d.index <= end)].sum()) if d is not None else 0.0
            price1 = price_asof(p, end)
            if price0 > 0 and price1 > 0:
                rec[f"ret_{h}y"] = (price1 + recv) / price0 - 1
            if d0 > 0 and d is not None:
                d1 = float(d[(d.index > end - pd.DateOffset(years=1)) & (d.index <= end)].sum())
                rec[f"dps_growth_{h}y"] = d1 / d0 - 1
            cy = cut_years(t, end)
            rec[f"cut_{h}y"] = int(any(asof.year < y <= end.year for y in cy))
        rows.append(rec)
    return pd.concat([panel.reset_index(drop=True), pd.DataFrame(rows)], axis=1)


_CACHE = "data/cache_hist_panel.pkl"


def load_cached(rebuild: bool = False) -> pd.DataFrame:
    """パネルを作り直すと数分かかるので、作ったものを置いておく。

    有報の取り込みが進んだら rebuild=True で作り直す。
    """
    import os
    if not rebuild and os.path.exists(_CACHE):
        return pd.read_pickle(_CACHE)
    p = attach_outcomes(build_panel())
    p.to_pickle(_CACHE)
    return p
