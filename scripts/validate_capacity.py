"""A層（増配余力）の検証。配当性向・営業CFの厚み・自己資本比率・ROEは効くのか。

【なぜ複数コホートにしたか】
最初は FY2022 起点の1コホート・3年しか測れなかった。1つでは「時期をまたいで
安定しているか」が確かめられず、足切りを既定にしてよいか判断できない。

過去の有報を取り込んで（scripts/fetch_edinet_history.py）FY2014 まで遡れたので、
本検証（株価・配当）と同じ 4コホート・5年で測り直す。

【起点の置き方】
FY○○ の有報は、決算期末から3ヶ月以内＝翌年6月までに出る。
そこで **起点は (FY+1)年6月30日**。その日までに公表されている情報だけを使う。

    FY2014 → 起点 2015-06-30 → 実績 2020-06-30
    FY2016 → 起点 2017-06-30 → 実績 2022-06-30
    FY2018 → 起点 2019-06-30 → 実績 2024-06-30
    FY2020 → 起点 2021-06-30 → 実績 2026-06-30

【学習と検証の分け方】
前半（FY2014・FY2016）で見て、後半（FY2018・FY2020）で答え合わせする。
全期間の最良値でチューニングしない。

使い方:
    uv run python scripts/validate_capacity.py
    uv run python scripts/validate_capacity.py --fy 2014,2016,2018,2020 --horizon 5
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np                                    # noqa: E402
import pandas as pd                                   # noqa: E402
from scipy.stats import spearmanr                     # noqa: E402

from modules.dividend_history import build_profile    # noqa: E402
from modules.store import read_df                     # noqa: E402
from modules.quality import price_asof, trim_frame    # noqa: E402

FACTORS = {
    "A_配当性向の低さ": "payout_neg",
    "A_営業CFの厚み": "ocf_cover",
    "A_自己資本比率": "equity_ratio",
    "A_ROE": "roe",
    "（比較）起点の利回り": "start_yield",
}
OUTCOMES = {"DPS成長": "fwd_dps_growth", "トータルリターン": "fwd_total_return",
            "減配の発生": "fwd_had_cut"}


def _hr(t: str) -> None:
    print("\n" + "─" * 80 + f"\n{t}\n" + "─" * 80)


def build_cohort(fy: int, horizon: int, prices: pd.DataFrame,
                 div: pd.DataFrame) -> pd.DataFrame:
    """FY○○ の財務を起点に、horizon 年先の実績と突き合わせる。"""
    asof = pd.Timestamp(f"{fy + 1}-06-30")
    end = asof + pd.DateOffset(years=horizon)
    if end > pd.Timestamp.today():
        return pd.DataFrame()

    fin = read_df("SELECT ticker, net_income, eps, dps, roe, equity_ratio, operating_cf "
                  "FROM edinet_summary WHERE fiscal_year = ?", (fy,))
    fin = fin[fin["eps"].notna() & (fin["eps"] != 0)]
    if fin.empty:
        return pd.DataFrame()

    fin["payout_neg"] = -(fin["dps"] / fin["eps"])
    shares = fin["net_income"] / fin["eps"]
    div_total = fin["dps"] * shares
    fin["ocf_cover"] = np.where(div_total > 0, fin["operating_cf"] / div_total, np.nan)

    px_all = {t: g.set_index("date")["close"].sort_index()
              for t, g in prices.groupby("ticker")}
    dv_all = {t: g.set_index("date")["amount"].sort_index()
              for t, g in div.groupby("ticker")}

    rows = []
    for r in fin.itertuples(index=False):
        px, dv = px_all.get(r.ticker), dv_all.get(r.ticker)
        if px is None or dv is None or px.empty or dv.empty:
            continue
        if px.index.max() < end - pd.Timedelta(days=45):
            continue
        # 欠けた区間の古い値を拾わない（quality.price_asof 参照）
        price0, price1 = price_asof(px, asof), price_asof(px, end)
        if not (price0 > 0 and price1 > 0):
            continue
        d0 = float(dv[(dv.index > asof - pd.DateOffset(years=1)) & (dv.index <= asof)].sum())
        if d0 <= 0:
            continue
        d1 = float(dv[(dv.index > end - pd.DateOffset(years=1)) & (dv.index <= end)].sum())
        received = float(dv[(dv.index > asof) & (dv.index <= end)].sum())

        # 減配の判定は本検証（modules/validation.py）と同じやり方に揃える。
        # 自前で年度を切ると
        #   ・起点の年から1年目への減配を見落とす
        #   ・期間の外（horizon+1年目）まで見てしまう
        # という取りこぼしと先読みが起きる。build_profile は記念配当の
        # スパイクを除いた年度系列を返すので、そのまま差分を見る。
        dv_win = dv[dv.index <= end].reset_index()
        dv_win.columns = ["date", "amount"]
        prof = build_profile(r.ticker, dv_win)
        series = pd.Series(prof.series).sort_index()
        in_window = series[(series.index > asof.year) & (series.index <= end.year)]
        had_cut = bool((in_window.diff().dropna() < 0).any())
        rows.append({
            "ticker": r.ticker, "fy": fy,
            "payout_neg": r.payout_neg, "ocf_cover": r.ocf_cover,
            "equity_ratio": r.equity_ratio, "roe": r.roe,
            "start_yield": d0 / price0,
            "fwd_dps_growth": d1 / d0 - 1,
            "fwd_total_return": (price1 + received) / price0 - 1,
            "fwd_had_cut": int(had_cut),
        })
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fy", default="2014,2016,2018,2020")
    ap.add_argument("--horizon", type=int, default=5)
    a = ap.parse_args()
    fys = [int(x) for x in a.fy.split(",")]

    print(f"A層（増配余力）の検証  起点 FY{fys} ／ 先 {a.horizon} 年")
    prices, _ = trim_frame(read_df("SELECT ticker, date, close FROM prices"))
    prices["date"] = pd.to_datetime(prices["date"])
    div = read_df("SELECT ticker, date, amount FROM dividends")
    div["date"] = pd.to_datetime(div["date"])

    frames = {}
    for fy in fys:
        c = build_cohort(fy, a.horizon, prices, div)
        if c.empty:
            print(f"  FY{fy}: 作れませんでした（財務か実績が足りない）")
            continue
        print(f"  FY{fy}（起点 {fy+1}-06-30 → 実績 {fy+1+a.horizon}-06-30）: {len(c):,} 銘柄")
        frames[fy] = c
    if not frames:
        print("コホートが1つも作れませんでした")
        return
    df = pd.concat(frames.values(), ignore_index=True)
    print(f"  合計 延べ {len(df):,} 銘柄 / {len(frames)} コホート")

    _hr("1. 各指標と実績の順位相関（全コホート結合）")
    tbl = {}
    for fname, col in FACTORS.items():
        tbl[fname] = {}
        for oname, ocol in OUTCOMES.items():
            m = df[col].notna() & df[ocol].notna()
            tbl[fname][oname] = (spearmanr(df.loc[m, col], df.loc[m, ocol]).statistic
                                 if m.sum() >= 100 else np.nan)
    print(pd.DataFrame(tbl).T.round(3).to_string())
    print("\n目安: |相関| < 0.05 はノイズ。0.10 を超えたら意味がある可能性がある。")

    if len(frames) > 1:
        _hr("2. コホート別（時期をまたいで安定しているか）")
        for oname, ocol in OUTCOMES.items():
            rows = {}
            for fy, c in frames.items():
                rows[fy] = {fn: (spearmanr(c.loc[c[col].notna() & c[ocol].notna(), col],
                                           c.loc[c[col].notna() & c[ocol].notna(), ocol]).statistic
                                 if (c[col].notna() & c[ocol].notna()).sum() >= 50 else np.nan)
                            for fn, col in FACTORS.items()}
            print(f"\n【{oname}】")
            print(pd.DataFrame(rows).round(3).to_string())
        print("\n符号がコホートごとに入れ替わる指標は、効いていないと考えるべき。")

    _hr("3. 分位別の実績（5分位・中央値）")
    for fname, col in FACTORS.items():
        m = df[col].notna()
        if m.sum() < 300:
            continue
        q = pd.qcut(df.loc[m, col].rank(method="first"), 5,
                    labels=[f"Q{i}" for i in range(1, 6)])
        g = df[m].groupby(q, observed=True).agg(
            DPS成長=("fwd_dps_growth", "median"),
            リターン=("fwd_total_return", "median"),
            減配率=("fwd_had_cut", "mean"))
        print(f"\n  【{fname}】（Q5が最上位）")
        print("    DPS成長  " + "  ".join(f"{i}:{v:+.0%}" for i, v in zip(g.index, g["DPS成長"])))
        print("    リターン " + "  ".join(f"{i}:{v:+.0%}" for i, v in zip(g.index, g["リターン"])))
        print("    減配率   " + "  ".join(f"{i}:{v:.0%}" for i, v in zip(g.index, g["減配率"])))

    _hr("4. 配当性向の帯ごとの実績（30〜50%が最高という前提は正しいか）")
    d = df.copy()
    d["payout"] = -d["payout_neg"]
    d = d[d["payout"].between(-0.5, 3.0)]
    bins = [-np.inf, 0.15, 0.30, 0.50, 0.70, 1.00, np.inf]
    labels = ["15%未満", "15-30%", "30-50%", "50-70%", "70-100%", "100%超"]
    d["帯"] = pd.cut(d["payout"], bins, labels=labels)
    g = d.groupby("帯", observed=True).agg(
        n=("ticker", "count"), DPS成長=("fwd_dps_growth", "median"),
        リターン=("fwd_total_return", "median"), 減配率=("fwd_had_cut", "mean"),
        起点の利回り=("start_yield", "median"))
    for c in ("DPS成長", "リターン", "減配率", "起点の利回り"):
        g[c] = (g[c] * 100).round(1)
    print(g.to_string())

    _hr("5. 高利回り × 増配余力の足切りは、減配を減らすか")
    for label, sub in [("全コホート", df)] + (
            [(f"FY{fy}", c) for fy, c in frames.items()] if len(frames) > 1 else []):
        hi = sub["start_yield"] >= sub["start_yield"].quantile(0.7)
        okp = (-sub["payout_neg"]).between(0.0, 0.5)
        okf = sub["ocf_cover"] >= 3
        print(f"\n  【{label}】")
        for nm, m in (("全体", pd.Series(True, index=sub.index)),
                      ("高利回り上位30%", hi),
                      ("  ＋ 配当性向50%以下", hi & okp),
                      ("  ＋ 営業CFが配当の3倍以上", hi & okp & okf)):
            s = sub[m]
            if len(s) < 30:
                print(f"    {nm:<26s} n={len(s):>5,}  （少なすぎる）")
                continue
            print(f"    {nm:<26s} n={len(s):>5,}  減配率 {s['fwd_had_cut'].mean():>5.1%}"
                  f"  リターン {s['fwd_total_return'].median():>+7.1%}"
                  f"  DPS成長 {s['fwd_dps_growth'].median():>+7.1%}")

    if len(frames) >= 4:
        _hr("6. 前半で見て、後半で答え合わせ")
        ys = sorted(frames)
        half = len(ys) // 2
        tr = pd.concat([frames[y] for y in ys[:half]], ignore_index=True)
        te = pd.concat([frames[y] for y in ys[half:]], ignore_index=True)
        print(f"  学習 FY{ys[:half]}（{len(tr):,}）／ 検証 FY{ys[half:]}（{len(te):,}）\n")
        for nm, sub in (("学習", tr), ("検証", te)):
            hi = sub["start_yield"] >= sub["start_yield"].quantile(0.7)
            okp = (-sub["payout_neg"]).between(0.0, 0.5)
            okf = sub["ocf_cover"] >= 3
            base = sub[hi]
            best = sub[hi & okp & okf]
            if len(base) < 30 or len(best) < 30:
                print(f"  {nm}: 少なすぎる")
                continue
            print(f"  {nm}：高利回りのみ 減配率 {base['fwd_had_cut'].mean():.1%} → "
                  f"足切りあり {best['fwd_had_cut'].mean():.1%}"
                  f"（{best['fwd_had_cut'].mean() - base['fwd_had_cut'].mean():+.1f}pt）"
                  f"／ リターン {base['fwd_total_return'].median():+.1%} → "
                  f"{best['fwd_total_return'].median():+.1%}")
        print("\n  **後半でも同じ向きに動いていれば、足切りを既定にしてよい。**")


if __name__ == "__main__":
    main()
