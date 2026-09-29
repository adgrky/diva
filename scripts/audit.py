"""アプリ全体の機械的な検算。お金の判断に使うので、感覚で「たぶん大丈夫」としない。

    uv run python scripts/audit.py

やること
    A 表示される数字を、独立に計算し直して突き合わせる
    B 画面をまたいだ判定の矛盾を探す（買い場と言いながら売れと言っていないか）
    C 空データ・0円・NaN で関数が落ちないか
    D 書き込み（買い・売り・判断・編集）を DBのコピー で実際に動かして確かめる

本番のDBには一切書き込まない。D はコピーを作ってそちらを触る。
"""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np                                            # noqa: E402
import pandas as pd                                           # noqa: E402

NG: list[str] = []


def chk(name: str, ok: bool, detail: str = "") -> None:
    print(("  OK  " if ok else "  NG  ") + name + (f"   {detail}" if detail else ""))
    if not ok:
        NG.append(name)


def section(title: str) -> None:
    print(f"\n=== {title} ===")


# ──────────────────────────────────────────── A 数字の検算
def audit_numbers() -> None:
    from modules.config import load_config
    from modules.portfolio import load_positions, sector_exposure, dividend_calendar
    from modules.store import latest_scores, read_df
    from modules.allocator import allocate, month_gaps
    from modules import sell_rules as SR, market

    cfg = load_config()
    pos = load_positions(cfg)
    if pos.empty:
        print("  （保有データが無いので飛ばします）")
        return

    section("A1 保有の集計")
    chk("評価額 = 株数 × 株価", np.allclose(pos["eval_value"], pos["shares"] * pos["last_close"],
                                       equal_nan=True))
    chk("取得額 = 株数 × 取得単価", np.allclose(pos["cost_value"], pos["shares"] * pos["avg_cost"],
                                        equal_nan=True))
    chk("年間配当 = 株数 × DPS",
        np.allclose(pos["annual_dividend"], pos["shares"] * pos["dps_latest"].fillna(0)))
    nisa, spec = pos[pos.account == "nisa"], pos[pos.account == "specific"]
    chk("NISA は非課税", np.allclose(nisa["annual_dividend_after_tax"], nisa["annual_dividend"]))
    chk("特定口座は 20.315% 課税",
        np.allclose(spec["annual_dividend_after_tax"],
                    spec["annual_dividend"] * (1 - cfg["portfolio"]["tax_rate_specific"])))
    chk("株価が欠けている保有がない", pos["last_close"].isna().sum() == 0)
    chk("業種が欠けている保有がない", pos["sector33"].isna().sum() == 0)

    section("A2 業種の配分")
    ex = sector_exposure(pos, cfg)
    chk("構成比の合計がちょうど100%", abs(ex["構成比"].sum() - 1.0) < 1e-9,
        f"{ex['構成比'].sum():.10f}")

    section("A3 資金配分の保存則（単元未満株）")
    sc = latest_scores()
    cand = sc[sc["gate_passed"] == 1].copy().reset_index(drop=True)
    raw = pd.DataFrame([json.loads(x)["raw"] for x in cand["detail_json"]])
    cand = pd.concat([cand, raw[["dividend_yield", "yield_percentile", "streak",
                                 "payout_months"]]], axis=1)
    tg = read_df("SELECT ticker, target_yield FROM watchlist "
                 "UNION SELECT ticker, target_yield FROM holdings").groupby("ticker").first()
    cand = cand.join(tg, on="ticker")
    gaps = month_gaps(pos, dividend_calendar(pos))
    for cash, n in ((500000, 12), (100000, 8), (30000, 5), (10000, 3), (3000, 2)):
        p = allocate(cash, cand, pos, cfg, max_names=n, per_name_cap_pct=1 / n,
                     require_below_target=False, month_gap=gaps, lot=1)
        if p.empty:
            chk(f"入金 {cash:,} で配分できる", False, "0銘柄")
            continue
        total, rest = p["投入額"].sum(), p.attrs["残り"]
        chk(f"入金 {cash:,}：投入 + 残り = 入金", abs(total + rest - cash) < 0.01,
            f"{total:,.0f} + {rest:,.0f}（{total / cash:.1%} を投入）")
        chk(f"入金 {cash:,}：株数が整数", (p["株数"] == p["株数"].astype(int)).all())
        chk(f"入金 {cash:,}：銘柄が重複しない", p["ticker"].is_unique)
    p100 = allocate(500000, cand, pos, cfg, max_names=12, per_name_cap_pct=0.15,
                    require_below_target=False, month_gap=gaps, lot=100)
    chk("単元モードでは100株の倍数", (p100["株数"] % 100 == 0).all())

    section("A3.5 配当の偏りの上限が効いているか")
    # 2026-09-16 の検証14・9 で入れた上限。評価額ではなく**配当**にかけている。
    cyc = set(cfg.get("cyclical", {}).get("sectors", []))
    cap_cyc = float(cfg.get("cyclical", {}).get("max_income_share", 1.0))
    cap_one = float(cfg["portfolio"].get("max_income_weight", 1.0))
    inc_now = pos.groupby("ticker")["annual_dividend"].sum()
    cyc_now = float(pos.loc[pos["sector33"].isin(cyc), "annual_dividend"].sum())
    base = float(pos["annual_dividend"].sum())
    for cash in (100_000, 1_000_000):
        plan = allocate(cash, cand, pos, cfg, max_names=30, lot=1,
                        require_below_target=False)
        if plan.empty:
            continue
        after = base + plan["年間配当"].sum()
        add_cyc = plan.loc[plan["業種"].isin(cyc), "年間配当"].sum()
        share = (cyc_now + add_cyc) / after if after else 0.0
        before = cyc_now / base if base else 0.0
        chk(f"入金 {cash:,}：景気敏感の配当シェアが悪化しない",
            share <= before + 1e-9, f"{before:.1%} → {share:.1%}")
        worst = 0.0
        for r in plan.itertuples():
            mine = float(inc_now.get(r.ticker, 0.0)) + r.年間配当
            worst = max(worst, mine / after if after else 0.0)
        chk(f"入金 {cash:,}：1銘柄の配当シェアが上限を超えない",
            worst <= cap_one + 1e-6 or worst <= (inc_now.max() / base if base else 1),
            f"最大 {worst:.1%}（上限 {cap_one:.0%}）")

    section("A4 売り判定")
    ev = SR.evaluate(pos, cfg)
    chk("優先度が 0〜100 に収まる", ev["整理の優先度"].between(0, 100).all())
    chk("理由が空の行がない", (ev["理由"].str.strip() != "").all())
    fin = ev[ev["sector33"].isin(SR._FINANCIAL)]
    chk("金融に営業CF・負債の基準を当てていない",
        fin[fin["理由"].str.contains("営業キャッシュフロー|借金が重く")].empty)
    chk("新規買いの条件を売り理由にしていない",
        ev[ev["理由"].str.contains("上場5年未満|売買代金|時価総額|財務未取得")].empty)

    section("A5 大人買いライン")
    snap = market.load_snapshot()
    if snap.get("pbr") and snap.get("nikkei"):
        lad = market.buy_ladder(snap, cfg, 1_000_000)
        want = [snap["nikkei"] * float(s["pbr"]) / snap["pbr"] for s in cfg["market"]["ladder"]]
        chk("日経の水準 = いまの日経 × 目標PBR ÷ いまのPBR",
            np.allclose(lad["日経平均の水準"].values, want))
        chk("投入割合の合計が100%", abs(lad["投入する割合"].sum() - 1.0) < 1e-9)

    section("A6 年度の切り方（作った配当明細で確かめる）")
    from modules.dividend_history import annual_dps, build_profile
    # 2026-09-16 に見つかったバグの再発防止。権利落ち日は年ごとに数日ずれる。
    # 直近の権利落ち日から1年ずつ遡って切ると、ズレが積み上がって
    # 「1年に3回ぶん入る年」ができ、その翌年が減配に見える。
    # ここは実データではなく**作った明細**で見る（実データが変わっても意味が変わらない）。
    drift = pd.DataFrame({
        "date": ["2021-03-30", "2021-09-29", "2022-03-30", "2022-09-29",
                 "2023-03-30", "2023-09-28", "2024-03-28", "2024-09-27",
                 "2025-03-28"],
        "amount": [8, 8, 9, 9, 10, 10, 14, 12, 16],
    })
    tbl = annual_dps(drift)
    chk("権利落ち日が数日ずれても1年は2回ぶん",
        (tbl["n_payments"].iloc[1:] == 2).all(), tbl["n_payments"].tolist())
    chk("増配しかしていない明細で減配が出ない",
        build_profile("TEST", drift).cuts_all == 0,
        f"減配 {build_profile('TEST', drift).cuts_all} 回／{dict(tbl['dps'])}")
    # 途中で切っても、全期間で見ても、同じ年度の配当は同じ額でなければならない
    cut_short = drift[drift["date"] <= "2024-09-27"]
    a = annual_dps(cut_short)["dps"]
    b = annual_dps(drift)["dps"]
    common = a.index.intersection(b.index)
    chk("途中で切っても年度ごとの配当が変わらない",
        bool((a[common] == b[common]).all()),
        f"切った版 {dict(a[common])} ／ 全期間 {dict(b[common])}")
    # 本物の減配は見落とさない
    real_cut = pd.DataFrame({
        "date": ["2022-03-30", "2022-09-29", "2023-03-30", "2023-09-28",
                 "2024-03-28", "2024-09-27"],
        "amount": [20, 20, 20, 20, 10, 10],
    })
    chk("本物の減配は拾える", build_profile("TEST", real_cut).cuts_all == 1,
        f"減配 {build_profile('TEST', real_cut).cuts_all} 回")
    # 中間配当だけ済んでいて期末配当がまだ、という時期（毎年9月〜3月）。
    # そのまま数えると全社が減配したように見える。
    mid_year = pd.concat([drift, pd.DataFrame({"date": ["2025-09-29"],
                                               "amount": [17]})], ignore_index=True)
    pr = build_profile("TEST", mid_year)
    chk("期末配当がまだの年を減配と数えない", pr.cuts_all == 0,
        f"減配 {pr.cuts_all} 回／{pr.series}")


# ──────────────────────────────────────────── B 画面をまたいだ矛盾
def audit_contradictions() -> None:
    from modules.config import load_config
    from modules.portfolio import load_positions, dividend_calendar
    from modules.store import latest_scores, read_df
    from modules.allocator import buy_priority, buy_gate, month_gaps
    from modules import sell_rules as SR
    from modules.monitor import build_alerts

    cfg = load_config()
    pos = load_positions(cfg)
    sc = latest_scores()
    if pos.empty or sc.empty:
        print("  （データが無いので飛ばします）")
        return
    ev = SR.evaluate(pos, cfg)
    sev = ev.drop_duplicates("ticker").set_index("ticker")["重さ"].to_dict()
    name = sc.set_index("ticker")["name"].to_dict()

    cand = sc[sc["gate_passed"] == 1].copy().reset_index(drop=True)
    raw = pd.DataFrame([json.loads(x)["raw"] for x in cand["detail_json"]])
    cand = pd.concat([cand, raw[["dividend_yield", "yield_percentile", "streak",
                                 "payout_months"]]], axis=1)
    tg = read_df("SELECT ticker, target_yield FROM watchlist "
                 "UNION SELECT ticker, target_yield FROM holdings").groupby("ticker").first()
    cand = cand.join(tg, on="ticker")
    gaps = month_gaps(pos, dividend_calendar(pos))
    bp, _ = buy_gate(buy_priority(cand, pos, cfg, gaps), cfg)
    top = bp.nlargest(50, "買い付け優先度")

    section("B 画面をまたいだ矛盾")
    bad = [t for t in top["ticker"] if sev.get(t) in ("売却を検討", "監視を強める")]
    chk("買い付け上位50に、売れと判定した銘柄がない", not bad,
        "／".join(name.get(t, t) for t in bad))
    tp = {t for t, s in sev.items() if s == "利確を検討"}
    chk("買い付け上位50に「利確を検討」がない", not [t for t in top["ticker"] if t in tp])
    held = set(pos["ticker"])
    addmore = set(cand[(cand["ticker"].isin(held))
                       & (cand["yield_percentile"].fillna(0) >= 0.7)]["ticker"])
    chk("「買い増しどき」と「利確を検討」が重ならない", not (addmore & tp),
        "／".join(name.get(t, t) for t in (addmore & tp)))
    chk("「買い増しどき」と「売却を検討」が重ならない",
        not {t for t in addmore if sev.get(t) == "売却を検討"})

    a = build_alerts(cfg)
    if not a.empty:
        reach = set(a[a.kind == "target_reached"].ticker)
        chk("監視の「買い場」に、売れと判定した銘柄がない",
            not [t for t in reach if sev.get(t) in ("売却を検討", "監視を強める")])
        chk("監視の「買い場」とトラップが同時に出ない",
            not (reach & set(a[a.kind == "trap"].ticker)))
        chk("同じ銘柄に重大警報が2つ出ない",
            not (a[a.severity == "high"].groupby("ticker").size() > 1).any())
        # 監視の種別を増やしたのに画面側に足し忘れると、警報が黙って消える
        import re
        ui = set(re.findall(r'"(\w+)":\s+\("',
                            (Path(__file__).resolve().parent.parent
                             / "views" / "3_monitor.py").read_text()))
        missing = set(a["kind"]) - ui
        chk("監視が出す警報を、画面がすべて知っている", not missing,
            "／".join(sorted(missing)))


# ──────────────────────────────────────────── C 空データ・異常値
def audit_edges() -> None:
    from modules.config import load_config
    from modules import portfolio as P, allocator as A, sell_rules as SR
    from modules import peers as PE, market as M, review as R, quality as Q
    from modules.dividend_history import next_ex_dates, build_profiles

    cfg = load_config()
    empty = pd.DataFrame()
    section("C 空データ・0円・NaN")
    cases = [
        ("業種の配分", lambda: P.sector_exposure(empty, cfg)),
        ("配当カレンダー", lambda: P.dividend_calendar(empty)),
        ("配当の自動生成", lambda: P.expected_dividends(empty, cfg)),
        ("売り判定", lambda: SR.evaluate(empty, cfg)),
        ("資金配分", lambda: A.allocate(100000, empty, empty, cfg)),
        ("買い付け優先度（列が欠けている）",
         lambda: A.buy_priority(pd.DataFrame(columns=[
             "yield_percentile", "dividend_yield", "health", "sector33",
             "payout_months", "last_close", "ticker"]), empty, cfg)),
        ("同業比較", lambda: PE.build(pd.DataFrame(columns=[
            "sector33", "gate_passed", "ticker", "detail_json"]), "機械", "9999.T")),
        ("大人買いライン（相場データ無し）", lambda: M.buy_ladder({}, cfg, 0)),
        ("相場の区分（データ無し）", lambda: M.regime({}, cfg)),
        ("判断の貼り付け", lambda: R.attach(empty)),
        ("権利落ち日（存在しない銘柄）", lambda: next_ex_dates(["0000.T"])),
        ("配当プロフィール（空）",
         lambda: build_profiles(pd.DataFrame(columns=["ticker", "date", "amount"]))),
        ("株価の破損検出（空）",
         lambda: Q.trim_frame(pd.DataFrame(columns=["ticker", "date", "close"]))),
    ]
    for label, fn in cases:
        try:
            fn()
            chk(label, True)
        except Exception as exc:
            chk(label, False, f"{type(exc).__name__}: {exc}")

    one = pd.DataFrame([{
        "account": "specific", "ticker": "9999.T", "code": "9999", "name": "テスト",
        "sector33": "機械", "shares": 1.0, "avg_cost": 0.0, "last_close": 100.0,
        "eval_value": 100.0, "cost_value": 0.0, "pnl": 100.0, "pnl_pct": np.nan,
        "annual_dividend": 0.0, "annual_dividend_after_tax": 0.0, "yoc": np.nan,
        "current_yield": np.nan, "dps_latest": 0.0, "health": np.nan, "streak": 0,
        "cuts_10y": 0, "streak_no_cut": np.nan, "gate_passed": 0, "gate_reason": "",
        "detail_json": None, "target_yield": 0.047}])
    for label, fn in (("1銘柄・取得額0の業種配分", lambda: P.sector_exposure(one, cfg)),
                      ("1銘柄・取得額0の売り判定", lambda: SR.evaluate(one, cfg))):
        try:
            fn(); chk(label, True)
        except Exception as exc:
            chk(label, False, f"{type(exc).__name__}: {exc}")


def audit_frozen_prices() -> None:
    """株価が止まった区間（yfinance の埋め草）が週足に残っていないか。

    残っていると、その区間をまたいだ過去検証のリターンが偽物になる
    （実測: 7564.T の FY2018 で ret_1y +1546%）。直し方は
    scripts/repair_frozen_prices.py。
    """
    from modules.quality import FROZEN_WEEKS, frozen_runs
    from modules.store import read_df

    section("C2 株価が止まった区間")
    px = read_df("SELECT ticker, date, close FROM prices")
    if px.empty:
        print("  （週足が無いので飛ばします）")
        return
    long = frozen_runs(px, FROZEN_WEEKS)
    chk(f"同じ終値が{FROZEN_WEEKS}週以上続く区間が無い", long.empty,
        f"{len(long)} 区間 / {long['ticker'].nunique()} 銘柄 → "
        "scripts/repair_frozen_prices.py で消す" if len(long) else "")
    for r in long.head(10).itertuples(index=False):
        print(f"       {r.ticker} {r.close:,.2f} {r.start}〜{r.end}（{r.weeks}週）")
    # 低位株は呼び値1円で本当に半年動かないことがあるので、ここは NG にせず見せるだけ
    mid = frozen_runs(px, 26)
    mid = mid[mid["weeks"] < FROZEN_WEEKS]
    if len(mid):
        print(f"  参考 同じ終値が26〜{FROZEN_WEEKS - 1}週続く区間 {len(mid)} か所"
              "（低位株なら本物のことがある）")
        for r in mid.head(5).itertuples(index=False):
            print(f"       {r.ticker} {r.close:,.2f} {r.start}〜{r.end}（{r.weeks}週）")

    # 上場廃止した銘柄は yfinance が新しい株価を返さなくなる。一覧に残っていれば
    # 週次更新の削除（weekly_scan.purge_delisted）が漏れている。
    # 保有・ウォッチ中は売却の記録が要るので、わざと残している。
    section("C3 上場廃止の取り残し")
    q = read_df("SELECT u.ticker, u.name, q.asof FROM universe u "
                "LEFT JOIN quotes q ON q.ticker = u.ticker")
    keep = set(read_df("SELECT ticker FROM holdings UNION SELECT ticker FROM watchlist")
               ["ticker"])
    newest = pd.to_datetime(q["asof"]).max()
    old = q[(pd.to_datetime(q["asof"]) < newest - pd.Timedelta(days=45))
            & ~q["ticker"].isin(keep)]
    chk("45日以上株価が更新されない銘柄が一覧に無い", old.empty,
        f"{len(old)} 銘柄（上場廃止の取り残しの疑い）: "
        + ", ".join(old["ticker"].head(10)) if len(old) else "")


# ──────────────────────────────────────────── D 書き込み（DBのコピーで）
def audit_writes() -> None:
    from modules.config import db_path
    real = db_path()
    if not real.exists():
        print("  （DBが無いので飛ばします）")
        return
    tmp = Path(tempfile.mkdtemp()) / "sandbox.db"
    shutil.copy2(real, tmp)

    import sqlite3
    import modules.config as C
    import modules.store as S
    orig = C.db_path
    C.db_path = lambda config=None: tmp
    S.db_path = C.db_path
    # クラウド(Turso)につながったままだと、保有・売買はコピーではなく本物に書かれる。
    guard = S.local_only()
    guard.__enter__()
    try:
        with S.connect() as probe:
            if not isinstance(probe, sqlite3.Connection):
                chk("書き込みの検算がパソコンのコピーだけに向いている", False,
                    "クラウドに繋がったままなので、書き込みの検算を中止")
                return
        chk("書き込みの検算がパソコンのコピーだけに向いている", True)
        S.init_db()
        from modules.config import load_config
        from modules.store import connect, read_df
        from modules import review as R
        from modules.portfolio import load_positions, expected_dividends
        cfg = load_config()

        def buy(account, ticker, name, shares, price):
            with connect() as conn:
                conn.execute("INSERT INTO transactions (date, account, ticker, name, type,"
                             " shares, price, fee, memo) VALUES (?,?,?,?,'buy',?,?,0,'監査')",
                             (date.today().isoformat(), account, ticker, name, shares, price))
                cur = conn.execute("SELECT shares, avg_cost FROM holdings"
                                   " WHERE account=? AND ticker=?", (account, ticker)).fetchone()
                if cur and cur["shares"]:
                    ns = float(cur["shares"]) + shares
                    nc = (float(cur["shares"]) * float(cur["avg_cost"] or 0)
                          + shares * price) / ns
                else:
                    ns, nc = shares, price
                conn.execute(
                    "INSERT OR REPLACE INTO holdings (account, ticker, name, shares, avg_cost,"
                    " target_yield, bottom_yield, updated_at) VALUES (?,?,?,?,?,"
                    " COALESCE((SELECT target_yield FROM holdings WHERE account=? AND ticker=?),"
                    " 0.047), (SELECT bottom_yield FROM holdings WHERE account=? AND ticker=?),"
                    " datetime('now'))",
                    (account, ticker, name, ns, nc, account, ticker, account, ticker))

        def sell(account, ticker, name, shares, price):
            with connect() as conn:
                conn.execute("INSERT INTO transactions (date, account, ticker, name, type,"
                             " shares, price, fee, memo) VALUES (?,?,?,?,'sell',?,?,0,'監査')",
                             (date.today().isoformat(), account, ticker, name, shares, price))
                cur = conn.execute("SELECT shares FROM holdings WHERE account=? AND ticker=?",
                                   (account, ticker)).fetchone()
                left = (float(cur["shares"]) if cur else 0.0) - shares
                if left <= 0.5:
                    conn.execute("DELETE FROM holdings WHERE account=? AND ticker=?",
                                 (account, ticker))
                    conn.execute("DELETE FROM holding_review WHERE account=? AND ticker=?",
                                 (account, ticker))
                else:
                    conn.execute("UPDATE holdings SET shares=?, updated_at=datetime('now')"
                                 " WHERE account=? AND ticker=?", (left, account, ticker))

        section("D 書き込み（DBのコピーで実際に動かす）")
        buy("specific", "9999.T", "監査テスト", 7, 1234.0)
        h = read_df("SELECT * FROM holdings WHERE ticker='9999.T'")
        chk("買うと保有ができる", len(h) == 1 and float(h.shares.iloc[0]) == 7.0)
        buy("specific", "9999.T", "監査テスト", 3, 2000.0)
        h = read_df("SELECT * FROM holdings WHERE ticker='9999.T'")
        want = (7 * 1234.0 + 3 * 2000.0) / 10
        chk("買い増しで平均取得単価が正しく動く",
            float(h.shares.iloc[0]) == 10.0 and abs(float(h.avg_cost.iloc[0]) - want) < 1e-9,
            f"{h.avg_cost.iloc[0]:.2f}（期待 {want:.2f}）")
        cost = float(h.avg_cost.iloc[0])
        sell("specific", "9999.T", "監査テスト", 4, 1500.0)
        h = read_df("SELECT * FROM holdings WHERE ticker='9999.T'")
        chk("一部売却で株数が減り、取得単価は据え置き",
            float(h.shares.iloc[0]) == 6.0 and abs(float(h.avg_cost.iloc[0]) - cost) < 1e-9)
        sell("specific", "9999.T", "監査テスト", 6, 1500.0)
        chk("全部売却で保有から消える",
            read_df("SELECT * FROM holdings WHERE ticker='9999.T'").empty)
        chk("売買の履歴は残る",
            len(read_df("SELECT * FROM transactions WHERE ticker='9999.T'")) == 4)

        buy("nisa", "9998.T", "監査1株", 1, 7210.0)
        buy("nisa", "9998.T", "監査1株", 1, 7300.0)
        h = read_df("SELECT * FROM holdings WHERE ticker='9998.T'")
        chk("1株ずつ買い増せる（単元未満株）",
            float(h.shares.iloc[0]) == 2.0 and abs(float(h.avg_cost.iloc[0]) - 7255.0) < 1e-9)
        sell("nisa", "9998.T", "監査1株", 1, 7500.0)
        chk("1株残る売却で消えない",
            len(read_df("SELECT * FROM holdings WHERE ticker='9998.T'")) == 1)
        sell("nisa", "9998.T", "監査1株", 1, 7500.0)
        chk("最後の1株を売ると消える",
            read_df("SELECT * FROM holdings WHERE ticker='9998.T'").empty)

        pos = load_positions(cfg)
        if not pos.empty:
            first = pos.iloc[0]
            R.save(first["account"], first["ticker"], "keep")
            chk("判断を記録できる", len(R.load()) == 1)
            R.save(first["account"], first["ticker"], "watch")
            d = R.load()
            chk("判断を上書きしても重複しない",
                len(d) == 1 and d.decision.iloc[0] == "watch")
            R.clear(first["account"], first["ticker"])
            chk("判断を取り消せる", R.load().empty)
            R.save(first["account"], first["ticker"], "keep")
            sell(first["account"], first["ticker"], first["name"],
                 float(first["shares"]), 1000.0)
            chk("保有が消えたら判断も消える", R.load().empty)

            # 比べるのは「どの設定でも入金済みになる古い配当」だけ。最近の権利落ちは
            # 入金までの日数が短い設定でだけ出てくるのが正しい動きで、件数が違って当然。
            # 実測（2026-09-29）: 8月末の権利落ち3件が 30日 でだけ出て、誤って NG になった。
            lags = (30, 45, 75, 110, 150)
            cutoff = (pd.Timestamp.today().normalize()
                      - pd.Timedelta(days=max(lags))).date()

            def settled(lag):
                df = expected_dividends(load_positions(cfg), cfg, 12, lag)
                return 0 if df.empty else int((df["権利落ち日"] <= cutoff).sum())
            counts = {lag: settled(lag) for lag in lags}
            chk("入金までの日数を変えても配当が二重計上にならない",
                len(set(counts.values())) == 1, str(counts))
    finally:
        guard.__exit__(None, None, None)
        C.db_path = orig
        S.db_path = orig
        shutil.rmtree(tmp.parent, ignore_errors=True)


# ──────────────────────────────────────────── E コードの健全性
def audit_code() -> None:
    """これまで実際に出たバグの「型」を機械で洗う。

    手で気づけたものだけ直していると、同じ型のバグが別の場所で再発する。
    実際に出た型:
        ・設定にあるのにコードが直書きしていて、つまみが効かない
        ・同じ概念に2つの閾値があり、画面ごとに答えが変わる
        ・置き換え済みの関数が残っていて、誤って使われる
        ・画面の入力がどこにも使われていない
    """
    import ast
    import re
    import yaml

    section("E コードの健全性")
    root = Path(__file__).resolve().parent.parent
    files = (sorted((root / "modules").glob("[!.]*.py")) + sorted((root / "views").glob("[!.]*.py"))
             + sorted((root / "scripts").glob("[!.]*.py")) + [root / "app.py"])
    src_by_file = {f: f.read_text() for f in files}
    allsrc = "\n".join(src_by_file.values())

    # E1 設定にあるのに読んでいない項目＝効かないつまみ
    cfg = yaml.safe_load((root / "config.yaml").read_text())

    def walk(d, path=()):
        for k, v in (d or {}).items():
            if isinstance(v, dict):
                yield from walk(v, path + (k,))
            else:
                yield path + (k,), v

    unread = []
    for path, v in walk(cfg):
        if isinstance(v, list) and all(isinstance(x, dict) for x in (v or [])):
            continue
        if not re.search(rf'["\']{re.escape(path[-1])}["\']', allsrc):
            unread.append(".".join(path))
    chk("設定の項目がすべてコードから読まれている（効かないつまみが無い）",
        not unread, "／".join(unread))

    # E2 画面の入力が使われているか
    dead_widgets = []
    for f in sorted((root / "views").glob("[!.]*.py")):
        src = src_by_file[f]
        for n in ast.walk(ast.parse(src)):
            if isinstance(n, ast.Assign) and isinstance(n.value, ast.Call):
                w = getattr(n.value.func, "attr", None)
                if w in ("number_input", "slider", "selectbox", "multiselect", "checkbox",
                         "radio", "text_input", "text_area", "date_input", "toggle"):
                    for t in n.targets:
                        if isinstance(t, ast.Name) and \
                                len(re.findall(rf"\b{re.escape(t.id)}\b", src)) <= 1:
                            dead_widgets.append(f"{f.name}:{n.lineno} {t.id}")
    chk("画面の入力がすべて実際に使われている", not dead_widgets, "／".join(dead_widgets))

    # E3 置き換え済みの関数が残っていないか
    defined = {}
    for f in files:
        for n in ast.walk(ast.parse(src_by_file[f])):
            if isinstance(n, ast.FunctionDef) and not n.name.startswith("_"):
                defined.setdefault(n.name, []).append(f"{f.name}:{n.lineno}")
    # 入口になる関数や、スクリプトから呼ばれるものは対象外
    keep = {"main", "flash", "show_flash"}
    dead = []
    for n, where in defined.items():
        if n in keep:
            continue
        # 定義行そのものを除いて、名前がどこかに現れるか。
        # 「fn(」だけを数えると、辞書やタプルに関数を入れて渡す書き方
        # （("capacity", score_capacity) など）を見落とす。
        occurrences = len(re.findall(rf"\b{re.escape(n)}\b", allsrc))
        if occurrences - len(where) <= 0:
            dead.append(f"{n}（{where[0]}）")
    chk("使われていない関数が残っていない", not dead, "／".join(sorted(dead)))

    # E4 監視と買う側で、トラップの足切りが揃っているか
    mon = src_by_file[root / "modules" / "monitor.py"]
    chk("監視のトラップ判定が、買う側と同じ設定を使っている",
        "penalty >= max_trap" in mon,
        "NG なら監視だけ別の数字を直書きしている")

    # E5 減配の判定が1か所に集約されているか
    # 実測で、検証スクリプトが自前で年度を切っていて
    #   ・起点の年から1年目への減配を見落とす
    #   ・期間の外（horizon+1年目）まで見てしまう（先読み）
    # という取りこぼしが起きた。判定が2通りあると、同じ銘柄に違う答えが出る。
    dupes = []
    for f in sorted((root / "scripts").glob("valid*.py")) + \
            sorted((root / "modules").glob("valid*.py")) + \
            [root / "modules" / "hist_panel.py"]:
        if not f.exists():
            continue
        src = src_by_file.get(f, f.read_text())
        # 減配を扱っているのに、判定を自前で書いている（共有の入り口を通っていない）
        uses_cut = "had_cut" in src or "cut_1y" in src or "cut_" in src and "cut_years" in src
        # 共有の入り口は3つ。どれかを通っていれば、判定は1か所に集約されている。
        #   modules.dividend_history.build_profile  … 判定そのもの
        #   modules.hist_panel.load_cached          … パネル（内部で build_profile）
        #   modules.validation.build_outcomes       … 本検証（内部で build_profile）
        shared = ("build_profile" in src or "load_cached" in src
                  or "from modules.hist_panel" in src
                  or "build_outcomes" in src)
        if uses_cut and not shared:
            dupes.append(f.name)
    chk("減配の判定が build_profile に統一されている", not dupes, "／".join(dupes))

    # E6 年度の切り方が1か所に集約されているか
    # 実測（2026-09-16）: income_risk が「4月〜翌3月」と決め打ちしていて、
    # dividend_history.annual_dps と定義が食い違っていた。同じ「2019年度の配当」が
    # 画面によって別の数字になる。年度を自前で切っているファイルを見つける。
    fy_dupes = []
    for f in sorted((root / "modules").glob("[!.]*.py")):
        if f.name in ("dividend_history.py", "hist_panel.py", "jp_calendar.py"):
            continue
        src = src_by_file.get(f, f.read_text())
        if '-04-01' in src and "annual_dps" not in src:
            fy_dupes.append(f.name)
    chk("年度の切り方が annual_dps に統一されている", not fy_dupes, "／".join(fy_dupes))

    # E7 数値の列に文字列が紛れ込んでいないか
    # 実測（2026-09-16）: yfinance が赤字の会社の PER に Infinity を返し、SQLite が
    # それを **文字列 "Infinity"** として保存していた。読み戻すと列全体が文字列になり、
    # `df["per"] > 0` で週次スキャンが落ちる。画面には何も出ないまま更新が止まる。
    from modules.store import read_df as _rdf
    numeric = {
        "snapshots": ["market_cap", "per", "pbr", "roe", "payout_ratio"],
        "quotes": ["last_close", "high_52w", "low_52w", "pos_52w", "avg_turnover"],
        "scores": ["total", "health", "trap_penalty"],
        "fundamentals": ["net_income", "operating_cf", "free_cf", "shares"],
        "edinet_summary": ["eps", "dps", "net_income", "operating_cf"],
    }
    dirty = []
    for table, cols in numeric.items():
        for c in cols:
            try:
                n = _rdf(f"SELECT COUNT(*) n FROM {table} "
                         f"WHERE {c} IS NOT NULL AND typeof({c}) = 'text'")["n"].iloc[0]
            except Exception:
                continue
            if int(n):
                dirty.append(f"{table}.{c} に {int(n)} 行")
    chk("数値の列に文字列が入っていない", not dirty, "／".join(dirty))

    # E8 config.yaml を書き換えたら、その場で効くか
    # 実測（2026-09-16）: load_config が lru_cache でプロセスに固めていたため、
    # 大人買いラインの段を書き換えてもアプリは古い段を出し続けた。しかも
    # 画面には何の手がかりも出ない。「コードに数値を直書きしない」と決めている以上、
    # 設定が効かないのは黙って壊れているのと同じ。
    import time as _t
    from modules.config import load_config as _lc
    cfg_path = root / "config.yaml"
    original = cfg_path.read_text(encoding="utf-8")
    before = _lc()["portfolio"]["max_sector_weight"]
    try:
        cfg_path.write_text(
            original.replace(f"max_sector_weight: {before}",
                             "max_sector_weight: 0.999"), encoding="utf-8")
        _t.sleep(0.02)
        after = _lc()["portfolio"]["max_sector_weight"]
    finally:
        cfg_path.write_text(original, encoding="utf-8")
        _t.sleep(0.02)
        _lc()
    chk("config.yaml を書き換えたら、その場で効く", after == 0.999,
        f"NG なら書き換えが効いていない（読めた値 {after}）")


def main() -> int:
    print("=" * 62)
    print("  diva  機械的な検算")
    print("=" * 62)
    audit_numbers()
    audit_contradictions()
    audit_edges()
    audit_frozen_prices()
    audit_writes()
    audit_code()
    print("\n" + "=" * 62)
    if NG:
        print(f"  ⚠️  NG {len(NG)} 件")
        for x in NG:
            print(f"     ・{x}")
        return 1
    print("  ✅ すべて通りました")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
