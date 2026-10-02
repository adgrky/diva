"""監視 — 保有と候補の異変を拾う。

発掘の裏返し。買ったあとに仮説が壊れていないかを機械が見張る。
アラートは「棄却条件に触れたか」を中心に組む。単に株価が下がったことは
アラートにしない（配当株では株価下落はむしろ買い場になる）。

【判定の出どころはひとつにする】
以前ここは、売る判定を自前で持っていた。その結果、

    極東証券        監視「目標利回りに到達＝買い場。良い知らせ」
                    整理「🔴 10年で5回も減配している」
    ヒラノテクシード  監視「買い場」／整理「🔴 直近の配当年度で減配した」

    （当時ここには日本製鉄の「配当性向2193%」も並べていたが、あの数字自体が
      取り込みの事故だった。権利落ち日と分割日が同じ日の配当に分割が当たらず、
      年間配当が3倍で積まれていた。2026-09-23 に修正済み）

という **正反対の指示が同時に出ていた**（実測6銘柄）。
さらに「採用基準を外れた」を high の警報にしていたため、160件中99件が
「新規で買う条件を満たさないだけ」の雑音で埋まっていた。

いまは
    売る側の判定 … modules/sell_rules.evaluate（起きた事実だけに反応）
    買う側の判定 … modules/allocator.buy_gate（配当継続とトラップで足切り）
を **そのまま使う**。画面が違っても答えは必ず一致する。
"""
from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd

from modules.store import connect, read_df

SEVERITY_ORDER = {"high": 0, "medium": 1, "low": 2}


def build_alerts(config: dict) -> pd.DataFrame:
    """保有＋ウォッチリストを対象に異変を検出する。"""
    targets = read_df(
        "SELECT ticker FROM holdings UNION SELECT ticker FROM watchlist"
    )["ticker"].tolist()
    if not targets:
        return pd.DataFrame()

    placeholders = ",".join("?" * len(targets))
    sc = read_df(
        f"SELECT * FROM scores WHERE asof = (SELECT MAX(asof) FROM scores) "
        f"AND ticker IN ({placeholders})", tuple(targets)
    ).set_index("ticker").drop(columns=["asof"], errors="ignore")
    uni = read_df(f"SELECT ticker, name, sector33 FROM universe WHERE ticker IN ({placeholders})",
                  tuple(targets)).set_index("ticker")
    q = read_df(f"SELECT ticker, last_close, pos_52w, high_52w, low_52w, ret_1y "
                f"FROM quotes WHERE ticker IN ({placeholders})",
                tuple(targets)).set_index("ticker")
    div = read_df(f"SELECT ticker, date, amount FROM dividends WHERE ticker IN ({placeholders})",
                  tuple(targets))
    watch = read_df("SELECT ticker, target_yield, bottom_yield, invalidation FROM watchlist"
                    ).set_index("ticker")
    hold = read_df("SELECT ticker, target_yield, bottom_yield FROM holdings"
                   ).groupby("ticker").first()

    from modules.dividend_history import build_profiles, fiscal_months, profiles_to_frame
    prof = profiles_to_frame(build_profiles(div, fiscal_months()))

    targets_yield = watch[["target_yield", "bottom_yield"]].combine_first(hold)
    df = uni.join([q, sc, prof, targets_yield], how="left")
    df["dividend_yield"] = np.where(df["last_close"] > 0,
                                    df["dps_latest"] / df["last_close"], np.nan)

    # 売り側の判定を一度だけ作って共有する（画面ごとに別の答えを出さないため）
    sell_verdict: dict = {}
    try:
        from modules import sell_rules
        from modules.portfolio import load_positions
        pos = load_positions(config)
        if not pos.empty:
            ev = sell_rules.evaluate(pos, config)
            if not ev.empty:
                # 同じ銘柄が2口座にあるときは重いほうを採用する
                ev = ev.sort_values("重さ", key=lambda s: s.map(
                    {k: i for i, k in enumerate(sell_rules.SEVERITY_ORDER)}))
                sell_verdict = {r["ticker"]: r for _, r in
                                ev.drop_duplicates("ticker").iterrows()}
    except Exception as exc:                                  # データ不足でも監視は動かす
        print(f"⚠️  売り判定を取り込めませんでした（続行）: {exc}")

    bg = config.get("buy_priority", {})
    min_health = float(bg.get("min_health", 40))
    max_trap = float(bg.get("max_trap_penalty", 10))

    now = datetime.now().isoformat(timespec="seconds")
    rows = []

    def add(ticker, severity, kind, message):
        rows.append({"detected_at": now, "ticker": ticker, "severity": severity,
                     "kind": kind, "message": message, "resolved": 0})

    for ticker, r in df.iterrows():
        name = r.get("name") or ticker

        broken = sell_verdict.get(ticker)

        # 1) 減配 — 配当株にとって最も重い異変
        # 売り判定の「直近の配当年度で減配した」と同じ出来事なので、1本にまとめる。
        # 別々に出すと同じ銘柄で2つ警報が鳴り、件数だけが膨らむ。
        cut_fired = False
        if pd.notna(r.get("growth_latest")) and r["growth_latest"] < -0.001:
            extra = ""
            if broken is not None and "減配" in broken["理由"]:
                extra = "　／　" + broken["根拠"].splitlines()[0].lstrip("- ")
            add(ticker, "high", "dividend_cut",
                f"{name}: 配当が前年比 {r['growth_latest']:.1%}"
                f"（{r['dps_prev']:.1f} → {r['dps_latest']:.1f}円）{extra}")
            cut_fired = True

        # 2) 配当が壊れた・原資が傷んだ
        # 判定は売り側と同じ modules/sell_rules を使う。ここで自前の基準を持つと
        # 「監視は買い場、整理は売れ」という矛盾が起きる（実測で6銘柄あった）。
        # 「採用基準を外れた」はそれだけでは異変ではない。新規で買わない理由であって、
        # 持っている株を手放す理由ではないから（実測で160件中99件がこれだった）。
        lead = broken["理由"].splitlines()[0].lstrip("- ") if broken is not None else ""
        if cut_fired and lead.startswith("直近の配当年度で減配した"):
            pass                       # 上の減配アラートと同じ出来事なので出さない
        elif broken is not None and broken["重さ"] == "売却を検討":
            add(ticker, "high", "dividend_broken",
                f"{name}: {broken['理由'].splitlines()[0].lstrip('・')}"
                f"（{broken['根拠'].splitlines()[0].lstrip('・')}）")
        elif broken is not None and broken["重さ"] == "監視を強める":
            add(ticker, "medium", "dividend_weak",
                f"{name}: {broken['理由'].splitlines()[0].lstrip('・')}"
                f"（{broken['根拠'].splitlines()[0].lstrip('・')}）")
        elif broken is not None and broken["重さ"] == "利確を検討":
            add(ticker, "low", "take_profit",
                f"{name}: {broken['理由'].splitlines()[0].lstrip('・')}"
                f"（{broken['根拠'].splitlines()[0].lstrip('・')}）")

        # 3) トラップ検出
        penalty = r.get("trap_penalty") or 0
        # 買う側の足切りと同じ値を使う。ここだけ別の数字にすると、
        # 「買う画面では弾くのに監視では警告しない」帯ができる。
        if penalty >= max_trap:
            add(ticker, "medium", "trap", f"{name}: 高配当トラップの兆候（減点 {penalty:.0f}）")

        # 4) 指値到達
        # **買い場と呼んでよいのは、買い側の足切りを通ったものだけ。**
        # 利回りが目標に届いた理由が「減配を市場が織り込んだ株価下落」なら、
        # それは買い場ではなく警告。買う画面と同じ基準（配当継続スコアと
        # トラップ減点）で振り分ける。
        ty = r.get("target_yield")
        if pd.notna(ty) and ty > 0 and pd.notna(r.get("dividend_yield")) \
                and r["dividend_yield"] >= ty:
            price = r["last_close"]
            head = (f"{name}: 目標利回り {ty:.2%} に到達"
                    f"（現在 {r['dividend_yield']:.2%} / ¥{price:,.0f}）")
            health = r.get("health")
            blocked = []
            if pd.isna(health) or float(health) < min_health:
                blocked.append(f"配当継続スコア {('—' if pd.isna(health) else f'{health:.0f}')}"
                               f"（{min_health:.0f}未満）")
            if penalty >= max_trap:
                blocked.append(f"高配当トラップの減点 {penalty:.0f}")
            if broken is not None and broken["重さ"] in ("売却を検討", "監視を強める"):
                blocked.append(broken["理由"].splitlines()[0].lstrip("- "))
            if blocked:
                add(ticker, "medium", "yield_trap",
                    head + " — ただし買ってはいけない：" + " / ".join(blocked))
            else:
                add(ticker, "low", "target_reached", head)

        # 5) 据え置きが続いている＝増配ストーリーが止まった
        if pd.notna(r.get("streak")) and r["streak"] == 0 and (r.get("streak_no_cut") or 0) >= 2:
            add(ticker, "low", "dividend_flat", f"{name}: 増配が止まっている（据え置き）")

    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    out["_order"] = out["severity"].map(SEVERITY_ORDER)
    return out.sort_values("_order").drop(columns="_order").reset_index(drop=True)


def save_alerts(alerts: pd.DataFrame) -> pd.DataFrame:
    """まだ記録していないアラートだけを登録し、その新規ぶんを返す。

    「基準を外れている」「増配が止まっている」は毎日変わらない状態であって、
    毎朝の出来事ではない。日付だけで重複を見ると、同じ 137 件が毎日届いて
    通知が読まれなくなる。銘柄×種別で過去すべてと突き合わせ、
    本当に新しいものだけを通知に回す。
    """
    if alerts is None or alerts.empty:
        return pd.DataFrame()
    with connect() as conn:
        existing = {
            (r["ticker"], r["kind"])
            for r in conn.execute(
                "SELECT ticker, kind FROM alerts WHERE resolved = 0"
            ).fetchall()
        }
        new = alerts[~alerts.apply(lambda r: (r["ticker"], r["kind"]) in existing, axis=1)]
        for _, r in new.iterrows():
            conn.execute(
                "INSERT INTO alerts (detected_at, ticker, severity, kind, message, resolved) "
                "VALUES (?, ?, ?, ?, ?, 0)",
                (r["detected_at"], r["ticker"], r["severity"], r["kind"], r["message"]),
            )
    return new


def format_for_push(alerts: pd.DataFrame, limit: int = 10) -> str:
    """スマホ通知用の本文。長いと読まれないので重要なものだけ。"""
    if alerts is None or alerts.empty:
        return "異変なし"
    icons = {"high": "🔴", "medium": "🟡", "low": "🔵"}
    lines = [f"{icons.get(r['severity'], '・')} {r['message']}"
             for _, r in alerts.head(limit).iterrows()]
    if len(alerts) > limit:
        lines.append(f"…ほか {len(alerts) - limit} 件")
    return "\n".join(lines)
