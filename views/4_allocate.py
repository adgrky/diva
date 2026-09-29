"""資金投入 — 「トレード利益を移した。どこに入れるか」に答える。

3つに分かれる。
    相場と現金  いくら入れて、いくら残すか（ヘムの3本目の柱）
    買う        どの銘柄に配るか・買ったら記録する
    整理する    どれから手をつけるか・判断を覚える・売ったら記録する

【発掘スコア順に買わない】
発掘スコアは「市場にまだ気づかれていないか」を含むので、見つける順位であって
買う順位ではない。買い付け優先度は √(割安度 × 配当継続) で別に出す。
"""
from __future__ import annotations

import json
from datetime import date

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from modules import market, review as R, sell_rules as SR
from modules.allocator import allocate, buy_gate, buy_priority, month_gaps
from modules.format import to_pct, yen, yen_short
from modules.portfolio import dividend_calendar, dividend_cash, freed_cash
from modules.store import connect, read_df, run_batch
from modules.ui import flash, show_flash, get_config, get_positions, get_scores, no_data_guard

st.title("💰 資金投入")
show_flash()

config = get_config()
scores = get_scores()
if no_data_guard(scores):
    st.stop()

positions = get_positions(config)
calendar = dividend_calendar(positions)
gaps = month_gaps(positions, calendar)
total_eval = positions["eval_value"].sum() if not positions.empty else 0.0


@st.cache_data(ttl=3600, show_spinner="相場の水準を確認中...")
def _market() -> dict:
    pbr = market.fetch_nikkei_pbr()
    price = market.fetch_nikkei_price()
    snap = market.snapshot(price, pbr)
    if snap:
        market.record_snapshot(snap, pbr)
    return snap or market.load_snapshot()


snap = _market()
regime_name, deploy_ratio, regime_note = market.regime(snap, config)

def _record_sell(account: str, ticker: str, name: str, shares: float,
                 price: float, when, tax: float = 0.0) -> None:
    """売却を記録して、保有の株数をその場で減らす。全部売れば保有から消す。

    税額も一緒に残す。「整理して生まれたお金をそのまま次の買い付けに回す」ときに、
    税引後の手取りでないと金額が合わないため。
    """
    _record_sells([dict(account=account, ticker=ticker, name=name, shares=shares,
                        price=price, when=when, tax=tax)])


def _record_sells(items: list[dict]) -> None:
    """複数の売却を1回で記録する。一覧からまとめて売ったとき用。

    クラウドDBだと1件ごとに往復が発生するので、保有株数の読み出しも書き込みも1回にまとめる。
    """
    with connect() as conn:
        held = {(r["account"], r["ticker"]): float(r["shares"]) for r in
                conn.execute("SELECT account, ticker, shares FROM holdings").fetchall()}
        stmts = []
        for it in items:
            key = (it["account"], it["ticker"])
            left = held.get(key, 0.0) - float(it["shares"])
            held[key] = left
            stmts.append((
                "INSERT INTO transactions (date, account, ticker, name, type, shares, price, "
                "fee, memo, tax) VALUES (?, ?, ?, ?, 'sell', ?, ?, 0, ?, ?)",
                (it["when"].isoformat(), it["account"], it["ticker"], it["name"],
                 float(it["shares"]), float(it["price"]), "整理タブから記録",
                 float(it.get("tax", 0.0)))))
            if left <= 0.5:
                stmts += [("DELETE FROM holdings WHERE account=? AND ticker=?", key),
                          ("DELETE FROM holding_review WHERE account=? AND ticker=?", key)]
            else:
                stmts.append(("UPDATE holdings SET shares=?, updated_at=datetime('now') "
                              "WHERE account=? AND ticker=?", (left, *key)))
        # 記録と株数の更新は片方だけ残ると二重売却になるので、まとめて通す。
        run_batch(conn, stmts)

@st.cache_data(ttl=600, show_spinner=False)
def _evaluate(pos: pd.DataFrame, _config: dict) -> pd.DataFrame:
    """整理候補の判定。画面を触るたびに数秒かかっていたので覚えておく。
    記録したときは st.cache_data.clear() で消えるので、判断の反映は遅れない。"""
    ev = SR.evaluate(pos, _config)
    return R.attach(ev) if not ev.empty else ev


tab_market, tab_buy, tab_sell = st.tabs(["📉 相場と現金", "🛒 買う", "🧹 整理する（売る）"])

# ═══════════════════════════════════════════════ 相場と現金
with tab_market:
    st.caption("ヘムの3本目の柱「暴落時の買い向かい」。"
               "現金を残しておき、下がるほど多く入れるための材料です。")
    # 4つ横に並ぶので、単位は見出しに逃がして値は数字だけにする。
    # 「63,484 円」のまま入れると、窓が狭いときに「63,48…」と切れる。
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("日経平均（円）",
              f"{snap.get('nikkei', 0):,.0f}" if snap.get("nikkei") else "—",
              help="終値。週足の最新")
    c2.metric("日経平均PBR", f"{snap.get('pbr', 0):.2f}" if snap.get("pbr") else "—",
              help="株価が1株純資産の何倍か。リーマン・ショックのときは0.8倍まで下がりました")
    c3.metric("10年での位置",
              f"上位 {1 - snap['pct_10y']:.0%}" if snap.get("pct_10y") is not None else "—",
              help="日経平均が、過去10年の分布の中でどのあたりか")
    c4.metric("200日線との差",
              f"{snap['vs_ma200']:+.1%}" if snap.get("vs_ma200") is not None else "—",
              help="200日移動平均からどれだけ離れているか")

    color = {"総力戦": "error", "大人買い": "error", "買い増し": "success",
             "平常": "info", "やや高い": "warning", "高い": "warning"}.get(regime_name, "info")
    getattr(st, color)(f"### 【{regime_name}】入金額の {deploy_ratio:.0%} を投入\n\n{regime_note}")

    st.markdown("#### 大人買いライン")
    reserve_pct = st.slider("総資産のうち、暴落用に取っておく現金の割合", 0, 60,
                            int(config["market"]["cash_reserve"] * 100), 5, format="%d%%") / 100
    reserve = total_eval * reserve_pct / (1 - reserve_pct) if reserve_pct < 1 else 0
    st.caption(f"いまの評価額 {yen_short(total_eval)} に対して、"
               f"**{yen_short(reserve)}** を現金で持っておく計算です。")

    ladder = market.buy_ladder(snap, config, reserve)
    if ladder["日経平均の水準"].notna().any():
        show = ladder.copy()
        show["日経平均の水準"] = show["日経平均の水準"].round(0)
        show["いまからの下落率"] = to_pct(show["いまからの下落率"]).round(1)
        show["投入する割合"] = to_pct(show["投入する割合"]).round(0)
        show["投入額"] = show["投入額"].round(0)
        show["到達済み"] = show["到達済み"].map({True: "✅ 到達", False: ""})
        st.dataframe(show, hide_index=True, width="stretch", column_config={
            "日経平均PBR": st.column_config.NumberColumn(format="%.2f 倍"),
            "日経平均の水準": st.column_config.NumberColumn(
                format="%d 円", help="この日経平均になったら、その行の金額を投入する"),
            "いまからの下落率": st.column_config.NumberColumn(format="%.1f%%"),
            "投入する割合": st.column_config.NumberColumn(
                format="%.0f%%", help="取ってある現金のうち、その水準で入れる割合"),
            "投入額": st.column_config.NumberColumn(format="¥%d"),
        })
        st.caption("PBRは「株価 ÷ 1株純資産」なので、1株純資産が変わらないとすれば "
                   "**目標PBRに対応する日経平均 = いまの日経平均 × 目標PBR ÷ いまのPBR** "
                   "で逆算できます。証券会社のアプリでこの価格にアラートを仕掛けておくと、"
                   "暴落のときに迷わず動けます。")

        fig = go.Figure()
        lv = ladder.dropna(subset=["日経平均の水準"])
        fig.add_bar(x=[f"PBR {p:.2f}" for p in lv["日経平均PBR"]],
                    y=lv["日経平均の水準"], marker_color="#4C8BF5",
                    text=[f"{v:,.0f}円" for v in lv["日経平均の水準"]], textposition="outside")
        if snap.get("nikkei"):
            fig.add_hline(y=snap["nikkei"], line_dash="dash", line_color="#E45756",
                          annotation_text=f"いま {snap['nikkei']:,.0f}円")
        fig.update_layout(height=320, margin=dict(l=10, r=10, t=30, b=10),
                          yaxis_title="日経平均（円）")
        st.plotly_chart(fig, width="stretch")
    else:
        st.warning("日経平均PBRが取れていないため、大人買いラインを出せません。")

    hist = read_df("SELECT * FROM market_history ORDER BY date")
    if len(hist) > 3:
        with st.expander(f"貯まっている相場の記録（{len(hist)} 日ぶん）"):
            st.caption("日経のサイトは当月ぶんしか返さないため、"
                       "スキャンのたびに少しずつ貯めて自前の履歴を作っています。")
            st.dataframe(hist.tail(60), hide_index=True, width="stretch")

# ═══════════════════════════════════════════════ 買う
with tab_buy:
    # ── 整理して生まれたお金を、そのまま次の買い付けに回せるようにする ──
    # 売ったあとに金額を手で足し算して打ち直すのでは、この使い方は続かない。
    freed = freed_cash(30)
    # 受け取った配当も買い付けの原資。再投資しないと「育つ」にならない。
    dcash = dividend_cash(90)
    if freed["残り"] > 0:
        st.success(
            f"🔁 **整理して生まれた余力が {yen(freed['残り'])} 残っています**"
            f"（直近30日に {freed['売却件数']} 件売った手取り {yen(freed['手取り'])} − "
            f"その後 {freed['買い付け件数']} 件買った {yen(freed['使った額'])}）。\n\n"
            "下の「整理で生まれた余力」に入れてあります。買い付けを記録すれば自動で減ります。")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        cash_in = st.number_input("入金額（円）", 0, 100_000_000, 500_000, 50_000, format="%d",
                                  help="トレード利益から新しく入れるお金")
        swap_in = st.number_input(
            "整理で生まれた余力（円）", 0, 100_000_000, int(freed["残り"]), 10_000, format="%d",
            help="持ち株を整理して作ったお金。**相場の水準による絞り込みはかけません**。"
                 "同じ市場の中で乗り換えるだけで、新しく市場に踏み込むわけではないからです")
        div_in = st.number_input(
            "受け取った配当（円）", 0, 100_000_000, int(dcash["受取額"]), 1_000, format="%d",
            help=f"直近90日に受け取った配当（税引後）の合計。"
                 f"いま {dcash['件数']} 件・{yen(dcash['受取額'])} 記録されています。"
                 "**再投資しないと配当は育ちません。**"
                 "相場の水準による絞り込みはかけません")
        account = st.radio("入れる口座", ["specific", "nisa"], index=0,
                           format_func=lambda a: "特定口座（課税 20.315%）" if a == "specific"
                           else "NISA（非課税）",
                           help="NISAは配当が非課税。同じ配当でも手取りが約2割変わります")
    with c2:
        use_regime = st.checkbox(
            f"相場の水準に合わせる（いま【{regime_name}】＝{deploy_ratio:.0%}）", value=True,
            help="高い局面では入金を使い切らず、暴落に備えて現金に積みます")
        max_names = st.number_input(
            "買う銘柄数", 1, 40, 12,
            help="ヘムは360〜400銘柄に超分散。機械的な基準で選び監視はアプリがやるので、"
                 "絞る理由は薄いです")
    with c3:
        min_yield = st.number_input("最低利回り（%）", 0.0, 8.0, 3.0, 0.25) / 100
        odd_lot = st.checkbox(
            "単元未満株（1株から）で買う", value=True,
            help="SBIのS株、楽天のかぶミニ、マネックスのワン株など。"
                 "毎月の入金で単元（10万〜70万円）を買えることは稀なので、こちらが既定です")
        even = st.checkbox("選んだ銘柄数で均等に配る", value=True,
                           help="外すと、1銘柄あたりの上限を自分で決められます")
    with c4:
        scope = st.radio("対象", ["未保有のみ", "保有の買い増しも含む"], index=1)
        require_target = st.checkbox("目標利回りに届いている銘柄だけ", value=False,
                                     help="届いていないなら待つ、というヘムの型")
        if even:
            per_cap = 1.0 / float(max_names)
            single_lot = True
            st.caption(f"1銘柄あたり **{per_cap:.1%}**（＝入金 ÷ {int(max_names)} 銘柄）")
        else:
            per_cap = st.slider("1銘柄あたりの上限（投入額に対する比率）", 5, 100, 15, 5,
                                format="%d%%") / 100
            single_lot = st.checkbox(
                "1単元だけは上限を超えて買う", value=True,
                help="単元で買うときだけ効きます。外すと、単元の値段が上限を超える銘柄は"
                     "すべて落ちます（実測で投入枠30万円のうち9.9万円しか配れませんでした）")
    lot = 1 if odd_lot else 100

    cap_g = config["buy_priority"]
    require_capacity = st.checkbox(
        f"増配余力の条件も満たすものだけにする"
        f"（配当性向 {cap_g.get('capacity_max_payout', 0.5):.0%}以下 ／ "
        f"FCFが配当の {cap_g.get('capacity_min_fcf_cover', 2.0):.0f}倍以上）",
        value=False,
        help="2026-09-15 の検証（EDINET の FY2022 起点・1,211銘柄・3年）で、"
             "高利回りにこれを重ねると **減配率が 8.8% → 5.4%** に下がり、"
             "リターンも +80.4% → +90.6% と上がりました。"
             "ただし**コホートが1つ・3年**しかなく、"
             "かけると候補の利回りが下がります（実測 4.17% → 3.82%）。"
             "だから既定では掛けていません。")

    # 相場の水準による絞り込みは「新しく入れるお金」にだけかける。
    # 整理して作ったお金は、同じ市場の中で乗り換えるだけなので絞らない。
    # ここを一緒くたに絞ると、整理するたびに市場から少しずつ降りることになる。
    cash_from_deposit = cash_in * (deploy_ratio if use_regime else 1.0)
    # 整理で作ったお金と受け取った配当は、すでに市場の中にあるお金なので絞らない
    cash = cash_from_deposit + swap_in + div_in
    if use_regime and deploy_ratio < 1.0 and cash_in > 0:
        msg = (f"【{regime_name}】のため、入金 {yen(cash_in)} のうち "
               f"**{yen(cash_from_deposit)}** を投入し、"
               f"**{yen(cash_in - cash_from_deposit)}** は暴落用の現金に積みます。")
        extra = []
        if swap_in > 0:
            extra.append(f"整理で作った {yen(swap_in)}")
        if div_in > 0:
            extra.append(f"受け取った配当 {yen(div_in)}")
        if extra:
            msg += ("\n\n" + "・".join(extra) + " は**そのまま全額**入れます"
                    "（すでに市場の中にあるお金なので絞りません）。"
                    f"合わせて **{yen(cash)}** を配ります。")
        st.info(msg)
    elif swap_in > 0 or div_in > 0:
        parts = [f"入金 {yen(cash_from_deposit)}"]
        if swap_in > 0:
            parts.append(f"整理で作った {yen(swap_in)}")
        if div_in > 0:
            parts.append(f"受け取った配当 {yen(div_in)}")
        st.info(" ＋ ".join(parts) + f" ＝ **{yen(cash)}** を配ります。")

    held = set(positions["ticker"]) if not positions.empty else set()
    cand = scores[scores["gate_passed"] == 1].copy().reset_index(drop=True)
    raw = pd.DataFrame([json.loads(x)["raw"] for x in cand["detail_json"]])
    # 増配余力の足切りに使うので、配当性向とFCFカバー率も持たせる
    cand = pd.concat([cand, raw[["dividend_yield", "yield_percentile", "streak",
                                 "payout_months", "payout_ratio", "fcf_cover"]]], axis=1)
    if scope == "未保有のみ":
        cand = cand[~cand["ticker"].isin(held)]
    cand = cand[cand["dividend_yield"].fillna(0) >= min_yield]
    targets = read_df("SELECT ticker, target_yield FROM watchlist "
                      "UNION SELECT ticker, target_yield FROM holdings").groupby("ticker").first()
    cand = cand.join(targets, on="ticker")

    scored = buy_priority(cand, positions, config, gaps)
    _, gate_reasons = buy_gate(scored, config, require_capacity=require_capacity)
    n_cut = int((gate_reasons != "").sum())
    if n_cut:
        st.caption(f"候補 {len(scored)} 件のうち **{n_cut} 件を足切り**しました"
                   f"（{gate_reasons[gate_reasons != ''].value_counts().to_dict()}）。"
                   "安くても、配当が危ない銘柄と高配当トラップの判定が出ている銘柄には入れません。")

    plan = allocate(cash, cand, positions, config, max_names=int(max_names),
                    per_name_cap_pct=float(per_cap), require_below_target=require_target,
                    month_gap=gaps, allow_single_lot=single_lot, lot=lot,
                    require_capacity=require_capacity)

    st.divider()
    if plan.empty:
        st.warning("条件に合う配分先がありません。銘柄数の上限を増やすか、利回りの下限を下げてください。")
    else:
        tax = config["portfolio"]["tax_rate_nisa"] if account == "nisa" \
            else config["portfolio"]["tax_rate_specific"]
        total_in, total_div = plan["投入額"].sum(), plan["年間配当"].sum()
        reserve_part = cash_in - cash_from_deposit   # 相場の水準に応じて意図的に残したぶん
        leftover = cash - total_in             # 単元に丸めきれず余ったぶん
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("投入額", yen(total_in))
        c2.metric("増える配当／年", yen(total_div), help="税引前")
        c3.metric("税引後", yen(total_div * (1 - tax)),
                  help="NISAは非課税、特定口座は20.315%を引いています")
        c4.metric("残る現金", yen(cash_in + swap_in + div_in - total_in),
                  help="暴落用に取っておくぶんと、単元に丸めて余ったぶんの合計")

        # 内訳を必ず出す。合計だけ出していたときは「入金50万・投入枠30万」と
        # 言いながら「現金に積む 40万」と表示され、何が起きているか分からなかった。
        unit_word = "1株の端数" if lot == 1 else "単元（100株）に丸めた端数"
        bits = [f"入金 {yen(cash_in)}"]
        if swap_in > 0:
            bits.append(f"整理で作った {yen(swap_in)}")
        if div_in > 0:
            bits.append(f"受け取った配当 {yen(div_in)}")
        src = "**" + " ＋ ".join(bits) + "**"
        st.caption(
            f"{src}　＝　投入 {yen(total_in)}　＋　"
            f"暴落用に取っておく {yen(reserve_part)}（【{regime_name}】のため入金の "
            f"{1 - deploy_ratio:.0%}）　＋　{unit_word} {yen(leftover)}"
            if use_regime and deploy_ratio < 1.0 and cash_in > 0 else
            f"{src}　＝　投入 {yen(total_in)}　＋　{unit_word} {yen(leftover)}")
        if leftover > 0:
            cheapest = (cand["last_close"].min() * lot) if not cand.empty else 0
            if lot == 1:
                st.caption(f"余った {yen(leftover)} は、**1株の端数**です"
                           f"（いちばん安い候補でも1株 {yen(cheapest)}）。"
                           "次の入金に足してください。")
            else:
                st.caption(
                    f"余った {yen(leftover)} は、**100株単位で買えるものが無くなった**ぶんです"
                    f"（いちばん安い候補でも1単元 {yen(cheapest)}）。"
                    "「単元未満株（1株から）で買う」に入れると、ほぼ全額を配れます。")
        n_relaxed = int(plan.get("上限を超えて1単元", pd.Series(dtype=bool)).sum())
        if n_relaxed:
            st.caption(f"うち **{n_relaxed} 銘柄**は、1単元の値段が上限を超えていますが"
                       "「1単元だけは上限を超えて買う」に従って入れています。")
        fee = float(plan["概算手数料"].sum())
        if fee > 0:
            st.caption(
                f"⚠️ **単元未満株は約定価格にスプレッドが乗ります**（0.2〜0.5%程度。"
                f"SBIのS株は買い無料・売り0.55%、楽天のかぶミニは0.22%）。"
                f"今回の概算は **{yen(fee)}**（0.22%で計算）。"
                f"増える年間配当 {yen(total_div)} に対して {fee / total_div:.1%} なので、"
                + ("**1年目で十分に回収できます**。" if fee < total_div * 0.3 else
                   "**配り先を絞って1銘柄あたりを大きくしたほうがよいかもしれません**。"))
        n_sec = plan["業種"].nunique()
        if n_sec < max(3, len(plan) // 2):
            top_sec = plan.groupby("業種")["投入額"].sum().idxmax()
            st.warning(f"今回の配分は **{n_sec} 業種**に偏っています（最大は {top_sec}）。"
                       "業種の上限はポートフォリオ全体に対してかかるので、"
                       "1回の入金では効きません。気になるなら銘柄数の上限を上げてください。")

        show = plan[["コード", "銘柄名", "業種", "配当月", "株価", "株数", "投入額", "利回り",
                     "年間配当", "買い付け優先度", "割安度", "自己利回り順位", "絶対利回り順位",
                     "継続", "補完度", "発掘スコア", "上限を超えて1単元",
                     "概算手数料"]].copy()
        show = show.rename(columns={"上限を超えて1単元": "上限超"})
        if lot == 1:
            show = show.drop(columns=["上限超"])
        else:
            show = show.drop(columns=["概算手数料"])
        show["利回り"] = to_pct(show["利回り"])
        show["税引後配当"] = (show["年間配当"] * (1 - tax)).round(0)
        for c in ("買い付け優先度", "割安度", "自己利回り順位", "絶対利回り順位",
                  "継続", "補完度", "発掘スコア"):
            show[c] = show[c].round(0)
        st.dataframe(show, hide_index=True, width="stretch", column_config={
            "株価": st.column_config.NumberColumn(format="¥%d"),
            "投入額": st.column_config.NumberColumn(format="¥%d"),
            "年間配当": st.column_config.NumberColumn(format="¥%d"),
            "税引後配当": st.column_config.NumberColumn(format="¥%d"),
            "利回り": st.column_config.NumberColumn(format="%.2f%%"),
            "買い付け優先度": st.column_config.ProgressColumn(
                format="%.0f", min_value=0, max_value=100,
                help="√(割安度 × 配当継続) × 補完度係数 − トラップ減点。発掘スコアとは別物"),
            "割安度": st.column_config.NumberColumn(
                format="%.0f", help="自己利回り順位60% + 候補内の絶対利回り順位40%"),
            "継続": st.column_config.NumberColumn(format="%.0f", help="配当継続スコア"),
            "補完度": st.column_config.NumberColumn(
                format="%.0f", help="空いている業種・空いている配当月を埋めるか（±10%のタイブレーク）"),
            "発掘スコア": st.column_config.NumberColumn(
                format="%.0f", help="参考表示。買う順位の決定には使っていません"),
            "上限超": st.column_config.CheckboxColumn(
                help="1単元の値段が「1銘柄あたりの上限」を超えているが、"
                     "1単元だけ入れた銘柄"),
            "概算手数料": st.column_config.NumberColumn(
                format="¥%d", help="単元未満株のスプレッド見込み（0.22%で計算）"),
        })
        st.caption(("株数は**1株単位**です（単元未満株）。" if lot == 1 else
                    "株数は単元（100株）に丸めています。")
                   + "**買い付け優先度の順に**、業種の上限と1銘柄あたりの上限を守りながら"
                     "埋めています。")

        with st.expander("なぜこの式なのか"):
            st.markdown("""
#### 発掘スコア順に買わない理由

発掘スコアには「見過ごされ度」が重み1.2で入っています。
**市場に気づかれていないことは、見つける理由にはなっても、いま買う理由になりません。**

#### 足し算ではなく掛け算にした理由

    買い付け優先度 = √(割安度 × 配当継続) × 補完度係数 − トラップ減点

加重和だと「片方が極端に高ければ、もう片方が低くても選ばれて」しまいます。
実測すると『割安度80超・配当継続35未満』の銘柄が13件あり、加重和では69点で
上位に入っていました（小松マテーレ 割安95・継続34 など）。

配当投資では「安いが配当が危ない」も「安全だが高い」も買いたくない。
**両方そろって初めて買う**という性質を、そのまま式にしています。

| | 加重和 | 幾何平均 |
|---|---|---|
| 割安100・継続100 | 100 | **100** |
| 割安100・継続25 | 68 | **50** |
| 割安60・継続60 | 60 | **60** |

#### 割安度に2つ混ぜている理由

「安い」には2つの意味があります。

- **自己利回り順位**（60%）… その銘柄自身の過去と比べて安いか＝**買う時期**の判断
- **絶対利回り順位**（40%）… 候補の中で利回りが高いか＝**いま受け取れる額**

自己利回り順位だけだと「いつも1.5%の銘柄が2.0%」を最高評価してしまい、
インカムとしては物足りない銘柄が上位に来ます。

#### 足切り

順位を付ける前に外します。**安いからといって、配当が危ない銘柄や
高配当トラップの判定が出ている銘柄に資金を入れる理由はありません。**

- 配当継続スコア 40未満
- 高配当トラップの減点 10以上
""")

        st.markdown("#### 買ったら記録する")
        st.caption("実際に発注して約定したら、ここで記録すると保有に反映されます。")
        edited = st.data_editor(
            pd.DataFrame({"記録する": True, "コード": plan["コード"].values,
                          "銘柄名": plan["銘柄名"].values, "株数": plan["株数"].values,
                          "約定単価": plan["株価"].round(1).values}),
            hide_index=True, width="stretch", key="buy_editor",
            column_config={
                "記録する": st.column_config.CheckboxColumn(),
                "コード": st.column_config.TextColumn(disabled=True),
                "銘柄名": st.column_config.TextColumn(disabled=True),
                "株数": st.column_config.NumberColumn(min_value=0, step=1),
                "約定単価": st.column_config.NumberColumn(format="¥%.1f", min_value=0.0)})
        buy_date = st.date_input("約定日", value=date.today(), key="buy_date")
        if st.button("この内容で買ったことにする", type="primary"):
            rows = edited[edited["記録する"] & (edited["株数"] > 0)]
            if rows.empty:
                st.warning("記録する行がありません")
            else:
                with connect() as conn:
                    for r in rows.itertuples(index=False):
                        ticker = f"{r.コード}.T"
                        shares, price = float(r.株数), float(r.約定単価)
                        conn.execute(
                            "INSERT INTO transactions (date, account, ticker, name, type, "
                            "shares, price, fee, memo) VALUES (?, ?, ?, ?, 'buy', ?, ?, 0, ?)",
                            (buy_date.isoformat(), account, ticker, r.銘柄名, shares, price,
                             "資金投入画面から記録"))
                        cur = conn.execute(
                            "SELECT shares, avg_cost FROM holdings WHERE account=? AND ticker=?",
                            (account, ticker)).fetchone()
                        if cur and cur["shares"]:
                            ns = float(cur["shares"]) + shares
                            nc = (float(cur["shares"]) * float(cur["avg_cost"] or 0)
                                  + shares * price) / ns
                        else:
                            ns, nc = shares, price
                        conn.execute(
                            "INSERT OR REPLACE INTO holdings (account, ticker, name, shares, "
                            "avg_cost, target_yield, bottom_yield, updated_at) VALUES (?, ?, ?, ?, ?,"
                            " COALESCE((SELECT target_yield FROM holdings WHERE account=? AND ticker=?), 0.047),"
                            " (SELECT bottom_yield FROM holdings WHERE account=? AND ticker=?),"
                            " datetime('now'))",
                            (account, ticker, r.銘柄名, ns, nc, account, ticker, account, ticker))
                flash(f"✅ {len(rows)} 銘柄・{yen((rows['株数'] * rows['約定単価']).sum())} "
                      "を買ったこととして記録しました。ポートフォリオに反映されています。")
                st.cache_data.clear()
                st.rerun()

# ═══════════════════════════════════════════════ 整理する
with tab_sell:
    ev = _evaluate(positions, config)
    counts = SR.summary(ev)

    st.markdown("#### いま何が起きているか")
    cs = st.columns(5)
    _HELP = {
        "売却を検討": "配当そのものが壊れた。減配が実際に起きた／利益を超えて配当している等、**事実**にだけ反応します",
        "監視を強める": "配当はまだ出ているが、原資が傷んでいる。次の決算で確かめるもの",
        "利確を検討": "壊れたのではなく育ちきった。ヘムの「上がりすぎたら売る」",
        "手入れ": "売る理由ではありません。額が小さい・口座が分かれているなどの片付け",
        "判定待ち": "財務や配当性向がまだ取れていないもの。判定していません",
    }
    for col, k in zip(cs, SR.SEVERITY_ORDER):
        # 5つ横に並ぶので見出しは短く。正式名は説明に入れる
        col.metric(f"{SR.SEVERITY[k][0]} {SR.SHORT[k]}", counts[k],
                   help=f"**{k}** — {_HELP[k]}")

    with st.expander("なぜこの基準なのか（買う基準をそのまま使わない理由）"):
        st.markdown("""
**買わない理由と、売る理由は別物です。**

買うときは選択肢が3,700あるので、少しでも引っかかれば見送ってよい。
けれど持っている株を売るのは、**税金・スプレッド・再投資先の確保**という
コストを払う行為です。同じ基準を当てると、こうなります。

以前はここに **保有114銘柄のうち84銘柄** が「重大」として並んでいました。中身は：

| 出ていたもの | 実際は |
|---|---|
| 三菱UFJ・群馬銀行「営業CFがマイナス」 | **銀行は貸出を増やすと営業CFがマイナスになる**。正常 |
| トヨタ「FCFで配当を賄えていない」 | 金融事業込みのFCF。売り理由ではない |
| MS&AD「配当性向が計算できない」 | 単なる**データ欠損**。連続増配17年 |
| 黒田グループ「上場5年未満」 | 新規上場は売り理由ではない |
| イントラスト「売買代金が薄い」 | 流動性は**買う前に確かめること** |

いまは **起きた事実にだけ反応**します。

- 🔴 **売却を検討** … 実際に減配した／10年で3回以上減配している／利益を超えて配当している
- 🟡 **監視を強める** … 営業CFが2期連続マイナス／2期連続赤字／減益が続き配当性向も高い／借金が重く自己資本が薄い／配当が3年すえ置きで性向が高い
- 🟢 **利確を検討** … 自己利回り順位が15%以下（この会社としては歴史的に高い株価）で含み益30%超／PER15倍超で利回りが目標の6割を下回った
- ⚪️ **手入れ** … 額が小さい・口座が分かれている。**売る理由ではありません**

**金融（銀行・保険・証券）には、営業CF・FCF・有利子負債の基準を当てていません。**
業態として営業CFがマイナスになるのが普通だからです。
また、上場年数・売買代金・時価総額は新規買いの条件であって、売り理由にはしていません。
""")

    st.divider()
    c1, c2, c3 = st.columns([1.6, 1, 1])
    with c1:
        pick_sev = st.multiselect(
            "見るもの", SR.SEVERITY_ORDER,
            default=[k for k in ("売却を検討", "監視を強める", "利確を検討") if counts[k]],
            format_func=lambda k: f"{SR.SEVERITY[k][0]} {k}（{counts[k]}）")
    with c2:
        hide_decided = st.checkbox("判断済みを隠す", value=True,
                                   help="「持ち続ける」「様子見」と決めたものを隠します")
    with c3:
        view_mode = st.radio("表示", ["1銘柄ずつ", "一覧"], horizontal=True,
                             help="1銘柄ずつなら理由が全部読めます")

    shown = ev[ev["重さ"].isin(pick_sev)] if pick_sev else ev
    if hide_decided:
        shown = shown[shown["判断"] == ""]
    shown = shown.reset_index(drop=True)

    tax_rate = config["portfolio"]["tax_rate_specific"]

    def _tax(row, shares, price) -> float:
        if row["account"] == "nisa":
            return 0.0
        gain = (price - float(row["avg_cost"] or 0)) * shares
        return max(gain, 0.0) * tax_rate

    # ────────────────────────────── 1銘柄ずつ
    if shown.empty:
        st.success("この条件に該当する、まだ判断していない保有はありません。")
    elif view_mode == "1銘柄ずつ":
        st.caption(f"**{len(shown)} 件**あります。1件ずつ決めると、次からは隠れます。")
        i = st.number_input("何件目", 1, len(shown), 1, 1,
                            key="sell_cursor") - 1
        r = shown.iloc[int(i)]
        st.progress((int(i) + 1) / len(shown), f"{int(i) + 1} / {len(shown)} 件目")

        acc_ja = "NISA" if r["account"] == "nisa" else "特定"
        st.markdown(f"### {SR.SEVERITY[r['重さ']][0]} {r['code']}　{r['name']}")
        st.caption(f"{r['sector33']}　／　{acc_ja}口座　／　"
                   f"整理の優先度 {r['整理の優先度']:.0f}")

        m = st.columns(5)
        m[0].metric("株数", f"{r['shares']:,.0f} 株")
        m[1].metric("株価", yen(r["last_close"]))
        # 5列に詰めると「¥190,…」と切れるので、金額は短縮表記にして正確な額は説明へ
        m[2].metric("評価額", yen_short(r["eval_value"]), help=yen(r["eval_value"]))
        m[3].metric("損益", f"{r['pnl_pct']:+.1%}" if pd.notna(r["pnl_pct"]) else "—",
                    yen_short((r["last_close"] - (r["avg_cost"] or 0)) * r["shares"]),
                    help=f"取得単価 {yen(r['avg_cost'])} → 株価 {yen(r['last_close'])}")
        m[4].metric("配当継続", f"{r['health']:.0f}" if pd.notna(r["health"]) else "—",
                    help="配当継続スコア。配当が続くか・増えるかだけを見た点数。50が真ん中")

        box = {"売却を検討": st.error, "監視を強める": st.warning,
               "利確を検討": st.success, "手入れ": st.info,
               "判定待ち": st.info}[r["重さ"]]
        box(f"**{r['重さ']}**\n\n{r['理由']}")
        c_a, c_b = st.columns(2)
        with c_a:
            st.markdown("**根拠になっている数字**")
            st.markdown(r["根拠"])
        with c_b:
            st.markdown("**どうするか**")
            st.markdown(r["やること"])
        if r["判断"]:
            st.caption(f"→ すでに「{r['判断']}」と記録しています（{r['判断日']}）")

        st.markdown("#### この銘柄をどうするか")
        st.caption("「売った」を選んで記録すると、**ポートフォリオの株数がその場で減ります**"
                   "（全部売れば保有から消えます）。買い足したときは 🛒 買う で記録してください。")
        d1, d2, d3, d4 = st.columns([1.2, 1, 1, 1])
        with d1:
            act = st.radio("判断", ["まだ決めない", "持ち続ける", "様子見", "売った"],
                           key=f"act_{r['account']}_{r['ticker']}", horizontal=False)
        with d2:
            sh = st.number_input("売った株数", 0.0, float(r["shares"]),
                                 float(r["shares"]), 1.0,
                                 key=f"sh_{r['account']}_{r['ticker']}")
        with d3:
            pr = st.number_input("約定単価（円）", 0.0, 10_000_000.0,
                                 float(r["last_close"] or 0), 0.5,
                                 key=f"pr_{r['account']}_{r['ticker']}")
        with d4:
            sd = st.date_input("約定日", value=date.today(),
                               key=f"sd_{r['account']}_{r['ticker']}")

        if act == "売った" and sh > 0:
            tax = _tax(r, sh, pr)
            e1, e2, e3 = st.columns(3)
            e1.metric("売却代金", yen(sh * pr))
            e2.metric("売却益の税金", yen(tax),
                      help="特定口座 20.315%。含み損なら0円。NISAは非課税")
            e3.metric("手取り", yen(sh * pr - tax))
            lost = sh * (r.get("dps_latest") or 0)
            if lost:
                st.caption(f"この売却で**年間 {yen(lost)} の配当がなくなります**。"
                           "同じ配当を取り戻すには、🛒 買う で入れ直す必要があります。")

        if st.button("この判断を記録する", type="primary", key=f"go_{r['account']}_{r['ticker']}"):
            if act == "まだ決めない":
                st.warning("判断が選ばれていません")
            elif act in ("持ち続ける", "様子見"):
                R.save(r["account"], r["ticker"], "keep" if act == "持ち続ける" else "watch")
                flash(f"✅ {r['name']} を「{act}」として記録しました。次から隠れます。")
                st.cache_data.clear()
                st.rerun()
            elif sh <= 0:
                st.warning("売った株数を入れてください")
            else:
                _record_sell(r["account"], r["ticker"], r["name"], sh, pr, sd,
                             tax=_tax(r, sh, pr))
                flash(f"✅ {r['name']} を {sh:,.0f} 株 {yen(sh * pr)} で売却として記録しました。"
                      "ポートフォリオの株数に反映されています。")
                st.cache_data.clear()
                st.rerun()

    # ────────────────────────────── 一覧
    # 表を触るたびにページ全体を計算し直すと遅いので、この表だけ動かす
    @st.fragment
    def _bulk_list(shown: pd.DataFrame) -> None:
        st.caption("理由の全文は「1銘柄ずつ」で読めます。"
                   "「判断」の列で選んでいけば、下のボタンでまとめて記録できます。"
                   "売った株数と約定単価は、はじめは全株・直近の株価が入っています。")
        v = pd.DataFrame({
            # スマホの幅でも「銘柄名」と「今回の判断」がはみ出さない並び
            "銘柄名": shown["name"].values,
            "今回の判断": "—",
            "口座": shown["account"].map({"specific": "特定", "nisa": "NISA"}).values,
            "売った株数": shown["shares"].astype(float).values,
            "約定単価": shown["last_close"].fillna(0).astype(float).values,
            "保有株数": shown["shares"].values,
            "": shown["印"].values,
            "重さ": shown["重さ"].values,
            "優先度": shown["整理の優先度"].values,
            "コード": shown["code"].values,
            "業種": shown["sector33"].values,
            "評価額": shown["eval_value"].values,
            "損益率": to_pct(shown["pnl_pct"]).values,
            "配当継続": shown["health"].round(0).values,
            "理由": shown["理由"].str.replace("\n- ", " ／ ").str.lstrip("- ").values,
            "判断": shown["判断"].values,
        })
        # 行の並びが変わると、前の編集が別の銘柄に当たってしまう。
        # 表示中の銘柄の組み合わせごとに別の表として扱う。
        ed_key = "bulk_sell_" + str(abs(hash(tuple(shown["account"] + shown["ticker"]))))
        locked = [c for c in v.columns if c not in ("今回の判断", "売った株数", "約定単価")]
        ed = st.data_editor(v, hide_index=True, width="stretch", height=520, key=ed_key,
                            disabled=locked, column_config={
            "銘柄名": st.column_config.TextColumn(pinned=True),
            "今回の判断": st.column_config.SelectboxColumn(
                "判断", options=["—", "売った", "持ち続ける", "様子見"], required=True, width="small"),
            "売った株数": st.column_config.NumberColumn(min_value=0.0, step=1.0, format="%.0f"),
            "約定単価": st.column_config.NumberColumn(min_value=0.0, step=0.5, format="¥%.1f"),
            "優先度": st.column_config.ProgressColumn(format="%.0f", min_value=0, max_value=100),
            "保有株数": st.column_config.NumberColumn(format="%.0f"),
            "評価額": st.column_config.NumberColumn(format="¥%d"),
            "損益率": st.column_config.NumberColumn(format="%.1f%%"),
            "配当継続": st.column_config.NumberColumn(format="%.0f"),
            "理由": st.column_config.TextColumn(width="large"),
            "判断": st.column_config.TextColumn("記録済み"),
        })

        sold = ed[(ed["今回の判断"] == "売った") & (ed["売った株数"] > 0)]
        kept = ed[ed["今回の判断"].isin(["持ち続ける", "様子見"])]
        over = sold[sold["売った株数"] > sold["保有株数"] + 1e-9]
        if not sold.empty:
            sd = st.date_input("約定日（まとめて記録する売却すべてに使います）",
                               value=date.today(), key="bulk_sell_date")
            taxes = [_tax(shown.iloc[i], r["売った株数"], r["約定単価"])
                     for i, r in sold.iterrows()]
            gross = float((sold["売った株数"] * sold["約定単価"]).sum())
            lost = float(sum(r["売った株数"] * (shown.iloc[i].get("dps_latest") or 0)
                             for i, r in sold.iterrows()))
            e1, e2, e3, e4 = st.columns(4)
            e1.metric("売却代金", yen(gross))
            e2.metric("売却益の税金", yen(sum(taxes)),
                      help="特定口座 20.315%。含み損なら0円。NISAは非課税")
            e3.metric("手取り", yen(gross - sum(taxes)))
            e4.metric("なくなる年間配当", yen(lost))
        if not over.empty:
            st.error("保有株数より多く売ったことになっている行があります: "
                     + "、".join(over["銘柄名"].astype(str)))

        n = len(sold) + len(kept)
        if st.button(f"この {n} 件をまとめて記録する", type="primary",
                     disabled=n == 0 or not over.empty, key="bulk_sell_go"):
            if not sold.empty:
                _record_sells([dict(
                    account=shown.iloc[i]["account"], ticker=shown.iloc[i]["ticker"],
                    name=shown.iloc[i]["name"], shares=r["売った株数"], price=r["約定単価"],
                    when=sd, tax=_tax(shown.iloc[i], r["売った株数"], r["約定単価"]))
                    for i, r in sold.iterrows()])
            for i, r in kept.iterrows():
                R.save(shown.iloc[i]["account"], shown.iloc[i]["ticker"],
                       "keep" if r["今回の判断"] == "持ち続ける" else "watch")
            st.session_state.pop(ed_key, None)
            flash(f"✅ 売却 {len(sold)} 件（{yen(gross) if not sold.empty else '¥0'}）"
                  f"・持ち続ける／様子見 {len(kept)} 件を記録しました。"
                  "ポートフォリオの株数に反映されています。")
            st.cache_data.clear()
            st.rerun(scope="app")

        st.download_button("この一覧をCSVで保存", v.to_csv(index=False).encode("utf-8-sig"),
                           f"整理候補_{date.today():%Y%m%d}.csv", "text/csv")

    if not shown.empty and view_mode == "一覧":
        _bulk_list(shown)

    st.divider()
    with st.expander("記録した判断を取り消す"):
        dec = R.load()
        if dec.empty:
            st.caption("記録された判断はまだありません")
        else:
            names = read_df("SELECT ticker, name FROM universe").set_index("ticker")["name"]
            dec["銘柄名"] = dec["ticker"].map(names)
            dec["判断"] = dec["decision"].map(R.DECISIONS)
            st.dataframe(dec[["account", "ticker", "銘柄名", "判断", "decided_at"]].rename(
                columns={"account": "口座", "ticker": "銘柄", "decided_at": "判断日"}),
                hide_index=True, width="stretch")
            undo = st.selectbox("取り消す銘柄", [""] + [
                f"{r.account}／{r.ticker} {r.銘柄名}" for r in dec.itertuples(index=False)])
            if undo and st.button("この判断を取り消す"):
                acc, rest = undo.split("／")
                R.clear(acc, rest.split(" ")[0])
                flash("✅ 判断を取り消しました。また整理の一覧に出てきます。")
                st.cache_data.clear()
                st.rerun()

    with st.expander("売買の記録"):
        tx = read_df("SELECT date, account, ticker, name, type, shares, price, memo "
                     "FROM transactions ORDER BY date DESC, id DESC LIMIT 200")
        if tx.empty:
            st.caption("まだ記録がありません")
        else:
            tx["account"] = tx["account"].map({"specific": "特定", "nisa": "NISA"}).fillna(tx["account"])
            tx["type"] = tx["type"].map({"buy": "買い", "sell": "売り", "dividend": "配当"}).fillna(tx["type"])
            # 配当は「株数1 × 単価＝受取額」で保存している。表示では実際の株数に戻す
            is_div = tx["type"] == "配当"
            amt = tx["shares"] * tx["price"]
            div_sh = tx["memo"].str.extract(r"／([\d,.]+)株")[0].str.replace(",", "")
            div_sh = pd.to_numeric(div_sh, errors="coerce")
            tx.loc[is_div, "shares"] = div_sh[is_div]
            tx.loc[is_div, "price"] = (amt / div_sh)[is_div]
            tx["金額"] = amt.round(0)
            st.dataframe(tx.rename(columns={
                "date": "日付", "account": "口座", "ticker": "銘柄", "name": "銘柄名",
                "type": "種別", "shares": "株数", "price": "単価", "memo": "メモ"}),
                hide_index=True, width="stretch", column_config={
                    "単価": st.column_config.NumberColumn(format="¥%.1f"),
                    "金額": st.column_config.NumberColumn(format="¥%d")})
