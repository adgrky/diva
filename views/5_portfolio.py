"""ポートフォリオ — 発掘した銘柄を入れる枠の現状。"""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from datetime import date

from modules.format import pct, to_pct, yen, yen_short
from modules import income_risk as IR
from modules.portfolio import (dividend_calendar, dividend_staircase, dividends_received,
                               expected_dividends, portfolio_vs_benchmark, record_equity,
                               sector_exposure)
from modules.store import connect, read_df
from modules.ui import flash, show_flash, get_config, get_next_ex_dates, get_positions

st.title("📊 ポートフォリオ")
show_flash()

config = get_config()
positions = get_positions(config)
if positions.empty:
    st.warning("保有データがありません。\n\n```bash\nuv run python scripts/import_legacy.py\n```")
    st.stop()

total_eval = positions["eval_value"].sum()
total_cost = positions["cost_value"].sum()
annual_div = positions["annual_dividend"].sum()
annual_div_at = positions["annual_dividend_after_tax"].sum()

# 指標は3列ずつ2段に分ける。5列に詰めると値が「1,27…」と切れて読めなくなる。
c1, c2, c3 = st.columns(3)
c1.metric("評価額", yen_short(total_eval))
c2.metric("損益", yen_short(total_eval - total_cost),
          f"{(total_eval / total_cost - 1):.1%}" if total_cost else None)
c3.metric("取得額", yen_short(total_cost))

c4, c5, c6 = st.columns(3)
c4.metric("年間配当（税引前）", yen_short(annual_div))
c5.metric("年間配当（税引後）", yen_short(annual_div_at),
          help="NISAは非課税、特定口座は20.315%を引いています")
c6.metric("YOC", pct(annual_div / total_cost) if total_cost else "—",
          help="取得額に対する利回り。いまの年間配当 ÷ 買ったときの金額。"
               "増配で育つとここが上がります")

n_ok = int((positions["gate_passed"] == 1).sum())
st.caption(f"保有 {len(positions)} 銘柄 ／ 1銘柄あたり平均 {yen_short(total_eval / len(positions))}"
           f" ／ 採用基準を満たすもの {n_ok} 銘柄")

# その日の評価額と年間配当を1日1行だけ残す。
# インカム投資の目的は「配当が育つこと」なので、推移が見えないと成果が分からない。
record_equity(positions, config)

tab1, tab7, tab9, tab8, tab2, tab3, tab6, tab4, tab5, tab10 = st.tabs(
    ["保有一覧", "配当の強さ", "増配の実績", "NISA枠", "業種の配分", "配当月",
     "次の権利落ち日", "配当の受取記録", "推移", "ほっといた場合との差"])

with tab1:
    edit_mode = st.toggle("株数と取得単価を直す", value=False,
                          help="売買を証券会社の画面で済ませたあと、ここで合わせられます。"
                               "株数を0にすると保有から消えます")
    src = positions.copy()
    src["判定"] = src.apply(
        lambda r: "◯ 基準を満たす" if r.get("gate_passed") == 1
        else (str(r.get("gate_reason")) if isinstance(r.get("gate_reason"), str)
              and r.get("gate_reason") else "—"), axis=1)

    if edit_mode:
        st.caption("**直した行は下のボタンを押すまで保存されません。** "
                   "株数を0にすると、その保有は消えます。"
                   "売買として記録を残したいときは 💰 資金投入 の「買う」「整理する」を使ってください"
                   "（そちらは取引の履歴も残ります）。")
        base = src[["account", "ticker", "code", "name", "shares", "avg_cost",
                    "last_close", "target_yield"]].copy()
        base["口座"] = base["account"].map({"specific": "特定", "nisa": "NISA"})
        ed = st.data_editor(
            base.rename(columns={"code": "コード", "name": "銘柄名", "shares": "株数",
                                 "avg_cost": "取得単価", "last_close": "株価",
                                 "target_yield": "目標利回り"})[
                ["コード", "銘柄名", "口座", "株数", "取得単価", "株価", "目標利回り"]],
            hide_index=True, width="stretch", height=560, key="holdings_editor",
            column_config={
                "コード": st.column_config.TextColumn(disabled=True),
                "銘柄名": st.column_config.TextColumn(disabled=True),
                "口座": st.column_config.TextColumn(disabled=True),
                "株価": st.column_config.NumberColumn(format="¥%d", disabled=True),
                "株数": st.column_config.NumberColumn(min_value=0.0, step=1.0),
                "取得単価": st.column_config.NumberColumn(format="¥%.1f", min_value=0.0),
                "目標利回り": st.column_config.NumberColumn(
                    format="%.3f", min_value=0.0, max_value=0.2,
                    help="0.047 なら 4.7%。指値の計算に使います"),
            })
        if st.button("この内容で保存する", type="primary"):
            n_upd = n_del = 0
            with connect() as conn:
                for orig, new_row in zip(base.itertuples(index=False),
                                         ed.itertuples(index=False)):
                    sh, cost, tgt = float(new_row.株数), float(new_row.取得単価), float(new_row.目標利回り)
                    if (abs(sh - float(orig.shares)) < 1e-9
                            and abs(cost - float(orig.avg_cost or 0)) < 1e-9
                            and abs(tgt - float(orig.target_yield or 0)) < 1e-9):
                        continue
                    if sh <= 0:
                        conn.execute("DELETE FROM holdings WHERE account=? AND ticker=?",
                                     (orig.account, orig.ticker))
                        n_del += 1
                    else:
                        conn.execute(
                            "UPDATE holdings SET shares=?, avg_cost=?, target_yield=?, "
                            "updated_at=datetime('now') WHERE account=? AND ticker=?",
                            (sh, cost, tgt, orig.account, orig.ticker))
                        n_upd += 1
            if n_upd or n_del:
                flash(f"✅ {n_upd} 銘柄を直し、{n_del} 銘柄を保有から外しました")
                st.cache_data.clear()
                st.rerun()
            else:
                st.info("変わったところがありません")
    else:
        view = src[[
            "code", "name", "sector33", "account", "shares", "avg_cost", "last_close",
            "eval_value", "pnl_pct", "dps_latest", "annual_dividend", "current_yield",
            "yoc", "streak", "health", "判定",
        ]].rename(columns={
            "code": "コード", "name": "銘柄名", "sector33": "業種", "account": "口座",
            "shares": "株数", "avg_cost": "取得単価", "last_close": "株価",
            "eval_value": "評価額", "pnl_pct": "損益率", "dps_latest": "DPS",
            "annual_dividend": "年間配当", "current_yield": "現在利回り",
            "yoc": "YOC", "streak": "連続増配", "health": "配当継続スコア",
        }).sort_values("評価額", ascending=False)
        for col in ("損益率", "現在利回り", "YOC"):
            view[col] = to_pct(view[col])
        view["口座"] = view["口座"].map({"specific": "特定", "nisa": "NISA"}).fillna(view["口座"])

        st.dataframe(view, hide_index=True, width="stretch", height=560, column_config={
            "取得単価": st.column_config.NumberColumn(format="¥%.1f"),
            "株価": st.column_config.NumberColumn(format="¥%d"),
            "評価額": st.column_config.NumberColumn(format="¥%d"),
            "年間配当": st.column_config.NumberColumn(format="¥%d"),
            "DPS": st.column_config.NumberColumn(format="¥%.1f"),
            "損益率": st.column_config.NumberColumn(format="%.1f%%"),
            "現在利回り": st.column_config.NumberColumn(format="%.2f%%"),
            "YOC": st.column_config.NumberColumn(
                format="%.2f%%", help="いまの年間配当 ÷ 買ったときの金額"),
            "連続増配": st.column_config.NumberColumn(format="%d 年"),
            "配当継続スコア": st.column_config.NumberColumn(
                format="%.0f", help="配当が続くか・増えるかだけを見た点数。"
                                    "空欄は採用基準を外れている銘柄（理由は「判定」列）"),
            "判定": st.column_config.TextColumn(width="large"),
        })
        st.download_button("保有一覧をCSVで保存",
                           view.to_csv(index=False).encode("utf-8-sig"),
                           f"保有一覧_{date.today():%Y%m%d}.csv", "text/csv")

with tab7:
    st.caption("**評価額の分散と、配当の分散は別物です。** "
               "検証で「最良の選び方でも5年で35〜40%が減配する」と出ました。"
               "選別では防ぎきれないので、**起きたときにどれだけ減るか**を見ます。")

    conc = IR.concentration(positions)
    if not conc:
        st.info("配当のデータがありません。")
    else:
        # 4つ横に並べると見出しが切れるので、3つに絞って言葉も短くする
        eff = conc["実質の分散銘柄数"]
        c1, c2, c3 = st.columns(3)
        c1.metric("配当が出ている銘柄",
                  f"{conc['配当が出ている銘柄数']} / {conc['銘柄数']}",
                  help="保有していても無配の銘柄は、インカムには効いていません")
        c2.metric("実質の分散", f"{eff:.1f} 銘柄",
                  f"見かけより {conc['配当が出ている銘柄数'] - eff:.0f} 少ない",
                  delta_color="off",
                  help="1 ÷ Σ(配当の構成比²)。50銘柄あっても1銘柄に半分が寄っていれば、"
                       "実質は数銘柄ぶんにしかなりません")
        c3.metric("上位5／上位10",
                  f"{conc['上位5の割合']:.0%} ／ {conc['上位10の割合']:.0%}",
                  help="年間配当のうち、上位5銘柄／上位10銘柄から出ている割合")

        ratio = eff / max(conc["配当が出ている銘柄数"], 1)
        if ratio < 0.35:
            st.warning(f"**配当が偏っています。** {conc['配当が出ている銘柄数']} 銘柄から"
                       f"配当が出ていますが、偏りを考えると **実質 {eff:.0f} 銘柄ぶん** "
                       "の分散しかありません。上位の銘柄が減配すると、インカム全体が大きく揺れます。")
        else:
            st.success(f"配当の分散は効いています（実質 {eff:.0f} 銘柄ぶん）。")

        src = conc["内訳"]
        names = positions.groupby("ticker")["name"].first()
        codes = positions.groupby("ticker")["code"].first()
        topn = src.head(15)
        fig = go.Figure(go.Bar(
            x=[f"{codes.get(t,'')} {str(names.get(t,t))[:8]}" for t in topn.index],
            y=topn.values, marker_color="#4C8BF5",
            text=[f"{v/conc['年間配当']:.1%}" for v in topn.values], textposition="outside"))
        fig.update_layout(height=320, yaxis_title="年間配当（円）",
                          title="配当の出どころ（上位15銘柄）",
                          margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig, width="stretch")

    st.divider()
    st.markdown("#### 不況に弱い業種に、配当がどれだけ寄っているか")
    cyc = IR.cyclical_exposure(positions, config)
    if cyc:
        c1, c2, c3 = st.columns(3)
        c1.metric("配当に占める割合", f"{cyc['配当に占める割合']:.1%}",
                  f"上限 {cyc['上限']:.0%}",
                  delta_color="inverse" if cyc["超過"] > 0 else "off",
                  help="不況のたびに配当を削る業種が、年間配当の何割を出しているか")
        c2.metric("評価額に占める割合", f"{cyc['評価額に占める割合']:.1%}",
                  help="高利回りな業種なので、評価額よりも配当のほうが偏りやすい")
        c3.metric("該当する銘柄", f"{cyc['銘柄数']} 銘柄")
        if cyc["超過"] > 0:
            st.warning(
                f"**上限 {cyc['上限']:.0%} を {cyc['超過']:.1%} 超えています"
                f"（年間 {yen(cyc['超過している配当額'])} ぶん）。**\n\n"
                "整理するときは、ここから減らすと不況耐性が上がります。"
                "避けてもリターンは落ちません（検証では高利回り候補の中で "
                "+52.1% → +54.3%、減配率 58.9% → 54.8%）。")
        else:
            st.success(f"上限 {cyc['上限']:.0%} の内側です。")
        st.dataframe(
            pd.DataFrame({"業種": cyc["業種別"].index,
                          "配当に占める割合": to_pct(cyc["業種別"].values).round(1)}),
            hide_index=True, width="stretch",
            column_config={"配当に占める割合":
                           st.column_config.NumberColumn(format="%.1f%%")})
        with st.expander("なぜこの9業種なのか"):
            st.markdown(
                "業種は**平時の減配は当てられません**（業種ごとの減配率の順位は、"
                "2014年と2020年で相関 0.08 しかない）。\n\n"
                "ところが**不況のときの落ち込みは、業種で強く決まっていました**。\n\n"
                "| 比べた2つの不況 | 業種ごとの落ち込みの順位相関 |\n|---|---|\n"
                "| リーマン ↔ コロナ | **+0.622** |\n"
                "| リーマン ↔ 東日本大震災 | +0.455 |\n\n"
                "毎回おなじ顔ぶれがやられます。上の9業種は**リーマンの実績だけ**で選び、"
                "コロナで答え合わせしたものです（40銘柄・等金額で、"
                "この9業種を25%までに抑えると配当の落ち込みが 12.1% → 11.6%、"
                "15%までなら 11.0%、ゼロなら 9.5%）。\n\n"
                "**1業種あたりの上限は効きません。** ランダムに20銘柄選ぶと、"
                "もう平均12.7業種にまたがっているからです（上限をかけても "
                "23.0% → 23.7% と変わらなかった）。効くのは**この9業種の合計**のほうです。")
    st.divider()
    st.markdown("#### 不況が来たら、配当はどれだけ落ちるか")
    st.caption("いま持っている銘柄が、**過去の不況で実際にどう振る舞ったか**を当てはめています。"
               "当時まだ配当が無かった銘柄は分かりません。"
               "分かるぶんで割合を出し、残りも同じように振る舞うとみなしています。")

    stress = IR.stress_test(positions)
    if stress.empty:
        st.info("過去の配当データが足りず、判定できません。")
    else:
        cols = st.columns(len(stress))
        for col, r in zip(cols, stress.itertuples()):
            col.metric(r.できごと, f"−{r.減った割合:.0%}",
                       f"残り {yen_short(r.残る年間配当)}", delta_color="off",
                       help=f"{r.年度}年度／{r.説明}／"
                            f"調べられたのは {r.調べられた銘柄} 銘柄"
                            f"（配当の {r.調べられた配当の割合:.0%}）")
        worst = stress.loc[stress["減った割合"].idxmax()]
        st.error(f"**{worst['できごと']}級が来ると、年間配当は "
                 f"{yen(conc['年間配当'])} → {yen(worst['残る年間配当'])} まで落ちます"
                 f"（−{worst['減った割合']:.0%}）。**\n\n"
                 "これは予想ではなく、**いま持っている銘柄が当時そう動いた**という記録です。"
                 "現金を残しておくのは、この局面で買い向かうためです。")

        show = pd.DataFrame({
            "できごと": stress["できごと"], "年度": stress["年度"],
            "調べられた銘柄": stress["調べられた銘柄"], "うち減配": stress["うち減配"],
            "減る割合": to_pct(stress["減った割合"]).round(1),
            "減る額": stress["いまの配当に当てはめた減少額"].round(0),
            "残る年間配当": stress["残る年間配当"].round(0),
            "説明": stress["説明"],
        })
        st.dataframe(show, hide_index=True, width="stretch", column_config={
            "減る割合": st.column_config.NumberColumn(format="%.1f%%"),
            "減る額": st.column_config.NumberColumn(format="¥%d"),
            "残る年間配当": st.column_config.NumberColumn(format="¥%d"),
            "説明": st.column_config.TextColumn(width="large")})

        pick = st.selectbox("どの銘柄が配当を減らしたか見る",
                            stress["できごと"].tolist(),
                            index=int(stress["減った割合"].values.argmax()))
        w = IR.worst_contributors(positions, pick, top=12)
        if w.empty:
            st.caption("この局面で減配した保有はありませんでした。")
        else:
            wv = w.copy()
            wv["減配率"] = to_pct(wv["減配率"]).round(0)
            st.dataframe(wv, hide_index=True, width="stretch", column_config={
                "当時の1株配当": st.column_config.NumberColumn(format="¥%.1f"),
                "その後の1株配当": st.column_config.NumberColumn(format="¥%.1f"),
                "減配率": st.column_config.NumberColumn(format="%.0f%%"),
                "いまの年間配当": st.column_config.NumberColumn(format="¥%d"),
                "失う配当": st.column_config.NumberColumn(format="¥%d")})

    st.divider()
    st.markdown("#### 業種の偏り — 評価額で見るか、配当で見るか")
    st.caption("業種の上限（20%）は**評価額**にかかっています。"
               "でもインカムを守るなら、見るべきは**配当の構成比**です。")
    ibs = IR.income_by_sector(positions)
    if not ibs.empty:
        top = ibs.head(12)
        fig = go.Figure()
        fig.add_bar(name="評価額の構成比", x=top["sector33"],
                    y=to_pct(top["評価額の構成比"]), marker_color="#9AA5B1")
        fig.add_bar(name="配当の構成比", x=top["sector33"],
                    y=to_pct(top["配当の構成比"]), marker_color="#4C8BF5")
        fig.add_hline(y=config["portfolio"]["max_sector_weight"] * 100,
                      line_dash="dash", annotation_text="上限20%")
        fig.update_layout(height=340, barmode="group", yaxis_title="構成比（%）",
                          margin=dict(l=10, r=10, t=10, b=10),
                          legend=dict(orientation="h", y=1.12))
        st.plotly_chart(fig, width="stretch")

        gap = ibs[ibs["配当の構成比"] > config["portfolio"]["max_sector_weight"]]
        if not gap.empty:
            st.warning("**配当の構成比が20%を超えている業種：**" + "、".join(
                f"{r['sector33']}（配当 {r['配当の構成比']:.0%} ／ 評価額 "
                f"{r['評価額の構成比']:.0%}）" for _, r in gap.iterrows())
                + "\n\nこの業種がまとめて減配すると、インカムが大きく揺れます。")

        iv = pd.DataFrame({
            "業種": ibs["sector33"], "銘柄数": ibs["銘柄数"],
            "評価額": ibs["評価額"].round(0), "年間配当": ibs["年間配当"].round(0),
            "評価額の構成比": to_pct(ibs["評価額の構成比"]).round(1),
            "配当の構成比": to_pct(ibs["配当の構成比"]).round(1),
            "差": to_pct(ibs["差"]).round(1)})
        st.dataframe(iv, hide_index=True, width="stretch", height=300, column_config={
            "評価額": st.column_config.NumberColumn(format="¥%d"),
            "年間配当": st.column_config.NumberColumn(format="¥%d"),
            "評価額の構成比": st.column_config.NumberColumn(format="%.1f%%"),
            "配当の構成比": st.column_config.NumberColumn(
                format="%.1f%%", help="インカムを守るなら、こちらを見ます"),
            "差": st.column_config.NumberColumn(
                format="%+.1f%%", help="プラスなら「評価額の割に配当を多く出している業種」")})

with tab8:
    from modules import nisa as NI

    st.markdown("#### NISA枠をどう使うか")
    st.caption("売らずに持ち続けるなら、NISAの値打ちは **配当への課税 20.315% が消えること** に"
               "ほぼ尽きます（値上がり益の非課税は、売らないかぎり実現しないので）。"
               "だから **枠には利回りが高い銘柄を入れる**。それだけです。")

    c1, c2 = st.columns([1, 1])
    monthly = c1.number_input("毎月いくら入金しますか", min_value=0, max_value=1_000_000,
                              value=100_000, step=10_000, format="%d",
                              help="新規買いを先に枠に入れ、余った年間枠で移し替えを考えます")
    horizon = c2.slider("何年持ち続ける前提か", 5, 40, 20, 5,
                        help="移し替えが得かどうかは、持つ年数で決まります")

    r = NI.plan(positions, float(monthly), int(horizon))
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("成長投資枠の残り", yen_short(r["残りの枠"]),
              f"生涯 {yen_short(NI.GROWTH_LIFETIME)} − 簿価 {yen_short(r['使った枠'])}",
              delta_color="off", help="枠は時価ではなく**簿価（取得額）**で数えます")
    m2.metric("埋めきるまで", f"最短 {r['最短何年']:.0f} 年",
              f"年 {yen_short(r['年間の枠'])} まで", delta_color="off")
    m3.metric("特定口座で払っている税", f"年 {yen_short(r['特定の税'])}",
              f"{horizon}年で {yen_short(r['特定の税'] * horizon)}", delta_color="off",
              help="いまの配当が変わらないとした場合。増配すればもっと増えます")
    m4.metric("利回り NISA / 特定", f"{r['NISAの利回り']:.2%} / {r['特定の利回り']:.2%}",
              help="NISAのほうが高ければ、枠の使い方としては正しい向きです")

    st.markdown("##### 今年の枠の使いみち")
    st.info(f"年間枠 {yen(NI.GROWTH_ANNUAL)} のうち、"
            f"**新規買いに {yen(r['今年の新規買いに使う枠'])}**、"
            f"**移し替えに使えるのが {yen(r['今年の移し替えに使える枠'])}**。\n\n"
            "移し替えより新規買いが優先です。新規買いには含み益への課税が無く、"
            "コストがゼロだからです。**2つは同じ年240万円を奪い合います。**")

    moves = r["移す銘柄"]
    if r["今年の移し替えに使える枠"] <= 0:
        st.success("入金だけで年間枠を使い切ります。移し替えを考える必要はありません。")
    elif moves.empty:
        st.info(f"{horizon}年持つ前提では、移して得になる銘柄がありません。")
    else:
        st.markdown(f"##### 今年移すなら、この順番（{horizon}年で {yen(r['移して得られる額'])} の得）")
        show = moves[["code", "name_jpx", "使う枠", "含み益", "annual_dividend",
                      "回収年数", f"{horizon}年の得"]].copy()
        show["回収年数"] = show["回収年数"].map(
            lambda v: "即得" if v <= 0 else (f"{v:.1f}年" if pd.notna(v) and v < 1e6 else "—"))
        if "一部だけ" in moves.columns and moves["一部だけ"].any():
            show["銘柄"] = [f"{n}{'（一部）' if part else ''}"
                          for n, part in zip(moves["name_jpx"], moves["一部だけ"])]
            show = show.drop(columns=["name_jpx"])
            show = show[["code", "銘柄", "使う枠", "含み益", "annual_dividend",
                         "回収年数", f"{horizon}年の得"]]
        show.columns = ["コード", "銘柄", "使う枠", "含み益", "年間配当",
                        "回収年数", f"{horizon}年の得"]
        st.dataframe(show, hide_index=True, width="stretch", column_config={
            "使う枠": st.column_config.NumberColumn(format="¥%d"),
            "含み益": st.column_config.NumberColumn(format="¥%d"),
            "年間配当": st.column_config.NumberColumn(format="¥%d"),
            f"{horizon}年の得": st.column_config.NumberColumn(format="¥%d"),
        })
        st.caption(f"全部移せるだけの枠があれば {yen(r['全部移した場合の得'])} の得になります"
                   f"（年間枠の制限で、今年はここまで）。"
                   "**枠は簿価で数えるので、売った枠が戻るのは翌年1月1日です。**")

    with st.expander("移すかどうかは、たった1つの式で決まる"):
        st.markdown(f"""
移すには一度売るので、**含み益に 20.315% の税がかかります**。
一方で、移したあとは毎年の配当が非課税になります。

| | |
|---|---|
| 移すコスト（1回だけ） | 含み益 × 20.315% |
| 移す便益（毎年） | 年間配当 × 20.315% |
| **回収にかかる年数** | **含み益 ÷ 年間配当** |

つまり **含み益が年間配当の何倍あるか** だけで決まります。
{horizon}年持つなら、回収年数が {horizon} 年より短い銘柄は移す価値があります。

**含み損の銘柄は話が逆**です。売れば損失が確定して同じ年の利益と通算できるので、
コストがマイナスになります。ただし **同じ年に他で利益を確定している場合だけ** なので、
この表では戻りを数えていません（数えなくても、コストがゼロなのでどのみち移す価値があります）。

**制度の確認（2026-09-16 に裏取り）**
成長投資枠 年240万円・生涯1,200万円（NISA全体1,800万円のうち）。
売却すると翌年1月1日に簿価ぶんの生涯枠が復活しますが、年間投資枠は復活しません。
""")

    mis = NI.misplaced(positions)
    if mis:
        lo, hi = mis["NISAにある低利回り"], mis["特定にある高利回り"]
        st.markdown("##### 置き場所が逆になっている銘柄")
        st.caption(f"保有全体の利回りの真ん中は {mis['境目の利回り']:.2%}。"
                   "それより低い銘柄がNISAにあり、高い銘柄が特定口座にあるなら、"
                   "枠の使い方としては損をしています。")
        k1, k2 = st.columns(2)
        k1.metric("NISAにある低利回り", f"{len(lo)} 銘柄", yen_short(lo["eval_value"].sum()),
                  delta_color="off")
        k2.metric("特定にある高利回り", f"{len(hi)} 銘柄", yen_short(hi["eval_value"].sum()),
                  delta_color="off")
        st.caption("ただし入れ替えるには NISA 側を売る必要があり、**枠が戻るのは翌年**。"
                   "年間枠240万円は復活しません。急ぐ理由はありません。"
                   "まず新規買いで枠を埋めるのが先です。")


with tab2:
    exposure = sector_exposure(positions, config)
    over = exposure[exposure["上限超過"]]
    if not over.empty:
        st.warning("上限（構成比20%）を超えている業種：" + "、".join(
            f"{r['sector33']}（{r['構成比']:.1%}）" for _, r in over.iterrows()))
    ex = exposure.rename(columns={"sector33": "業種"}).copy()
    fig = go.Figure(go.Bar(
        x=to_pct(ex["構成比"]), y=ex["業種"], orientation="h",
        marker_color=["#E45756" if o else "#4C8BF5" for o in ex["上限超過"]]))
    fig.add_vline(x=config["portfolio"]["max_sector_weight"] * 100,
                  line_dash="dash", annotation_text="上限")
    fig.update_layout(height=max(400, 22 * len(ex)), xaxis_title="構成比（%）",
                      margin=dict(l=10, r=10, t=10, b=10),
                      yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig, width="stretch")
    ex["構成比"] = to_pct(ex["構成比"])
    st.dataframe(ex, hide_index=True, width="stretch", column_config={
        "評価額": st.column_config.NumberColumn(format="¥%d"),
        "年間配当": st.column_config.NumberColumn(format="¥%d"),
        "構成比": st.column_config.NumberColumn(format="%.1f%%"),
        "上限までの余裕": st.column_config.NumberColumn(format="¥%d"),
    })

with tab3:
    cal = dividend_calendar(positions)
    if cal.empty:
        st.caption("配当履歴が不足しています")
    else:
        fig = go.Figure(go.Bar(x=cal["label"], y=cal["amount"], marker_color="#4C8BF5"))
        fig.update_layout(height=340, yaxis_title="受取見込み（円）",
                          margin=dict(l=10, r=10, t=10, b=10),
                          xaxis=dict(categoryorder="array",
                                     categoryarray=[f"{m}月" for m in range(1, 13)]))
        st.plotly_chart(fig, width="stretch")
        peak = cal.loc[cal["amount"].idxmax()]
        share = peak["amount"] / cal["amount"].sum() if cal["amount"].sum() else 0
        empty_months = [r["label"] for _, r in cal.iterrows() if r["amount"] == 0]
        st.caption(f"直近1年の権利落ち月ベース。最も多いのは **{peak['label']}**"
                   f"（{yen(peak['amount'])}・全体の {share:.0%}）。")
        if empty_months:
            st.info(f"**受け取りがゼロの月：{'、'.join(empty_months)}**　\n"
                    "🔭 発掘 の「配当がある月」でこれらの月を選ぶと、"
                    "受取を平準化できる銘柄を探せます。")

with tab6:
    st.caption("**権利落ち日までに持っていれば、その回の配当を受け取れます。** "
               "yfinance の権利確定日は実測で1,272社のうち5社しか入っていなかったので、"
               "**配当履歴から予測**しています（会社の発表ではありません）。"
               "日本株の権利落ち日は決算期末に固定されていて、実質その月の最終営業日です。")

    ex = get_next_ex_dates(tuple(sorted(set(positions["ticker"]))))
    m = pd.DataFrame()
    if ex.empty:
        st.info("配当履歴が足りず、予測できませんでした。")
    else:
        pos_sum = positions.groupby("ticker").agg(
            name=("name", "first"), code=("code", "first"),
            sector33=("sector33", "first"), shares=("shares", "sum"),
            eval_value=("eval_value", "sum")).reset_index()
        m = ex.merge(pos_sum, on="ticker", how="inner")
        m["受取見込み"] = (m["shares"] * m["1株配当の目安"]).round(0)

    if not ex.empty and m.empty:
        st.info("保有銘柄と予測を突き合わせられませんでした。")
    elif not ex.empty:
        soon = m[m["あと何日"] <= 30]
        c1, c2, c3 = st.columns(3)
        c1.metric("30日以内に権利落ち", f"{len(soon)} 銘柄")
        c2.metric("その受取見込み", yen(soon["受取見込み"].sum()),
                  help="30日以内に権利落ちする銘柄の配当の合計（税引前）")
        # 「2026-09-30（あと15日）」だと幅が足りず「あと 1…」と切れる。
        # 日付を値に、日数は下の差分に回す。
        c3.metric("いちばん近い日", str(m["次の権利落ち日"].iloc[0]),
                  f"あと {int(m['あと何日'].iloc[0])} 日", delta_color="off")

        if not soon.empty:
            st.warning(f"**{m['次の権利落ち日'].iloc[0]} に {int((m['次の権利落ち日'] == m['次の権利落ち日'].iloc[0]).sum())} 銘柄**が"
                       "権利落ちします。買い増すならこの日までです。")

        days = st.slider("何日先まで見るか", 15, 400, 120, 15, format="%d 日")
        view = m[m["あと何日"] <= days].copy()
        show = pd.DataFrame({
            "次の権利落ち日": view["次の権利落ち日"].astype(str),
            "あと何日": view["あと何日"],
            "コード": view["code"], "銘柄名": view["name"], "業種": view["sector33"],
            "株数": view["shares"], "1株配当の目安": view["1株配当の目安"],
            "受取見込み（税引前）": view["受取見込み"], "評価額": view["eval_value"],
            "根拠": view["根拠"],
        })
        st.dataframe(show, hide_index=True, width="stretch", height=420, column_config={
            "あと何日": st.column_config.NumberColumn(format="%d 日"),
            "1株配当の目安": st.column_config.NumberColumn(
                format="¥%.2f", help="前回の同じ月の実績。会社予想ではありません"),
            "受取見込み（税引前）": st.column_config.NumberColumn(format="¥%d"),
            "評価額": st.column_config.NumberColumn(format="¥%d"),
            "根拠": st.column_config.TextColumn(width="medium"),
        })

        by_date = view.groupby("次の権利落ち日")["受取見込み"].sum().reset_index()
        if len(by_date) > 1:
            fig = go.Figure(go.Bar(x=by_date["次の権利落ち日"].astype(str),
                                   y=by_date["受取見込み"], marker_color="#4C8BF5"))
            fig.update_layout(height=280, yaxis_title="受取見込み（円・税引前）",
                              margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig, width="stretch")
        st.download_button("この一覧をCSVで保存", show.to_csv(index=False).encode("utf-8-sig"),
                           f"権利落ち日_{date.today():%Y%m%d}.csv", "text/csv")

with tab4:
    st.caption("実際に受け取った配当を記録します。**税引後の手取り額**を入れてください。"
               "予想ではなく実績が貯まると、インカムが本当に育っているかが分かります。")
    got = dividends_received(config)

    if not got.empty:
        this_year = got[got["年"] == date.today().year]
        c1, c2, c3 = st.columns(3)
        c1.metric(f"{date.today().year}年の受取", yen(this_year["受取額"].sum()),
                  help="税引後の手取り額の合計")
        c2.metric("累計の受取", yen(got["受取額"].sum()))
        c3.metric("記録件数", f"{len(got)} 件")

        by_year = got.groupby("年")["受取額"].sum().reset_index()
        fig = go.Figure(go.Bar(x=by_year["年"].astype(str), y=by_year["受取額"],
                               marker_color="#4C8BF5",
                               text=[f"{v:,.0f}円" for v in by_year["受取額"]],
                               textposition="outside"))
        fig.update_layout(height=300, yaxis_title="受け取った配当（円・税引後）",
                          margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig, width="stretch")
    else:
        st.info("まだ記録がありません。証券会社の配当金計算書を見ながら、下で足していけます。")

    st.markdown("#### 保有と配当履歴から自動で作る")
    st.caption("**証券会社の計算書を1件ずつ写す必要はありません。** "
               "権利落ち日と、いまの株数から、受け取ったはずの配当を組み立てます。"
               "金額はその場で直せます。")
    c1, c2, c3 = st.columns([1, 1, 2])
    with c1:
        months = st.selectbox("さかのぼる期間", [6, 12, 24, 36], index=1,
                              format_func=lambda m: f"過去 {m} か月")
    with c2:
        lag = st.number_input("権利落ちから入金までの日数", 30, 150, 75, 5,
                              help="日本株はおおむね2〜3か月後に入金されます")
    with c3:
        st.caption("⚠️ 株数は**いまの保有数**で計算しています。"
                   "その時点で株数が違っていた銘柄は、下の表で直してください。"
                   "すでに記録した月は出てきません。")

    exp = expected_dividends(positions, config, months_back=int(months), lag_days=int(lag))
    if exp.empty:
        st.success("自動で作れる未記録の配当はありません（すべて記録済みか、履歴がありません）")
    else:
        st.info(f"**{len(exp)} 件・合計 {yen(exp['受取額（税引後）'].sum())}（税引後）** "
                "が未記録です。内容を確かめて、下のボタンでまとめて記録できます。")
        ed = st.data_editor(
            exp.drop(columns=["_ticker", "_account"]),
            hide_index=True, width="stretch", height=380, key="auto_div_editor",
            column_config={
                "記録する": st.column_config.CheckboxColumn(),
                "コード": st.column_config.TextColumn(disabled=True),
                "銘柄名": st.column_config.TextColumn(disabled=True),
                "口座": st.column_config.TextColumn(disabled=True),
                "権利落ち日": st.column_config.DateColumn(disabled=True),
                "受取日": st.column_config.DateColumn(),
                "株数": st.column_config.NumberColumn(min_value=0.0, step=1.0),
                "1株配当": st.column_config.NumberColumn(format="¥%.2f", disabled=True),
                "税引前": st.column_config.NumberColumn(format="¥%d", disabled=True),
                "受取額（税引後）": st.column_config.NumberColumn(format="¥%d", min_value=0.0),
            })
        picked = ed[ed["記録する"] & (ed["受取額（税引後）"] > 0)]
        st.caption(f"チェックが入っているのは **{len(picked)} 件・"
                   f"{yen(picked['受取額（税引後）'].sum())}** です。")
        if st.button(f"この {len(picked)} 件を記録する", type="primary",
                     disabled=picked.empty):
            with connect() as conn:
                for i, r in picked.iterrows():
                    src = exp.loc[i]
                    conn.execute(
                        "INSERT INTO transactions (date, account, ticker, name, type, "
                        "shares, price, fee, memo, ref_date) "
                        "VALUES (?, ?, ?, ?, 'dividend', 1, ?, 0, ?, ?)",
                        (str(r["受取日"]), src["_account"], src["_ticker"], src["銘柄名"],
                         float(r["受取額（税引後）"]),
                         f"自動生成（権利落ち {src['権利落ち日']}／{r['株数']:.0f}株）",
                         str(src["権利落ち日"])))
            flash(f"✅ 配当の受取 {len(picked)} 件・"
                  f"{yen(picked['受取額（税引後）'].sum())}（税引後）を記録しました。"
                  "間違えたときは下の「まとめて取り消す」で戻せます。")
            st.cache_data.clear()
            st.rerun()

    with st.expander("1件だけ手で足す"):
        holds = positions[["ticker", "name", "account", "shares"]].drop_duplicates()
        labels = holds.apply(
            lambda r: f"{r['ticker'][:-2]} {r['name']}（{'NISA' if r['account'] == 'nisa' else '特定'}）",
            axis=1).tolist()
        lookup = dict(zip(labels, holds.itertuples(index=False)))
        c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
        with c1:
            pick = st.selectbox("銘柄", [""] + labels)
        with c2:
            recv_date = st.date_input("受取日", value=date.today())
        with c3:
            amount = st.number_input("受取額（税引後・円）", 0, 10_000_000, 0, 100)
        with c4:
            st.write("")
            st.write("")
            add = st.button("記録する")
        if add:
            if not pick or amount <= 0:
                st.warning("銘柄と受取額を入れてください")
            else:
                row = lookup[pick]
                with connect() as conn:
                    conn.execute(
                        "INSERT INTO transactions (date, account, ticker, name, type, shares, "
                        "price, fee, memo) VALUES (?, ?, ?, ?, 'dividend', 1, ?, 0, ?)",
                        (recv_date.isoformat(), row.account, row.ticker, row.name,
                         float(amount), "手入力（税引後）"))
                flash(f"✅ {row.name} の配当 {yen(amount)} を記録しました")
                st.cache_data.clear()
                st.rerun()

    with st.expander("記録した配当を取り消す"):
        st.caption("まとめて記録したあとで間違いに気づいたときに使います。"
                   "**消した配当は、また「自動で作る」の一覧に戻ってきます。**")
        tx = read_df("SELECT id, date, account, ticker, name, shares, price, memo "
                     "FROM transactions WHERE type='dividend' ORDER BY date DESC, id DESC")
        if tx.empty:
            st.caption("取り消せる記録はありません")
        else:
            tx["受取額"] = tx["shares"] * tx["price"]
            tx["区分"] = tx["memo"].fillna("").map(
                lambda m: "自動生成" if str(m).startswith("自動生成") else "手入力")
            c1, c2 = st.columns([1, 2])
            with c1:
                scope = st.radio("取り消す範囲", ["選んだ日にまとめて記録したぶん",
                                             "自動生成したものすべて", "1件だけ"])
            targets = pd.DataFrame()
            if scope == "選んだ日にまとめて記録したぶん":
                with c2:
                    days = sorted(tx["date"].unique(), reverse=True)
                    pick_day = st.selectbox(
                        "記録した受取日", days,
                        format_func=lambda d: f"{d}（{int((tx['date'] == d).sum())} 件・"
                                              f"{(tx.loc[tx['date'] == d, '受取額'].sum()):,.0f} 円）")
                    targets = tx[tx["date"] == pick_day]
            elif scope == "自動生成したものすべて":
                targets = tx[tx["区分"] == "自動生成"]
                with c2:
                    st.caption(f"対象は **{len(targets)} 件・"
                               f"{targets['受取額'].sum():,.0f} 円**（手入力したものは残ります）")
            else:
                with c2:
                    labels = {f"{r.date} {r.name}（{r.受取額:,.0f}円）": r.id
                              for r in tx.head(300).itertuples(index=False)}
                    pick_one = st.selectbox("取り消す記録", [""] + list(labels))
                    if pick_one:
                        targets = tx[tx["id"] == labels[pick_one]]

            if not targets.empty:
                st.warning(f"**{len(targets)} 件・{yen(targets['受取額'].sum())}** を取り消します。"
                           "元に戻せません。")
                if st.checkbox("この内容で取り消すことを確認した", key="undo_div_confirm"):
                    if st.button("取り消す", type="primary"):
                        ids = [int(i) for i in targets["id"]]
                        with connect() as conn:
                            conn.executemany("DELETE FROM transactions WHERE id = ?",
                                             [(i,) for i in ids])
                        flash(f"🗑 配当の受取記録 {len(ids)} 件・"
                              f"{yen(targets['受取額'].sum())} を取り消しました。"
                              "「自動で作る」の一覧に戻っています。", "info")
                        st.cache_data.clear()
                        st.rerun()

    if not got.empty:
        st.markdown("#### 記録の一覧")
        view = got[["date", "account", "ticker", "name", "受取額"]].copy()
        view["date"] = view["date"].dt.strftime("%Y-%m-%d")
        view["account"] = view["account"].map({"specific": "特定", "nisa": "NISA"}).fillna(view["account"])
        st.dataframe(view.rename(columns={
            "date": "受取日", "account": "口座", "ticker": "銘柄", "name": "銘柄名"}).iloc[::-1],
            hide_index=True, width="stretch", height=300,
            column_config={"受取額": st.column_config.NumberColumn(format="¥%d")})

with tab5:
    hist = read_df("SELECT * FROM equity_history ORDER BY date")
    st.caption("アプリを開くたびに、その日の評価額と年間配当を1行だけ残しています。"
               "**インカム投資で見るべきは年間配当のほう**です。"
               "評価額は市場が決めますが、年間配当は増配と買い増しで自分が育てられます。")

    if hist.empty:
        st.info("まだ記録がありません。次にこの画面を開いたときから貯まります。")
    else:
        # 前のアプリから引き継いだ期間は「その日に記録した値」ではなく、
        # 『今の保有のまま過去も持っていたら』を過去株価から計算し直した推計。
        # 実測と同じ線で描くと、分からないはずの過去が分かっているように見える。
        if "source" not in hist.columns:
            hist["source"] = "snapshot"
        hist["source"] = hist["source"].fillna("snapshot")
        est = hist[hist["source"] == "backfill"]
        real = hist[hist["source"] != "backfill"]

        if not est.empty:
            st.info(
                f"**{est['date'].min()}〜{est['date'].max()} の {len(est)} 日ぶんは推計です。** "
                "前に使っていた持株管理アプリから引き継いだもので、当時の保有数が"
                "残っていないため『**今の保有のまま過去も持っていたら**評価額はいくらだったか』"
                "を過去の株価から計算し直した値です。"
                + (f"実際にその日に記録した値は {real['date'].min()} からの {len(real)} 日ぶん。"
                   if not real.empty else "実測はまだ1日もありません。"),
                icon="📎")

        # 増減は実測どうしでしか測らない。推計期間の年間配当は今の値を全日に
        # 流用しただけなので、そこを起点に引くと増えていない配当が増えて見える。
        movable = len(real) >= 2
        base = real if movable else hist
        first, last = base.iloc[0], base.iloc[-1]

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("記録した日数", f"{len(real)} 日",
                  help=(f"実測のみ。{real['date'].min()} から {real['date'].max()} まで"
                        if not real.empty else "実測はまだありません")
                       + (f"（ほかに推計 {len(est)} 日ぶん）" if not est.empty else ""))
        c2.metric("年間配当（税引前）", yen(hist.iloc[-1]["annual_dividend"]),
                  yen(last["annual_dividend"] - first["annual_dividend"]) if movable else None,
                  help="いまの保有株数 × 直近の1株配当。増配と買い増しで増えます。"
                       "増減は実測を始めてからの変化です")
        c3.metric("年間配当（税引後）", yen(hist.iloc[-1]["annual_dividend_after_tax"]),
                  yen(last["annual_dividend_after_tax"] - first["annual_dividend_after_tax"])
                  if movable else None)
        c4.metric("YOC", pct(hist.iloc[-1]["yoc"], 2)
                  if pd.notna(hist.iloc[-1]["yoc"]) else "—",
                  f"{(last['yoc'] - first['yoc']) * 100:+.2f}pt"
                  if movable and pd.notna(first["yoc"]) else None,
                  help="いまの年間配当 ÷ 買ったときの金額。増配で育つとここが上がります")

        if len(hist) < 2:
            st.info("グラフは2日ぶん貯まってから出ます。いまは1日ぶんしかありません。")
        else:
            c1, c2 = st.columns(2)
            with c1:
                # 評価額は推計にも意味がある。「今の持ち株がこの半年どう動いたか」は
                # 過去株価から正しく再現できるので、薄い線で続きとして見せる。
                fig = go.Figure()
                if not est.empty:
                    fig.add_scatter(x=est["date"], y=est["total_eval"], name="評価額（推計）",
                                    line=dict(color="#4C8BF5", width=1.5, dash="dot"),
                                    opacity=0.55)
                fig.add_scatter(x=real["date"], y=real["total_eval"], name="評価額（実測）",
                                line=dict(color="#4C8BF5"))
                fig.add_scatter(x=hist["date"], y=hist["total_cost"], name="取得額",
                                line=dict(color="#999", dash="dot"))
                if not est.empty and not real.empty:
                    fig.add_vline(x=real["date"].min(), line_width=1,
                                  line_dash="dash", line_color="#888",
                                  annotation_text="ここから実測",
                                  annotation_position="top left")
                fig.update_layout(height=300, yaxis_title="円", title="評価額と取得額",
                                  margin=dict(l=10, r=10, t=40, b=10))
                st.plotly_chart(fig, width="stretch")
            with c2:
                # 年間配当は推計を描かない。引き継いだ期間の配当は今の値を全日に
                # 流用しただけで、過去について何も語っていない。横ばいの線を引くと
                # 「その間ずっと配当は動かなかった」という嘘になる。
                fig = go.Figure()
                fig.add_scatter(x=real["date"], y=real["annual_dividend"],
                                name="税引前", line=dict(color="#E45756"))
                fig.add_scatter(x=real["date"], y=real["annual_dividend_after_tax"],
                                name="税引後", line=dict(color="#E45756", dash="dot"))
                fig.update_layout(height=300, yaxis_title="円",
                                  title="年間配当（こちらが本番／実測のみ）",
                                  margin=dict(l=10, r=10, t=40, b=10))
                st.plotly_chart(fig, width="stretch")
                if len(real) < 2:
                    st.caption("実測が2日ぶん貯まるとここに線が出ます。"
                               "引き継いだ期間の配当は推計なので描いていません。")

        # 生の列名のまま出すと何の数字か分からないので、必ず日本語に直して単位をつける
        show = pd.DataFrame({
            "日付": hist["date"],
            "種別": hist["source"].map({"backfill": "推計", "snapshot": "実測"}),
            "評価額": hist["total_eval"].round(0),
            "取得額": hist["total_cost"].round(0),
            "損益": (hist["total_eval"] - hist["total_cost"]).round(0),
            "年間配当（税引前）": hist["annual_dividend"].round(0),
            "年間配当（税引後）": hist["annual_dividend_after_tax"].round(0),
            "保有銘柄数": hist["holdings_count"],
            "YOC": to_pct(hist["yoc"]).round(2),
        }).iloc[::-1]
        st.dataframe(show, hide_index=True, width="stretch", height=280, column_config={
            "種別": st.column_config.TextColumn(
                help="実測＝その日に記録した値／推計＝今の保有を過去株価に当てはめた値"),
            "評価額": st.column_config.NumberColumn(format="¥%d"),
            "取得額": st.column_config.NumberColumn(format="¥%d"),
            "損益": st.column_config.NumberColumn(format="¥%d"),
            "年間配当（税引前）": st.column_config.NumberColumn(format="¥%d"),
            "年間配当（税引後）": st.column_config.NumberColumn(format="¥%d"),
            "保有銘柄数": st.column_config.NumberColumn(format="%d 銘柄"),
            "YOC": st.column_config.NumberColumn(
                format="%.2f%%", help="いまの年間配当 ÷ 買ったときの金額"),
        })


with tab9:
    st.caption("**増配で配当が育っているか**を見る。評価額は市場が決めるが、"
               "ここは企業の増配と自分の買い増しだけが動かす。")

    an = positions.copy()
    for col in ("streak", "streak_no_cut", "cuts_10y"):
        if col not in an.columns:
            an[col] = pd.NA
    an["streak"] = pd.to_numeric(an["streak"], errors="coerce").fillna(0)

    # 配当性向はスコアの内訳に入っている（画面をまたいで同じ値を使う）
    def _raw_payout(js):
        if not isinstance(js, str):
            return None
        try:
            import json
            return json.loads(js).get("raw", {}).get("payout_ratio")
        except Exception:
            return None

    an["payout_ratio"] = pd.to_numeric(
        an["detail_json"].map(_raw_payout), errors="coerce")

    # 平均は銘柄数ではなく**年間配当で重みづけ**する。1株だけ持っている銘柄と
    # 1,100株持っている銘柄を同じ1票にすると、実際に受け取る配当の姿とずれる。
    w = an["annual_dividend"].fillna(0)
    wsum = float(w.sum())

    def _wavg(col: pd.Series) -> float | None:
        m = col.notna() & (w > 0)
        return float((col[m] * w[m]).sum() / w[m].sum()) if m.any() and w[m].sum() else None

    avg_streak = _wavg(an["streak"])
    n_10plus = int((an["streak"] >= 10).sum())
    avg_payout = _wavg(an["payout_ratio"])
    n_high = int((an["payout_ratio"] > 0.8).sum())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("連続増配年数（配当で加重）", f"{avg_streak:.1f} 年" if avg_streak else "—",
              help="銘柄数ではなく、受け取る配当の大きさで重みをつけた平均")
    c2.metric("10年以上の連続増配", f"{n_10plus} 銘柄",
              help=f"保有 {len(an)} 銘柄のうち")
    c3.metric("配当性向（配当で加重）", pct(avg_payout) if avg_payout else "—",
              help="配当 ÷ 純利益。30〜50%が健全。低すぎるのも還元する気が薄い")
    c4.metric("配当性向 80%超", f"{n_high} 銘柄",
              help="利益のほとんどを配当に回している状態。減益が来ると減配しやすい")

    st.divider()

    st.markdown("##### 連続増配年数 ベスト15")
    st.caption("⚠️ **この年数は「これから増配する」という意味ではありません。** "
               "🧪検証で調べたところ、連続増配年数は将来のリターンを説明しませんでした。"
               "増配を続けてきた実績の記録として見てください。"
               "据え置きの年があると止まりますが、記念配当の年は飛ばして数えています。")
    top = an[an["streak"] > 0].nlargest(15, "streak")
    if top.empty:
        st.info("連続増配中の保有はありません。")
    else:
        show = pd.DataFrame({
            "コード": top["code"],
            "銘柄名": top["name"],
            "口座": top["account"].map({"specific": "特定", "nisa": "NISA"}),
            "連続増配": top["streak"].astype(int),
            "減配なし": pd.to_numeric(top["streak_no_cut"], errors="coerce"),
            "年間配当": top["annual_dividend"].round(0),
            "YOC": to_pct(top["yoc"]).round(2),
            "配当性向": to_pct(top["payout_ratio"]).round(1),
        })
        st.dataframe(show, hide_index=True, width="stretch", column_config={
            "連続増配": st.column_config.NumberColumn(format="%d 年"),
            "減配なし": st.column_config.NumberColumn(format="%d 年"),
            "年間配当": st.column_config.NumberColumn(format="¥%d"),
            "YOC": st.column_config.NumberColumn(format="%.2f%%"),
            "配当性向": st.column_config.NumberColumn(format="%.1f%%"),
        })

    st.divider()

    st.markdown("##### 配当性向が高い保有 — 減配が起きるとしたらここ")
    st.caption("配当性向は **配当 ÷ 純利益**。70%を超えると利益の変動をそのまま配当が受ける。"
               "100%を超えていれば、利益を超えて配当を出している状態。"
               "ここに出ること自体は売る理由になりません（売りは"
               "**実際に減配が起きてから**）。買い増しを止める材料として見てください。")
    risky = an[an["payout_ratio"] > 0.7].nlargest(15, "payout_ratio")
    if risky.empty:
        st.success("配当性向が70%を超える保有はありません。")
    else:
        show = pd.DataFrame({
            "コード": risky["code"],
            "銘柄名": risky["name"],
            "口座": risky["account"].map({"specific": "特定", "nisa": "NISA"}),
            "配当性向": to_pct(risky["payout_ratio"]).round(1),
            "連続増配": risky["streak"].astype(int),
            "10年の減配回数": pd.to_numeric(risky["cuts_10y"], errors="coerce"),
            "年間配当": risky["annual_dividend"].round(0),
            "配当継続スコア": pd.to_numeric(risky["health"], errors="coerce").round(1),
        })
        st.dataframe(show, hide_index=True, width="stretch", column_config={
            "配当性向": st.column_config.NumberColumn(format="%.1f%%"),
            "連続増配": st.column_config.NumberColumn(format="%d 年"),
            "10年の減配回数": st.column_config.NumberColumn(format="%d 回"),
            "年間配当": st.column_config.NumberColumn(format="¥%d"),
            "配当継続スコア": st.column_config.NumberColumn(
                format="%.1f", help="市場の評価を入れず、配当が続くか・増えるかだけを見たスコア"),
        })
        n_over100 = int((risky["payout_ratio"] > 1.0).sum())
        if n_over100:
            st.warning(f"うち {n_over100} 銘柄は配当性向が100%を超えています。"
                       "利益を超えて配当を出している状態なので、🚨監視の減配シグナルを確認してください。")

    st.divider()

    st.markdown("##### 年間の受取配当 — 階段が上がっているか")
    stair = dividend_staircase(positions, config)
    if stair.empty:
        st.info("配当履歴がまだありません。画面左の「📥 データを更新」を押すと貯まります。")
    else:
        st.caption("**いまの保有数のまま過去も持っていたら**、年ごとにいくら受け取っていたか。"
                   "実際の受取額ではありません（当時の株数は残っていないため）。"
                   "買い増した銘柄ほど過去が大きく出るので、"
                   "**増配率そのものではなく、いまの持ち株が育ってきた形**として見てください。"
                   f"今年（{pd.Timestamp.today().year}年）はまだ権利落ちが済んでいない分があるので入れていません。")

        fig = go.Figure()
        fig.add_bar(x=stair["年"], y=stair["特定（税引後）"], name="特定（税引後）",
                    marker_color="#4C8BF5")
        fig.add_bar(x=stair["年"], y=stair["NISA（非課税）"], name="NISA（非課税）",
                    marker_color="#54A24B")
        fig.update_layout(barmode="stack", height=320, yaxis_title="円",
                          xaxis=dict(dtick=1),
                          margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig, width="stretch")

        n_years = len(stair)
        if n_years >= 2:
            first_y, last_y = stair.iloc[0], stair.iloc[-1]
            cagr = ((last_y["合計"] / first_y["合計"]) ** (1 / (n_years - 1)) - 1
                    if first_y["合計"] > 0 else None)
            up = int((stair["前年比"] > 0).sum())
            c1, c2, c3 = st.columns(3)
            c1.metric(f"{int(first_y['年'])}年 → {int(last_y['年'])}年",
                      yen(last_y["合計"]), yen(last_y["合計"] - first_y["合計"]))
            c2.metric("年率", pct(cagr) if cagr is not None else "—",
                      help="この期間の伸びを1年あたりに直したもの")
            c3.metric("前年より増えた年", f"{up} / {n_years - 1} 年")

        show = stair.copy()
        show["年"] = show["年"].astype(int).astype(str) + "年"
        show["前年比"] = to_pct(show["前年比"]).round(1)
        st.dataframe(show.iloc[::-1], hide_index=True, width="stretch", height=260,
                     column_config={
                         "特定（税引後）": st.column_config.NumberColumn(format="¥%d"),
                         "NISA（非課税）": st.column_config.NumberColumn(format="¥%d"),
                         "合計": st.column_config.NumberColumn(format="¥%d"),
                         "前年比": st.column_config.NumberColumn(format="%.1f%%"),
                     })


with tab10:
    st.caption("**自分で選んだ結果が、指数をただ買って放っておいた場合より良かったのか。** "
               "対照を置かずに自分の成績だけ見ると、相場が上がっただけの期間を"
               "実力だと取り違えます。")

    BM_LABELS = {
        "1306.T": "TOPIX（ETF）",
        "1577.T": "日本高配当株70（ETF）",
        "1489.T": "日経高配当株50（ETF）",
    }
    have = read_df("SELECT symbol, COUNT(*) AS n, MIN(date) AS a, MAX(date) AS b "
                   "FROM benchmarks GROUP BY symbol")
    if have.empty:
        st.info("比較するデータがまだありません。\n\n"
                "```bash\nuv run python scripts/fetch_benchmarks.py\n```")
    else:
        picked = st.multiselect(
            "比べる相手", list(BM_LABELS.keys()),
            default=[s for s in ("1306.T", "1577.T") if s in set(have["symbol"])],
            format_func=lambda s: BM_LABELS.get(s, s))
        cmp_df = portfolio_vs_benchmark(positions, picked)

        if cmp_df.empty or len(cmp_df) < 2:
            st.info("推移が2日ぶん貯まると比べられます。")
        else:
            first_d, last_d = cmp_df["date"].iloc[0], cmp_df["date"].iloc[-1]
            st.caption(f"{first_d:%Y-%m-%d} を100として、{last_d:%Y-%m-%d} まで。"
                       "どちらも**配当・分配金を受け取って持ち続けた**前提です"
                       "（片方だけ配当を抜くと、配当を出している側が一方的に低く出ます）。"
                       "税金はどちらも引いていません。"
                       "売買でお金が出入りした日は、その分を差し引いてから増減を測っています。")

            fig = go.Figure()
            fig.add_scatter(x=cmp_df["date"], y=cmp_df["自分の持ち株"],
                            name="自分の持ち株", line=dict(color="#E45756", width=2.5))
            palette = ["#4C8BF5", "#54A24B", "#B279A2"]
            for i, s in enumerate([c for c in cmp_df.columns
                                   if c not in ("date", "自分の持ち株")]):
                fig.add_scatter(x=cmp_df["date"], y=cmp_df[s], name=BM_LABELS.get(s, s),
                                line=dict(color=palette[i % len(palette)],
                                          width=1.5, dash="dot"))
            fig.add_hline(y=100, line_width=1, line_color="#bbb")
            fig.update_layout(height=360, yaxis_title="起点=100",
                              margin=dict(l=10, r=10, t=20, b=10),
                              legend=dict(orientation="h", yanchor="bottom", y=1.0))
            st.plotly_chart(fig, width="stretch")

            last = cmp_df.iloc[-1]
            mine = last["自分の持ち株"] - 100.0
            cols = st.columns(1 + len([c for c in cmp_df.columns
                                       if c not in ("date", "自分の持ち株")]))
            cols[0].metric("自分の持ち株", f"{mine:+.1f}%")
            for i, s in enumerate([c for c in cmp_df.columns
                                   if c not in ("date", "自分の持ち株")]):
                theirs = last[s] - 100.0
                cols[i + 1].metric(BM_LABELS.get(s, s), f"{theirs:+.1f}%",
                                   f"{mine - theirs:+.1f}pt",
                                   help="自分との差。マイナスなら、選ぶ手間をかけて"
                                        "この指数に負けていたということ")

            diffs = {s: mine - (last[s] - 100.0)
                     for s in cmp_df.columns if s not in ("date", "自分の持ち株")}
            if diffs and max(diffs.values()) < 0:
                worst = min(diffs, key=diffs.get)
                st.warning(
                    f"**この期間は、選んだ結果が指数に負けています**"
                    f"（{BM_LABELS.get(worst, worst)}に {abs(diffs[worst]):.1f}pt）。"
                    "ただし判断はこの1回では決まりません。期間が短いほど差は運で動きます。"
                    "同じ差が期間を変えても続くようなら、銘柄を選ぶこと自体を見直す材料になります。",
                    icon="⚠️")
            elif diffs and min(diffs.values()) > 0:
                st.success(
                    "この期間は、選んだ結果がどの指数も上回っています。"
                    "期間が短いうちは運の割合が大きいので、続くかどうかを見てください。",
                    icon="✅")

            st.caption(f"※ 比較できるのは推移が貯まっている {len(cmp_df)} 日ぶんだけです。"
                       "期間を伸ばすには推移を貯め続ける必要があります。")
