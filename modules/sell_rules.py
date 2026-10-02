"""売る理由を、事実にもとづいて判定する。

【設計の前提】
以前は `review_candidates` が「新規買いのゲートを外れたか」で整理候補を出していた。
実測すると保有114銘柄のうち84銘柄が「重大」になり、その中身は

    三菱UFJ・群馬銀行  営業CFがマイナス  → 銀行は貸出を増やすと営業CFがマイナスになる。正常
    トヨタ自動車       FCFで配当を賄えていない → 金融事業込みのFCF。売り理由ではない
    MS&AD             配当性向が計算できない → 単なるデータ欠損。連続増配17年
    黒田グループ       上場5年未満 → 新規上場は売り理由ではない
    イントラスト       売買代金が薄い → 流動性は「買う前に確かめること」

という、売る理由になっていないものばかりだった。

**買わない理由と、売る理由は別物。**
買うときは選択肢が3,700あるので厳しくてよい。持っている株を売るのは
税金・スプレッド・再投資先の確保というコストを払う行為なので、
「配当が壊れた」という事実にだけ反応させる。

【4つの段階】
    🔴 売却を検討   配当そのものが壊れた（事実として起きたこと）
    🟡 監視を強める  配当の原資が傷んでいる（まだ配当は出ている）
    🟢 利確を検討   ヘムの「上がりすぎたら売る」。壊れたのではなく育ちきった
    ⚪️ 手入れ       売る理由ではない。額が小さい・口座が分かれている等

【適用しないもの】
    金融（銀行・保険・証券）には営業CF・FCF・有利子負債の基準を当てない
    データが欠けているだけのものは「判定待ち」に回し、売り理由にしない
    上場年数・売買代金・時価総額は新規買いの条件であって売り理由にしない
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from modules.store import read_df

# 重大度の並び順と、優先度の基準点
# 基準点。ここを高くしすぎると上限100で頭打ちになり、同じ重さの中で
# 金額の差が効かなくなる（実測で1株3,240円の銘柄が84万円の銘柄より上に来ていた）。
SEVERITY = {
    "売却を検討": ("🔴", 72),
    "監視を強める": ("🟡", 44),
    "利確を検討": ("🟢", 26),
    "手入れ": ("⚪️", 4),
    "判定待ち": ("⏳", 0),
}
SEVERITY_ORDER = list(SEVERITY)

# 指標に5つ並べると「🟡 監視を強…」と切れるので、短い見出しも持っておく
SHORT = {"売却を検討": "売却", "監視を強める": "監視", "利確を検討": "利確",
         "手入れ": "手入れ", "判定待ち": "判定待ち"}

# 累進配当・DOE を掲げている会社の扱い（2026-10-02 に見直し）。
# 以前は方針を見ずに「10年で3回減配」「配当継続スコアが低く減配歴あり」を
# 売り・買い増し停止の理由にしていた。しかし減配の多くは方針を掲げる前の話で、
# 掲げたあとは配当を減らさない圧力が内側から働く。手元の全市場データで確かめると、
#     2016〜23年に減配歴がある会社の、2024〜26年の減配率
#         累進配当を掲げる   2.9%（68社）
#         DOE を掲げる       7.0%（86社）
#         どちらも無い      10.1%（227社）
# 方針は「今の有報」の文言なので後知恵が混ざる（減配した会社は累進を書きにくい）。
# それでも、過去の減配回数だけで売りを勧める根拠にはならない。
# そこで、方針を掲げていて**直近3年に減配が無い**会社には、過去の減配回数と
# 配当継続スコアの低さを売り理由として使わない。方針を掲げながら実際に減配したら、
# それは約束を破ったので従来どおり 🔴 にする。
_COMMITTED = ("累進配当", "DOE（純資産配当率）")


def _policy(raw: dict) -> dict:
    v = raw.get("policy_flags")
    if isinstance(v, dict):
        return v
    try:
        return json.loads(v) if v else {}
    except Exception:
        return {}


# 営業CF・FCF・有利子負債の基準を当てない業種。
# 銀行は貸出が増えると営業CFがマイナスになり、保険は責任準備金で歪む。
_FINANCIAL = {"銀行業", "保険業", "証券、商品先物取引業", "その他金融業"}


def _periods(tickers: list[str]) -> dict[str, pd.DataFrame]:
    """銘柄ごとの期別財務（新しい期が先頭）。"""
    if not tickers:
        return {}
    ph = ",".join("?" * len(tickers))
    f = read_df(
        f"SELECT ticker, fiscal_end, net_income, operating_cf, total_equity, total_assets "
        f"FROM fundamentals WHERE ticker IN ({ph}) ORDER BY ticker, fiscal_end DESC",
        tuple(tickers))
    return {t: g.reset_index(drop=True) for t, g in f.groupby("ticker")} if not f.empty else {}


def _dps_series(tickers: list[str]) -> dict[str, pd.Series]:
    if not tickers:
        return {}
    ph = ",".join("?" * len(tickers))
    div = read_df(f"SELECT ticker, date, amount FROM dividends WHERE ticker IN ({ph})",
                  tuple(tickers))
    if div.empty:
        return {}
    from modules.dividend_history import build_profiles, fiscal_months
    return {t: pd.Series(p.series).sort_index()
            for t, p in build_profiles(div, fiscal_months()).items()}


def _consecutive(vals: list[float], test) -> int:
    """先頭（直近）から何期連続で test を満たすか。"""
    n = 0
    for v in vals:
        if v is None or (isinstance(v, float) and np.isnan(v)) or not test(v):
            break
        n += 1
    return n


def evaluate(pos: pd.DataFrame, config: dict) -> pd.DataFrame:
    """保有1行ごとに、売る理由を判定して返す。

    返る列
        重さ            売却を検討 / 監視を強める / 利確を検討 / 手入れ / 判定待ち
        印              🔴🟡🟢⚪️⏳
        理由            そのまま読める日本語（複数は改行区切り）
        根拠            数値の実物
        やること        次の一手
        整理の優先度    0-100
    """
    if pos is None or pos.empty:
        return pd.DataFrame()

    sell = config.get("sell", {})
    total = float(pos["eval_value"].sum()) or 1.0
    tickers = sorted(set(pos["ticker"]))
    fund = _periods(tickers)
    dps = _dps_series(tickers)
    dup = pos.groupby("ticker")["account"].nunique()

    rows = []
    for _, r in pos.iterrows():
        t = r["ticker"]
        sector = str(r.get("sector33") or "")
        is_fin = sector in _FINANCIAL
        detail = {}
        try:
            detail = json.loads(r.get("detail_json") or "{}")
        except Exception:
            detail = {}
        raw = detail.get("raw", {}) if isinstance(detail, dict) else {}
        traps = detail.get("traps", []) if isinstance(detail, dict) else []

        payout = raw.get("payout_ratio")
        streak = int(r.get("streak") or 0)
        cuts = int(r.get("cuts_10y") or 0)
        no_cut = r.get("streak_no_cut")
        policy = _policy(raw)
        committed = [k for k in _COMMITTED if policy.get(k)]
        held_3y = pd.notna(no_cut) and int(no_cut) >= 3
        # 方針を掲げていて、直近3年は減配していない＝過去の減配は方針以前の話とみなす
        protected = bool(committed) and held_3y
        policy_note = "・".join(k.replace("（純資産配当率）", "") for k in committed)
        f = fund.get(t)
        ni = f["net_income"].tolist() if f is not None else []
        ocf = f["operating_cf"].tolist() if f is not None else []

        found: list[tuple[str, str, str, str]] = []   # (重さ, 理由, 根拠, やること)

        # ── 🔴 配当そのものが壊れた ───────────────────────────────
        if pd.notna(no_cut) and int(no_cut) == 0 and cuts >= 1 and committed:
            found.append(("売却を検討", f"{policy_note}を掲げているのに、直近の配当年度で減配した",
                          f"10年の減配 {cuts} 回／配当方針: {policy_note}",
                          "方針どおりなら起きないはずの減配。会社が方針を変えたか、"
                          "記念配当・特別配当がなくなっただけかを決算資料で確かめる。"
                          "後者なら売る理由ではない"))
        elif pd.notna(no_cut) and int(no_cut) == 0 and cuts >= 1:
            # 2026-09-16 の検証（延べ16,950 銘柄・年）で、この理由の中身が変わった。
            # **減配した銘柄が持ち続けると損をする、という証拠は無い。**
            # 同じ年・同じ利回り帯でそろえて比べると、5年リターンの差は +1.0pt。
            # 1年目だけは負ける（+1.7% 対 +7.1%）が、そこから戻ってくる。
            # 配当も、減配した年を1.0として3年で1.20倍と、減配しなかった銘柄と
            # 同じペースで回復していた。
            # 売る根拠は「下がり続けるから」ではなく、次の2つ。
            #   ・また減配する確率が 1.41倍（3年以内 39.5% 対 28.1%）
            #   ・乗り換え先のほうが明確に強い（高利回り×減配なしの5年リターン
            #     +37.7% 対 減配した銘柄 +24.2%。13.5ポイント差）
            found.append(("売却を検討", "直近の配当年度で減配した",
                          f"10年の減配 {cuts} 回／連続増配は途切れている"
                          "／過去の実測では、減配した銘柄はその後3年でまた減配する確率が1.41倍",
                          "増配を前提に買ったなら前提が消えている。"
                          "株価は5年で戻ることが多いので急ぐ必要はないが、"
                          "同じ資金を『高利回り かつ 減配歴なし』に移すと"
                          "5年リターンの中央値が13ポイント高い。乗り換え先を決めてから動く"))
        if cuts >= int(sell.get("cuts_10y_serious", 3)) and not protected:
            if held_3y:
                # 減配はしているが直近3年は無い。過去の事実で「売却」にはしない
                found.append(("監視を強める", f"10年で {cuts} 回減配している（直近3年は減配なし）",
                              f"減配 {cuts} 回／減配なし継続 {int(no_cut)} 年／連続増配 {streak} 年",
                              "景気で配当が動きやすい会社。売る理由ではないが、"
                              "不況のときに配当が減る前提で持つ"))
            else:
                found.append(("売却を検討", f"10年で {cuts} 回も減配している",
                              f"減配 {cuts} 回／連続増配 {streak} 年",
                              "景気で配当を動かす会社。増配を積み上げる器ではない"))
        if payout is not None and payout > float(sell.get("payout_unsustainable", 1.0)) \
                and streak < int(sell.get("payout_grace_streak", 10)):
            # 配当性向は「直近の実績」で出している。利益が一時的に吹き飛んだ年は、
            # 恒常的に払いすぎている会社と同じ見た目になる。どちらなのかは
            # 前の年までの利益と比べないと分からないので、ここで分けて書く。
            # 会社が決算短信で出す配当性向は**その期の予想**に対する数字なので、
            # こういう年はそちらと大きく食い違う（実測: 日本製鉄 2026年3月期は
            # 純利益が 3,502億→172億 と95%落ち、実績ベースでは700%超になる）。
            prior = [v for v in ni[1:4] if v is not None and v > 0]
            base_ni = float(np.median(prior)) if prior else None
            collapsed = (base_ni is not None and ni and ni[0] is not None
                         and ni[0] < base_ni * 0.4)
            if collapsed and committed:
                found.append(("監視を強める",
                              f"配当が利益を超えている（今期の利益が落ち込んだが、{policy_note}で配当を維持）",
                              f"配当性向 {payout:.0%}（100%超）／純利益 "
                              f"{ni[0]/1e8:,.0f}億（前の年までは {base_ni/1e8:,.0f}億）"
                              f"／配当方針: {policy_note}",
                              "利益が落ちた年に配当を据え置くのは方針どおりの動き。"
                              "**利益が戻るかどうか**だけを次の決算で見る"))
            elif collapsed:
                found.append(("売却を検討", "配当が利益を超えている（今期の利益が落ち込んだため）",
                              f"配当性向 {payout:.0%}（100%超）／純利益 "
                              f"{ni[0]/1e8:,.0f}億（前の年までは {base_ni/1e8:,.0f}億）"
                              f"／連続増配 {streak} 年",
                              "恒常的に払いすぎているのではなく、利益が落ちた年。"
                              "会社が出す配当性向（今期予想に対する数字）とは大きく食い違う。"
                              "**利益が戻るかどうか**で判断が変わるので、これ単独では動かない。"
                              "減配が実際に起きているかを先に見る"))
            elif committed:
                # DOE・累進は利益ではなく純資産や前年の配当で額を決めるので、
                # 利益が落ちた年は配当性向が100%を超えうる。それ自体は方針どおり
                found.append(("監視を強める",
                              f"利益を超えて配当を出している（{policy_note}で配当を決める会社）",
                              f"配当性向 {payout:.0%}（100%超）／連続増配 {streak} 年"
                              f"／配当方針: {policy_note}",
                              "利益ではなく純資産や前年の配当を基準に額を決めているので、"
                              "利益が落ちた年は100%を超えることがある。"
                              "2年続いて利益が戻らないなら、方針の見直し（減配）を疑う"))
            else:
                found.append(("売却を検討", "利益を超えて配当を出している",
                              f"配当性向 {payout:.0%}（100%超）／連続増配 {streak} 年",
                              "取り崩しで配当を維持している。続かない"))

        # ── 🟡 原資が傷んでいる（金融には当てない） ─────────────────
        if not is_fin:
            # 営業CF・赤字の連続は、2026-09-16 の検証では**弱い**兆候だった。
            #   営業CFが赤字   減配率リフト 1.11（学習）/ 1.13（検証）
            #   最終赤字       リフト 1.10 / 1.26
            # 1.3 に届かないので、これ単独では動かない材料として扱う。根拠にその旨を
            # 書いて、ケンが重みを付けられるようにする。
            neg_ocf = _consecutive(ocf, lambda v: v < 0)
            if neg_ocf >= int(sell.get("neg_ocf_years", 2)):
                found.append(("監視を強める", f"営業キャッシュフローが {neg_ocf} 期連続でマイナス",
                              f"直近の営業CF {', '.join(f'{v/1e8:,.0f}億' for v in ocf[:3] if v is not None)}"
                              "／過去の実測では、この兆候だけでの減配率は平均の1.1倍どまり",
                              "配当の原資そのものが無い。ただし単独では弱い材料。"
                              "配当性向や利回り順位と重なったときに重く見る"))
            neg_ni = _consecutive(ni, lambda v: v < 0)
            if neg_ni >= int(sell.get("loss_years", 2)):
                found.append(("監視を強める", f"最終赤字が {neg_ni} 期連続",
                              f"直近の純利益 {', '.join(f'{v/1e8:,.0f}億' for v in ni[:3] if v is not None)}"
                              "／この兆候だけでの減配率は平均の1.1〜1.3倍",
                              "赤字のまま配当を続けると自己資本が削れる。"
                              "他の兆候と重なっているかを見る"))
            # 「借金が重く自己資本が薄い」は 2026-09-16 の検証で**外した**。
            # 自己資本比率30%未満の会社の2年以内の減配率は、平均の 0.95倍（学習）/
            # 0.81倍（検証）で、むしろ**減配しにくい側**だった。設備産業・通信・
            # 鉄道のように、借金を抱えたまま安定した配当を出し続ける会社が日本には多い。
            # 理屈は通っていても、データが支持しない材料で売りを勧めてはいけない。

        # 減益が続いている × 配当性向が高い＝次の減配の予備軍
        dec_ni = 0
        for a, b in zip(ni, ni[1:]):
            if a is None or b is None or np.isnan(a) or np.isnan(b) or a >= b:
                break
            dec_ni += 1
        if dec_ni >= int(sell.get("earnings_slide_years", 2)) and payout is not None \
                and payout > float(sell.get("payout_watch", 0.60)):
            found.append(("監視を強める", f"純利益が {dec_ni} 期連続で減っていて、配当性向も高い",
                          f"配当性向 {payout:.0%}／純利益 {dec_ni} 期連続の減益",
                          "利益が戻らなければ、配当性向が100%に届いて減配になる"))

        for tr in traps:
            tr = str(tr)
            # 銀行・保険の営業CFは預金・貸出の増減で大きく振れるので、
            # 「利益に営業CFが追いついていない」は金融には意味がない
            if is_fin and "営業CF" in tr:
                continue
            # 利益の連続減少（減配率1.2倍前後）と利益・営業CFの乖離（1.15倍）は弱い兆候。
            # 累進・DOE を掲げて配当を守っている会社では売り材料にしない
            if protected and (tr.startswith("利益が連続で減少") or "営業CF" in tr):
                continue
            found.append(("監視を強める", tr, "スキャン時の減点判定",
                          "増配の前提が細っていないか、次の決算で確かめる"))

        # 配当継続スコアが著しく低く、かつ実際に減配歴がある
        hl = r.get("health")
        if pd.notna(hl) and hl < float(sell.get("health_danger", 30)) and cuts >= 1 \
                and not protected:
            found.append(("監視を強める", "配当継続スコアが著しく低く、減配歴もある",
                          f"配当継続スコア {hl:.0f}（50が真ん中）／10年の減配 {cuts} 回"
                          + (f"／配当方針: {policy_note}" if committed else ""),
                          "増配余力・増配意思・原資成長がそろって弱い。"
                          "買い増しは、利益が戻ってからでよい"))

        # ── 🟡 増配が止まった ────────────────────────────────────
        s = dps.get(t)
        if s is not None and len(s) >= 4:
            recent = s.iloc[-4:]
            flat = int((recent.diff().dropna().abs() < 1e-9).sum())
            if flat >= int(sell.get("flat_dps_years", 3)) and payout is not None \
                    and payout > float(sell.get("flat_payout_floor", 0.50)):
                found.append(("監視を強める", f"配当が {flat} 年すえ置きのまま、配当性向だけが高い",
                              f"1株配当 {'→'.join(f'{v:.0f}円' for v in recent)}／"
                              f"配当性向 {payout:.0%}",
                              "増配で育つ前提が崩れている。増配余地のある銘柄に入れ替える価値がある"))

        # ── 🟢 利確を検討（ヘムの「上がりすぎたら売る」） ──────────────
        yp = raw.get("yield_percentile")
        per = raw.get("per")
        pnl = r.get("pnl_pct")
        cur_y = r.get("current_yield")
        tgt = r.get("target_yield")
        if yp is not None and pd.notna(pnl) \
                and yp <= float(sell.get("rich_yield_percentile", 0.15)) \
                and pnl >= float(sell.get("rich_min_gain", 0.30)):
            found.append(("利確を検討", "この会社としては、めったにない高い株価になっている",
                          f"自己利回り順位 {yp:.0%}（低いほど株価が高い）／含み益 {pnl:+.0%}"
                          + (f"／PER {per:.1f}倍" if per else ""),
                          "ヘムは配当性向50%超・PER15倍超で利確する。"
                          "同じ配当を、より安い銘柄で買い直せる"))
        # PER で見る分岐には、**自己利回り順位が低いこと**を必ず重ねる。
        # これが無いと「もともと低利回りの銘柄」を全部拾ってしまう。
        # 実測: JR東日本は自己利回り順位87%＝**自分史上いちばん安い水準**なのに
        # 「上がりすぎた」と出ていた（PER16.1・利回り2.06% < 目標4.7%×0.6）。
        # それは「買うべきでなかった理由」であって「いま売る理由」ではない。
        elif per is not None and per > float(sell.get("rich_per", 15.0)) \
                and pd.notna(pnl) and pnl >= float(sell.get("rich_min_gain", 0.30)) \
                and yp is not None \
                and yp <= float(sell.get("rich_yield_percentile_loose", 0.50)) \
                and cur_y is not None and pd.notna(cur_y) and pd.notna(tgt) \
                and cur_y < float(tgt) * float(sell.get("rich_yield_vs_target", 0.6)):
            found.append(("利確を検討", "利回りが目標を大きく下回るところまで買われた",
                          f"PER {per:.1f}倍／現在利回り {cur_y:.2%}（目標 {float(tgt):.2%}）／"
                          f"自己利回り順位 {yp:.0%}／含み益 {pnl:+.0%}",
                          "インカムの効率が落ちている。利回りの高い銘柄に移す候補"))

        # ── ⚪️ 手入れ（売り理由ではない） ─────────────────────────
        share = r["eval_value"] / total
        if share < float(sell.get("tiny_position", 0.003)):
            found.append(("手入れ", f"保有額が小さい（全体の {share:.2%}）",
                          f"評価額 {r['eval_value']:,.0f} 円",
                          "売るか、単元を足して意味のある大きさにするか。持ち続けても害はない"))
        if dup.get(t, 1) > 1 and share < float(sell.get("tiny_position", 0.003)) * 3:
            found.append(("手入れ", "同じ銘柄が2つの口座に分かれていて、こちら側が小さい",
                          f"{int(dup[t])} 口座／こちらは全体の {share:.2%}",
                          "NISA側に寄せると配当の手取りが約2割増える。まとめる候補"))

        # ── ⏳ 判定待ち ─────────────────────────────────────────
        if not found:
            missing = []
            if payout is None:
                missing.append("配当性向")
            if not ni:
                missing.append("財務")
            if missing:
                rows.append(_row(r, "判定待ち",
                                 [(f"{'・'.join(missing)}がまだ取れていない", "—",
                                   "`uv run python scripts/weekly_scan.py --missing-only` で埋まる")],
                                 share, total))
            continue

        # 最も重いものを代表にする
        sev = min((x[0] for x in found), key=lambda s: SEVERITY_ORDER.index(s))
        rows.append(_row(r, sev, [(f[1], f[2], f[3]) for f in found if f[0] == sev]
                         + [(f"（{f[0]}）{f[1]}", f[2], f[3]) for f in found if f[0] != sev],
                         share, total, n_rules=len(found)))

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["_o"] = out["重さ"].map({k: i for i, k in enumerate(SEVERITY_ORDER)})
    return out.sort_values(["_o", "整理の優先度"], ascending=[True, False]).drop(columns="_o")


def _row(r: pd.Series, sev: str, found: list[tuple[str, str, str]],
         share: float, total: float, n_rules: int = 1) -> dict:
    mark, base = SEVERITY[sev]
    if sev == "判定待ち":
        return {
            "account": r["account"], "ticker": r["ticker"], "code": r.get("code"),
            "name": r.get("name"), "sector33": r.get("sector33"),
            "shares": r.get("shares"), "avg_cost": r.get("avg_cost"),
            "last_close": r.get("last_close"), "eval_value": r.get("eval_value"),
            "pnl_pct": r.get("pnl_pct"), "health": r.get("health"),
            "streak": r.get("streak"), "dps_latest": r.get("dps_latest"),
            "target_yield": r.get("target_yield"), "構成比": share, "重さ": sev, "印": mark,
            "理由": "\n".join(f"- {a}" for a, _, _ in found),
            "根拠": "\n".join(f"- {b}" for _, b, _ in found),
            "やること": "\n".join(f"- {c}" for _, _, c in found),
            "件数": len(found), "整理の優先度": 0.0,
        }
    # 同じ重さの中では、金額が大きいものと、含み損で税金がかからないものを先に。
    size = min(share / 0.02, 1.0) * 100          # 全体の2%で満点
    pnl = r.get("pnl_pct")
    tax_ease = 50.0 if pd.isna(pnl) else float(np.clip(-pnl * 100, -50, 50) + 50)
    prio = base + 3 * (n_rules - 1) + 0.15 * size + 0.03 * tax_ease
    return {
        "account": r["account"], "ticker": r["ticker"], "code": r.get("code"),
        "name": r.get("name"), "sector33": r.get("sector33"),
        "shares": r.get("shares"), "avg_cost": r.get("avg_cost"),
        "last_close": r.get("last_close"), "eval_value": r.get("eval_value"),
        "pnl_pct": pnl, "health": r.get("health"), "streak": r.get("streak"),
        "dps_latest": r.get("dps_latest"), "target_yield": r.get("target_yield"),
        "構成比": share,
        "重さ": sev, "印": mark,
        # Markdown の箇条書きにする。「・」＋改行1つだと Markdown が
        # 1行に潰してしまい、複数の理由がつながって読めなくなる。
        "理由": "\n".join(f"- {a}" for a, _, _ in found),
        "根拠": "\n".join(f"- {b}" for _, b, _ in found),
        "やること": "\n".join(f"- {c}" for _, _, c in found),
        "件数": len(found),
        "整理の優先度": round(min(prio, 100.0), 0),
    }


def summary(ev: pd.DataFrame) -> dict[str, int]:
    if ev is None or ev.empty:
        return {k: 0 for k in SEVERITY_ORDER}
    c = ev["重さ"].value_counts().to_dict()
    return {k: int(c.get(k, 0)) for k in SEVERITY_ORDER}
