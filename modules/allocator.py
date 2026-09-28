"""資金投入の配分エンジン。

「トレードで出た利益を移した。どこに入れるか」に答えるのが仕事。

【発掘スコア順に買わない】
発掘スコア（total）には「見過ごされ度」が入っている。市場に気づかれていないことは
**見つける理由**にはなっても、**いま買う理由**にはならない。
限られた資金の配り先を決めるときは、次の4つで優劣をつける。

    割安度        その銘柄自身の過去と比べて安いか（検証で唯一きれいに効いた）
    配当継続      買ったあと減配しないか
    目標到達      目標利回りに届いているか（届いていないなら待つのがヘム流）
    補完度        空いている業種・空いている配当月を埋めるか

重みは config.yaml の buy_priority。

最適化ソルバは使わない。上位候補から順に「入れられるか」を判定していくだけで
十分な精度が出るうえ、なぜその配分になったかを1行ずつ説明できる。
説明できない配分は使えない。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

_LOT = 100  # 東証の売買単位（2018年以降は原則100株）

# 単元未満株（1株から買える仕組み）。SBIのS株、楽天のかぶミニ、マネックスのワン株など。
# 毎月の入金額で単元（10万〜70万円）を買えることは稀なので、こちらが既定。
_ODD_LOT = 1


def _lot_size(price: float, budget: float, lot: int = _LOT) -> int:
    if not price or price <= 0:
        return 0
    return int(budget // (price * lot)) * lot


def buy_priority(cand: pd.DataFrame, positions: pd.DataFrame, config: dict,
                 month_gap: dict[int, float] | None = None) -> pd.DataFrame:
    """買い付け優先度（0〜100）と、その内訳を付けて返す。

    【なぜ加重和ではなく幾何平均か】
    加重和だと「片方が極端に高ければ、もう片方が低くても選ばれる」。
    実測すると『割安度80超・配当継続35未満』の銘柄が13件あり、加重和では
    69点で上位に入っていた（小松マテーレ 割安95・継続34 など）。
    配当投資では「安いが配当が危ない」も「安全だが高い」もどちらも買いたくない。
    **両方そろって初めて買う**という性質を、そのまま式にする。

        割安度100・継続100 → 100
        割安度100・継続25  →  50   （加重和なら68になってしまう）
        割安度60・継続60   →  60

    補完度は順位を大きく動かすものではなく、同点のときに
    「空いている業種・空いている配当月を埋めるほう」を選ぶための係数（±10%）。

    トラップ減点は必ず引く。実測で、引かないと買い付け上位50のうち7銘柄が
    高配当トラップ判定（最大19点）の銘柄だった。
    """
    cap = config["portfolio"]["max_sector_weight"]
    # 重みは必ず config から読む。ここに数字を直書きすると、config.yaml を直しても
    # 何も起きない「効かないつまみ」になる（実際そうなっていた）。
    bp = config.get("buy_priority", {})
    w_self = float(bp.get("self_yield_weight", 0.6))
    w_abs = float(bp.get("abs_yield_weight", 0.4))
    swing = float(bp.get("fit_swing", 0.2))
    total_eval = positions["eval_value"].sum() if not positions.empty else 0.0
    sector_now = (positions.groupby("sector33")["eval_value"].sum()
                  if not positions.empty else pd.Series(dtype=float))

    c = cand.copy()

    # ── 割安度 ──
    # 「安い」には2つの意味がある。
    #   自己利回り順位 … その銘柄自身の過去と比べて安いか（買う時期の判断）
    #   絶対利回り順位 … 候補の中で利回りが高いか（いま受け取れる額）
    # 前者だけだと「いつも1.5%の銘柄が2.0%」を最高評価してしまい、
    # インカムとしては物足りない銘柄が上位に来る。両方を混ぜる。
    self_rank = (c["yield_percentile"].fillna(0.5) * 100).clip(0, 100)
    abs_rank = c["dividend_yield"].rank(pct=True) * 100
    c["_割安度"] = (w_self * self_rank + w_abs * abs_rank.fillna(50)).clip(0, 100)
    c["_自己利回り順位"] = self_rank
    c["_絶対利回り順位"] = abs_rank.fillna(50)

    # ── 配当の質 ──
    c["_継続"] = c["health"].fillna(0).clip(0, 100)

    # ── 補完度（タイブレーク）──
    if total_eval > 0:
        used = c["sector33"].map(sector_now).fillna(0.0)
        room = ((cap * total_eval - used) / (cap * total_eval)).clip(0, 1)
    else:
        room = pd.Series(1.0, index=c.index)
    if month_gap:
        def _month_score(s: str) -> float:
            months = [int(m.replace("月", "")) for m in str(s).split("・") if m]
            if not months:
                return 0.5
            return float(np.mean([month_gap.get(m, 0.5) for m in months]))
        month_fit = c["payout_months"].fillna("").map(_month_score)
    else:
        month_fit = pd.Series(0.5, index=c.index)
    c["_補完度"] = ((room + month_fit) / 2 * 100).clip(0, 100)

    # ── 目標到達度（表示用。順位には使わず、絞り込みの条件として使う）──
    # 目標利回りが結合されていない画面から呼ばれることもあるので、既定値で埋める
    if "target_yield" not in c.columns:
        c["target_yield"] = np.nan
    tgt = c["target_yield"].fillna(0.047).replace(0, np.nan)
    c["_目標到達"] = (c["dividend_yield"] / tgt * 100).clip(0, 200).fillna(0)

    base = np.sqrt(c["_割安度"].clip(lower=0) * c["_継続"].clip(lower=0))
    tiebreak = (1 - swing / 2) + swing * c["_補完度"] / 100
    # trap_penalty 列が無い呼び出し元があるので、Series で受けてから引く
    # （c.get(..., 0) は列が無いと int の 0 を返すため .fillna で落ちる）
    penalty = c["trap_penalty"].fillna(0) if "trap_penalty" in c.columns \
        else pd.Series(0.0, index=c.index)
    # 0〜100 に収める。補完度の幅を広げると100を超えることがあり、
    # 画面のバー（上限100）が振り切れて読めなくなる。
    c["買い付け優先度"] = (base * tiebreak - penalty).clip(lower=0, upper=100)
    return c


def buy_gate(c: pd.DataFrame, config: dict,
             require_capacity: bool = False) -> tuple[pd.DataFrame, pd.Series]:
    """買ってはいけないものを外す。

    順位を付ける前の足切り。安いからといって、配当が危ない銘柄や
    高配当トラップの判定が出ている銘柄に資金を入れる理由はない。

    require_capacity を立てると、増配余力の条件も重ねる。
    2026-09-15 の検証（EDINET の FY2022 起点・1,211銘柄・3年）で、
    高利回りにこれを重ねると減配率が 8.8% → 5.4% に下がり、
    リターンも +80.4% → +90.6% と上がった。
    ただしコホートが1つ・3年しかなく、かけると候補の利回りが落ちる。
    **既定では掛けない。** ケンが画面で選ぶ。
    """
    g = config["buy_priority"]
    reasons = pd.Series("", index=c.index)
    trap = (c["trap_penalty"].fillna(0) if "trap_penalty" in c.columns
            else pd.Series(0.0, index=c.index))
    bad_trap = trap >= float(g["max_trap_penalty"])
    reasons[bad_trap] = "高配当トラップの判定が出ている"
    low_health = c["health"].fillna(0) < float(g["min_health"])
    reasons[low_health & (reasons == "")] = "配当継続スコアが低い"

    if require_capacity:
        max_p = float(g.get("capacity_max_payout", 0.50))
        min_f = float(g.get("capacity_min_fcf_cover", 2.0))
        if "payout_ratio" in c.columns:
            bad = c["payout_ratio"].fillna(9.9) > max_p
            reasons[bad & (reasons == "")] = f"配当性向が{max_p:.0%}を超えている"
        if "fcf_cover" in c.columns:
            bad = c["fcf_cover"].fillna(0) < min_f
            reasons[bad & (reasons == "")] = f"FCFが配当の{min_f:.0f}倍に届かない"
    return c[reasons == ""].copy(), reasons


def month_gaps(positions: pd.DataFrame, calendar: pd.DataFrame | None) -> dict[int, float]:
    """月ごとの「受け取りの少なさ」を 0〜1 で返す。空いている月ほど1に近い。"""
    if calendar is None or calendar.empty:
        return {}
    amt = calendar.set_index("month")["amount"]
    peak = amt.max()
    if not peak:
        return {}
    return {int(m): float(1 - v / peak) for m, v in amt.items()}


def allocate(cash: float, candidates: pd.DataFrame, positions: pd.DataFrame,
             config: dict, max_names: int = 12,
             per_name_cap_pct: float = 0.25,
             require_below_target: bool = True,
             month_gap: dict[int, float] | None = None,
             allow_single_lot: bool = True,
             lot: int = _LOT,
             require_capacity: bool = False) -> pd.DataFrame:
    """入金額を候補に割り振る。

    Parameters
    ----------
    max_names : int
        1回の投入で買う銘柄数の上限。ヘムは360〜400銘柄に超分散している。
        機械的な基準で選び、監視はアプリがやるのだから、絞る理由は薄い。
    per_name_cap_pct : float
        1銘柄あたりの投入上限（今回の入金額に対する比率）
    require_below_target : bool
        目標利回りに届いている銘柄だけを対象にする
    allow_single_lot : bool
        上限に収まらなくても、1単元だけなら買うことを許す
    lot : int
        売買の刻み。100 なら単元、1 なら単元未満株（S株・かぶミニ・ワン株）。
        毎月の入金で単元を買えることは稀なので、既定は1株。

    【なぜ2周するか】※ lot=1 のときは1周目でほぼ配り切れるので効きません
    日本株は100株単位でしか買えない。1単元の値段は10万〜70万円が普通なので、
    「1銘柄あたり入金の15%まで」といった上限を素直に当てると、単元がその上限を
    超える銘柄が全部落ちる。実測では、入金50万円（投入枠30万円）に対して
    **上位25銘柄のうち買えたのは3銘柄だけ**で、¥99,430 しか配れずに
    ¥200,570 が宙に浮いていた。しかも画面にはそれが「暴落用の現金」に見えていた。

    そこで2周する。
        1周目  上限を守って、分散を優先して配る
        2周目  余ったお金で、上限を超えても1単元だけなら買う
               （ただし1銘柄が入金の single_lot_max を超えることはしない）
    """
    if candidates is None or candidates.empty or cash <= 0:
        return pd.DataFrame()

    cap_sector = config["portfolio"]["max_sector_weight"]
    total_after = (positions["eval_value"].sum() if not positions.empty else 0.0) + cash
    sector_now = (positions.groupby("sector33")["eval_value"].sum()
                  if not positions.empty else pd.Series(dtype=float))

    # ── 配当の偏りに上限をかける ──
    # 評価額の上限だけでは、インカムの偏りは止まらない。高利回りの銘柄は
    # 評価額より配当のほうが大きく偏る（実測: 景気敏感業種は評価額22.2%に対して
    # 配当29.5%）。守りたいのはインカムなので、配当側にも上限を置く。
    cap_income = float(config["portfolio"].get("max_income_weight", 1.0))
    cyc = config.get("cyclical", {})
    cyc_sectors = set(cyc.get("sectors", []))
    cap_cyc = float(cyc.get("max_income_share", 1.0))
    income_now = (positions.groupby("ticker")["annual_dividend"].sum()
                  if not positions.empty and "annual_dividend" in positions.columns
                  else pd.Series(dtype=float))
    cyc_income_now = float(
        positions.loc[positions["sector33"].isin(cyc_sectors), "annual_dividend"].sum()
        if not positions.empty and "annual_dividend" in positions.columns
        and cyc_sectors else 0.0)
    income_base = float(income_now.sum())

    c = candidates[candidates["last_close"].fillna(0) > 0].copy()
    if require_below_target and "target_yield" in c.columns:
        c = c[c["dividend_yield"].fillna(0) >= c["target_yield"].fillna(0)]
    if c.empty:
        return pd.DataFrame()

    c = buy_priority(c, positions, config, month_gap)
    c, _ = buy_gate(c, config, require_capacity=require_capacity)
    if c.empty:
        return pd.DataFrame()
    # 発掘スコアではなく買い付け優先度の順に配る
    c = c.sort_values("買い付け優先度", ascending=False)

    per_name_cap = cash * per_name_cap_pct
    single_lot_max = cash * float(config["buy_priority"].get("single_lot_max", 0.5))
    lot = max(int(lot), 1)
    # 端数で ¥297 のような細切れが1行できても、管理の手間が増えるだけで配当は増えない。
    # 1銘柄あたりの最低金額を決めて、それに満たない配分は作らない。
    min_ticket = max(float(config["buy_priority"].get("min_ticket_yen", 3000)),
                     cash * float(config["buy_priority"].get("min_ticket_pct", 0.02)))
    remaining = cash
    picks = []
    taken: set = set()

    def _try(r, cap_amount: float, relaxed: bool) -> bool:
        nonlocal remaining
        price = r["last_close"]
        lot_cost = price * lot
        if len(picks) >= max_names or remaining < lot_cost or r["ticker"] in taken:
            return False
        sector = r.get("sector33")
        used = float(sector_now.get(sector, 0.0)) + sum(
            p["投入額"] for p in picks if p["業種"] == sector)
        room = cap_sector * total_after - used
        if room <= 0 or lot_cost > room:
            return False
        shares = _lot_size(price, min(cap_amount, remaining, room), lot)
        if shares < lot:
            if not relaxed or lot_cost > min(remaining, room, single_lot_max):
                return False
            shares = lot          # 上限は超えるが1単元だけ入れる
        amount = shares * price

        # 配当の偏りの上限。買ったあとの姿で判定する。
        y = float(r.get("dividend_yield") or 0.0)
        if y > 0 and income_base > 0:
            added = sum(p["年間配当"] for p in picks)
            total_income = income_base + added + amount * y
            if total_income > 0:
                mine = float(income_now.get(r["ticker"], 0.0)) + amount * y
                if mine / total_income > cap_income:
                    # 上限に収まる金額まで減らす。単元に満たなければ見送る
                    # 増やす配当 a は (いまの配当 + a) / (ほかの配当 + a) ≤ 上限 を満たす最大値。
                    # 分母に「削る前の金額」を入れたまま解くと、上限を少し超える。
                    held = float(income_now.get(r["ticker"], 0.0))
                    room_income = (cap_income * (income_base + added) - held) / (1 - cap_income) \
                        if cap_income < 1 else amount * y
                    shares = _lot_size(price, max(0.0, room_income / y), lot)
                    if shares < lot:
                        return False
                    amount = shares * price
                if sector in cyc_sectors:
                    cyc_after = cyc_income_now + sum(
                        p["年間配当"] for p in picks if p["業種"] in cyc_sectors)
                    if (cyc_after + amount * y) / total_income > cap_cyc:
                        return False
        if amount < min_ticket and remaining > min_ticket:
            return False        # 細切れは作らない（お金が尽きかけている時だけ許す）
        picks.append(_pick(r, sector, shares, amount, cap_sector * total_after - used,
                           relaxed and amount > cap_amount, lot))
        taken.add(r["ticker"])
        remaining -= amount
        return True

    for _, r in c.iterrows():
        _try(r, per_name_cap, relaxed=False)
    if allow_single_lot and lot > 1:
        # 余ったお金で、上限を超えても1単元だけなら買う2周目
        # （1株から買えるなら1周目で配り切れるので、この2周目は要らない）
        for _, r in c.iterrows():
            if remaining <= 0 or len(picks) >= max_names:
                break
            _try(r, per_name_cap, relaxed=True)

    # ── 端数の積み増し ──
    # 1株単位で配ると、1銘柄ごとに「株価 − 1円」までの端数が残る。銘柄数の上限に
    # 達していると新しい銘柄を足せないので、実測では入金10万円のうち1.2万円、
    # 3万円のうち0.8万円が宙に浮いていた。すでに選んだ銘柄に1株ずつ積み増して埋める。
    if lot == 1 and picks and remaining > 0:
        topup_cap = per_name_cap * float(
            config["buy_priority"].get("topup_cap_multiple", 1.5))
        by_ticker = {p["ticker"]: p for p in picks}
        progressed = True
        while progressed and remaining > 0:
            progressed = False
            for _, r in c.iterrows():
                pk = by_ticker.get(r["ticker"])
                if pk is None:
                    continue
                price = pk["株価"]
                if price <= 0 or price > remaining or pk["投入額"] + price > topup_cap:
                    continue
                sector = pk["業種"]
                used = float(sector_now.get(sector, 0.0)) + sum(
                    q["投入額"] for q in picks if q["業種"] == sector)
                if price > cap_sector * total_after - used:
                    continue
                # 配当の偏りの上限は積み増しにも当てる。1周目で上限まで削った銘柄に
                # ここで1株ずつ足すと、上限を素通りしてしまう（1銘柄で配当の16%になった）。
                y = float(r.get("dividend_yield") or 0.0)
                if y > 0 and income_base > 0:
                    total_income = income_base + sum(q["年間配当"] for q in picks) + price * y
                    mine = float(income_now.get(pk["ticker"], 0.0)) + pk["年間配当"] + price * y
                    if mine / total_income > cap_income:
                        continue
                    if sector in cyc_sectors:
                        cyc_after = cyc_income_now + sum(
                            q["年間配当"] for q in picks if q["業種"] in cyc_sectors) + price * y
                        if cyc_after / total_income > cap_cyc:
                            continue
                pk["株数"] += 1
                pk["投入額"] += price
                pk["年間配当"] = pk["投入額"] * (r.get("dividend_yield") or 0)
                pk["概算手数料"] = _odd_lot_fee(pk["投入額"], pk["株数"], lot)
                remaining -= price
                progressed = True

    out = pd.DataFrame(picks)
    if not out.empty:
        out.attrs["残り"] = remaining
    return out


def _pick(r, sector, shares, amount, sector_room, relaxed: bool, lot: int) -> dict:
    return {
        "ticker": r["ticker"],
        "コード": r.get("code"),
        "銘柄名": r.get("name"),
        "業種": sector,
        "株価": r["last_close"],
        "株数": shares,
        "投入額": amount,
        "利回り": r.get("dividend_yield"),
        "年間配当": amount * (r.get("dividend_yield") or 0),
        "買い付け優先度": r["買い付け優先度"],
        "割安度": r["_割安度"],
        "自己利回り順位": r["_自己利回り順位"],
        "絶対利回り順位": r["_絶対利回り順位"],
        "継続": r["_継続"],
        "目標到達": r["_目標到達"],
        "補完度": r["_補完度"],
        "トラップ減点": r.get("trap_penalty", 0),
        "発掘スコア": r.get("total"),
        "配当月": r.get("payout_months"),
        "業種の空き枠": max(sector_room, 0.0),
        "上限を超えて1単元": relaxed,
        "概算手数料": _odd_lot_fee(amount, shares, lot),
    }


def _odd_lot_fee(amount: float, shares: int, lot: int) -> float:
    """単元未満株のおおよその手数料（スプレッド込み）。

    単元未満株は売買手数料が無料でも **約定価格に 0.2〜0.5% のスプレッドが乗る**
    ことが多い（SBIのS株は買い無料・売り0.55%、楽天のかぶミニはスプレッド0.22%）。
    無視すると小口に配るほど有利に見えてしまうので、0.22% を概算で置いておく。
    単元で買うなら手数料はほぼ無視できるので0。
    """
    return round(amount * 0.0022, 0) if lot < 100 else 0.0