"""増配期待スコア。

「市場が正しく評価していない増配期待企業」を5層＋減点で定義する。
各層は 0〜100 に正規化し、**内訳をすべて開示する**（カルテで全部見せる）。

層の意味:
    A capacity    増配を「出せる」か（配当余力・BSの厚み）
    B willingness 増配を「する気がある」か（実績・方針）★日本株の肝
    C growth      配当の元手が伸びているか
    D neglect     市場に見過ごされているか（カバー不足・PBR1倍割れ）
    E valuation   その銘柄自身の過去と比べて割安か
    F trap        高配当トラップの減点

正規化はゲート通過銘柄の中でのパーセンタイル順位を使う。絶対値のスケーリングは
業種や市場環境で意味が変わるうえ、閾値を勘で決めることになるため使わない。
ただし「配当性向のスイートスポット」のように絶対値に意味がある指標だけは
絶対評価のカーブを当てる。

重みは config.yaml。初期値は全層 1.0 の等ウェイトで、
scripts/validate_score.py の検証を通った層だけ重みを上げる。勘で動かさない。
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

LAYERS = ["capacity", "willingness", "growth", "neglect", "valuation"]

# 配当継続スコア（health）に使う層。
# 発掘スコア（total）には「見過ごされ度」と「割安」が入っているが、これは
# 「市場がまだ気づいていないか」を測る指標であって、「配当が続くか」ではない。
# 実測: 三菱商事の見過ごされ度は8点、JR東日本は7点、NTTは19点。大型株は
# 定義上ここで沈む。この total をそのまま保有の評価に使うと、累進配当で
# 10年連続増配している三菱商事が「スコアが低い」という理由だけで整理候補に並ぶ。
# 保有を評価するときは、配当の余力・意思・原資だけを見る。
HEALTH_LAYERS = ["capacity", "willingness", "growth"]


# ──────────────────────────────── 正規化 ────────────────────────────────

def pct_rank(s: pd.Series, ascending: bool = True) -> pd.Series:
    """0〜100 のパーセンタイル順位。欠損は 50（中立）に置く。

    欠損を 0 にすると「データが無い」と「最悪」が区別できなくなり、
    データ欠損の多い小型株が不当に沈む。中立に置くのが正しい。
    """
    r = s.rank(pct=True, ascending=ascending, na_option="keep") * 100
    return r.fillna(50.0)


def payout_curve(payout: pd.Series, low: float, high: float, too_low: float) -> pd.Series:
    """配当性向のスイートスポット評価。

    低いほど良いわけではない。15%未満は「還元する気がない」として減点し、
    30〜50% を満点、そこから上下に離れるほど落とす。
    """
    def score(x):
        if pd.isna(x) or x < 0:
            return 50.0
        if x < too_low:
            return 100.0 * (x / too_low) * 0.5          # 0〜50 点
        if x < low:
            return 50.0 + 50.0 * (x - too_low) / (low - too_low)
        if x <= high:
            return 100.0
        if x >= 1.0:
            return 0.0
        return 100.0 * (1.0 - (x - high) / (1.0 - high))
    return payout.map(score)


# ──────────────────────────────── ゲート ────────────────────────────────

def apply_gate(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """ゲート判定。通らない理由を必ず文字列で残す（なぜ落ちたか説明できるように）。

    財務の足切りは2本立てにしている。自己資本比率30%の一本槍だと、
    通信・鉄道・電力といった設備産業が丸ごと落ちる（実測: NTT の自己資本比率は
    20.8%）。日本の配当株投資でこれらを捨てるのは本末転倒なので、
    「自己資本が厚い」か「レバレッジは高いが営業CFで十分返せる」かの
    どちらかを満たせば通す。
    """
    g = config["gate"]
    fin_sectors = set(config["universe"]["financial_sectors"])
    reasons = pd.Series([[] for _ in range(len(df))], index=df.index)

    def fail(mask: pd.Series, label: str) -> None:
        mask = mask.fillna(False)
        for i in df.index[mask]:
            reasons[i].append(label)

    is_fin = df["sector33"].isin(fin_sectors)

    # 財務を取りに行っていない銘柄を「基準未満」と書くと、落選理由が読めなくなる
    # （実測で 3,079 銘柄が「財務が基準未満」と表示され、その大半は単に
    #  粗いふるいの段階で対象外になっていただけだった）。データ欠如は別扱いにする。
    no_fund = df["equity_ratio"].isna() & df["net_income"].isna()
    fail(no_fund, "財務未取得（粗いふるいで対象外）")

    # 流動性・規模・上場年数
    fail(df["listing_years"] < g["min_listing_years"], f"上場{g['min_listing_years']}年未満")
    fail(df["market_cap_oku"] < g["min_market_cap_oku"], f"時価総額{g['min_market_cap_oku']}億未満")
    fail(df["avg_turnover_man"] < g["min_avg_turnover_man"], f"売買代金{g['min_avg_turnover_man']}万未満")

    # 配当実績
    fail(df["years_paying"] < g["min_dividend_years"], f"配当実績{g['min_dividend_years']}年未満")
    fail(df["cuts_10y"] > g["max_dividend_cuts_10y"], f"10年で減配{g['max_dividend_cuts_10y']}回超")
    fail(df["dps_latest"] <= 0, "無配")

    # 収益の安定（財務が無い銘柄には重ねて表示しない）
    has_fund = ~no_fund
    fail(has_fund & (df["n_periods"] < g["min_fiscal_periods"]), "決算データが少なく判定不能")
    ocf_ratio = df["positive_ocf_years"] / df["n_periods"].replace(0, np.nan)
    fail(has_fund & (ocf_ratio < g["min_positive_ocf_ratio"]), "営業CFがマイナスの期が多い")
    fail(has_fund & (df["loss_years"] > g["max_loss_years_5y"]), "赤字期が多い")

    # 財務の健全性（設備産業を落とさないよう2本立て）
    equity_ok = df["equity_ratio"] >= g["min_equity_ratio"]
    leveraged_ok = (
        (df["equity_ratio"] >= g["min_equity_ratio_leveraged"])
        & (df["net_debt_to_ocf"] <= g["max_net_debt_to_ocf"])
    )
    fin_ok = df["equity_ratio"] >= g["min_equity_ratio_financial"]
    bs_ok = np.where(is_fin, fin_ok.fillna(False), (equity_ok | leveraged_ok).fillna(False))
    fail(has_fund & ~pd.Series(bs_ok, index=df.index), "財務が基準未満")

    # 配当の持続性
    fail(has_fund & (df["payout_ratio"] > g["max_payout_ratio"]),
         f"配当性向{g['max_payout_ratio']:.0%}超")
    fail(has_fund & (df["fcf_payout_ratio"] > g["max_fcf_payout_ratio"]),
         "FCFで配当を賄えていない")
    # 財務はあるのに配当性向が計算できない＝赤字などで判定不能
    fail(has_fund & df["payout_ratio"].isna(), "配当性向が計算できない（赤字等）")

    out = df.copy()
    out["gate_reason"] = reasons.map(lambda xs: " / ".join(xs))
    out["gate_passed"] = out["gate_reason"] == ""
    return out


# ──────────────────────────────── 各層 ────────────────────────────────

def score_capacity(df: pd.DataFrame, config: dict) -> tuple[pd.Series, dict]:
    s = config["scoring"]
    parts = {
        "配当性向の位置": payout_curve(df["payout_ratio"], s["payout_ideal_low"],
                                  s["payout_ideal_high"], s["payout_too_low"]),
        # ヘムの「配当性向 < 配当利回り×10」を連続値にしたもの。高利回りと
        # 低配当性向を同時に要求するので、割安さと増配余力が分離されない。
        "ヘム指数（利回り×10÷配当性向）": pct_rank(df.get("hem_ratio")),
        "FCF配当カバー率": pct_rank(df["fcf_cover"]),
        "ネットキャッシュ比率": pct_rank(df["net_cash_ratio"]),
        "有利子負債の軽さ": pct_rank(df["net_debt_to_ocf"], ascending=False),
    }
    return pd.DataFrame(parts).mean(axis=1), parts


def score_willingness(df: pd.DataFrame, config: dict) -> tuple[pd.Series, dict]:
    # 減配なしの継続年数は「方針の固さ」そのもの。連続増配より緩い基準だが、
    # 累進配当（減らさない）を掲げる企業を拾えるので日本株では重要。
    policy = df.get("policy_bonus", pd.Series(0.0, index=df.index)).fillna(0.0)
    parts = {
        "連続増配年数": pct_rank(df["streak"]),
        "減配なし継続年数": pct_rank(df["streak_no_cut"]),
        "DPS 5年成長": pct_rank(df["cagr_5y"]),
        "DPS 10年成長": pct_rank(df["cagr_10y"]),
        # 有報の「配当政策」から検出した方針。累進配当・DOE・配当性向目標を
        # 掲げている会社は、利益が一時的に落ちても配当を維持する圧力が働く。
        "配当方針の明示": policy * 100.0,
    }
    return pd.DataFrame(parts).mean(axis=1), parts


def score_growth(df: pd.DataFrame, config: dict) -> tuple[pd.Series, dict]:
    parts = {
        "純利益の伸び": pct_rank(df["ni_cagr"]),
        "営業CFの伸び": pct_rank(df["ocf_cagr"]),
        "ROE": pct_rank(df["roe"]),
        "営業利益率": pct_rank(df["operating_margin"]),
    }
    return pd.DataFrame(parts).mean(axis=1), parts


def score_neglect(df: pd.DataFrame, config: dict) -> tuple[pd.Series, dict]:
    # 規模が小さく・薄く・機関投資家に持たれていないほど、アナリストのカバーが
    # 薄い＝価格に情報が織り込まれていない可能性が高い。日本市場で構造的に
    # 残り続けている数少ない非効率がここ。
    pbr_below1 = (df["pbr"] < 1.0).fillna(False)
    roe_improving = (df["ni_cagr"] > 0).fillna(False)
    # 東証の資本効率改善要請が最も効く位置＝PBR1倍割れ かつ 利益が伸びている
    tse_pressure = (pbr_below1 & roe_improving).astype(float) * 100.0
    parts = {
        "規模の小ささ": pct_rank(df["market_cap_oku"], ascending=False),
        "売買代金の薄さ": pct_rank(df["avg_turnover_man"], ascending=False),
        "機関投資家保有の低さ": pct_rank(df["held_pct_institutions"], ascending=False),
        "PBR1倍割れ×増益": tse_pressure,
    }
    return pd.DataFrame(parts).mean(axis=1), parts


def score_valuation(df: pd.DataFrame, config: dict) -> tuple[pd.Series, dict]:
    # 自己ヒストリカル利回りパーセンタイルが中心。絶対利回りは業種特性に
    # 引きずられるので順位付けの主役にはしない。
    parts = {
        "自己利回りの高さ": (df["yield_percentile"] * 100).fillna(50.0),
        "業種内の利回り順位": pct_rank(df.groupby("sector33")["dividend_yield"].rank(pct=True)),
        "PER の低さ": pct_rank(df["per"].where(df["per"] > 0), ascending=False),
        "PBR の低さ": pct_rank(df["pbr"].where(df["pbr"] > 0), ascending=False),
    }
    return pd.DataFrame(parts).mean(axis=1), parts


# ──────────────────────────────── 合成 ────────────────────────────────

def compute_health(df: pd.DataFrame, config: dict,
                   trap_penalty: pd.Series | None = None) -> pd.DataFrame:
    """配当継続スコアを、財務が取れている全銘柄について計算する。

    ゲートを通った銘柄だけに付けると、保有114銘柄のうち83銘柄が空欄になり
    「持っている株を評価する」という目的を果たせない。基準を外れている銘柄こそ
    「どれくらい危ないのか」を数字で知りたい。

    順位の母集団は「財務が取れている全銘柄」。ゲート通過銘柄だけを母集団にすると、
    基準を外れた銘柄が全員100点満点の外側に出てしまい比較にならない。
    """
    pool = df[df["net_income"].notna() | df["equity_ratio"].notna()].copy()
    if pool.empty:
        return pd.DataFrame()
    layers = {}
    for name, fn in (("capacity", score_capacity), ("willingness", score_willingness),
                     ("growth", score_growth)):
        layers[name], _ = fn(pool, config)
    lf = pd.DataFrame(layers)
    penalty = (trap_penalty.reindex(pool.index).fillna(0.0)
               if trap_penalty is not None else pd.Series(0.0, index=pool.index))
    out = pd.DataFrame({"health": (lf[HEALTH_LAYERS].mean(axis=1) - penalty).clip(lower=0.0)})
    # 基準を外れた銘柄にも実数を残す。ケンの保有114銘柄のうち83銘柄は基準を外れて
    # いるので、ここに実数が無いとカルテが空になり、売り判定も「判定待ち」になる。
    # 発掘スコアの内訳（層ごとの順位）は付けない。あれはゲート通過銘柄の中での
    # 順位なので、母集団の違う銘柄に付けると意味の違う数字になる。
    out["raw_json"] = [
        json.dumps({"raw": {k: _round(pool.loc[t, k]) for k in _RAW_KEYS
                            if k in pool.columns}}, ensure_ascii=False)
        for t in pool.index
    ]
    return out


def compute_scores(df: pd.DataFrame, config: dict, trap_penalty: pd.Series | None = None,
                   trap_detail: pd.Series | None = None) -> pd.DataFrame:
    """ゲート通過銘柄に発掘スコアを付ける。

    Parameters
    ----------
    df : DataFrame
        index=ticker。必要な列は apply_gate と各 score_* が参照するもの。
    """
    gated = df[df["gate_passed"]].copy()
    if gated.empty:
        return pd.DataFrame()

    w = config["weights"]
    layer_scores, details = {}, {}
    for name, fn in (("capacity", score_capacity), ("willingness", score_willingness),
                     ("growth", score_growth), ("neglect", score_neglect),
                     ("valuation", score_valuation)):
        s, parts = fn(gated, config)
        layer_scores[name] = s
        details[name] = pd.DataFrame(parts)

    layers = pd.DataFrame(layer_scores)
    weights = pd.Series({k: float(w[k]) for k in LAYERS})
    total = (layers[LAYERS] * weights).sum(axis=1) / weights.sum()

    penalty = (trap_penalty.reindex(gated.index).fillna(0.0)
               if trap_penalty is not None else pd.Series(0.0, index=gated.index))
    total = (total - penalty * float(w["trap_penalty"])).clip(lower=0.0)

    out = layers.copy()
    out["trap_penalty"] = penalty
    out["total"] = total
    out["gate_passed"] = 1

    # 内訳は JSON にして丸ごと残す。カルテで「なぜこの点なのか」を全部見せるため。
    detail_rows = []
    for t in gated.index:
        rec = {layer: {k: _round(details[layer].loc[t, k]) for k in details[layer].columns}
               for layer in details}
        rec["raw"] = {k: _round(gated.loc[t, k]) for k in _RAW_KEYS if k in gated.columns}
        if trap_detail is not None and t in trap_detail.index:
            rec["traps"] = trap_detail.loc[t]
        detail_rows.append(json.dumps(rec, ensure_ascii=False))
    out["detail_json"] = detail_rows
    return out


_RAW_KEYS = ["dividend_yield", "yield_percentile", "dps_latest", "hem_ratio",
             "policy_bonus", "edinet_years", "eps_cagr", "ex_dividend_date", "payout_months",
             "streak", "streak_no_cut",
             "cuts_10y", "cagr_5y", "cagr_10y", "payout_ratio", "fcf_cover",
             "net_cash_ratio", "net_debt_to_ocf", "equity_ratio", "roe", "per", "pbr",
             "market_cap_oku", "avg_turnover_man", "ni_cagr", "ocf_cagr", "pos_52w",
             "operating_margin", "held_pct_institutions", "fcf_payout_ratio",
             "payout_basis", "payout_ratio_prev", "policy_flags"]


def _round(v):
    if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
        return None
    if isinstance(v, (int, np.integer)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        return round(float(v), 4)
    return v
