"""東証の営業日と、売りの判定。"""
from __future__ import annotations

import json
from datetime import date

import pandas as pd
import pytest

import modules.store as S
from modules import sell_rules as SR
from modules.config import load_config
from modules.jp_calendar import is_trading_day, last_trading_day, shift_trading_days


@pytest.mark.parametrize("d, ok", [
    (date(2023, 1, 4), True),     # 元日が日曜の年の大発会
    (date(2034, 1, 4), True),
    (date(2023, 1, 2), False),    # 振替休日（東証も休み）
    (date(2026, 5, 6), False),    # 5/3 日曜の振替
    (date(2026, 9, 22), False),   # 敬老の日と秋分の日に挟まれた国民の休日
    (date(2026, 12, 30), True),   # 大納会
    (date(2026, 12, 31), False),
    (date(2026, 9, 28), True),
])
def test_trading_days(d, ok):
    assert is_trading_day(d) is ok


def test_shift_and_month_end():
    assert last_trading_day(2026, 12) == date(2026, 12, 30)
    assert shift_trading_days(date(2026, 9, 18), 1) == date(2026, 9, 24)   # 連休をまたぐ


def _pos(**kw):
    base = {"account": "specific", "ticker": "1.T", "name": "テスト", "sector33": "卸売業",
            "eval_value": 1_000_000.0, "pnl_pct": 0.1, "streak": 5, "cuts_10y": 0,
            "streak_no_cut": 5, "health": 60.0, "current_yield": 0.04, "target_yield": 0.04,
            "detail_json": json.dumps({"raw": {"payout_ratio": 0.4}})}
    base.update(kw)
    other = dict(base, ticker="2.T", name="その他", eval_value=9_000_000.0)
    return pd.DataFrame([base, other])


def _fund(local_db, ni, ocf):
    rows = [{"ticker": "1.T", "fiscal_end": f"{2025 - i}-03-31", "net_income": n,
             "operating_cf": o} for i, (n, o) in enumerate(zip(ni, ocf))]
    S.upsert_df("fundamentals", pd.DataFrame(rows), ["ticker", "fiscal_end", "net_income",
                                                     "operating_cf"])


def _first(ev):
    return ev[ev["ticker"] == "1.T"].iloc[0] if "ticker" in ev.columns else ev.iloc[0]


def test_just_cut_is_sell_candidate(local_db):
    _fund(local_db, [1e9] * 4, [1e9] * 4)
    ev = SR.evaluate(_pos(cuts_10y=1, streak_no_cut=0, streak=0), load_config())
    assert _first(ev)["重さ"] == "売却を検討"


def test_payout_over_100_after_profit_collapse_is_explained(local_db):
    _fund(local_db, [1e8, 3e9, 3e9, 3e9], [1e9] * 4)
    detail = json.dumps({"raw": {"payout_ratio": 7.0}})
    ev = SR.evaluate(_pos(detail_json=detail, streak=2), load_config())
    r = _first(ev)
    assert r["重さ"] == "売却を検討" and "落ち込んだ" in r["理由"]


def test_negative_cf_is_only_watch_and_not_for_banks(local_db):
    _fund(local_db, [1e9] * 4, [-1e9, -1e9, 1e9, 1e9])
    cfg = load_config()
    assert _first(SR.evaluate(_pos(), cfg))["重さ"] == "監視を強める"
    ev = SR.evaluate(_pos(sector33="銀行業"), cfg)
    assert "1.T" not in set(ev.get("ticker", pd.Series(dtype=str))) \
        or _first(ev)["重さ"] != "監視を強める"


def test_missing_data_is_pending(local_db):
    ev = SR.evaluate(_pos(detail_json="{}"), load_config())
    assert _first(ev)["重さ"] == "判定待ち"
    assert SR.summary(ev)["判定待ち"] >= 1


def _raw(**kw):
    return json.dumps({"raw": {"payout_ratio": 0.4, **kw}}, ensure_ascii=False)


def test_progressive_dividend_past_cuts_are_not_a_sell_reason(local_db):
    """累進配当を掲げ、直近3年は減配なし → 過去の減配回数・低スコアで売りを勧めない。"""
    _fund(local_db, [1e9] * 4, [1e9] * 4)
    pol = json.dumps({"累進配当": True}, ensure_ascii=False)
    ev = SR.evaluate(_pos(cuts_10y=4, streak_no_cut=5, health=20.0,
                          detail_json=_raw(policy_flags=pol)), load_config())
    mine = ev[ev["ticker"] == "1.T"] if not ev.empty else ev
    assert mine.empty or (mine.iloc[0]["重さ"] not in ("売却を検討", "監視を強める")
                          and "減配" not in mine.iloc[0]["理由"])


def test_same_past_cuts_without_policy_are_watch_not_sell(local_db):
    """方針が無くても、直近3年に減配が無ければ過去の回数だけで🔴にはしない。"""
    _fund(local_db, [1e9] * 4, [1e9] * 4)
    ev = SR.evaluate(_pos(cuts_10y=4, streak_no_cut=5), load_config())
    assert _first(ev)["重さ"] == "監視を強める"


def test_progressive_dividend_actual_cut_is_still_sell(local_db):
    _fund(local_db, [1e9] * 4, [1e9] * 4)
    pol = json.dumps({"累進配当": True}, ensure_ascii=False)
    ev = SR.evaluate(_pos(cuts_10y=1, streak_no_cut=0, streak=0,
                          detail_json=_raw(policy_flags=pol)), load_config())
    r = _first(ev)
    assert r["重さ"] == "売却を検討" and "掲げているのに" in r["理由"]


def test_doe_payout_over_100_is_watch(local_db):
    _fund(local_db, [1e9, 1.5e9, 1.5e9, 1.5e9], [1e9] * 4)
    pol = json.dumps({"DOE（純資産配当率）": True}, ensure_ascii=False)
    ev = SR.evaluate(_pos(detail_json=_raw(payout_ratio=1.2, policy_flags=pol), streak=2),
                     load_config())
    assert _first(ev)["重さ"] == "監視を強める"
