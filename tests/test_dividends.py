"""配当履歴: 年度の切り方・増配/減配の数え方・分割の当て直し。"""
from __future__ import annotations

import pandas as pd

from modules.bulk_fetch import fix_same_day_split_dividends
from modules.dividend_history import annual_dps, build_profile, flag_spikes, profiles_to_frame


def _div(rows, ticker="1.T"):
    return pd.DataFrame([{"ticker": ticker, "date": d, "amount": a} for d, a in rows])


def test_march_fy_is_not_split_by_drifting_ex_dates():
    """1332 日本水産: 権利落ち日が 3/30→3/28 とずれても、同じ年度に3回入らない。"""
    d = _div([("2022-09-29", 9), ("2023-03-30", 10), ("2023-09-28", 10), ("2024-03-28", 14),
              ("2024-09-27", 12), ("2025-03-28", 16)])
    a = annual_dps(d)
    assert a["n_payments"].tolist() == [2, 2, 2]
    assert a["dps"].tolist() == [19, 24, 28]


def test_cut_off_at_interim_does_not_shift_fiscal_year():
    full = _div([(f"{y}-03-28", 10 + y - 2015) for y in range(2015, 2025)]
                + [(f"{y}-09-27", 10 + y - 2015) for y in range(2015, 2025)])
    a = annual_dps(full)
    assert (a["n_payments"] == 2).all()
    assert (a["dps"].diff().dropna() > 0).all()      # 毎年増配として読める


def test_streak_and_cuts():
    d = _div([(f"{y}-03-28", v) for y, v in
              zip(range(2012, 2025), [10, 11, 12, 12, 13, 9, 10, 11, 12, 13, 14, 15, 16])])
    p = build_profile("1.T", d)
    assert p.cuts_10y == 1 and p.cuts_all == 1
    assert p.streak == 7
    assert p.dps_latest == 16


def test_commemorative_spike_is_flagged():
    dps = pd.Series([10.0, 11.0, 20.0, 12.0, 13.0], index=range(2020, 2025))
    assert flag_spikes(dps).tolist() == [False, False, True, False, False]
    d = _div([(f"{y}-03-28", v) for y, v in zip(range(2016, 2025),
                                                 [8, 9, 10, 11, 20, 12, 13, 14, 15])])
    p = build_profile("1.T", d)
    assert p.has_spike and p.cuts_10y == 0


def test_empty_profiles_frame_keeps_columns():
    f = profiles_to_frame({})
    assert "dps_latest" in f.columns and f.empty


def test_same_day_split_dividend_is_divided():
    """日本製鉄 5401: 2025-09-29 に 5:1 分割、同じ日の配当 60円 が調整されずに入っていた。"""
    div = _div([("2024-03-28", 16), ("2024-09-27", 16), ("2025-03-28", 12),
                ("2025-09-29", 60), ("2026-03-30", 12)], "5401.T")
    splits = pd.DataFrame([{"ticker": "5401.T", "date": "2025-09-29", "ratio": 5.0}])
    out, changed = fix_same_day_split_dividends(div, splits)
    assert out.loc[out["date"] == "2025-09-29", "amount"].item() == 12
    assert len(changed) == 1


def test_same_day_split_leaves_real_increase_alone():
    """分割と同じ日でも、比率ぶん大きくなっていなければ触らない（記念配当・本物の増配）。"""
    div = _div([("2024-03-28", 40), ("2024-09-27", 40), ("2025-03-28", 42),
                ("2025-09-29", 45), ("2026-03-30", 46)], "9433.T")
    splits = pd.DataFrame([{"ticker": "9433.T", "date": "2025-09-29", "ratio": 2.0}])
    out, changed = fix_same_day_split_dividends(div, splits)
    assert changed.empty and out["amount"].tolist() == div["amount"].tolist()
