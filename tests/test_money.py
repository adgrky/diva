"""お金の計算: 書式・NISA・資金配分。"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from modules import allocator as A
from modules import nisa as N
from modules.config import load_config
from modules.format import pct, to_pct, yen, yen_short


@pytest.fixture(scope="module")
def cfg():
    return load_config()


def test_format():
    assert yen(1234567) == "¥1,234,567"
    assert yen(float("nan")) == "—" and pct(None) == "—"
    assert pct(0.0595, 2) == "5.95%"
    assert yen_short(-230_000) == "-23万円"
    assert yen_short(3.2e8) == "3億円"
    assert to_pct(0.0595) == pytest.approx(5.95)       # 100分の1表示の再発防止
    assert to_pct(pd.Series([0.1234]), 1).tolist() == [12.3]


def _pos(rows):
    return pd.DataFrame(rows, columns=["account", "ticker", "shares", "cost_value", "eval_value",
                                       "annual_dividend", "sector33"])


def test_nisa_room_counts_book_value():
    p = _pos([("nisa", "1.T", 100, 3_000_000, 5_000_000, 150_000, "a"),
              ("specific", "2.T", 100, 1_000_000, 1_000_000, 40_000, "b")])
    r = N.room(p)
    assert r["使った枠"] == 3_000_000 and r["残りの枠"] == 9_000_000
    assert r["最短何年"] == 4
    assert r["特定の税"] == pytest.approx(40_000 * N.TAX)


def test_nisa_plan_new_buys_first_then_moves_within_annual_cap():
    p = _pos([("specific", "1.T", 1000, 1_000_000, 2_000_000, 100_000, "a"),
              ("specific", "2.T", 1000, 3_000_000, 3_000_000, 150_000, "b")])
    plan = N.plan(p, monthly_deposit=100_000, horizon=20)
    assert plan["今年の新規買いに使う枠"] == 1_200_000
    assert plan["今年の移し替えに使える枠"] == 1_200_000
    moves = plan["移す銘柄"]
    assert moves["使う枠"].sum() <= 1_200_000 + 1e-6
    # 含み益ゼロの 2.T のほうが枠あたりの得が大きいので先に入る
    assert moves.iloc[0]["ticker"] == "2.T" and bool(moves.iloc[0]["一部だけ"])


def _cands(n=8, price=1000.0, sector=None):
    return pd.DataFrame({
        "ticker": [f"{i}.T" for i in range(n)], "code": [str(i) for i in range(n)],
        "name": [f"c{i}" for i in range(n)],
        "sector33": sector or [f"s{i % 4}" for i in range(n)],
        "last_close": price, "dividend_yield": np.linspace(0.04, 0.06, n),
        "target_yield": 0.035, "yield_percentile": 0.7, "health": 70.0,
        "trap_penalty": 0.0, "payout_months": "3月・9月", "total": 60.0})


def test_allocate_never_spends_more_than_cash(cfg):
    for lot in (1, 100):
        out = A.allocate(300_000, _cands(price=500.0), pd.DataFrame(), cfg, lot=lot)
        assert not out.empty
        assert out["投入額"].sum() <= 300_000 + 1e-6
        assert (out["株数"] % lot == 0).all()
        assert out.attrs["残り"] == pytest.approx(300_000 - out["投入額"].sum())


def test_allocate_skips_traps_and_low_health(cfg):
    c = _cands()
    c.loc[0, "trap_penalty"] = 50
    c.loc[1, "health"] = 10
    out = A.allocate(300_000, c, pd.DataFrame(), cfg, lot=1)
    assert not {"0.T", "1.T"} & set(out["ticker"])


def test_allocate_respects_sector_cap(cfg):
    pos = pd.DataFrame({"ticker": ["x.T"], "sector33": ["s0"], "eval_value": [300_000.0],
                        "annual_dividend": [12_000.0]})
    import copy
    c2 = copy.deepcopy(cfg)
    c2["portfolio"]["max_income_weight"] = 1.0     # 業種の上限だけを見る
    out = A.allocate(1_700_000, _cands(sector=["s0"] * 8), pos, c2, lot=1)
    cap = cfg["portfolio"]["max_sector_weight"] * 2_000_000
    assert not out.empty
    assert 300_000 + out["投入額"].sum() <= cap + 1e-6


def test_allocate_topup_respects_income_cap(cfg):
    """1株単位の端数の積み増しでも、1銘柄の配当の上限を超えない。"""
    cap = cfg["portfolio"]["max_income_weight"]
    pos = pd.DataFrame({"ticker": [f"h{i}.T" for i in range(20)], "sector33": ["zz"] * 20,
                        "eval_value": [500_000.0] * 20, "annual_dividend": [2_000.0] * 20})
    c = _cands(n=2, price=100.0)
    out = A.allocate(400_000, c, pos, cfg, lot=1, max_names=2)
    assert not out.empty
    total_income = 40_000 + out["年間配当"].sum()
    for _, r in out.iterrows():
        assert r["年間配当"] / total_income <= cap + 1e-6, r["ticker"]
