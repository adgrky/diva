"""store.py: パソコンとクラウドの振り分け、書き込みの安全性。"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

import modules.store as S


def test_local_only_mode_writes_to_file(local_db):
    S.upsert_df("holdings", pd.DataFrame([{"account": "nisa", "ticker": "1111.T", "shares": 100}]),
                ["account", "ticker", "shares"])
    assert S.read_df("SELECT shares FROM holdings").shares.tolist() == [100]


def test_cloud_tables_go_to_cloud_and_prices_stay_local(turso):
    S.upsert_df("holdings", pd.DataFrame([{"account": "nisa", "ticker": "1111.T", "shares": 100}]),
                ["account", "ticker", "shares"])
    S.upsert_df("prices", pd.DataFrame([{"ticker": "1111.T", "date": "2026-09-25", "close": 500.0}]),
                ["ticker", "date", "close"])
    assert turso.rows("SELECT ticker, shares FROM holdings") == [("1111.T", 100.0)]
    assert turso.rows("SELECT name FROM sqlite_master WHERE name='prices'") == []
    with S.local_only():
        assert S.read_df("SELECT * FROM holdings").empty          # パソコン側には来ていない
        assert len(S.read_df("SELECT * FROM prices")) == 1


def test_every_cloud_table_exists_in_cloud_ddl():
    """CLOUD_TABLES に足したのに _DDL_CLOUD を忘れると、クラウドで no such table になる。"""
    import re
    created = set(re.findall(r"CREATE TABLE IF NOT EXISTS (\w+)", S._DDL_CLOUD))
    assert S.CLOUD_TABLES == created


def test_local_only_guard_keeps_cloud_untouched(turso):
    """audit.py の書き込み検算がクラウドの本物の保有を売却扱いにしていた不具合の再発防止。"""
    S.upsert_df("holdings", pd.DataFrame([{"account": "nisa", "ticker": "1111.T", "shares": 100}]),
                ["account", "ticker", "shares"])
    before = turso.calls
    with S.local_only():
        with S.connect() as conn:
            conn.execute("DELETE FROM holdings")
            conn.execute("INSERT INTO transactions (date, type, memo) VALUES ('2026-01-01','sell','監査')")
        S.upsert_df("holdings", pd.DataFrame([{"account": "x", "ticker": "9999.T", "shares": 1}]),
                    ["account", "ticker", "shares"])
    assert turso.calls == before
    assert turso.rows("SELECT ticker FROM holdings") == [("1111.T",)]
    assert turso.rows("SELECT COUNT(*) FROM transactions") == [(0,)]


def test_audit_write_check_does_not_touch_cloud(turso, monkeypatch):
    """audit_writes を丸ごと走らせても、クラウドの保有・売買・判断は1行も変わらない。"""
    S.upsert_df("holdings", pd.DataFrame([{"account": "nisa", "ticker": "1111.T", "name": "本物",
                                           "shares": 100, "avg_cost": 900.0}]),
                ["account", "ticker", "name", "shares", "avg_cost"])
    S.upsert_df("holding_review", pd.DataFrame([{"account": "nisa", "ticker": "1111.T",
                                                 "decision": "keep"}]),
                ["account", "ticker", "decision"])
    S.upsert_df("dividends", pd.DataFrame({"ticker": "1111.T",
                                           "date": [f"{y}-03-28" for y in range(2018, 2026)],
                                           "amount": [10.0 + y for y in range(8)]}),
                ["ticker", "date", "amount"])
    S.upsert_df("quotes", pd.DataFrame([{"ticker": "1111.T", "last_close": 1200.0}]),
                ["ticker", "last_close"])
    snap = lambda: (turso.rows("SELECT * FROM holdings"), turso.rows("SELECT * FROM transactions"),
                    turso.rows("SELECT * FROM holding_review"))
    before = snap()
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location(
        "audit", Path(__file__).resolve().parent.parent / "scripts" / "audit.py")
    audit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(audit)
    audit.audit_writes()
    assert snap() == before
    assert not S._force_local


def test_upsert_drops_infinity_and_nan(local_db):
    df = pd.DataFrame([{"ticker": "1.T", "asof": "2026-09-25", "total": math.inf, "health": np.nan}])
    S.upsert_df("scores", df, ["ticker", "asof", "total", "health"])
    r = S.read_df("SELECT total, health FROM scores")
    assert r.total.isna().all() and r.health.isna().all()


def test_turso_args_keep_numpy_numbers_numeric(turso):
    """np.int64 を文字列で送ると TEXT で保存され、数値の比較が効かなくなる。"""
    with S.connect() as conn:
        conn.execute("INSERT INTO holdings (account, ticker, shares, avg_cost) VALUES (?,?,?,?)",
                     ("nisa", "1.T", np.int64(100), np.float64(12.5)))
        conn.execute("INSERT INTO scores (ticker, asof, gate_passed, total) VALUES (?,?,?,?)",
                     ("1.T", "2026-09-25", np.bool_(True), float("inf")))
    assert turso.rows("SELECT typeof(shares), typeof(avg_cost) FROM holdings") == [("real", "real")]
    assert turso.rows("SELECT gate_passed, total FROM scores") == [(1, None)]


def test_turso_rows_behave_like_sqlite_rows(turso):
    S.upsert_df("holdings", pd.DataFrame([{"account": "nisa", "ticker": "1.T", "shares": 3}]),
                ["account", "ticker", "shares"])
    with S.connect() as conn:
        row = conn.execute("SELECT account, shares FROM holdings").fetchone()
        assert row["shares"] == 3 and row[0] == "nisa"
        assert dict(row) == {"account": "nisa", "shares": 3}
        cur = conn.execute("INSERT INTO alerts (detected_at, ticker) VALUES ('d', '1.T')")
        assert cur.lastrowid == 1


def test_cloud_upsert_is_chunked_small(turso):
    n = 120
    df = pd.DataFrame({"ticker": [f"{i}.T" for i in range(n)], "code": [str(i) for i in range(n)]})
    before = turso.calls
    assert S.upsert_df("universe", df, ["ticker", "code"]) == n
    assert turso.calls - before == 3      # 50行ずつ
    assert turso.rows("SELECT COUNT(*) FROM universe") == [(n,)]


def test_batch_is_all_or_nothing_on_cloud(turso):
    S.upsert_df("holdings", pd.DataFrame([{"account": "nisa", "ticker": "1.T", "shares": 10}]),
                ["account", "ticker", "shares"])
    turso.fail_on = "UPDATE holdings"
    with pytest.raises(Exception):
        with S.connect() as conn:
            S.run_batch(conn, [
                ("INSERT INTO transactions (date, ticker, type, shares) VALUES (?,?,?,?)",
                 ("2026-09-25", "1.T", "sell", 4)),
                ("UPDATE holdings SET shares=? WHERE ticker=?", (6, "1.T")),
            ])
    assert turso.rows("SELECT COUNT(*) FROM transactions") == [(0,)]
    assert turso.rows("SELECT shares FROM holdings") == [(10.0,)]


def test_batch_commits_on_cloud(turso):
    with S.connect() as conn:
        S.run_batch(conn, [
            ("INSERT INTO transactions (date, ticker, type) VALUES (?,?,?)", ("d", "1.T", "sell")),
            ("INSERT INTO transactions (date, ticker, type) VALUES (?,?,?)", ("d", "2.T", "sell")),
        ])
    assert turso.rows("SELECT COUNT(*) FROM transactions") == [(2,)]


def test_batch_refuses_mixing_cloud_and_local(turso):
    with pytest.raises(ValueError):
        with S.connect() as conn:
            S.run_batch(conn, [("DELETE FROM holdings", ()), ("DELETE FROM prices", ())])


def test_turso_refuses_plain_http():
    with pytest.raises(ValueError):
        S._TursoHttpConn("http://example.com", "t")
    assert S._TursoHttpConn("libsql://x.turso.io", "t")._base == "https://x.turso.io"
