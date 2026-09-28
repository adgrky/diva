"""scripts/repair_audit_leak.py: 検算が本物のクラウドに残したものを元に戻せるか。"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd

import modules.store as S


def _load():
    spec = importlib.util.spec_from_file_location(
        "repair", Path(__file__).resolve().parent.parent / "scripts" / "repair_audit_leak.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _leaked_state(turso):
    # パソコンのDB: クラウド同期より前の、正しい保有と判断
    with S.local_only():
        S.upsert_df("holdings", pd.DataFrame([{"account": "nisa", "ticker": "1111.T", "name": "本物",
                                               "shares": 100, "avg_cost": 900.0,
                                               "target_yield": 0.05}]),
                    ["account", "ticker", "name", "shares", "avg_cost", "target_yield"])
        S.upsert_df("holding_review", pd.DataFrame([{"account": "nisa", "ticker": "1111.T",
                                                     "decision": "watch", "decided_at": "2026-09-01"}]),
                    ["account", "ticker", "decision", "decided_at"])
    # クラウド: 検算に本物を売られ、偽の売買が残った状態
    rows = [("2026-09-26", "specific", "9999.T", "監査テスト", "buy", 7, 1234.0, "監査"),
            ("2026-09-26", "specific", "9999.T", "監査テスト", "sell", 7, 1500.0, "監査"),
            ("2026-09-26", "nisa", "1111.T", "本物", "sell", 100, 1000.0, "監査"),
            ("2026-09-20", "nisa", "2222.T", "普通の売買", "sell", 5, 3000.0, "整理タブから記録")]
    with S.connect() as conn:
        conn.executemany("INSERT INTO transactions (date, account, ticker, name, type, shares, "
                         "price, memo) VALUES (?,?,?,?,?,?,?,?)", rows)
        conn.execute("INSERT INTO holdings (account, ticker, name, shares) "
                     "VALUES ('nisa','3333.T','別の保有',50)")


def test_dry_run_changes_nothing(turso, monkeypatch, capsys):
    _leaked_state(turso)
    before = turso.rows("SELECT * FROM transactions"), turso.rows("SELECT * FROM holdings")
    monkeypatch.setattr(sys, "argv", ["repair"])
    assert _load().main() == 0
    assert (turso.rows("SELECT * FROM transactions"), turso.rows("SELECT * FROM holdings")) == before
    assert "1111.T" in capsys.readouterr().out


def test_apply_restores_holding_and_removes_fakes(turso, monkeypatch):
    _leaked_state(turso)
    monkeypatch.setattr(sys, "argv", ["repair", "--apply"])
    assert _load().main() == 0
    assert turso.rows("SELECT memo FROM transactions") == [("整理タブから記録",)]
    assert sorted(turso.rows("SELECT ticker, shares, avg_cost, target_yield FROM holdings")) == [
        ("1111.T", 100.0, 900.0, 0.05), ("3333.T", 50.0, None, None)]
    assert turso.rows("SELECT decision FROM holding_review WHERE ticker='1111.T'") == [("watch",)]
    # 2回目は何もしない
    assert _load().main() == 0
    assert len(turso.rows("SELECT * FROM holdings")) == 2
