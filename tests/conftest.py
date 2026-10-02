"""テスト共通の土台。

本物のDB（パソコン・クラウドとも）には絶対に触らない。
- パソコンのDBは一時フォルダのものに差し替える
- クラウド(Turso)は、HTTPの代わりにメモリ上の SQLite で応答する偽物に差し替える
"""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import modules.config as C  # noqa: E402
import modules.store as S  # noqa: E402


@pytest.fixture(autouse=True)
def _no_real_db(tmp_path, monkeypatch):
    """全テストで、本物の secrets とDBの場所を読まないようにする。"""
    for k in C._SECRET_KEYS:
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr(S, "_bridged", True)
    db = tmp_path / "test.db"
    monkeypatch.setattr(C, "db_path", lambda config=None: db)
    monkeypatch.setattr(S, "db_path", lambda config=None: db)
    yield db


@pytest.fixture
def local_db(_no_real_db):
    S.init_db()
    return _no_real_db


class FakeTurso:
    """Turso の /v2/pipeline をメモリ上の SQLite で真似る（execute / batch / close）。"""

    def __init__(self):
        self.db = sqlite3.connect(":memory:", isolation_level=None)
        self.calls = 0
        self.fail_on: str | None = None   # この文字列を含むSQLでエラーを返す

    @staticmethod
    def _arg(a):
        t, v = a["type"], a["value"]
        if t == "null":
            return None
        if t == "integer":
            assert isinstance(v, str), "integer は文字列で送る決まり"
            return int(v)
        if t == "float":
            assert isinstance(v, (int, float)), "float は数値で送る決まり"
            return float(v)
        return v

    def _run(self, stmt):
        sql = stmt["sql"]
        if self.fail_on and self.fail_on in sql:
            raise sqlite3.OperationalError(f"fake failure: {sql}")
        cur = self.db.execute(sql, [self._arg(a) for a in stmt.get("args", [])])
        cols = [{"name": d[0]} for d in (cur.description or [])]
        rows = []
        for r in cur.fetchall():
            cells = []
            for v in r:
                if v is None:
                    cells.append({"type": "null", "value": None})
                elif isinstance(v, int):
                    cells.append({"type": "integer", "value": str(v)})
                elif isinstance(v, float):
                    cells.append({"type": "float", "value": v})
                else:
                    cells.append({"type": "text", "value": v})
            rows.append(cells)
        return {"cols": cols, "rows": rows, "last_insert_rowid": str(cur.lastrowid)
                if cur.lastrowid is not None else None}

    def _ok(self, i, results, errors):
        return results[i] is not None and errors[i] is None

    def _cond(self, c, results, errors):
        if c is None:
            return True
        if c["type"] == "ok":
            return self._ok(c["step"], results, errors)
        if c["type"] == "error":
            return errors[c["step"]] is not None
        if c["type"] == "not":
            return not self._cond(c["cond"], results, errors)
        raise NotImplementedError(c)

    def pipeline(self, requests):
        self.calls += 1
        out = []
        for req in requests:
            if req["type"] == "close":
                out.append({"type": "ok", "response": {"type": "close"}})
            elif req["type"] == "execute":
                try:
                    out.append({"type": "ok", "response": {"type": "execute",
                                                           "result": self._run(req["stmt"])}})
                except sqlite3.Error as e:
                    out.append({"type": "error", "error": {"message": str(e)}})
            elif req["type"] == "batch":
                steps = req["batch"]["steps"]
                results, errors = [None] * len(steps), [None] * len(steps)
                for i, st in enumerate(steps):
                    if not self._cond(st.get("condition"), results, errors):
                        continue
                    try:
                        results[i] = self._run(st["stmt"])
                    except sqlite3.Error as e:
                        errors[i] = {"message": str(e)}
                out.append({"type": "ok", "response": {"type": "batch", "result": {
                    "step_results": results, "step_errors": errors}}})
        return out

    def rows(self, sql, params=()):
        return self.db.execute(sql, params).fetchall()


@pytest.fixture
def turso(local_db, monkeypatch):
    """クラウド同期が有効な状態を作る。返り値の .rows() でクラウド側の中身を見られる。"""
    fake = FakeTurso()
    monkeypatch.setenv("TURSO_DATABASE_URL", "libsql://fake.example")
    monkeypatch.setenv("TURSO_AUTH_TOKEN", "t")
    monkeypatch.setattr(S._TursoHttpConn, "_http_pipeline",
                        lambda self, reqs, retries=0: fake.pipeline(reqs))
    S.init_db()   # クラウド側にもテーブルを作る
    return fake
