"""SQLite 永続化層。

ORM は使わず生 SQL。DB はローカルが唯一の正で、保有情報は外に出さない。

テーブルの役割:
    universe      JPX 由来の全上場銘柄（コード・銘柄名・33業種・市場区分）
    prices        週足の終値と出来高。全市場3,700銘柄の日足を持つと2,000万行を超えるため
                  週足に落とす。利回りパーセンタイルもバリュエーション履歴も週足で足りる
    quotes        銘柄ごとの最新値。日足から計算した52週レンジ・20日平均売買代金など
    dividends     配当落ち日ベースの1回ごとの配当額（分割調整済み）
    splits        株式分割・併合（DPS を遡って調整するために必要）
    fundamentals  財務5期分。絞り込み後の銘柄だけ埋まる
    scores        日付つきスナップショット。検証の生命線なので上書きしない
                  total  = 発掘スコア（市場に見過ごされているか を含む）
                  health = 配当継続スコア（配当が続くか だけを見る。保有の評価用）
    holdings      保有（口座別）
    transactions  売買記録
    watchlist     監視銘柄と目標利回り
    alerts        監視が拾った異変
    edinet_index  証券コード → 有価証券報告書の docID
    edinet_summary 有報の「主要な経営指標等の推移」5年分（日本基準の正確な値）
    company_profile 事業の内容・従業員数・権利確定日・会社予想配当
    settings      key-value
    market_history 日経平均とPBRの日次記録（暴落時の買い向かいの判断に使う）
    holding_review 整理の判断（持ち続ける/様子見）。同じ銘柄が毎回並ばないように
    equity_history 評価額と年間配当の推移。インカムが育っているかを見る
"""
from __future__ import annotations

import math

import json
import os
import sqlite3
import urllib.request
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, Iterator

import pandas as pd

from modules.config import db_path

# 保有・売買・監視など「積立の管理」に関わるテーブルと、それを画面に出すのに
# 要る軽量な参照データ（銘柄名・現在値・配当履歴・スコア・財務・相場水準）を
# クラウド(Turso)に同期する。重すぎる週足の株価履歴(prices)だけはパソコン専用。
# TURSO_DATABASE_URL/TURSO_AUTH_TOKEN が環境変数にあれば、これらのテーブルへの
# SQL はすべて自動でクラウド側に振り分けられる（呼び出し側は意識しなくてよい）。
CLOUD_TABLES = {
    "holdings", "transactions", "watchlist", "alerts",
    "settings", "holding_review", "equity_history",
    "universe", "quotes", "dividends", "scores",
    "fundamentals", "market_history", "benchmarks",
}

_DDL = """
CREATE TABLE IF NOT EXISTS universe (
    ticker      TEXT PRIMARY KEY,   -- 9432.T
    code        TEXT NOT NULL,      -- 9432
    name        TEXT,
    sector33    TEXT,               -- JPX の 33業種区分（日本語）
    market      TEXT,               -- プライム（内国株式）等
    scale       TEXT,               -- 規模区分
    updated_at  TEXT
);

CREATE TABLE IF NOT EXISTS prices (          -- 週足
    ticker  TEXT NOT NULL,
    date    TEXT NOT NULL,                    -- その週の最終営業日
    close   REAL,
    volume  REAL,
    PRIMARY KEY (ticker, date)
);
CREATE INDEX IF NOT EXISTS idx_prices_date ON prices(date);

CREATE TABLE IF NOT EXISTS quotes (
    ticker          TEXT PRIMARY KEY,
    asof            TEXT,
    last_close      REAL,
    high_52w        REAL,
    low_52w         REAL,
    pos_52w         REAL,    -- 52週レンジ内の位置 0=安値 1=高値
    avg_turnover    REAL,    -- 20日平均売買代金（円）
    ret_1y          REAL,
    listing_start   TEXT,    -- 株価データの最初の日（上場経過年数の代用）
    n_bars          INTEGER
);

CREATE TABLE IF NOT EXISTS dividends (
    ticker  TEXT NOT NULL,
    date    TEXT NOT NULL,          -- 権利落ち日
    amount  REAL NOT NULL,          -- 1株あたり（分割調整済み）
    PRIMARY KEY (ticker, date)
);

CREATE TABLE IF NOT EXISTS splits (
    ticker  TEXT NOT NULL,
    date    TEXT NOT NULL,
    ratio   REAL NOT NULL,
    PRIMARY KEY (ticker, date)
);

CREATE TABLE IF NOT EXISTS fundamentals (
    ticker          TEXT NOT NULL,
    fiscal_end      TEXT NOT NULL,  -- 決算期末 YYYY-MM-DD
    net_income      REAL,
    revenue         REAL,
    operating_income REAL,
    operating_cf    REAL,
    free_cf         REAL,
    total_equity    REAL,
    total_assets    REAL,
    total_debt      REAL,
    cash            REAL,
    shares          REAL,
    PRIMARY KEY (ticker, fiscal_end)
);

CREATE TABLE IF NOT EXISTS snapshots (
    ticker      TEXT NOT NULL,
    asof        TEXT NOT NULL,      -- スナップショット日
    market_cap  REAL,
    per         REAL,
    pbr         REAL,
    roe         REAL,
    payout_ratio REAL,
    held_pct_institutions REAL,
    PRIMARY KEY (ticker, asof)
);

CREATE TABLE IF NOT EXISTS scores (
    ticker      TEXT NOT NULL,
    asof        TEXT NOT NULL,
    total       REAL,
    capacity    REAL,
    willingness REAL,
    growth      REAL,
    neglect     REAL,
    valuation   REAL,
    trap_penalty REAL,
    health      REAL,               -- 配当継続スコア（発掘スコアとは別物）
    gate_passed INTEGER,
    gate_reason TEXT,
    detail_json TEXT,               -- 各指標の生値。カルテで内訳を開示するため
    PRIMARY KEY (ticker, asof)
);

CREATE TABLE IF NOT EXISTS holdings (
    account     TEXT NOT NULL,      -- specific / nisa
    ticker      TEXT NOT NULL,
    name        TEXT,
    shares      REAL NOT NULL,
    avg_cost    REAL,
    target_yield REAL,
    bottom_yield REAL,
    updated_at  TEXT,
    PRIMARY KEY (account, ticker)
);

CREATE TABLE IF NOT EXISTS transactions (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    date    TEXT NOT NULL,
    account TEXT,
    ticker  TEXT,
    name    TEXT,
    type    TEXT,                   -- buy / sell / dividend
    shares  REAL,
    price   REAL,
    fee     REAL,
    memo    TEXT,
    ref_date TEXT,                   -- 配当なら権利落ち日。二重計上を防ぐ鍵
    tax     REAL                     -- 売却時にかかる税額（特定口座）。手取りの計算に使う
);

CREATE TABLE IF NOT EXISTS watchlist (
    ticker      TEXT PRIMARY KEY,
    name        TEXT,
    target_yield REAL,
    bottom_yield REAL,
    source      TEXT,               -- manual / scout（発掘由来）
    thesis      TEXT,               -- なぜ増配が続くと考えるか
    invalidation TEXT,              -- 棄却条件
    added_at    TEXT,
    note        TEXT
);

CREATE TABLE IF NOT EXISTS alerts (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    detected_at TEXT NOT NULL,
    ticker      TEXT,
    severity    TEXT,               -- high / medium / low
    kind        TEXT,               -- dividend_cut / ocf_decline / limit_reached ...
    message     TEXT,
    resolved    INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS settings (
    key         TEXT PRIMARY KEY,
    value       TEXT,
    updated_at  TEXT
);

CREATE TABLE IF NOT EXISTS edinet_index (
    code        TEXT PRIMARY KEY,   -- 証券コード4桁
    doc_id      TEXT NOT NULL,
    filer_name  TEXT,
    period_end  TEXT,
    submit_date TEXT,
    fetched_at  TEXT                -- XBRL を取得済みなら日時
);

CREATE TABLE IF NOT EXISTS edinet_summary (   -- 主要な経営指標等の推移（5年分）
    ticker          TEXT NOT NULL,
    fiscal_year     INTEGER NOT NULL,
    basis           TEXT,               -- 連結 / 単体（混ぜると無意味になるので記録する）
    sales           REAL,
    ordinary_income REAL,
    net_income      REAL,
    eps             REAL,
    dps             REAL,
    payout_ratio    REAL,
    roe             REAL,
    equity_ratio    REAL,
    net_assets      REAL,
    total_assets    REAL,
    operating_cf    REAL,
    employees       REAL,
    PRIMARY KEY (ticker, fiscal_year)
);

CREATE TABLE IF NOT EXISTS company_profile (
    ticker          TEXT PRIMARY KEY,
    business_ja     TEXT,    -- 有報の「事業の内容」（日本語）
    business_en     TEXT,    -- yfinance の英文サマリ（有報が無いときの代替）
    employees       REAL,
    ex_dividend_date TEXT,   -- 次回（直近）の権利確定日
    dividend_rate   REAL,    -- 会社予想の年間配当
    industry_en     TEXT,
    dividend_policy TEXT,    -- 有報の「配当政策」本文
    policy_flags    TEXT,    -- 累進配当・DOE・配当性向目標などの検出結果（JSON）
    policy_score    REAL,    -- 0〜1。増配意思スコアに使う
    updated_at      TEXT
);

CREATE TABLE IF NOT EXISTS equity_history (   -- 資産と配当の推移（日次スナップショット）
    date            TEXT PRIMARY KEY,
    total_eval      REAL,
    total_cost      REAL,
    annual_dividend REAL,
    annual_dividend_after_tax REAL,
    holdings_count  INTEGER,
    yoc             REAL,
    source          TEXT            -- snapshot（その日に記録）/ backfill（過去株価からの推計）
);

CREATE TABLE IF NOT EXISTS market_history (   -- 相場の水準（自前で貯める）
    date            TEXT PRIMARY KEY,
    nikkei          REAL,
    pbr_weighted    REAL,
    pbr_index       REAL,
    pct_10y         REAL,
    vs_ma200        REAL
);

CREATE TABLE IF NOT EXISTS benchmarks (   -- 「ほっといた場合」の比較対象
    symbol  TEXT NOT NULL,                -- 1306.T（TOPIX連動ETF）等
    date    TEXT NOT NULL,
    close   REAL,                         -- 分配金を再投資した後の値
    PRIMARY KEY (symbol, date)
);

CREATE TABLE IF NOT EXISTS holding_review (   -- 整理の判断を覚えておく
    account     TEXT NOT NULL,
    ticker      TEXT NOT NULL,
    decision    TEXT,               -- keep（持ち続ける）/ watch（様子見）/ sold（売った）
    decided_at  TEXT,
    note        TEXT,
    PRIMARY KEY (account, ticker)
);

CREATE TABLE IF NOT EXISTS scan_runs (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at  TEXT,
    finished_at TEXT,
    kind        TEXT,               -- weekly / daily
    n_universe  INTEGER,
    n_passed    INTEGER,
    note        TEXT
);
"""

# クラウド(Turso)側に作るテーブルは CLOUD_TABLES の分だけ。上の _DDL から該当部分を
# そのまま抜き出したもの。発掘用の重いテーブル(prices 等)はクラウドには作らない。
_DDL_CLOUD = """
CREATE TABLE IF NOT EXISTS universe (
    ticker      TEXT PRIMARY KEY,
    code        TEXT NOT NULL,
    name        TEXT,
    sector33    TEXT,
    market      TEXT,
    scale       TEXT,
    updated_at  TEXT
);

CREATE TABLE IF NOT EXISTS quotes (
    ticker          TEXT PRIMARY KEY,
    asof            TEXT,
    last_close      REAL,
    high_52w        REAL,
    low_52w         REAL,
    pos_52w         REAL,
    avg_turnover    REAL,
    ret_1y          REAL,
    listing_start   TEXT,
    n_bars          INTEGER
);

CREATE TABLE IF NOT EXISTS dividends (
    ticker  TEXT NOT NULL,
    date    TEXT NOT NULL,
    amount  REAL NOT NULL,
    PRIMARY KEY (ticker, date)
);

CREATE TABLE IF NOT EXISTS scores (
    ticker      TEXT NOT NULL,
    asof        TEXT NOT NULL,
    total       REAL,
    capacity    REAL,
    willingness REAL,
    growth      REAL,
    neglect     REAL,
    valuation   REAL,
    trap_penalty REAL,
    health      REAL,
    gate_passed INTEGER,
    gate_reason TEXT,
    detail_json TEXT,
    PRIMARY KEY (ticker, asof)
);

CREATE TABLE IF NOT EXISTS fundamentals (
    ticker          TEXT NOT NULL,
    fiscal_end      TEXT NOT NULL,
    net_income      REAL,
    revenue         REAL,
    operating_income REAL,
    operating_cf    REAL,
    free_cf         REAL,
    total_equity    REAL,
    total_assets    REAL,
    total_debt      REAL,
    cash            REAL,
    shares          REAL,
    PRIMARY KEY (ticker, fiscal_end)
);

CREATE TABLE IF NOT EXISTS market_history (
    date            TEXT PRIMARY KEY,
    nikkei          REAL,
    pbr_weighted    REAL,
    pbr_index       REAL,
    pct_10y         REAL,
    vs_ma200        REAL
);

CREATE TABLE IF NOT EXISTS benchmarks (
    symbol  TEXT NOT NULL,
    date    TEXT NOT NULL,
    close   REAL,
    PRIMARY KEY (symbol, date)
);

CREATE TABLE IF NOT EXISTS holdings (
    account     TEXT NOT NULL,
    ticker      TEXT NOT NULL,
    name        TEXT,
    shares      REAL NOT NULL,
    avg_cost    REAL,
    target_yield REAL,
    bottom_yield REAL,
    updated_at  TEXT,
    PRIMARY KEY (account, ticker)
);

CREATE TABLE IF NOT EXISTS transactions (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    date    TEXT NOT NULL,
    account TEXT,
    ticker  TEXT,
    name    TEXT,
    type    TEXT,
    shares  REAL,
    price   REAL,
    fee     REAL,
    memo    TEXT,
    ref_date TEXT,
    tax     REAL
);

CREATE TABLE IF NOT EXISTS watchlist (
    ticker      TEXT PRIMARY KEY,
    name        TEXT,
    target_yield REAL,
    bottom_yield REAL,
    source      TEXT,
    thesis      TEXT,
    invalidation TEXT,
    added_at    TEXT,
    note        TEXT
);

CREATE TABLE IF NOT EXISTS alerts (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    detected_at TEXT NOT NULL,
    ticker      TEXT,
    severity    TEXT,
    kind        TEXT,
    message     TEXT,
    resolved    INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS settings (
    key         TEXT PRIMARY KEY,
    value       TEXT,
    updated_at  TEXT
);

CREATE TABLE IF NOT EXISTS equity_history (
    date            TEXT PRIMARY KEY,
    total_eval      REAL,
    total_cost      REAL,
    annual_dividend REAL,
    annual_dividend_after_tax REAL,
    holdings_count  INTEGER,
    yoc             REAL,
    source          TEXT
);

CREATE TABLE IF NOT EXISTS holding_review (
    account     TEXT NOT NULL,
    ticker      TEXT NOT NULL,
    decision    TEXT,
    decided_at  TEXT,
    note        TEXT,
    PRIMARY KEY (account, ticker)
);
"""


class _TursoRow(tuple):
    """sqlite3.Row 互換。列名でも位置でも引ける。"""
    def __new__(cls, cols, vals):
        obj = super().__new__(cls, vals)
        obj._cols = cols
        return obj

    def __getitem__(self, key):
        if isinstance(key, str):
            return tuple.__getitem__(self, self._cols.index(key))
        return tuple.__getitem__(self, key)

    def keys(self):
        return self._cols


class _TursoHttpCursor:
    def __init__(self, cols, rows, lastrowid=None):
        self.description = [(c, None, None, None, None, None, None) for c in cols] if cols else None
        self._rows = rows
        self._pos = 0
        self.lastrowid = lastrowid

    def fetchone(self):
        if self._pos >= len(self._rows):
            return None
        row = self._rows[self._pos]
        self._pos += 1
        return row

    def fetchall(self):
        rows = self._rows[self._pos:]
        self._pos = len(self._rows)
        return rows


class _TursoHttpConn:
    """Turso HTTP v2/pipeline API を sqlite3.Connection 互換に薄くラップ。

    libsql-experimental（Rustコンパイル必須）に依存せず、標準ライブラリの
    urllib だけで動く。stock-recommender の modules/trading_log.py と同じ手法。
    """

    def __init__(self, url: str, token: str):
        self._base = url.replace("libsql://", "https://")
        self._token = token

    def _http_pipeline(self, requests: list) -> list:
        payload = json.dumps({"requests": requests}).encode("utf-8")
        req = urllib.request.Request(
            f"{self._base}/v2/pipeline",
            data=payload,
            headers={"Authorization": f"Bearer {self._token}", "Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read())["results"]

    @staticmethod
    def _py_to_arg(v):
        if v is None:
            return {"type": "null", "value": None}
        if isinstance(v, bool):
            return {"type": "integer", "value": str(int(v))}
        if isinstance(v, int):
            return {"type": "integer", "value": str(v)}
        if isinstance(v, float):
            if not math.isfinite(v):
                return {"type": "null", "value": None}
            return {"type": "float", "value": v}
        # numpy の数値（np.int64 など）を文字列として送らない。ローカルの sqlite3 は
        # 受け付けずに落ちるが、こちらは黙って TEXT で保存し、数値の比較が効かなくなる。
        if hasattr(v, "item") and not isinstance(v, (str, bytes)):
            try:
                return _TursoHttpConn._py_to_arg(v.item())
            except (TypeError, ValueError):
                pass
        return {"type": "text", "value": str(v)}

    @staticmethod
    def _parse_result(result: dict) -> tuple[list, list]:
        cols = [c["name"] for c in result.get("cols", [])]
        rows = []
        for raw_row in result.get("rows", []):
            vals = []
            for cell in raw_row:
                t, v = cell.get("type"), cell.get("value")
                if t == "null" or v is None:
                    vals.append(None)
                elif t == "integer":
                    vals.append(int(v))
                elif t in ("real", "float"):
                    vals.append(float(v))
                else:
                    vals.append(v)
            rows.append(_TursoRow(cols, vals))
        return cols, rows

    def execute(self, sql: str, params=()):
        if not isinstance(params, (tuple, list)):
            params = (params,)
        args = [self._py_to_arg(p) for p in params]
        stmt: dict = {"sql": sql}
        if args:
            stmt["args"] = args
        results = self._http_pipeline([{"type": "execute", "stmt": stmt}, {"type": "close"}])
        res = results[0]
        if res.get("type") == "error":
            raise Exception(res.get("error", {}).get("message", "Turso HTTP error"))
        result = res["response"]["result"]
        last_id = result.get("last_insert_rowid")
        last_id = int(last_id) if last_id is not None else None
        cols, rows = self._parse_result(result)
        return _TursoHttpCursor(cols, rows, lastrowid=last_id)

    def executemany(self, sql: str, seq_of_params):
        seq = list(seq_of_params)
        if not seq:
            return _TursoHttpCursor([], [])
        requests = []
        for params in seq:
            if not isinstance(params, (tuple, list)):
                params = (params,)
            args = [self._py_to_arg(p) for p in params]
            stmt: dict = {"sql": sql}
            if args:
                stmt["args"] = args
            requests.append({"type": "execute", "stmt": stmt})
        requests.append({"type": "close"})
        results = self._http_pipeline(requests)
        for res in results:
            if res.get("type") == "error":
                raise Exception(res.get("error", {}).get("message", "Turso executemany error"))
        return _TursoHttpCursor([], [])

    def executescript(self, script: str):
        stmts = [s.strip() for s in script.split(";") if s.strip()]
        requests = [{"type": "execute", "stmt": {"sql": s}} for s in stmts]
        requests.append({"type": "close"})
        results = self._http_pipeline(requests)
        for res in results:
            if res.get("type") == "error":
                msg = res.get("error", {}).get("message", "")
                if "already exists" not in msg.lower():
                    raise Exception(f"Turso executescript error: {msg}")

    def batch(self, stmts: list[tuple[str, tuple]]) -> None:
        """複数の書き込みを1つのトランザクションで通す。途中で失敗したら全部取り消す。

        execute() は1文ごとに別のHTTPで自動コミットされるので、売却の記録と
        保有の株数の更新を別々に送ると、間で落ちたとき片方だけが残る。
        """
        steps = [{"stmt": {"sql": "BEGIN"}}]
        for sql, params in stmts:
            stmt: dict = {"sql": sql}
            args = [self._py_to_arg(p) for p in params]
            if args:
                stmt["args"] = args
            steps.append({"condition": {"type": "ok", "step": len(steps) - 1}, "stmt": stmt})
        n = len(steps)
        steps.append({"condition": {"type": "ok", "step": n - 1}, "stmt": {"sql": "COMMIT"}})
        steps.append({"condition": {"type": "not", "cond": {"type": "ok", "step": n}},
                      "stmt": {"sql": "ROLLBACK"}})
        res = self._http_pipeline([{"type": "batch", "batch": {"steps": steps}},
                                   {"type": "close"}])[0]
        if res.get("type") == "error":
            raise Exception(res.get("error", {}).get("message", "Turso batch error"))
        errors = res["response"]["result"].get("step_errors", [])
        for err in errors[:n + 1]:
            if err:
                raise Exception(f"Turso batch error: {err.get('message', err)}")

    def commit(self):
        pass  # HTTP API は自動コミット

    def close(self):
        pass


def _table_names_in(sql: str) -> set[str]:
    """雑な正規表現ではなく、SQLに単語として現れるテーブル名を拾う。"""
    import re
    words = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", sql))
    return words


class _DualConn:
    """1本の DB 操作の裏で、SQL に含まれるテーブル名によって
    ローカル SQLite と Turso クラウドのどちらか一方に自動で振り分ける。

    CLOUD_TABLES と発掘用テーブルを同じクエリで JOIN する箇所は無い
    （store.py 定義時点で確認済み）ので、SQL 単位の振り分けで安全に成立する。
    """

    def __init__(self, local_conn: sqlite3.Connection, cloud_conn: _TursoHttpConn):
        self._local = local_conn
        self._cloud = cloud_conn

    def _route(self, sql: str):
        names = _table_names_in(sql)
        if names & CLOUD_TABLES:
            return self._cloud
        return self._local

    def execute(self, sql: str, params=()):
        return self._route(sql).execute(sql, params)

    def executemany(self, sql: str, seq_of_params):
        return self._route(sql).executemany(sql, seq_of_params)

    def batch(self, stmts: list[tuple[str, tuple]]) -> None:
        targets = {id(self._route(sql)) for sql, _ in stmts}
        if len(targets) > 1:
            raise ValueError("クラウドとパソコンのテーブルを1つのトランザクションにはできません")
        target = self._route(stmts[0][0]) if stmts else self._local
        if target is self._cloud:
            self._cloud.batch(stmts)
        else:
            for sql, params in stmts:
                self._local.execute(sql, params)

    def executescript(self, script: str):
        self._local.executescript(script)
        self._cloud.executescript(_DDL_CLOUD)

    def commit(self):
        self._local.commit()

    def close(self):
        self._local.close()


_bridged = False
_force_local = False


@contextmanager
def local_only() -> Iterator[None]:
    """この中の connect() はクラウドに一切つながず、パソコンのDBだけを使う。

    検算のように「DBのコピーで書き込みを試す」処理のためのもの。db_path を
    差し替えても、TURSO_DATABASE_URL があれば保有・売買はクラウドに振り分けられる。
    実測: audit.py の書き込み検算が、クラウドの本物の保有を1銘柄ずつ
    1,000円で売却扱いにして消し、偽の売買記録を残していた。
    """
    global _force_local
    prev, _force_local = _force_local, True
    try:
        yield
    finally:
        _force_local = prev


def _ensure_secrets() -> None:
    """接続を作る直前に、必ず secrets を環境変数へ通す。

    スクリプト側の呼び忘れで、クラウドではなくパソコンのDBに書いてしまうのを防ぐ。
    実測（2026-09-23）: weekly_scan.py が呼んでおらず、スキャン結果が4日ぶん
    アプリに届いていなかった。エラーは出ないので気づけない。
    """
    global _bridged
    if _bridged:
        return
    _bridged = True
    try:
        from modules.config import bridge_secrets_to_env
        bridge_secrets_to_env()
    except Exception:
        pass


@contextmanager
def connect(path: Path | None = None) -> Iterator[sqlite3.Connection]:
    _ensure_secrets()
    turso_url = os.environ.get("TURSO_DATABASE_URL")
    turso_token = os.environ.get("TURSO_AUTH_TOKEN")
    local_conn = sqlite3.connect(str(path or db_path()))
    local_conn.row_factory = sqlite3.Row
    try:
        local_conn.execute("PRAGMA journal_mode=WAL")
        local_conn.execute("PRAGMA synchronous=NORMAL")
        if turso_url and turso_token and not _force_local:
            conn = _DualConn(local_conn, _TursoHttpConn(turso_url, turso_token))
        else:
            conn = local_conn
        yield conn
        conn.commit()
    finally:
        local_conn.close()


def run_batch(conn, stmts: list[tuple[str, tuple]]) -> None:
    """複数の書き込みを「全部通るか、全部通らないか」で実行する。

    ローカルの sqlite3 は connect() の with を抜けるまでが1トランザクションなので
    そのまま流すだけでよい。クラウド行きは1回のHTTPにまとめて送る。
    """
    if hasattr(conn, "batch"):
        conn.batch(stmts)
    else:
        for sql, params in stmts:
            conn.execute(sql, params)


# 既存テーブルに後から足した列。CREATE TABLE IF NOT EXISTS は列を追加しないので、
# ここで ALTER TABLE を当てる。すでにある場合の例外は握りつぶす。
_MIGRATIONS = [
    ("scores", "health", "REAL"),
    ("edinet_summary", "basis", "TEXT"),
    ("company_profile", "dividend_policy", "TEXT"),
    ("company_profile", "policy_flags", "TEXT"),
    ("company_profile", "policy_score", "REAL"),
    # 配当の受取記録の二重計上を防ぐ。入金日は「権利落ちから何日後か」の設定で動くので、
    # 入金日で重複を判定すると設定を変えるたびに同じ配当をもう一度記録できてしまう
    # （実測: 入金までの日数を75日→45日にすると、記録済みの222件が再び未記録として出た）。
    # 権利落ち日は動かないので、こちらを鍵にする。
    ("transactions", "ref_date", "TEXT"),
    # 売ったときの税額。整理で生まれた「手取り」を正確に積み上げて、
    # そのお金をそのまま次の買い付けに回せるようにする。
    ("transactions", "tax", "REAL"),
    # 推移の1行が「実際に記録した日」か「過去株価から計算し直した推計」かを分ける。
    # portfolio-manager から引き継いだ2〜6月ぶんは、当時の保有数が残っていないので
    # 『今の保有のまま過去も持っていたら』という前提で計算されている。実測値と混ぜて
    # 1本の線にすると、増えていない配当が増えたように見えるので必ず区別する。
    ("equity_history", "source", "TEXT"),
]


def init_db(path: Path | None = None) -> None:
    with connect(path) as conn:
        conn.executescript(_DDL)
        for table, column, coltype in _MIGRATIONS:
            try:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {coltype}")
            except Exception:
                pass   # 既にある（ローカル/クラウドどちらでも列追加の重複はここで握りつぶす）


def upsert_df(table: str, df: pd.DataFrame, columns: Iterable[str],
              path: Path | None = None, chunk: int = 5000) -> int:
    """DataFrame を INSERT OR REPLACE でまとめて書き込む。"""
    columns = list(columns)
    if df is None or df.empty:
        return 0
    # クラウド(Turso)行きは1回に詰め込みすぎるとタイムアウトする
    # （実測: chunk=300 で 87,017行の書き込みが途中で落ちた。50なら安定）。
    # 呼び出し側は行き先を意識しなくてよい約束なので、ここで自動的に絞る。
    _ensure_secrets()
    if table in CLOUD_TABLES and os.environ.get("TURSO_DATABASE_URL") and not _force_local:
        chunk = min(chunk, 50)
    sub = df.reindex(columns=columns)
    placeholders = ",".join("?" * len(columns))
    sql = f"INSERT OR REPLACE INTO {table} ({','.join(columns)}) VALUES ({placeholders})"
    # 無限大を通さない。yfinance は赤字の会社の PER に Infinity を返すことがあり
    # （実測: 2676.T / 3543.T）、SQLite はそれを **文字列 "Infinity"** として
    # 書き込む。読み戻すと列全体が文字列になり、`df["per"] > 0` で落ちる。
    # 週次スキャンが止まる原因になるので、書き込む手前で必ず落とす。
    def _clean(v):
        if v is None:
            return None
        if isinstance(v, float) and not math.isfinite(v):
            return None
        try:
            if pd.isna(v):
                return None
        except (TypeError, ValueError):
            pass
        return v

    rows = [tuple(_clean(v) for v in rec) for rec in sub.itertuples(index=False, name=None)]
    with connect(path) as conn:
        for i in range(0, len(rows), chunk):
            conn.executemany(sql, rows[i:i + chunk])
    return len(rows)


def read_df(sql: str, params: tuple = (), path: Path | None = None) -> pd.DataFrame:
    with connect(path) as conn:
        cur = conn.execute(sql, params)
        cols = [d[0] for d in cur.description] if cur.description else []
        rows = [tuple(r) for r in cur.fetchall()]
        return pd.DataFrame(rows, columns=cols)


def latest_scores(path: Path | None = None) -> pd.DataFrame:
    """最新スナップショットのスコアをユニバース情報つきで返す。

    画面はこれだけあれば描ける。ここを pipeline 側に置いていたせいで、
    表を見るだけの操作でも yfinance・curl_cffi・peewee・bs4 まで読み込まれていた
    （実測 1.17秒）。取得系と閲覧系は import の経路から分けておく。
    """
    return read_df("""
        SELECT s.*, u.name, u.sector33, u.market, u.code,
               q.last_close, q.pos_52w, q.avg_turnover
        FROM scores s
        JOIN universe u ON u.ticker = s.ticker
        LEFT JOIN quotes q ON q.ticker = s.ticker
        WHERE s.asof = (SELECT MAX(asof) FROM scores)
    """, path=path)