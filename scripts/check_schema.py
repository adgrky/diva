"""コードの中のSQLを、実際のDBに食わせて確かめる。

【なぜ要るか】
列を足したあとアプリだけを起動すると「no such column」で画面ごと落ちる。
実測で、transactions に ref_date を足したあと、ポートフォリオ画面が
まるごと落ちる状態になっていた（マイグレーションを走らせるのがスクリプト側
だけで、アプリ起動時には走っていなかった）。

正規表現でSQLを読み解こうとすると JOIN や複数文で誤検出する。
**SQLite 自身に EXPLAIN させる**のがいちばん確実。構文も列名もテーブル名も、
本物のパーサが判定してくれる。実際のデータは1行も読まない。

    uv run python scripts/check_schema.py
"""
from __future__ import annotations

import ast
import re
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modules.config import db_path              # noqa: E402
from modules.store import init_db               # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
_SQL_HEAD = re.compile(r"^\s*(SELECT|INSERT|UPDATE|DELETE|REPLACE)\b", re.I)


def sql_literals(path: Path) -> list[tuple[int, str]]:
    """ファイル中の「SQLに見える文字列」を、連結も含めて取り出す。"""
    tree = ast.parse(path.read_text())
    found: list[tuple[int, str]] = []

    # f文字列の中の「文字どおりの部分」も ast.walk では Constant として出てくる。
    # そのまま拾うと "… WHERE ticker IN (" のような切れ端を1本のSQLとみなして
    # 「構文が壊れている」と誤検出するので、あらかじめ除いておく。
    inside_fstring = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.JoinedStr):
            for sub in ast.walk(node):
                inside_fstring.add(id(sub))

    def value_of(node) -> str | None:
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        if isinstance(node, ast.JoinedStr):       # f文字列は値が読めないので飛ばす
            return None
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            a, b = value_of(node.left), value_of(node.right)
            return None if a is None or b is None else a + b
        return None

    for node in ast.walk(tree):
        if id(node) in inside_fstring:
            continue
        v = value_of(node)
        if v and _SQL_HEAD.match(v):
            found.append((getattr(node, "lineno", 0), " ".join(v.split())))
    return found


def main() -> int:
    init_db()                       # 先にマイグレーションを済ませてから照合する
    conn = sqlite3.connect(f"file:{db_path()}?mode=ro", uri=True)

    files = sorted(list((ROOT / "modules").glob("[!.]*.py"))
                   + list((ROOT / "views").glob("[!.]*.py"))
                   + list((ROOT / "scripts").glob("[!.]*.py")) + [ROOT / "app.py"])
    n_sql = 0
    bad: list[tuple[str, int, str, str]] = []
    for f in files:
        for lineno, sql in sql_literals(f):
            if "{" in sql:                       # 組み立て途中のものは飛ばす
                sql = re.sub(r"\{[^}]*\}", "?", sql)
            n_sql += 1
            try:
                conn.execute("EXPLAIN " + sql, [None] * sql.count("?"))
            except sqlite3.Error as exc:
                bad.append((f.name, lineno, sql[:90], str(exc)))
    conn.close()

    print(f"照合したファイル {len(files)} 個 ／ SQL {n_sql} 本")
    if bad:
        print(f"\n⚠️  DBに通らないSQLがあります（{len(bad)} 件）:")
        for name, lineno, sql, err in bad:
            print(f"\n   {name}:{lineno}  {err}")
            print(f"      {sql}")
        print("\n   列を足したなら modules/store.py の _DDL と _MIGRATIONS に書いてください。")
        return 1
    print("✅ コードの中のSQLは、すべて実際のDBで通ります")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
