"""検算(audit.py)がクラウドに書いてしまった偽の売買を取り除き、消えた保有を戻す。

何が起きていたか:
    audit.py の書き込み検算は「DBのコピー」で動かすつもりだったが、db_path を
    差し替えてもクラウド(Turso)行きのテーブルは本物に書かれていた。更新.command の
    最後で毎回走るので、1回ごとに次のことが本物のクラウドに起きる。
      - 9999.T / 9998.T の偽の売買 8件（memo='監査'）
      - 保有の先頭1銘柄を 1,000円で全株売却扱い（memo='監査'）→ 保有から消える
      - その銘柄の整理の判断(holding_review)が消える

直し方:
    - memo='監査' の売買記録を消す
    - 偽の売却で消えた保有を、売却記録の株数で戻す。取得単価・目標利回りは
      パソコンのDB(クラウド同期より前の値)にあればそこから取る
    - 整理の判断もパソコンのDBにあれば戻す

使い方:
    python scripts/repair_audit_leak.py          # 何をするかを表示するだけ
    python scripts/repair_audit_leak.py --apply  # 実際に直す
"""
from __future__ import annotations

import argparse
import os
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modules.config import bridge_secrets_to_env, db_path  # noqa: E402
from modules.store import _TursoHttpConn  # noqa: E402

FAKE_TICKERS = ("9999.T", "9998.T")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="実際に書き込む")
    args = ap.parse_args()

    bridge_secrets_to_env()
    url, token = os.environ.get("TURSO_DATABASE_URL"), os.environ.get("TURSO_AUTH_TOKEN")
    if not (url and token):
        print("クラウド(Turso)の設定が見つかりません。直すものはありません。")
        return 0
    cloud = _TursoHttpConn(url, token)

    leaked = cloud.execute(
        "SELECT id, date, account, ticker, name, type, shares, price FROM transactions "
        "WHERE memo = '監査' ORDER BY id").fetchall()
    if not leaked:
        print("検算が残した記録はありません。")
        return 0

    local = None
    if db_path().exists():
        local = sqlite3.connect(str(db_path()))
        local.row_factory = sqlite3.Row

    # 偽の売却で消えた本物の保有
    restores = []
    for r in leaked:
        if r["type"] != "sell" or r["ticker"] in FAKE_TICKERS:
            continue
        acc, tk = r["account"], r["ticker"]
        if cloud.execute("SELECT 1 FROM holdings WHERE account=? AND ticker=?",
                         (acc, tk)).fetchone():
            continue   # 手で入れ直し済み
        old = local.execute("SELECT * FROM holdings WHERE account=? AND ticker=?",
                            (acc, tk)).fetchone() if local else None
        rev = local.execute("SELECT * FROM holding_review WHERE account=? AND ticker=?",
                            (acc, tk)).fetchone() if local else None
        restores.append((r, old, rev))

    print(f"検算が残した売買記録: {len(leaked)} 件（すべて消します）")
    for r in leaked:
        print(f"  #{r['id']} {r['date']} {r['account']} {r['ticker']} {r['name']} "
              f"{r['type']} {r['shares']:g}株 @{r['price']:,.0f}")
    print(f"\n戻す保有: {len(restores)} 銘柄")
    for r, old, rev in restores:
        cost = f"取得単価 {old['avg_cost']:,.1f}" if old and old["avg_cost"] else "取得単価 不明（あとで手で入れてください）"
        print(f"  {r['account']} {r['ticker']} {r['name']} {r['shares']:g}株 / {cost}"
              + (f" / 整理の判断 {rev['decision']}" if rev else ""))

    if not args.apply:
        print("\n表示だけです。直すには --apply を付けて実行してください。")
        return 0

    for r, old, rev in restores:
        cloud.execute(
            "INSERT OR REPLACE INTO holdings (account, ticker, name, shares, avg_cost, "
            "target_yield, bottom_yield, updated_at) VALUES (?,?,?,?,?,?,?,datetime('now'))",
            (r["account"], r["ticker"], r["name"], float(r["shares"]),
             old["avg_cost"] if old else None,
             old["target_yield"] if old else 0.047,
             old["bottom_yield"] if old else None))
        if rev:
            cloud.execute(
                "INSERT OR REPLACE INTO holding_review (account, ticker, decision, decided_at, note) "
                "VALUES (?,?,?,?,?)",
                (rev["account"], rev["ticker"], rev["decision"], rev["decided_at"], rev["note"]))
    cloud.executemany("DELETE FROM transactions WHERE id = ?", [(r["id"],) for r in leaked])
    for tk in FAKE_TICKERS:
        cloud.execute("DELETE FROM holdings WHERE ticker = ?", (tk,))
    print("\n直しました。アプリ左下の「🔄 読み込み直す」を押してください。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
