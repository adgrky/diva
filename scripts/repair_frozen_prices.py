"""すでに貯めてある週足のうち、株価が止まっている行を消して「欠損」に戻す。

yfinance は売買の無い日・データの無い区間に「出来高0・直前の終値のまま」の埋め草を
返す（詳しくは modules/quality.py の FROZEN_WEEKS 付近）。取り込み側は直したので、
これから取るぶんは入らない。ただし週足の書き込みは上書き（INSERT OR REPLACE）なので、
**すでに入っている埋め草の行は取り直しても消えない**。このスクリプトで消す。

消す行は次の2種類。どちらも区間の最初の1行（本物の最後の取引）は残す。
    ① 出来高0 で、前の週と終値が同じ週（1週間まるごと埋め草）
    ② ①を消した後も、同じ終値が FROZEN_WEEKS 週以上続く区間の2行目以降

消す前に、消す行を data/backup/ に CSV で書き出す（元に戻せるように）。
週足（prices）はパソコン専用のテーブルなので、クラウドには触らない。

使い方:
    uv run python scripts/repair_frozen_prices.py            # 何を消すかを見るだけ
    uv run python scripts/repair_frozen_prices.py --apply    # 実際に消す
    uv run python scripts/repair_frozen_prices.py --db コピー.db --apply
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modules.config import db_path                      # noqa: E402
from modules.quality import FROZEN_WEEKS, frozen_mask, frozen_runs  # noqa: E402


def rows_to_delete(p: pd.DataFrame) -> pd.DataFrame:
    """消す行（ticker, date, close, volume, why）。p は ticker・date 順。"""
    prev = p.groupby("ticker", sort=False)["close"].shift()
    filler = p["volume"].fillna(0).eq(0) & p["close"].eq(prev)
    rest = p[~filler]
    frozen = frozen_mask(rest)
    out = pd.concat([p[filler].assign(why="出来高0の埋め草"),
                     rest[frozen].assign(why=f"同値{FROZEN_WEEKS}週以上")])
    return out.sort_values(["ticker", "date"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="実際に消す（無ければ見るだけ）")
    ap.add_argument("--db", type=Path, default=None, help="対象のDB（既定はアプリのDB）")
    args = ap.parse_args()

    path = args.db or db_path()
    conn = sqlite3.connect(str(path))
    p = pd.read_sql("SELECT ticker, date, close, volume FROM prices ORDER BY ticker, date", conn)
    before = frozen_runs(p, 26)
    dele = rows_to_delete(p)

    print(f"対象: {path}")
    print(f"週足 {len(p):,} 行のうち、消す行 {len(dele):,} 行（{dele['ticker'].nunique()} 銘柄）")
    print(dele["why"].value_counts().to_string())
    print(f"同じ終値が26週以上続く区間: {len(before)} 区間 / {before['ticker'].nunique()} 銘柄"
          f"（消した後の見込み: "
          f"{len(frozen_runs(p.drop(dele.index), 26))} 区間）")
    top = (dele.groupby("ticker").agg(行数=("date", "size"), 最初=("date", "min"),
                                      最後=("date", "max"))
           .sort_values("行数", ascending=False).head(15))
    print("\n消す行の多い銘柄:\n" + top.to_string())

    if not args.apply:
        print("\n（見るだけ。消すときは --apply を付ける）")
        return 0
    if dele.empty:
        return 0

    bdir = Path(path).parent / "backup"
    bdir.mkdir(parents=True, exist_ok=True)
    bfile = bdir / f"prices_frozen_{datetime.now():%Y%m%d_%H%M%S}.csv"
    dele.to_csv(bfile, index=False)
    with conn:
        conn.executemany("DELETE FROM prices WHERE ticker = ? AND date = ?",
                         list(dele[["ticker", "date"]].itertuples(index=False, name=None)))
    left = pd.read_sql("SELECT COUNT(*) AS n FROM prices", conn)["n"].iloc[0]
    conn.close()
    print(f"\n消しました: {len(dele):,} 行 → 残り {left:,} 行")
    print(f"消した行の控え: {bfile}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
