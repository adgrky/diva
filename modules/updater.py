"""アプリの中からデータ更新を始める。Streamlit に依存しない。

以前は Finder で `更新.command` を探してダブルクリックする必要があった。
画面のボタンから同じ更新（scripts/scheduled_update.py）を裏で走らせ、
進み具合はログの末尾を読んで画面に出す。

【パソコンで開いているときだけ動く】
DB の正はパソコンの SQLite（週足の株価・有報の全文はクラウドに無い）なので、
Streamlit Cloud（スマホなど）から開いたときは更新を始められない。

【二重に走らせない】
scheduled_update.py は走っているあいだ data/.update.lock を置き、
あとから来たほうは黙って降りる。土曜の自動更新と重なっても安全。
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOCK = ROOT / "data" / ".update.lock"
LOG = ROOT / "logs" / "manual_update.log"
_PY = ROOT / ".venv" / "bin" / "python"


def can_run_here() -> bool:
    """このパソコンの上でアプリが動いているか（クラウドでは False）。"""
    return _PY.exists() and (ROOT / "data" / "scout.db").exists()


def is_running() -> bool:
    # scheduled_update.py と同じ基準（3時間より古い鍵は、落ちた更新の残骸とみなす）
    return LOCK.exists() and time.time() - LOCK.stat().st_mtime < 3 * 3600


def start(full: bool = False) -> bool:
    """更新を裏で始める。すでに走っていれば何もしない。"""
    if not can_run_here() or is_running():
        return False
    LOG.parent.mkdir(parents=True, exist_ok=True)
    cmd = [str(_PY), str(ROOT / "scripts" / "scheduled_update.py")]
    if full:
        cmd.append("--all")
    with open(LOG, "w", encoding="utf-8") as out:
        out.write(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] アプリから更新を開始\n")
        out.flush()
        # 画面を閉じても更新は最後まで走らせたいので、アプリとは別のプロセスにする
        subprocess.Popen(cmd, cwd=ROOT, stdout=out, stderr=subprocess.STDOUT,
                         stdin=subprocess.DEVNULL, start_new_session=True,
                         env={**os.environ, "PYTHONUNBUFFERED": "1"})
    # 鍵が置かれるまで少し待つ（押した直後の画面で「走っていない」と出ないように）
    for _ in range(20):
        if LOCK.exists():
            break
        time.sleep(0.25)
    return True


def status(tail: int = 6) -> dict:
    """{running, started, log（末尾）, finished, ok}"""
    out = {"running": is_running(), "started": None, "log": [], "finished": False, "ok": None}
    if not LOG.exists():
        return out
    lines = [ln.rstrip() for ln in LOG.read_text(encoding="utf-8", errors="ignore")
             .splitlines() if ln.strip()]
    out["started"] = datetime.fromtimestamp(LOG.stat().st_ctime)
    # 銘柄ごとの細かい進捗は読みにくいので、段階の見出しと％だけ残す
    keep = [ln for ln in lines if ("▶" in ln or "✅" in ln or "⚠" in ln or "Stage" in ln
                                   or "%" in ln or "完了" in ln or "失敗" in ln)]
    out["log"] = keep[-tail:] if keep else lines[-tail:]
    done = any("合計" in ln for ln in lines)
    out["finished"] = done and not out["running"]
    if out["finished"]:
        out["ok"] = not any("失敗:" in ln or "⚠️" in ln for ln in lines)
    return out
