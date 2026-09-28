"""設定と共通パス。

Streamlit に依存しない。ヘッドレスなスクリプトからも同じ設定を読む。
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import yaml

APP_DIR = Path(__file__).resolve().parent.parent


@lru_cache(maxsize=4)
def _read_config(_stamp: float) -> dict:
    with open(APP_DIR / "config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_config() -> dict:
    """設定を読む。**config.yaml を書き換えたら、その場で効く。**

    以前はプロセスのあいだ1回だけ読んで固めていた。そのせいで、閾値や重みを
    直してもアプリを再起動するまで反映されず、しかも画面には何の手がかりも
    出なかった（実測: 大人買いラインの段を書き換えたのに、古い段が表示され続けた）。
    「コードに数値を直書きしない」と決めている以上、設定を直したら効かないと困る。

    ファイルの更新時刻をキーにして読み直す。中身が変わっていなければ
    キャッシュがそのまま返るので、読み込みの負担は増えない。
    """
    try:
        stamp = (APP_DIR / "config.yaml").stat().st_mtime
    except OSError:
        stamp = 0.0
    return _read_config(stamp)


def db_path(config: dict | None = None) -> Path:
    config = config or load_config()
    p = APP_DIR / config["paths"]["db"]
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def legacy_dir(config: dict | None = None) -> Path:
    config = config or load_config()
    raw = Path(config["paths"]["legacy_dir"]).expanduser()
    # 絶対パスならそのまま。相対パスだけアプリの場所からたどる。
    return raw.resolve() if raw.is_absolute() else (APP_DIR / raw).resolve()


_SECRET_KEYS = ("EDINET_API_KEY", "NTFY_TOPIC", "TURSO_DATABASE_URL", "TURSO_AUTH_TOKEN",
                "APP_PASSWORD")


def bridge_secrets_to_env() -> None:
    """secrets.toml の中身を os.environ に流し込む。

    **これを通していないと、クラウド(Turso)ではなくパソコンのDBに書いてしまう。**
    `store.connect()` は TURSO_DATABASE_URL が環境変数にあるかどうかだけで
    書き込み先を決めるので、環境変数が無いスクリプトは黙ってローカルに書く。

    実測（2026-09-23）: weekly_scan.py がこれを呼んでおらず、2026-09-20 に
    クラウド同期を入れて以降、毎週のスキャン結果がアプリに一度も届いていなかった。
    アプリ側のスコアは 2026-09-19 のまま止まり、配当データを直しても
    画面の配当性向が変わらなかった。エラーは出ない。ただ古いまま見え続ける。

    そのため **store.connect() から自動で呼ぶ**ようにしてある。スクリプト側の
    呼び忘れで同じことが起きないようにするため。

    読み取りは tomllib（標準ライブラリ）で直接行う。Streamlit 経由にすると
    modules/ が Streamlit に依存してしまい、GitHub Actions から動かせなくなる。
    """
    if all(k in os.environ for k in _SECRET_KEYS):
        return
    import tomllib
    for path in (APP_DIR / ".streamlit" / "secrets.toml",
                 Path.home() / ".streamlit" / "secrets.toml"):
        if not path.exists():
            continue
        try:
            with open(path, "rb") as f:
                data = tomllib.load(f)
        except Exception:
            continue
        for key in _SECRET_KEYS:
            if key not in os.environ and isinstance(data.get(key), (str, int, float)):
                os.environ[key] = str(data[key])
