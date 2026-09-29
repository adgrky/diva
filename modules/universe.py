"""JPX 公式の上場銘柄一覧からユニバースを作る。

JPX の Excel は「コード・銘柄名・33業種区分・市場区分・規模区分」を持っており、
日本株の業種分類としてはこれが唯一の正。yfinance の英語 sector は使わない。

実ファイル名・パスは JPX 側で不定期に変わる（過去 data_j.xls → data_j.xlsx で
旧 URL が 404 になった）ため、掲載ページからリンクを解決してから取得する。
"""
from __future__ import annotations

import io
import re

import pandas as pd
import requests

_JPX_PAGE_URL = "https://www.jpx.co.jp/markets/statistics-equities/misc/01.html"
_JPX_BASE = "https://www.jpx.co.jp"
_HTTP_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36"
    )
}


def _resolve_jpx_list_url() -> str:
    resp = requests.get(_JPX_PAGE_URL, headers=_HTTP_HEADERS, timeout=30)
    resp.raise_for_status()
    matches = re.findall(r'href="([^"]*data_j\.xls[x]?)"', resp.text)
    if not matches:
        raise RuntimeError("JPX ページに data_j ファイルのリンクが見つからない")
    href = matches[0]
    return href if href.startswith("http") else _JPX_BASE + href


def to_ticker(code: str) -> str:
    """証券コード -> yfinance シンボル。4桁・英数字混在（例 135A）の両方に対応。"""
    return f"{str(code).strip().upper()}.T"


def fetch_universe(markets: list[str] | None = None) -> pd.DataFrame:
    """JPX から上場銘柄一覧を取得する。

    Returns
    -------
    pd.DataFrame
        columns: ticker, code, name, sector33, market, scale
    """
    url = _resolve_jpx_list_url()
    resp = requests.get(url, headers=_HTTP_HEADERS, timeout=120)
    resp.raise_for_status()
    df = pd.read_excel(io.BytesIO(resp.content))

    required = ["コード", "銘柄名", "33業種区分", "市場・商品区分"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"JPX Excel の列構成が変わった可能性がある: {missing} / 実際={list(df.columns)}")

    out = pd.DataFrame({
        "code": df["コード"].astype(str).str.strip(),
        "name": df["銘柄名"].astype(str).str.strip(),
        "sector33": df["33業種区分"].astype(str).str.strip(),
        "market": df["市場・商品区分"].astype(str).str.strip(),
        "scale": df.get("規模区分", pd.Series([""] * len(df))).astype(str).str.strip(),
    })
    if markets:
        out = out[out["market"].isin(markets)]
    # ETF・REIT・優先出資証券などは 33業種区分が「-」になる
    out = out[(out["code"] != "") & (out["sector33"] != "-")]
    out["ticker"] = out["code"].map(to_ticker)
    return out[["ticker", "code", "name", "sector33", "market", "scale"]].reset_index(drop=True)

# ─── 上場廃止（2026-09-26 追加）
# 上の一覧は「前月末時点」なので、月の途中で廃止された銘柄が翌月まで残る。
# 実測: 1909.T（9/14 廃止）・2180.T（9/16 廃止）が8月末の一覧に載ったまま、
# yfinance から終値 約160億円・出来高0 の壊れた値が返ってきていた。
# JPX の上場廃止銘柄ページには「廃止予定」も載るので、廃止日で区切って使う。
_JPX_DELISTED_URL = "https://www.jpx.co.jp/listing/stocks/delisted/index.html"


def fetch_delisted() -> pd.DataFrame:
    """JPX の上場廃止銘柄一覧（予定を含む）。

    Returns
    -------
    pd.DataFrame
        columns: ticker, name, delist_date(Timestamp), reason
    """
    import html as _html

    resp = requests.get(_JPX_DELISTED_URL, headers=_HTTP_HEADERS, timeout=30)
    resp.raise_for_status()
    resp.encoding = resp.apparent_encoding or "utf-8"
    rows = []
    for tr in re.findall(r"<tr[^>]*>(.*?)</tr>", resp.text, re.S):
        cells = [_html.unescape(re.sub(r"<[^>]+>", "", c)).strip()
                 for c in re.findall(r"<td[^>]*>(.*?)</td>", tr, re.S)]
        if len(cells) < 5 or not re.fullmatch(r"\d{4}/\d{2}/\d{2}", cells[0]):
            continue
        rows.append({"ticker": to_ticker(cells[2]), "name": cells[1],
                     "delist_date": pd.Timestamp(cells[0].replace("/", "-")),
                     "reason": cells[4]})
    if not rows:
        raise RuntimeError("JPX 上場廃止ページの表が読めない（ページの作りが変わった可能性）")
    return pd.DataFrame(rows)


def delisted_to_purge(listed: set[str], known: set[str], delisted: pd.DataFrame | None,
                      protected: set[str], today: pd.Timestamp) -> tuple[set[str], set[str]]:
    """消してよい上場廃止銘柄と、廃止したが保有・ウォッチ中で残す銘柄を返す。

    listed    … JPX の最新一覧にある銘柄
    known     … いまDBの一覧（universe）にある銘柄
    delisted  … fetch_delisted() の結果（取れなかったら None）
    protected … 保有・ウォッチリストの銘柄。売却の記録やカルテに要るので消さない

    廃止日が今日より後（予定）の銘柄はまだ消さない。取引はできるので。
    """
    gone = set(known) - set(listed)
    if delisted is not None and not delisted.empty:
        gone |= set(delisted.loc[delisted["delist_date"] <= today, "ticker"]) & set(known)
    return gone - protected, gone & protected
