"""EDINET（金融庁の電子開示システム）から有価証券報告書を取る。

【なぜ必要か】
yfinance の財務は4〜5期分しか返らず、しかも日本基準の指標（配当性向・自己資本比率）は
欠損や誤りがある（実測で配当性向480%という値を確認）。増配の持続性を判定するには
正確な長期データが要る。

【なぜ有報が効率的か】
有価証券報告書の冒頭「主要な経営指標等の推移」には、**5年分**の
    1株当たり当期純利益 / 1株当たり配当額 / 配当性向 / 自己資本利益率 /
    自己資本比率 / 営業活動によるキャッシュ・フロー / 売上高 / 従業員数
が構造化データ（XBRL）で入っている。1社1ファイル（約300KB）で5年分が揃う。
さらに「事業の内容」が日本語で入っているので、何をやっている会社かも取れる。

APIは無料。利用にはアカウント登録とAPIキーが必要（環境変数 EDINET_API_KEY）。
"""
from __future__ import annotations

import io
import json
import os
import re
import time
import urllib.parse
import urllib.request
import zipfile
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

_BASE = "https://api.edinet-fsa.go.jp/api/v2"
_UA = {"User-Agent": "diva/1.0"}
_DOC_TYPE_YUHO = "120"          # 有価証券報告書

# 「主要な経営指標等の推移」の要素名 → こちらでの列名。
# IFRS を採用している会社は要素名が別（末尾が IFRSSummaryOfBusinessResults）。
# 実測: INPEX は IFRS のため日本基準の要素には単体の値しか入っておらず、
# IFRS 版を見ないと連結のEPSが2年分しか取れなかった。
# 同じ列に複数の要素が対応する場合は、先に書いたほうが優先される。
_SUMMARY_ELEMENTS = {
    # 売上・収益
    "NetSalesSummaryOfBusinessResults": "sales",
    "RevenueIFRSSummaryOfBusinessResults": "sales",
    # 利益
    "OrdinaryIncomeLossSummaryOfBusinessResults": "ordinary_income",
    "ProfitLossBeforeTaxIFRSSummaryOfBusinessResults": "ordinary_income",
    "ProfitLossAttributableToOwnersOfParentSummaryOfBusinessResults": "net_income",
    "ProfitLossAttributableToOwnersOfParentIFRSSummaryOfBusinessResults": "net_income",
    "NetIncomeLossSummaryOfBusinessResults": "net_income",
    # 1株あたり
    "BasicEarningsLossPerShareSummaryOfBusinessResults": "eps",
    "BasicEarningsLossPerShareIFRSSummaryOfBusinessResults": "eps",
    # 注意: 有報の1株配当は株式分割が調整されていない。増配率の計算には使わず、
    # 表示のみに使う（成長率は分割調整済みの配当履歴から計算する）。
    "DividendPaidPerShareSummaryOfBusinessResults": "dps",
    "PayoutRatioSummaryOfBusinessResults": "payout_ratio",
    # 財務の健全性
    "RateOfReturnOnEquitySummaryOfBusinessResults": "roe",
    "RateOfReturnOnEquityIFRSSummaryOfBusinessResults": "roe",
    "EquityToAssetRatioSummaryOfBusinessResults": "equity_ratio",
    # 注意: EquityToAssetRatioIFRSSummaryOfBusinessResults は名前に反して
    # 「1株当たり親会社所有者帰属持分（BPS）」が入っている（実測: 三菱商事 2578.33、
    # NTT 119.47）。自己資本比率はこちらが正しい。
    "RatioOfOwnersEquityToGrossAssetsIFRSSummaryOfBusinessResults": "equity_ratio",
    "NetAssetsSummaryOfBusinessResults": "net_assets",
    "EquityAttributableToOwnersOfParentIFRSSummaryOfBusinessResults": "net_assets",
    "TotalAssetsSummaryOfBusinessResults": "total_assets",
    "TotalAssetsIFRSSummaryOfBusinessResults": "total_assets",
    # キャッシュフロー
    "NetCashProvidedByUsedInOperatingActivitiesSummaryOfBusinessResults": "operating_cf",
    "CashFlowsFromUsedInOperatingActivitiesIFRSSummaryOfBusinessResults": "operating_cf",
    "NumberOfEmployees": "employees",
}

# contextRef の接頭辞 → 何年前か
_PERIOD_OFFSET = {
    "CurrentYear": 0, "Prior1Year": 1, "Prior2Year": 2,
    "Prior3Year": 3, "Prior4Year": 4,
}


class EdinetError(RuntimeError):
    pass


def api_key() -> str:
    """APIキーを環境変数から取る。無ければ stock-recommender の .env も見る。"""
    key = os.getenv("EDINET_API_KEY")
    if key:
        return key
    # 同じキーを2箇所で管理したくないので、既に登録済みの場所を参照する
    fallback = Path.home() / "dev" / "stock-recommender" / ".env"
    if fallback.exists():
        for line in fallback.read_text(encoding="utf-8").splitlines():
            if line.startswith("EDINET_API_KEY="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise EdinetError(
        "EDINET_API_KEY が設定されていません。\n"
        "  https://api.edinet-fsa.go.jp/api/auth/index.aspx?mode=1 で無料登録し、\n"
        "  このプロジェクトの .env に EDINET_API_KEY=... を書いてください。"
    )


def _get(path: str, params: dict, timeout: int = 180) -> bytes:
    params = {**params, "Subscription-Key": api_key()}
    url = f"{_BASE}/{path}?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers=_UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def list_documents(day: date, retries: int = 2) -> list[dict]:
    """その日に提出された有価証券報告書（上場企業のみ）を返す。"""
    for attempt in range(retries + 1):
        try:
            data = json.loads(_get("documents.json",
                                   {"date": day.isoformat(), "type": "2"}, timeout=60))
            break
        except Exception:
            if attempt == retries:
                return []
            time.sleep(2 * (attempt + 1))
    out = []
    for r in data.get("results") or []:
        if r.get("docTypeCode") != _DOC_TYPE_YUHO:
            continue
        sec = r.get("secCode")
        if not sec:
            continue
        out.append({
            # secCode は5桁（証券コード4桁＋末尾0）。末尾を落として4桁に戻す。
            "code": str(sec)[:4],
            "doc_id": r["docID"],
            "filer_name": r.get("filerName"),
            "period_end": r.get("periodEnd"),
            "submit_date": day.isoformat(),
        })
    return out


def build_index(days_back: int = 400, end: date | None = None,
                progress=None) -> pd.DataFrame:
    """直近N日ぶんの提出一覧を舐めて、銘柄コード→有報 のひも付けを作る。

    有報は決算期末から3ヶ月以内に出る。3月決算が大半なので6月に集中するが、
    他の決算期もあるため1年ぶんを見る。
    """
    end = end or date.today()
    rows = []
    for i in range(days_back):
        day = end - timedelta(days=i)
        if progress and i % 20 == 0:
            progress(i / days_back, f"提出一覧 {day} まで遡り中（{len(rows)} 件）")
        rows.extend(list_documents(day))
        time.sleep(0.15)   # EDINET に負荷をかけない
    if progress:
        progress(1.0, f"提出一覧 完了（{len(rows)} 件）")
    if not rows:
        return pd.DataFrame(columns=["code", "doc_id", "filer_name", "period_end", "submit_date"])
    df = pd.DataFrame(rows)
    # 同じ会社が複数年ぶん出てくるので、期末が新しいものを残す
    return df.sort_values("period_end").drop_duplicates("code", keep="last").reset_index(drop=True)


def _cache_path(doc_id: str) -> Path:
    d = Path(__file__).resolve().parent.parent / "data" / "edinet_cache"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{doc_id}.xbrl.gz"


def download_xbrl(doc_id: str, retries: int = 2, use_cache: bool = True) -> str | None:
    """有報のXBRL本体（テキスト）を取る。

    1社あたり1.8MB・全体で1時間以上かかるので、圧縮してディスクに残す。
    解析の仕方を直したときに、取り直さずに読み直せるようにするため。
    """
    import gzip
    cache = _cache_path(doc_id)
    if use_cache and cache.exists():
        try:
            return gzip.decompress(cache.read_bytes()).decode("utf-8", "ignore")
        except Exception:
            pass

    for attempt in range(retries + 1):
        try:
            blob = _get(f"documents/{doc_id}", {"type": "1"})
            break
        except Exception:
            if attempt == retries:
                return None
            time.sleep(3 * (attempt + 1))
    try:
        z = zipfile.ZipFile(io.BytesIO(blob))
    except zipfile.BadZipFile:
        return None
    cands = [n for n in z.namelist() if n.endswith(".xbrl") and "PublicDoc" in n]
    if not cands:
        return None
    text = z.read(cands[0]).decode("utf-8", "ignore")
    try:
        cache.write_bytes(gzip.compress(text.encode("utf-8"), 6))
    except Exception:
        pass
    return text


_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"[ \t　]+")


def parse_business(raw: str, limit: int = 600) -> str:
    """「事業の内容」を日本語の平文で取り出す。"""
    m = re.search(r"<jpcrp[^ >]*:DescriptionOfBusinessTextBlock[^>]*>(.*?)</jpcrp[^ >]*:"
                  r"DescriptionOfBusinessTextBlock>", raw, re.S)
    if not m:
        return ""
    import html
    text = html.unescape(m.group(1))
    text = _TAG_RE.sub(" ", text)
    text = html.unescape(text)
    text = _WS_RE.sub(" ", text).replace("\n", " ").strip()
    return text[:limit]


# 配当方針のキーワード。日本株の「増配意思」は、実績よりも会社が何を約束したかに出る。
# 累進配当（減らさないと約束）や DOE（純資産に対して配当を決める）を掲げた会社は、
# 利益が一時的に落ちても配当を維持・増加させる圧力が内側から働く。
# 「累進的な配当」「累進的に」の言い回しと、全角の「ＤＯＥ」を以前は取りこぼしていた
# （実測: 有報1,272社のうち累進26社・DOE32社。コンコルディア・双日・しずおかFGなど）。
_POLICY_PATTERNS = {
    "累進配当": (r"累進(的)?(な)?配当|累進的に|減配(は)?(行わ|いたしま|しま)(ない|せん)"
               r"|前年度の配当金を下限", 0.30),
    "DOE（純資産配当率）": (r"DOE|ＤＯＥ|株主資本配当率|純資産配当率", 0.25),
    "配当性向の目標": (r"配当性向[^。]{0,20}?(\d{2})\s*[%％][^。]{0,10}?(以上|目標|とし|を目指)", 0.20),
    "連続増配を明言": (r"連続(して)?増配|増配を(継続|続け)", 0.15),
    "安定配当": (r"安定(的)?(な|に)?配当", 0.05),
    "自己株式の取得": (r"自己株式の取得|自社株買い", 0.05),
}


def parse_dividend_policy(raw: str, limit: int = 500) -> tuple[str, dict, float]:
    """「配当政策」の本文と、そこから読み取れる方針のフラグ・スコアを返す。

    Returns
    -------
    (本文, {方針名: True}, 0〜1のスコア)
    """
    m = re.search(r"<jpcrp[^ >]*:DividendPolicyTextBlock[^>]*>(.*?)"
                  r"</jpcrp[^ >]*:DividendPolicyTextBlock>", raw, re.S)
    if not m:
        return "", {}, 0.0
    import html
    text = html.unescape(m.group(1))
    text = _TAG_RE.sub(" ", text)
    text = html.unescape(text)
    text = _WS_RE.sub(" ", text).replace("\n", " ").strip()

    flags, score = policy_from_text(text)
    return text[:limit], flags, score


def policy_from_text(text: str) -> tuple[dict, float]:
    """配当政策の本文から、方針のフラグと 0〜1 のスコアを出す。

    保存済みの本文から何度でも作り直せるよう、取得とは切り離してある。
    言い回しの取りこぼしを直したとき、有報を取り直さずに全社へ反映できる。
    """
    flags, score = {}, 0.0
    for label, (pattern, weight) in _POLICY_PATTERNS.items():
        if re.search(pattern, text or ""):
            flags[label] = True
            score += weight
    return flags, min(score, 1.0)


def parse_summary(raw: str, period_end: str) -> pd.DataFrame:
    """「主要な経営指標等の推移」から5年分の指標を取り出す。

    【連結と単体を絶対に混ぜない】
    有報の5年表は「連結」と「提出会社（単体）」の2本立てで、指標によって
    どちらか片方しか載っていないことがある。指標ごとに「あるほうを使う」と、
    連結と単体が同じ行に混ざって無意味な数字になる。

    実測した事故: INPEX（1605）は EPS が連結で 153.87 円、配当性向が単体基準で
    5.240（＝524%）と載っており、そのまま混ぜたため「配当性向70%超」で
    足切りされていた。単体EPSは 9.16 円なので 48円÷9.16円＝524% は単体としては
    正しく、連結ベースなら 48÷153.87＝31% で基準内。

    そこで、会社ごとに基準をひとつ選ぶ。EPS（または純利益）が3年以上そろって
    いるほうを採用し、すべての指標をその基準から取る。足りない指標は埋めない。
    配当性向だけは、採用した基準のEPSと1株配当から自前で計算できるので補う。
    """
    end_year = int(str(period_end)[:4]) if period_end else None

    # (col, basis, year) -> value
    facts: dict[tuple[str, str, int], float] = {}
    for element, col in _SUMMARY_ELEMENTS.items():
        pattern = rf'<jpcrp[^ >]*:{element}[^>]*contextRef="([^"]+)"[^>]*>([^<]*)<'
        for ctx, val in re.findall(pattern, raw):
            prefix = next((p for p in _PERIOD_OFFSET if ctx.startswith(p)), None)
            if prefix is None:
                continue
            basis = "単体" if "NonConsolidatedMember" in ctx else "連結"
            year = (end_year - _PERIOD_OFFSET[prefix]) if end_year else -_PERIOD_OFFSET[prefix]
            try:
                num = float(str(val).replace(",", "").strip())
            except ValueError:
                continue
            # 同じ (列, 基準, 年) に複数の要素が当たることがある（日本基準とIFRSの併記）。
            # 先に定義した要素を優先する。
            facts.setdefault((col, basis, year), num)

    if not facts:
        return pd.DataFrame()

    def _n(basis: str) -> int:
        return sum(1 for (c, b, _) in facts if b == basis and c in ("eps", "net_income"))

    basis = "連結" if _n("連結") >= 3 else "単体"
    years = sorted({y for (_, b, y) in facts if b == basis})
    if not years:
        basis = "単体" if basis == "連結" else "連結"
        years = sorted({y for (_, b, y) in facts if b == basis})
    if not years:
        return pd.DataFrame()

    rows = []
    for y in years:
        rec = {"fiscal_year": y, "basis": basis}
        for col in set(_SUMMARY_ELEMENTS.values()):
            rec[col] = facts.get((col, basis, y))
        # 配当性向を dps ÷ eps で補うことはしない。有報の1株配当は
        # **株式分割が調整されていない**（実測: NTT は 115円 → 120円 → 5.10円 と
        # 25:1 の分割前後で連続していない）。一方 EPS は調整済みのことがあり、
        # 割ると無意味な値になる。開示されている配当性向が無ければ空のままにして、
        # 株価側の分割調整済み配当から計算した値（attach_fundamentals）を使う。
        pass
        rows.append(rec)

    df = pd.DataFrame(rows)
    # 1株配当は提出会社（単体）側にしか載らないことが多い。基準に関係なく同じ値なので補う。
    if df["dps"].isna().all():
        other = "単体" if basis == "連結" else "連結"
        df["dps"] = [facts.get(("dps", other, y)) for y in years]
    return _sanitize(df).sort_values("fiscal_year").reset_index(drop=True)


# 比率として妥当な範囲。要素名の意味が taxonomy の版で変わることがあるため、
# 値そのものを見て弾く（実測: 自己資本比率の要素に1株当たり純資産が入っていた）。
_RATIO_RANGE = {
    "payout_ratio": (-5.0, 5.0),
    "roe": (-2.0, 2.0),
    "equity_ratio": (0.0, 1.0),
}


def _sanitize(df: pd.DataFrame) -> pd.DataFrame:
    """比率の列が明らかにおかしい値を落とす。"""
    for col, (lo, hi) in _RATIO_RANGE.items():
        if col in df.columns:
            v = pd.to_numeric(df[col], errors="coerce")
            df[col] = v.where((v >= lo) & (v <= hi))
    return df


def fetch_company(doc_id: str, period_end: str) -> dict:
    """1社ぶんの5年指標・事業内容・配当方針をまとめて取る。"""
    raw = download_xbrl(doc_id)
    if raw is None:
        return {"summary": pd.DataFrame(), "business": "", "policy": "",
                "policy_flags": {}, "policy_score": 0.0}
    policy, flags, score = parse_dividend_policy(raw)
    return {
        "summary": parse_summary(raw, period_end),
        "business": parse_business(raw),
        "policy": policy,
        "policy_flags": flags,
        "policy_score": score,
    }
