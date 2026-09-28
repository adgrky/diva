"""東証の営業日カレンダー。

【なぜ自前で持つか】
権利落ち日は「その月の最終営業日の1営業日前」にほぼ固定されている
（実測: 2021〜2026年の25,085件のうち 23,643件＝94.2% がこの位置）。
これを土日だけで計算すると、祝日をまたぐ月でずれる。

**ずれる方向が問題**で、予測が実際より後ろにずれると
「その日までに買えば間に合う」と表示された日にはもう権利が落ちている。
配当を1回取り逃がすので、祝日をきちんと入れる。

東証の休みは「土日 ＋ 国民の祝日 ＋ 年末年始（12/31〜1/3）」。
"""
from __future__ import annotations

from datetime import date, timedelta
from functools import lru_cache


def _nth_monday(year: int, month: int, nth: int) -> date:
    d = date(year, month, 1)
    d += timedelta(days=(7 - d.weekday()) % 7)      # その月の最初の月曜
    return d + timedelta(days=7 * (nth - 1))


def _shunbun(year: int) -> int:
    """春分の日（1980〜2099年に有効な近似式）。"""
    return int(20.8431 + 0.242194 * (year - 1980) - (year - 1980) // 4)


def _shubun(year: int) -> int:
    """秋分の日（同上）。"""
    return int(23.2488 + 0.242194 * (year - 1980) - (year - 1980) // 4)


@lru_cache(maxsize=64)
def holidays(year: int) -> frozenset[date]:
    """その年の東証の休業日（土日を除く）。"""
    h = {
        date(year, 1, 1),                    # 元日
        _nth_monday(year, 1, 2),             # 成人の日
        date(year, 2, 11),                   # 建国記念の日
        date(year, 2, 23),                   # 天皇誕生日（2020年〜）
        date(year, _shunbun(year) and 3, _shunbun(year)),   # 春分の日
        date(year, 4, 29),                   # 昭和の日
        date(year, 5, 3), date(year, 5, 4), date(year, 5, 5),  # 憲法記念日・みどり・こども
        _nth_monday(year, 7, 3),             # 海の日
        date(year, 8, 11),                   # 山の日
        _nth_monday(year, 9, 3),             # 敬老の日
        date(year, _shubun(year) and 9, _shubun(year)),     # 秋分の日
        _nth_monday(year, 10, 2),            # スポーツの日
        date(year, 11, 3),                   # 文化の日
        date(year, 11, 23),                  # 勤労感謝の日
    }
    # 振替休日：日曜と重なった祝日は翌平日へ
    for d in sorted(h):
        if d.weekday() == 6:
            nxt = d + timedelta(days=1)
            while nxt in h or nxt.weekday() >= 5:
                nxt += timedelta(days=1)
            h.add(nxt)
    # 国民の休日：祝日に挟まれた平日（秋分の日と敬老の日の間など）
    for d in sorted(h):
        mid = d + timedelta(days=1)
        if (mid not in h and mid.weekday() < 5
                and mid + timedelta(days=1) in h):
            h.add(mid)
    # 東証だけの休み（大納会の翌日から大発会の前日まで）。祝日ではないので、
    # 振替休日・国民の休日の計算より後で足す。先に入れると、元日が日曜の年に
    # 振替が 1/2・1/3 を飛び越えて大発会の 1/4 を休みにしてしまう（2023年・2034年）。
    h |= {date(year, 12, 31), date(year, 1, 2), date(year, 1, 3)}
    return frozenset(h)


def is_trading_day(d: date) -> bool:
    return d.weekday() < 5 and d not in holidays(d.year)


def prev_trading_day(d: date) -> date:
    d -= timedelta(days=1)
    while not is_trading_day(d):
        d -= timedelta(days=1)
    return d


def shift_trading_days(d: date, n: int) -> date:
    """n 営業日ずらす（マイナスで過去）。"""
    step = 1 if n > 0 else -1
    for _ in range(abs(n)):
        d += timedelta(days=step)
        while not is_trading_day(d):
            d += timedelta(days=step)
    return d


def last_trading_day(year: int, month: int) -> date:
    """その月の最終営業日。"""
    nxt = date(year + (month == 12), (month % 12) + 1, 1)
    d = nxt - timedelta(days=1)
    while not is_trading_day(d):
        d -= timedelta(days=1)
    return d


def trading_days_before_month_end(d: date) -> int:
    """その日が、月の最終営業日の何営業日前か。"""
    end = last_trading_day(d.year, d.month)
    n = 0
    cur = end
    while cur > d and n < 40:
        cur = prev_trading_day(cur)
        n += 1
    return n
