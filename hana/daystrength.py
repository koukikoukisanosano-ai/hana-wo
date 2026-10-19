"""店全体の「日の強さ」。日付末尾（旧イベント日かどうか）と曜日ごとに、
ハナハナ全台の「判定設定が高設定だった台の割合」と平均差枚を集計し、今日と今後1週間の強さを見積もる。

63日分のデータで、0・5のつく日は全台平均+172枚・高設定っぽい台36%（その他の日は−64枚・25%）。
その日より前のデータだけで見積もった強さと実際の強さの相関は0.37（42日で検証）。
台ごとの差より、日による差のほうがはっきり出ている。
"""
from __future__ import annotations

from datetime import date as Date, timedelta

import numpy as np
import pandas as pd

WEEKDAY = "月火水木金土日"


def _day_table(hist: pd.DataFrame, high_min: int) -> pd.DataFrame:
    h = hist.dropna(subset=["judge"])
    h = h[h["games"] >= 1000]
    t = h.assign(hi=(h["judge"] >= high_min).astype(float), plus=(h["diff"] > 0).astype(float)) \
         .groupby("date").agg(hi=("hi", "mean"), diff=("diff", "mean"), plus=("plus", "mean"), n=("hi", "size")).reset_index()
    t = t[t["n"] >= 15]
    t["date"] = pd.to_datetime(t["date"])
    return t


def group_of(d, event_digits) -> str:
    return "旧イベ日" if d.day % 10 in event_digits else "通常日"


def estimate(hist: pd.DataFrame, target: Date, event_digits, high_min: int = 5, days: int = 7) -> dict:
    t = _day_table(hist, high_min)
    if len(t) < 14:
        return {}
    t["g"] = [group_of(d, event_digits) for d in t["date"]]
    t["wd"] = t["date"].dt.weekday
    base = float(t["hi"].mean())
    base_diff = float(t["diff"].mean())

    def est(d: Date) -> dict:
        g = group_of(d, event_digits)
        gr = t[t["g"] == g]
        rate = (gr["hi"].sum() + 5 * base) / (len(gr) + 5)
        wr = t[t["wd"] == d.weekday()]
        w = (wr["hi"].sum() + 10 * base) / (len(wr) + 10)
        rate = rate * w / base
        dd = t[t["g"] == g]["diff"]
        return {"date": d, "group": g, "rate": float(rate), "ratio": float(rate / base),
                "g_days": int(len(gr)), "g_rate": float(gr["hi"].mean()) if len(gr) else np.nan,
                "g_diff": float(dd.mean()) if len(dd) else np.nan,
                "g_plus": float(gr["plus"].mean()) if len(gr) else np.nan,
                "w_rate": float(wr["hi"].mean()) if len(wr) else np.nan, "w_days": int(len(wr))}

    today = est(target)
    ahead = [est(target + timedelta(days=k)) for k in range(days)]
    return {"base": base, "base_diff": base_diff, "n_days": int(len(t)), "today": today, "ahead": ahead}


def level(ratio: float) -> tuple[str, str]:
    if ratio >= 1.2:
        return "強い", "hot"
    if ratio >= 1.05:
        return "やや強い", "warm"
    if ratio > 0.92:
        return "普通", "norm"
    return "弱い", "cold"
