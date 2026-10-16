"""みんレポ（min-repo.com）からの取得。

・オーギヤWOのページ: https://min-repo.com/tag/wo/ （2021/10〜。2026/7/30 以降は更新が止まっている）
・日別ページ → 機種ページ(?kishu=機種名) に台番ごとの 差枚/G数/合成確率
・台番ページ(?num=台番) に BB/RB 回数

使い方は2通り:
  1) 補完（precise）: アナスロに無い新しい日があれば、台番ページまで開いて BB/RB を正確に取る
  2) 過去分（estimate）: アナスロより古い日を、機種ページだけ（1日1〜2ページ）で取り、
     BB/RB は 差枚・G数・合成確率 から推定する（推定値として区別して保存）
"""
from __future__ import annotations

import logging
import re
import time
from datetime import date as Date
from urllib.parse import quote, urljoin

import numpy as np
from bs4 import BeautifulSoup

from ..http import Fetcher, RateLimited
from .anaslo import to_int

log = logging.getLogger("hana.minrepo")

_MD = re.compile(r"(?:(\d{4})/)?(\d{1,2})/(\d{1,2})\((.)\)")
_PUB = re.compile(r"(\d{4})年(\d{1,2})月(\d{1,2})日")
_WD = "月火水木金土日"


def infer_year(month: int, day: int, weekday_char: str, today: Date) -> int | None:
    for y in (today.year, today.year - 1, today.year - 2):
        try:
            d = Date(y, month, day)
        except ValueError:
            continue
        if d <= today and _WD[d.weekday()] == weekday_char:
            return y
    return None


def parse_tag_page(html: str, base_url: str, today: Date) -> tuple[list[dict], list[tuple[str, str]], int]:
    """タグページ → (日別サマリー, [(date, 日別URL)], 最終ページ番号)"""
    soup = BeautifulSoup(html, "lxml")
    days, links = [], []
    for tr in soup.find_all("tr"):
        tds = tr.find_all("td")
        if len(tds) < 4:
            continue
        m = _MD.search(tds[0].get_text())
        a = tds[0].find("a", href=True)
        if not m:
            continue
        y = int(m.group(1)) if m.group(1) else infer_year(int(m.group(2)), int(m.group(3)), m.group(4), today)
        if not y:
            continue
        d = f"{y:04d}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"
        days.append({"date": d, "total_diff": to_int(tds[1].get_text()), "avg_diff": to_int(tds[2].get_text()),
                     "avg_games": to_int(tds[3].get_text())})
        if a and re.search(r"min-repo\.com/\d+/?$", urljoin(base_url, a["href"])):
            links.append((d, urljoin(base_url, a["href"])))
    last_page = 1
    for a in soup.find_all("a", href=True):
        m = re.search(r"/page/(\d+)/?$", a["href"])
        if m:
            last_page = max(last_page, int(m.group(1)))
    return days, links, last_page


def parse_day_models(html: str, keyword: str) -> tuple[list[str], list[dict]]:
    """日別ページ → (対象機種名リスト, バラエティ行[{unit, model, diff, games}])"""
    soup = BeautifulSoup(html, "lxml")
    models = []
    for a in soup.find_all("a", href=True):
        if "kishu=" in a["href"] and "kishu=all" not in a["href"] and keyword in a.get_text():
            name = a.get_text(strip=True)
            if name not in models:
                models.append(name)
    variety = []
    for table in soup.find_all("table"):
        head = [th.get_text(strip=True) for th in table.find_all("th")]
        if head[:3] == ["機種", "台番", "差枚"]:
            for tr in table.find_all("tr")[1:]:
                tds = [td.get_text(strip=True) for td in tr.find_all("td")]
                if len(tds) >= 4 and keyword in tds[0] and tds[0] not in models:
                    variety.append({"model": tds[0], "unit": to_int(tds[1]), "diff": to_int(tds[2]),
                                    "games": to_int(tds[3])})
    return models, variety


def parse_kishu_page(html: str) -> list[dict]:
    """機種ページ → [{unit, diff, games, total}]（total は合成確率から逆算したボーナス合計）"""
    soup = BeautifulSoup(html, "lxml")
    out = []
    for li in soup.select("ul.slump_list > li"):
        a = li.find("a")
        table = li.find("table")
        if not a or not table:
            continue
        unit = to_int(a.get_text())
        tds = [td.get_text(strip=True) for td in table.find_all("td")]
        if unit is None or len(tds) < 3:
            continue
        diff, games = to_int(tds[0]), to_int(tds[1])
        m = re.search(r"1/([\d.]+)", tds[2])
        total = None
        if m and games:
            denom = float(m.group(1))
            total = int(round(games / denom)) if denom > 0 else 0
        out.append({"unit": unit, "diff": diff, "games": games, "total": total})
    return out


def parse_unit_page(html: str) -> dict | None:
    """台番ページ → {model, diff, games, bb, rb}"""
    soup = BeautifulSoup(html, "lxml")
    res = {}
    for table in soup.find_all("table"):
        rows = table.find_all("tr")
        if len(rows) < 2:
            continue
        head = [c.get_text(strip=True) for c in rows[0].find_all(["th", "td"])]
        vals = [c.get_text(strip=True) for c in rows[1].find_all(["th", "td"])]
        if head[:2] == ["機種", "差枚"] and len(vals) >= 3:
            res.update(model=vals[0], diff=to_int(vals[1]), games=to_int(vals[2]))
        elif head[:2] == ["BB", "RB"] and len(vals) >= 2:
            res.update(bb=to_int(vals[0]), rb=to_int(vals[1]))
    return res if "bb" in res else None


def post_date(html: str, fallback: str) -> str:
    """日別ページの本当の日付（タイトルの月日 + 投稿日の年）"""
    soup = BeautifulSoup(html, "lxml")
    title = soup.title.get_text() if soup.title else ""
    m = _MD.search(title)
    p = _PUB.search(soup.get_text(" ", strip=True))
    if m and p:
        py, pm = int(p.group(1)), int(p.group(2))
        mm, dd = int(m.group(2)), int(m.group(3))
        y = py - 1 if mm > pm else py
        return f"{y:04d}-{mm:02d}-{dd:02d}"
    return fallback


# ---- BB/RB の推定（過去分用） ----
DEFAULT_COEF = (242.0, 131.0, -1.19)   # 差枚 ≒ a*BB + b*RB + c*G（WOの実データで校正した値）


def fit_payout_coef(df) -> tuple[float, float, float]:
    d = df.dropna(subset=["bb", "rb", "games", "diff"])
    d = d[d["games"] >= 1000]
    if len(d) < 200:
        return DEFAULT_COEF
    X = d[["bb", "rb", "games"]].to_numpy(float)
    y = d["diff"].to_numpy(float)
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    a, b, c = coef
    if not (150 < a < 350 and 60 < b < 200 and -3 < c < 0):
        return DEFAULT_COEF
    return float(a), float(b), float(c)


def estimate_bb_rb(diff, games, total, coef) -> tuple[int | None, int | None]:
    if diff is None or games is None or total is None:
        return None, None
    a, b, c = coef
    bb = (diff - c * games - b * total) / (a - b)
    bb = int(round(min(max(bb, 0), total)))
    return bb, total - bb


# ---- 取得処理 ----
class MinRepo:
    def __init__(self, db, fetcher: Fetcher, tag_url: str, keyword: str, interval: float = 6):
        self.db, self.f, self.tag_url, self.kw, self.interval = db, fetcher, tag_url.rstrip("/") + "/", keyword, interval

    def refresh_index(self, today: Date, all_pages: bool = False) -> list[tuple[str, str]]:
        html = self.f.get(self.tag_url, interval=self.interval)
        days, links, last = parse_tag_page(html, self.tag_url, today)
        if all_pages:
            for p in range(2, last + 1):
                h = self.f.get(f"{self.tag_url}page/{p}/", referer=self.tag_url, interval=self.interval)
                d2, l2, _ = parse_tag_page(h, self.tag_url, today)
                days += d2
                links += l2
        self.db.upsert_hall_days(days, "minrepo")
        self.db.upsert_index(links, "minrepo")
        return links

    def fetch_day(self, date: str, url: str, precise: bool, coef=DEFAULT_COEF) -> int:
        html = self.f.get(url, referer=self.tag_url, interval=self.interval)
        date = post_date(html, date)
        models, variety = parse_day_models(html, self.kw)
        rows = []
        for model in models:
            kurl = f"{url.rstrip('/')}/?kishu={quote(model)}"
            units = parse_kishu_page(self.f.get(kurl, referer=url, interval=self.interval))
            for u in units:
                rows.append({"date": date, "unit": u["unit"], "model": model, "games": u["games"],
                             "diff": u["diff"], "total": u["total"]})
        for v in variety:
            rows.append({"date": date, "unit": v["unit"], "model": v["model"], "games": v["games"],
                         "diff": v["diff"], "total": None})
        if precise:
            out = []
            for r in rows:
                h = self.f.get(f"{url.rstrip('/')}/?num={r['unit']}", referer=url, interval=self.interval)
                info = parse_unit_page(h)
                if info:
                    out.append({**r, "bb": info.get("bb"), "rb": info.get("rb")})
            n = self.db.upsert_units(out, "minrepo")
        else:
            out = []
            for r in rows:
                bb, rb = estimate_bb_rb(r["diff"], r["games"], r["total"], coef)
                if bb is not None:
                    out.append({**r, "bb": bb, "rb": rb})
            n = self.db.upsert_units(out, "minrepo_est", est=True)
        self.db.log_fetch(date, "minrepo", "ok" if out else "empty", n)
        return n

    def fill_recent(self, today: Date, after: str | None, max_days: int = 5) -> int:
        """アナスロの最新日より新しい日がみんレポにあれば正確に取得。"""
        if not after:
            return 0
        links = self.refresh_index(today)
        done = self.db.fetched_dates("minrepo")
        todo = [(d, u) for d, u in sorted(links, reverse=True) if (after is None or d > after) and d not in done]
        n = 0
        for d, u in todo[:max_days]:
            try:
                n += self.fetch_day(d, u, precise=True)
            except RateLimited:
                log.warning("みんレポ: アクセス制限。今回は中断します")
                break
            except Exception as e:
                log.warning("みんレポ: %s の取得失敗: %s", d, e)
        return n

    def backfill_history(self, today: Date, before: str, since: str, coef, deadline: float | None = None,
                         max_days: int = 2000) -> int:
        """アナスロより古い日を推定値で取得。"""
        links = self.refresh_index(today, all_pages=True)
        done = self.db.fetched_dates("minrepo")
        todo = [(d, u) for d, u in sorted(links, reverse=True) if since <= d < before and d not in done]
        log.info("みんレポ過去分: 対象 %d日", len(todo))
        n = 0
        for d, u in todo[:max_days]:
            if deadline and time.time() > deadline:
                break
            try:
                n += self.fetch_day(d, u, precise=False, coef=coef)
                log.info("みんレポ過去分: %s 取得", d)
            except RateLimited:
                log.warning("みんレポ: アクセス制限。今回は中断します")
                break
            except Exception as e:
                log.warning("みんレポ: %s の取得失敗: %s", d, e)
        return n
