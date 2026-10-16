"""アナスロ（ana-slo.com）からの取得。

一覧ページ: 日付ごとの総差枚・平均差枚・平均G（約4年分）と、詳細ページへのリンク（約13か月分）。
詳細ページ: 全台の 機種名/台番号/G数/差枚/BB/RB（#all_data_table）。
詳細ページは一覧ページから開いたときだけ表示される（Referer が必要）ので、一覧→詳細の順で開く。
"""
from __future__ import annotations

import logging
import re
import time
from urllib.parse import unquote

from bs4 import BeautifulSoup

from ..http import Fetcher, RateLimited

log = logging.getLogger("hana.anaslo")
SOURCE = "anaslo"

_DATE_IN_URL = re.compile(r"/(\d{4}-\d{2}-\d{2})-[^/]*-data/?$")
_DATE_TEXT = re.compile(r"(\d{4})/(\d{1,2})/(\d{1,2})")


def to_int(s) -> int | None:
    if s is None:
        return None
    s = str(s).strip().replace(",", "").replace("+", "").replace("枚", "").replace("G", "")
    s = s.replace("−", "-").replace("–", "").replace("―", "").replace("ー", "")
    if s in ("", "-"):
        return None
    try:
        return int(float(s))
    except ValueError:
        return None


def parse_list(html: str) -> tuple[list[dict], list[tuple[str, str]]]:
    """一覧ページ → (日別サマリー, [(date, 詳細URL)])"""
    soup = BeautifulSoup(html, "lxml")
    days: list[dict] = []
    links: dict[str, str] = {}
    for row in soup.select("div.date-table div.table-row"):
        cells = row.select("div.table-data-cell")
        if len(cells) < 4:
            continue
        m = _DATE_TEXT.search(cells[0].get_text())
        if not m:
            continue
        date = f"{int(m.group(1)):04d}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"
        days.append({
            "date": date,
            "total_diff": to_int(cells[1].get_text()),
            "avg_diff": to_int(cells[2].get_text()),
            "avg_games": to_int(cells[3].get_text()),
        })
        a = cells[0].find("a", href=True)
        if a:
            links[date] = a["href"]
    # 念のため: 表の形が変わってもリンクだけは拾う
    if not links:
        for a in soup.find_all("a", href=True):
            m = _DATE_IN_URL.search(unquote(a["href"]))
            if m:
                links.setdefault(m.group(1), a["href"])
    return days, sorted(links.items())


def parse_detail(html: str, date: str) -> list[dict]:
    """詳細ページ → 全台の行"""
    soup = BeautifulSoup(html, "lxml")
    table = soup.find("table", id="all_data_table")
    if table is None:
        return []
    header = [th.get_text(strip=True) for th in table.find("tr").find_all(["th", "td"])]
    col = {name: i for i, name in enumerate(header)}

    def idx(*names):
        for n in names:
            if n in col:
                return col[n]
        return None

    i_model, i_unit, i_g = idx("機種名", "機種"), idx("台番号", "台番"), idx("G数", "総回転", "回転数")
    i_diff, i_bb, i_rb = idx("差枚", "差枚数"), idx("BB", "BIG"), idx("RB", "REG")
    rows = []
    for tr in table.find_all("tr")[1:]:
        tds = [td.get_text(strip=True) for td in tr.find_all("td")]
        if len(tds) < len(header) - 1 or i_unit is None:
            continue
        unit = to_int(tds[i_unit])
        if unit is None:
            continue
        rows.append({
            "date": date,
            "unit": unit,
            "model": tds[i_model] if i_model is not None else "",
            "games": to_int(tds[i_g]) if i_g is not None else None,
            "diff": to_int(tds[i_diff]) if i_diff is not None else None,
            "bb": to_int(tds[i_bb]) if i_bb is not None else None,
            "rb": to_int(tds[i_rb]) if i_rb is not None else None,
        })
    return rows


def update(db, fetcher: Fetcher, list_url: str, max_pages: int = 10, backoff_min: float = 20,
           deadline: float | None = None) -> dict:
    """一覧を取得し、未取得の日の詳細を新しい順に取りに行く。"""
    stats = {"list_days": 0, "new_days": 0, "rows": 0, "rate_limited": False, "remaining": 0}
    html = fetcher.get(list_url, interval=5)
    days, links = parse_list(html)
    db.upsert_hall_days(days, SOURCE)
    db.upsert_index(links, SOURCE)
    stats["list_days"] = len(days)

    done = db.fetched_dates(SOURCE)
    todo = [(d, u) for d, u in sorted(links, reverse=True) if d not in done]
    stats["remaining"] = len(todo)
    log.info("アナスロ: 一覧 %d日 / 詳細リンク %d日 / 未取得 %d日", len(days), len(links), len(todo))

    backed_off = False
    for date, url in todo[:max_pages]:
        if deadline and time.time() > deadline:
            break
        try:
            page = fetcher.get(url, referer=list_url)
        except RateLimited:
            if backed_off:
                log.warning("アナスロ: アクセス制限が続くため今回はここまで（次回続きから）")
                stats["rate_limited"] = True
                break
            log.warning("アナスロ: アクセス制限(403)。%d分休んでから再開します", backoff_min)
            time.sleep(backoff_min * 60)
            backed_off = True
            try:
                fetcher.get(list_url, interval=5)
                page = fetcher.get(url, referer=list_url)
            except RateLimited:
                stats["rate_limited"] = True
                break
        except Exception as e:  # 通信エラー等
            log.warning("アナスロ: %s の取得に失敗: %s", date, e)
            db.log_fetch(date, SOURCE, "error", 0)
            continue
        rows = parse_detail(page, date)
        n = db.upsert_units(rows, SOURCE)
        db.log_fetch(date, SOURCE, "ok" if rows else "empty", n)
        stats["new_days"] += 1
        stats["rows"] += n
        stats["remaining"] -= 1
        log.info("アナスロ: %s 取得 (%d台)", date, len(rows))
    return stats
