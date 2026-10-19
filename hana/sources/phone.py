"""iPhone（ショートカット）から送られてきたアナスロのページ断片を取り込む。

iPhone 側はアナスロの日別ページを開き、次の正規表現に一致した部分だけを送ってくる:
  <title>…</title> | 店全体の集計表(total_get_medals_table) | ハナハナの台の行
送り先: リポジトリの inbox/ フォルダ（1ページ = 1ファイル、1行目がページのURL）
"""
from __future__ import annotations

import logging
import re
from datetime import date as Date, datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import unquote

from .anaslo import parse_detail, to_int

log = logging.getLogger("hana.phone")
SOURCE = "anaslo"
JST = timezone(timedelta(hours=9))

# iPhone のショートカットで使う正規表現（ガイドページにも同じものを載せる）
PHONE_REGEX = (r'<title>[^<]*</title>|'   # 店全体の集計表は重いので外した（届けば使う）
               r'<td class="fixed01"[^>]*>[^<]*ハナハナ[^<]*</td>(?:\s*<td[^>]*>[^<]*</td>){10}')
ROW_RE = re.compile(r'<td class="fixed01"[^>]*>[^<]*ハナハナ[^<]*</td>(?:\s*<td[^>]*>[^<]*</td>){10}')
TOTAL_RE = re.compile(r'<table[^>]*total_get_medals_table[\s\S]*?</table>')
TITLE_RE = re.compile(r'<title>([^<]*)</title>')
DATE_IN_URL = re.compile(r"/(\d{4}-\d{2}-\d{2})-[^/]*-data/?")
HEADER = ("<tr><th>機種名</th><th>台番号</th><th>G数</th><th>差枚</th><th>BB</th><th>RB</th><th>ART</th>"
          "<th>合成確率</th><th>BB確率</th><th>RB確率</th><th>ART確率</th></tr>")


def parse_blob(text: str) -> dict:
    """送られてきたテキスト → {url, date, kind, rows, total}"""
    text = text.replace("\r\n", "\n")
    first, _, rest = text.partition("\n")
    url = first.strip()
    m = DATE_IN_URL.search(unquote(url))
    date = m.group(1) if m else None
    title = TITLE_RE.search(rest)
    title = title.group(1).strip() if title else ""
    rows_html = ROW_RE.findall(rest)
    rows = []
    if date and rows_html:
        table = '<table id="all_data_table">' + HEADER + "".join(f"<tr>{r}</tr>" for r in rows_html) + "</table>"
        seen = set()
        for r in parse_detail(table, date):
            if r["unit"] not in seen:
                seen.add(r["unit"])
                rows.append(r)
    total = None
    tm = TOTAL_RE.search(rest)
    if tm:
        cells = re.findall(r"<td[^>]*>([^<]*)</td>", tm.group(0))
        if len(cells) >= 3:
            total = {"date": date, "total_diff": to_int(cells[0]), "avg_diff": to_int(cells[1]),
                     "avg_games": to_int(cells[2])}
    if rows or total:
        kind = "ok"
    elif "データまとめ" in title:
        kind = "nodata"          # ページはあるがハナハナの行がない
    elif title:
        kind = "notfound"        # 404 などの別ページ
    else:
        kind = "blocked"         # 中身が空 = アクセス拒否の可能性
    return {"url": url, "date": date, "title": title, "kind": kind, "rows": rows, "total": total}


def new_stats() -> dict:
    return {"files": 0, "days": 0, "rows": 0, "blocked": 0, "notfound": 0, "dates": []}


def handle_blob(db, text: str, stats: dict, label: str = "") -> None:
    """iPhoneから届いた1ページ分（1行目URL＋照合結果）を取り込む。"""
    stats["files"] += 1
    info = parse_blob(text)
    d = info["date"]
    if not d:
        log.warning("受信 %s: URLから日付が分かりません（%s）", label, info["url"][:80])
        return
    if info["kind"] == "ok":
        n = db.upsert_units(info["rows"], SOURCE)
        if info["total"]:
            db.upsert_hall_days([info["total"]], SOURCE)
        db.log_fetch(d, SOURCE, "ok" if n else "empty", n)
        stats["days"] += 1
        stats["rows"] += n
        stats["dates"].append(d)
        log.info("iPhoneから受信: %s ハナハナ %d台", d, n)
    elif info["kind"] == "blocked":
        stats["blocked"] += 1
        db.log_fetch(d, SOURCE, "blocked", 0)
        log.warning("iPhoneから受信: %s は中身が空でした（アクセス拒否の可能性）", d)
    else:
        stats["notfound"] += 1
        db.log_fetch(d, SOURCE, "empty", 0)
        log.info("iPhoneから受信: %s はデータなし（%s）", d, info["title"][:40])


def process(db, root: Path, stats: dict | None = None) -> dict:
    """inbox/ フォルダに置かれたファイルを取り込む（予備の受け取り口）。"""
    stats = stats or new_stats()
    inbox = root / "inbox"
    if not inbox.exists():
        return stats
    for p in sorted(x for x in inbox.iterdir() if x.is_file() and not x.name.startswith(".")):
        try:
            handle_blob(db, p.read_text(encoding="utf-8", errors="replace"), stats, p.name)
        except Exception as e:
            log.warning("受信ファイル %s を読めませんでした: %s", p.name, e)
        p.unlink()
    return stats


def _norm_handout(handout: dict, now: datetime) -> None:
    """古い形式（{日付: 最初に載せた日時}）を {日付: {"fails": 回数}} に直す。"""
    for k, v in list(handout.items()):
        if k.startswith("_"):
            continue
        if isinstance(v, str):
            try:
                age = now - datetime.fromisoformat(v)
            except Exception:
                age = timedelta(0)
            handout[k] = {"fails": 1 if age > timedelta(hours=3) else 0}
        elif not isinstance(v, dict):
            handout[k] = {"fails": 0}


def next_urls(db, today: Date, url_tmpl: str, n: int = 1, window_days: int = 390,
              handout: dict | None = None, rotate_hours: float = 3.0) -> list[str]:
    """iPhone が次に取りに行くページ。まだ取れていない日を、新しい日付から順に。

    1つのページで毎回ショートカットが止まると、その日付が先頭にある間は何も届かなくなる。
    そこで、先頭に rotate_hours 時間置いても届かなかった日付は「失敗1回」と数えて後ろに回す
    （この関数は1時間ごとに呼ばれる）。並び順は 失敗回数が少ない順 → 新しい日付順。
    直近2日分は予想に一番効くので、失敗2回までは先頭に置く。
    handout: {"_front": {"date", "since"}, 日付: {"fails", "last_fail"}}（data/handout.json）
    """
    with db.conn() as c:
        rows = c.execute("SELECT date, status, fetched_at FROM fetch_log WHERE source=?", (SOURCE,)).fetchall()
    log_by = {d: (s, t) for d, s, t in rows}
    now = datetime.now(JST).replace(tzinfo=None)
    handout = handout if handout is not None else {}
    _norm_handout(handout, now)

    # 先頭に置いていた日付が、時間がたっても届かなかった → 失敗として数える
    front = handout.get("_front") or {}
    fd = front.get("date")
    if fd and log_by.get(fd, ("", ""))[0] != "ok":
        try:
            since = datetime.fromisoformat(front.get("since", ""))
        except Exception:
            since = now
        if now - since >= timedelta(hours=rotate_hours):
            e = handout.setdefault(fd, {"fails": 0})
            e["fails"] = int(e.get("fails", 0)) + 1
            e["last_fail"] = now.isoformat(timespec="seconds")
            handout.pop("_front", None)
            fd = None

    cands = []
    # 当日分はまだ公開されないので前日から。前日分も朝までは公開されていないことがあるので7時から
    first_k = 1 if now.hour >= 7 else 2
    for k in range(first_k, window_days):
        d = today - timedelta(days=k)
        key = d.isoformat()
        st = log_by.get(key)
        if st:
            status, t = st
            if status == "ok":
                handout.pop(key, None)
                continue
            try:
                last = datetime.fromisoformat(t)
            except Exception:
                last = now - timedelta(days=30)
            recent = k <= 10
            # 「空ページ」「データなし」で届いた日は、最近なら数時間おき、古ければ1日おきに取り直す
            wait = timedelta(hours=3) if recent else timedelta(days=1)
            if now - last < wait:
                continue
            if not recent and status == "empty":
                continue        # 古い日でページ自体がない → もう取りに行かない
        fails = int(handout.get(key, {}).get("fails", 0))
        eff = max(0, fails - 2) if k <= 2 else fails
        cands.append((eff, k, key))
    cands.sort()
    chosen = [key for _, _, key in cands[:n]]
    if chosen and chosen[0] != fd:
        handout["_front"] = {"date": chosen[0], "since": now.isoformat(timespec="seconds")}
    out = [url_tmpl.format(date=key) for key in chosen]
    if not out:   # 空だとiPhoneのショートカットがエラーになるので、前日分を1件だけ入れておく
        out.append(url_tmpl.format(date=(today - timedelta(days=1)).isoformat()))
    return out
