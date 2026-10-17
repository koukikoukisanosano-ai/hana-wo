"""GitHub の「iPhone受信箱」Issue に届いたコメントを取り込む。

iPhone のショートカットは、1ページごとに次の内容をコメントとして投稿する:
  1行目: アナスロのページURL
  2行目以降: ページ内で正規表現に一致した部分（タイトル・集計表・ハナハナの行）
取り込んだコメントは削除する（通知もIssueも増えない）。持ち主以外のコメントは無視して削除。
"""
from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

from . import phone

log = logging.getLogger("hana.gh_inbox")
API = "https://api.github.com"
TITLE = "iPhone受信箱"


def _session(token: str) -> requests.Session:
    s = requests.Session()
    s.headers.update({"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json",
                      "X-GitHub-Api-Version": "2022-11-28"})
    return s


def ensure_issue(s: requests.Session, repo: str) -> int:
    r = s.get(f"{API}/repos/{repo}/issues", params={"state": "all", "per_page": 100}, timeout=30)
    r.raise_for_status()
    for it in r.json():
        if it.get("title") == TITLE and "pull_request" not in it:
            return it["number"]
    r = s.post(f"{API}/repos/{repo}/issues", timeout=30, json={
        "title": TITLE,
        "body": "iPhoneのショートカットが、ここにアナスロのデータをコメントで送ります。\n"
                "届いたコメントは自動で取り込まれ、すぐに削除されます。このIssueは閉じないでください。"})
    r.raise_for_status()
    n = r.json()["number"]
    s.put(f"{API}/repos/{repo}/issues/{n}/lock", json={"lock_reason": "resolved"}, timeout=30)
    log.info("受信箱のIssue #%d を作成しました", n)
    return n


def _debug_record(root: Path, c: dict, body: str) -> None:
    """届いた中身の概要を data/debug/inbox.json に残す（原因調査用・最新20件）。"""
    p = root / "data" / "debug" / "inbox.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    try:
        items = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        items = []
    first, _, rest = body.partition("\n")
    items.append({
        "received": datetime.now(timezone(timedelta(hours=9))).strftime("%Y-%m-%d %H:%M:%S"),
        "url": first.strip()[:200], "total_len": len(body), "rest_len": len(rest),
        "lines": body.count("\n") + 1,
        "has_title": bool(re.search(r"<title>", rest)),
        "has_total": "total_get_medals_table" in rest,
        "n_rows": len(re.findall(r'class="fixed01"', rest)),
        "head": rest[:600],
    })
    p.write_text(json.dumps(items[-20:], ensure_ascii=False, indent=1), encoding="utf-8")


def process(db, stats: dict, root: Path | None = None) -> int | None:
    token = os.environ.get("GITHUB_TOKEN")
    repo = os.environ.get("GITHUB_REPOSITORY")
    owner = (os.environ.get("GITHUB_REPOSITORY_OWNER") or "").lower()
    if not token or not repo:
        return None
    s = _session(token)
    number = ensure_issue(s, repo)
    url = f"{API}/repos/{repo}/issues/{number}/comments"
    comments = []
    page = 1
    while True:
        r = s.get(url, params={"per_page": 100, "page": page}, timeout=30)
        r.raise_for_status()
        batch = r.json()
        comments += batch
        if len(batch) < 100:
            break
        page += 1
    for c in sorted(comments, key=lambda c: c["created_at"]):
        who = (c.get("user") or {}).get("login", "").lower()
        if who == owner:
            if root is not None:
                try:
                    _debug_record(root, c, c.get("body") or "")
                except Exception as e:
                    log.warning("記録に失敗: %s", e)
            try:
                phone.handle_blob(db, c.get("body") or "", stats, f"comment {c['id']}")
            except Exception as e:
                log.warning("コメント %s を取り込めませんでした: %s", c["id"], e)
        else:
            log.info("持ち主以外のコメントを削除: %s", who)
        s.delete(f"{API}/repos/{repo}/issues/comments/{c['id']}", timeout=30)
    return number
