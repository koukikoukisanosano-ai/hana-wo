"""公式データサイト（テラスモバイル2）の構造調査。結果を data/diag_tera/ に保存。"""
from __future__ import annotations

import json
from pathlib import Path

from .http import DEFAULT_UA

BASE = "https://ogiya.pt.teramoba2.com/wo/"


def run(root: Path) -> None:
    out_dir = root / "data" / "diag_tera"
    out_dir.mkdir(parents=True, exist_ok=True)
    from playwright.sync_api import sync_playwright
    report = {"steps": []}
    with sync_playwright() as p:
        b = p.chromium.launch(headless=True)
        ctx = b.new_context(user_agent=DEFAULT_UA, locale="ja-JP", viewport={"width": 390, "height": 844})
        pg = ctx.new_page()
        net = []
        pg.on("response", lambda r: net.append({"url": r.url, "status": r.status,
                                                 "type": r.headers.get("content-type", "")[:40],
                                                 "method": r.request.method}))

        def snap(name):
            html = pg.content()
            (out_dir / f"{name}.html").write_text(html, encoding="utf-8")
            text = pg.inner_text("body")[:4000] if pg.query_selector("body") else ""
            links = pg.eval_on_selector_all("a", "els => els.map(e => [e.innerText.trim().slice(0,40), e.href]).slice(0,120)")
            report["steps"].append({"name": name, "url": pg.url, "title": pg.title(), "text": text, "links": links})

        pg.goto(BASE + "rack_info/", wait_until="networkidle", timeout=60000)
        snap("01_entry")
        btn = pg.get_by_text("サイトを閲覧する")
        if btn.count():
            btn.first.click()
            pg.wait_for_load_state("networkidle", timeout=60000)
            pg.wait_for_timeout(3000)
            snap("02_after_enter")
        # 機種一覧・スロットっぽいリンクをたどる
        for kw in ("スロット", "パチスロ", "20円", "ハナハナ"):
            loc = pg.get_by_text(kw, exact=False)
            if loc.count():
                try:
                    loc.first.click()
                    pg.wait_for_load_state("networkidle", timeout=60000)
                    pg.wait_for_timeout(2000)
                    snap(f"03_{kw}")
                except Exception as e:
                    report["steps"].append({"name": f"click_{kw}", "error": str(e)[:200]})
        report["network"] = [n for n in net if not any(x in n["url"] for x in (".png", ".jpg", ".gif", ".svg", ".woff", ".css", "fonts."))][:200]
        b.close()
    (out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
