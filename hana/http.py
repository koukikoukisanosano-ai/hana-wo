"""データサイトへのアクセス。

相手サイトに負担をかけないよう、1ページごとに十分な間隔を空けます。
アナスロは短時間に続けて開くと 403 を返すため、403 が出たら長めに休んで再開し、
それでもダメならその回は打ち切ります（次回の自動実行で続きから取得）。
"""
from __future__ import annotations

import logging
import random
import time

from urllib.parse import unquote

import requests

log = logging.getLogger("hana.http")

DEFAULT_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/129.0 Safari/537.36"
)


class RateLimited(Exception):
    """サイト側から一時的に拒否された（403/429）。"""


class Fetcher:
    def __init__(self, interval_sec: float = 60, jitter_sec: float = 20, user_agent: str = DEFAULT_UA,
                 timeout: float = 60, engine: str = "requests"):
        self.interval = float(interval_sec)
        self.jitter = float(jitter_sec)
        self.timeout = timeout
        self.engine = engine
        self._last = 0.0
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": user_agent,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "ja,en-US;q=0.8,en;q=0.6",
        })
        self._pw = None  # playwright (任意)
        self.user_agent = user_agent
        # engine: requests / playwright / auto（まず requests、拒否されたら playwright に切り替え）
        self.mode = "playwright" if engine == "playwright" else "requests"

    def _wait(self, interval: float | None = None) -> None:
        gap = (self.interval if interval is None else interval) + random.uniform(0, self.jitter)
        remain = self._last + gap - time.time()
        if remain > 0:
            time.sleep(remain)

    def get(self, url: str, referer: str | None = None, interval: float | None = None) -> str:
        self._wait(interval)
        try:
            if self.mode == "playwright":
                return self._get_playwright(url, referer)
            headers = {"Referer": referer} if referer else {}
            r = self.session.get(url, headers=headers, timeout=self.timeout)
        finally:
            self._last = time.time()
        challenged = r.headers.get("cf-mitigated") == "challenge" or (
            r.status_code in (403, 503) and "challenge-platform" in r.text[:20000])
        if r.status_code in (403, 429) or challenged:
            if self.engine == "auto" and self.mode == "requests" and playwright_available():
                log.info("通常の方法では表示を拒否されたため、ブラウザ方式（Chromium）に切り替えます")
                self.mode = "playwright"
                time.sleep(10)
                try:
                    return self._get_playwright(url, referer)
                finally:
                    self._last = time.time()
            raise RateLimited(f"{r.status_code} {url}")
        r.raise_for_status()
        r.encoding = r.apparent_encoding if not r.encoding or r.encoding.lower() == "iso-8859-1" else r.encoding
        return r.text

    # ---- Playwright（requestsで弾かれる環境向けの予備手段） ----
    def _get_playwright(self, url: str, referer: str | None) -> str:
        if self._pw is None:
            from playwright.sync_api import sync_playwright  # 任意依存
            self._pw_ctx = sync_playwright().start()
            browser = self._pw_ctx.chromium.launch(headless=True)
            ctx = browser.new_context(user_agent=self.user_agent, locale="ja-JP")
            self._pw = ctx.new_page()
        page = self._pw
        if referer:
            if unquote(page.url) != unquote(referer):
                page.goto(referer, wait_until="domcontentloaded", timeout=self.timeout * 1000)
            res = page.evaluate(
                "async (u) => { const r = await fetch(u); return [r.status, await r.text()]; }", url)
            status, text = res
        else:
            resp = page.goto(url, wait_until="domcontentloaded", timeout=self.timeout * 1000)
            status, text = (resp.status if resp else 0), page.content()
        if status in (403, 429):
            raise RateLimited(f"{status} {url}")
        if status >= 400:
            raise requests.HTTPError(f"{status} {url}")
        return text

    def close(self) -> None:  # noqa: D401
        if self._pw is not None:
            try:
                self._pw_ctx.stop()
            except Exception:
                pass
            self._pw = None


def playwright_available() -> bool:
    try:
        import playwright.sync_api  # noqa: F401
        return True
    except Exception:
        return False
