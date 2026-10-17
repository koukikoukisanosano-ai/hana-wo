"""使い方（プロジェクトのフォルダで）:

  python -m hana daily              毎朝用: 新しいデータを取得して今日の予想ページを作る
  python -m hana predict [--date 2026-10-11] [--open]
  python -m hana update [--max-pages 8]
  python -m hana backfill [--hours 8]   過去データを時間をかけて取得（アクセス制限に配慮してゆっくり）
  python -m hana backtest [--days 90] [--lag 1]
  python -m hana minrepo-history [--since 2024-01-01] [--hours 3]   みんレポの古いデータ（任意）
  python -m hana status
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import webbrowser
from datetime import date as Date, datetime, timedelta, timezone
from pathlib import Path

from . import backtest as BT
from . import config as C
from . import model as M
from . import report as R
from .db import DB
from .http import Fetcher, RateLimited
from .sources import anaslo
from .sources import minrepo as MR

JST = timezone(timedelta(hours=9))
log = logging.getLogger("hana")


def setup_logging(root: Path) -> None:
    (root / "logs").mkdir(exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(root / "logs" / "hana.log", encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    lg = logging.getLogger("hana")
    lg.setLevel(logging.INFO)
    lg.handlers[:] = [fh, sh]
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


class FileLock:
    """取得処理が同時に2つ動かないようにするロック（PC再起動やプロセス終了で自動解除）。"""
    def __init__(self, path: Path):
        self.path = path
        self.f = None

    def acquire(self) -> bool:
        import os
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.f = open(self.path, "a+")
        try:
            if os.name == "nt":
                import msvcrt
                self.f.seek(0)
                msvcrt.locking(self.f.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.f, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return True
        except OSError:
            return False


def today_jst() -> Date:
    return datetime.now(JST).date()


def make_fetcher(cfg: dict) -> Fetcher:
    f = cfg["fetch"]
    return Fetcher(interval_sec=f.get("interval_sec", 75), jitter_sec=f.get("jitter_sec", 30),
                   engine=f.get("engine", "requests"))


def write_status(root: Path, **kw) -> None:
    p = root / "data" / "status.json"
    try:
        cur = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    except Exception:
        cur = {}
    cur.update(kw)
    p.write_text(json.dumps(cur, ensure_ascii=False, indent=1), encoding="utf-8")


def cmd_update(cfg, db, max_pages=None, deadline=None) -> dict:
    stats = {}
    if not cfg["fetch"].get("anaslo_direct", True):
        return cmd_minrepo_recent(cfg, db, stats)
    f = make_fetcher(cfg)
    now = datetime.now(JST).strftime("%Y-%m-%d %H:%M")
    try:
        stats = anaslo.update(db, f, cfg["hall"]["anaslo_list_url"],
                              max_pages=max_pages or cfg["fetch"].get("daily_max_pages", 8),
                              backoff_min=cfg["fetch"].get("backoff_min", 20), deadline=deadline)
        ok = stats.get("new_days", 0) > 0 or not stats.get("rate_limited")
        write_status(C.ROOT, anaslo_checked=now, anaslo_result="ok" if ok else "limited",
                     anaslo_message="" if ok else "アクセス制限で一部取得できず（次回続きから）",
                     **({"anaslo_ok": now} if ok else {}))
    except RateLimited as e:
        log.warning("アナスロ: 一覧ページもアクセス制限中。次回の実行で取得します")
        stats["rate_limited"] = True
        write_status(C.ROOT, anaslo_checked=now, anaslo_result="blocked", anaslo_message=f"アクセスを拒否されました（{e}）")
    except Exception as e:
        log.error("アナスロの取得でエラー: %s", e)
        msg = str(e)
        if any(k in msg for k in ("ConnectionError", "ProxyError", "Max retries", "timed out", "NameResolution")):
            msg = "サイトに接続できませんでした"
        write_status(C.ROOT, anaslo_checked=now, anaslo_result="error", anaslo_message=msg[:120])
    f.close()
    return cmd_minrepo_recent(cfg, db, stats)


def cmd_minrepo_recent(cfg, db, stats: dict) -> dict:
    # みんレポ: アナスロの最新日より新しい日があれば補完
    try:
        cov = db.coverage()
        mf = Fetcher(interval_sec=cfg["fetch"].get("minrepo_interval_sec", 8), jitter_sec=3)
        mr = MR.MinRepo(db, mf, cfg["hall"]["minrepo_tag_url"], cfg["target"].get("keyword", "ハナハナ"),
                        interval=cfg["fetch"].get("minrepo_interval_sec", 8))
        n = mr.fill_recent(today_jst(), after=cov.get("unit_last"))
        if n:
            log.info("みんレポで %d 台日を補完", n)
    except Exception as e:
        log.warning("みんレポの確認でエラー（無視して続行）: %s", e)
    return stats


def cmd_inbox(cfg, db, root: Path) -> dict:
    """iPhone から届いたデータを取り込み、次に取りに行くページの一覧を更新する。"""
    from .sources import gh_inbox, phone
    st = phone.process(db, root)
    try:
        num = gh_inbox.process(db, st, root)
        if num:
            write_status(root, inbox_issue=num)
    except Exception as e:
        log.warning("受信箱(Issue)の確認でエラー: %s", e)
    now = datetime.now(JST).strftime("%Y-%m-%d %H:%M")
    if st["files"]:
        result = "ok" if st["days"] else ("blocked" if st["blocked"] else "nodata")
        kw = {"phone_last": now, "phone_result": result,
              "phone_message": f"{st['days']}日分を受信" + (f"・空ページ{st['blocked']}件" if st["blocked"] else "")}
        if st["days"]:
            kw["phone_ok"] = now
        write_status(root, **kw)
        log.info("iPhoneからの受信: %s", st)
    write_next(cfg, db, root)
    return st


def write_next(cfg, db, root: Path) -> list[str]:
    from .sources import phone
    tmpl = cfg["hall"].get("anaslo_day_url",
                           "https://ana-slo.com/{date}-%e3%82%aa%e3%83%bc%e3%82%ae%e3%83%a4wo-data/")
    n = int(cfg.get("phone", {}).get("pages_per_run", 4))
    hp = root / "data" / "handout.json"
    try:
        handout = json.loads(hp.read_text(encoding="utf-8")) if hp.exists() else {}
    except Exception:
        handout = {}
    urls = phone.next_urls(db, today_jst(), tmpl, n=n, handout=handout)
    hp.write_text(json.dumps(dict(sorted(handout.items())), ensure_ascii=False, indent=1), encoding="utf-8")
    txt = "\n".join(urls)          # 末尾に改行を付けない（iPhone側で空の行がエラーになるため）
    (root / "data" / "next.txt").write_text(txt, encoding="utf-8")
    (root / "output").mkdir(exist_ok=True)
    (root / "output" / "next.txt").write_text(txt, encoding="utf-8")
    log.info("iPhoneが次に取りに行くページ: %d件", len(urls))
    return urls


def cmd_backfill(cfg, db, hours: float, on_progress=None) -> None:
    deadline = time.time() + hours * 3600
    while time.time() < deadline:
        st = cmd_update(cfg, db, max_pages=60, deadline=deadline)
        log.info("取得状況: %s", st)
        if on_progress:
            on_progress()
        if st.get("remaining", 0) <= 0:
            log.info("アナスロの取得可能な過去データはすべて取得しました")
            break
        if st.get("rate_limited"):
            wait = cfg["fetch"].get("backoff_min", 20) * 3
            log.info("%d分休んでから続けます", wait)
            time.sleep(min(wait * 60, max(0, deadline - time.time())))


def load_backtest(root: Path) -> dict | None:
    p = root / "data" / "backtest.json"
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None


def cmd_backtest(cfg, db, root: Path, days=90, lag=None) -> dict:
    units, hall = db.units(cfg["target"].get("keyword", "ハナハナ")), db.hall_days()
    if lag is None:
        lag = estimate_lag(db)
    res = BT.run(units, hall, C.model_settings(cfg), days=days, lag=lag)
    import numpy as np

    def plain(v):
        if isinstance(v, (bool, np.bool_)):
            return bool(v)
        if isinstance(v, (int, np.integer)):
            return int(v)
        if isinstance(v, (float, np.floating)):
            return float(v)
        return v
    out = {k: plain(v) for k, v in res.items() if k != "table"}
    out["when"] = f"{datetime.now():%Y-%m-%d}"
    (root / "data").mkdir(exist_ok=True)
    (root / "data" / "backtest.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    if "table" in res:
        res["table"].to_csv(root / "data" / "backtest_days.csv", index=False, encoding="utf-8-sig")
    return out


def estimate_lag(db) -> int:
    """データ公開の遅れ（日）。最近の取得記録から推定。"""
    cov = db.coverage()
    if not cov.get("unit_last"):
        return 1
    last = Date.fromisoformat(cov["unit_last"])
    return max(1, min(7, (today_jst() - last).days))


def cmd_predict(cfg, db, root: Path, target: Date, open_browser=False) -> Path | None:
    units, hall = db.units(cfg["target"].get("keyword", "ハナハナ")), db.hall_days()
    if units.empty:
        log.error("データがありません。先に update / backfill を実行してください")
        return None
    s = C.model_settings(cfg)
    m = M.train(units, hall, s)
    pred = M.predict(m, target)
    pred_dir = root / "data" / "predictions"
    path = R.build(m, pred, target, cfg["hall"].get("name", ""), cfg["predict"].get("top_n", 10),
                   load_backtest(root), db.coverage(), root / "output", extra_warns=status_warnings(root),
                   pred_dir=pred_dir, info_line=status_line(root))
    if cfg.get("storage") == "csv":   # 答え合わせ用に予想を保存
        pred_dir.mkdir(parents=True, exist_ok=True)
        pred[["rank", "unit", "score"]].round(4).to_csv(pred_dir / f"{target:%Y-%m-%d}.csv", index=False)
    log.info("予想ページを作成: %s", path)
    top = pred.head(5)
    log.info("上位: %s", "  ".join(f"{int(r.unit)}({r.score*100:.1f}%)" for r in top.itertuples()))
    if open_browser:
        webbrowser.open((root / "output" / "latest.html").resolve().as_uri())
    return path


def status_line(root: Path) -> str:
    p = root / "data" / "status.json"
    try:
        st = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return ""
    if st.get("mode") == "phone":
        return f"iPhoneからの最終受信: {st.get('phone_last', 'まだありません')}" + (f"（{st['phone_message']}）" if st.get('phone_message') else "")
    return ""


def status_warnings(root: Path) -> list[str]:
    p = root / "data" / "status.json"
    if not p.exists():
        return []
    try:
        st = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return []
    out = []
    if st.get("mode") == "phone":
        last = st.get("phone_last")
        if not last:
            out.append("iPhoneからのデータ送信がまだありません。設定ガイドの手順でショートカットを設定してください。")
        else:
            try:
                age = datetime.now(JST).replace(tzinfo=None) - datetime.strptime(last, "%Y-%m-%d %H:%M")
                if age > timedelta(hours=36):
                    out.append(f"iPhoneからのデータ送信が止まっています（最後の受信 {last}）。ショートカットのオートメーションを確認してください。")
            except ValueError:
                pass
            if st.get("phone_result") == "blocked":
                out.append(f"最後の受信（{last}）ではアナスロのページが空でした。アクセス制限の可能性があります（自動で再試行します）。")
        return out
    if st.get("anaslo_result") in ("blocked", "error"):
        last_ok = st.get("anaslo_ok", "なし")
        out.append(f"アナスロからデータを取得できていません（最終確認 {st.get('anaslo_checked')}：{st.get('anaslo_message', '')}）。"
                   f"最後に取得できたのは {last_ok} です。取得できるまでは手元のデータで予想しています。")
    return out


def cmd_import_seed(db, root: Path) -> None:
    import re
    seed = root / "data" / "seed"
    p = seed / "anaslo_hana.tsv"
    if p.exists():
        rows = []
        for line in p.read_text(encoding="utf-8").splitlines():
            d, model, unit, g, diff, bb, rb = line.split("\t")
            rows.append({"date": d, "unit": int(unit), "model": model, "games": anaslo.to_int(g),
                         "diff": anaslo.to_int(diff), "bb": anaslo.to_int(bb), "rb": anaslo.to_int(rb)})
        db.upsert_units(rows, "seed")
        for d in sorted({r["date"] for r in rows}):
            db.log_fetch(d, "anaslo", "ok", sum(1 for r in rows if r["date"] == d))
        log.info("初期データ %d 台日を取り込みました", len(rows))
    p = seed / "anaslo_hall_days.csv"
    if p.exists():
        rx = re.compile(r"^(\d{4}/\d{2}/\d{2}),(–|[+\-][\d,]+?),(–|[+\-][\d,]+?),([\d,]+)$")
        rows = []
        for line in p.read_text(encoding="utf-8").splitlines():
            m = rx.match(line.strip())
            if m:
                rows.append({"date": m.group(1).replace("/", "-"), "total_diff": anaslo.to_int(m.group(2)),
                             "avg_diff": anaslo.to_int(m.group(3)), "avg_games": anaslo.to_int(m.group(4))})
        db.upsert_hall_days(rows, "anaslo")
        log.info("店全体の日別データ %d 日を取り込みました", len(rows))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="hana", description="オーギヤWO ハナハナ予想")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("daily")
    p = sub.add_parser("predict")
    p.add_argument("--date")
    p.add_argument("--open", action="store_true")
    p = sub.add_parser("update")
    p.add_argument("--max-pages", type=int)
    p = sub.add_parser("backfill")
    p.add_argument("--hours", type=float, default=8)
    p.add_argument("--max-minutes", type=float)
    p = sub.add_parser("backtest")
    p.add_argument("--days", type=int, default=90)
    p.add_argument("--lag", type=int)
    p = sub.add_parser("minrepo-history")
    p.add_argument("--since", default="2024-01-01")
    p.add_argument("--hours", type=float, default=3)
    sub.add_parser("status")
    sub.add_parser("import-seed")
    sub.add_parser("diag")
    sub.add_parser("inbox")
    a = ap.parse_args(argv)

    root = C.ROOT
    setup_logging(root)
    cfg = C.load()
    csv_mode = cfg.get("storage") == "csv"
    db_path = root / "data" / "hana.db"
    if csv_mode and db_path.exists():
        db_path.unlink()          # CSV が正。毎回CSVから作り直す
    db = DB(db_path)
    if csv_mode:
        n = db.import_csv(root / "data")
        if n == 0:
            cmd_import_seed(db, root)
        log.info("保存データを読み込み: %s", db.coverage())

    lock = FileLock(root / "data" / "fetch.lock")
    fetch_ok = True
    if a.cmd in ("update", "backfill", "daily", "minrepo-history"):
        fetch_ok = lock.acquire()
        if not fetch_ok:
            log.info("別の取得処理が実行中のため、今回はデータ取得をスキップします")

    if a.cmd == "update":
        if fetch_ok:
            log.info("結果: %s", cmd_update(cfg, db, a.max_pages))
    elif a.cmd == "backfill":
        if not fetch_ok:
            return 0
        hours = a.max_minutes / 60 if a.max_minutes else a.hours
        save = (lambda: db.export_csv(root / "data", keyword=cfg["target"].get("keyword", "ハナハナ"))) if csv_mode else None
        cmd_backfill(cfg, db, hours, on_progress=save)
        try:
            cmd_backtest(cfg, db, root)
        except Exception as e:
            log.warning("検証でエラー（予想は続行）: %s", e)
        cmd_predict(cfg, db, root, today_jst())
    elif a.cmd == "predict":
        target = Date.fromisoformat(a.date) if a.date else today_jst()
        cmd_predict(cfg, db, root, target, a.open or cfg["predict"].get("open_browser", False))
    elif a.cmd == "backtest":
        r = cmd_backtest(cfg, db, root, a.days, a.lag)
        log.info("検証結果: %s", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()})
    elif a.cmd == "inbox":
        cmd_inbox(cfg, db, root)
        if today_jst().weekday() == 0 or load_backtest(root) is None:
            try:
                cmd_backtest(cfg, db, root)
            except Exception as e:
                log.warning("検証でエラー（予想は続行）: %s", e)
        cmd_predict(cfg, db, root, today_jst())
        write_next(cfg, db, root)
    elif a.cmd == "daily":
        cmd_inbox(cfg, db, root)
        if fetch_ok:
            cmd_update(cfg, db)
        if today_jst().weekday() == 0 or load_backtest(root) is None:   # 週1回 検証を更新
            try:
                cmd_backtest(cfg, db, root)
            except Exception as e:
                log.warning("検証でエラー（予想は続行）: %s", e)
        cmd_predict(cfg, db, root, today_jst(), cfg["predict"].get("open_browser", False))
        write_next(cfg, db, root)
    elif a.cmd == "minrepo-history":
        if not fetch_ok:
            return 0
        cov = db.coverage()
        units = db.units(cfg["target"].get("keyword", "ハナハナ"))
        coef = MR.fit_payout_coef(units[units["source"].isin(["anaslo", "seed", "minrepo"])])
        log.info("BB/RB推定の係数: %s", coef)
        allu = db.units()
        an = allu[allu["source"] == "anaslo"]["date"]
        first = an.min() if len(an) else (cov.get("unit_first") or "2099-01-01")
        f = Fetcher(interval_sec=cfg["fetch"].get("minrepo_interval_sec", 8), jitter_sec=4)
        mr = MR.MinRepo(db, f, cfg["hall"]["minrepo_tag_url"], cfg["target"].get("keyword", "ハナハナ"),
                        interval=cfg["fetch"].get("minrepo_interval_sec", 8))
        n = mr.backfill_history(today_jst(), before=first, since=a.since, coef=coef,
                                deadline=time.time() + a.hours * 3600)
        log.info("みんレポ過去分 %d 台日を保存", n)
    elif a.cmd == "status":
        print(json.dumps(db.coverage(), ensure_ascii=False, indent=1))
        print("アナスロ未取得:", len(set(db.index("anaslo")) - db.fetched_dates("anaslo")), "日")
    elif a.cmd == "import-seed":
        cmd_import_seed(db, root)
    elif a.cmd == "diag":
        from . import diag, diag_tera
        r = diag.run(root)
        try:
            diag_tera.run(root)
        except Exception as e:
            log.warning("公式サイト調査でエラー: %s", e)
        log.info("診断: %s", json.dumps(r, ensure_ascii=False)[:1500])
    if csv_mode and a.cmd in ("update", "backfill", "daily", "minrepo-history", "import-seed", "inbox"):
        db.export_csv(root / "data", keyword=cfg["target"].get("keyword", "ハナハナ"))
        log.info("データを保存しました")
    return 0


if __name__ == "__main__":
    sys.exit(main())
