"""SQLite保存（data/hana.db）。"""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import pandas as pd

SCHEMA = """
CREATE TABLE IF NOT EXISTS unit_day (
    date    TEXT NOT NULL,          -- YYYY-MM-DD
    unit    INTEGER NOT NULL,       -- 台番
    model   TEXT NOT NULL,          -- 機種名
    games   INTEGER,                -- G数
    diff    INTEGER,                -- 差枚
    bb      INTEGER,
    rb      INTEGER,
    source  TEXT NOT NULL,          -- anaslo / minrepo / minrepo_est / seed
    est     INTEGER NOT NULL DEFAULT 0,  -- 1 = BB/RBが推定値
    PRIMARY KEY (date, unit)
);
CREATE INDEX IF NOT EXISTS ix_unit_day_model ON unit_day(model);
CREATE TABLE IF NOT EXISTS hall_day (
    date        TEXT PRIMARY KEY,
    total_diff  INTEGER,
    avg_diff    INTEGER,
    avg_games   INTEGER,
    source      TEXT
);
CREATE TABLE IF NOT EXISTS day_index (
    date    TEXT NOT NULL,
    source  TEXT NOT NULL,
    url     TEXT NOT NULL,
    PRIMARY KEY (date, source)
);
CREATE TABLE IF NOT EXISTS fetch_log (
    date        TEXT NOT NULL,
    source      TEXT NOT NULL,
    status      TEXT NOT NULL,      -- ok / empty / error
    n_rows      INTEGER,
    fetched_at  TEXT NOT NULL,
    PRIMARY KEY (date, source)
);
"""

# 優先度: 同じ日・台に複数ソースがあれば精度の高い方を残す
SOURCE_RANK = {"anaslo": 3, "minrepo": 2, "seed": 2, "minrepo_est": 1}


class DB:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.conn() as c:
            c.executescript(SCHEMA)

    @contextmanager
    def conn(self):
        c = sqlite3.connect(self.path)
        try:
            yield c
            c.commit()
        finally:
            c.close()

    # ---- 書き込み ----
    def upsert_units(self, rows: list[dict], source: str, est: bool = False) -> int:
        """rows: date, unit, model, games, diff, bb, rb。優先度の低いソースでは上書きしない。"""
        if not rows:
            return 0
        rank = SOURCE_RANK.get(source, 0)
        n = 0
        with self.conn() as c:
            for r in rows:
                cur = c.execute("SELECT source FROM unit_day WHERE date=? AND unit=?", (r["date"], r["unit"])).fetchone()
                if cur and SOURCE_RANK.get(cur[0], 0) > rank:
                    continue
                c.execute(
                    "INSERT OR REPLACE INTO unit_day(date,unit,model,games,diff,bb,rb,source,est) VALUES (?,?,?,?,?,?,?,?,?)",
                    (r["date"], int(r["unit"]), r["model"], r.get("games"), r.get("diff"), r.get("bb"), r.get("rb"),
                     source, 1 if est else 0),
                )
                n += 1
        return n

    def upsert_hall_days(self, rows: list[dict], source: str) -> None:
        with self.conn() as c:
            for r in rows:
                c.execute(
                    "INSERT INTO hall_day(date,total_diff,avg_diff,avg_games,source) VALUES (?,?,?,?,?) "
                    "ON CONFLICT(date) DO UPDATE SET "
                    "total_diff=COALESCE(excluded.total_diff, hall_day.total_diff), "
                    "avg_diff=COALESCE(excluded.avg_diff, hall_day.avg_diff), "
                    "avg_games=COALESCE(excluded.avg_games, hall_day.avg_games), source=excluded.source",
                    (r["date"], r.get("total_diff"), r.get("avg_diff"), r.get("avg_games"), source),
                )

    def upsert_index(self, items: list[tuple[str, str]], source: str) -> None:
        with self.conn() as c:
            c.executemany("INSERT OR REPLACE INTO day_index(date,source,url) VALUES (?,?,?)",
                          [(d, source, u) for d, u in items])

    def log_fetch(self, date: str, source: str, status: str, n_rows: int) -> None:
        with self.conn() as c:
            c.execute("INSERT OR REPLACE INTO fetch_log(date,source,status,n_rows,fetched_at) VALUES (?,?,?,?,?)",
                      (date, source, status, n_rows, datetime.now().isoformat(timespec="seconds")))

    # ---- 読み出し ----
    def fetched_dates(self, source: str, statuses=("ok", "empty")) -> set[str]:
        q = "SELECT date FROM fetch_log WHERE source=? AND status IN (%s)" % ",".join("?" * len(statuses))
        with self.conn() as c:
            return {r[0] for r in c.execute(q, (source, *statuses))}

    def index(self, source: str) -> dict[str, str]:
        with self.conn() as c:
            return dict(c.execute("SELECT date,url FROM day_index WHERE source=?", (source,)).fetchall())

    def units(self, keyword: str | None = None) -> pd.DataFrame:
        with self.conn() as c:
            if keyword:
                df = pd.read_sql_query("SELECT * FROM unit_day WHERE model LIKE ? ORDER BY date, unit", c,
                                       params=(f"%{keyword}%",))
            else:
                df = pd.read_sql_query("SELECT * FROM unit_day ORDER BY date, unit", c)
        return df

    def hall_days(self) -> pd.DataFrame:
        with self.conn() as c:
            return pd.read_sql_query("SELECT * FROM hall_day ORDER BY date", c)

    # ---- CSV で保存・復元（クラウド実行用。CSVをリポジトリに保存する） ----
    TABLES = {
        "units.csv": ("unit_day", ["date", "unit", "model", "games", "diff", "bb", "rb", "source", "est"]),
        "hall_days.csv": ("hall_day", ["date", "total_diff", "avg_diff", "avg_games", "source"]),
        "day_index.csv": ("day_index", ["date", "source", "url"]),
        "fetch_log.csv": ("fetch_log", ["date", "source", "status", "n_rows", "fetched_at"]),
    }

    def import_csv(self, folder: Path) -> int:
        import csv
        n = 0
        with self.conn() as c:
            for fname, (table, cols) in self.TABLES.items():
                p = Path(folder) / fname
                if not p.exists():
                    continue
                with open(p, encoding="utf-8", newline="") as f:
                    rows = [tuple((r.get(k) if r.get(k) not in ("", None) else None) for k in cols)
                            for r in csv.DictReader(f)]
                c.executemany(f"INSERT OR REPLACE INTO {table}({','.join(cols)}) VALUES ({','.join('?' * len(cols))})", rows)
                n += len(rows)
        return n

    def export_csv(self, folder: Path, keyword: str | None = None) -> None:
        """keyword を指定すると、台データは機種名にそれを含む行だけ保存（容量節約）。"""
        import csv
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        with self.conn() as c:
            for fname, (table, cols) in self.TABLES.items():
                q = f"SELECT {','.join(cols)} FROM {table}"
                params: tuple = ()
                if table == "unit_day" and keyword:
                    q += " WHERE model LIKE ?"
                    params = (f"%{keyword}%",)
                q += " ORDER BY " + ", ".join(cols[:2])
                rows = c.execute(q, params).fetchall()
                tmp = folder / (fname + ".tmp")
                with open(tmp, "w", encoding="utf-8", newline="") as f:
                    w = csv.writer(f, lineterminator="\n")
                    w.writerow(cols)
                    w.writerows(rows)
                tmp.replace(folder / fname)

    def coverage(self) -> dict:
        with self.conn() as c:
            r = c.execute("SELECT MIN(date), MAX(date), COUNT(DISTINCT date), COUNT(*) FROM unit_day").fetchone()
            h = c.execute("SELECT MIN(date), MAX(date), COUNT(*) FROM hall_day").fetchone()
            by_src = c.execute("SELECT source, COUNT(DISTINCT date) FROM unit_day GROUP BY source").fetchall()
        return {"unit_first": r[0], "unit_last": r[1], "unit_days": r[2], "unit_rows": r[3],
                "hall_first": h[0], "hall_last": h[1], "hall_days": h[2], "by_source": dict(by_src)}
