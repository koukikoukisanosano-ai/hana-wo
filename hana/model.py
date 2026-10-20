"""店の癖を学習して、対象日の各台の「高設定確率」を出す。

考え方
------
1. 過去の各台・各日について、BB/RB/G数から「高設定(既定: 5以上)だった確率」を推定（estimate.py）。
2. それを“正解ラベル”として、次のような条件ごとの高設定率を数える（＝店の癖）。
     ・日付の末尾（0のつく日、5のつく日…）  ・曜日  ・ゾロ目日など特定日
     ・台番（その台自体の入りやすさ）  ・台番末尾  ・台番末尾と日付末尾の一致
     ・角台  ・前回の差枚（凹み→上げ／勝ち→据え置き・下げ）  ・前回の高設定挙動  ・連続凹み日数
     ・店全体の差枚（アナスロ一覧の約4年分）から見た、その日付の強さ
   件数が少ない条件は平均に寄せる（縮小推定）ので、偶然の偏りに振り回されにくい。
3. 条件ごとの“上げ幅”を、過去データで重みを調整したロジスティック回帰でまとめて確率にする。
   重みの調整は「その日より前のデータだけ」で計算した値で行う（未来の情報を使わない）。
"""
from __future__ import annotations

import calendar
from dataclasses import dataclass, field
from datetime import date as Date, timedelta

import numpy as np
import pandas as pd

from .estimate import add_estimates

WEEKDAY = "月火水木金土日"

# 因子名 → (表示名, 縮小の強さm)
FACTORS = {
    # 63日分の検証（各日をその日より前のデータだけで予想）で効いたものだけを使う。
    # 台番・末尾・角台・台ごとの実績・前日差枚などは、翌日以降ほとんど続かず精度を下げたので外した。
    "f_digit":    ("日付末尾", 90.0),
    "f_wday":     ("曜日", 90.0),
    "f_special":  ("特定日", 90.0),
    "f_seq2":     ("2日前→前日の流れ", 20.0),
    "f_seq3":     ("3日前→前日の流れ", 30.0),
}
HALL_FACTORS = {"h_digit": "店全体・日付末尾", "h_wday": "店全体・曜日"}


@dataclass
class Settings:
    keyword: str = "ハナハナ"
    high_min: int = 5
    label_min_games: int = 1500
    event_digits: tuple[int, ...] = (0, 5)
    prev_max_gap: int = 1          # 「前回」として使うのは何日前まで（1=前日のみ）
    islands: list[list[int]] | None = None   # [[開始, 終了], ...] 未指定なら連番の塊を島とみなす
    l2: float = 4.0                # 重みを1（素直な足し算）に寄せる強さ
    min_train_days: int = 45       # これ未満のデータ日数なら重み調整をせず素直に足す
    judge_min_games: int = 2000    # これ未満のG数の日は設定を判定しない


# ---------------------------------------------------------------- 前処理
def prepare(units: pd.DataFrame, cfg: Settings) -> tuple[pd.DataFrame, np.ndarray]:
    df = units[units["model"].str.contains(cfg.keyword, na=False)].copy()
    df = df.dropna(subset=["games", "bb", "rb"])
    df["date"] = pd.to_datetime(df["date"])
    df, prior = add_estimates(df, high_min=cfg.high_min, min_games=300, judge_min_games=cfg.judge_min_games)
    # 答え = その日の合算で「設定{high_min}以上」と判定されたか（サイトの判定設定表と同じ物差し）。
    # 判定できない日（G数不足）は学習に使わない。
    judged = df["judge"].notna().to_numpy()
    info = np.where(judged, 1.0, 0.0)
    info = np.where(df.get("est", 0) == 1, info * 0.5, info)
    df["w"] = info
    df["y"] = (df["judge"] >= cfg.high_min).astype(float).where(judged, 0.0)
    return df.sort_values(["date", "unit"]).reset_index(drop=True), prior


def corners(unit_list: list[int], islands: list[list[int]] | None) -> dict[int, str]:
    """台番 → 'corner'（角）/ 'corner2'（角の隣）/ 'mid'"""
    units = sorted(set(int(u) for u in unit_list))
    runs: list[list[int]] = []
    if islands:
        for a, b in islands:
            runs.append([u for u in units if a <= u <= b])
    else:
        cur: list[int] = []
        for u in units:
            if cur and u != cur[-1] + 1:
                runs.append(cur)
                cur = []
            cur.append(u)
        if cur:
            runs.append(cur)
    pos = {u: "mid" for u in units}
    for r in runs:
        if not r:
            continue
        for u in (r[0], r[-1]):
            pos[u] = "corner"
        for u in (r[1] if len(r) > 3 else None, r[-2] if len(r) > 3 else None):
            if u is not None and pos[u] == "mid":
                pos[u] = "corner2"
    return pos


def day_keys(d: pd.Series, cfg: Settings) -> pd.DataFrame:
    dd = d.dt.day
    special = np.where((dd == 11) | (dd == 22), "ゾロ目日",
               np.where(dd.to_numpy() == d.dt.month.to_numpy(), "月日ゾロ目",
               np.where(dd == d.dt.days_in_month, "月末", "通常")))
    return pd.DataFrame({"k_digit": (dd % 10).astype(int), "k_wday": d.dt.weekday.astype(int),
                         "k_special": special}, index=d.index)


def diff_bucket(x) -> str:
    if x is None or pd.isna(x):
        return "なし"
    if x <= -1500:
        return "大凹み"
    if x <= -500:
        return "凹み"
    if x < 500:
        return "平行"
    if x < 1500:
        return "勝ち"
    return "大勝ち"


def add_prev(df: pd.DataFrame, cfg: Settings) -> pd.DataFrame:
    """各行に『前回（max_gap日以内）』の情報と連続凹み日数を付ける。"""
    df = df.sort_values(["unit", "date"]).copy()
    g = df.groupby("unit")
    prev_date = g["date"].shift(1)
    gap = (df["date"] - prev_date).dt.days
    ok = gap.notna() & (gap <= cfg.prev_max_gap)
    df["prev_diff"] = np.where(ok, g["diff"].shift(1), np.nan)
    df["prev_p"] = np.where(ok, g["p_high"].shift(1), np.nan)
    low = (df["diff"] <= -500) & (df["p_high"].fillna(0) < 0.15)
    # 直前までの連続凹み日数（連続日のみ）
    streak = np.zeros(len(df), dtype=int)
    units, dates, lows = df["unit"].to_numpy(), df["date"].to_numpy(), low.to_numpy()
    for i in range(1, len(df)):
        if units[i] == units[i - 1] and (dates[i] - dates[i - 1]) <= np.timedelta64(cfg.prev_max_gap, "D"):
            streak[i] = streak[i - 1] + 1 if lows[i - 1] else 0
    df["streak"] = streak
    df["recent_judge"], df["recent_n"] = recent_judge(df, df["unit"].to_numpy(), df["date"].to_numpy())
    st = judge_state(df["judge"])
    for k in (1, 2, 3):
        okk = (df["date"] - g["date"].shift(k)).dt.days == k
        df[f"st{k}"] = np.where(okk, pd.Series(st, index=df.index).groupby(df["unit"]).shift(k), None)
    return df.sort_values(["date", "unit"]).reset_index(drop=True)


def judge_state(j) -> np.ndarray:
    """判定設定 → 高（5・6）/ 中（3・4）/ 低（1・2）。判定なしは None。"""
    j = pd.Series(j)
    return np.where(j.isna(), None, np.where(j >= 5, "高", np.where(j >= 3, "中", "低")))


def seq_key(*states) -> str:
    if any(s is None or (isinstance(s, float) and np.isnan(s)) for s in states):
        return "なし"
    return "→".join(states)


def recent_judge(hist: pd.DataFrame, units, dates, days: int = 7):
    """各(台, 日)について、その日より前の days 日間の判定設定の平均と日数。"""
    h = hist.dropna(subset=["judge"])[["unit", "date", "judge"]]
    by = {u: (g["date"].to_numpy(), g["judge"].to_numpy(float)) for u, g in h.groupby("unit")}
    mean = np.full(len(units), np.nan)
    n = np.zeros(len(units), dtype=int)
    span = np.timedelta64(days, "D")
    for i, (u, d) in enumerate(zip(units, dates)):
        if u not in by:
            continue
        dd, jj = by[u]
        d = np.datetime64(d)
        m = (dd < d) & (dd >= d - span)
        if m.any():
            mean[i] = jj[m].mean()
            n[i] = int(m.sum())
    return mean, n


def recent_bucket(mean, n) -> str:
    if n < 3 or pd.isna(mean):
        return "なし"
    if mean < 2.5:
        return "低め"
    if mean < 3.5:
        return "普通"
    return "高め"


def add_keys(df: pd.DataFrame, cfg: Settings, pos: dict[int, str]) -> pd.DataFrame:
    df = df.copy()
    dk = day_keys(df["date"], cfg)
    df = pd.concat([df, dk], axis=1)
    df["k_unit"] = df["unit"].astype(int)
    df["k_udigit"] = (df["unit"] % 10).astype(int)
    event = df["k_digit"].isin(cfg.event_digits)
    df["k_match"] = (np.where(event, "旧イベ日", "通常日") + "・"
                     + np.where(df["k_udigit"] == df["k_digit"], "一致", "不一致"))
    df["k_corner"] = df["unit"].map(pos).fillna("mid")
    df["k_prevdiff"] = [diff_bucket(x) for x in df["prev_diff"]]
    df["k_prevhigh"] = np.where(df["prev_p"].isna(), "なし",
                         np.where(df["prev_p"] >= 0.30, "高設定挙動", "通常"))
    df["k_streak"] = np.where(df["streak"] >= 3, "3日以上", df["streak"].astype(int).astype(str) + "日")
    df["k_recent"] = [recent_bucket(m, n) for m, n in zip(df["recent_judge"], df["recent_n"])]
    df["k_seq2"] = [seq_key(a, b) for a, b in zip(df["st2"], df["st1"])]
    df["k_seq3"] = [seq_key(a, b, c) for a, b, c in zip(df["st3"], df["st2"], df["st1"])]
    return df


# ---------------------------------------------------------------- 条件ごとの上げ幅（縮小推定）
def _logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def cum_tables(df: pd.DataFrame, key: str) -> pd.DataFrame:
    """key×日付ごとの累積 Σw·y, Σw（その日を含む）"""
    t = df.groupby([key, "date"]).agg(wy=("wy", "sum"), w=("w", "sum")).reset_index().sort_values("date")
    t["cwy"] = t.groupby(key)["wy"].cumsum()
    t["cw"] = t.groupby(key)["w"].cumsum()
    return t[[key, "date", "cwy", "cw"]]


def lift_asof(rows: pd.DataFrame, hist: pd.DataFrame, key: str, m: float, cutoff_col: str) -> np.ndarray:
    """rows の各行について、cutoff_col 以前のデータだけで key の上げ幅（対数オッズ差）を出す。"""
    hist = hist.copy()
    hist["wy"] = hist["w"] * hist["y"]
    ct = cum_tables(hist, key)
    tot = hist.groupby("date").agg(wy=("wy", "sum"), w=("w", "sum")).sort_index().cumsum().reset_index()
    tot.columns = ["date", "twy", "tw"]
    q = rows[[key, cutoff_col]].copy()
    q["_i"] = np.arange(len(q))
    q = q.rename(columns={cutoff_col: "date"}).sort_values("date")
    q = pd.merge_asof(q, tot, on="date", direction="backward")
    ct = ct.sort_values("date")
    q[key] = q[key].astype(ct[key].dtype) if len(ct) else q[key]
    q = pd.merge_asof(q, ct, on="date", by=key, direction="backward")
    q = q.sort_values("_i")
    base = (q["twy"] / q["tw"]).to_numpy()
    cwy, cw = q["cwy"].fillna(0).to_numpy(), q["cw"].fillna(0).to_numpy()
    rate = (cwy + m * base) / (cw + m)
    out = _logit(rate) - _logit(base)
    return np.nan_to_num(out, nan=0.0)


def hall_lift_asof(rows: pd.DataFrame, hall: pd.DataFrame, key: str, cutoff_col: str, m: float = 15.0) -> np.ndarray:
    """店全体の平均差枚（全機種）から見た、その日付条件の強さ（標準化・縮小済み）。"""
    h = hall.dropna(subset=["avg_diff"]).copy()
    if h.empty:
        return np.zeros(len(rows))
    h["date"] = pd.to_datetime(h["date"])
    h = pd.concat([h, day_keys(h["date"], Settings()).set_index(h.index)], axis=1).sort_values("date")
    h["y2"] = h["avg_diff"] ** 2
    h["one"] = 1.0
    tot = h[["date", "avg_diff", "y2", "one"]].set_index("date").cumsum().reset_index()
    tot.columns = ["date", "ts", "ts2", "tn"]
    ct = h[[key, "date", "avg_diff", "one"]].copy()
    ct["cs"] = ct.groupby(key)["avg_diff"].cumsum()
    ct["cn"] = ct.groupby(key)["one"].cumsum()
    ct = ct[[key, "date", "cs", "cn"]].sort_values("date")
    q = rows[[key, cutoff_col]].copy()
    q["_i"] = np.arange(len(q))
    q = q.rename(columns={cutoff_col: "date"}).sort_values("date")
    q = pd.merge_asof(q, tot, on="date", direction="backward")
    q[key] = q[key].astype(ct[key].dtype)
    q = pd.merge_asof(q, ct, on="date", by=key, direction="backward").sort_values("_i")
    mean = q["ts"] / q["tn"]
    sd = np.sqrt(np.maximum(q["ts2"] / q["tn"] - mean ** 2, 1.0))
    cs, cn = q["cs"].fillna(0), q["cn"].fillna(0)
    shr = (cs + m * mean) / (cn + m)
    return np.nan_to_num(((shr - mean) / sd).to_numpy(), nan=0.0)


# ---------------------------------------------------------------- 学習・予測
def build_X(rows: pd.DataFrame, hist: pd.DataFrame, hall: pd.DataFrame, cutoff_col: str) -> pd.DataFrame:
    X = pd.DataFrame(index=rows.index)
    for f, (_, m) in FACTORS.items():
        X[f] = lift_asof(rows, hist, "k" + f[1:], m, cutoff_col)
    X["h_digit"] = hall_lift_asof(rows, hall, "k_digit", cutoff_col)
    X["h_wday"] = hall_lift_asof(rows, hall, "k_wday", cutoff_col)
    return X


PREV_FACTORS = ("f_prevdiff", "f_prevhigh", "f_streak")   # 判定の流れは前日がなくても最新日までで代用するので外す


def neutralize_prev(X: pd.DataFrame, rows: pd.DataFrame) -> None:
    """前日データがまだ無い（データ公開の遅れ等）台は、前日系の要因を0にする。"""
    mask = rows["prev_p"].isna().to_numpy()
    for c in PREV_FACTORS:
        X.loc[mask, c] = 0.0


def fit_combiner(X: np.ndarray, y: np.ndarray, w: np.ndarray, l2: float, base_logit: float,
                 n_hall: int) -> np.ndarray:
    """重み付きロジスティック回帰（ソフトラベル）。係数は1（ホール指標は0）に寄せる正則化つき。"""
    n, k = X.shape
    A = np.hstack([np.ones((n, 1)), X])
    target = np.r_[base_logit, np.ones(k - n_hall), np.zeros(n_hall)]
    beta = target.copy()
    P = np.diag(np.r_[0.0, np.full(k, l2)])
    for _ in range(50):
        p = 1 / (1 + np.exp(-(A @ beta)))
        g = A.T @ (w * (y - p)) - P @ (beta - target)
        H = (A * (w * p * (1 - p))[:, None]).T @ A + P + 1e-6 * np.eye(k + 1)
        step = np.linalg.solve(H, g)
        beta += step
        if np.abs(step).max() < 1e-6:
            break
    # 重みは0以上に（「店全体が強い日なのに確率を下げる」のような逆向きの重みは使わない）
    beta[1:] = np.maximum(beta[1:], 0.0)
    for _ in range(30):   # 切片だけ合わせ直す
        p = 1 / (1 + np.exp(-(A @ beta)))
        g0 = float(np.sum(w * (y - p)))
        h0 = float(np.sum(w * p * (1 - p))) + 1e-9
        beta[0] += g0 / h0
        if abs(g0 / h0) < 1e-7:
            break
    return beta


@dataclass
class Model:
    cfg: Settings
    prior: np.ndarray
    beta: np.ndarray
    cols: list[str]
    hist: pd.DataFrame
    hall: pd.DataFrame
    n_days: int
    tuned: bool
    base_rate: float
    summaries: dict = field(default_factory=dict)


def train(units: pd.DataFrame, hall: pd.DataFrame, cfg: Settings, until: Date | None = None) -> Model:
    """until より前（until自体は含まない）のデータで学習。"""
    df, prior = prepare(units, cfg)
    if until is not None:
        df = df[df["date"] < pd.Timestamp(until)]
        hall = hall[pd.to_datetime(hall["date"]) < pd.Timestamp(until)]
    df = add_prev(df, cfg)
    pos = corners(df.loc[df["date"] == df["date"].max(), "unit"].tolist() if len(df) else [], cfg.islands)
    df = add_keys(df, cfg, pos)
    cols = list(FACTORS) + list(HALL_FACTORS)
    n_days = df["date"].nunique()
    base_rate = float((df["w"] * df["y"]).sum() / max(df["w"].sum(), 1e-9)) if len(df) else 0.1
    base_logit = float(_logit(base_rate))
    beta = np.r_[base_logit, np.ones(len(FACTORS)), np.zeros(len(HALL_FACTORS))]
    tuned = False
    if n_days >= cfg.min_train_days:
        rows = df.copy()
        rows["cut"] = rows["date"] - pd.Timedelta(days=1)   # その日より前だけで計算
        X = build_X(rows, df, hall, "cut")
        neutralize_prev(X, rows)
        first = df["date"].min() + pd.Timedelta(days=21)    # 最初の3週間は助走期間
        m = (rows["date"] >= first) & (rows["w"] > 0)
        if m.sum() > 200:
            beta = fit_combiner(X.loc[m, cols].to_numpy(), rows.loc[m, "y"].to_numpy(),
                                rows.loc[m, "w"].to_numpy(), cfg.l2, base_logit, len(HALL_FACTORS))
            tuned = True
    model = Model(cfg, prior, beta, cols, df, hall, n_days, tuned, base_rate)
    model.summaries = summarize_habits(df, hall)
    return model


def candidates(model: Model, target: Date) -> pd.DataFrame:
    """対象日に予想する台（直近データ日の設置台）と、その特徴。"""
    df = model.hist
    if df.empty:
        return pd.DataFrame()
    last = df["date"].max()
    # 直近1週間に登場した台（ある日のデータが一部欠けても台が消えないように）
    recent = df[df["date"] >= last - pd.Timedelta(days=7)].sort_values("date")
    cur = recent.groupby("unit").tail(1)[["unit", "model"]].copy()
    t = pd.Timestamp(target)
    cur["date"] = t
    pos = corners(cur["unit"].tolist(), model.cfg.islands)
    # 前回情報
    lastrows = df.sort_values("date").groupby("unit").tail(1).set_index("unit")
    gap = (t - lastrows["date"]).dt.days
    ok = gap <= model.cfg.prev_max_gap
    cur["prev_date"] = cur["unit"].map(lastrows["date"])
    cur["prev_diff"] = cur["unit"].map(lastrows["diff"].where(ok))
    cur["prev_p"] = cur["unit"].map(lastrows["p_high"].where(ok))
    lows = (lastrows["diff"] <= -500) & (lastrows["p_high"].fillna(0) < 0.15)
    cur["streak"] = [int((lastrows.loc[u, "streak"] + 1) if (ok.get(u, False) and lows.get(u, False)) else 0)
                     for u in cur["unit"]]
    jd = df.dropna(subset=["judge"]).set_index(["unit", "date"])["judge"]
    # 前日のデータがまだ公開されていない朝は、取れている最新日までの流れで代用する
    have = set(df.dropna(subset=["judge"])["date"])
    off = 0
    while off < 3 and (t - pd.Timedelta(days=1 + off)) not in have:
        off += 1
    for k in (1, 2, 3):
        dk = t - pd.Timedelta(days=k + off)
        cur[f"st{k}"] = [judge_state([jd.get((u, dk), np.nan)])[0] for u in cur["unit"]]
    cur["recent_judge"], cur["recent_n"] = recent_judge(df, cur["unit"].to_numpy(),
                                                        np.full(len(cur), np.datetime64(t)))
    cur = add_keys(cur.reset_index(drop=True), model.cfg, pos)
    return cur


def predict(model: Model, target: Date) -> pd.DataFrame:
    rows = candidates(model, target)
    if rows.empty:
        return rows
    rows["cut"] = rows["date"] - pd.Timedelta(days=1)
    X = build_X(rows, model.hist, model.hall, "cut")
    neutralize_prev(X, rows)
    A = np.hstack([np.ones((len(rows), 1)), X[model.cols].to_numpy()])
    rows["score"] = 1 / (1 + np.exp(-(A @ model.beta)))
    contrib = X[model.cols].to_numpy() * model.beta[1:]
    for i, c in enumerate(model.cols):
        rows["c_" + c] = contrib[:, i]
        rows["x_" + c] = X[c].to_numpy()
    rows = rows.sort_values("score", ascending=False).reset_index(drop=True)
    rows["rank"] = np.arange(1, len(rows) + 1)
    return rows


# ---------------------------------------------------------------- 店の癖の要約（結果ページ用）
def summarize_habits(df: pd.DataFrame, hall: pd.DataFrame) -> dict:
    out = {}
    if df.empty:
        return out
    base = (df["w"] * df["y"]).sum() / max(df["w"].sum(), 1e-9)

    def table(key, label_fn, min_w=1.0):
        t = df.groupby(key).apply(lambda g: pd.Series({
            "rate": (g["w"] * g["y"]).sum() / max(g["w"].sum(), 1e-9),
            "days": g["date"].nunique(), "n": int((g["w"] > 0).sum()),
            "avg_diff": g["diff"].mean()}), include_groups=False).reset_index()
        t = t[t["n"] >= min_w]
        t["label"] = [label_fn(v) for v in t[key]]
        t["ratio"] = t["rate"] / base if base > 0 else np.nan
        return t
    out["base"] = base
    out["digit"] = table("k_digit", lambda v: f"{v}のつく日")
    out["wday"] = table("k_wday", lambda v: f"{WEEKDAY[v]}曜")
    out["special"] = table("k_special", str)
    out["udigit"] = table("k_udigit", lambda v: f"末尾{v}")
    out["match"] = table("k_match", lambda v: v.split("・")[0] + ("・末尾一致" if v.endswith("・一致") else "・末尾不一致"))
    out["corner"] = table("k_corner", lambda v: {"corner": "角台", "corner2": "角2", "mid": "中"}.get(v, v))
    out["prevdiff"] = table("k_prevdiff", lambda v: f"前日{v}")
    out["prevhigh"] = table("k_prevhigh", lambda v: f"前日{v}")
    out["streak"] = table("k_streak", lambda v: f"凹み{v}")
    out["recent"] = table("k_recent", lambda v: f"直近7日の判定が{v}")
    out["seq2"] = table("k_seq2", lambda v: f"判定の流れ {v}")
    out["seq3"] = table("k_seq3", lambda v: f"判定の流れ {v}")
    out["unit"] = table("k_unit", lambda v: f"{v}番")
    # 店全体（全機種）の日付末尾別 平均差枚
    h = hall.dropna(subset=["avg_diff"]).copy()
    if not h.empty:
        h["date"] = pd.to_datetime(h["date"])
        h["digit"] = h["date"].dt.day % 10
        h["wday"] = h["date"].dt.weekday
        out["hall_digit"] = h.groupby("digit")["avg_diff"].agg(["mean", "count"]).reset_index()
        out["hall_wday"] = h.groupby("wday")["avg_diff"].agg(["mean", "count"]).reset_index()
        out["hall_span"] = (h["date"].min().date(), h["date"].max().date(), len(h))
    return out


_REASON_KEYS = {
    "f_digit": ("digit", "k_digit"), "f_wday": ("wday", "k_wday"), "f_special": ("special", "k_special"),
    "f_unit": ("unit", "k_unit"), "f_udigit": ("udigit", "k_udigit"), "f_match": ("match", "k_match"),
    "f_corner": ("corner", "k_corner"), "f_prevdiff": ("prevdiff", "k_prevdiff"),
    "f_prevhigh": ("prevhigh", "k_prevhigh"), "f_streak": ("streak", "k_streak"),
    "f_recent": ("recent", "k_recent"),
    "f_seq2": ("seq2", "k_seq2"), "f_seq3": ("seq3", "k_seq3"),
}


def reasons(row: pd.Series, model: Model, top: int = 4, min_effect: float = 0.06) -> list[tuple[str, float, str]]:
    """結果ページに出す根拠: (短い表示, 効き具合, 詳細)。プラス要因・マイナス要因の大きい順。"""
    items = []
    s = model.summaries
    for c in model.cols:
        v = float(row.get("c_" + c, 0.0))
        if abs(v) < min_effect:
            continue
        if c in _REASON_KEYS:
            tname, key = _REASON_KEYS[c]
            t = s.get(tname)
            hit = t[t[key] == row[key]] if t is not None and len(t) else None
            if hit is None or hit.empty:
                continue
            h = hit.iloc[0]
            if c == "f_unit" and h["days"] < 10:      # 日数が少ない台の個性は表示しない
                continue
            label = str(h["label"])
            if c == "f_unit":
                label = "この台の実績"
            if c == "f_recent" and pd.notna(row.get("recent_judge")):
                label = f"直近7日の判定 平均{row['recent_judge']:.1f}"
            if c == "f_prevdiff" and pd.notna(row.get("prev_diff")):
                label += f"({int(row['prev_diff']):+,}枚)"
            text = f"{label} ×{h['ratio']:.2f}"
            detail = f"{h['label']}: 高設定率 {h['rate']*100:.1f}%（平常 {s['base']*100:.1f}%）・{int(h['days'])}日分"
        elif c in ("h_digit", "h_wday"):
            tab = s.get("hall_digit" if c == "h_digit" else "hall_wday")
            k = int(row["k_digit"] if c == "h_digit" else row["k_wday"])
            if tab is None or tab.empty:
                continue
            col = "digit" if c == "h_digit" else "wday"
            hit = tab[tab[col] == k]
            name = f"{k}のつく日" if c == "h_digit" else f"{WEEKDAY[k]}曜"
            mean = float(hit["mean"].iloc[0]) if len(hit) else float("nan")
            text = f"店全体の{name} {mean:+.0f}枚"
            detail = f"店全体（全機種）の{name}の平均差枚 {mean:+.0f}枚/台（全日平均 {tab['mean'].mean():+.0f}枚）"
        else:
            continue
        items.append((text, v, detail))
    items.sort(key=lambda t: -abs(t[1]))
    return items[:top]


def target_day_info(target: Date, cfg: Settings) -> dict:
    d = target
    tags = []
    if d.day % 10 in cfg.event_digits:
        tags.append(f"{d.day % 10}のつく日（旧イベント日）")
    if d.day in (11, 22):
        tags.append("ゾロ目日")
    if d.day == d.month:
        tags.append("月日ゾロ目")
    if d.day == calendar.monthrange(d.year, d.month)[1]:
        tags.append("月末")
    return {"date": d, "wday": WEEKDAY[d.weekday()], "digit": d.day % 10, "tags": tags}
