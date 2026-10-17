"""ハナハナの高設定期待度を「査定点（＋／－）」で出す。

しくみ
------
0. 各台・各日について、BIG・REG回数とG数から「高設定（既定: 設定5以上）だった確率」を推定する（estimate.py）。
   これを過去の“答え”として使う。差枚は運の影響が大きいので答えには使わない。

1. 日の強さ（今日は島全体に高設定が多そうか）
   日付末尾・曜日・特定日（ゾロ目など）・店全体（全機種）の傾向 から、その日の島全体の高設定率を見積もる。
   全台に同じだけ効くので、台の順位は変えない。

2. 台ごとの査定（同じ日の中で、どの台に入りやすいか）
   この台の実績・台番末尾・日付と台番末尾の一致・台の位置・前日の差枚・連続凹み・最近の高設定挙動
   の条件ごとに、＋（入りやすい）／－（入りにくい）の点をつけて合計する。
   - 点は「同じ日の島全体の平均と比べて、その条件の台はどうだったか」で決める（日の強弱が混ざらない）。
   - 全条件を同時に当てはめて点を決めるので、同じ事実を二重に数えない
     （例: 686番だけが強い → 末尾6・角2まで強く見える、を防ぐ）。
   - データの少ない条件は0点に寄せる（縮小）。データがたまるほど点差がはっきりする。
   - 10点で入りやすさ約2倍、－10点で約半分。

期待度 = 日の強さ × 2^(台の合計点/10)（ロジスティックで0〜100%に収める）
"""
from __future__ import annotations

import calendar
from dataclasses import dataclass, field
from datetime import date as Date

import numpy as np
import pandas as pd

from .estimate import add_estimates

WEEKDAY = "月火水木金土日"
PTS = 10.0 / np.log(2.0)          # 対数オッズ → 点（10点でオッズ2倍）

# 台ごとの査定: key -> (表示名, 縮小の強さ m)。m は「何台日ぶんのデータで実力の半分を信じるか」
UNIT_FACTORS = {
    "unit":   ("この台の実績", 15.0),
    "udigit": ("台番末尾", 45.0),
    "match":  ("日付と台番末尾", 45.0),
    "pos":    ("台の位置", 60.0),
    "prev":   ("前日の差枚", 40.0),
    "streak": ("連続凹み", 40.0),
    "recent": ("最近の高設定挙動", 40.0),
}
# 日の強さ: key -> (表示名, m)。島全体で1日あたり約25台日ぶん
DAY_FACTORS = {
    "digit":   ("日付末尾", 75.0),
    "wday":    ("曜日", 75.0),
    "special": ("特定日", 75.0),
}
HALL_M = 500.0                     # 店全体（全機種）の傾向の縮小の強さ
PREV_DEP = ("prev", "streak", "recent")   # 前日のデータが無いと判定できない条件
POS_LABEL = {"corner": "角台", "corner2": "角2", "mid": "島の中"}
RECENT_LEVELS = ("前日", "2〜3日前", "4〜7日前", "8日以上なし")


@dataclass
class Settings:
    keyword: str = "ハナハナ"
    high_min: int = 5
    label_min_games: int = 1500
    event_digits: tuple[int, ...] = (0, 5)
    prev_max_gap: int = 1          # 「前日」として使うのは何日前まで（1=前日のみ）
    islands: list[list[int]] | None = None   # [[開始, 終了], ...] 未指定なら連番の塊を島とみなす
    half_life: float = 180.0       # 古いデータほど軽く扱う（この日数で重み半分）
    min_tune_days: int = 60        # これ以上のデータ日数で縮小の強さを自動調整


# ---------------------------------------------------------------- 前処理
def _logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def prepare(units: pd.DataFrame, cfg: Settings) -> tuple[pd.DataFrame, np.ndarray]:
    df = units[units["model"].str.contains(cfg.keyword, na=False)].copy()
    df = df.dropna(subset=["games", "bb", "rb"])
    df["date"] = pd.to_datetime(df["date"])
    df, prior = add_estimates(df, high_min=cfg.high_min, min_games=300)
    info = np.clip(df["games"].to_numpy(float) / 6000.0, 0.0, 1.0)
    info = np.where(df["games"] < cfg.label_min_games, 0.0, info)
    est = df["est"].fillna(0).to_numpy() if "est" in df else np.zeros(len(df))
    info = np.where(est == 1, info * 0.5, info)
    df["w"] = info
    df["y"] = df["p_high"].fillna(0.0)
    return df.sort_values(["date", "unit"]).reset_index(drop=True), prior


def high_threshold(prior: np.ndarray, high_min: int) -> float:
    """「高設定挙動」とみなす高設定確率のライン（店の平均の2.5倍。0.15〜0.5）。"""
    return float(np.clip(2.5 * prior[high_min - 1:].sum(), 0.15, 0.5))


def corners(unit_list: list[int], islands: list[list[int]] | None) -> dict[int, str]:
    """台番 → 'corner'（角）/ 'corner2'（角の隣）/ 'mid'"""
    units = sorted(set(int(u) for u in unit_list))
    runs = island_runs(units, islands)
    pos = {u: "mid" for u in units}
    for r in runs:
        if not r:
            continue
        for u in (r[0], r[-1]):
            pos[u] = "corner"
        if len(r) > 3:
            for u in (r[1], r[-2]):
                if pos[u] == "mid":
                    pos[u] = "corner2"
    return pos


def island_runs(units: list[int], islands: list[list[int]] | None) -> list[list[int]]:
    units = sorted(set(int(u) for u in units))
    if islands:
        return [[u for u in units if a <= u <= b] for a, b in islands]
    runs: list[list[int]] = []
    cur: list[int] = []
    for u in units:
        if cur and u != cur[-1] + 1:
            runs.append(cur)
            cur = []
        cur.append(u)
    if cur:
        runs.append(cur)
    return runs


def day_keys(d: pd.Series) -> pd.DataFrame:
    dd = d.dt.day
    special = np.where((dd == 11) | (dd == 22), "ゾロ目日",
               np.where(dd.to_numpy() == d.dt.month.to_numpy(), "月日ゾロ目",
               np.where(dd == d.dt.days_in_month, "月末", "通常日")))
    return pd.DataFrame({"k_digit": (dd % 10).astype(int), "k_wday": d.dt.weekday.astype(int),
                         "k_special": special}, index=d.index)


def diff_bucket(x) -> str:
    if x is None or pd.isna(x):
        return ""
    if x <= -1500:
        return "大凹み"
    if x <= -500:
        return "凹み"
    if x < 500:
        return "平行"
    if x < 1500:
        return "勝ち"
    return "大勝ち"


def add_history(df: pd.DataFrame, cfg: Settings, thr: float) -> pd.DataFrame:
    """各行に、その日より前のデータから分かること（前日の差枚・連続凹み・最近の高設定挙動）を付ける。
    結果が未定の行（予想対象日）も入れてよい（diff・p_high が NaN）。"""
    df = df.sort_values(["unit", "date"]).reset_index(drop=True)
    n = len(df)
    units = df["unit"].to_numpy()
    dates = df["date"].to_numpy().astype("datetime64[D]")
    diff = df["diff"].to_numpy(float)
    ph = df["p_high"].to_numpy(float)
    prev_ok = np.zeros(n, bool)
    prev_diff = np.full(n, np.nan)
    prev_p = np.full(n, np.nan)
    streak = np.zeros(n, int)
    recent = np.full(n, "", dtype=object)
    gap_max = np.timedelta64(cfg.prev_max_gap, "D")
    day = np.timedelta64(1, "D")
    cur = None
    first = last_date = last_high = None
    run = 0
    for i in range(n):
        if units[i] != cur:
            cur, first, last_date, last_high, run = units[i], dates[i], None, None, 0
        ok = last_date is not None and (dates[i] - last_date) <= gap_max
        if ok:
            prev_ok[i] = True
            prev_diff[i] = diff[i - 1]
            prev_p[i] = ph[i - 1]
            streak[i] = run
            if last_high is not None:
                g = int((dates[i] - last_high) / day)
                recent[i] = "前日" if g <= 1 else "2〜3日前" if g <= 3 else "4〜7日前" if g <= 7 else "8日以上なし"
            elif (dates[i] - first) >= 8 * day:
                recent[i] = "8日以上なし"
        has = not (np.isnan(diff[i]) and np.isnan(ph[i]))
        if has:
            lose = (not np.isnan(diff[i])) and diff[i] <= -500
            run = (run + 1 if lose else 0) if ok else (1 if lose else 0)
            if not np.isnan(ph[i]) and ph[i] >= thr:
                last_high = dates[i]
            last_date = dates[i]
    df["prev_ok"] = prev_ok
    df["prev_diff"] = prev_diff
    df["prev_p"] = prev_p
    df["streak"] = streak
    df["recent"] = recent
    return df.sort_values(["date", "unit"]).reset_index(drop=True)


def add_keys(df: pd.DataFrame, cfg: Settings, pos: dict[int, str]) -> pd.DataFrame:
    df = df.copy()
    dk = day_keys(df["date"])
    for c in dk.columns:
        df[c] = dk[c]
    u = df["unit"].astype(int)
    df["k_unit"] = u.astype(str)
    df["k_udigit"] = (u % 10).astype(str)
    event = df["k_digit"].isin(cfg.event_digits)
    match = (u % 10) == df["k_digit"]
    df["k_match"] = np.where(~match, "不一致", np.where(event, "旧イベ日に一致", "通常日に一致"))
    df["k_pos"] = u.map(pos).fillna("mid")
    po = df["prev_ok"].to_numpy(bool)
    df["k_prev"] = np.where(po, [diff_bucket(x) for x in df["prev_diff"]], "")
    st = df["streak"].to_numpy(int)
    df["k_streak"] = np.where(po, np.where(st >= 3, "3日以上", np.where(st == 2, "2日", "0〜1日")), "")
    df["k_recent"] = np.where(po, df["recent"].astype(str), "")
    return df


def featurize(units: pd.DataFrame, cfg: Settings) -> tuple[pd.DataFrame, np.ndarray, float]:
    df, prior = prepare(units, cfg)
    thr = high_threshold(prior, cfg.high_min)
    pos = corners(df.loc[df["date"] == df["date"].max(), "unit"].tolist() if len(df) else [], cfg.islands)
    df = add_history(df, cfg, thr)
    df = add_keys(df, cfg, pos)
    return df, prior, thr


# ---------------------------------------------------------------- 表示用の言葉
def level_label(f: str, lv: str) -> str:
    if f == "unit":
        return f"{lv}番"
    if f == "udigit":
        return f"末尾{lv}"
    if f == "match":
        return {"旧イベ日に一致": "日付と末尾が一致（旧イベ日）", "通常日に一致": "日付と末尾が一致",
                "不一致": "日付と末尾が不一致"}.get(lv, lv)
    if f == "pos":
        return POS_LABEL.get(lv, lv)
    if f == "prev":
        return f"前日{lv}"
    if f == "streak":
        return {"0〜1日": "連続凹みなし", "2日": "2日連続凹み", "3日以上": "3日以上連続凹み"}.get(lv, lv)
    if f == "recent":
        return {"前日": "前日に高設定挙動", "2〜3日前": "2〜3日前に高設定挙動", "4〜7日前": "4〜7日前に高設定挙動",
                "8日以上なし": "1週間以上 高設定挙動なし"}.get(lv, lv)
    if f == "digit":
        return f"{lv}のつく日"
    if f == "wday":
        return f"{WEEKDAY[int(lv)]}曜"
    return str(lv)


LEVEL_HINT = {
    ("prev", "大凹み"): "-1500枚以下", ("prev", "凹み"): "-500〜-1499枚", ("prev", "平行"): "±499枚",
    ("prev", "勝ち"): "+500〜+1499枚", ("prev", "大勝ち"): "+1500枚以上",
    ("recent", "前日"): "据え置きの傾向", ("streak", "0〜1日"): "",
}


# ---------------------------------------------------------------- あてはめ
def onehot(df: pd.DataFrame, factors, vocab: dict | None = None):
    if vocab is None:
        vocab = {f: sorted(v for v in df["k_" + f].astype(str).unique() if v != "") for f in factors}
    cols = [(f, lv) for f in factors for lv in vocab.get(f, [])]
    X = np.zeros((len(df), len(cols)))
    j = 0
    for f in factors:
        vals = df["k_" + f].astype(str).to_numpy()
        for lv in vocab.get(f, []):
            X[:, j] = vals == lv
            j += 1
    return X, cols, vocab


def fit_logit(X: np.ndarray, y: np.ndarray, w: np.ndarray, offset: np.ndarray, lam: np.ndarray,
              iters: int = 60) -> np.ndarray:
    """重み付き・ソフトラベルのロジスティック回帰。係数ごとに0へ寄せる強さ lam（切片は自由）。"""
    n, k = X.shape
    A = np.hstack([np.ones((n, 1)), X])
    P = np.r_[0.0, lam]
    beta = np.zeros(k + 1)
    sw = max(float(w.sum()), 1e-9)
    beta[0] = float(_logit(np.sum(w * y) / sw) - np.sum(w * offset) / sw)
    for _ in range(iters):
        p = _sigmoid(offset + A @ beta)
        g = A.T @ (w * (y - p)) - P * beta
        H = (A * (w * p * (1 - p))[:, None]).T @ A + np.diag(P) + 1e-8 * np.eye(k + 1)
        step = np.linalg.solve(H, g)
        beta += step
        if np.abs(step).max() < 1e-7:
            break
    return beta


def hall_lift_asof(rows: pd.DataFrame, hall: pd.DataFrame, key: str, cutoff_col: str, m: float = 15.0) -> np.ndarray:
    """店全体の平均差枚（全機種）から見た、その日付条件の強さ（標準化・縮小済み。cutoff以前のデータのみ）。"""
    if hall is None or len(hall) == 0:
        return np.zeros(len(rows))
    h = hall.dropna(subset=["avg_diff"]).copy()
    if h.empty:
        return np.zeros(len(rows))
    h["date"] = pd.to_datetime(h["date"])
    h = pd.concat([h, day_keys(h["date"]).set_index(h.index)], axis=1).sort_values("date")
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


def _day_table(df: pd.DataFrame, cfg: Settings) -> pd.DataFrame:
    last = df["date"].max()
    d = df[df["w"] > 0]
    t = d.assign(wy=d["w"] * d["y"]).groupby("date").agg(wy=("wy", "sum"), W=("w", "sum")).reset_index()
    t["ybar"] = (t["wy"] / t["W"]).clip(0.003, 0.997)
    t["decay"] = 0.5 ** ((last - t["date"]).dt.days / cfg.half_life)
    dk = day_keys(t["date"])
    return pd.concat([t, dk], axis=1)


def _day_X(t: pd.DataFrame, hall: pd.DataFrame, vocab=None):
    X, cols, vocab = onehot(t, list(DAY_FACTORS), vocab)
    q = t[["date", "k_digit", "k_wday"]].copy()
    q["cut"] = q["date"] - pd.Timedelta(days=1)
    hz = np.c_[hall_lift_asof(q, hall, "k_digit", "cut"), hall_lift_asof(q, hall, "k_wday", "cut")]
    return np.hstack([X, hz]), cols + [("hall", "digit"), ("hall", "wday")], vocab


@dataclass
class Fit:
    base_rate: float
    day_vocab: dict
    day_cols: list
    day_beta: np.ndarray
    unit_vocab: dict
    unit_cols: list
    unit_beta: np.ndarray
    strength: float
    tuned: bool
    n_days: int
    pos_auto: bool


def _unit_lam(cols, v: float, strength: float, pos_auto: bool) -> np.ndarray:
    lam = []
    for f, _ in cols:
        m = UNIT_FACTORS[f][1] * (2.0 if (f == "pos" and pos_auto) else 1.0)
        lam.append(m * v * strength)
    return np.array(lam)


def _unit_offsets(df: pd.DataFrame, t: pd.DataFrame) -> np.ndarray:
    return _logit(df["date"].map(t.set_index("date")["ybar"]).to_numpy(float))


def fit(df: pd.DataFrame, hall: pd.DataFrame, cfg: Settings, strength: float | None = None) -> Fit:
    """特徴量つきの過去データ df で、日の強さと台ごとの査定点をあてはめる。"""
    tr = df[df["w"] > 0]
    n_days = int(tr["date"].nunique())
    base = float((tr["w"] * tr["y"]).sum() / max(tr["w"].sum(), 1e-9)) if len(tr) else 0.1
    v = base * (1 - base)
    pos_auto = not cfg.islands
    if len(tr) == 0:
        return Fit(base, {}, [], np.r_[_logit(base)], {}, [], np.zeros(1), 1.0, False, 0, pos_auto)
    t = _day_table(tr, cfg)
    # 1) 日の強さ
    Xd, dcols, dvocab = _day_X(t, hall)
    lam_d = np.array([DAY_FACTORS[f][1] * v if f in DAY_FACTORS else HALL_M * v for f, _ in dcols])
    day_beta = fit_logit(Xd, t["ybar"].to_numpy(), (t["W"] * t["decay"]).to_numpy(), np.zeros(len(t)), lam_d)
    # 2) 台ごとの査定（オフセット = その日の島全体の平均 → 同じ日の中での比較になる）
    last = tr["date"].max()
    w = tr["w"].to_numpy() * 0.5 ** ((last - tr["date"]).dt.days.to_numpy() / cfg.half_life)
    off = _unit_offsets(tr, t)
    Xu, ucols, uvocab = onehot(tr, list(UNIT_FACTORS))
    tuned = False
    if strength is None:
        strength = 1.0
        if n_days >= cfg.min_tune_days:
            strength, tuned = _tune_strength(tr, t, cfg, v, pos_auto), True
    unit_beta = fit_logit(Xu, tr["y"].to_numpy(), w, off, _unit_lam(ucols, v, strength, pos_auto))
    # 表示用に、条件ごとに「よくある台（データ上の平均）」が0点になるよう切片へ寄せ替える（予想値は変わらない）
    _recenter(unit_beta, Xu, ucols, w)
    _recenter(day_beta, Xd, dcols, (t["W"] * t["decay"]).to_numpy(), skip=("hall",))
    return Fit(base, dvocab, dcols, day_beta, uvocab, ucols, unit_beta, strength, tuned, n_days, pos_auto)


def _recenter(beta: np.ndarray, X: np.ndarray, cols: list, w: np.ndarray, skip=()) -> None:
    facs = {f for f, _ in cols if f not in skip}
    for fac in facs:
        idx = [j for j, (f, _) in enumerate(cols) if f == fac]
        sub = X[:, idx]
        known = sub.sum(axis=1) > 0           # 判定できない行（前日データなし等）は0点のまま
        if not known.any():
            continue
        wk = w[known]
        mean = float(np.sum(wk * (sub[known] @ beta[1 + np.array(idx)])) / max(wk.sum(), 1e-12))
        beta[1 + np.array(idx)] -= mean
        beta[0] += mean          # 判定できる行の予想値は変わらない。判定できない行は「平均的な台」扱いになる


def _tune_strength(tr: pd.DataFrame, t: pd.DataFrame, cfg: Settings, v: float, pos_auto: bool) -> float:
    """古い75%の日であてはめ、新しい25%の日で当たり具合（同じ日の中での順位の確からしさ）を比べて、縮小の強さを選ぶ。"""
    dates = np.sort(tr["date"].unique())
    cut = dates[int(len(dates) * 0.75)]
    a, b = tr[tr["date"] < cut], tr[tr["date"] >= cut]
    if len(a) < 200 or len(b) < 100:
        return 1.0
    last = a["date"].max()
    wa = a["w"].to_numpy() * 0.5 ** ((last - a["date"]).dt.days.to_numpy() / cfg.half_life)
    Xa, cols, vocab = onehot(a, list(UNIT_FACTORS))
    Xb, _, _ = onehot(b, list(UNIT_FACTORS), vocab)
    oa, ob = _unit_offsets(a, t), _unit_offsets(b, t)
    yb, wb = b["y"].to_numpy(), b["w"].to_numpy()
    best, best_loss = 1.0, np.inf
    for s in (0.5, 1.0, 2.0, 4.0, 8.0):
        beta = fit_logit(Xa, a["y"].to_numpy(), wa, oa, _unit_lam(cols, v, s, pos_auto))
        p = np.clip(_sigmoid(ob + beta[0] + Xb @ beta[1:]), 1e-6, 1 - 1e-6)
        loss = -np.sum(wb * (yb * np.log(p) + (1 - yb) * np.log(1 - p)))
        if loss < best_loss:
            best, best_loss = s, loss
    return best


def unit_contrib(f: Fit, rows: pd.DataFrame) -> pd.DataFrame:
    """各行・各条件の対数オッズの寄与（予想対象日の行にも使える）。"""
    X, cols, _ = onehot(rows, list(UNIT_FACTORS), f.unit_vocab)
    out = pd.DataFrame(0.0, index=rows.index, columns=list(UNIT_FACTORS))
    if len(cols):
        b = f.unit_beta[1:]
        for j, (fac, _) in enumerate(cols):
            out[fac] += X[:, j] * b[j]
    return out


def day_contrib(f: Fit, target: Date, hall: pd.DataFrame) -> tuple[float, dict]:
    t = pd.DataFrame({"date": [pd.Timestamp(target)]})
    t = pd.concat([t, day_keys(t["date"])], axis=1)
    X, cols, _ = _day_X(t, hall, f.day_vocab)
    parts: dict[str, float] = {}
    for j, (fac, _) in enumerate(cols):
        parts[fac] = parts.get(fac, 0.0) + float(X[0, j] * f.day_beta[1 + j])
    logit = float(f.day_beta[0] + sum(parts.values()))
    return logit, parts


# ---------------------------------------------------------------- 学習・予測（結果ページ用）
@dataclass
class Model:
    cfg: Settings
    prior: np.ndarray
    thr: float
    fit: Fit
    hist: pd.DataFrame
    hall: pd.DataFrame
    n_days: int
    tuned: bool
    base_rate: float
    evidence: dict = field(default_factory=dict)
    summaries: dict = field(default_factory=dict)
    day: dict = field(default_factory=dict)


def train(units: pd.DataFrame, hall: pd.DataFrame, cfg: Settings, until: Date | None = None) -> Model:
    """until より前（until自体は含まない）のデータで学習。"""
    df, prior, thr = featurize(units, cfg)
    if until is not None:
        df = df[df["date"] < pd.Timestamp(until)]
        hall = hall[pd.to_datetime(hall["date"]) < pd.Timestamp(until)]
    f = fit(df, hall, cfg)
    model = Model(cfg, prior, thr, f, df, hall, f.n_days, f.tuned, f.base_rate)
    model.evidence = evidence(df, f, cfg)
    model.summaries = hall_summary(hall)
    return model


def candidates(model: Model, target: Date) -> pd.DataFrame:
    """対象日に予想する台（直近1週間に登場した台）と、その特徴。"""
    df = model.hist
    if df.empty:
        return pd.DataFrame()
    last = df["date"].max()
    recent = df[df["date"] >= last - pd.Timedelta(days=7)].sort_values("date")
    cur = recent.groupby("unit").tail(1)[["unit", "model"]].copy()
    cur["date"] = pd.Timestamp(target)
    for c in ("diff", "p_high", "games", "bb", "rb"):
        cur[c] = np.nan
    base_cols = ["unit", "model", "date", "diff", "p_high", "games", "bb", "rb"]
    panel = pd.concat([df[base_cols], cur[base_cols]], ignore_index=True)
    panel = add_history(panel, model.cfg, model.thr)
    rows = panel[panel["date"] == pd.Timestamp(target)].copy()
    pos = corners(rows["unit"].tolist(), model.cfg.islands)
    return add_keys(rows.reset_index(drop=True), model.cfg, pos)


def predict(model: Model, target: Date) -> pd.DataFrame:
    rows = candidates(model, target)
    if rows.empty:
        return rows
    f = model.fit
    day_logit, parts = day_contrib(f, target, model.hall)
    uc = unit_contrib(f, rows)
    rows["score"] = _sigmoid(day_logit + f.unit_beta[0] + uc.sum(axis=1).to_numpy())
    for c in UNIT_FACTORS:
        rows["c_" + c] = uc[c].to_numpy()
    rows["pts"] = uc.sum(axis=1).to_numpy() * PTS
    model.day = {"logit": day_logit, "parts": parts,
                 "pts": (day_logit - float(_logit(f.base_rate))) * PTS,
                 "rate": float(_sigmoid(day_logit))}
    rows = rows.sort_values(["score", "unit"], ascending=[False, True]).reset_index(drop=True)
    rows["rank"] = np.arange(1, len(rows) + 1)
    return rows


# ---------------------------------------------------------------- 査定表（各条件の点数と、その根拠）
def evidence(df: pd.DataFrame, f: Fit, cfg: Settings) -> dict:
    tr = df[df["w"] > 0]
    out: dict = {}
    if tr.empty:
        return out
    t = _day_table(tr, cfg)
    exp = tr["date"].map(t.set_index("date")["ybar"]).to_numpy(float)
    tr = tr.assign(wy=tr["w"] * tr["y"], we=tr["w"] * exp)
    beta = dict(zip(f.unit_cols, f.unit_beta[1:]))
    for fac in UNIT_FACTORS:
        g = tr[tr["k_" + fac].astype(str) != ""].groupby(tr["k_" + fac].astype(str))
        tab = pd.DataFrame({"obs": g["wy"].sum(), "exp": g["we"].sum(), "n": g["w"].sum(),
                            "days": g["date"].nunique()}).reset_index(names="level")
        tab["ratio"] = tab["obs"] / tab["exp"].clip(lower=1e-9)
        tab["pts"] = [beta.get((fac, lv), 0.0) * PTS for lv in tab["level"]]
        tab["label"] = [level_label(fac, lv) for lv in tab["level"]]
        out[fac] = tab
    dbeta = dict(zip(f.day_cols, f.day_beta[1:]))
    base = f.base_rate
    for fac in DAY_FACTORS:
        tt = t.assign(lv=t["k_" + fac].astype(str), wyb=t["W"] * t["ybar"])
        g = tt.groupby("lv")
        tab = pd.DataFrame({"rate": g["wyb"].sum() / g["W"].sum(), "days": g["date"].nunique()}).reset_index(names="level")
        tab["ratio"] = tab["rate"] / base if base > 0 else np.nan
        tab["pts"] = [dbeta.get((fac, lv), 0.0) * PTS for lv in tab["level"]]
        tab["label"] = [level_label(fac, lv) for lv in tab["level"]]
        out["day_" + fac] = tab
    out["hall_coef"] = {k: dbeta.get(("hall", k), 0.0) for k in ("digit", "wday")}
    return out


def hall_summary(hall: pd.DataFrame) -> dict:
    out: dict = {}
    if hall is None or len(hall) == 0:
        return out
    h = hall.dropna(subset=["avg_diff"]).copy()
    if h.empty:
        return out
    h["date"] = pd.to_datetime(h["date"])
    h["digit"] = h["date"].dt.day % 10
    h["wday"] = h["date"].dt.weekday
    out["hall_digit"] = h.groupby("digit")["avg_diff"].agg(["mean", "count"]).reset_index()
    out["hall_wday"] = h.groupby("wday")["avg_diff"].agg(["mean", "count"]).reset_index()
    out["hall_mean"] = float(h["avg_diff"].mean())
    out["hall_span"] = (h["date"].min().date(), h["date"].max().date(), len(h))
    return out


def reasons(row: pd.Series, model: Model, min_pts: float = 0.5) -> list[tuple[str, float, str]]:
    """台ごとの査定: (表示, 点, 詳細) を点の大きい順（＋も－も）。"""
    items = []
    ev = model.evidence
    for fac, (name, _) in UNIT_FACTORS.items():
        pts = float(row.get("c_" + fac, 0.0)) * PTS
        if abs(pts) < min_pts:
            continue
        lv = str(row.get("k_" + fac, ""))
        text = "この台の実績" if fac == "unit" else level_label(fac, lv)
        if fac == "prev" and pd.notna(row.get("prev_diff")):
            text += f"（{int(row['prev_diff']):+,}枚）"
        detail = name
        tab = ev.get(fac)
        if tab is not None and len(tab):
            hit = tab[tab["level"] == lv]
            if len(hit):
                h = hit.iloc[0]
                detail = (f"{name}「{level_label(fac, lv)}」: 同じ日の島平均と比べて高設定率 ×{h['ratio']:.2f}"
                          f"（{int(h['days'])}日分のデータ）")
        items.append((text, pts, detail))
    items.sort(key=lambda t: -abs(t[1]))
    return items


def day_reasons(model: Model, target: Date, min_pts: float = 0.5) -> list[tuple[str, float, str]]:
    parts = model.day.get("parts", {})
    k = {"digit": str(target.day % 10), "wday": str(target.weekday())}
    sp = day_keys(pd.Series([pd.Timestamp(target)]))["k_special"].iloc[0]
    k["special"] = sp
    out = []
    for fac, (name, _) in DAY_FACTORS.items():
        pts = parts.get(fac, 0.0) * PTS
        if abs(pts) >= min_pts:
            out.append((level_label(fac, k[fac]), pts, f"{name}（ハナハナの過去データ）"))
    hp = parts.get("hall", 0.0) * PTS
    if abs(hp) >= min_pts:
        out.append(("店全体の傾向", hp, "店全体（全機種）の平均差枚から見た、日付末尾・曜日の強さ"))
    out.sort(key=lambda t: -abs(t[1]))
    return out


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
