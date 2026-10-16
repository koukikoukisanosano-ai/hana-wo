"""過去の日で「その日より前のデータだけで予想していたら当たっていたか」を検証する。"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import model as M


def run(units: pd.DataFrame, hall: pd.DataFrame, cfg: M.Settings, days: int = 60, lag: int = 1,
        top_k: int = 3, refit_every: int = 7) -> dict:
    df, prior = M.prepare(units, cfg)
    df = M.add_prev(df, cfg)
    pos = M.corners(df.loc[df["date"] == df["date"].max(), "unit"].tolist(), cfg.islands)
    df = M.add_keys(df, cfg, pos)
    all_dates = sorted(df["date"].unique())
    if len(all_dates) < cfg.min_train_days + 5:
        return {"ok": False, "reason": f"データが{len(all_dates)}日分しかありません（{cfg.min_train_days + 5}日以上で検証できます）"}

    rows = df.copy()
    rows["cut"] = rows["date"] - pd.Timedelta(days=lag)
    if lag > 1:  # 前日データが無い想定
        rows.loc[:, "prev_p"] = np.nan
    X = M.build_X(rows, df, hall, "cut")
    M.neutralize_prev(X, rows)
    cols = list(M.FACTORS) + list(M.HALL_FACTORS)
    first = df["date"].min() + pd.Timedelta(days=21)

    test_dates = [d for d in all_dates[-days:] if d >= df["date"].min() + pd.Timedelta(days=cfg.min_train_days)]
    beta, last_fit = None, None
    per_day = []
    for d in test_dates:
        if beta is None or (d - last_fit).days >= refit_every:
            train_m = (rows["date"] <= d - pd.Timedelta(days=lag)) & (rows["date"] >= first) & (rows["w"] > 0)
            tr = rows[train_m]
            base = float((tr["w"] * tr["y"]).sum() / max(tr["w"].sum(), 1e-9))
            if train_m.sum() > 200:
                beta = M.fit_combiner(X.loc[train_m, cols].to_numpy(), tr["y"].to_numpy(), tr["w"].to_numpy(),
                                      cfg.l2, float(M._logit(base)), len(M.HALL_FACTORS))
            else:
                beta = np.r_[M._logit(base), np.ones(len(M.FACTORS)), np.zeros(len(M.HALL_FACTORS))]
            last_fit = d
        day = rows["date"] == d
        dd = rows[day & (rows["games"] >= cfg.label_min_games)].copy()
        if len(dd) < top_k * 2:
            continue
        A = np.hstack([np.ones((len(dd), 1)), X.loc[dd.index, cols].to_numpy()])
        dd["score"] = 1 / (1 + np.exp(-(A @ beta)))
        dd = dd.sort_values("score", ascending=False)
        top = dd.head(top_k)
        best = set(dd.sort_values("e_set", ascending=False).head(top_k)["unit"])
        per_day.append({
            "date": d.date(),
            "n": len(dd),
            "top_p": top["p_high"].mean(), "all_p": dd["p_high"].mean(),
            "top_e": top["e_set"].mean(), "all_e": dd["e_set"].mean(),
            "top_diff": top["diff"].mean(), "all_diff": dd["diff"].mean(),
            "hit": len(best & set(top["unit"])) / top_k,
            "rand_hit": top_k / len(dd),
            "picks": ",".join(str(u) for u in top["unit"]),
        })
    if not per_day:
        return {"ok": False, "reason": "検証できる日がありませんでした"}
    t = pd.DataFrame(per_day)
    # 順位相関（予想スコア vs 推定設定）の平均
    return {
        "ok": True, "days": len(t), "lag": lag, "top_k": top_k,
        "top_e": t["top_e"].mean(), "all_e": t["all_e"].mean(),
        "top_p": t["top_p"].mean(), "all_p": t["all_p"].mean(),
        "top_diff": t["top_diff"].mean(), "all_diff": t["all_diff"].mean(),
        "hit": t["hit"].mean(), "rand_hit": t["rand_hit"].mean(),
        "win_days": float((t["top_diff"] > t["all_diff"]).mean()),
        "table": t,
    }
