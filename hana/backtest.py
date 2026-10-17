"""過去の日で「その日より前のデータだけで予想していたら当たっていたか」を検証する。"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import model as M


def run(units: pd.DataFrame, hall: pd.DataFrame, cfg: M.Settings, days: int = 60, lag: int = 1,
        top_k: int = 3, refit_every: int = 7, min_history: int = 28) -> dict:
    df, prior, thr = M.featurize(units, cfg)
    all_dates = sorted(df["date"].unique())
    if len(all_dates) < min_history + 7:
        return {"ok": False, "reason": f"データが{len(all_dates)}日分しかありません（{min_history + 7}日分以上で検証できます）"}
    first = df["date"].min()
    test_dates = [d for d in all_dates[-days:] if (d - first).days >= min_history]
    fit, last_fit = None, None
    per_day = []
    for d in test_dates:
        if fit is None or (d - last_fit).days >= refit_every:
            fit = M.fit(df[df["date"] <= d - pd.Timedelta(days=lag)], hall, cfg)
            last_fit = d
        dd = df[(df["date"] == d) & (df["games"] >= cfg.label_min_games)].copy()
        if len(dd) < top_k * 2:
            continue
        if lag > 1:   # 前日のデータがまだ無い想定
            for c in M.PREV_DEP:
                dd["k_" + c] = ""
        dd["score"] = M.unit_contrib(fit, dd).sum(axis=1).to_numpy()
        dd = dd.sort_values(["score", "unit"], ascending=[False, True])
        top = dd.head(top_k)
        best = set(dd.sort_values("e_set", ascending=False).head(top_k)["unit"])
        rc = pd.Series(dd["score"].to_numpy()).rank().corr(pd.Series(dd["p_high"].to_numpy()).rank())
        per_day.append({
            "date": d.date(), "n": len(dd),
            "top_p": top["p_high"].mean(), "all_p": dd["p_high"].mean(),
            "top_e": top["e_set"].mean(), "all_e": dd["e_set"].mean(),
            "top_diff": top["diff"].mean(), "all_diff": dd["diff"].mean(),
            "hit": len(best & set(top["unit"])) / top_k, "rand_hit": top_k / len(dd),
            "rank_corr": rc if np.isfinite(rc) else 0.0,
            "picks": ",".join(str(u) for u in top["unit"]),
        })
    if not per_day:
        return {"ok": False, "reason": "検証できる日がありませんでした"}
    t = pd.DataFrame(per_day)
    return {
        "ok": True, "days": len(t), "lag": lag, "top_k": top_k,
        "top_e": t["top_e"].mean(), "all_e": t["all_e"].mean(),
        "top_p": t["top_p"].mean(), "all_p": t["all_p"].mean(),
        "top_diff": t["top_diff"].mean(), "all_diff": t["all_diff"].mean(),
        "hit": t["hit"].mean(), "rand_hit": t["rand_hit"].mean(),
        "rank_corr": t["rank_corr"].mean(),
        "win_days": float((t["top_diff"] > t["all_diff"]).mean()),
        "table": t,
    }
