"""癖を仕込んだ架空データで、学習ロジックが癖を見つけられるかを確認する開発用スクリプト。"""
from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hana import backtest, model as M  # noqa: E402
from hana.specs import SPECS  # noqa: E402

SPEC = list(SPECS.values())[0]
UNITS = list(range(668, 688)) + list(range(693, 704))


def simulate(n_days=400, seed=1, habits=True):
    rng = np.random.default_rng(seed)
    start = date(2025, 9, 1)
    rows, hall = [], []
    prev = {u: 2 for u in UNITS}
    low_streak = {u: 0 for u in UNITS}
    for k in range(n_days):
        d = start + timedelta(days=k)
        dig = d.day % 10
        event = dig in (0, 5)
        today = {}
        for u in UNITS:
            s = rng.choice([1, 2, 3, 4], p=[0.45, 0.3, 0.17, 0.08])
            if habits:
                if event and u % 10 == dig and rng.random() < 0.6:
                    s = rng.choice([5, 6])
                if prev[u] == 6 and rng.random() < 0.4:
                    s = 6
                if u == 703 and rng.random() < 0.15:
                    s = 6
                if low_streak[u] >= 2 and rng.random() < 0.25:
                    s = 5
            today[u] = s
        if habits and event:
            for u in rng.choice(UNITS, 2, replace=False):
                today[u] = 6
        tot = []
        for u in UNITS:
            s = today[u]
            g = int(np.clip(rng.normal(4000 + 600 * s, 1800), 300, 10500))
            bb = rng.poisson(g / SPEC.bb[s - 1])
            rb = rng.poisson(g / SPEC.rb[s - 1])
            diff = int(242 * bb + 131 * rb - 1.19 * g + rng.normal(0, 280))
            rows.append({"date": d.isoformat(), "unit": u, "model": SPEC.name, "games": g, "diff": diff,
                         "bb": bb, "rb": rb, "est": 0})
            tot.append(diff)
            low_streak[u] = low_streak[u] + 1 if (diff < -500 and s <= 3) else 0
        prev = today
        hall.append({"date": d.isoformat(), "avg_diff": float(np.mean(tot)) + (150 if event else -20) + rng.normal(0, 60)})
    return pd.DataFrame(rows), pd.DataFrame(hall)


if __name__ == "__main__":
    for habits in (True, False):
        units, hall = simulate(habits=habits)
        cfg = M.Settings()
        m = M.train(units, hall, cfg)
        print(f"\n=== 癖{'あり' if habits else 'なし'} === 学習日数 {m.n_days} 重み調整 {m.tuned}")
        print("係数:", dict(zip(["切片"] + m.cols, np.round(m.beta, 2))))
        s = m.summaries
        print(s["match"][["label", "rate", "days"]].round(3).to_string(index=False))
        print(s["prevhigh"][["label", "rate", "days"]].round(3).to_string(index=False))
        r = backtest.run(units, hall, cfg, days=90, lag=1)
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items() if k != "table"})
        r6 = backtest.run(units, hall, cfg, days=90, lag=6)
        print("lag6:", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r6.items() if k != "table"})
        # 0のつく日の予想例
        p = M.predict(m, date(2026, 10, 10))
        print("10/10の上位:", p.head(5)[["unit", "score"]].round(3).to_dict("records"))
