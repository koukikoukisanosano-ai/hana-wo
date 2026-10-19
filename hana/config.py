from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def load(path: Path | None = None) -> dict:
    p = path or ROOT / "config.yaml"
    with open(p, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    cfg.setdefault("hall", {})
    cfg.setdefault("target", {})
    cfg.setdefault("fetch", {})
    cfg.setdefault("predict", {})
    return cfg


def model_settings(cfg: dict):
    from .model import Settings
    t, p, h = cfg["target"], cfg["predict"], cfg["hall"]
    islands = t.get("islands") or None
    return Settings(
        keyword=t.get("keyword", "ハナハナ"),
        high_min=int(t.get("high_min", 5)),
        reg_weight=float(t.get("reg_weight", 1.5)),
        judge_min_games=int(t.get("judge_min_games", 2000)),
        event_digits=tuple(h.get("event_digits", [0, 5])),
        prev_max_gap=int(p.get("prev_max_gap", 1)),
        islands=[list(map(int, x)) for x in islands] if islands else None,
    )
