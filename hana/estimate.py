"""BIG・REG回数とゲーム数から、その日の設定を推定する（合算のみ。差枚は使わない）。

各設定 s について BIG回数 ~ Poisson(G / BIG分母_s)、REG回数 ~ Poisson(G / REG分母_s)
として尤度を計算する（BIGとREGは同じ重み＝合算で見るのと同じ）。

・判定設定: その日の数字だけで一番近い設定（店の傾向は混ぜない）。台ごとに毎日記録する。
・高設定確率 p_high: 店全体の設定配分（EMで推定）を掛けた事後確率。予想の学習に使う。

差枚・プラス終了・期待獲得枚数は、ほとんどがボーナス回数で決まり二重に数えることになるので使わない。
検証（63日分）: 全台・全日の合算 1/167 → 設定2.89、各台の判定設定の平均 2.80（日ごとのずれ平均0.30）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .specs import spec_for

N_SET = 6
HIGH_MIN_DEFAULT = 5


def loglik_matrix(df: pd.DataFrame) -> np.ndarray:
    """行ごと・設定ごとの対数尤度 (n, 6)。定数項は省略。"""
    out = np.zeros((len(df), N_SET))
    games = df["games"].to_numpy(dtype=float)
    bb = df["bb"].to_numpy(dtype=float)
    rb = df["rb"].to_numpy(dtype=float)
    models = df["model"].to_numpy()
    cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for i in range(len(df)):
        m = models[i]
        if m not in cache:
            sp = spec_for(m)
            cache[m] = (1.0 / np.array(sp.bb), 1.0 / np.array(sp.rb))
        pb, pr = cache[m]
        g = max(games[i], 0.0)
        lam_b, lam_r = g * pb, g * pr
        # Poisson log-likelihood (k*log(lam) - lam), lam=0 のとき寄与0
        with np.errstate(divide="ignore", invalid="ignore"):
            lb = np.where(lam_b > 0, bb[i] * np.log(lam_b) - lam_b, 0.0)
            lr = np.where(lam_r > 0, rb[i] * np.log(lam_r) - lam_r, 0.0)
        out[i] = lb + lr
    return out


def posterior(loglik: np.ndarray, prior: np.ndarray) -> np.ndarray:
    lp = loglik + np.log(np.clip(prior, 1e-9, None))
    lp -= lp.max(axis=1, keepdims=True)
    p = np.exp(lp)
    return p / p.sum(axis=1, keepdims=True)


def fit_prior(loglik: np.ndarray, iters: int = 200, smooth: float = 1.0) -> np.ndarray:
    """店全体の設定配分をEMで推定（ディリクレ平滑化つき）。"""
    if len(loglik) == 0:
        return np.full(N_SET, 1.0 / N_SET)
    prior = np.array([0.40, 0.25, 0.15, 0.10, 0.06, 0.04])  # 一般的な初期値
    for _ in range(iters):
        post = posterior(loglik, prior)
        new = post.sum(axis=0) + smooth
        new /= new.sum()
        if np.abs(new - prior).max() < 1e-6:
            prior = new
            break
        prior = new
    return prior


def add_estimates(df: pd.DataFrame, prior: np.ndarray | None = None,
                  high_min: int = HIGH_MIN_DEFAULT, min_games: int = 300,
                  judge_min_games: int = 2000) -> tuple[pd.DataFrame, np.ndarray]:
    """df に p_high（高設定確率）, e_set（期待設定）, p_set1..6, judge（判定設定）, gassan, reg_den を追加して返す。"""
    df = df.copy()
    ok = df["games"].fillna(0) >= min_games
    ll = loglik_matrix(df[ok])
    if prior is None:
        prior = fit_prior(ll)
    post = posterior(ll, prior) if len(ll) else np.zeros((0, N_SET))
    for s in range(N_SET):
        df[f"p_set{s+1}"] = np.nan
        df.loc[ok, f"p_set{s+1}"] = post[:, s] if len(post) else np.nan
    settings = np.arange(1, N_SET + 1)
    df["e_set"] = np.nan
    df["p_high"] = np.nan
    if len(post):
        df.loc[ok, "e_set"] = post @ settings
        df.loc[ok, "p_high"] = post[:, high_min - 1:].sum(axis=1)
    # 判定設定（その日の数字だけ。店の傾向は混ぜない）
    g = df["games"].astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        df["gassan"] = np.where(df["bb"] + df["rb"] > 0, g / (df["bb"] + df["rb"]), np.nan)
        df["reg_den"] = np.where(df["rb"] > 0, g / df["rb"], np.nan)
    df["judge"] = np.nan
    jok = (df["games"].fillna(0) >= judge_min_games).to_numpy()
    if jok.any():
        lj = loglik_matrix(df[jok])
        df.loc[jok, "judge"] = lj.argmax(axis=1) + 1
    return df, prior


def gassan_to_setting(gassan: float, model: str = "") -> float:
    """合算分母 → 設定（スペック表の間を直線でつなぐ。1〜6の範囲に収める）。"""
    sp = spec_for(model)
    g = 1.0 / (1.0 / np.array(sp.bb) + 1.0 / np.array(sp.rb))
    return float(np.interp(-gassan, -g, np.arange(1, N_SET + 1)))
