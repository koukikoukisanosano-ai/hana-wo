"""BIG・REG回数とゲーム数から、その日の設定をベイズ推定する。

各設定 s について BIG回数 ~ Poisson(G / BIG分母_s)、REG回数 ~ Poisson(G / REG分母_s)
として尤度を計算し、店全体の設定配分（事前分布）を掛けて事後確率を出す。
REGは設定差が大きいので、REG側の尤度を reg_weight 倍（既定1.5）して重く見る。

「判定設定」= その日のBIG・REG（合算）だけで一番近い設定（店の傾向は混ぜない）。台ごとに毎日記録する。
店の設定配分は全台・全日のデータからEMアルゴリズムで推定する（＝その店の“使い方”）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .specs import spec_for

N_SET = 6
HIGH_MIN_DEFAULT = 5


def loglik_matrix(df: pd.DataFrame, reg_weight: float = 1.0) -> np.ndarray:
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
        out[i] = lb + reg_weight * lr
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
                  reg_weight: float = 1.5, judge_min_games: int = 2000) -> tuple[pd.DataFrame, np.ndarray]:
    """df に p_high（高設定確率）, e_set（期待設定）, p_set1..6, judge（判定設定）, gassan, reg_den を追加して返す。"""
    df = df.copy()
    ok = df["games"].fillna(0) >= min_games
    ll = loglik_matrix(df[ok], reg_weight)
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
        lj = loglik_matrix(df[jok], reg_weight)
        df.loc[jok, "judge"] = lj.argmax(axis=1) + 1
    return df, prior
