"""BIG・REG回数とゲーム数から、その日の設定をベイズ推定する。

各設定 s について BIG回数 ~ Poisson(G / BIG分母_s)、REG回数 ~ Poisson(G / REG分母_s)
として尤度を計算し、店全体の設定配分（事前分布）を掛けて事後確率を出す。
店の設定配分は全台・全日のデータからEMアルゴリズムで推定する（＝その店の“使い方”）。
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


def class_llr(loglik: np.ndarray, prior: np.ndarray, high_min: int = HIGH_MIN_DEFAULT) -> np.ndarray:
    """「高設定(high_min以上)」と「それ以外」の対数尤度比。各グループ内の設定の割合は店全体の配分を使う。
    事前確率 π の台なら、事後確率 = sigmoid(logit(π) + 尤度比)。"""
    if len(loglik) == 0:
        return np.zeros(0)
    k = high_min - 1
    lp = loglik + np.log(np.clip(prior, 1e-9, None))
    mx = lp.max(axis=1, keepdims=True)
    hi = np.log(np.exp(lp[:, k:] - mx).sum(axis=1)) - np.log(prior[k:].sum())
    lo = np.log(np.exp(lp[:, :k] - mx).sum(axis=1)) - np.log(prior[:k].sum())
    return hi - lo


def add_estimates(df: pd.DataFrame, prior: np.ndarray | None = None,
                  high_min: int = HIGH_MIN_DEFAULT, min_games: int = 300) -> tuple[pd.DataFrame, np.ndarray]:
    """df に p_high（高設定確率）, e_set（期待設定）, p_set1..6 を追加して返す。"""
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
    df["llr"] = 0.0          # ゲーム数が少ない日は情報なし（尤度比0）
    if len(post):
        df.loc[ok, "e_set"] = post @ settings
        df.loc[ok, "p_high"] = post[:, high_min - 1:].sum(axis=1)
        df.loc[ok, "llr"] = class_llr(ll, prior, high_min)
    return df, prior
