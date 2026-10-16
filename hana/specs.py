"""ハナハナシリーズの設定別スペック。

数値はメーカー公表値（ボーナス確率の分母）。新しいハナハナ機種が店に入ったら
ここに追記すると、その機種も設定推定の対象になります。
"""
from __future__ import annotations

import unicodedata
from dataclasses import dataclass


@dataclass(frozen=True)
class Spec:
    name: str
    bb: tuple[float, ...]       # 設定1..6 のBIG確率の分母
    rb: tuple[float, ...]       # 設定1..6 のREG確率の分母
    payout: tuple[float, ...]   # 設定1..6 の機械割(%)
    verified: bool = True

    @property
    def settings(self) -> tuple[int, ...]:
        return tuple(range(1, len(self.bb) + 1))


# キー: 機種名に含まれる文字列（正規化後）
SPECS: dict[str, Spec] = {
    # ハナハナホウオウ～天翔～-30（2022年導入）公表値
    "ハナハナホウオウ~天翔~": Spec(
        name="ハナハナホウオウ～天翔～-30",
        bb=(297, 284, 273, 262, 249, 236),
        rb=(496, 458, 425, 397, 366, 337),
        payout=(97, 99, 101, 103, 106, 109),
    ),
}

# 未登録のハナハナ機種に使う仮スペック（天翔と同等と仮定）。結果ページで警告表示されます。
GENERIC = Spec(
    name="ハナハナ(スペック未登録)",
    bb=(297, 284, 273, 262, 249, 236),
    rb=(496, 458, 425, 397, 366, 337),
    payout=(97, 99, 101, 103, 106, 109),
    verified=False,
)


def normalize(name: str) -> str:
    s = unicodedata.normalize("NFKC", name or "")
    s = s.replace("〜", "~").replace("～", "~").replace(" ", "").replace("　", "")
    return s


def spec_for(model: str) -> Spec:
    n = normalize(model)
    for key, spec in SPECS.items():
        if normalize(key) in n:
            return spec
    return GENERIC
