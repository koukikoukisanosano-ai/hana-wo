"""予想結果を1枚のHTMLにする（output/latest.html と output/日付.html）。"""
from __future__ import annotations

import html
from datetime import date as Date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from . import model as M

E = html.escape

CSS = """
:root{--bg:#f6f4ef;--card:#fff;--ink:#1d1b18;--sub:#6b665d;--line:#e4dfd4;--accent:#c8324b;--accent2:#e98a2b;
--good:#1f8a5b;--bad:#9a948a;--heat0:#f1ede4;--heat1:#f9d7c4;--heat2:#f4a98a;--heat3:#e66a5a;--heat4:#c8324b;--warn:#fff4dc;--warnline:#e9c46a}
@media (prefers-color-scheme:dark){:root{--bg:#141311;--card:#1e1c19;--ink:#efebe3;--sub:#a59f94;--line:#34312b;--accent:#ff6b81;
--accent2:#ffad5c;--good:#4cc28c;--bad:#7d776d;--heat0:#26231f;--heat1:#4a2f27;--heat2:#7a3a2c;--heat3:#b04838;--heat4:#e2566a;--warn:#2b2516;--warnline:#8a6d2a}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);
font-family:"Hiragino Sans","Yu Gothic UI","Meiryo",system-ui,sans-serif;font-size:15px;line-height:1.55}
.wrap{max-width:1180px;margin:0 auto;padding:22px 18px 60px}
header{display:flex;flex-wrap:wrap;align-items:flex-end;gap:10px 22px;border-bottom:3px solid var(--ink);padding-bottom:12px}
h1{font-size:26px;margin:0;letter-spacing:.02em}h1 small{font-size:14px;color:var(--sub);font-weight:500;margin-left:8px}
.date{font-size:20px;font-weight:700}.tag{display:inline-block;background:var(--accent);color:#fff;border-radius:3px;padding:1px 8px;font-size:13px;margin-left:6px}
.status{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:10px;margin:16px 0}
.box{background:var(--card);border:1px solid var(--line);border-radius:6px;padding:10px 14px}
.box b{display:block;font-size:22px}.box span{color:var(--sub);font-size:13px}
.warn{background:var(--warn);border:1px solid var(--warnline);border-radius:6px;padding:10px 14px;margin:10px 0;font-size:14px}
h2{font-size:18px;margin:30px 0 10px;padding-left:10px;border-left:5px solid var(--accent)}
table{width:100%;border-collapse:collapse;background:var(--card);border:1px solid var(--line)}
th,td{padding:7px 9px;border-bottom:1px solid var(--line);text-align:left;vertical-align:middle}
th{font-size:12px;color:var(--sub);font-weight:600;background:var(--bg);position:sticky;top:0}
td.num{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}
tr.top td{background:color-mix(in srgb,var(--accent) 7%,var(--card))}
.rank{font-weight:800;font-size:18px;width:42px;text-align:center}.unit{font-weight:800;font-size:20px;letter-spacing:.02em}
.bar{position:relative;height:10px;background:var(--heat0);border-radius:5px;min-width:110px;overflow:hidden}
.bar i{position:absolute;left:0;top:0;bottom:0;background:linear-gradient(90deg,var(--accent2),var(--accent));border-radius:5px}
.pct{font-weight:700;font-size:16px}.ratio{color:var(--sub);font-size:12px}
.chip{display:inline-block;border-radius:3px;padding:0 6px;margin:2px 4px 2px 0;font-size:12px;border:1px solid var(--line)}
.chip.p{border-color:var(--good);color:var(--good)}.chip.m{color:var(--bad)}
.prev{font-size:12px;color:var(--sub);white-space:nowrap}.prev b{color:var(--ink);font-weight:600}
.map{display:flex;flex-direction:column;gap:12px}.island{display:flex;flex-wrap:wrap;gap:4px}
.cell{width:56px;height:56px;border-radius:5px;display:flex;flex-direction:column;align-items:center;justify-content:center;
font-weight:700;border:1px solid var(--line);font-size:15px}.cell small{font-size:11px;font-weight:500}
.grid2{display:grid;grid-template-columns:repeat(auto-fit,minmax(330px,1fr));gap:14px}
.mini td,.mini th{padding:5px 8px;font-size:13px}.hi{color:var(--accent);font-weight:700}
.note{color:var(--sub);font-size:13px}footer{margin-top:36px;color:var(--sub);font-size:12px;border-top:1px solid var(--line);padding-top:12px}
svg.spark{display:block}
.scroll{overflow-x:auto;-webkit-overflow-scrolling:touch}
@media (max-width:700px){
.wrap{padding:14px 12px 40px}
header{gap:6px 12px}h1{font-size:19px}h1 small{display:block;margin:2px 0 0}.date{font-size:17px}
.status{grid-template-columns:repeat(2,1fr);gap:8px}.box{padding:8px 10px}.box b{font-size:19px}.box span{font-size:12px}
h2{font-size:17px;margin:24px 0 8px}
table.rk,table.rk tbody{display:block;border:none;background:none}
table.rk tr.hd{display:none}
table.rk tr{display:grid;grid-template-columns:30px 62px 1fr;gap:4px 8px;align-items:center;background:var(--card);
border:1px solid var(--line);border-radius:8px;padding:10px 10px;margin-bottom:8px}
table.rk tr.top{border-color:color-mix(in srgb,var(--accent) 45%,var(--line))}
table.rk td{display:block;border:none;padding:0;background:none!important}
table.rk td.rank{font-size:16px;width:auto}table.rk td.unit{font-size:22px}
table.rk td.reasons,table.rk td.prevc{grid-column:1/-1}
table.rk td.sparkc{display:none}
.bar{min-width:0}
.cell{width:46px;height:46px;font-size:13px}
.grid2{grid-template-columns:1fr}
.mini td,.mini th{padding:4px 6px;font-size:12px}
}
"""


def heat_class(p: float, base: float) -> str:
    r = p / base if base > 0 else 1
    return "var(--heat4)" if r >= 1.8 else "var(--heat3)" if r >= 1.4 else "var(--heat2)" if r >= 1.15 \
        else "var(--heat1)" if r >= 0.95 else "var(--heat0)"


def spark(vals: list[float]) -> str:
    if not vals:
        return ""
    w, h = 84, 22
    xs = np.linspace(2, w - 2, len(vals)) if len(vals) > 1 else np.array([w / 2])
    ys = [h - 2 - (min(max(v, 1), 6) - 1) / 5 * (h - 4) for v in vals]
    pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys))
    mid = h - 2 - (3.5 - 1) / 5 * (h - 4)
    dots = "".join(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="1.8" fill="{"var(--accent)" if v >= 4.2 else "var(--sub)"}"/>'
                   for x, y, v in zip(xs, ys, vals))
    return (f'<svg class="spark" width="{w}" height="{h}" viewBox="0 0 {w} {h}" aria-label="直近の推定設定">'
            f'<line x1="0" x2="{w}" y1="{mid:.1f}" y2="{mid:.1f}" stroke="var(--line)" stroke-dasharray="2 2"/>'
            f'<polyline points="{pts}" fill="none" stroke="var(--sub)" stroke-width="1"/>{dots}</svg>')


def fmt_int(v, sign=False) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "-"
    return f"{int(v):+,}" if sign else f"{int(v):,}"


def habit_table(t: pd.DataFrame, base: float, title: str, extra: pd.DataFrame | None = None, key=None) -> str:
    if t is None or len(t) == 0:
        return ""
    rows = []
    for _, r in t.iterrows():
        ratio = r["rate"] / base if base > 0 else 1
        cls = ' class="hi"' if ratio >= 1.3 and r["days"] >= 5 else ""
        rows.append(f"<tr><td>{E(str(r['label']))}</td><td class='num'{cls}>{r['rate']*100:.1f}%</td>"
                    f"<td class='num'>×{ratio:.2f}</td><td class='num'>{fmt_int(r['avg_diff'], True)}</td>"
                    f"<td class='num'>{int(r['days'])}日</td></tr>")
    return (f"<div><h3 style='font-size:15px;margin:6px 0'>{E(title)}</h3><table class='mini'><tr><th>条件</th><th>高設定率</th>"
            f"<th>平常比</th><th>平均差枚</th><th>日数</th></tr>{''.join(rows)}</table></div>")


def check_section(hist: pd.DataFrame, pred_dir: Path | None, target: Date, top_k: int = 3, show: int = 7) -> str:
    """過去の予想（data/predictions/*.csv）と、その後に公開された実際のデータの答え合わせ。"""
    if pred_dir is None or not Path(pred_dir).exists() or hist.empty:
        return ""
    rows = []
    for f in sorted(Path(pred_dir).glob("*.csv"), reverse=True):
        try:
            d = pd.Timestamp(f.stem)
        except ValueError:
            continue
        if d.date() >= target:
            continue
        day = hist[hist["date"] == d]
        if day.empty:
            continue
        pr = pd.read_csv(f)
        picks = pr.sort_values("rank").head(top_k)["unit"].astype(int).tolist()
        act = day.set_index("unit")
        got = [u for u in picks if u in act.index]
        if not got:
            continue
        sub = act.loc[got]
        best = set(day.sort_values("e_set", ascending=False).head(top_k)["unit"].astype(int))
        rows.append({
            "date": d.date(), "picks": got, "act": sub,
            "top_diff": float(sub["diff"].mean()), "all_diff": float(day["diff"].mean()),
            "top_e": float(sub["e_set"].mean()), "all_e": float(day["e_set"].mean()),
            "hit": len(best & set(got)),
        })
    if not rows:
        return ("<h2>予想の答え合わせ</h2><p class='note'>予想した日のデータが公開されると、ここに予想上位"
                f"{top_k}台の実際の結果が表示されます。</p>")
    n = len(rows)
    win = sum(r["top_diff"] > r["all_diff"] for r in rows)
    avg_top = np.mean([r["top_diff"] for r in rows])
    avg_all = np.mean([r["all_diff"] for r in rows])
    hits = sum(r["hit"] for r in rows)
    summary = (f"<div class='status'><div class='box'><b>{win}/{n}日</b><span>予想上位{top_k}台が全台平均を上回った日</span></div>"
               f"<div class='box'><b>{avg_top:+,.0f}枚</b><span>予想上位{top_k}台の平均差枚（全台平均 {avg_all:+,.0f}枚）</span></div>"
               f"<div class='box'><b>{hits}/{n * top_k}台</b><span>その日の推定設定上位{top_k}台に入った台数</span></div></div>")
    trs = []
    for r in rows[:show]:
        cells = " ".join(
            f"<span class='chip {'p' if a['diff'] > 0 else 'm'}'>{u}: {fmt_int(a['diff'], True)}枚 推定{a['e_set']:.1f}</span>"
            for u, a in r["act"].iterrows())
        trs.append(f"<tr><td class='num'>{r['date']:%m/%d}</td><td>{cells}</td>"
                   f"<td class='num'>{r['top_diff']:+,.0f}</td><td class='num'>{r['all_diff']:+,.0f}</td></tr>")
    return (f"<h2>予想の答え合わせ</h2>{summary}<div class='scroll'><table class='mini'><tr><th>日付</th><th>予想上位{top_k}台の実際</th>"
            f"<th>上位平均</th><th>全台平均</th></tr>{''.join(trs)}</table></div>"
            "<p class='note'>この仕組みが毎朝出した予想を、後から公開された実データで採点したものです（推定＝BB・REGから推定した設定）。</p>")


def build(model: M.Model, pred: pd.DataFrame, target: Date, hall_name: str, top_n: int,
          bt: dict | None, cov: dict, out_dir: Path, extra_warns: list[str] | None = None,
          pred_dir: Path | None = None, info_line: str = "") -> Path:
    cfg = model.cfg
    info = M.target_day_info(target, cfg)
    hist = model.hist
    last_date = hist["date"].max().date() if len(hist) else None
    lag = (target - last_date).days if last_date else None
    base = model.base_rate
    n_units = len(pred)
    exp_high = float(pred["score"].sum()) if n_units else 0.0
    avg_high = base * n_units

    tags = "".join(f"<span class='tag'>{E(t)}</span>" for t in info["tags"])
    head = (f"<header><h1>{E(hall_name)} ハナハナ予想<small>高設定(設定{cfg.high_min}以上)期待度</small></h1>"
            f"<div class='date'>{target.year}年{target.month}月{target.day}日({info['wday']}){tags}</div></header>")

    boxes = [
        (f"{exp_high:.1f}台", f"今日の高設定 期待台数（全{n_units}台・平常{avg_high:.1f}台）"),
        (f"{last_date:%m/%d}" if last_date else "-", f"データの最新日（{lag}日前）" if lag is not None else "データなし"),
        (f"{model.n_days}日分", "学習に使ったデータ"),
        ("調整済" if model.tuned else "蓄積中", "要因の重み（45日分以上で自動調整）"),
    ]
    status = "<div class='status'>" + "".join(f"<div class='box'><b>{E(a)}</b><span>{E(b)}</span></div>" for a, b in boxes) + "</div>"
    warns = []
    if lag is not None and lag > cfg.prev_max_gap:
        y = target - pd.Timedelta(days=1)
        warns.append(f"前日（{y.month}/{y.day}）のデータがまだ公開されていないため、"
                     "「前日の凹み・据え置き」系の要因は使わずに予想しています。データサイトの更新は数日遅れることがあります。")
    if model.n_days < 30:
        warns.append(f"まだ{model.n_days}日分しかデータがありません。過去データの取得が進むほど店の癖が反映されます"
                     "（初回の取得は数時間〜1日程度かかります）。")
    from .specs import spec_for
    unver = sorted({m for m in pred.get("model", []) if not spec_for(m).verified})
    if unver:
        warns.append("スペック未登録の機種があります（仮の数値で推定中）: " + "、".join(unver) + "。hana/specs.py に追記してください。")
    warns = list(extra_warns or []) + warns
    warn_html = "".join(f"<div class='warn'>{E(w)}</div>" for w in warns)

    # 直近の推移
    recent = hist[hist["date"] > hist["date"].max() - pd.Timedelta(days=14)] if len(hist) else hist
    by_unit = {u: g.sort_values("date") for u, g in recent.groupby("unit")} if len(recent) else {}
    last_by_unit = hist.sort_values("date").groupby("unit").tail(1).set_index("unit") if len(hist) else pd.DataFrame()

    trs = []
    for _, r in pred.iterrows():
        u = int(r["unit"])
        p = float(r["score"])
        ratio = p / base if base > 0 else 1
        chips = "".join(f"<span class='chip {'p' if v > 0 else 'm'}' title='{E(d)}'>{'＋' if v > 0 else '－'}{E(t)}</span>"
                        for t, v, d in M.reasons(r, model))
        lr = last_by_unit.loc[u] if u in last_by_unit.index else None
        prev = "-"
        if lr is not None:
            prev = (f"<div class='prev'>{lr['date']:%m/%d} <b>{fmt_int(lr['games'])}G</b> "
                    f"BB{int(lr['bb'])}/RB{int(lr['rb'])} <b>{fmt_int(lr['diff'], True)}枚</b> "
                    f"推定{lr['e_set']:.1f}</div>")
        sp = spark(list(by_unit[u]["e_set"].fillna(3).tail(10))) if u in by_unit else ""
        width = min(100, p / max(pred["score"].max(), 1e-9) * 100)
        trs.append(
            f"<tr class='{'top' if r['rank'] <= top_n else ''}'><td class='rank'>{int(r['rank'])}</td>"
            f"<td class='unit'>{u}</td>"
            f"<td class='score'><div class='pct'>{p*100:.1f}%</div><div class='bar'><i style='width:{width:.0f}%'></i></div>"
            f"<div class='ratio'>平常の×{ratio:.2f}</div></td><td class='reasons'>{chips or '<span class=note>特になし</span>'}</td>"
            f"<td class='prevc'>{prev}</td><td class='sparkc'>{sp}</td></tr>")
    table = ("<table class='rk'><tr class='hd'><th>順位</th><th>台番</th><th>高設定期待度</th><th>主な根拠</th><th>前回データ</th>"
             "<th>直近の推定設定</th></tr>" + "".join(trs) + "</table>")

    # 島マップ
    units_sorted = sorted(pred["unit"].astype(int).tolist())
    sc = dict(zip(pred["unit"].astype(int), pred["score"]))
    runs: list[list[int]] = []
    if cfg.islands:
        runs = [[u for u in units_sorted if a <= u <= b] for a, b in cfg.islands]
    else:
        cur: list[int] = []
        for u in units_sorted:
            if cur and u != cur[-1] + 1:
                runs.append(cur)
                cur = []
            cur.append(u)
        if cur:
            runs.append(cur)
    isl = "".join("<div class='island'>" + "".join(
        f"<div class='cell' style='background:{heat_class(sc[u], base)}'>{u}<small>{sc[u]*100:.0f}%</small></div>"
        for u in run) + "</div>" for run in runs if run)
    mapsec = f"<h2>島マップ</h2><div class='map'>{isl}</div><p class='note'>色が濃いほど高設定期待度が高い台。</p>"

    # 店の癖
    s = model.summaries
    habit = ""
    if s:
        parts = [habit_table(s.get("digit"), base, "日付の末尾別（ハナハナ）"),
                 habit_table(s.get("wday"), base, "曜日別（ハナハナ）"),
                 habit_table(s.get("match"), base, "台番末尾と日付末尾の一致"),
                 habit_table(s.get("prevdiff"), base, "前日の差枚 → 当日"),
                 habit_table(s.get("prevhigh"), base, "前日の挙動 → 当日（据え置き傾向）"),
                 habit_table(s.get("corner"), base, "台の位置"),
                 habit_table(s.get("special"), base, "特定日")]
        ut = s.get("unit")
        if ut is not None and len(ut):
            ut = ut.sort_values("rate", ascending=False)
            parts.append(habit_table(pd.concat([ut.head(6), ut.tail(4)]), base, "台番別（入りやすい台・入りにくい台）"))
        hd = s.get("hall_digit")
        if hd is not None and len(hd):
            span = s.get("hall_span")
            rows = "".join(f"<tr><td>{int(r['digit'])}のつく日</td><td class='num'>{fmt_int(r['mean'], True)}</td>"
                           f"<td class='num'>{int(r['count'])}日</td></tr>" for _, r in hd.iterrows())
            parts.append(f"<div><h3 style='font-size:15px;margin:6px 0'>店全体（全機種）の平均差枚・日付末尾別</h3>"
                         f"<table class='mini'><tr><th>日付</th><th>平均差枚/台</th><th>日数</th></tr>{rows}</table>"
                         f"<p class='note'>{span[0]}〜{span[1]}（{span[2]}日）</p></div>")
        habit = (f"<h2>学習した店の癖</h2><p class='note'>高設定率 = BB・REG回数から推定した「設定{cfg.high_min}以上だった確率」の平均"
                 f"（ゲーム数が少ない日は重みを下げて集計）。全体平均は {base*100:.1f}%。日数が少ない条件は偶然の可能性があります。</p>"
                 f"<div class='grid2'>{''.join(p for p in parts if p)}</div>")

    # 検証
    btsec = "<h2>過去データでの検証</h2>"
    if bt and bt.get("ok"):
        btsec += (f"<div class='status'>"
                  f"<div class='box'><b>{bt['hit']*100:.1f}%</b><span>上位{bt['top_k']}台予想が、その日の推定上位{bt['top_k']}台に入った率"
                  f"（ランダムなら{bt['rand_hit']*100:.1f}%）</span></div>"
                  f"<div class='box'><b>{bt['top_diff']:+,.0f}枚</b><span>予想上位{bt['top_k']}台の平均差枚（全台平均 {bt['all_diff']:+,.0f}枚）</span></div>"
                  f"<div class='box'><b>{bt['win_days']*100:.0f}%</b><span>予想上位が全台平均を上回った日の割合</span></div>"
                  f"<div class='box'><b>{bt['days']}日</b><span>検証日数（データ公開の遅れ {bt['lag']}日を想定）</span></div></div>"
                  f"<p class='note'>各日について、その日より前のデータだけで予想し直して答え合わせした結果です（{bt.get('when', '')}時点）。</p>")
    else:
        reason = bt.get("reason") if bt else "まだ実行していません"
        btsec += f"<p class='note'>{E(reason)}</p>"

    cov_txt = (f"データ: {cov.get('unit_first')}〜{cov.get('unit_last')}（{cov.get('unit_days')}日・{cov.get('unit_rows'):,}台日）"
               if cov.get("unit_rows") else "データなし")
    from datetime import timedelta, timezone
    now_jst = datetime.now(timezone(timedelta(hours=9)))
    foot = (f"<footer>{E(cov_txt)}／取得元: アナスロ・みんレポ（公開データ）／作成 {now_jst:%Y-%m-%d %H:%M}<br>"
            + (f"{E(info_line)}<br>" if info_line else "") +
            "この予想は公開データから計算した確率であり、結果を保証するものではありません。遊技は余裕のある範囲でお楽しみください。</footer>")

    doc = (f"<!doctype html><html lang='ja'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
           f"<title>ハナハナ予想 {target:%m/%d}</title><style>{CSS}</style></head><body><div class='wrap'>"
           f"{head}{status}{warn_html}<h2>台別ランキング</h2>{table}{mapsec}{check_section(hist, pred_dir, target)}"
           f"{habit}{btsec}{foot}</div></body></html>")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{target:%Y-%m-%d}.html"
    path.write_text(doc, encoding="utf-8")
    (out_dir / "latest.html").write_text(doc, encoding="utf-8")
    (out_dir / "index.html").write_text(doc, encoding="utf-8")
    cols = ["rank", "unit", "model", "score"]
    pred.assign(reasons=[" / ".join(t for t, _, _ in M.reasons(r, model)) for _, r in pred.iterrows()])[
        cols + ["reasons"]].to_csv(out_dir / f"{target:%Y-%m-%d}.csv", index=False, encoding="utf-8-sig")
    return path
