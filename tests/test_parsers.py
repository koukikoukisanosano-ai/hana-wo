"""取得部分のテスト（実際のページの構造を写したHTMLで確認）。 python tests/test_parsers.py"""
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hana.sources import anaslo, minrepo  # noqa: E402

ANASLO_LIST = """<div class="date-list-area"><div class="date-table" id="table">
<div class="table-row" id="tr"><div class="table-head-cell">日付</div></div>
<div class="table-row" id="tr">
<div class="table-data-cell"><a href="https://ana-slo.com/2026-10-03-%e3%82%aa%e3%83%bc%e3%82%ae%e3%83%a4wo-data/">2026/10/03(土)</a></div>
<div class="table-data-cell">+36,029</div><div class="table-data-cell">+118</div><div class="table-data-cell">4,365</div>
<div class="table-data-cell">42.3%(129/305)</div></div>
<div class="table-row" id="tr">
<div class="table-data-cell"><a href="https://ana-slo.com/2026-10-04-%e3%82%aa%e3%83%bc%e3%82%ae%e3%83%a4wo-data/">2026/10/04(日)</a></div>
<div class="table-data-cell">–</div><div class="table-data-cell">–</div><div class="table-data-cell">4,521</div>
<div class="table-data-cell">–</div></div>
<div class="table-row" id="tr">
<div class="table-data-cell">2021/12/18(土)</div><div class="table-data-cell">–</div><div class="table-data-cell">–</div>
<div class="table-data-cell">4,332</div><div class="table-data-cell">–</div></div>
</div></div>"""

ANASLO_DETAIL = """<html><head><title>2026/10/03 オーギヤWO データまとめ - アナスロ</title></head><body>
<table id="all_data_table" class="fixed_get_medals_table">
<thead><tr><th class="fixed01">機種名</th><th>台番号</th><th>G数</th><th>差枚</th><th>BB</th><th>RB</th><th>ART</th>
<th>合成確率</th><th>BB確率</th><th>RB確率</th><th>ART確率</th></tr></thead><tbody>
<tr><td class="fixed01" style="font-size:14px;line-height:150%;">ハナハナホウオウ～天翔～-30</td><td class="table_cells">668</td>
<td class="table_cells">5,405</td><td class="table_cells">-2,460</td><td class="table_cells">10</td><td class="table_cells">12</td>
<td class="table_cells">0</td><td class="table_cells">1/245.7</td><td class="table_cells">1/540.5</td><td class="table_cells">1/450.4</td>
<td class="table_cells">1/0.0</td></tr>
<tr><td class="fixed01">ハナハナホウオウ～天翔～-30</td><td class="table_cells">669</td><td class="table_cells">7,979</td>
<td class="table_cells">+2,938</td><td class="table_cells">42</td><td class="table_cells">19</td><td class="table_cells">0</td>
<td class="table_cells">1/130.8</td><td class="table_cells">1/190.0</td><td class="table_cells">1/419.9</td><td class="table_cells">1/0.0</td></tr>
<tr><td class="fixed01">スマスロ ハナビ</td><td class="table_cells">719</td><td class="table_cells">4,375</td><td class="table_cells">+406</td>
<td class="table_cells">16</td><td class="table_cells">9</td><td class="table_cells">0</td><td class="table_cells">1/175.0</td>
<td class="table_cells">1/273.4</td><td class="table_cells">1/486.1</td><td class="table_cells">1/0.0</td></tr>
</tbody></table></body></html>"""

MINREPO_TAG = """<table><tbody><tr><th>日付</th><th>総差枚</th><th>平均差枚</th><th>平均G</th><th>機種・末尾</th></tr><tr>
<td><a href="https://min-repo.com/3258624/">7/30(木)</a></td><td>+137,855</td><td>+452</td><td>5,546</td><td>◎Lパチスロ炎炎ノ消防隊2<br></td></tr>
<tr><td><a href="https://min-repo.com/3256933/">7/29(水)</a></td><td>-</td><td>-</td><td>2,671</td><td></td></tr>
<tr><td><a href="https://min-repo.com/1234567/">2025/6/7(土)</a></td><td>+16,034</td><td>+53</td><td>5,113</td><td></td></tr>
</tbody></table>
<a class="page larger" href="https://min-repo.com/tag/wo/page/2/">2</a><a class="last" href="https://min-repo.com/tag/wo/page/4/">最後 »</a>"""

MINREPO_DAY = """<html><head><title>7/30(木) オーギヤWO | スロット差枚データ詳細 – みんレポ</title></head><body>
<div>オーギヤWO 愛知県 2026年7月31日前日</div>
<table class="kishu _2dai"><tbody><tr data-count="0"><th>機種</th><th class="samai_cell">平均差枚</th><th>平均G数</th><th>勝率</th><th>出率</th></tr>
<tr data-count="31"><td><a href="?kishu=%E3%83%8F%E3%83%8A%E3%83%8F%E3%83%8A%E3%83%9B%E3%82%A6%E3%82%AA%E3%82%A6%EF%BD%9E%E5%A4%A9%E7%BF%94%EF%BD%9E-30">ハナハナホウオウ～天翔～-30</a></td>
<td>412</td><td>7,700</td><td>20/31</td><td>103%</td></tr></tbody></table>
<table class="kishu"><tbody><tr><th>機種</th><th>台番</th><th class="samai_cell">差枚</th><th>G数</th><th>出率</th></tr>
<tr><td><a href="?num=719">新ハナビ</a></td><td>719</td><td>274</td><td>5,091</td><td>101.8%</td></tr>
<tr><td><a href="?num=999">キングハナハナ-30</a></td><td>999</td><td>-1,200</td><td>6,000</td><td>95%</td></tr></tbody></table>
</body></html>"""

MINREPO_KISHU = """<div class="tab_content"><h2>ハナハナホウオウ～天翔～-30　グラフ一覧</h2><ul class="slump_list">
<li><a href="https://min-repo.com/3258624/?num=668">668</a><canvas id="graph1"></canvas><table><tbody><tr><th class="samai_cell">差枚</th>
<th>G数</th><th>合成</th></tr><tr><td class="samai_cell">597</td><td>8,475</td><td>1/170</td></tr></tbody></table></li><script>var x=1;</script>
<li><a href="https://min-repo.com/3258624/?num=669">669</a><canvas id="graph2"></canvas><table><tbody><tr><th class="samai_cell">差枚</th>
<th>G数</th><th>合成</th></tr><tr><td class="samai_cell">2,739</td><td>9,462</td><td>1/137</td></tr></tbody></table></li></ul></div>"""

MINREPO_UNIT = """<table class="kishu"><tr><th>機種</th><th>差枚</th><th>G数</th><th>出率</th></tr>
<tr><td>ハナハナホウオウ～天翔～-30</td><td>597</td><td>8,475</td><td>102.3%</td></tr></table>
<table><tr><th>BB</th><th>RB</th><th>合成</th><th>BB率</th><th>RB率</th></tr><tr><td>33</td><td>17</td><td>1/170</td><td>1/257</td><td>1/499</td></tr></table>"""


def test_all():
    days, links = anaslo.parse_list(ANASLO_LIST)
    assert [d["date"] for d in days] == ["2026-10-03", "2026-10-04", "2021-12-18"], days
    assert days[0]["total_diff"] == 36029 and days[0]["avg_diff"] == 118 and days[0]["avg_games"] == 4365
    assert days[1]["total_diff"] is None and days[1]["avg_games"] == 4521
    assert [d for d, _ in links] == ["2026-10-03", "2026-10-04"]

    rows = anaslo.parse_detail(ANASLO_DETAIL, "2026-10-03")
    assert len(rows) == 3
    r = rows[0]
    assert (r["unit"], r["games"], r["diff"], r["bb"], r["rb"]) == (668, 5405, -2460, 10, 12), r
    assert rows[1]["diff"] == 2938 and rows[2]["model"] == "スマスロ ハナビ"

    days, links, last = minrepo.parse_tag_page(MINREPO_TAG, "https://min-repo.com/tag/wo/", date(2026, 10, 11))
    assert links[0] == ("2026-07-30", "https://min-repo.com/3258624/"), links
    assert links[2][0] == "2025-06-07" and last == 4
    assert days[0]["total_diff"] == 137855 and days[1]["avg_diff"] is None

    models, variety = minrepo.parse_day_models(MINREPO_DAY, "ハナハナ")
    assert models == ["ハナハナホウオウ～天翔～-30"], models
    assert variety == [{"model": "キングハナハナ-30", "unit": 999, "diff": -1200, "games": 6000}], variety
    assert minrepo.post_date(MINREPO_DAY, "x") == "2026-07-30"

    units = minrepo.parse_kishu_page(MINREPO_KISHU)
    assert units[0] == {"unit": 668, "diff": 597, "games": 8475, "total": 50}, units
    assert units[1]["total"] == 69

    info = minrepo.parse_unit_page(MINREPO_UNIT)
    assert info["bb"] == 33 and info["rb"] == 17 and info["games"] == 8475

    bb, rb = minrepo.estimate_bb_rb(597, 8475, 50, minrepo.DEFAULT_COEF)
    assert abs(bb - 33) <= 5 and bb + rb == 50, (bb, rb)   # 推定なので誤差あり（重み半分で使用）
    print("OK: すべての解析テストに合格")


if __name__ == "__main__":
    test_all()
