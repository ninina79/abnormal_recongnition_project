"""
生成张拉异常冒烟测试用 CSV（仅 time, force_left, force_right, dis_left, dis_right）。

用法（项目根目录）:
    python tools/generate_anomaly_smoke_test_zhangla.py

输出:
    data/anomaly_smoke_test_zhangla.csv

测试说明:
  - 实时页「目标力值」请设为 1000 kN（与脚本 TARGET 一致）。
  - 在 config.py 中设置 SIMULATION_CSV_FILENAME = "data/anomaly_smoke_test_zhangla.csv"
    或使用绝对路径指向生成的文件。
  - 本数据为工程化捏造曲线，用于触发尽可能多的规则项；不保证与真实千斤顶曲线一致。
"""
from __future__ import annotations

import csv
import os
from datetime import datetime, timedelta

TARGET = 1000.0
DT = 1.0
BASE = datetime(2023, 11, 18, 8, 18, 0)
OUT = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "data", "anomaly_smoke_test_zhangla.csv")
)


def row(t_sec: float, fl: float, fr: float, dl: float, dr: float) -> dict:
    ts = BASE + timedelta(seconds=float(t_sec))
    return {
        "time": ts.strftime("%Y-%m-%d %H:%M:%S"),
        "force_left": round(fl, 2),
        "force_right": round(fr, 2),
        "dis_left": round(dl, 3),
        "dis_right": round(dr, 3),
    }


def avg(r: dict) -> float:
    return (float(r["force_left"]) + float(r["force_right"])) / 2.0


def main() -> None:
    rows: list[dict] = []
    t = 0.0

    # ---------- 0) 起步：位移很小、力较快上升 → 刚度比偏高（易触发 high_stiffness）----------
    fl, fr = 40.0, 42.0
    dl, dr = 0.12, 0.10
    for _ in range(6):
        fl += 95.0
        fr += 95.0
        dl += 0.02
        dr += 0.015
        rows.append(row(t, fl, fr, dl, dr))
        t += DT

    # ---------- 1) 平滑加载到 ~32% 目标（越过 INITIAL 20% 后突降规则才生效）----------
    a0 = avg(rows[-1])
    for i in range(22):
        p = (i + 1) / 22.0
        a = a0 + (320.0 - a0) * p
        fl, fr = a - 8.0, a + 8.0
        dl += 0.28
        dr += 0.26
        rows.append(row(t, fl, fr, dl, dr))
        t += DT

    # ---------- 2) 极快升载 → speed_too_fast（近 8 s 平均速率显著高于上限）----------
    cur = avg(rows[-1])
    dl, dr = rows[-1]["dis_left"], rows[-1]["dis_right"]
    for _ in range(10):
        cur = min(cur + 48.0, 920.0)
        fl, fr = cur - 10.0, cur + 10.0
        dl += 0.32
        dr += 0.30
        rows.append(row(t, fl, fr, dl, dr))
        t += DT

    # ---------- 3) 极慢爬升 → speed_too_slow ----------
    cur = avg(rows[-1])
    for _ in range(28):
        cur = min(cur + 0.32, 755.0)
        fl, fr = cur - 6.0, cur + 6.0
        dl += 0.02
        dr += 0.02
        rows.append(row(t, fl, fr, dl, dr))
        t += DT

    # ---------- 4) 力缓升、位移快增 → 刚度比偏低（low_stiffness）----------
    cur = avg(rows[-1])
    for _ in range(10):
        cur = min(cur + 5.0, 780.0)
        dl += 2.1
        dr += 2.0
        fl, fr = cur - 5.0, cur + 5.0
        rows.append(row(t, fl, fr, dl, dr))
        t += DT

    # ---------- 5) 力较快升、位移慢增（相对上一段刚度升高，辅助速度突变历史）----------
    cur = avg(rows[-1])
    for _ in range(8):
        cur = min(cur + 22.0, 900.0)
        dl += 0.06
        dr += 0.05
        fl, fr = cur - 5.0, cur + 5.0
        rows.append(row(t, fl, fr, dl, dr))
        t += DT

    # ---------- 6) 加载至 ≥98.5% 目标，再单步大突降（force_drop_alarm）----------
    cur = avg(rows[-1])
    dl, dr = rows[-1]["dis_left"], rows[-1]["dis_right"]
    while cur < 0.985 * TARGET:
        cur += 11.0
        dl += 0.10
        dr += 0.09
        fl, fr = cur - 6.0, cur + 6.0
        rows.append(row(t, fl, fr, dl, dr))
        t += DT
    cur -= 115.0  # >10% 目标
    fl, fr = cur - 5.0, cur + 5.0
    rows.append(row(t, fl, fr, dl, dr))
    t += DT
    for _ in range(8):
        cur = min(cur + 18.0, 992.0)
        dl += 0.07
        dr += 0.07
        fl, fr = cur - 6.0, cur + 6.0
        rows.append(row(t, fl, fr, dl, dr))
        t += DT

    # ---------- 7) 8%~10% 单步突降 → force_drop_warning ----------
    cur = avg(rows[-1])
    dl, dr = rows[-1]["dis_left"], rows[-1]["dis_right"]
    cur -= 86.0
    fl, fr = cur - 4.0, cur + 4.0
    rows.append(row(t, fl, fr, dl, dr))
    t += DT
    for _ in range(6):
        cur = min(cur + 22.0, 1004.0)
        dl += 0.06
        dr += 0.06
        fl, fr = cur - 5.0, cur + 5.0
        rows.append(row(t, fl, fr, dl, dr))
        t += DT

    # ---------- 8) 加载阶段超张（>105%）----------
    rows.append(row(t, 1065.0, 1065.0, dl + 0.04, dr + 0.04))
    t += DT
    rows.append(row(t, 1000.0, 1000.0, dl + 0.05, dr + 0.05))
    t += DT
    dl, dr = rows[-1]["dis_left"], rows[-1]["dis_right"]

    # ---------- 9) 连续 ≥3 点达到持荷阈值 ----------
    for i in range(5):
        cur = 1002.0 + (i % 3) * 0.4
        dl += 0.04
        dr += 0.04
        fl, fr = cur - 6.0, cur + 6.0
        rows.append(row(t, fl, fr, dl, dr))
        t += DT

    # ---------- 10) 持荷 ≥95 s（满足 min_holding_time_before_unloading 100s 接近，后续再加）----------
    t_hold0 = t
    dl0, dr0 = dl, dr
    while t - t_hold0 < 95.0:
        cur = 1000.0 + (t - t_hold0) * 0.008
        dl = dl0 + (t - t_hold0) * 0.010
        dr = dr0 + (t - t_hold0) * 0.009
        fl, fr = cur - 4.0, cur + 4.0
        rows.append(row(t, fl, fr, dl, dr))
        t += DT

    # ---------- 11) 左右力差大 → force_diff_excessive / force_imbalance ----------
    for i in range(12):
        fl = 1140.0 - i * 2.5
        fr = 860.0 + i * 2.5
        dl += 0.04
        dr += 0.04
        rows.append(row(t, fl, fr, dl, dr))
        t += DT

    # ---------- 12) 恢复平衡 ----------
    for _ in range(42):
        cur = 1000.0
        fl, fr = cur - 5.0, cur + 5.0
        dl += 0.03
        dr += 0.03
        rows.append(row(t, fl, fr, dl, dr))
        t += DT

    # ---------- 13) 持荷力相对参考缓慢下降 → holding_force_drop ----------
    for i in range(16):
        cur = 1000.0 - 8.5 * i
        fl, fr = cur - 3.0, cur + 3.0
        dl += 0.02
        dr += 0.02
        rows.append(row(t, fl, fr, dl, dr))
        t += DT

    # ---------- 14) 拉回至目标附近 ----------
    for _ in range(28):
        cur = min(avg(rows[-1]) + 10.0, 1000.0)
        fl, fr = cur - 5.0, cur + 5.0
        dl += 0.03
        dr += 0.03
        rows.append(row(t, fl, fr, dl, dr))
        t += DT

    # ---------- 15) 两端力变化率差大 → side_speed_unsync ----------
    for i in range(10):
        fl = 1000.0 + (i + 1) * 58.0
        fr = 1000.0 - (i + 1) * 50.0
        dl += 0.03
        dr += 0.02
        rows.append(row(t, fl, fr, dl, dr))
        t += DT

    # ---------- 16) 再稳定一段（补足持荷总时长）----------
    for _ in range(30):
        cur = 1000.0
        fl, fr = cur - 4.0, cur + 4.0
        dl += 0.02
        dr += 0.02
        rows.append(row(t, fl, fr, dl, dr))
        t += DT

    # ---------- 17) 持荷单步大突降 + 数秒平台（断丝/滑丝观察窗）----------
    cur = avg(rows[-1])
    dl, dr = rows[-1]["dis_left"], rows[-1]["dis_right"]
    cur -= 96.0
    fl, fr = cur - 2.0, cur + 2.0
    rows.append(row(t, fl, fr, dl, dr))
    t += DT
    for _ in range(5):
        rows.append(row(t, fl, fr, dl, dr))
        t += DT

    # ---------- 18) 欠张（持荷满 20s 后由规则判 under_tension）----------
    for i in range(40):
        cur = 900.0 - i * 1.2
        fl, fr = cur - 6.0, cur + 6.0
        dl += 0.02
        dr += 0.02
        rows.append(row(t, fl, fr, dl, dr))
        t += DT

    # ---------- 19) 单端超张 ----------
    for _ in range(5):
        rows.append(row(t, 1095.0, 955.0, dl, dr))
        dl += 0.02
        dr += 0.02
        t += DT

    # ---------- 20) 进入卸载（<90% 目标，且持荷已超过 100 s）----------
    cur = avg(rows[-1])
    dl, dr = rows[-1]["dis_left"], rows[-1]["dis_right"]
    while cur > 0.87 * TARGET:
        cur -= 38.0
        dl -= 0.05
        dr -= 0.05
        fl, fr = cur - 4.0, cur + 4.0
        rows.append(row(t, fl, fr, max(0.0, dl), max(0.0, dr)))
        t += DT

    # ---------- 21) 卸载过快（force_rate < UNLOADING_RATE_MIN）----------
    while cur > 100.0:
        cur -= 92.0
        dl -= 0.07
        dr -= 0.07
        fl, fr = cur - 2.0, cur + 2.0
        rows.append(row(t, fl, fr, max(0.0, dl), max(0.0, dr)))
        t += DT

    while cur > 2.0:
        cur = max(cur - 20.0, 0.0)
        fl = fr = cur
        dl = max(0.0, dl - 0.08)
        dr = max(0.0, dr - 0.08)
        rows.append(row(t, fl, fr, dl, dr))
        t += DT

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fieldnames = ["time", "force_left", "force_right", "dis_left", "dis_right"]
    with open(OUT, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print(f"Wrote {len(rows)} rows -> {OUT}")
    print("目标力值请用 1000 kN；SIMULATION_CSV_FILENAME 指向本文件。")


if __name__ == "__main__":
    main()
