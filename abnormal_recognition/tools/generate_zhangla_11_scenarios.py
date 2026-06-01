"""
生成 11 条张拉测试 CSV（每条对应一类规则冒烟场景）。
列：time, force_left, force_right, dis_left, dis_right
目标力请与前端/配置一致使用 1000 kN（RuleEngine(target_force=1000)）。

输出目录：data/test_scenarios/
运行：python tools/generate_zhangla_11_scenarios.py
可选：python tools/generate_zhangla_11_scenarios.py --verify  （用 OnlineFeatureBuilder + RuleEngine 回放首条告警类型）
"""
from __future__ import annotations

import argparse
import csv
import os
import random
import sys
from datetime import datetime, timedelta

# 保证可从仓库根目录运行
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from feature_engine import OnlineFeatureBuilder
from rule_engine import RuleEngine

DATA_DIR = os.path.join(_REPO_ROOT, "data", "test_scenarios")
TARGET = 1000.0
T0 = datetime(2026, 5, 14, 8, 0, 0)


def _t(i: int) -> str:
    return (T0 + timedelta(seconds=int(i))).strftime("%Y-%m-%d %H:%M:%S")


def _write(name: str, rows: list[tuple[float, float, float, float, float]]) -> str:
    os.makedirs(DATA_DIR, exist_ok=True)
    path = os.path.join(DATA_DIR, name)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["time", "force_left", "force_right", "dis_left", "dis_right"])
        for i, (fl, fr, dl, dr) in enumerate(rows):
            w.writerow([_t(i), f"{fl:.3f}", f"{fr:.3f}", f"{dl:.4f}", f"{dr:.4f}"])
    return path


def scenario_01_speed_too_fast() -> list[tuple[float, float, float, float, float]]:
    """近窗平均张拉速度持续过快：>18.2 kN/s 连续 3 点。"""
    rows: list[tuple[float, float, float, float, float]] = []
    for i in range(0, 25):
        f = min(220.0, 9.0 * i)
        d = 0.02 * f
        rows.append((f, f, d, d))
    base = 220.0
    for j in range(25, 55):
        f = base + 22.0 * (j - 24)
        d = 0.02 * f
        rows.append((f, f, d, d))
    return rows


def scenario_02_speed_too_slow() -> list[tuple[float, float, float, float, float]]:
    """近窗平均张拉速度持续过慢：自始保持极低斜率，避免先出现「突然变慢」。"""
    rows = []
    for j in range(0, 100):
        f = 220.0 + 0.22 * j
        d = 0.02 * f
        rows.append((f, f, d, d))
    return rows


def scenario_03_speed_sudden_slow() -> list[tuple[float, float, float, float, float]]:
    """
    平均速度相对近期突然变慢：固定随机种子生成的力增量序列（120 点），
    在 8s 滑动窗下可出现「近期高速率」后突降，触发 speed_sudden_slow。
    """
    random.seed(1)
    f = 250.0
    forces: list[float] = []
    for i in range(120):
        if i < 40:
            df = random.uniform(8, 12)
        elif i < 45:
            df = random.uniform(0.5, 1.5)
        else:
            df = random.uniform(7, 11)
        f = min(920.0, f + df)
        forces.append(round(f, 3))
    rows: list[tuple[float, float, float, float, float]] = []
    for fv in forces:
        d = 0.02 * fv
        rows.append((fv, fv, d, d))
    return rows


def scenario_04_speed_sudden_fast() -> list[tuple[float, float, float, float, float]]:
    """平均速度相对近期突然变快：近期约 2 kN/s，窗内速率跃升至 >6 kN/s。"""
    rows = []
    for i in range(0, 45):
        f = min(300.0, 6.8 * i)
        d = 0.02 * f
        rows.append((f, f, d, d))
    f0 = 300.0
    for j in range(45, 53):
        f = f0 + 2.0 * (j - 44)
        d = 0.02 * f
        rows.append((f, f, d, d))
    for j in range(53, 65):
        f = rows[-1][0] + 55.0
        d = 0.02 * f
        rows.append((f, f, d, d))
    return rows


def scenario_05_low_stiffness() -> list[tuple[float, float, float, float, float]]:
    """刚度比过低：loading、total_delta_dis>1、force_avg/total_dis < 5。"""
    rows = []
    for i in range(0, 60):
        f = 380.0 + 0.5 * i
        dl = 40.0 + 0.35 * i
        dr = 40.0 + 0.35 * i
        rows.append((f, f, dl, dr))
    return rows


def scenario_06_high_stiffness() -> list[tuple[float, float, float, float, float]]:
    """刚度比过高：先缓升力且总位移<1，再高持力、缓增位移使刚度>200。"""
    rows = []
    for i in range(0, 45):
        f = 11.0 * i
        dl = dr = 0.11
        rows.append((f, f, dl, dr))
    for j in range(45, 90):
        t = j - 44
        f = 495.0 + 0.85 * t
        dl = 0.11 + 0.024 * t
        dr = 0.11 + 0.024 * t
        rows.append((f, f, dl, dr))
    return rows


def scenario_07_force_drop_loading() -> list[tuple[float, float, float, float, float]]:
    """张拉阶段单步大突降：>=10% 目标力（>=100 kN）。"""
    rows = []
    for i in range(0, 55):
        f = min(850.0, 16.0 * i)
        d = 0.02 * f
        rows.append((f, f, d, d))
    rows.append((750.0, 750.0, 0.02 * 750.0, 0.02 * 750.0))
    for j in range(56, 70):
        f = 750.0 + 5.0 * (j - 55)
        d = 0.02 * f
        rows.append((f, f, d, d))
    return rows


def scenario_08_side_over() -> list[tuple[float, float, float, float, float]]:
    """单端超张：加载末段一侧>1080 kN，平均力未超 5% 超张。"""
    rows = []
    for i in range(0, 55):
        f = min(820.0, 15.2 * i)
        d = 0.02 * f
        rows.append((f, f, d, d))
    rows.append((1090.0, 710.0, 0.02 * 900.0, 0.02 * 900.0))
    for j in range(56, 70):
        f = 900.0 + 3.5 * (j - 55)
        d = 0.02 * f
        rows.append((f + 35.0, f - 35.0, d, d))
    return rows


def scenario_09_force_imbalance() -> list[tuple[float, float, float, float, float]]:
    """持荷左右力差比超阈：进入持荷后 >20s 再拉开差值。"""
    rows = []
    for i in range(0, 70):
        f = min(1000.0, 14.5 * i)
        d = 0.02 * f
        rows.append((f, f, d, d))
    for j in range(70, 155):
        d = 20.0
        rows.append((1000.0, 1000.0, d, d))
    for j in range(155, 180):
        rows.append((1045.0, 955.0, 20.0, 20.0))
    return rows


def scenario_10_holding_slip_stable() -> list[tuple[float, float, float, float, float]]:
    """持荷突降后观察窗内平稳（断丝/滑丝类）。"""
    rows = []
    for i in range(0, 65):
        f = min(1000.0, 15.8 * i)
        d = 0.02 * f
        rows.append((f, f, d, d))
    for j in range(65, 135):
        d = 22.0
        rows.append((1000.0, 1000.0, d, d))
    rows.append((900.0, 900.0, 22.0, 22.0))
    for k in range(8):
        f = 903.0 + float((k % 3) - 1)
        rows.append((f, f, 22.0, 22.0))
    return rows


def scenario_11_normal() -> list[tuple[float, float, float, float, float]]:
    """无异常：平滑加载—持荷—卸载，速度与刚度处于常见区间。"""
    rows = []
    for i in range(0, 100):
        f = min(1000.0, 10.5 * i)
        d = 0.02 * f
        rows.append((f, f, d, d))
    for j in range(100, 420):
        rows.append((1000.0, 1000.0, 20.0, 20.0))
    f = 1000.0
    for j in range(420, 520):
        f = max(50.0, f - 9.0)
        d = 0.02 * f
        rows.append((f, f, d, d))
    return rows


SCENARIOS = [
    ("01_speed_too_fast.csv", scenario_01_speed_too_fast, "speed_too_fast"),
    ("02_speed_too_slow.csv", scenario_02_speed_too_slow, "speed_too_slow"),
    ("03_speed_sudden_slow.csv", scenario_03_speed_sudden_slow, "speed_sudden_slow"),
    ("04_speed_sudden_fast.csv", scenario_04_speed_sudden_fast, "speed_sudden_fast"),
    ("05_low_stiffness.csv", scenario_05_low_stiffness, "low_stiffness"),
    ("06_high_stiffness.csv", scenario_06_high_stiffness, "high_stiffness"),
    ("07_force_drop_loading.csv", scenario_07_force_drop_loading, "force_drop_alarm"),
    ("08_side_over_tension.csv", scenario_08_side_over, "side_over_tension"),
    ("09_force_imbalance_holding.csv", scenario_09_force_imbalance, "force_imbalance"),
    ("10_holding_drop_slip_like.csv", scenario_10_holding_slip_stable, "force_drop_alarm_slip"),
    ("11_normal.csv", scenario_11_normal, "none"),
]


def _replay_alerts(path: str, target: float = TARGET) -> tuple[list[str], list[tuple[str, str]]]:
    """返回 (每步按顺序的告警类型列表, 所有 (type, message) 列表)。"""
    fe = OnlineFeatureBuilder(window_size=10, target_force=target)
    re = RuleEngine(target_force=target)
    ordered: list[str] = []
    pairs: list[tuple[str, str]] = []
    import csv as _csv

    with open(path, "r", encoding="utf-8-sig") as f:
        r = _csv.DictReader(f)
        for idx, row in enumerate(r):
            raw = {
                "time": float(idx),
                "force_left": float(row["force_left"]),
                "force_right": float(row["force_right"]),
                "dis_left": float(row["dis_left"]),
                "dis_right": float(row["dis_right"]),
            }
            feat = fe.update(raw)
            res = re.check(feat)
            if re.consume_unloading_after_hold_drop():
                fe.force_transition_to_unloading()
            for a in res.get("alerts") or []:
                t = str(a.get("type") or "")
                m = str(a.get("message") or "")
                if t:
                    ordered.append(t)
                    pairs.append((t, m))
    return ordered, pairs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true", help="回放并打印每条首触发的告警类型")
    args = ap.parse_args()

    written = []
    for fname, builder, _ in SCENARIOS:
        rows = builder()
        written.append(_write(fname, rows))

    print(f"已写入 {len(written)} 个文件到: {DATA_DIR}")
    for p in written:
        print(" ", os.path.basename(p))

    if args.verify:
        print("\n--verify：首条告警类型 / 是否满足期望（10 号为持荷平稳突降类 force_drop_alarm 文案）")
        for (fname, _, want), p in zip(SCENARIOS, written):
            ordered, pairs = _replay_alerts(p)
            uniq: list[str] = []
            seen: set[str] = set()
            for t in ordered:
                if t not in seen:
                    seen.add(t)
                    uniq.append(t)
            first = ordered[0] if ordered else "(无)"
            if want == "none":
                ok = len(ordered) == 0
            elif want == "force_drop_alarm_slip":
                ok = any(
                    t == "force_drop_alarm" and "突降后" in m for t, m in pairs
                )
            else:
                ok = want in uniq
            mark = "OK" if ok else "CHECK"
            print(f"  [{mark}] {fname}: first={first} uniq={uniq[:6]} want={want}")


if __name__ == "__main__":
    main()
