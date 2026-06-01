# IF_LOF_DATA.py
#模型处理的数据进行分类

import os
import json
import pandas as pd

from rule_engine import RealTimeTensionMonitor
from feature_engine import OnlineFeatureBuilder


# =========================================================
# 1. 路径配置
# =========================================================
DATA_DIR = r"E:\ABNORMAL_RECOGNITION\DATA\processed\zhangla_data"
JSON_DIR = r"E:\ABNORMAL_RECOGNITION\DATA\processed\json"
OUTPUT_BASE_DIR = r"E:\ABNORMAL_RECOGNITION\DATA\processed\if_lof_training_data"

NORMAL_DIR = os.path.join(OUTPUT_BASE_DIR, "1_normal")
WARNING_DIR = os.path.join(OUTPUT_BASE_DIR, "2_warning_recoverable")
ALARM_DIR = os.path.join(OUTPUT_BASE_DIR, "3_alarm")
REPORT_PATH = os.path.join(OUTPUT_BASE_DIR, "classification_report.csv")


# =========================================================
# 2. 硬异常类型（训练时排除）
# =========================================================
HARD_ANOMALY_TYPES = {
    # 超限类
    "over_tension",
    "side_over_tension",

    # 两端严重不同步
    "side_unsync_alarm",

    # 速度严重异常
    "speed_too_fast",
    "speed_sudden_fast",

    # 力值下降严重异常
    "force_drop_alarm",

    # 持荷阶段异常
    "hold_fluctuation",
    "hold_time_short",

    # final_check 硬异常
    "under_tension",
    "no_holding_stage",
    "hold_time_insufficient",
}

# =========================================================
# 3. 软异常类型（训练时保留为 normal）
# =========================================================
SOFT_ANOMALY_TYPES = {
    # 两端轻微不同步
    "side_unsync",

    # 速度轻微异常
    "speed_too_slow",
    "speed_sudden_slow",

    # 力值轻微下降
    "force_drop_warning",

    # final_check 软异常
    "unload_not_finished",
}

# =========================================================
# 4. 窗口配置
# =========================================================
WINDOW_SIZE = 5          # 窗口大小（行数）
WINDOW_STEP = 5          # 窗口步长（不重叠）


# =========================================================
# 5. 工具函数
# =========================================================
def ensure_dirs():
    os.makedirs(NORMAL_DIR, exist_ok=True)
    os.makedirs(WARNING_DIR, exist_ok=True)
    os.makedirs(ALARM_DIR, exist_ok=True)


def get_csv_files(data_dir):
    files = [
        os.path.join(data_dir, f)
        for f in os.listdir(data_dir)
        if f.endswith(".csv")
    ]
    return sorted(files)


def parse_integrated_filename(csv_file_path):
    fname = os.path.basename(csv_file_path)
    name = fname

    if name.startswith("integrated_"):
        name = name[len("integrated_"):]

    if name.endswith(".csv"):
        name = name[:-4]

    segments = name.split("-")

    if len(segments) < 4:
        raise ValueError(f"文件名格式不符合: {fname}")

    return {
        "project_id": segments[0],
        "task_id": segments[1],
        "strand_ids": [segments[2], segments[3]],
        "json_name": f"{segments[0]}-{segments[1]}.json",
    }


def get_target_params_from_json(csv_file_path):
    info = parse_integrated_filename(csv_file_path)
    json_path = os.path.join(JSON_DIR, info["json_name"])

    if not os.path.exists(json_path):
        raise FileNotFoundError(f"找不到 JSON: {json_path}")

    with open(json_path, "r", encoding="utf-8-sig") as f:
        data = json.load(f)

    strand_ids = set(str(x) for x in info["strand_ids"])
    target_values = []
    theory_values = []

    for item in data:
        if str(item.get("id")) in strand_ids:
            t = item.get("targetPulling")
            d = item.get("theoryDis")

            if t is not None:
                target_values.append(float(t))

            if d is not None:
                theory_values.append(float(d))

    if not target_values:
        raise ValueError(f"JSON 中未找到目标力: {info['strand_ids']}")

    return {
        "target_force": sum(target_values) / len(target_values),
        "theory_dis": sum(theory_values) / len(theory_values) if theory_values else 0.0,
    }


# =========================================================
# 6. 窗口标签判定
# =========================================================
def classify_window_label(rule_results_in_window):
    """
    对一个窗口内的所有规则结果进行标签判定。

    宽松标签策略：
    - 出现硬异常 -> alarm
    - 只出现软异常 -> normal（用于训练）
    - 没有异常 -> normal

    同时输出严格标签用于测试：
    - 出现硬异常 -> alarm
    - 出现软异常 -> warning
    - 没有异常 -> normal
    """

    has_hard = False
    has_soft = False

    for result in rule_results_in_window:
        status = result.get("status", "normal")
        rtype = result.get("type", "normal")

        if status in ("warning", "alarm"):
            if rtype in HARD_ANOMALY_TYPES:
                has_hard = True
            elif rtype in SOFT_ANOMALY_TYPES:
                has_soft = True
            else:
                # 未归类的异常类型，保守处理为硬异常
                has_hard = True

    if has_hard:
        train_label = "alarm"
        strict_label = "alarm"
    elif has_soft:
        train_label = "normal"       # 宽松：软异常仍作为 normal 训练
        strict_label = "warning"     # 严格：用于测试时区分
    else:
        train_label = "normal"
        strict_label = "normal"

    return train_label, strict_label


# =========================================================
# 7. 单文件处理
# =========================================================
def process_one_file(csv_file_path):
    file_name = os.path.basename(csv_file_path)

    params = get_target_params_from_json(csv_file_path)
    target_force = params["target_force"]
    theory_dis = params["theory_dis"]

    df = pd.read_csv(csv_file_path, encoding="utf-8-sig")

    if df.empty:
        raise ValueError("CSV 为空")

    # 初始化规则引擎和特征构建器
    engine = RealTimeTensionMonitor(
        target_force=target_force,
        theory_dis=theory_dis,
        strand_id=file_name,
    )

    feature_builder = OnlineFeatureBuilder(
        window_size=5,
        target_force=target_force,
    )

    # =====================================================
    # 逐行计算特征并运行规则引擎
    # =====================================================
    all_features = []
    all_rule_results = []

    for _, row in df.iterrows():
        time_str = str(row["time"]).split(" ")[-1]

        force_left = float(row.get("force_left", 0))
        force_right = float(row.get("force_right", 0))
        dis_left = float(row.get("dis_left", 0))
        dis_right = float(row.get("dis_right", 0))

        features = feature_builder.update(
            time=time_str,
            force_left=force_left,
            force_right=force_right,
            dis_left=dis_left,
            dis_right=dis_right,
        )

        rule_result = engine.check_point(features, return_dict=True)

        all_features.append(features)
        all_rule_results.append(rule_result)

    # =====================================================
    # 按窗口划分并打标签
    # =====================================================
    window_rows = []
    total_points = len(all_features)

    for start in range(0, total_points - WINDOW_SIZE + 1, WINDOW_STEP):
        end = start + WINDOW_SIZE

        window_results = all_rule_results[start:end]
        window_features = all_features[start:end]

        train_label, strict_label = classify_window_label(window_results)

        # 取窗口最后一个点的特征作为代表
        last_feat = window_features[-1]

        row = dict(last_feat)
        row["window_start_idx"] = start
        row["window_end_idx"] = end - 1
        row["train_label"] = train_label
        row["strict_label"] = strict_label

        # 记录窗口内的异常类型
        anomaly_types = []

        for r in window_results:
            if r.get("status") in ("warning", "alarm"):
                anomaly_types.append(r.get("type", "unknown"))

        row["window_anomaly_types"] = "|".join(anomaly_types) if anomaly_types else ""

        window_rows.append(row)

    # =====================================================
    # final_check
    # =====================================================
    final_results = engine.final_check()
    has_final_hard = False

    for item in final_results:
        if item.get("status") in ("warning", "alarm"):
            has_final_hard = True

    return {
        "file_name": file_name,
        "target_force": target_force,
        "total_points": total_points,
        "total_windows": len(window_rows),
        "has_final_hard": has_final_hard,
        "window_rows": window_rows,
    }


# =========================================================
# 8. 批量处理
# =========================================================
def classify_all():
    ensure_dirs()

    csv_files = get_csv_files(DATA_DIR)

    if not csv_files:
        raise FileNotFoundError(f"没有找到 CSV: {DATA_DIR}")

    print("=" * 70)
    print("开始训练数据筛选")
    print("=" * 70)
    print(f"数据目录: {DATA_DIR}")
    print(f"文件数: {len(csv_files)}")
    print(f"窗口大小: {WINDOW_SIZE}，步长: {WINDOW_STEP}")
    print()

    all_normal = []
    all_warning = []
    all_alarm = []
    report_rows = []

    for idx, csv_path in enumerate(csv_files, 1):
        file_name = os.path.basename(csv_path)

        try:
            result = process_one_file(csv_path)

            n_normal = 0
            n_warning = 0
            n_alarm = 0

            for row in result["window_rows"]:
                label = row["train_label"]

                if label == "normal":
                    all_normal.append(row)
                    n_normal += 1
                elif label == "alarm":
                    all_alarm.append(row)
                    n_alarm += 1

                # 严格标签用于统计
                if row["strict_label"] == "warning":
                    all_warning.append(row)
                    n_warning += 1

            report_rows.append({
                "file": file_name,
                "target_force": result["target_force"],
                "total_points": result["total_points"],
                "total_windows": result["total_windows"],
                "normal_windows": n_normal,
                "warning_windows": n_warning,
                "alarm_windows": n_alarm,
                "has_final_hard": result["has_final_hard"],
            })

            print(
                f"[{idx}/{len(csv_files)}] {file_name} | "
                f"窗口={result['total_windows']} | "
                f"normal={n_normal} warning={n_warning} alarm={n_alarm}"
            )

        except Exception as e:
            print(f"[{idx}/{len(csv_files)}] {file_name} | 失败: {e}")

            report_rows.append({
                "file": file_name,
                "target_force": "",
                "total_points": "",
                "total_windows": "",
                "normal_windows": "",
                "warning_windows": "",
                "alarm_windows": "",
                "has_final_hard": "",
            })

    # =====================================================
    # 保存
    # =====================================================
    if all_normal:
        pd.DataFrame(all_normal).to_csv(
            os.path.join(NORMAL_DIR, "1_normal.csv"),
            index=False,
            encoding="utf-8-sig",
        )

    if all_warning:
        pd.DataFrame(all_warning).to_csv(
            os.path.join(WARNING_DIR, "2_warning_recoverable.csv"),
            index=False,
            encoding="utf-8-sig",
        )

    if all_alarm:
        pd.DataFrame(all_alarm).to_csv(
            os.path.join(ALARM_DIR, "3_alarm.csv"),
            index=False,
            encoding="utf-8-sig",
        )

    pd.DataFrame(report_rows).to_csv(
        REPORT_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    print()
    print("=" * 70)
    print("筛选完成")
    print("=" * 70)
    print(f"normal 窗口数（训练用）: {len(all_normal)}")
    print(f"warning 窗口数（测试用）: {len(all_warning)}")
    print(f"alarm 窗口数（测试用）:   {len(all_alarm)}")
    print()
    print(f"normal 数据: {NORMAL_DIR}")
    print(f"warning 数据: {WARNING_DIR}")
    print(f"alarm 数据: {ALARM_DIR}")
    print(f"分类报告: {REPORT_PATH}")


if __name__ == "__main__":
    classify_all()