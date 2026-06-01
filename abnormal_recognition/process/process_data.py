import pandas as pd
import os
import json

# ==========================================
# 1. 滤波器定义（EWMA，因果实时）
# ==========================================
def ewma_filter(x: pd.Series, alpha: float = 0.25) -> pd.Series:
    """
    EWMA 滤波器，适用于时序数据。
    y[t] = alpha * x[t] + (1-alpha) * y[t-1]
    """
    if not (0 < alpha <= 1):
        raise ValueError("alpha must be in (0, 1].")
    x = x.astype(float)
    y = x.copy()
    for i in range(1, len(x)):
        y.iloc[i] = alpha * x.iloc[i] + (1.0 - alpha) * y.iloc[i-1]
    return y


# ==========================================
# 2. 路径配置
# ==========================================
PROCESSED_BASE = r"E:\ABNORMAL_RECOGNITION\DATA\processed_1"
TWO_DIR = os.path.join(PROCESSED_BASE, 'two')
JSON_DIR = os.path.join(PROCESSED_BASE, 'json')
SAVE_DIR = os.path.join(PROCESSED_BASE, 'integrated_two')

os.makedirs(SAVE_DIR, exist_ok=True)


def get_target_params(group_key):
    """
    从 JSON 中查找目标力值和理论伸长量
    """
    try:
        segments = group_key.split('-')
        json_fname = f"{segments[0]}-{segments[1]}.json"
        strand_ids = [segments[2], segments[3]]

        json_path = os.path.join(JSON_DIR, json_fname)
        if os.path.exists(json_path):
            with open(json_path, 'r', encoding='utf-8-sig') as f:
                data = json.load(f)
                for item in data:
                    if str(item.get('id')) in strand_ids:
                        return {
                            'target_f': float(item.get('targetPulling', 1000.0)),
                            'theory_dis': float(item.get('theoryDis', 0.0))
                        }
    except Exception as e:
        print(f"  [警告] 解析 JSON 出错 {group_key}: {e}")

    return {
        'target_f': 1000.0,
        'theory_dis': 0.0
    }


def process_two_strand_data():
    all_files = [f for f in os.listdir(TWO_DIR) if f.endswith('.csv')]
    strand_groups = {}

    # 按前四段分组
    for fname in all_files:
        segments = fname.split('-')
        if len(segments) >= 4:
            group_key = "-".join(segments[:4])
            strand_groups.setdefault(group_key, []).append(fname)

    print(f"找到 {len(strand_groups)} 组待整合数据...")

    for key, files in strand_groups.items():
        if len(files) != 2:
            continue

        try:
            params = get_target_params(key)
            target_f = params['target_f']

            # ================================
            # 1. 读取原始数据
            # ================================
            df_l = pd.read_csv(os.path.join(TWO_DIR, files[0]))
            df_r = pd.read_csv(os.path.join(TWO_DIR, files[1]))

            df_l.columns = ['time', 'f_raw_left', 'd_raw_left']
            df_r.columns = ['time', 'f_raw_right', 'd_raw_right']

            final_df = pd.merge(df_l, df_r, on='time')

            if final_df.empty:
                print(f"[跳过] {key} 合并为空")
                continue

            # ================================
            # 2. 对原始力值进行 EWMA 滤波 (alpha=0.25)
            # ================================
            # 确保按时间排序（原始数据可能已有序，但保险）
            final_df = final_df.sort_values('time').reset_index(drop=True)

            final_df["force_left_raw"] = final_df["f_raw_left"]   # 保留原始值
            final_df["force_right_raw"] = final_df["f_raw_right"]

            # 应用 EWMA 滤波
            final_df["force_left"] = ewma_filter(final_df["f_raw_left"], alpha=0.25)
            final_df["force_right"] = ewma_filter(final_df["f_raw_right"], alpha=0.25)

            # ================================
            # 3. 基于滤波后力值计算衍生特征
            # ================================
            final_df["force_avg"] = (final_df["force_left"] + final_df["force_right"]) / 2
            final_df["force_sum"] = final_df["force_left"] + final_df["force_right"]
            final_df["force_diff"] = (final_df["force_left"] - final_df["force_right"]).abs()
            final_df["force_diff_ratio"] = final_df["force_diff"] / (final_df["force_avg"].abs() + 1e-6)

            # ================================
            # 4. 位移（不滤波）
            # ================================
            final_df["dis_left"] = final_df["d_raw_left"]
            final_df["dis_right"] = final_df["d_raw_right"]
            final_df["total_delta_dis"] = final_df["dis_left"] + final_df["dis_right"]
            final_df["dis_diff"] = (final_df["dis_left"] - final_df["dis_right"]).abs()

            # ================================
            # 5. 时间处理
            # ================================
            final_df["time_dt"] = pd.to_datetime(final_df["time"])
            final_df = final_df.sort_values("time_dt").reset_index(drop=True)
            final_df["dt"] = final_df["time_dt"].diff().dt.total_seconds().fillna(1)
            final_df.loc[final_df["dt"] <= 0, "dt"] = 1

            # ================================
            # 6. 力速度（基于滤波后力值）
            # ================================
            final_df["left_force_rate"] = final_df["force_left"].diff().fillna(0) / final_df["dt"]
            final_df["right_force_rate"] = final_df["force_right"].diff().fillna(0) / final_df["dt"]
            final_df["force_rate"] = final_df["force_avg"].diff().fillna(0) / final_df["dt"]
            final_df["force_rate_diff"] = (final_df["left_force_rate"] - final_df["right_force_rate"]).abs()
            final_df["force_rate_ratio"] = final_df["force_rate_diff"] / (final_df["force_rate"].abs() + 1e-6)

            # ================================
            # 7. 力加速度（基于滤波后力值）
            # ================================
            final_df["left_force_acc"] = final_df["left_force_rate"].diff().fillna(0) / final_df["dt"]
            final_df["right_force_acc"] = final_df["right_force_rate"].diff().fillna(0) / final_df["dt"]
            final_df["force_acc"] = final_df["force_rate"].diff().fillna(0) / final_df["dt"]

            # ================================
            # 8. 位移速度
            # ================================
            final_df["left_dis_rate"] = final_df["dis_left"].diff().fillna(0) / final_df["dt"]
            final_df["right_dis_rate"] = final_df["dis_right"].diff().fillna(0) / final_df["dt"]
            final_df["dis_rate_diff"] = (final_df["left_dis_rate"] - final_df["right_dis_rate"]).abs()

            # ================================
            # 9. 统计特征
            # ================================
            final_df["stiffness_ratio"] = final_df["force_avg"] / (final_df["total_delta_dis"] + 1e-6)
            final_df["force_std_5s"] = final_df["force_avg"].rolling(window=5).std().fillna(0)
            final_df["force_diff_std_5s"] = final_df["force_diff"].rolling(window=5).std().fillna(0)
            final_df["force_rate_std_5s"] = final_df["force_rate"].rolling(window=5).std().fillna(0)

            # ================================
            # 10. 输出字段
            # ================================
            output_cols = [
                "time",
                "force_left_raw",   # 原始左力值
                "force_right_raw",  # 原始右力值
                "force_left",       # 滤波后左力值
                "force_right",      # 滤波后右力值
                "dis_left",
                "dis_right",

                "force_avg",
                "force_sum",
                "force_diff",
                "force_diff_ratio",

                "total_delta_dis",
                "dis_diff",

                "left_force_rate",
                "right_force_rate",
                "force_rate",
                "force_rate_diff",
                "force_rate_ratio",

                "left_force_acc",
                "right_force_acc",
                "force_acc",

                "left_dis_rate",
                "right_dis_rate",
                "dis_rate_diff",

                "stiffness_ratio",
                "force_std_5s",
                "force_diff_std_5s",
                "force_rate_std_5s",
            ]

            save_path = os.path.join(SAVE_DIR, f"integrated_{key}.csv")
            final_df[output_cols].to_csv(save_path, index=False, encoding="utf-8-sig")
            print(f"[成功] {key} | 行数: {len(final_df)}")

        except Exception as e:
            print(f"[失败] {key}: {e}")


if __name__ == "__main__":
    process_two_strand_data()