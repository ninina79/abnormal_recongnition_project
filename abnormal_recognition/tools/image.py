# import os
# import pandas as pd
# import matplotlib.pyplot as plt
#
#
# # =========================================================
# # 1. 路径配置
# # =========================================================
# # DATA_DIR = r"E:\ABNORMAL_RECOGNITION\DATA\processed_test\rule_engine_classified\3_other_anomaly"
# DATA_DIR = r"E:\ABNORMAL_RECOGNITION\DATA\processed_test\rule_engine_classified\1_pass"
# # SAVE_DIR = r"E:\ABNORMAL_RECOGNITION\DATA\processed_test\rule_engine_classified\\plots_force_avg_time"
# SAVE_DIR = r"E:\ABNORMAL_RECOGNITION\DATA\processed_test\rule_engine_classified\\plots_force_avg_time1"
#
#
# os.makedirs(SAVE_DIR, exist_ok=True)
#
#
# # =========================================================
# # 2. 中文显示设置
# # =========================================================
# plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
# plt.rcParams["axes.unicode_minus"] = False
#
#
# # =========================================================
# # 3. 获取 CSV 文件
# # =========================================================
# def get_csv_files(data_dir):
#     files = []
#     for file_name in os.listdir(data_dir):
#         if file_name.endswith(".csv"):
#             files.append(os.path.join(data_dir, file_name))
#     return sorted(files)
#
#
# # =========================================================
# # 4. 读取单个 CSV（已修正）
# # =========================================================
# def load_data(csv_path):
#     df = pd.read_csv(csv_path, encoding="utf-8-sig")
#
#     if "time" not in df.columns:
#         raise ValueError("CSV 文件中缺少 time 列")
#     if "force_avg" not in df.columns:
#         raise ValueError("CSV 文件中缺少 force_avg 列")
#
#     # 解析时间字符串
#     df["time_dt"] = pd.to_datetime(df["time"], errors="coerce")
#     # 计算相对秒数（从第一行开始）
#     df["time_axis"] = (df["time_dt"] - df["time_dt"].iloc[0]).dt.total_seconds()
#
#     # 数值化 force_avg
#     df["force_avg"] = pd.to_numeric(df["force_avg"], errors="coerce")
#
#     # 剔除缺失值
#     df = df.dropna(subset=["time_axis", "force_avg"]).reset_index(drop=True)
#
#     if df.empty:
#         raise ValueError("time 或 force_avg 清洗后为空")
#
#     x = df["time_axis"]
#     y = df["force_avg"]
#     return x, y
#
#
# # =========================================================
# # 5. 绘制单个文件
# # =========================================================
# def plot_one_file(csv_path, save_dir):
#     file_name = os.path.basename(csv_path)
#     base_name = os.path.splitext(file_name)[0]
#
#     x, y = load_data(csv_path)
#
#     plt.figure(figsize=(12, 6))
#     plt.plot(x, y, linewidth=2, label="平均张拉力 force_avg")
#     plt.title(f"平均张拉力-时间曲线\n{file_name}")
#     plt.xlabel("时间 / s")
#     plt.ylabel("平均张拉力 / kN")
#     plt.grid(alpha=0.3)
#     plt.legend()
#     plt.tight_layout()
#
#     save_path = os.path.join(save_dir, f"{base_name}_force_avg_time.png")
#     plt.savefig(save_path, dpi=300)
#     plt.close()
#
#     print(f"[完成] {file_name} -> {save_path}")
#
#
# # =========================================================
# # 6. 批量绘图
# # =========================================================
# def batch_plot_force_avg_time():
#     csv_files = get_csv_files(DATA_DIR)
#
#     if not csv_files:
#         raise FileNotFoundError(f"没有找到 CSV 文件: {DATA_DIR}")
#
#     print("=" * 60)
#     print("开始批量绘制平均张拉力-时间曲线")
#     print("=" * 60)
#     print(f"数据目录: {DATA_DIR}")
#     print(f"保存目录: {SAVE_DIR}")
#     print(f"CSV 文件数量: {len(csv_files)}")
#     print()
#
#     success_count = 0
#     fail_count = 0
#
#     for csv_path in csv_files:
#         try:
#             plot_one_file(csv_path, SAVE_DIR)
#             success_count += 1
#         except Exception as e:
#             fail_count += 1
#             print(f"[失败] {os.path.basename(csv_path)} | 原因: {e}")
#
#     print()
#     print("=" * 60)
#     print("批量绘图完成")
#     print("=" * 60)
#     print(f"成功: {success_count}")
#     print(f"失败: {fail_count}")
#
#
# # =========================================================
# # 7. 主入口
# # =========================================================
# if __name__ == "__main__":
#     batch_plot_force_avg_time()
#
# #
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# 读取 CSV 文件（请替换为你的实际文件路径）
file_path = "data/test_scenarios/01_speed_too_fast.csv"
df = pd.read_csv(file_path)

# 解析时间列
df['time'] = pd.to_datetime(df['time'])

# 计算左右力之和
df['force_sum'] = df['force_left'] + df['force_right']

# 绘图
plt.figure(figsize=(12, 6))
plt.plot(df['time'], df['force_sum'], linewidth=1, color='blue')
plt.xlabel('Time')
plt.ylabel('Total Force (force_left + force_right)')
plt.title('Total Force over Time')
plt.grid(True, linestyle='--', alpha=0.6)
plt.xticks(rotation=45)
plt.tight_layout()

# 显示图形
plt.show()



