import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ================= 配置 Matplotlib 解决中文乱码 =================
plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows 请用 SimHei，Mac 请改用 Arial Unicode MS
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号


def extract_speeds_for_window(data_dir: str, window_size: int, force_min_pct: float = 0.20,
                              force_max_pct: float = 0.95):
    """
    内部提取函数：计算特定窗口大小下的基准速度
    """
    file_pattern = os.path.join(data_dir, "integrated_*.csv")
    csv_files = glob.glob(file_pattern)

    if not csv_files:
        return None

    file_representative_speeds = []

    for file in csv_files:
        try:
            df = pd.read_csv(file, usecols=['force_avg'])
            if len(df) <= window_size:
                continue

            # 计算宏观窗口速度
            window_speed = (df['force_avg'] - df['force_avg'].shift(window_size)) / window_size

            # === 物理阶段切分 (20% ~ 95%) ===
            max_force = df['force_avg'].max()
            if max_force < 100:
                continue

            loading_mask = (df['force_avg'] >= max_force * force_min_pct) & (
                        df['force_avg'] <= max_force * force_max_pct)
            active_speeds = window_speed[loading_mask].dropna()

            if not active_speeds.empty:
                file_representative_speeds.append(active_speeds.median())

        except Exception:
            pass

    final_speeds = np.array(file_representative_speeds)
    if len(final_speeds) == 0:
        return None

    # 计算众数基准 (最高柱子)
    counts, bins = np.histogram(final_speeds, bins=40)
    max_bin_idx = np.argmax(counts)
    global_baseline = (bins[max_bin_idx] + bins[max_bin_idx + 1]) / 2.0

    return final_speeds, global_baseline


def plot_multiple_windows(data_dir: str, windows=[10, 15, 20, 25, 30,35,40]):
    """
    一键生成多个时间窗口的对比图
    """
    print(f"🚀 开始批量分析，测试窗口: {windows}秒...")
    print(f"数据切分范围: 20% ~ 95% Fmax")

    num_plots = len(windows)
    fig, axes = plt.subplots(num_plots, 1, figsize=(10, 3.5 * num_plots))
    if num_plots == 1: axes = [axes]  # 兼容单个窗口的情况

    for ax, w in zip(axes, windows):
        print(f"⏳ 正在处理 {w}秒 窗口数据...")
        result = extract_speeds_for_window(data_dir, window_size=w, force_min_pct=0.20, force_max_pct=0.95)

        if result is None:
            ax.set_title(f"{w}秒窗口: 数据不足", color='red')
            continue

        final_speeds, baseline = result

        # 绘图
        sns.histplot(final_speeds, bins=40, kde=True, color='#409EFF', stat='density', alpha=0.6, ax=ax)
        ax.axvline(baseline, color='#F56C6C', linestyle='-', linewidth=2.5, label=f'基准点: {baseline:.2f} kN/s')

        # 美化子图
        ax.set_title(f'窗口大小 {w} 秒 速度分布 (有效文件数: {len(final_speeds)})', fontsize=12, fontweight='bold')
        ax.set_ylabel('密度')
        ax.legend(loc='upper right')
        ax.grid(axis='y', alpha=0.3, linestyle='--')

        # 限制X轴范围，排除极端离群值让图更好看 (取1%和99%分位数)
        q_low, q_high = np.percentile(final_speeds, [1, 99])
        ax.set_xlim(max(0, q_low - 5), q_high + 5)

    plt.xlabel('张拉速度 (kN/s)', fontsize=12)
    plt.tight_layout()

    # 保存结果图
    plot_path = os.path.join(os.path.dirname(data_dir), "Window_Size_Comparison.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ 分析完成！对比图已保存至: {plot_path}")
    plt.show()


if __name__ == "__main__":
    # 请确保此路径是你电脑上实际存放 integrated_*.csv 文件的路径
    TARGET_DIR = r"E:\ABNORMAL_RECOGNITION\DATA\processed_test\integrated_two"
    plot_multiple_windows(TARGET_DIR, windows=[5,10, 15])