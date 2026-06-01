"""
classify_zhangla_data.py
功能：从 integrated_two 文件夹中读取每个完整张拉CSV文件，
      提取张拉阶段数据（从第一行到 force_avg 首次达到 target_force * 0.98），
      保存到 zhangla_data 文件夹。
"""

import pandas as pd
import os
import json

# ==========================================
# 路径配置
# ==========================================
PROCESSED_BASE = r"E:\ABNORMAL_RECOGNITION\DATA\processed_1"
INPUT_DIR = os.path.join(PROCESSED_BASE, 'integrated_two')
JSON_DIR = os.path.join(PROCESSED_BASE, 'json')
OUTPUT_DIR = os.path.join(PROCESSED_BASE, 'zhangla_data')

os.makedirs(OUTPUT_DIR, exist_ok=True)

# 切分阈值：达到目标力值的这个比例就认为张拉阶段结束
TENSION_RATIO_THRESHOLD = 0.98

# 张拉阶段最少数据点数，低于此值的文件跳过
MIN_TENSION_ROWS = 15

# 默认目标力值（当 JSON 匹配失败时使用）
DEFAULT_TARGET_FORCE = 1000.0


def get_target_force(group_key):
    """
    从 JSON 中查找目标力值。
    group_key 格式: "梁号ID-任务ID-通道ID1-通道ID2"
    """
    try:
        segments = group_key.split('-')
        if len(segments) < 4:
            return DEFAULT_TARGET_FORCE

        json_fname = f"{segments[0]}-{segments[1]}.json"
        strand_ids = [segments[2], segments[3]]

        json_path = os.path.join(JSON_DIR, json_fname)
        if not os.path.exists(json_path):
            print(f"  [警告] JSON文件不存在: {json_fname}")
            return DEFAULT_TARGET_FORCE

        with open(json_path, 'r', encoding='utf-8-sig') as f:
            data = json.load(f)
            for item in data:
                if str(item.get('id')) in strand_ids:
                    target = item.get('targetPulling')
                    if target is not None:
                        return float(target)

        print(f"  [警告] JSON中未找到匹配通道: {strand_ids}")
        return DEFAULT_TARGET_FORCE

    except Exception as e:
        print(f"  [警告] 解析JSON出错 {group_key}: {e}")
        return DEFAULT_TARGET_FORCE


def extract_tensioning_phase(df, target_force):
    """
    从完整张拉数据中提取张拉阶段。

    张拉阶段定义：从第一行到 force_avg 首次达到 target_force * TENSION_RATIO_THRESHOLD。

    返回:
        tensioning_df: 张拉阶段的 DataFrame，如果提取失败返回 None
        cut_index: 切分点的行索引
        cut_force: 切分点的 force_avg 值
        cut_ratio: 切分点力值占目标力值的比例
    """
    threshold = target_force * TENSION_RATIO_THRESHOLD

    # 找到 force_avg 首次达到阈值的行
    above_threshold = df[df['force_avg'] >= threshold]

    if above_threshold.empty:
        max_force = df['force_avg'].max()
        max_ratio = max_force / target_force if target_force > 0 else 0
        return None, -1, max_force, max_ratio

    cut_index = above_threshold.index[0]
    tensioning_df = df.loc[:cut_index].copy()
    cut_force = df.loc[cut_index, 'force_avg']
    cut_ratio = cut_force / target_force if target_force > 0 else 0

    return tensioning_df, cut_index, cut_force, cut_ratio


def split_all_files():
    """
    遍历所有 integrated CSV 文件，提取张拉阶段并保存。
    """
    all_files = [f for f in os.listdir(INPUT_DIR) if f.endswith('.csv')]
    print(f"找到 {len(all_files)} 个CSV文件待处理\n")

    success_count = 0
    skip_no_threshold = 0
    skip_too_short = 0
    fail_count = 0
    results = []

    for i, fname in enumerate(all_files):
        # 从文件名提取 group_key
        # 文件名格式: integrated_{梁号ID}-{任务ID}-{通道ID1}-{通道ID2}.csv
        if fname.startswith('integrated_'):
            group_key = fname[len('integrated_'):-len('.csv')]
        else:
            group_key = fname[:-len('.csv')]

        try:
            filepath = os.path.join(INPUT_DIR, fname)
            df = pd.read_csv(filepath)

            if df.empty or 'force_avg' not in df.columns:
                print(f"  [{i+1}/{len(all_files)}] [跳过] {fname} - 空文件或缺少force_avg列")
                fail_count += 1
                continue

            total_rows = len(df)
            target_force = get_target_force(group_key)

            tensioning_df, cut_index, cut_force, cut_ratio = extract_tensioning_phase(
                df, target_force
            )

            if tensioning_df is None:
                print(
                    f"  [{i+1}/{len(all_files)}] [跳过] {fname} - "
                    f"force_avg未达到阈值 "
                    f"(最大值={cut_force:.1f}, 目标={target_force:.1f}, "
                    f"比例={cut_ratio:.2%}, 阈值={TENSION_RATIO_THRESHOLD:.0%})"
                )
                skip_no_threshold += 1
                continue

            tension_rows = len(tensioning_df)

            if tension_rows < MIN_TENSION_ROWS:
                print(
                    f"  [{i+1}/{len(all_files)}] [跳过] {fname} - "
                    f"张拉阶段太短 ({tension_rows}行 < {MIN_TENSION_ROWS}行)"
                )
                skip_too_short += 1
                continue

            # 保存张拉阶段数据，文件名保持原名不变
            save_path = os.path.join(OUTPUT_DIR, fname)
            tensioning_df.to_csv(save_path, index=False, encoding='utf-8-sig')

            success_count += 1
            results.append({
                'file': fname,
                'total_rows': total_rows,
                'tension_rows': tension_rows,
                'target_force': target_force,
                'cut_force': cut_force,
                'cut_ratio': cut_ratio,
            })

            if (i + 1) % 100 == 0 or (i + 1) == len(all_files):
                print(
                    f"  [{i+1}/{len(all_files)}] [成功] {fname} - "
                    f"总行数={total_rows}, 张拉阶段={tension_rows}行 "
                    f"({tension_rows/total_rows:.1%}), "
                    f"目标力={target_force:.1f}, "
                    f"切分点力值={cut_force:.1f} ({cut_ratio:.2%})"
                )

        except Exception as e:
            print(f"  [{i+1}/{len(all_files)}] [失败] {fname}: {e}")
            fail_count += 1

    # ==========================================
    # 汇总统计
    # ==========================================
    print("\n" + "=" * 60)
    print("处理完成 - 汇总统计")
    print("=" * 60)
    print(f"总文件数:          {len(all_files)}")
    print(f"成功提取:          {success_count}")
    print(f"跳过(未达阈值):    {skip_no_threshold}")
    print(f"跳过(数据太短):    {skip_too_short}")
    print(f"失败:              {fail_count}")

    if results:
        tension_rows_list = [r['tension_rows'] for r in results]
        ratios = [r['tension_rows'] / r['total_rows'] for r in results]

        print(f"\n张拉阶段行数统计:")
        print(f"  最小值:  {min(tension_rows_list)}")
        print(f"  最大值:  {max(tension_rows_list)}")
        print(f"  平均值:  {sum(tension_rows_list) / len(tension_rows_list):.1f}")
        print(f"  中位数:  {sorted(tension_rows_list)[len(tension_rows_list)//2]}")

        print(f"\n张拉阶段占比统计:")
        print(f"  最小占比: {min(ratios):.1%}")
        print(f"  最大占比: {max(ratios):.1%}")
        print(f"  平均占比: {sum(ratios) / len(ratios):.1%}")

    print(f"\n输出目录: {OUTPUT_DIR}")


if __name__ == "__main__":
    split_all_files()