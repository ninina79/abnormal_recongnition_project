import os
import shutil
#文件分类，一根钢筋是几个文件，将两个文件的提出来

def classify_steel_strands(raw_dir, output_base):
    """
    第一步：根据 CSV 文件名的前四段逻辑进行钢束物理分类
    并将 JSON 文件保存到指定的 json 子目录中
    """
    # 1. 初始化分类文件夹 (one, two, ..., other)
    folder_map = {1: 'one', 2: 'two', 3: 'three', 4: 'four', 5: 'five', 6: 'six'}
    for name in list(folder_map.values()) + ['other']:
        os.makedirs(os.path.join(output_base, name), exist_ok=True)

    # 创建 json 文件夹
    json_target_dir = os.path.join(output_base, 'json')
    os.makedirs(json_target_dir, exist_ok=True)

    # 2. 获取目录下所有 CSV 和 JSON
    all_files = os.listdir(raw_dir)
    csv_files = [f for f in all_files if f.endswith('.csv')]
    json_files = [f for f in all_files if f.endswith('.json')]

    print(f"--- 开始扫描原始目录: {raw_dir} ---")
    print(f"检测到 {len(csv_files)} 个数据文件和 {len(json_files)} 个项目说明文件。\n")

    # 3. 按前四段进行分组识别钢束
    strand_groups = {}
    for fname in csv_files:
        segments = fname.split('-')
        if len(segments) >= 4:
            group_key = "-".join(segments[:4])
            if group_key not in strand_groups:
                strand_groups[group_key] = []
            strand_groups[group_key].append(fname)

    # 4. 执行 CSV 文件分类拷贝
    count_stats = {name: 0 for name in list(folder_map.values()) + ['other']}

    for key, files in strand_groups.items():
        file_count = len(files)
        folder_name = folder_map.get(file_count, 'other')
        target_dir = os.path.join(output_base, folder_name)

        for f in files:
            src = os.path.join(raw_dir, f)
            dst = os.path.join(target_dir, f)
            shutil.copy2(src, dst)

        count_stats[folder_name] += 1
        print(f"已识别钢束: {key} | 文件数: {file_count} -> 分类至: {folder_name}")

    # 5. 将 JSON 文件保存到 E:\ABNORMAL_RECOGNITION\think\processed\json
    if json_files:
        print(f"\n--- 正在将 JSON 存入专用目录: {json_target_dir} ---")
        for j in json_files:
            shutil.copy2(os.path.join(raw_dir, j), os.path.join(json_target_dir, j))
            print(f"已保存 JSON: {j}")

    print("\n" + "=" * 30)
    print("分类任务完成！统计结果见上文。")
    print("=" * 30)


if __name__ == "__main__":
    # 配置路径
    RAW_DIR = r"E:\ABNORMAL_RECOGNITION\jiaoda_data"
    PROCESSED_BASE = r"E:\ABNORMAL_RECOGNITION\DATA\processed_1"
    # 就是最原始的那个数据，一个json加上好几个数据文件
    if os.path.exists(RAW_DIR):
        classify_steel_strands(RAW_DIR, PROCESSED_BASE)
    else:
        print(f"错误：找不到原始路径 {RAW_DIR}")