"""
TCN_DATA.py
功能：从 zhangla_data 文件夹读取张拉阶段CSV文件，
      进行工程归一化 → 滑动窗口划分 → 窗口质量检查 → 训练/验证拆分 → 保存。
      输出供 train_tcn_trend_model.py 直接加载训练。
"""

import pandas as pd
import numpy as np
import os
import json

# ==========================================
# 路径配置
# ==========================================
PROCESSED_BASE = r"E:\ABNORMAL_RECOGNITION\DATA\processed"
INPUT_DIR = os.path.join(PROCESSED_BASE, 'zhangla_data')
JSON_DIR = os.path.join(PROCESSED_BASE, 'json')
OUTPUT_DIR = os.path.join(PROCESSED_BASE, 'tcn_training_data')

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================
# 窗口参数
# ==========================================
INPUT_WINDOW = 10          # TCN输入步数
PREDICT_HORIZON = 5        # TCN预测步数
TOTAL_WINDOW = INPUT_WINDOW + PREDICT_HORIZON  # 总窗口长度 = 15
SLIDE_STEP = 2             # 滑动步长

# ==========================================
# 训练/验证拆分
# ==========================================
VAL_RATIO = 0.15           # 验证集比例
RANDOM_SEED = 42

# ==========================================
# 特征列配置（与 process_data.py 输出列名一致）
# ==========================================

# 原始特征列（从CSV中读取，用于检查是否存在）
RAW_FEATURE_COLS = [
    'force_left', 'force_right', 'force_avg',
    'dis_left', 'dis_right', 'total_delta_dis',
]

# 归一化后用于TCN训练的输入特征列
TCN_INPUT_COLS = [
    'norm_force_left',
    'norm_force_right',
    'norm_force_avg',
    'norm_force_diff',
    'norm_force_rate',
    'norm_left_force_rate',
    'norm_right_force_rate',
    'norm_force_rate_diff',
    'norm_force_acc',
    'dis_left',
    'dis_right',
    'total_delta_dis',
    'dis_diff',
    'dis_rate',
    'force_disp_ratio',
    'left_right_force_diff',
    'stiffness_ratio_norm',
    'force_std_5s_norm',
]

# TCN预测目标列
TARGET_COLS = [
    'norm_force_avg',
    'norm_force_diff',
    'norm_force_rate',
]

# ==========================================
# 质量检查参数
# ==========================================
NORM_FORCE_MIN = 0.0
NORM_FORCE_MAX = 1.15
FORCE_JUMP_THRESHOLD = 0.08
DISP_BACKWARD_THRESHOLD = 0.5
FORCE_STAGNATION_THRESHOLD = 0.005
DISP_STAGNATION_THRESHOLD = 0.05
FORCE_DECLINE_TOLERANCE = 0.01
FORCE_RATE_ABNORMAL = 0.1
FORCE_DISP_CORR_MIN = 0.3

# 默认目标力值
DEFAULT_TARGET_FORCE = 1000.0


def get_target_force(group_key):
    """从 JSON 中查找目标力值"""
    try:
        segments = group_key.split('-')
        if len(segments) < 4:
            return DEFAULT_TARGET_FORCE

        json_fname = f"{segments[0]}-{segments[1]}.json"
        strand_ids = [segments[2], segments[3]]

        json_path = os.path.join(JSON_DIR, json_fname)
        if not os.path.exists(json_path):
            return DEFAULT_TARGET_FORCE

        with open(json_path, 'r', encoding='utf-8-sig') as f:
            data = json.load(f)
            for item in data:
                if str(item.get('id')) in strand_ids:
                    target = item.get('targetPulling')
                    if target is not None:
                        return float(target)

        return DEFAULT_TARGET_FORCE

    except Exception:
        return DEFAULT_TARGET_FORCE


def engineer_normalize(df, target_force):
    """
    工程归一化 + 派生特征计算。
    列名与 process_data.py / feature_engine.py 的输出保持一致。
    """
    tf = target_force if target_force > 0 else 1.0

    # 力值归一化
    df['norm_force_left'] = df['force_left'] / tf
    df['norm_force_right'] = df['force_right'] / tf
    df['norm_force_avg'] = df['force_avg'] / tf
    df['norm_force_diff'] = df['force_diff'] / tf if 'force_diff' in df.columns else (
        (df['force_left'] - df['force_right']).abs() / tf
    )

    # 力值变化率归一化
    if 'force_rate' in df.columns:
        df['norm_force_rate'] = df['force_rate'] / tf
    else:
        df['norm_force_rate'] = df['norm_force_avg'].diff().fillna(0)

    if 'left_force_rate' in df.columns:
        df['norm_left_force_rate'] = df['left_force_rate'] / tf
    else:
        df['norm_left_force_rate'] = df['norm_force_left'].diff().fillna(0)

    if 'right_force_rate' in df.columns:
        df['norm_right_force_rate'] = df['right_force_rate'] / tf
    else:
        df['norm_right_force_rate'] = df['norm_force_right'].diff().fillna(0)

    if 'force_rate_diff' in df.columns:
        df['norm_force_rate_diff'] = df['force_rate_diff'] / tf
    else:
        df['norm_force_rate_diff'] = (
            df['norm_left_force_rate'] - df['norm_right_force_rate']
        ).abs()

    # 力值加速度归一化
    if 'force_acc' in df.columns:
        df['norm_force_acc'] = df['force_acc'] / tf
    else:
        df['norm_force_acc'] = df['norm_force_rate'].diff().fillna(0)

    # 位移变化率
    if 'total_delta_dis' in df.columns:
        df['dis_rate'] = df['total_delta_dis'].diff().fillna(0)
    else:
        df['dis_rate'] = 0.0

    # 力值-位移比（避免除零）
    dis_rate_safe = df['dis_rate'].replace(0, np.nan)
    df['force_disp_ratio'] = (df['norm_force_rate'] / dis_rate_safe).fillna(0)
    df['force_disp_ratio'] = df['force_disp_ratio'].clip(-10, 10)

    # 左右力值差（归一化后）
    df['left_right_force_diff'] = df['norm_force_left'] - df['norm_force_right']

    # 刚度比归一化
    if 'stiffness_ratio' in df.columns:
        # 用简单的 clip + 缩放
        df['stiffness_ratio_norm'] = df['stiffness_ratio'].clip(-100, 100) / 100.0
    else:
        df['stiffness_ratio_norm'] = (
            df['norm_force_avg'] / (df['total_delta_dis'] + 1e-6)
        ).clip(-10, 10)

    # 力值标准差归一化
    if 'force_std_5s' in df.columns:
        df['force_std_5s_norm'] = df['force_std_5s'] / tf
    else:
        df['force_std_5s_norm'] = (
            df['norm_force_avg'].rolling(window=5).std().fillna(0)
        )

    return df


def check_window_quality(window_df):
    """
    检查一个窗口是否适合TCN训练。
    返回 (is_valid, reason)。
    """
    # 一、数据完整性检查
    if window_df[TCN_INPUT_COLS].isnull().any().any():
        return False, "存在缺失值"

    if np.isinf(window_df[TCN_INPUT_COLS].values).any():
        return False, "存在无穷值"

    norm_force = window_df['norm_force_avg'].values
    if (norm_force < NORM_FORCE_MIN).any() or (norm_force > NORM_FORCE_MAX).any():
        return False, "力值超出合理范围"

    # 二、张拉过程异常检查
    # 位移：优先用 total_delta_dis
    disp = window_df['total_delta_dis'].values
    force_rate = window_df['norm_force_rate'].values

    # 力值跳变
    force_diff = np.abs(np.diff(norm_force))
    if (force_diff > FORCE_JUMP_THRESHOLD).any():
        return False, "力值跳变过大"

    # 位移回退
    disp_diff = np.diff(disp)
    if (disp_diff < -DISP_BACKWARD_THRESHOLD).any():
        return False, "位移回退过大"

    # 力值停滞
    force_range = norm_force.max() - norm_force.min()
    if force_range < FORCE_STAGNATION_THRESHOLD:
        return False, "力值停滞"

    # 位移停滞
    disp_range = disp.max() - disp.min()
    if disp_range < DISP_STAGNATION_THRESHOLD:
        return False, "位移停滞"

    # 力值整体下降
    half = len(norm_force) // 2
    first_half_mean = norm_force[:half].mean()
    second_half_mean = norm_force[half:].mean()
    if second_half_mean < first_half_mean - FORCE_DECLINE_TOLERANCE:
        return False, "力值整体下降"

    # 力值变化率异常
    valid_rate = force_rate[1:] if len(force_rate) > 1 else force_rate
    if (np.abs(valid_rate) > FORCE_RATE_ABNORMAL).any():
        return False, "力值变化率异常"

    # 力值-位移协调性
    if len(norm_force) >= 5:
        corr = np.corrcoef(norm_force, disp)[0, 1]
        if not np.isnan(corr) and corr < FORCE_DISP_CORR_MIN:
            return False, "力值-位移协调性差"

    # 本脚本窗口来自张拉阶段 CSV；两端力差/左右趋势一致性仅在持荷阶段做业务判定，
    # 不在此作为窗口剔除条件（与 RuleEngine / RealTimeTensionMonitor 一致）。

    return True, "合格"


def process_single_file(filepath, fname):
    """处理单个张拉阶段CSV文件，返回合格的窗口数组"""
    if fname.startswith('integrated_'):
        group_key = fname[len('integrated_'):-len('.csv')]
    else:
        group_key = fname[:-len('.csv')]

    df = pd.read_csv(filepath)

    stats = {
        'total_rows': len(df),
        'total_windows': 0,
        'valid_windows': 0,
        'reject_reasons': {}
    }

    if len(df) < TOTAL_WINDOW:
        stats['skip_reason'] = f"数据行数不足 ({len(df)} < {TOTAL_WINDOW})"
        return [], [], stats

    # 检查必要列
    missing_cols = [c for c in RAW_FEATURE_COLS if c not in df.columns]
    if missing_cols:
        stats['skip_reason'] = f"缺少列: {missing_cols}"
        return [], [], stats

    target_force = get_target_force(group_key)
    if target_force <= 0:
        stats['skip_reason'] = f"目标力值无效: {target_force}"
        return [], [], stats

    # 工程归一化 + 派生特征
    df = engineer_normalize(df, target_force)

    # 检查归一化后的列是否都存在
    missing_input = [c for c in TCN_INPUT_COLS if c not in df.columns]
    if missing_input:
        stats['skip_reason'] = f"归一化后缺少输入列: {missing_input}"
        return [], [], stats

    missing_target = [c for c in TARGET_COLS if c not in df.columns]
    if missing_target:
        stats['skip_reason'] = f"归一化后缺少目标列: {missing_target}"
        return [], [], stats

    # 滑动窗口划分 + 质量检查
    windows_X = []
    windows_Y = []

    for start in range(0, len(df) - TOTAL_WINDOW + 1, SLIDE_STEP):
        end = start + TOTAL_WINDOW
        window_df = df.iloc[start:end].copy()
        stats['total_windows'] += 1

        is_valid, reason = check_window_quality(window_df)

        if not is_valid:
            reason_key = reason.split('(')[0].strip()
            stats['reject_reasons'][reason_key] = (
                stats['reject_reasons'].get(reason_key, 0) + 1
            )
            continue

        # 提取特征矩阵和目标矩阵
        feature_values = window_df[TCN_INPUT_COLS].values   # (TOTAL_WINDOW, n_input)
        target_values = window_df[TARGET_COLS].values        # (TOTAL_WINDOW, n_target)

        # 分割为输入和预测目标
        X = feature_values[:INPUT_WINDOW]                    # (INPUT_WINDOW, n_input)
        Y = target_values[INPUT_WINDOW:]                     # (PREDICT_HORIZON, n_target)

        windows_X.append(X)
        windows_Y.append(Y)
        stats['valid_windows'] += 1

    return windows_X, windows_Y, stats


def main():
    """主函数：遍历所有文件，提取合格窗口，拆分训练/验证集，保存"""
    all_files = [f for f in os.listdir(INPUT_DIR) if f.endswith('.csv')]
    print(f"找到 {len(all_files)} 个张拉阶段CSV文件")
    print(f"窗口参数: 输入={INPUT_WINDOW}, 预测={PREDICT_HORIZON}, "
          f"总长={TOTAL_WINDOW}, 步长={SLIDE_STEP}")
    print(f"输入特征数: {len(TCN_INPUT_COLS)}, 目标数: {len(TARGET_COLS)}")
    print(f"验证集比例: {VAL_RATIO}")
    print()

    all_X = []
    all_Y = []

    total_files = len(all_files)
    success_files = 0
    skip_files = 0
    total_valid_windows = 0
    total_checked_windows = 0
    all_reject_reasons = {}

    for i, fname in enumerate(all_files):
        filepath = os.path.join(INPUT_DIR, fname)

        try:
            windows_X, windows_Y, stats = process_single_file(filepath, fname)

            total_checked_windows += stats['total_windows']

            if 'skip_reason' in stats:
                skip_files += 1
                if (i + 1) % 500 == 0:
                    print(f"  [{i+1}/{total_files}] [跳过] {fname}: "
                          f"{stats['skip_reason']}")
                continue

            if len(windows_X) > 0:
                all_X.extend(windows_X)
                all_Y.extend(windows_Y)
                total_valid_windows += stats['valid_windows']
                success_files += 1

                for reason, count in stats['reject_reasons'].items():
                    all_reject_reasons[reason] = (
                        all_reject_reasons.get(reason, 0) + count
                    )
            else:
                skip_files += 1
                for reason, count in stats['reject_reasons'].items():
                    all_reject_reasons[reason] = (
                        all_reject_reasons.get(reason, 0) + count
                    )

            if (i + 1) % 500 == 0:
                print(f"  [{i+1}/{total_files}] 已处理, "
                      f"累计合格窗口: {total_valid_windows}")

        except Exception as e:
            print(f"  [{i+1}/{total_files}] [错误] {fname}: {e}")
            skip_files += 1

    # ==========================================
    # 检查结果
    # ==========================================
    if len(all_X) == 0:
        print("\n没有合格的窗口数据，请检查数据质量或放宽筛选条件。")
        print("\n窗口拒绝原因统计:")
        for reason, count in sorted(
            all_reject_reasons.items(), key=lambda x: -x[1]
        ):
            print(f"  {reason}: {count}")
        return

    X_array = np.array(all_X, dtype=np.float32)
    Y_array = np.array(all_Y, dtype=np.float32)

    print(f"\n总合格窗口: {len(X_array)}")
    print(f"  X shape: {X_array.shape}")
    print(f"  Y shape: {Y_array.shape}")

    # ==========================================
    # 训练/验证拆分（按样本随机拆分）
    # ==========================================
    np.random.seed(RANDOM_SEED)
    n_total = len(X_array)
    n_val = max(1, int(n_total * VAL_RATIO))
    n_train = n_total - n_val

    indices = np.random.permutation(n_total)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:]

    X_train = X_array[train_idx]
    Y_train = Y_array[train_idx]
    X_val = X_array[val_idx]
    Y_val = Y_array[val_idx]

    print(f"\n训练集: {X_train.shape[0]} 个窗口")
    print(f"验证集: {X_val.shape[0]} 个窗口")

    # ==========================================
    # 保存训练集和验证集
    # ==========================================
    train_path = os.path.join(OUTPUT_DIR, 'tcn_train_windows.npz')
    val_path = os.path.join(OUTPUT_DIR, 'tcn_val_windows.npz')

    np.savez_compressed(train_path, X=X_train, Y=Y_train)
    np.savez_compressed(val_path, X=X_val, Y=Y_val)

    print(f"\n训练集已保存: {train_path}")
    print(f"验证集已保存: {val_path}")

    # ==========================================
    # 保存配置文件（供 train_tcn_trend_model.py 读取）
    # ==========================================
    data_config = {
        "input_window": INPUT_WINDOW,
        "predict_horizon": PREDICT_HORIZON,
        "slide_step": SLIDE_STEP,
        "n_input_features": len(TCN_INPUT_COLS),
        "n_target_features": len(TARGET_COLS),
        "input_cols": TCN_INPUT_COLS,
        "target_cols": TARGET_COLS,
        "n_train": int(n_train),
        "n_val": int(n_val),
        "val_ratio": VAL_RATIO,
        "random_seed": RANDOM_SEED,
    }

    config_path = os.path.join(OUTPUT_DIR, 'tcn_data_config.json')
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(data_config, f, ensure_ascii=False, indent=2)

    print(f"配置文件已保存: {config_path}")

    # ==========================================
    # 汇总统计
    # ==========================================
    print("\n" + "=" * 60)
    print("TCN窗口数据提取完成 - 汇总统计")
    print("=" * 60)
    print(f"总文件数:          {total_files}")
    print(f"有效文件数:        {success_files}")
    print(f"跳过文件数:        {skip_files}")
    print(f"总检查窗口数:      {total_checked_windows}")
    print(f"合格窗口数:        {total_valid_windows}")
    if total_checked_windows > 0:
        print(f"窗口合格率:        "
              f"{total_valid_windows/total_checked_windows:.1%}")
    print(f"\n数据形状:")
    print(f"  X_train: {X_train.shape}  "
          f"(样本数, 输入步数={INPUT_WINDOW}, "
          f"特征数={len(TCN_INPUT_COLS)})")
    print(f"  Y_train: {Y_train.shape}  "
          f"(样本数, 预测步数={PREDICT_HORIZON}, "
          f"目标数={len(TARGET_COLS)})")
    print(f"  X_val:   {X_val.shape}")
    print(f"  Y_val:   {Y_val.shape}")
    print(f"\n输入特征列: {TCN_INPUT_COLS}")
    print(f"目标列: {TARGET_COLS}")

    if all_reject_reasons:
        print(f"\n窗口拒绝原因统计:")
        for reason, count in sorted(
            all_reject_reasons.items(), key=lambda x: -x[1]
        ):
            pct = (count / total_checked_windows * 100
                   if total_checked_windows > 0 else 0)
            print(f"  {reason}: {count} ({pct:.1f}%)")

    total_size = (
        os.path.getsize(train_path) + os.path.getsize(val_path)
    ) / 1024 / 1024
    print(f"\n总文件大小: {total_size:.1f} MB")


if __name__ == "__main__":
    main()