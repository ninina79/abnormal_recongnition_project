# train_IF_LOF_model.py
#
# Isolation Forest + LOF 辅助异常评分模型训练
#
# 学习正常张拉模式，辅助判断当前窗口特征是否偏离正常分布
#
# 模型输出：
#   model_status: "normal" | "assist_warning" | "assist_alarm"
#   model_score: 0.0 ~ 1.0
#
# 训练流程不变：
#   1. 读取 rule_engine_classified 输出的 1_normal.csv
#   2. 提取 FEATURE_COLS 特征
#   3. 对极端值做 clip 处理
#   4. StandardScaler 标准化
#   5. 训练 Isolation Forest
#   6. 训练 LOF
#   7. 在训练集上计算原始分数并保存归一化分位数
#   8. 计算 fusion_score 并确定 warning / alarm 阈值
#   9. 保存 model_bundle.pkl

import os
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor


# =========================================================
# 1. 路径配置
# =========================================================
CLASSIFIED_DIR = r"E:\ABNORMAL_RECOGNITION\DATA\processed\if_lof_training_data"
NORMAL_CSV = os.path.join(CLASSIFIED_DIR, "1_normal", "1_normal.csv")
WARNING_CSV = os.path.join(CLASSIFIED_DIR, "2_warning_recoverable", "2_warning_recoverable.csv")
ALARM_CSV = os.path.join(CLASSIFIED_DIR, "3_alarm", "3_alarm.csv")

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)


# =========================================================
# 2. 训练参数
# =========================================================
RANDOM_STATE = 42

# Isolation Forest 参数
IFOREST_N_ESTIMATORS = 300
IFOREST_CONTAMINATION = 0.01    # 训练数据是 normal，contamination 设小

# LOF 参数
LOF_CONTAMINATION = 0.01
LOF_N_NEIGHBORS_RATIO = 0.005   # 邻居数 = 样本数 * 0.005，最小 10，最大 50

# 融合权重
FUSION_WEIGHT_IFOREST = 0.5
FUSION_WEIGHT_LOF = 0.5

# 阈值分位数
WARNING_QUANTILE = 0.95         # 训练集 95% 分位数作为 warning 阈值
ALARM_QUANTILE = 0.99           # 训练集 99% 分位数作为 alarm 阈值

# 极端值 clip 范围
CLIP_CONFIGS = {
    "force_rate_ratio": (-10.0, 10.0),
    "stiffness_ratio": (-1000.0, 1000.0),
    "force_diff_ratio": (0.0, 5.0),
}


# =========================================================
# 3. 模型训练用的特征列
# =========================================================
# 注意：训练与推理均使用原始特征（非 norm_ 前缀的归一化特征）。
# StandardScaler 在训练时拟合，推理时通过 model_bundle["scaler"] 统一变换。
# 若修改此列表，务必同步检查 model_detector.py 的推理逻辑。
FEATURE_COLS = [
    "force_left",
    "force_right",
    "force_avg",
    "force_diff",
    "force_diff_ratio",

    "left_force_rate",
    "right_force_rate",
    "force_rate",
    "force_rate_diff",
    "force_rate_ratio",

    "left_force_acc",
    "right_force_acc",
    "force_acc",

    "total_delta_dis",
    "dis_diff",
    "left_dis_rate",
    "right_dis_rate",
    "dis_rate_diff",

    "stiffness_ratio",
    "force_std_5s",
    "force_diff_std_5s",
    "force_rate_std_5s",
]


# =========================================================
# 4. 数据加载
# =========================================================
def load_training_data():
    """
    读取 1_normal.csv 作为训练数据。
    """
    if not os.path.exists(NORMAL_CSV):
        raise FileNotFoundError(
            f"找不到训练数据: {NORMAL_CSV}\n"
            f"请先运行 rule_engine_classified.py 生成训练数据。"
        )

    df = pd.read_csv(NORMAL_CSV, encoding="utf-8-sig")

    if df.empty:
        raise ValueError("训练数据为空")

    # 检查特征列
    missing_cols = [col for col in FEATURE_COLS if col not in df.columns]

    if missing_cols:
        raise ValueError(f"训练数据缺少特征列: {missing_cols}")

    print(f"训练数据加载完成: {len(df)} 个 normal 窗口")

    return df


def load_test_data():
    """
    读取 warning 和 alarm 数据用于测试评估。
    """
    test_dfs = []

    if os.path.exists(WARNING_CSV):
        warning_df = pd.read_csv(WARNING_CSV, encoding="utf-8-sig")

        if not warning_df.empty:
            # 校验测试数据是否包含全部特征列
            missing = [col for col in FEATURE_COLS if col not in warning_df.columns]
            if missing:
                print(f"  [警告] warning 测试数据缺少特征列: {missing}，跳过该数据集")
            else:
                test_dfs.append(("warning", warning_df))
                print(f"warning 测试数据: {len(warning_df)} 个窗口")

    if os.path.exists(ALARM_CSV):
        alarm_df = pd.read_csv(ALARM_CSV, encoding="utf-8-sig")

        if not alarm_df.empty:
            missing = [col for col in FEATURE_COLS if col not in alarm_df.columns]
            if missing:
                print(f"  [警告] alarm 测试数据缺少特征列: {missing}，跳过该数据集")
            else:
                test_dfs.append(("alarm", alarm_df))
                print(f"alarm 测试数据: {len(alarm_df)} 个窗口")

    return test_dfs


# =========================================================
# 5. 数据预处理
# =========================================================
def preprocess_features(df, feature_cols, clip_configs=None):
    """
    提取特征列，处理 inf / nan / 极端值。
    """
    X = df[feature_cols].copy()

    # 替换 inf
    X = X.replace([np.inf, -np.inf], np.nan)

    # 填充 nan
    X = X.fillna(0)

    # clip 极端值
    if clip_configs:
        for col, (low, high) in clip_configs.items():
            if col in X.columns:
                X[col] = X[col].clip(low, high)

    return X


# =========================================================
# 6. 分数归一化（训练和预测统一使用）
# =========================================================
def compute_score_stats(raw_scores):
    """
    计算分数的 1% 和 99% 分位数，用于归一化。
    """
    q01 = float(np.percentile(raw_scores, 1))
    q99 = float(np.percentile(raw_scores, 99))

    return {"q01": q01, "q99": q99}


def normalize_scores_batch(raw_scores, stats):
    """
    批量归一化。
    """
    q01 = stats["q01"]
    q99 = stats["q99"]

    if abs(q99 - q01) < 1e-12:
        return np.zeros_like(raw_scores)

    scores = (raw_scores - q01) / (q99 - q01)

    return np.clip(scores, 0.0, 1.0)


# =========================================================
# 7. 模型训练
# =========================================================
def train_isolation_forest(X_scaled):
    """
    训练 Isolation Forest。
    """
    model = IsolationForest(
        n_estimators=IFOREST_N_ESTIMATORS,
        contamination=IFOREST_CONTAMINATION,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    model.fit(X_scaled)

    # decision_function 越小越异常，取负号后越大越异常
    raw_scores = -model.decision_function(X_scaled)

    stats = compute_score_stats(raw_scores)
    norm_scores = normalize_scores_batch(raw_scores, stats)

    print(f"  Isolation Forest 训练完成")
    print(f"    原始分数范围: [{raw_scores.min():.4f}, {raw_scores.max():.4f}]")
    print(f"    归一化 q01={stats['q01']:.4f}, q99={stats['q99']:.4f}")
    print(f"    归一化分数范围: [{norm_scores.min():.4f}, {norm_scores.max():.4f}]")

    return model, raw_scores, norm_scores, stats


def train_lof(X_scaled):
    """
    训练 Local Outlier Factor。
    """
    n_samples = X_scaled.shape[0]

    # 动态计算邻居数
    n_neighbors = int(n_samples * LOF_N_NEIGHBORS_RATIO)
    n_neighbors = max(10, min(50, n_neighbors))

    print(f"  LOF 邻居数: {n_neighbors}（样本数 {n_samples}）")

    model = LocalOutlierFactor(
        n_neighbors=n_neighbors,
        contamination=LOF_CONTAMINATION,
        novelty=True,
    )

    model.fit(X_scaled)

    raw_scores = -model.decision_function(X_scaled)

    stats = compute_score_stats(raw_scores)
    norm_scores = normalize_scores_batch(raw_scores, stats)

    print(f"  LOF 训练完成")
    print(f"    原始分数范围: [{raw_scores.min():.4f}, {raw_scores.max():.4f}]")
    print(f"    归一化 q01={stats['q01']:.4f}, q99={stats['q99']:.4f}")
    print(f"    归一化分数范围: [{norm_scores.min():.4f}, {norm_scores.max():.4f}]")

    return model, raw_scores, norm_scores, stats


# =========================================================
# 8. 融合分数与阈值
# =========================================================
def compute_fusion_scores(iforest_norm, lof_norm):
    """
    计算融合分数。
    """
    fusion = (
        FUSION_WEIGHT_IFOREST * iforest_norm
        + FUSION_WEIGHT_LOF * lof_norm
    )

    return fusion


def compute_thresholds(fusion_scores):
    """
    用训练集的融合分数确定 warning 和 alarm 阈值。
    """
    warning_threshold = float(np.percentile(fusion_scores, WARNING_QUANTILE * 100))
    alarm_threshold = float(np.percentile(fusion_scores, ALARM_QUANTILE * 100))

    # 确保 alarm > warning
    if alarm_threshold <= warning_threshold:
        alarm_threshold = warning_threshold + 0.01

    return warning_threshold, alarm_threshold


# =========================================================
# 9. 测试评估
# =========================================================
def evaluate_on_test(model_bundle, test_datasets):
    """
    在 warning 和 alarm 数据上评估模型表现。
    """
    if not test_datasets:
        print("没有测试数据，跳过评估。")
        return {}

    scaler = model_bundle["scaler"]
    iforest = model_bundle["iforest"]
    lof = model_bundle["lof"]
    feature_cols = model_bundle["feature_cols"]
    iforest_stats = model_bundle["score_stats"]["iforest"]
    lof_stats = model_bundle["score_stats"]["lof"]
    warning_threshold = model_bundle["warning_threshold"]
    alarm_threshold = model_bundle["alarm_threshold"]

    results = {}

    for label, df in test_datasets:
        X = preprocess_features(df, feature_cols, CLIP_CONFIGS)
        X_scaled = scaler.transform(X)

        raw_if = -iforest.decision_function(X_scaled)
        raw_lof = -lof.decision_function(X_scaled)

        norm_if = normalize_scores_batch(raw_if, iforest_stats)
        norm_lof = normalize_scores_batch(raw_lof, lof_stats)

        fusion = compute_fusion_scores(norm_if, norm_lof)

        n_total = len(fusion)
        n_alarm = int(np.sum(fusion >= alarm_threshold))
        n_warning = int(np.sum(
            (fusion >= warning_threshold) & (fusion < alarm_threshold)
        ))
        n_normal = int(np.sum(fusion < warning_threshold))

        detection_rate = (n_alarm + n_warning) / n_total if n_total > 0 else 0.0

        results[label] = {
            "total": n_total,
            "detected_alarm": n_alarm,
            "detected_warning": n_warning,
            "detected_normal": n_normal,
            "detection_rate": detection_rate,
            "fusion_mean": float(fusion.mean()),
            "fusion_max": float(fusion.max()),
            "fusion_min": float(fusion.min()),
        }

        print(f"\n  [{label}] 测试结果:")
        print(f"    总窗口数: {n_total}")
        print(f"    检出为 alarm: {n_alarm}")
        print(f"    检出为 warning: {n_warning}")
        print(f"    检出为 normal: {n_normal}")
        print(f"    检出率: {detection_rate * 100:.1f}%")
        print(f"    融合分数: mean={fusion.mean():.4f}, max={fusion.max():.4f}")

    return results


# =========================================================
# 10. 主训练流程
# =========================================================
def train_fusion_model():
    """
    完整训练流程。
    """
    print("=" * 70)
    print("融合异常检测模型训练")
    print("=" * 70)
    print(f"训练数据: {NORMAL_CSV}")
    print(f"模型保存: {MODEL_DIR}")
    print(f"特征数量: {len(FEATURE_COLS)}")
    print(f"融合权重: IF={FUSION_WEIGHT_IFOREST}, LOF={FUSION_WEIGHT_LOF}")
    print(f"阈值分位: warning={WARNING_QUANTILE}, alarm={ALARM_QUANTILE}")
    print()

    # =====================================================
    # 1. 加载数据
    # =====================================================
    print("-" * 70)
    print("1. 加载训练数据")
    print("-" * 70)

    train_df = load_training_data()
    test_datasets = load_test_data()

    print()

    # =====================================================
    # 2. 预处理
    # =====================================================
    print("-" * 70)
    print("2. 特征预处理")
    print("-" * 70)

    X = preprocess_features(train_df, FEATURE_COLS, CLIP_CONFIGS)

    print(f"  训练样本数: {X.shape[0]}")
    print(f"  特征维度: {X.shape[1]}")

    # 检查数据质量
    zero_cols = []

    for col in FEATURE_COLS:
        if X[col].std() < 1e-10:
            zero_cols.append(col)

    if zero_cols:
        print(f"  [警告] 以下特征方差接近 0，可能无效: {zero_cols}")
        print(f"  建议检查数据或从 FEATURE_COLS 中移除。")

    print()

    # =====================================================
    # 3. 标准化
    # =====================================================
    print("-" * 70)
    print("3. 标准化")
    print("-" * 70)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print(f"  StandardScaler 拟合完成")
    print()

    # =====================================================
    # 4. 训练模型
    # =====================================================
    print("-" * 70)
    print("4. 训练模型")
    print("-" * 70)

    iforest, if_raw, if_norm, if_stats = train_isolation_forest(X_scaled)
    lof, lof_raw, lof_norm, lof_stats = train_lof(X_scaled)

    print()

    # =====================================================
    # 5. 计算融合分数和阈值
    # =====================================================
    print("-" * 70)
    print("5. 融合分数与阈值")
    print("-" * 70)

    fusion_scores = compute_fusion_scores(if_norm, lof_norm)
    warning_threshold, alarm_threshold = compute_thresholds(fusion_scores)

    print(f"  融合分数范围: [{fusion_scores.min():.4f}, {fusion_scores.max():.4f}]")
    print(f"  融合分数均值: {fusion_scores.mean():.4f}")
    print(f"  融合分数标准差: {fusion_scores.std():.4f}")
    print(f"  warning 阈值 (p{WARNING_QUANTILE * 100:.0f}): {warning_threshold:.4f}")
    print(f"  alarm 阈值 (p{ALARM_QUANTILE * 100:.0f}): {alarm_threshold:.4f}")

    # 训练集上的分布
    n_train = len(fusion_scores)
    n_train_alarm = int(np.sum(fusion_scores >= alarm_threshold))
    n_train_warning = int(np.sum(
        (fusion_scores >= warning_threshold) & (fusion_scores < alarm_threshold)
    ))
    n_train_normal = n_train - n_train_alarm - n_train_warning

    print(f"  训练集分布: normal={n_train_normal}, warning={n_train_warning}, alarm={n_train_alarm}")
    print()

    # =====================================================
    # 6. 组装报告与模型（先不保存，等评估完成后统一写入）
    # =====================================================
    print("-" * 70)
    print("6. 组装模型与报告")
    print("-" * 70)

    model_bundle = {
        "iforest": iforest,
        "lof": lof,
        "scaler": scaler,
        "feature_cols": FEATURE_COLS,
        "score_stats": {
            "iforest": if_stats,
            "lof": lof_stats,
        },
        "fusion_weights": {
            "iforest": FUSION_WEIGHT_IFOREST,
            "lof": FUSION_WEIGHT_LOF,
        },
        "warning_threshold": warning_threshold,
        "alarm_threshold": alarm_threshold,
        "training_info": {
            "n_samples": int(n_train),
            "n_features": len(FEATURE_COLS),
            "iforest_contamination": IFOREST_CONTAMINATION,
            "lof_contamination": LOF_CONTAMINATION,
            "warning_quantile": WARNING_QUANTILE,
            "alarm_quantile": ALARM_QUANTILE,
            "clip_configs": CLIP_CONFIGS,
        },
    }

    report = {
        "model_path": os.path.join(MODEL_DIR, "fusion_model_bundle.pkl"),
        "training_data": NORMAL_CSV,
        "n_training_samples": int(n_train),
        "n_features": len(FEATURE_COLS),
        "feature_cols": FEATURE_COLS,
        "iforest_params": {
            "n_estimators": IFOREST_N_ESTIMATORS,
            "contamination": IFOREST_CONTAMINATION,
        },
        "lof_params": {
            "contamination": LOF_CONTAMINATION,
            "n_neighbors_ratio": LOF_N_NEIGHBORS_RATIO,
        },
        "fusion_weights": {
            "iforest": FUSION_WEIGHT_IFOREST,
            "lof": FUSION_WEIGHT_LOF,
        },
        "score_stats": {
            "iforest": if_stats,
            "lof": lof_stats,
        },
        "warning_threshold": warning_threshold,
        "alarm_threshold": alarm_threshold,
        "training_distribution": {
            "normal": n_train_normal,
            "warning": n_train_warning,
            "alarm": n_train_alarm,
        },
        "fusion_score_stats": {
            "mean": float(fusion_scores.mean()),
            "std": float(fusion_scores.std()),
            "min": float(fusion_scores.min()),
            "max": float(fusion_scores.max()),
            "p50": float(np.percentile(fusion_scores, 50)),
            "p90": float(np.percentile(fusion_scores, 90)),
            "p95": float(np.percentile(fusion_scores, 95)),
            "p99": float(np.percentile(fusion_scores, 99)),
        },
    }

    model_path = os.path.join(MODEL_DIR, "fusion_model_bundle.pkl")
    joblib.dump(model_bundle, model_path)
    print(f"  模型已保存: {model_path}")

    # =====================================================
    # 7. 测试评估（如有测试数据，结果直接并入 report）
    # =====================================================
    if test_datasets:
        print("-" * 70)
        print("7. 测试评估")
        print("-" * 70)

        eval_results = evaluate_on_test(model_bundle, test_datasets)
        report["evaluation"] = eval_results

        print(f"\n  评估结果已并入报告")

    # =====================================================
    # 8. 统一保存训练报告（仅写入一次）
    # =====================================================
    report_path = os.path.join(MODEL_DIR, "training_report.json")

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"  训练报告: {report_path}")
    print()

    # =====================================================
    # 9. 保存训练集分数（可选，用于分析）
    # =====================================================
    score_df = train_df.copy()
    score_df["iforest_raw"] = if_raw
    score_df["lof_raw"] = lof_raw
    score_df["iforest_norm"] = if_norm
    score_df["lof_norm"] = lof_norm
    score_df["fusion_score"] = fusion_scores

    score_df["predicted_status"] = "normal"
    score_df.loc[
        score_df["fusion_score"] >= warning_threshold, "predicted_status"
    ] = "unknown_warning"
    score_df.loc[
        score_df["fusion_score"] >= alarm_threshold, "predicted_status"
    ] = "unknown_alarm"

    score_path = os.path.join(MODEL_DIR, "training_scores.csv")
    score_df.to_csv(score_path, index=False, encoding="utf-8-sig")

    print()
    print("-" * 70)
    print("训练完成")
    print("-" * 70)
    print(f"  模型文件:   {model_path}")
    print(f"  训练报告:   {report_path}")
    print(f"  训练集分数: {score_path}")
    print(f"  warning 阈值: {warning_threshold:.4f}")
    print(f"  alarm 阈值:   {alarm_threshold:.4f}")

    return model_bundle, report


# =========================================================
# 11. 主入口
# =========================================================
if __name__ == "__main__":
    train_fusion_model()