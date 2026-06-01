"""
Isolation Forest + LOF 异常检测器
从 fusion_model_bundle.pkl 加载模型（与 train_IF_LOF_model.py 保存格式一致）
"""
import os
import numpy as np
import joblib
from config import MODEL_DIR, IF_LOF_FEATURE_COLS


class IFLOFDetector:
    """IF + LOF 融合异常检测器"""

    def __init__(self, model_dir=None):
        self.model_dir = model_dir or MODEL_DIR
        self.if_model = None
        self.lof_model = None
        self.scaler = None
        self.feature_cols = IF_LOF_FEATURE_COLS
        self.loaded = False
        self._load_models()

    def _load_models(self):
        """加载模型文件"""
        bundle_path = os.path.join(self.model_dir, "fusion_model_bundle.pkl")

        if os.path.exists(bundle_path):
            # 方式1：从 bundle 加载（train_IF_LOF_model.py 保存的格式）
            bundle = joblib.load(bundle_path)
            self.if_model = bundle["iforest"]
            self.lof_model = bundle["lof"]
            self.scaler = bundle["scaler"]
            if "feature_cols" in bundle:
                self.feature_cols = bundle["feature_cols"]
            self.loaded = True
            print("[IFLOFDetector] 从 fusion_model_bundle.pkl 加载成功")
        else:
            # 方式2：从独立文件加载（兼容旧格式）
            if_path = os.path.join(self.model_dir, "isolation_forest_model.pkl")
            lof_path = os.path.join(self.model_dir, "lof_model.pkl")
            scaler_path = os.path.join(self.model_dir, "if_lof_scaler.pkl")

            if os.path.exists(if_path) and os.path.exists(lof_path):
                self.if_model = joblib.load(if_path)
                self.lof_model = joblib.load(lof_path)
                if os.path.exists(scaler_path):
                    self.scaler = joblib.load(scaler_path)
                self.loaded = True
                print("[IFLOFDetector] 从独立文件加载成功")
            else:
                print("[IFLOFDetector] 警告：未找到模型文件，检测功能不可用")

    def detect(self, features_dict):
        """
        对单个数据点进行异常检测

        Args:
            features_dict: feature_engine.update() 返回的完整特征字典

        Returns:
            dict: {
                "is_anomaly": bool,
                "if_score": float,  # IF 异常分数
                "lof_score": float, # LOF 异常分数
                "if_label": int,    # IF 标签 (1=正常, -1=异常)
                "lof_label": int,   # LOF 标签 (1=正常, -1=异常)
                "fusion_label": int # 融合标签
            }
        """
        if not self.loaded:
            return {
                "is_anomaly": False,
                "if_score": 0.0,
                "lof_score": 0.0,
                "if_label": 1,
                "lof_label": 1,
                "fusion_label": 1,
            }

        # 提取特征向量
        try:
            feature_vector = np.array(
                [[features_dict[col] for col in self.feature_cols]]
            )
        except KeyError as e:
            print(f"[IFLOFDetector] 特征缺失: {e}")
            return {
                "is_anomaly": False,
                "if_score": 0.0,
                "lof_score": 0.0,
                "if_label": 1,
                "lof_label": 1,
                "fusion_label": 1,
            }

        # 标准化
        if self.scaler is not None:
            feature_vector = self.scaler.transform(feature_vector)

        # IF 检测
        if_label = int(self.if_model.predict(feature_vector)[0])
        if_score = float(self.if_model.decision_function(feature_vector)[0])

        # LOF 检测
        lof_label = int(self.lof_model.predict(feature_vector)[0])
        lof_score = float(self.lof_model.decision_function(feature_vector)[0])

        # 融合策略：两者都判定为异常才认为是异常（保守策略）
        fusion_label = -1 if (if_label == -1 and lof_label == -1) else 1
        is_anomaly = fusion_label == -1

        return {
            "is_anomaly": is_anomaly,
            "if_score": if_score,
            "lof_score": lof_score,
            "if_label": if_label,
            "lof_label": lof_label,
            "fusion_label": fusion_label,
        }