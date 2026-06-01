"""
TCN 趋势预测器
从 tcn_trend_bundle.pkl 加载模型（与 train_tcn_trend_model.py 保存格式一致）
输入序列窗口，输出未来多步预测及残差异常判定
"""
import os
import pickle
import collections
import numpy as np
import torch
from config import MODEL_DIR, TCN_FEATURE_COLS, TCN_INPUT_WINDOW
import torch.nn as nn


class CausalConv1d(nn.Module):
    """因果卷积"""
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=self.padding, dilation=dilation,
        )

    def forward(self, x):
        out = self.conv(x)
        if self.padding > 0:
            out = out[:, :, :-self.padding]
        return out


class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout):
        super().__init__()
        self.conv1 = CausalConv1d(in_channels, out_channels, kernel_size, dilation)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(dropout)

        self.conv2 = CausalConv1d(out_channels, out_channels, kernel_size, dilation)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(dropout)

        self.downsample = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels else None
        )
        self.relu_out = nn.ReLU()

    def forward(self, x):
        out = self.drop1(self.relu1(self.bn1(self.conv1(x))))
        out = self.drop2(self.relu2(self.bn2(self.conv2(out))))
        res = x if self.downsample is None else self.downsample(x)
        return self.relu_out(out + res)


class TCNEncoder(nn.Module):
    def __init__(self, input_size, num_channels, kernel_size, dropout):
        super().__init__()
        layers = []
        for i, out_ch in enumerate(num_channels):
            in_ch = input_size if i == 0 else num_channels[i - 1]
            dilation = 2 ** i
            layers.append(TemporalBlock(in_ch, out_ch, kernel_size, dilation, dropout))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class TCNTrendModel(nn.Module):
    def __init__(self, input_size, output_size, predict_horizon,
                 num_channels, kernel_size, dropout):
        super().__init__()
        self.predict_horizon = predict_horizon
        self.output_size = output_size

        self.encoder = TCNEncoder(input_size, num_channels, kernel_size, dropout)

        encoder_out_dim = num_channels[-1]
        self.predictor = nn.Sequential(
            nn.Linear(encoder_out_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, predict_horizon * output_size),
        )

    def forward(self, x):
        enc = self.encoder(x)
        last_step = enc[:, :, -1]
        flat = self.predictor(last_step)
        out = flat.view(-1, self.predict_horizon, self.output_size)
        return out


class TrendPredictor:
    """TCN 趋势预测与异常检测"""

    def __init__(self, model_dir="models"):
        self.model_dir = model_dir or MODEL_DIR
        self.model = None
        self.input_scaler = None
        self.target_scaler = None
        self.residual_stats = None
        self.config = {}
        self.loaded = False

        # 目标力值（由外部设置）
        self._target_force = 1000.0

        # 输入序列缓冲区
        self.seq_len = TCN_INPUT_WINDOW
        self.feature_cols = TCN_FEATURE_COLS
        self.buffer = collections.deque(maxlen=self.seq_len)

        # 用于残差计算的历史：存储每步的归一化目标值（norm_force_avg等）
        self._norm_target_buffer = collections.deque(maxlen=self.seq_len + 10)

        self._load_model()

    def _load_model(self):
        """加载 TCN 模型 bundle"""
        bundle_path = os.path.join(self.model_dir, "tcn_trend_bundle.pkl")

        if not os.path.exists(bundle_path):
            print("[TrendPredictor] 警告：未找到 tcn_trend_bundle.pkl，预测功能不可用")
            return

        try:
            with open(bundle_path, "rb") as f:
                bundle = pickle.load(f)

            # 提取配置
            self.seq_len = bundle.get("input_window", TCN_INPUT_WINDOW)
            input_size = bundle["input_size"]
            output_size = bundle["output_size"]
            predict_horizon = bundle["predict_horizon"]
            num_channels = bundle.get("num_channels", [64, 64, 32])
            kernel_size = bundle.get("kernel_size", 5)
            dropout = bundle.get("dropout", 0.15)

            # 重建模型
            self.model = TCNTrendModel(
                input_size=input_size,
                output_size=output_size,
                predict_horizon=predict_horizon,
                num_channels=num_channels,
                kernel_size=kernel_size,
                dropout=dropout,
            )
            self.model.load_state_dict(bundle["model_state_dict"])
            self.model.eval()

            # 加载 scaler 和残差统计
            self.input_scaler = bundle.get("input_scaler")
            self.target_scaler = bundle.get("target_scaler")
            self.residual_stats = bundle.get("residual_stats")

            # 加载阈值
            self._warning_threshold = bundle.get("warning_threshold", 0.05)
            self._alarm_threshold = bundle.get("alarm_threshold", 0.1)

            # 更新特征列（如果 bundle 中有保存）
            if "input_cols" in bundle:
                self.feature_cols = bundle["input_cols"]

            # 目标列
            self._target_cols = bundle.get("target_cols", [
                "norm_force_avg", "norm_force_diff", "norm_force_rate"
            ])

            # 更新缓冲区大小
            self.buffer = collections.deque(maxlen=self.seq_len)
            self._norm_target_buffer = collections.deque(maxlen=self.seq_len + 10)

            self.config = bundle
            self.loaded = True
            print(f"[TrendPredictor] 模型加载成功，输入窗口={self.seq_len}，"
                  f"特征维度={input_size}，预测步数={predict_horizon}")
            print(f"[TrendPredictor] warning阈值={self._warning_threshold:.4f}, "
                  f"alarm阈值={self._alarm_threshold:.4f}")

        except Exception as e:
            print(f"[TrendPredictor] 模型加载失败: {e}")
            import traceback
            traceback.print_exc()

    def feed_point(self, features_dict):
        """
        将 feature_engine 输出映射为 TCN 训练时的归一化特征，
        并同时记录该点对应的归一化目标值（用于残差计算）。
        """
        if not self.loaded:
            return

        try:
            # 获取目标力值（用于归一化）
            tf = features_dict.get("_target_force", self._target_force)
            if tf <= 0:
                tf = 1000.0

            force_left = features_dict["force_left"]
            force_right = features_dict["force_right"]
            force_avg = features_dict["force_avg"]
            force_diff = features_dict["force_diff"]
            force_rate = features_dict["force_rate"]
            left_force_rate = features_dict["left_force_rate"]
            right_force_rate = features_dict["right_force_rate"]
            force_rate_diff = features_dict["force_rate_diff"]
            force_acc = features_dict["force_acc"]
            dis_left = features_dict["dis_left"]
            dis_right = features_dict["dis_right"]
            total_delta_dis = features_dict["total_delta_dis"]
            dis_diff = features_dict["dis_diff"]
            dis_rate = features_dict["dis_rate"]
            force_disp_ratio = features_dict["force_disp_ratio"]
            stiffness_ratio = features_dict["stiffness_ratio"]
            force_std_5s = features_dict["force_std_5s"]

            # 构建与 TCN_DATA.py 中 TCN_INPUT_COLS 完全一致的特征向量
            feature_vector = [
                force_left / tf,                              # norm_force_left
                force_right / tf,                             # norm_force_right
                force_avg / tf,                               # norm_force_avg
                force_diff / tf,                              # norm_force_diff
                force_rate / tf,                              # norm_force_rate
                left_force_rate / tf,                         # norm_left_force_rate
                right_force_rate / tf,                        # norm_right_force_rate
                force_rate_diff / tf,                         # norm_force_rate_diff
                force_acc / tf,                               # norm_force_acc
                dis_left,                                     # dis_left
                dis_right,                                    # dis_right
                total_delta_dis,                              # total_delta_dis
                dis_diff,                                     # dis_diff
                dis_rate,                                     # dis_rate
                max(-10.0, min(10.0, force_disp_ratio)),      # force_disp_ratio (clipped)
                (force_left - force_right) / tf,              # left_right_force_diff
                max(-10.0, min(10.0, stiffness_ratio / 100.0)),  # stiffness_ratio_norm
                force_std_5s / tf,                            # force_std_5s_norm
            ]

            self.buffer.append(feature_vector)

            # 同时记录该点的归一化目标值（与 TARGET_COLS 对应）
            # TARGET_COLS = ['norm_force_avg', 'norm_force_diff', 'norm_force_rate']
            norm_target = [
                force_avg / tf,       # norm_force_avg
                force_diff / tf,      # norm_force_diff
                force_rate / tf,      # norm_force_rate
            ]
            self._norm_target_buffer.append(norm_target)

        except KeyError as e:
            print(f"[TrendPredictor] 特征缺失: {e}")

    def predict(self):
        """
        使用当前缓冲区数据进行预测。

        Returns:
            dict or None: {
                "predicted_force": list,   # 未来多步预测的真实力值 (kN)
                "is_anomaly": bool,
                "residual": float,         # 归一化空间的残差
                "confidence": float,
            }
            缓冲区未满时返回 None
        """
        if not self.loaded:
            return None

        if len(self.buffer) < self.seq_len:
            return None

        # 构建输入张量: (seq_len, n_features)
        input_array = np.array(list(self.buffer), dtype=np.float32)

        # 用 input_scaler 做 StandardScaler 标准化
        if self.input_scaler is not None:
            input_array = self.input_scaler.transform(input_array)

        # 转为 (1, n_features, seq_len) — TCN 要求 channels first
        input_tensor = torch.FloatTensor(input_array.T).unsqueeze(0)

        # 推理
        with torch.no_grad():
            output = self.model(input_tensor)  # (1, predict_horizon, output_size)

        # output 是在 target_scaler 标准化空间中的预测
        prediction_scaled = output.squeeze(0).numpy()  # (predict_horizon, output_size)

        # 用 target_scaler 反标准化 → 得到归一化空间的值（norm_force_avg 等，范围约 0~1）
        if self.target_scaler is not None:
            prediction_norm = self.target_scaler.inverse_transform(prediction_scaled)
        else:
            prediction_norm = prediction_scaled

        # 将归一化值转换为真实力值 (kN)
        # prediction_norm[:, 0] 对应 norm_force_avg = force_avg / target_force
        tf = self._target_force if self._target_force > 0 else 1000.0
        predicted_force_kn = (prediction_norm[:, 0] * tf).tolist()

        # ============ 残差异常判定 ============
        is_anomaly = False
        residual = 0.0
        confidence = 0.0

        if hasattr(self, "_last_prediction_norm") and self._last_prediction_norm is not None:
            # 当前实际的归一化目标值
            if len(self._norm_target_buffer) > 0:
                actual_norm = np.array(self._norm_target_buffer[-1])  # (n_target,)
                # 上一次预测的第一步（归一化空间）
                expected_norm = self._last_prediction_norm[0]  # (n_target,)

                # 计算残差（在归一化空间中，这样与训练时的残差统计可比）
                residual = float(np.mean(np.abs(actual_norm - expected_norm)))

                # 基于 bundle 中保存的阈值判定
                if residual > self._alarm_threshold:
                    is_anomaly = True
                    confidence = min(residual / self._alarm_threshold, 3.0) / 3.0
                elif residual > self._warning_threshold:
                    is_anomaly = True
                    confidence = 0.5 * min(residual / self._alarm_threshold, 1.0)

        # 保存本次预测的归一化结果供下次比较
        self._last_prediction_norm = prediction_norm.copy()

        return {
            "predicted_force": predicted_force_kn,
            "is_anomaly": is_anomaly,
            "residual": residual,
            "confidence": confidence,
        }

    def reset(self):
        """重置缓冲区"""
        self.buffer.clear()
        self._norm_target_buffer.clear()
        self._last_prediction_norm = None