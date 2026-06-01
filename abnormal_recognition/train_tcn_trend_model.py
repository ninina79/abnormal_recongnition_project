"""
train_tcn_trend_model.py
功能：训练 TCN 滑动窗口趋势预测模型
      输入: 过去 input_window 个时间步的特征
      输出: 未来 predict_horizon 个时间步的目标值

保存: tcn_trend_bundle.pkl（模型权重 + scaler + 阈值 + 配置）
"""

import os
import json
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler


# =========================================================
# 1. 配置
# =========================================================
TCN_DATA_DIR = r"E:\ABNORMAL_RECOGNITION\DATA\processed\tcn_training_data"
MODEL_OUTPUT_DIR = "models"

# 训练超参数
BATCH_SIZE = 256
EPOCHS = 80
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
PATIENCE = 12          # 早停耐心
SCHEDULER_PATIENCE = 5  # 学习率衰减耐心

# TCN 网络参数
NUM_CHANNELS = [64, 64, 32]   # 每层的通道数
KERNEL_SIZE = 5
DROPOUT = 0.15

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================================================
# 2. TCN 基础模块
# =========================================================
class CausalConv1d(nn.Module):
    """因果卷积：只看过去，不看未来"""
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
    """TCN 编码器：将输入序列编码为固定维度向量"""
    def __init__(self, input_size, num_channels, kernel_size, dropout):
        super().__init__()
        layers = []
        for i, out_ch in enumerate(num_channels):
            in_ch = input_size if i == 0 else num_channels[i - 1]
            dilation = 2 ** i
            layers.append(TemporalBlock(in_ch, out_ch, kernel_size, dilation, dropout))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """
        x: (batch, input_size, seq_len)
        return: (batch, num_channels[-1], seq_len)
        """
        return self.network(x)


class TCNTrendModel(nn.Module):
    """
    TCN 趋势预测模型
    输入: (batch, input_size, input_window)
    输出: (batch, predict_horizon, output_size)
    """
    def __init__(self, input_size, output_size, predict_horizon,
                 num_channels, kernel_size, dropout):
        super().__init__()
        self.predict_horizon = predict_horizon
        self.output_size = output_size

        self.encoder = TCNEncoder(input_size, num_channels, kernel_size, dropout)

        # 取 TCN 最后一个时间步的输出，通过 MLP 预测未来
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
        """
        x: (batch, input_size, input_window)
        return: (batch, predict_horizon, output_size)
        """
        enc = self.encoder(x)              # (batch, channels, input_window)
        last_step = enc[:, :, -1]          # (batch, channels) — 取最后时间步
        flat = self.predictor(last_step)   # (batch, predict_horizon * output_size)
        out = flat.view(-1, self.predict_horizon, self.output_size)
        return out


# =========================================================
# 3. 数据加载与预处理
# =========================================================
def load_data():
    """加载 TCN_DATA.py 生成的窗口数据"""
    train_path = os.path.join(TCN_DATA_DIR, "tcn_train_windows.npz")
    val_path = os.path.join(TCN_DATA_DIR, "tcn_val_windows.npz")
    config_path = os.path.join(TCN_DATA_DIR, "tcn_data_config.json")

    for p in [train_path, val_path, config_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"文件不存在: {p}")

    with open(config_path, "r", encoding="utf-8") as f:
        data_config = json.load(f)

    train_data = np.load(train_path)
    val_data = np.load(val_path)

    X_train = train_data["X"]  # (N_train, input_window, n_input)
    Y_train = train_data["Y"]  # (N_train, predict_horizon, n_target)
    X_val = val_data["X"]
    Y_val = val_data["Y"]

    print(f"训练集: X={X_train.shape}, Y={Y_train.shape}")
    print(f"验证集: X={X_val.shape}, Y={Y_val.shape}")

    return X_train, Y_train, X_val, Y_val, data_config


def fit_scalers(X_train, Y_train):
    """
    对输入和目标分别拟合 StandardScaler。
    将 3D 数据展平为 2D 拟合，再还原。
    """
    N, T_in, D_in = X_train.shape
    N2, T_out, D_out = Y_train.shape

    input_scaler = StandardScaler()
    target_scaler = StandardScaler()

    # 展平: (N * T, D)
    X_flat = X_train.reshape(-1, D_in)
    Y_flat = Y_train.reshape(-1, D_out)

    input_scaler.fit(X_flat)
    target_scaler.fit(Y_flat)

    return input_scaler, target_scaler


def apply_scaler_3d(data, scaler):
    """对 3D 数据 (N, T, D) 应用 scaler"""
    N, T, D = data.shape
    flat = data.reshape(-1, D)
    scaled = scaler.transform(flat)
    return scaled.reshape(N, T, D).astype(np.float32)


def inverse_scaler_3d(data, scaler):
    """对 3D 数据 (N, T, D) 反向 scaler"""
    N, T, D = data.shape
    flat = data.reshape(-1, D)
    inv = scaler.inverse_transform(flat)
    return inv.reshape(N, T, D).astype(np.float32)


# =========================================================
# 4. 训练循环
# =========================================================
def train_model(model, train_loader, val_loader, target_scaler):
    """训练模型，返回训练历史"""
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=SCHEDULER_PATIENCE, factor=0.5, verbose=True
    )
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(1, EPOCHS + 1):
        # --- 训练 ---
        model.train()
        train_losses = []
        for X_batch, Y_batch in train_loader:
            X_batch = X_batch.to(DEVICE)
            Y_batch = Y_batch.to(DEVICE)

            # TCN 输入格式: (batch, features, seq_len)
            X_in = X_batch.permute(0, 2, 1)

            pred = model(X_in)
            loss = criterion(pred, Y_batch)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_losses.append(loss.item())

        avg_train = np.mean(train_losses)

        # --- 验证 ---
        model.eval()
        val_losses = []
        with torch.no_grad():
            for X_batch, Y_batch in val_loader:
                X_batch = X_batch.to(DEVICE)
                Y_batch = Y_batch.to(DEVICE)
                X_in = X_batch.permute(0, 2, 1)
                pred = model(X_in)
                loss = criterion(pred, Y_batch)
                val_losses.append(loss.item())

        avg_val = np.mean(val_losses)
        scheduler.step(avg_val)

        history["train_loss"].append(avg_train)
        history["val_loss"].append(avg_val)

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{EPOCHS} | "
                  f"Train Loss: {avg_train:.6f} | Val Loss: {avg_val:.6f}")

        # 早停
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"\n早停于 Epoch {epoch}，最佳验证损失: {best_val_loss:.6f}")
                break

    model.load_state_dict(best_state)
    return model, history, best_val_loss


# =========================================================
# 5. 计算残差统计与阈值
# =========================================================
def compute_residual_stats(model, val_loader, target_scaler):
    """
    在验证集上计算 1-step-ahead 残差统计，用于推理时的异常判定。
    这里用所有预测步的残差来估计分布。
    """
    model.eval()
    all_residuals = []

    with torch.no_grad():
        for X_batch, Y_batch in val_loader:
            X_batch = X_batch.to(DEVICE)
            X_in = X_batch.permute(0, 2, 1)
            pred = model(X_in).cpu().numpy()       # (batch, horizon, n_target)
            true = Y_batch.numpy()                  # (batch, horizon, n_target)

            # 反标准化
            pred_raw = inverse_scaler_3d(pred, target_scaler)
            true_raw = inverse_scaler_3d(true, target_scaler)

            residuals = np.abs(pred_raw - true_raw)  # (batch, horizon, n_target)
            # 取每个样本所有步的平均残差
            mean_res = residuals.mean(axis=1)         # (batch, n_target)
            all_residuals.append(mean_res)

    all_residuals = np.concatenate(all_residuals, axis=0)  # (N_val, n_target)

    stats = {}
    for i in range(all_residuals.shape[1]):
        col_res = all_residuals[:, i]
        stats[f"target_{i}"] = {
            "mean": float(np.mean(col_res)),
            "std": float(np.std(col_res)),
            "p90": float(np.percentile(col_res, 90)),
            "p95": float(np.percentile(col_res, 95)),
            "p99": float(np.percentile(col_res, 99)),
        }

    # 综合残差（所有目标列的平均）
    combined = all_residuals.mean(axis=1)
    stats["combined"] = {
        "mean": float(np.mean(combined)),
        "std": float(np.std(combined)),
        "p90": float(np.percentile(combined, 90)),
        "p95": float(np.percentile(combined, 95)),
        "p99": float(np.percentile(combined, 99)),
    }

    # 阈值：warning 用 p90，alarm 用 p99
    warning_threshold = float(stats["combined"]["p90"])
    alarm_threshold = float(stats["combined"]["p99"])

    return stats, warning_threshold, alarm_threshold


# =========================================================
# 6. 保存 bundle
# =========================================================
def save_bundle(model, input_scaler, target_scaler, data_config,
                residual_stats, warning_threshold, alarm_threshold,
                history, best_val_loss):
    """将所有推理需要的东西打包成一个 pkl"""
    os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)

    bundle = {
        # 模型结构参数
        "input_size": data_config["n_input_features"],
        "output_size": data_config["n_target_features"],
        "input_window": data_config["input_window"],
        "predict_horizon": data_config["predict_horizon"],
        "num_channels": NUM_CHANNELS,
        "kernel_size": KERNEL_SIZE,
        "dropout": DROPOUT,

        # 特征列名
        "input_cols": data_config["input_cols"],
        "target_cols": data_config["target_cols"],

        # 模型权重
        "model_state_dict": {
            k: v.cpu() for k, v in model.state_dict().items()
        },

        # Scaler
        "input_scaler": input_scaler,
        "target_scaler": target_scaler,

        # 残差统计与阈值
        "residual_stats": residual_stats,
        "warning_threshold": warning_threshold,
        "alarm_threshold": alarm_threshold,

        # 训练信息
        "best_val_loss": float(best_val_loss),
        "train_history": history,
        "train_config": {
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "patience": PATIENCE,
            "device": str(DEVICE),
        },
    }

    bundle_path = os.path.join(MODEL_OUTPUT_DIR, "tcn_trend_bundle.pkl")
    with open(bundle_path, "wb") as f:
        pickle.dump(bundle, f)

    print(f"\nBundle 已保存: {bundle_path}")
    print(f"  输入维度: {bundle['input_size']}")
    print(f"  输出维度: {bundle['output_size']}")
    print(f"  输入窗口: {bundle['input_window']}")
    print(f"  预测步数: {bundle['predict_horizon']}")
    print(f"  Warning 阈值: {warning_threshold:.6f}")
    print(f"  Alarm 阈值: {alarm_threshold:.6f}")
    print(f"  最佳验证损失: {best_val_loss:.6f}")

    return bundle_path


# =========================================================
# 7. 主流程
# =========================================================
def main():
    print("=" * 60)
    print("TCN 趋势预测模型训练")
    print(f"设备: {DEVICE}")
    print("=" * 60)

    # 1. 加载数据
    print("\n[1/6] 加载数据...")
    X_train, Y_train, X_val, Y_val, data_config = load_data()

    # 2. 拟合 Scaler
    print("\n[2/6] 拟合 StandardScaler...")
    input_scaler, target_scaler = fit_scalers(X_train, Y_train)

    X_train_s = apply_scaler_3d(X_train, input_scaler)
    Y_train_s = apply_scaler_3d(Y_train, target_scaler)
    X_val_s = apply_scaler_3d(X_val, input_scaler)
    Y_val_s = apply_scaler_3d(Y_val, target_scaler)

    print(f"  Scaler 拟合完成")

    # 3. 构建 DataLoader
    print("\n[3/6] 构建 DataLoader...")
    train_dataset = TensorDataset(
        torch.from_numpy(X_train_s),
        torch.from_numpy(Y_train_s),
    )
    val_dataset = TensorDataset(
        torch.from_numpy(X_val_s),
        torch.from_numpy(Y_val_s),
    )

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=0, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=0, pin_memory=True,
    )

    print(f"  训练批次: {len(train_loader)}, 验证批次: {len(val_loader)}")

    # 4. 构建模型
    print("\n[4/6] 构建 TCNTrendModel...")
    model = TCNTrendModel(
        input_size=data_config["n_input_features"],
        output_size=data_config["n_target_features"],
        predict_horizon=data_config["predict_horizon"],
        num_channels=NUM_CHANNELS,
        kernel_size=KERNEL_SIZE,
        dropout=DROPOUT,
    ).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  总参数: {total_params:,}")
    print(f"  可训练参数: {trainable_params:,}")

    # 5. 训练
    print("\n[5/6] 开始训练...")
    model, history, best_val_loss = train_model(
        model, train_loader, val_loader, target_scaler
    )

    # 6. 计算残差统计与阈值
    print("\n[6/6] 计算残差统计...")
    residual_stats, warning_threshold, alarm_threshold = compute_residual_stats(
        model, val_loader, target_scaler
    )

    print(f"  残差统计:")
    for key, val in residual_stats.items():
        print(f"    {key}: mean={val['mean']:.4f}, std={val['std']:.4f}, "
              f"p90={val['p90']:.4f}, p95={val['p95']:.4f}, p99={val['p99']:.4f}")

    # 7. 保存
    print("\n保存模型 bundle...")
    bundle_path = save_bundle(
        model, input_scaler, target_scaler, data_config,
        residual_stats, warning_threshold, alarm_threshold,
        history, best_val_loss,
    )

    print("\n" + "=" * 60)
    print("训练完成")
    print("=" * 60)

    return bundle_path


# =========================================================
# 8. 入口
# =========================================================
if __name__ == "__main__":
    main()