"""
数据模拟器
支持从 CSV 文件读取或生成合成数据
用于在没有真实传感器时测试系统
"""
import os
import csv
import math
import random
from datetime import datetime

from config import DATA_DIR, SAMPLE_INTERVAL


class DataSimulator:
    """张拉数据模拟器"""

    def __init__(self, csv_path=None, target_force=1000.0):
        """
        Args:
            csv_path: CSV 数据文件路径，None 则使用合成数据
            target_force: 目标张拉力值
        """
        self.target_force = target_force
        self.csv_path = csv_path
        self.data_rows = []
        self.current_index = 0
        self.finished = False

        if csv_path and os.path.exists(csv_path):
            self._load_csv(csv_path)
        else:
            self._generate_synthetic_data()

    @staticmethod
    def _parse_time_to_epoch(raw, row_index):
        """将 time 列转为 Unix 时间戳（秒）；无法解析时用行号占位。"""
        if raw is None or (isinstance(raw, str) and not str(raw).strip()):
            return float(row_index)
        try:
            return float(raw)
        except (TypeError, ValueError):
            pass
        s = str(raw).strip()
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M:%S.%f"):
            try:
                return datetime.strptime(s[:26], fmt).timestamp()
            except ValueError:
                continue
        try:
            return datetime.fromisoformat(s.replace(" ", "T", 1)).timestamp()
        except ValueError:
            return float(row_index)

    def _load_csv(self, csv_path):
        """从 CSV 文件加载数据（time 可为数值秒或日期时间字符串，统一为从首行起的经过秒数）"""
        raw_rows = []
        with open(csv_path, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                raw_rows.append(row)

        abs_times = [
            self._parse_time_to_epoch(row.get("time"), i) for i, row in enumerate(raw_rows)
        ]
        t0 = abs_times[0] if abs_times else 0.0

        for i, row in enumerate(raw_rows):
            t_elapsed = float(abs_times[i] - t0)
            self.data_rows.append({
                "time": round(t_elapsed, 3),
                "force_left": float(row.get("force_left", 0) or 0),
                "force_right": float(row.get("force_right", 0) or 0),
                "dis_left": float(row.get("dis_left", 0) or 0),
                "dis_right": float(row.get("dis_right", 0) or 0),
            })
        print(f"[DataSimulator] 从 CSV 加载 {len(self.data_rows)} 条数据")

    def _generate_synthetic_data(self):
        """生成合成张拉数据（加载-持荷-卸载三阶段）"""
        dt = SAMPLE_INTERVAL
        target = self.target_force
        rows = []

        # 加载阶段：0 ~ 60s，力值从 0 线性增加到目标值
        loading_duration = 60.0
        loading_steps = int(loading_duration / dt)
        for i in range(loading_steps):
            t = i * dt
            progress = t / loading_duration
            base_force = target * progress
            # 添加微小噪声和左右不对称
            noise_l = random.gauss(0, target * 0.005)
            noise_r = random.gauss(0, target * 0.005)
            force_left = base_force + noise_l + target * 0.01
            force_right = base_force + noise_r - target * 0.01
            # 位移与力值近似线性关系
            dis_left = progress * 25.0 + random.gauss(0, 0.1)
            dis_right = progress * 24.5 + random.gauss(0, 0.1)
            rows.append({
                "time": round(t, 1),
                "force_left": round(max(0, force_left), 2),
                "force_right": round(max(0, force_right), 2),
                "dis_left": round(max(0, dis_left), 3),
                "dis_right": round(max(0, dis_right), 3),
            })

        # 持荷阶段：60s ~ 120s，力值维持在目标值附近
        holding_duration = 60.0
        holding_steps = int(holding_duration / dt)
        for i in range(holding_steps):
            t = loading_duration + i * dt
            # 力值在目标值附近小幅波动
            force_left = target + random.gauss(0, target * 0.003) + target * 0.01
            force_right = target + random.gauss(0, target * 0.003) - target * 0.01
            # 位移缓慢增加（蠕变）
            creep = i * dt * 0.01
            dis_left = 25.0 + creep + random.gauss(0, 0.05)
            dis_right = 24.5 + creep + random.gauss(0, 0.05)
            rows.append({
                "time": round(t, 1),
                "force_left": round(force_left, 2),
                "force_right": round(force_right, 2),
                "dis_left": round(dis_left, 3),
                "dis_right": round(dis_right, 3),
            })

        # 卸载阶段：120s ~ 150s，力值逐渐降低
        unloading_duration = 30.0
        unloading_steps = int(unloading_duration / dt)
        for i in range(unloading_steps):
            t = loading_duration + holding_duration + i * dt
            progress = i / unloading_steps
            base_force = target * (1 - progress)
            force_left = base_force + random.gauss(0, target * 0.003) + target * 0.01
            force_right = base_force + random.gauss(0, target * 0.003) - target * 0.01
            dis_left = 25.0 * (1 - progress * 0.3) + random.gauss(0, 0.05)
            dis_right = 24.5 * (1 - progress * 0.3) + random.gauss(0, 0.05)
            rows.append({
                "time": round(t, 1),
                "force_left": round(max(0, force_left), 2),
                "force_right": round(max(0, force_right), 2),
                "dis_left": round(max(0, dis_left), 3),
                "dis_right": round(max(0, dis_right), 3),
            })

        self.data_rows = rows
        print(f"[DataSimulator] 生成合成数据 {len(rows)} 条")

    def next_point(self):
        """
        获取下一个数据点

        Returns:
            dict or None: 数据点字典，数据耗尽返回 None
        """
        if self.current_index >= len(self.data_rows):
            self.finished = True
            return None

        point = self.data_rows[self.current_index]
        self.current_index += 1
        return point

    def reset(self):
        """重置到起始位置"""
        self.current_index = 0
        self.finished = False

    @property
    def total_points(self):
        return len(self.data_rows)

    @property
    def progress(self):
        if len(self.data_rows) == 0:
            return 0
        return self.current_index / len(self.data_rows)