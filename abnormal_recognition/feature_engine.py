"""
在线特征计算引擎
负责从原始数据点计算所有下游模块需要的特征

输出两套特征：
1. base_features: 22维，供 IF/LOF 和 rule_engine 使用
2. tcn_features: 15维，供 TCN 模型使用（包含 target_force 归一化）
"""
import collections
import math
import numpy as np
from bisect import bisect_right


class OnlineFeatureBuilder:
    """在线特征构建器，逐点更新"""

    def __init__(self, window_size=10, target_force=None):
        """
        Args:
            window_size: 滑动窗口大小（用于计算标准差等统计量）
            target_force: 目标张拉力值（从前端传入）
        """
        self.window_size = window_size
        self.target_force = target_force if target_force else 1000.0

        # 历史窗口（存储最近 window_size 个点的特征）
        self.force_avg_window = collections.deque(maxlen=window_size)
        self.force_diff_window = collections.deque(maxlen=window_size)
        self.force_rate_window = collections.deque(maxlen=window_size)

        # 前一步状态
        self.prev_force_left = None
        self.prev_force_right = None
        self.prev_force_avg = None
        self.prev_left_force_rate = None
        self.prev_right_force_rate = None
        self.prev_dis_left = None
        self.prev_dis_right = None

        # 步数计数
        self.step = 0

        # 阶段状态
        self.phase = "loading"

        # 阶段切换连续确认计数器
        self._phase_confirm_counter = 0
        self._pending_phase = None

        # 欠张拉：张拉阶段力值长时间持平计数（|force_rate| 很小）
        self._loading_plateau_count = 0
        # 持荷是否属于“未达目标力”进入的欠张拉持荷（卸载判据不同）
        self._holding_sub_target = False
        self._holding_peak_force = 0.0

        # 记录进入持荷阶段的时间，用于避免“持荷早期掉力”被过早识别为卸载
        self._holding_start_time = None
        self._holding_elapsed_time = 0.0

        # (time, force_avg) 单调序列，用于张拉阶段近 W 秒平均速率 loading_avg_rate_kn_s（kN/s）
        self._tf_hist = collections.deque(maxlen=3000)

    def set_target_force(self, target_force):
        """更新目标力值"""
        self.target_force = target_force

    def reset(self):
        """重置状态（新一轮张拉时调用）"""
        self.__init__(window_size=self.window_size, target_force=self.target_force)

    @staticmethod
    def _interp_force_at_points(pts, t_query):
        """pts: [(t, f), ...] 按 t 升序；在 t_query 处线性插值力值。"""
        if not pts:
            return None
        ts = [p[0] for p in pts]
        fs = [p[1] for p in pts]
        if t_query <= ts[0]:
            return float(fs[0])
        if t_query >= ts[-1]:
            return float(fs[-1])
        i = bisect_right(ts, t_query) - 1
        t0, f0 = ts[i], fs[i]
        t1, f1 = ts[i + 1], fs[i + 1]
        if abs(t1 - t0) < 1e-12:
            return float(f0)
        return float(f0 + (t_query - t0) * (f1 - f0) / (t1 - t0))

    def _compute_loading_avg_rate_kn_s(self, time_val, force_avg):
        """
        近 LOADING_SPEED_AVERAGE_WINDOW_S 秒平均张拉速率 (kN/s)：
        (F(t) - F(t-W)) / W，其中 F(t-W) 由历史点线性插值。
        历史不含当前点；未满 W 秒跨度时返回 None。
        """
        from config import LOADING_SPEED_AVERAGE_WINDOW_S

        w = float(LOADING_SPEED_AVERAGE_WINDOW_S)
        if w <= 0:
            return None
        pts = list(self._tf_hist)
        if len(pts) < 1:
            return None
        if time_val - pts[0][0] + 1e-9 < w:
            return None
        t_cut = time_val - w
        if t_cut < pts[0][0]:
            return None
        f_past = self._interp_force_at_points(pts, t_cut)
        if f_past is None:
            return None
        return (float(force_avg) - f_past) / w

    def _append_tf_hist(self, time_val, force_avg):
        if self._tf_hist and abs(time_val - self._tf_hist[-1][0]) < 1e-9:
            self._tf_hist[-1] = (time_val, float(force_avg))
        else:
            self._tf_hist.append((time_val, float(force_avg)))

    def update(self, raw_point):
        """
        输入一个原始数据点，输出计算好的特征字典

        Args:
            raw_point: dict with keys: time, force_left, force_right, dis_left, dis_right

        Returns:
            dict: 包含所有计算特征的字典
        """
        time_val = float(raw_point.get("time", 0))
        force_left = float(raw_point["force_left"])
        force_right = float(raw_point["force_right"])
        dis_left = float(raw_point["dis_left"])
        dis_right = float(raw_point["dis_right"])

        # 基础计算
        force_avg = (force_left + force_right) / 2.0
        force_diff = abs(force_left - force_right)
        force_diff_ratio = force_diff / force_avg if force_avg > 1e-6 else 0.0
        total_delta_dis = dis_left + dis_right
        dis_diff = abs(dis_left - dis_right)

        # 变化率（一阶差分）
        if self.prev_force_left is not None:
            left_force_rate = force_left - self.prev_force_left
            right_force_rate = force_right - self.prev_force_right
            force_rate = force_avg - self.prev_force_avg
            left_dis_rate = dis_left - self.prev_dis_left
            right_dis_rate = dis_right - self.prev_dis_right
            dis_rate = (left_dis_rate + right_dis_rate)
        else:
            left_force_rate = 0.0
            right_force_rate = 0.0
            force_rate = 0.0
            left_dis_rate = 0.0
            right_dis_rate = 0.0
            dis_rate = 0.0

        # 力值变化率差异
        force_rate_diff = abs(left_force_rate - right_force_rate)
        force_rate_ratio = (
            force_rate_diff / abs(force_rate) if abs(force_rate) > 1e-6 else 0.0
        )
        dis_rate_diff = abs(left_dis_rate - right_dis_rate)

        # 加速度（二阶差分）
        if self.prev_left_force_rate is not None:
            left_force_acc = left_force_rate - self.prev_left_force_rate
            right_force_acc = right_force_rate - self.prev_right_force_rate
            force_acc = (left_force_acc + right_force_acc) / 2.0
        else:
            left_force_acc = 0.0
            right_force_acc = 0.0
            force_acc = 0.0

        # 刚度比
        stiffness_ratio = (
            force_avg / total_delta_dis if total_delta_dis > 1e-6 else 0.0
        )

        # 力-位移比（TCN 用）
        force_disp_ratio = (
            force_rate / dis_rate if abs(dis_rate) > 1e-6 else 0.0
        )

        # 张拉阶段规则用：近 N 秒平均张拉速率 (kN/s)，未满窗为 None
        loading_avg_rate_kn_s = self._compute_loading_avg_rate_kn_s(time_val, force_avg)
        self._append_tf_hist(time_val, force_avg)

        # 滑动窗口统计量
        self.force_avg_window.append(force_avg)
        self.force_diff_window.append(force_diff)
        self.force_rate_window.append(force_rate)

        force_std_5s = float(np.std(self.force_avg_window)) if len(self.force_avg_window) >= 3 else 0.0
        force_diff_std_5s = float(np.std(self.force_diff_window)) if len(self.force_diff_window) >= 3 else 0.0
        force_rate_std_5s = float(np.std(self.force_rate_window)) if len(self.force_rate_window) >= 3 else 0.0

        # TCN 专用：目标力值归一化
        force_ratio = force_avg / self.target_force if self.target_force > 0 else 0.0

        # 阶段判定（带连续确认；增加持荷最小时长保护）
        self._update_phase(time_val, force_avg, force_rate, force_std_5s)

        # 更新状态
        self.prev_force_left = force_left
        self.prev_force_right = force_right
        self.prev_force_avg = force_avg
        self.prev_left_force_rate = left_force_rate
        self.prev_right_force_rate = right_force_rate
        self.prev_dis_left = dis_left
        self.prev_dis_right = dis_right
        self.step += 1

        # 构建输出特征字典
        features = {
            # 原始值（保留用于前端显示和数据库存储）
            "time": time_val,
            "force_left": force_left,
            "force_right": force_right,
            "dis_left": dis_left,
            "dis_right": dis_right,
            # IF/LOF 22维特征
            "force_avg": force_avg,
            "force_diff": force_diff,
            "force_diff_ratio": force_diff_ratio,
            "left_force_rate": left_force_rate,
            "right_force_rate": right_force_rate,
            "force_rate": force_rate,
            "loading_avg_rate_kn_s": loading_avg_rate_kn_s,
            "force_rate_diff": force_rate_diff,
            "force_rate_ratio": force_rate_ratio,
            "left_force_acc": left_force_acc,
            "right_force_acc": right_force_acc,
            "force_acc": force_acc,
            "total_delta_dis": total_delta_dis,
            "dis_diff": dis_diff,
            "left_dis_rate": left_dis_rate,
            "right_dis_rate": right_dis_rate,
            "dis_rate_diff": dis_rate_diff,
            "stiffness_ratio": stiffness_ratio,
            "force_std_5s": force_std_5s,
            "force_diff_std_5s": force_diff_std_5s,
            "force_rate_std_5s": force_rate_std_5s,
            # TCN 额外特征
            "dis_rate": dis_rate,
            "force_disp_ratio": force_disp_ratio,
            "force_ratio": force_ratio,
            # 阶段
            "phase": self.phase,
            # 是否因“未达目标力但力值持平”而进入持荷（便于前端与欠张逻辑展示）
            "holding_sub_target": (
                bool(self._holding_sub_target) if self.phase == "holding" else False
            ),
            # 调试字段：便于确认是否因为持荷时间保护而暂未进入卸载
            "holding_elapsed_time": self._holding_elapsed_time,
        }

        return features

    def _update_phase(self, time_val, force_avg, force_rate, force_std_5s):
        """根据力值、变化率和持荷持续时间判定当前阶段（带连续确认机制）"""
        from config import PHASE_THRESHOLDS, PHASE_CONFIRM_COUNT, SAMPLE_INTERVAL

        ratio_to_target = force_avg / self.target_force if self.target_force > 0 else 0
        thr = PHASE_THRESHOLDS
        load_to_hold = thr["loading_to_holding_ratio"]
        min_hold_s = float(thr.get("min_holding_time_before_unloading_s", 0.0))

        # 计算期望阶段
        expected_phase = self.phase

        if self.phase == "loading":
            reached_target = ratio_to_target >= load_to_hold
            if reached_target:
                self._loading_plateau_count = 0
            else:
                min_r = thr.get("under_tension_plateau_min_ratio", 0.35)
                abs_max = thr.get("under_tension_plateau_force_rate_abs_max", 4.0)
                n_min_cfg = int(thr.get("under_tension_plateau_consecutive_min", 25))
                min_plateau_s = float(
                    thr.get("under_tension_plateau_min_duration_s", 2.5)
                )
                dt = float(SAMPLE_INTERVAL) if SAMPLE_INTERVAL else 1.0
                n_min = max(
                    n_min_cfg,
                    int(math.ceil(min_plateau_s / max(dt, 1e-9))),
                )
                std_max = thr.get("under_tension_plateau_std_max", 12.0)
                flat = abs(force_rate) <= abs_max
                stable_window = (
                    len(self.force_avg_window) >= 3 and force_std_5s <= std_max
                )
                if (
                    ratio_to_target >= min_r
                    and flat
                    and stable_window
                ):
                    self._loading_plateau_count += 1
                else:
                    self._loading_plateau_count = 0

            plateau_to_hold = (
                not reached_target
                and self._loading_plateau_count >= n_min
            )
            if reached_target or plateau_to_hold:
                expected_phase = "holding"

        elif self.phase == "holding":
            # 更新持荷已持续时间。没有记录起点时用当前时间补齐，避免历史状态为空。
            if self._holding_start_time is None:
                self._holding_start_time = time_val
            self._holding_elapsed_time = max(0.0, time_val - self._holding_start_time)
            enough_hold_time = self._holding_elapsed_time >= min_hold_s

            self._holding_peak_force = max(self._holding_peak_force, force_avg)
            drop_from_peak = thr.get("sub_target_holding_drop_from_peak", 0.08)

            # 关键修改：持荷时间未达到 min_holding_time_before_unloading_s 前，不允许切换到 unloading。
            # 这样持荷早期的 10% 掉力会继续以 holding 阶段送入 RuleEngine，优先判为滑丝/断丝异常。
            if enough_hold_time:
                if self._holding_sub_target:
                    if self._holding_peak_force > 1e-6:
                        if force_avg < self._holding_peak_force * (1.0 - drop_from_peak):
                            expected_phase = "unloading"
                    if force_rate < thr["unloading_rate_threshold"]:
                        expected_phase = "unloading"
                else:
                    if ratio_to_target < thr["holding_force_drop_ratio"]:
                        expected_phase = "unloading"
                    elif force_rate < thr["unloading_rate_threshold"]:
                        expected_phase = "unloading"

        # unloading 阶段不再转换

        # 连续确认机制
        if expected_phase != self.phase:
            if self._pending_phase == expected_phase:
                self._phase_confirm_counter += 1
            else:
                self._pending_phase = expected_phase
                self._phase_confirm_counter = 1

            if self._phase_confirm_counter >= PHASE_CONFIRM_COUNT:
                prev_phase = self.phase
                self.phase = expected_phase
                self._pending_phase = None
                self._phase_confirm_counter = 0

                if prev_phase == "loading" and self.phase == "holding":
                    self._holding_sub_target = ratio_to_target < load_to_hold
                    self._holding_peak_force = force_avg
                    self._holding_start_time = time_val
                    self._holding_elapsed_time = 0.0
                    self._loading_plateau_count = 0

        else:
            # 当前满足现有阶段，重置计数器
            self._pending_phase = None
            self._phase_confirm_counter = 0

    def force_transition_to_unloading(self):
        """
        由 RuleEngine 在「持荷突降后观察窗内持续回落」判据满足时调用，
        将当前阶段置为卸载（与突降后正常卸载区分，避免误报断丝）。
        """
        if self.phase != "holding":
            return
        self.phase = "unloading"
        self._pending_phase = None
        self._phase_confirm_counter = 0
        self._holding_start_time = None
        self._holding_elapsed_time = 0.0

    def get_base_feature_vector(self, features):
        """
        从特征字典中提取 IF/LOF 模型需要的 22 维向量
        """
        from config import IF_LOF_FEATURE_COLS
        return [features[col] for col in IF_LOF_FEATURE_COLS]

    def get_tcn_feature_vector(self, features):
        """
        从特征字典中提取 TCN 模型需要的特征向量
        """
        from config import TCN_FEATURE_COLS
        return [features[col] for col in TCN_FEATURE_COLS]