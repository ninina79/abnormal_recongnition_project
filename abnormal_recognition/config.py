"""
系统配置文件
"""
import os

# =========================================================
# Flask 配置
# =========================================================
FLASK_HOST = "0.0.0.0"
FLASK_PORT = 5000
FLASK_DEBUG = True

# =========================================================
# 数据库配置 (MySQL)
# =========================================================
DB_CONFIG = {
    "host": "127.0.0.1",
    "port": 3306,
    "user": "root",
    "password": "jy121206030508",
    "database": "tension_monitor",
    "charset": "utf8mb4",
}

# =========================================================
# 路径配置
# =========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


def resolve_simulation_csv_path():
    """
    将 SIMULATION_CSV_FILENAME 解析为可读 CSV 的绝对路径。
    支持：绝对路径；相对项目根目录的路径（如 data/xxx.csv）；仅文件名时尝试 DATA_DIR/文件名。
    无配置或文件不存在时返回 None（DataSimulator 将使用合成数据）。
    """
    raw = SIMULATION_CSV_FILENAME
    if raw is None:
        return None
    s = str(raw).strip()
    if not s:
        return None
    p = os.path.expanduser(s)
    if os.path.isabs(p):
        return os.path.normpath(p) if os.path.isfile(p) else None
    rel = os.path.normpath(os.path.join(BASE_DIR, p))
    if os.path.isfile(rel):
        return rel
    under_data = os.path.normpath(os.path.join(DATA_DIR, os.path.basename(p)))
    if os.path.isfile(under_data):
        return under_data
    return None


# 张拉模拟 CSV：写死为相对项目根的路径、绝对路径，或仅文件名（在 data/ 下查找）。
# 也可改为从环境变量读取，例如：
# SIMULATION_CSV_FILENAME = os.environ.get("TENSION_SIMULATION_CSV", "").strip() or None
SIMULATION_CSV_FILENAME = (
    "data/integrated_614532288100831232-615240266495954944-615258739682971656-615258739682971672.csv"
)

# 实时监测页 Chart.js 保留的最大点数（由 Flask 注入到 realtime.html，与前端一致）
REALTIME_CHART_MAX_POINTS = 500

# =========================================================
# 数据采集配置
# =========================================================
SAMPLE_INTERVAL = 0.1    # 采样间隔（秒）
PUSH_INTERVAL = 0.1       # WebSocket 推送间隔（秒）

# 完成后摘要：进入持荷后跳过前 N 秒（力与油压稳定），再对 force_avg 取中位数与目标力算偏差率
HOLDING_STABLE_MEDIAN_SKIP_S = 50.0

# 持荷阶段：进入持荷后满本秒数，再采集下述窗口内的力值做一次性欠张/超张/单端超张判定
HOLDING_UNDER_OVER_JUDGE_DELAY_S = 20.0
# 判定窗口长度（秒）：在 [DELAY, DELAY+WINDOW] 内累积 force_avg，窗满后用中位数判一次并锁定
HOLDING_UNDER_OVER_JUDGE_WINDOW_S = 8.0

# =========================================================
# 规则检测阈值配置
# 说明：RuleEngine 与 RealTimeTensionMonitor 共用这组阈值，避免实时检测和训练数据标注标准不一致。
# 力值上下限：FORCE_LOWER_THRESHOLD（欠张）、FORCE_THRESHOLD（超张）、SIDE_FORCE_THRESHOLD（单端超张）。
# =========================================================

# ---------- 力值上下限（持荷/实时规则共用） ----------
FORCE_THRESHOLD = 0.05           # 平均力超过目标×(1+本值) 判超张拉（5%）
FORCE_LOWER_THRESHOLD = 0.05     # 平均力低于目标×(1-本值) 判欠张拉（5%，与超张对称）
SIDE_FORCE_THRESHOLD = 0.08    # 单端张拉力超过目标×(1+本值) 判单端超张（8%）

# ---------- 持荷阶段 ----------
HOLD_TIME_MIN = 300              # 持荷时间不少于 300s
FORCE_FLUCTUATION = 0.08         # 持荷阶段：当前平均力偏离目标力超过 8% 判持荷波动异常
FORCE_DROP_PERCENT = 0.10        # 持荷阶段：相对持荷参考力下降超过 10% 判严重异常
BROKEN_WIRE_DROP = 0.08          # 相邻点平均力单步下降超过目标力×本值，视为一次「突降」候选（张拉立即告警；持荷进入 2s 确认）

# 持荷单步突降后的确认（与 feature_engine 的 time 字段一致，单位为秒）
FORCE_DROP_POST_CONFIRM_S = 2.0          # 观察窗时长：突降后据此区分「平稳=滑丝/断丝」与「继续降=卸载」
FORCE_DROP_STABLE_BAND_RATIO = 0.012     # 窗内 (max-min)/target 低于此认为力值在突降后水平附近平稳
FORCE_DROP_CONTINUE_RATIO = 0.015        # 窗内相对突降锚点再降超过 target×本值，认为「一直在降」

# ---------- 加载阶段整体张拉速度（基于「近 N 秒平均张拉速率」，单位 kN/s） ----------
# 在线特征由 feature_engine 计算 loading_avg_rate_kn_s = (F(t)-F(t-W)) / W（F 在 t-W 处线性插值），
# 再用 find_speed.py 在 zhangla_data 上统计分布后，替换下面 SPEED_NORMAL_LOW/HIGH。
LOADING_SPEED_AVERAGE_WINDOW_S = 8.0   # 张拉阶段速度判定使用的滑动平均窗长（秒）
SPEED_NORMAL_LOW = 1.0           # 正常平均张拉速度下限（kN/s，对应近 W 秒平均，W 见上一行）
SPEED_NORMAL_HIGH = 14.0         # 正常平均张拉速度上限（kN/s，对应近 W 秒平均）
SPEED_SLOW_FACTOR = 0.70         # 低于 SPEED_NORMAL_LOW × 本系数，判为速度过慢候选
SPEED_FAST_FACTOR = 1.30         # 高于 SPEED_NORMAL_HIGH × 本系数，判为速度过快候选
SPEED_CONSECUTIVE_LIMIT = 3      # 持续过慢/过快确认次数

# 速度突变判断
SPEED_JUMP_REFERENCE_COUNT = 3
SPEED_SUDDEN_SLOW_RATIO = 0.20
SPEED_SUDDEN_FAST_RATIO = 3.0
SPEED_JUMP_CONSECUTIVE_LIMIT = 2

# ---------- 两端同步（仅持荷阶段：RuleEngine 与 RealTimeTensionMonitor 判力差/速度差；张拉与卸载不判） ----------
FORCE_IMBALANCE_RATIO = 0.08             # 左右端力差超过目标力 8%
FORCE_IMBALANCE_MIN_KN = 20.0            # 最小力差阈值，避免小目标力下过敏感
SIDE_SPEED_DIFF_RATIO = 0.45             # RealTimeTensionMonitor 侧使用的左右速度差比例
SIDE_IMBALANCE_WARNING_LIMIT = 2
SIDE_IMBALANCE_ALARM_LIMIT = 5

# RuleEngine 持荷阶段：两端力值变化率差（左右不同步）固定阈值。
RULE_SIDE_SPEED_DIFF_MAX = 50.0          # kN/step；SAMPLE_INTERVAL=1s 时近似 kN/s

# RuleEngine 保留项：刚度和卸载速度规则。
STIFFNESS_RATIO_MIN = 5.0
STIFFNESS_RATIO_MAX = 200.0
UNLOADING_RATE_MIN = -80.0
FORCE_DIFF_ABS_MAX = 100.0

# =========================================================
# 阶段判定配置
# =========================================================
PHASE_THRESHOLDS = {
    "loading_to_holding_ratio": 0.98,       # 平均力/目标力 ≥ 本值 → 主路径进入持荷
    "holding_force_drop_ratio": 0.90,       # 持荷阶段力值降到目标力90%以下进入卸载（已达目标力时）
    "unloading_rate_threshold": -5.0,       # 力值变化率低于此值进入卸载

    # 如果你已经采用上一版 feature_engine.py 中的“持荷时间保护”逻辑，保留该项即可生效。
    # 作用：持荷未达到该时长前，不允许仅因低于90%或卸载速率判为 unloading，便于先报滑丝/断丝类异常。
    # 如果暂时不想启用，可以改为 0.0。
    "min_holding_time_before_unloading_s": 100.0,

    # ---------- 欠张拉：未达目标力时凭“力值长时间持平”进入持荷（阈值过低易过早显示持荷） ----------
    # 平均力至少达到目标的该比例后才允许用持平判据
    "under_tension_plateau_min_ratio": 0.90,
    # 相邻采样点平均力变化 |Δforce_avg| 小于该值视为“持平”（与 feature_engine 中 force_rate 定义一致：每点差分，单位 kN）
    "under_tension_plateau_force_rate_abs_max": 8,
    # 连续多少点满足“持平”；实际还会与 under_tension_plateau_min_duration_s、SAMPLE_INTERVAL 取较大点数
    "under_tension_plateau_consecutive_min": 30,
    # 持平至少持续本秒数（换算为最少采样点数），避免高采样率下亚秒级误判持荷
    "under_tension_plateau_min_duration_s": 2.5,
    # 持平判据生效时，最近 window 内力值标准差须低于此值（kN），避免仍在明显爬升时误判
    "under_tension_plateau_std_max": 8.0,
    # ---------- 欠张拉持荷阶段：进入卸载 ----------
    # 当持荷时平均力从未达到 loading_to_holding_ratio 时，用“相对持荷段峰值回落比例”判卸载，而非与目标力 90% 比较
    "sub_target_holding_drop_from_peak": 0.10,
}

# 阶段切换连续确认次数
PHASE_CONFIRM_COUNT = 3

# =========================================================
# IF/LOF 特征列
# =========================================================
IF_LOF_FEATURE_COLS = [
    "force_avg", "force_diff", "force_diff_ratio",
    "left_force_rate", "right_force_rate", "force_rate",
    "force_rate_diff", "force_rate_ratio",
    "left_force_acc", "right_force_acc", "force_acc",
    "total_delta_dis", "dis_diff",
    "left_dis_rate", "right_dis_rate", "dis_rate_diff",
    "stiffness_ratio",
    "force_std_5s", "force_diff_std_5s", "force_rate_std_5s",
    "dis_rate", "force_disp_ratio",
]

# =========================================================
# TCN 特征列
# =========================================================
TCN_FEATURE_COLS = [
    "norm_force_left",
    "norm_force_right",
    "norm_force_avg",
    "norm_force_diff",
    "norm_force_rate",
    "norm_left_force_rate",
    "norm_right_force_rate",
    "norm_force_rate_diff",
    "norm_force_acc",
    "dis_left",
    "dis_right",
    "total_delta_dis",
    "dis_diff",
    "dis_rate",
    "force_disp_ratio",
    "left_right_force_diff",
    "stiffness_ratio_norm",
    "force_std_5s_norm",
]

TCN_INPUT_WINDOW = 10