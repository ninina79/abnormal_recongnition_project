import statistics
from datetime import datetime
from typing import Optional, Dict, Any

from config import (
    LOADING_SPEED_AVERAGE_WINDOW_S,
    FORCE_THRESHOLD,
    FORCE_LOWER_THRESHOLD,
    SIDE_FORCE_THRESHOLD,
    HOLD_TIME_MIN,
    FORCE_FLUCTUATION,
    FORCE_DROP_PERCENT,
    BROKEN_WIRE_DROP,
    FORCE_DROP_POST_CONFIRM_S,
    FORCE_DROP_STABLE_BAND_RATIO,
    FORCE_DROP_CONTINUE_RATIO,
    SPEED_NORMAL_LOW,
    SPEED_NORMAL_HIGH,
    SPEED_SLOW_FACTOR,
    SPEED_FAST_FACTOR,
    SPEED_CONSECUTIVE_LIMIT,
    SPEED_JUMP_REFERENCE_COUNT,
    SPEED_SUDDEN_SLOW_RATIO,
    SPEED_SUDDEN_FAST_RATIO,
    SPEED_JUMP_CONSECUTIVE_LIMIT,
    FORCE_IMBALANCE_RATIO,
    FORCE_IMBALANCE_MIN_KN,
    SIDE_SPEED_DIFF_RATIO,
    SIDE_IMBALANCE_WARNING_LIMIT,
    SIDE_IMBALANCE_ALARM_LIMIT,
    RULE_SIDE_SPEED_DIFF_MAX,
    STIFFNESS_RATIO_MIN,
    STIFFNESS_RATIO_MAX,
    UNLOADING_RATE_MIN,
    FORCE_DIFF_ABS_MAX,
    HOLDING_UNDER_OVER_JUDGE_DELAY_S,
    HOLDING_UNDER_OVER_JUDGE_WINDOW_S,
)

# =========================================================
# 2. 阶段划分配置
# =========================================================
INITIAL_STAGE_RATIO = 0.20       # 目标力 20% 以下视为初始阶段
HOLD_START_RATIO = 0.98          # 达到目标力 98% 进入持荷阶段
HOLD_END_RATIO = 0.90            # 持荷后低于目标力 90% 视为进入卸载阶段
UNLOAD_FINISH_RATIO = 0.10       # 低于目标力 10% 视为卸载完成

class RuleEngine:
    """
    基于规则的实时异常检测。
    覆盖张拉、持荷、卸载；其中左右力差、力差比与两端速度差仅在持荷阶段判定。
    """

    def __init__(self, target_force=1000.0):
        self.target_force = float(target_force)

        # 规则阈值配置：保留为字典，便于前端调试时统一查看。
        # 其中平均张拉力速度、持荷波动、持荷下降等核心阈值来自 config.py，
        # 与 RealTimeTensionMonitor 保持同一来源。
        self.thresholds = {
            # 力值差异相关
            "force_diff_ratio_max": FORCE_IMBALANCE_RATIO,
            "force_diff_abs_max": FORCE_DIFF_ABS_MAX,
            # 两端速度不同步：用户要求先保留 50.0
            "side_speed_diff_max": RULE_SIDE_SPEED_DIFF_MAX,
            # 位移：左右位移差不作为异常（不配置 dis_diff 阈值）
            "stiffness_ratio_min": STIFFNESS_RATIO_MIN,
            "stiffness_ratio_max": STIFFNESS_RATIO_MAX,
            # 持荷阶段
            "holding_force_drop_max": FORCE_DROP_PERCENT,
            "holding_force_fluctuation_max": FORCE_FLUCTUATION,
            "force_lower_ratio": FORCE_LOWER_THRESHOLD,
            # 卸载阶段
            "unloading_rate_min": UNLOADING_RATE_MIN,
        }

        # 持荷阶段参考力值
        self.holding_reference_force = None

        # 平均张拉力速度检测状态（与 RealTimeTensionMonitor 口径一致）
        self.speed_normal_low = float(SPEED_NORMAL_LOW)
        self.speed_normal_high = float(SPEED_NORMAL_HIGH)
        self.speed_slow_limit = self.speed_normal_low * float(SPEED_SLOW_FACTOR)
        self.speed_fast_limit = self.speed_normal_high * float(SPEED_FAST_FACTOR)
        self.speed_slow_counter = 0
        self.speed_fast_counter = 0
        self.speed_sudden_slow_counter = 0
        self.speed_sudden_fast_counter = 0
        self.speed_history = []

        # 单步突降：张拉立即告警；持荷经 FORCE_DROP_POST_CONFIRM_S 区分滑丝/卸载
        self._prev_force_avg_for_drop = None
        self._holding_drop_watch = None  # dict: t0, anchor, fmin, fmax
        self._request_unloading_after_hold_drop = False

        # 持荷欠张/超张：满延迟后采集窗口样本，窗满一次性判定后锁定
        self._holding_force_judged = False
        self._holding_judge_samples = []  # list[(force_avg, force_left, force_right)]

    def set_target_force(self, target_force):
        """更新目标力值"""
        self.target_force = float(target_force)

    def _reset_speed_state(self):
        """离开 loading 阶段时，重置平均张拉力速度检测状态。"""
        self.speed_slow_counter = 0
        self.speed_fast_counter = 0
        self.speed_sudden_slow_counter = 0
        self.speed_sudden_fast_counter = 0
        self.speed_history = []

    def _check_average_loading_speed(self, speed_kn_per_s):
        """
        检测 loading 阶段的平均张拉速度异常。
        speed_kn_per_s：近 LOADING_SPEED_AVERAGE_WINDOW_S 秒的平均张拉速率 (kN/s)，
        由 feature_engine 的 loading_avg_rate_kn_s 提供；未满窗时为 None，本函数不判。
        """
        if speed_kn_per_s is None:
            return None

        speed = float(speed_kn_per_s)

        if speed < self.speed_slow_limit:
            self.speed_slow_counter += 1
            self.speed_fast_counter = 0
        elif speed > self.speed_fast_limit:
            self.speed_fast_counter += 1
            self.speed_slow_counter = 0
        else:
            self.speed_slow_counter = 0
            self.speed_fast_counter = 0

        alert = None

        if self.speed_fast_counter >= SPEED_CONSECUTIVE_LIMIT:
            alert = {
                "type": "speed_too_fast",
                "severity": "critical",
                "message": (
                    f"整体平均张拉速度持续过快（近{LOADING_SPEED_AVERAGE_WINDOW_S:.0f}s平均）：当前 {speed:.2f}kN/s，"
                    f"高于阈值 {self.speed_fast_limit:.2f}kN/s，"
                    f"已连续 {self.speed_fast_counter} 个点。"
                    "【常见原因】未分级升压、油泵流量过大或误开快进；后果：冲击滑丝/断丝、锚区压碎、伸长读数失真。"
                    "【隐患】升载过快易形成冲击与局部应力集中，夹片跟进不良时滑丝、咬丝风险升高；"
                    "油压—伸长双控难以同步校核，端部混凝土劈裂或锚下局部承压超限概率增大。"
                ),
            }
        elif self.speed_slow_counter >= SPEED_CONSECUTIVE_LIMIT:
            alert = {
                "type": "speed_too_slow",
                "severity": "warning",
                "message": (
                    f"整体平均张拉速度持续过慢（近{LOADING_SPEED_AVERAGE_WINDOW_S:.0f}s平均）：当前 {speed:.2f}kN/s，"
                    f"低于阈值 {self.speed_slow_limit:.2f}kN/s，"
                    f"已连续 {self.speed_slow_counter} 个点。"
                    "【常见原因】油泵供油不足、滤网堵塞或千斤顶摩阻大、操作过慢；后果：油温升高损密封、松弛与夹片回缩偏大。"
                    "【隐患】长时间高压易使油温与密封件工况变差，摩阻与回缩读数漂移；"
                    "若伴随供油不稳，真实控制力与记录值可能偏离，影响张拉质量判定。"
                ),
            }

        # 速度突变判断，与 RealTimeTensionMonitor 的判断口径保持一致。
        if alert is None and len(self.speed_history) >= SPEED_JUMP_REFERENCE_COUNT:
            recent = self.speed_history[-SPEED_JUMP_REFERENCE_COUNT:]
            recent_avg = sum(recent) / len(recent)

            if recent_avg > 0:
                if speed < recent_avg * SPEED_SUDDEN_SLOW_RATIO:
                    self.speed_sudden_slow_counter += 1
                    self.speed_sudden_fast_counter = 0
                elif speed > recent_avg * SPEED_SUDDEN_FAST_RATIO:
                    self.speed_sudden_fast_counter += 1
                    self.speed_sudden_slow_counter = 0
                else:
                    self.speed_sudden_slow_counter = 0
                    self.speed_sudden_fast_counter = 0

                if self.speed_sudden_slow_counter >= SPEED_JUMP_CONSECUTIVE_LIMIT:
                    alert = {
                        "type": "speed_sudden_slow",
                        "severity": "warning",
                        "message": (
                            f"整体平均张拉速度突然下降（近{LOADING_SPEED_AVERAGE_WINDOW_S:.0f}s平均）：当前 {speed:.2f}kN/s，"
                            f"近期均值 {recent_avg:.2f}kN/s。"
                            "【常见原因】油路泄漏、密封失效或孔道卡阻突变；后果：伸长与力值不对应，易误判张拉到位或掩盖断丝风险。"
                            "【隐患】骤变常见于油路节流、内泄或孔道/转向局部卡阻变化，力—伸长对应关系易失真；"
                            "若存在松脱或局部失载，未察觉则可能导致控制力不足或记录误判。"
                        ),
                    }
                elif self.speed_sudden_fast_counter >= SPEED_JUMP_CONSECUTIVE_LIMIT:
                    alert = {
                        "type": "speed_sudden_fast",
                        "severity": "critical",
                        "message": (
                            f"整体平均张拉速度突然升高（近{LOADING_SPEED_AVERAGE_WINDOW_S:.0f}s平均）：当前 {speed:.2f}kN/s，"
                            f"近期均值 {recent_avg:.2f}kN/s。"
                            "【常见原因】阀门卡滞突然开大、误碰快进或泵况突变；后果：冲击断丝/超张、锚混凝土压碎、夹片跟进不及。"
                            "【隐患】冲击式升载易损伤钢绞线与锚具齿口，夹片未及时跟进时滑丝、断丝风险显著上升；"
                            "梁体局部承压与偏心效应加剧。"
                        ),
                    }

        self.speed_history.append(float(speed))
        if len(self.speed_history) > 20:
            self.speed_history.pop(0)

        return alert

    def _check_force_magnitude_alerts(self, features, *, check_under: bool):
        """
        平均力、单端力相对目标力的欠张 / 超张（与 RealTimeTensionMonitor._check_force_limit 同口径）。
        check_under：持荷阶段为 True；张拉阶段不判欠张（力值尚在上升过程中）。
        """
        alerts = []
        if self.target_force <= 0:
            return alerts

        force_avg = float(features.get("force_avg", 0) or 0)
        force_left = float(features.get("force_left", 0) or 0)
        force_right = float(features.get("force_right", 0) or 0)

        lower = self.target_force * (1.0 - float(FORCE_LOWER_THRESHOLD))
        upper = self.target_force * (1.0 + float(FORCE_THRESHOLD))
        side_upper = self.target_force * (1.0 + float(SIDE_FORCE_THRESHOLD))

        if check_under and force_avg < lower:
            alerts.append({
                "type": "under_tension",
                "severity": "critical",
                "message": (
                    f"欠张拉：当前平均力 {force_avg:.1f}kN，低于目标力允许下限 {lower:.1f}kN "
                    f"（目标 {self.target_force:.1f}kN，允许偏低 "
                    f"{float(FORCE_LOWER_THRESHOLD) * 100:.1f}%）。"
                    "【常见原因】油压表偏低标定、千斤顶内摩阻未扣、孔道摩阻过大或过早锚固；后果：梁体开裂、下挠，承载储备不足。"
                    "【隐患】有效预应力不足，使用阶段易出现裂缝宽度与挠度偏大，抗裂与刚度储备下降，耐久性变差。"
                ),
            })

        if force_avg > upper:
            alerts.append({
                "type": "over_tension",
                "severity": "critical",
                "message": (
                    f"超张拉：当前平均力 {force_avg:.1f}kN，超过目标力允许上限 {upper:.1f}kN "
                    f"（允许偏高 {float(FORCE_THRESHOLD) * 100:.1f}%）。"
                    "【常见原因】油压表偏高标定、误读油压或程序超张；后果：断丝、锚区压碎、反拱过大与端部纵向裂缝。"
                    "【隐患】钢绞线屈服或断裂风险升高，锚下混凝土压溃、端部劈裂及上拱过大导致桥面或邻跨线形恶化；"
                    "可能损伤永久锚固体系。"
                ),
            })

        max_side = max(force_left, force_right)
        if max_side > side_upper:
            side = "左端" if force_left >= force_right else "右端"
            alerts.append({
                "type": "side_over_tension",
                "severity": "critical",
                "message": (
                    f"{side}单端超张拉：左 {force_left:.1f}kN，右 {force_right:.1f}kN，"
                    f"单端允许上限 {side_upper:.1f}kN。"
                    "【常见原因】两端不同步、单侧仪表/千斤顶故障或孔道摩阻严重不均；后果：梁体侧弯扭转、预应力偏心分布。"
                    "【隐患】截面应力偏心，腹板或底板易出现斜向或局部裂缝；一侧锚具与转向块超载，压溃或扭曲变形风险上升。"
                ),
            })

        return alerts

    def _reset_holding_force_judge_state(self):
        """离开持荷或重置会话时清空持荷力值带判定状态。"""
        self._holding_force_judged = False
        self._holding_judge_samples = []

    def _collect_and_judge_holding_force_once(self, features):
        """
        持荷满 DELAY 后，在 [DELAY, DELAY+WINDOW] 内累积样本；
        窗满后以中位数做一次性欠张/超张/单端超张判定，之后不再更新。
        """
        if self._holding_force_judged:
            return []

        het = float(features.get("holding_elapsed_time", 0.0) or 0.0)
        delay = float(HOLDING_UNDER_OVER_JUDGE_DELAY_S)
        window = float(HOLDING_UNDER_OVER_JUDGE_WINDOW_S)
        window_end = delay + window

        if het < delay:
            return []

        force_avg = float(features.get("force_avg", 0) or 0)
        force_left = float(features.get("force_left", 0) or 0)
        force_right = float(features.get("force_right", 0) or 0)

        if het < window_end:
            self._holding_judge_samples.append((force_avg, force_left, force_right))
            return []

        samples = list(self._holding_judge_samples)
        if not samples:
            samples = [(force_avg, force_left, force_right)]

        rep_features = dict(features)
        rep_features["force_avg"] = statistics.median([s[0] for s in samples])
        rep_features["force_left"] = statistics.median([s[1] for s in samples])
        rep_features["force_right"] = statistics.median([s[2] for s in samples])

        alerts = self._check_force_magnitude_alerts(rep_features, check_under=True)
        self._holding_force_judged = True
        self._holding_judge_samples = []
        return alerts

    def consume_unloading_after_hold_drop(self):
        """持荷突降后若判为「继续降=卸载」，由 app 在 check 后调用一次并切入卸载阶段。"""
        v = self._request_unloading_after_hold_drop
        self._request_unloading_after_hold_drop = False
        return v

    def _process_step_force_drop(self, features, alerts):
        """
        相邻采样点平均力单步下降（相对目标力）：
        - 张拉：达到 BROKEN_WIRE_DROP 即报突降异常（大则 force_drop_alarm，否则 warning）。
        - 持荷：达到 BROKEN_WIRE_DROP 起算观察窗；满 FORCE_DROP_POST_CONFIRM_S 后，
          窗内相对锚点继续明显下降则置 request_unloading；否则窗内起伏小则报滑丝/断丝类突降。
        """
        phase = features.get("phase", "loading")
        t = float(features.get("time", 0.0))
        curr = float(features.get("force_avg", 0.0) or 0.0)
        tf = self.target_force
        if tf <= 0:
            self._prev_force_avg_for_drop = curr
            return

        if phase == "unloading":
            self._holding_drop_watch = None
            self._prev_force_avg_for_drop = curr
            return

        if curr / tf < INITIAL_STAGE_RATIO:
            self._holding_drop_watch = None
            self._prev_force_avg_for_drop = curr
            return

        if self._holding_drop_watch is not None:
            if phase != "holding":
                self._holding_drop_watch = None
            else:
                w = self._holding_drop_watch
                w["fmin"] = min(w["fmin"], curr)
                w["fmax"] = max(w["fmax"], curr)
                if t - w["t0"] >= FORCE_DROP_POST_CONFIRM_S:
                    anchor = w["anchor"]
                    band = w["fmax"] - w["fmin"]
                    continued = (anchor - w["fmin"]) >= FORCE_DROP_CONTINUE_RATIO * tf
                    stable = band <= FORCE_DROP_STABLE_BAND_RATIO * tf
                    self._holding_drop_watch = None
                    if continued:
                        self._request_unloading_after_hold_drop = True
                        self._prev_force_avg_for_drop = curr
                        return
                    if stable:
                        alerts.append({
                            "type": "force_drop_alarm",
                            "severity": "critical",
                            "message": (
                                f"持荷突降后约{FORCE_DROP_POST_CONFIRM_S:.0f}s 力值在突降后水平附近平稳："
                                f"疑似断丝/滑丝（锚点 {anchor:.1f}kN，窗内起伏 {band:.1f}kN）。"
                                "【常见原因】断丝、严重滑丝或锚固异常；后果：该束有效预应力骤降，"
                                "内力重分布致开裂与挠度风险升高，须停机确认是否退锚更换。"
                                "【隐患】该束预应力突然损失，同束钢绞线内力重分布，残余承载与变形控制削弱；"
                                "若继续发展可演变为逐根滑脱，影响结构安全与验收。"
                            ),
                        })
                        self._prev_force_avg_for_drop = curr
                        return
                    self._prev_force_avg_for_drop = curr
                    return

        prev = self._prev_force_avg_for_drop
        if prev is None:
            self._prev_force_avg_for_drop = curr
            return

        delta = curr - prev
        if delta >= 0:
            self._prev_force_avg_for_drop = curr
            return

        drop_ratio = abs(delta) / tf
        if drop_ratio < BROKEN_WIRE_DROP:
            self._prev_force_avg_for_drop = curr
            return

        if phase == "loading":
            self._holding_drop_watch = None
            if drop_ratio >= FORCE_DROP_PERCENT:
                alerts.append({
                    "type": "force_drop_alarm",
                    "severity": "critical",
                    "message": (
                        f"张拉阶段力值突降：单步下降 {abs(delta):.1f}kN（{drop_ratio*100:.1f}% 目标力），"
                        f"疑似设备或锚固异常，请核查。"
                        "【常见原因】断丝、油管爆裂、工具锚松动或千斤顶严重内泄；后果：该束力值急降，须停机泄压排查，避免带病锁定。"
                        "【隐患】除断丝外亦可能为油路失压、千斤顶内泄或工具锚松动；"
                        "若未处置继续张拉，控制力与伸长量双控失真，最终有效预应力不可控。"
                    ),
                })
            else:
                alerts.append({
                    "type": "force_drop_warning",
                    "severity": "warning",
                    "message": (
                        f"张拉阶段力值明显下降：单步下降 {abs(delta):.1f}kN（{drop_ratio*100:.1f}% 目标力）。"
                        "【常见原因】轻微滑丝、回缩偏大、油路瞬态或量测干扰；后果：表观力高于实际锁定力，后期损失偏大。"
                        "【隐患】提示回缩异常、轻微滑丝或测量/油路瞬态问题；"
                        "若属实易导致张拉记录与真实受力不符，后期预应力损失偏大。"
                    ),
                })
        elif phase == "holding":
            self._holding_drop_watch = {
                "t0": t,
                "anchor": curr,
                "fmin": curr,
                "fmax": curr,
            }

        self._prev_force_avg_for_drop = curr

    def check(self, features):
        """
        对当前数据点进行规则检测。
        返回结构保持不变：
        {
            "has_anomaly": bool,
            "alerts": list
        }
        """
        alerts = []
        phase = features.get("phase", "loading")
        force_rate = features.get("force_rate", 0)

        self._process_step_force_drop(features, alerts)

        # ===== 张拉阶段特有规则 =====
        if phase == "loading":
            # 未判为持荷前已超设计力：超张（与 RealTimeTensionMonitor 一致，不判欠张）
            alerts.extend(
                self._check_force_magnitude_alerts(features, check_under=False)
            )

            # 规则3：平均张拉速度异常（近 N 秒平均 kN/s，与 RealTimeTensionMonitor 一致）
            speed_kn_s = features.get("loading_avg_rate_kn_s")
            speed_alert = (
                self._check_average_loading_speed(speed_kn_s)
                if speed_kn_s is not None
                else None
            )
            if speed_alert is not None:
                alerts.append(speed_alert)

            # 规则4：刚度比异常（可能滑丝或卡住）
            stiffness = features.get("stiffness_ratio", 0)
            total_dis = features.get("total_delta_dis", 0)
            if total_dis > 1.0:  # 位移足够大时才判断
                if stiffness < self.thresholds["stiffness_ratio_min"]:
                    alerts.append({
                        "type": "low_stiffness",
                        "severity": "critical",
                        "message": (
                            f"刚度比过低({stiffness:.1f})，可能发生滑丝。"
                            "【常见原因】夹片未咬紧、钢绞线屈服或孔道异常宽松；后果：易欠张，滑丝可发展为脱锚。"
                            "【隐患】力已较高而伸长不足，常见为夹片跟进不良或回缩；"
                            "锁定后易形成较大预应力损失，裂缝与挠度控制裕度下降。"
                        ),
                    })
                elif stiffness > self.thresholds["stiffness_ratio_max"]:
                    alerts.append({
                        "type": "high_stiffness",
                        "severity": "warning",
                        "message": (
                            f"刚度比过高({stiffness:.1f})，可能夹片卡住。"
                            "【常见原因】孔道杂物卡阻、波纹管变形或绞线缠绕；后果：力到设计而伸长不足，有效预应力偏低。"
                            "【隐患】孔道摩阻异常增大或夹片咬死时，单侧超载与局部承压风险上升；"
                            "伸长量达标困难，与设计假定偏差大。"
                        ),
                    })
        else:
            # RealTimeTensionMonitor 中，非 loading 阶段会重置速度计数。
            self._reset_speed_state()
            if phase != "holding":
                self._reset_holding_force_judge_state()

        # ===== 持荷阶段特有规则 =====
        if phase == "holding":
            # 规则1–2：左右力值差异、两端速度差（仅持荷；张拉与卸载阶段不判）
            force_diff_ratio = features.get("force_diff_ratio", 0)
            force_diff = features.get("force_diff", 0)
            if force_diff_ratio > self.thresholds["force_diff_ratio_max"]:
                alerts.append({
                    "type": "force_imbalance",
                    "severity": "warning",
                    "message": (
                        f"持荷阶段左右力值不平衡：差异比={force_diff_ratio:.3f} "
                        f"(阈值{self.thresholds['force_diff_ratio_max']}), "
                        f"差值={force_diff:.1f}kN。"
                        "【常见原因】双泵供油不均、操作不同步或孔道摩阻不对称；后果：梁体扭转偏心、纵向裂缝风险上升。"
                        "【隐患】两端不同步使截面受扭、应力偏心，腹板或翼缘易出现斜裂；"
                        "长期表现为线形与内力分布与设计不符。"
                    ),
                })

            if force_diff > self.thresholds["force_diff_abs_max"]:
                alerts.append({
                    "type": "force_diff_excessive",
                    "severity": "critical",
                    "message": (
                        f"持荷阶段左右力值差异过大：{force_diff:.1f}kN > "
                        f"{self.thresholds['force_diff_abs_max']}kN。"
                        "【常见原因】一端严重泄漏或设备故障致双端力差极大；后果：单侧承压超载，须立即停拉纠偏。"
                        "【隐患】严重偏心张拉，一侧锚具与混凝土局部承压接近或超过极限，"
                        "压溃、锚板变形或扭转开裂风险高，应立即停拉排查。"
                    ),
                })

            force_rate_diff = features.get("force_rate_diff", 0)
            if force_rate_diff > self.thresholds["side_speed_diff_max"]:
                alerts.append({
                    "type": "side_speed_unsync",
                    "severity": "warning",
                    "message": (
                        f"持荷阶段两端速度不同步：左右力值变化率差={force_rate_diff:.2f}kN/step "
                        f"(阈值>{self.thresholds['side_speed_diff_max']:.2f}kN/step)。"
                        "【常见原因】手动不同步、油路进气/单端泄漏或一端千斤顶摩阻大；后果：附加弯扭、先到端松弛偏大。"
                        "【隐患】双端张拉“同伸同缩”失控，伸长与油压双控易失真，"
                        "转向块与接缝处应力峰值升高，局部开裂风险增加。"
                    ),
                })

            force_avg = features.get("force_avg", 0)

            # 记录进入持荷时的参考力值
            if self.holding_reference_force is None:
                self.holding_reference_force = force_avg

            # 持荷力值相对目标：欠张 / 超张 / 单端超张（满 DELAY 后 8s 窗中位数一次性判定）
            alerts.extend(self._collect_and_judge_holding_force_once(features))

            lower_ok = self.target_force * (1.0 - float(FORCE_LOWER_THRESHOLD))
            upper_ok = self.target_force * (1.0 + float(FORCE_THRESHOLD))

            # 规则5：持荷阶段力值下降
            if self.holding_reference_force > 0:
                drop_ratio = (
                    (self.holding_reference_force - force_avg)
                    / self.holding_reference_force
                )
                if drop_ratio > self.thresholds["holding_force_drop_max"]:
                    alerts.append({
                        "type": "holding_force_drop",
                        "severity": "critical",
                        "message": (
                            f"持荷阶段力值下降{drop_ratio*100:.1f}% "
                            f"(参考值={self.holding_reference_force:.1f}kN, "
                            f"当前={force_avg:.1f}kN)。"
                            "【常见原因】千斤顶内泄、保压阀失效、滑丝或夹片回缩偏大；后果：欠张，滑丝可发展为脱锚。"
                            "【隐患】提示锚具回缩、夹片松弛或油路保压不良；"
                            "锁定吨位低于设计时，使用阶段裂缝与挠度易超标。"
                        ),
                    })

            # 规则6：持荷阶段波动过大（仅当平均力已在目标允许区间内，避免欠张被误报为波动）。
            if self.target_force > 0 and lower_ok <= force_avg <= upper_ok:
                fluctuation = abs(force_avg - self.target_force) / self.target_force
                if fluctuation > self.thresholds["holding_force_fluctuation_max"]:
                    alerts.append({
                        "type": "hold_fluctuation",
                        "severity": "critical",
                        "message": (
                            f"持荷阶段波动过大：当前波动率 {fluctuation * 100:.1f}%，"
                            f"超过允许范围 {self.thresholds['holding_force_fluctuation_max'] * 100:.1f}% "
                            f"(目标力={self.target_force:.1f}kN, 当前={force_avg:.1f}kN)。"
                            "【常见原因】油泵脉动、油路进气、蓄能器失效或夹片微滑；后果：锁定力难认定，锚固评定争议大。"
                            "【隐患】保压系统不稳定导致真实锁定力难以认定，验收争议大；"
                            "亦可能掩盖缓慢滑丝或泄漏，后期预应力损失不确定。"
                        ),
                    })

        # ===== 卸载阶段特有规则 =====
        elif phase == "unloading":
            self._reset_holding_force_judge_state()
            # 规则7：卸载速度过快
            if force_rate < self.thresholds["unloading_rate_min"]:
                alerts.append({
                    "type": "unloading_too_fast",
                    "severity": "warning",
                    "message": (
                        f"卸载速度过快：{force_rate:.2f}kN/step "
                        f"(阈值{self.thresholds['unloading_rate_min']})。"
                        "【常见原因】回油阀开度过大、无节流或误操作；后果：冲击卸载易夹片崩出、锚固失效、端部混凝土受拉开裂。"
                        "【隐患】回油过快易对夹片与钢绞线产生冲击，带丝、错牙或锚圈变形；"
                        "回缩量读数失真，影响锚固质量评定。"
                    ),
                })

            # 重置持荷参考值
            self.holding_reference_force = None

        return {
            "has_anomaly": len(alerts) > 0,
            "alerts": alerts,
        }

    def reset(self):
        """重置状态"""
        self.holding_reference_force = None
        self._reset_speed_state()
        self._prev_force_avg_for_drop = None
        self._holding_drop_watch = None
        self._request_unloading_after_hold_drop = False
        self._reset_holding_force_judge_state()


class RealTimeTensionMonitor:
    def __init__(
        self,
        target_force,
        theory_dis=0.0,
        strand_id="N/A",
        speed_normal_low=SPEED_NORMAL_LOW,
        speed_normal_high=SPEED_NORMAL_HIGH,
    ):
        self.target_force = float(target_force)
        self.theory_dis = float(theory_dis)
        self.strand_id = strand_id

        self.speed_normal_low = float(speed_normal_low)
        self.speed_normal_high = float(speed_normal_high)
        self.speed_slow_limit = self.speed_normal_low * float(SPEED_SLOW_FACTOR)
        self.speed_fast_limit = self.speed_normal_high * float(SPEED_FAST_FACTOR)

        self.phase = "initial"
        self.history = []
        self.speed_history = []

        self.holding_started = False
        self.holding_finished = False
        self.hold_start_time = None
        self.hold_end_time = None
        self.hold_duration = 0.0

        self.unloading_started = False
        self.unloading_finished = False
        self.unload_start_time = None
        self.unload_end_time = None

        self.max_tension = 0.0
        self.last_displacement = 0.0

        self.speed_slow_counter = 0
        self.speed_fast_counter = 0
        self.speed_sudden_slow_counter = 0
        self.speed_sudden_fast_counter = 0

        self.side_imbalance_counter = 0

        self._mon_prev_force_drop = None
        self._mon_holding_drop_watch = None

        self.hold_time_alarm_reported = False
        self.unload_alarm_reported = False

    def _parse_time(self, value):
        if isinstance(value, datetime):
            return value

        value = str(value)

        for fmt in ("%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y/%m/%d %H:%M:%S"):
            try:
                return datetime.strptime(value, fmt)
            except ValueError:
                pass

        try:
            return datetime.fromisoformat(value)
        except ValueError:
            return datetime.now()

    def _make_result(
        self,
        status="normal",
        level=0,
        type_="normal",
        msg="运行正常",
        features=None,
    ):
        return {
            "status": status,
            "level": level,
            "type": type_,
            "msg": msg,
            "phase": self.phase,
            "features": features or {},
        }

    def _result_to_old_tuple(self, result):
        if result["status"] == "alarm":
            return "anomaly", result["msg"]

        if result["status"] == "warning":
            return "anomaly", result["msg"]

        return "normal", result["msg"]

    def _force_ratio(self, force):
        if self.target_force <= 0:
            return 0.0
        return float(force) / self.target_force

    def _force_imbalance_threshold(self):
        return max(
            FORCE_IMBALANCE_MIN_KN,
            self.target_force * FORCE_IMBALANCE_RATIO
        )

    def _update_phase_state(self, curr_time, tension):
        ratio = self._force_ratio(tension)

        if self.unloading_finished:
            self.phase = "finished"
            return None

        if not self.holding_started:
            if ratio < INITIAL_STAGE_RATIO:
                self.phase = "initial"
            elif ratio < HOLD_START_RATIO:
                self.phase = "loading"
            else:
                self.phase = "holding"
                self.holding_started = True
                self.hold_start_time = curr_time
                self.hold_duration = 0.0
            return None

        if self.holding_started and not self.unloading_started:
            if ratio >= HOLD_END_RATIO:
                self.phase = "holding"
                if self.hold_start_time is not None:
                    self.hold_duration = (
                        curr_time - self.hold_start_time
                    ).total_seconds()
                return None

            self.phase = "unloading"
            self.holding_finished = True
            self.unloading_started = True
            self.unload_start_time = curr_time
            self.hold_end_time = curr_time

            if self.hold_start_time is not None:
                self.hold_duration = (
                    self.hold_end_time - self.hold_start_time
                ).total_seconds()

            if self.hold_duration < HOLD_TIME_MIN and not self.hold_time_alarm_reported:
                self.hold_time_alarm_reported = True
                return self._make_result(
                    status="alarm",
                    level=2,
                    type_="hold_time_short",
                    msg=(
                        f"持荷时间不足：本次持荷 {self.hold_duration:.1f}s，"
                        f"低于要求的 {HOLD_TIME_MIN}s。"
                        "【常见原因】未达规定持荷即回油；后果：锁定力与伸长对应关系不可靠，易欠张。"
                        "【隐患】混凝土弹性压缩、锚具变形与部分松弛损失尚未充分稳定即卸载，"
                        "锁定吨位与伸长量对应关系存疑，易导致有效预应力低于设计、后期裂缝与挠度偏大。"
                    )
                )

            return self._make_result(
                status="normal",
                level=0,
                type_="holding_finished",
                msg=f"持荷完成：本次持荷 {self.hold_duration:.1f}s，进入卸载阶段。"
            )

        if self.unloading_started:
            self.phase = "unloading"

            if ratio <= UNLOAD_FINISH_RATIO:
                self.phase = "finished"
                self.unloading_finished = True
                self.unload_end_time = curr_time

                return self._make_result(
                    status="normal",
                    level=0,
                    type_="unload_finished",
                    msg=(
                        f"卸载完成：当前力值 {tension:.2f}kN，"
                        f"约为目标力的 {ratio * 100:.1f}%。"
                    )
                )

        return None

    def _append_history(self, features):
        self.history.append(features)

        if len(self.history) > 200:
            self.history.pop(0)

    def _check_force_limit(self, features):
        if self.phase == "holding" and float(self.hold_duration) < float(
            HOLDING_UNDER_OVER_JUDGE_DELAY_S
        ):
            return None
        force_avg = features["force_avg"]
        force_left = features["force_left"]
        force_right = features["force_right"]

        if force_avg > self.target_force * (1 + FORCE_THRESHOLD):
            return self._make_result(
                status="alarm",
                level=2,
                type_="over_tension",
                msg=(
                    f"平均张拉力超限：当前平均力 {force_avg:.2f}kN，"
                    f"超过目标力允许上限。"
                    "【常见原因】油压表偏高标定、误读油压或程序超张；后果：断丝、锚区压碎、反拱过大与端部开裂。"
                    "【隐患】材料与锚下局部承压逼近或超过极限，钢绞线屈服/断裂、混凝土压溃及线形上拱过大风险升高。"
                ),
                features=features,
            )

        max_side = max(force_left, force_right)

        if max_side > self.target_force * (1 + SIDE_FORCE_THRESHOLD):
            side = "左端" if force_left >= force_right else "右端"
            return self._make_result(
                status="alarm",
                level=2,
                type_="side_over_tension",
                msg=(
                    f"{side}单端张拉力超限：左端 {force_left:.2f}kN，"
                    f"右端 {force_right:.2f}kN。"
                    "【常见原因】两端不同步、单侧仪表/千斤顶故障或孔道摩阻极不均；后果：梁体侧弯扭转、预应力偏心。"
                    "【隐患】截面偏心受力，腹板斜裂、锚板压溃或扭转变形风险大。"
                ),
                features=features,
            )

        return None

    def _check_under_tension_hold(self, features):
        """持荷阶段平均力低于目标允许下限时判欠张（与 RuleEngine 同口径）。"""
        if self.phase != "holding" or self.target_force <= 0:
            return None
        if float(self.hold_duration) < float(HOLDING_UNDER_OVER_JUDGE_DELAY_S):
            return None
        force_avg = float(features["force_avg"])
        lower = self.target_force * (1.0 - float(FORCE_LOWER_THRESHOLD))
        if force_avg < lower:
            return self._make_result(
                status="alarm",
                level=2,
                type_="under_tension",
                msg=(
                    f"欠张拉：当前平均力 {force_avg:.1f}kN，"
                    f"低于目标力允许下限 {lower:.1f}kN（目标 {self.target_force:.1f}kN）。"
                    "【常见原因】油压表偏低标定、内摩阻未扣、孔道摩阻大或过早锚固；后果：开裂、下挠，承载储备不足。"
                    "【隐患】有效预应力不足，抗裂与刚度储备下降，使用阶段裂缝与挠度易超标。"
                ),
                features=features,
            )
        return None

    def _check_side_sync(self, features):
        # 与 RuleEngine 一致：以 feature_engine 的 phase 为准，仅在持荷判两端同步
        if features.get("phase") != "holding":
            self.side_imbalance_counter = 0
            return None

        force_left = features["force_left"]
        force_right = features["force_right"]
        force_diff = features["force_diff"]
        force_diff_ratio = features["force_diff_ratio"]
        force_rate_diff = abs(features.get("force_rate_diff", 0.0))

        threshold = self._force_imbalance_threshold()

        force_unbalanced = force_diff >= threshold
        speed_unbalanced = force_rate_diff >= max(
            self.speed_normal_low * SIDE_SPEED_DIFF_RATIO,
            0.5
        )

        if force_unbalanced:
            self.side_imbalance_counter += 1
        else:
            self.side_imbalance_counter = 0
            return None

        leading_side = "左端" if force_left > force_right else "右端"
        lagging_side = "右端" if force_left > force_right else "左端"

        if (
            self.side_imbalance_counter >= SIDE_IMBALANCE_ALARM_LIMIT
            or force_diff_ratio >= 0.12
            or (force_unbalanced and speed_unbalanced and self.side_imbalance_counter >= 3)
        ):
            return self._make_result(
                status="alarm",
                level=2,
                type_="side_unsync_alarm",
                msg=(
                    f"持荷阶段两端严重不同步：疑似{leading_side}力偏高、"
                    f"{lagging_side}偏低。左端 {force_left:.2f}kN，"
                    f"右端 {force_right:.2f}kN，力差 {force_diff:.2f}kN，"
                    f"速度差 {force_rate_diff:.2f}kN/s。"
                    "【常见原因】一端严重泄漏或同步失控；后果：单侧承压超载，须立即停拉。"
                    "【隐患】扭转与偏心显著，一侧锚固与混凝土局部承压超载，开裂与压溃风险高，应立即停拉纠偏。"
                ),
                features=features,
            )

        if self.side_imbalance_counter >= SIDE_IMBALANCE_WARNING_LIMIT:
            return self._make_result(
                status="warning",
                level=1,
                type_="side_unsync",
                msg=(
                    f"持荷阶段左右端不同步预警：左端 {force_left:.2f}kN，"
                    f"右端 {force_right:.2f}kN，力差 {force_diff:.2f}kN，"
                    f"已连续 {self.side_imbalance_counter} 个点偏大。"
                    "【常见原因】双泵供油不均、操作不同步；后果：扭转偏心、纵向裂缝风险上升。"
                    "【隐患】双控伸长与油压对应关系易偏离设计，截面应力不均，接缝与转向处开裂风险上升。"
                ),
                features=features,
            )

        return None

    def _check_global_speed(self, features):
        if self.phase != "loading":
            self.speed_slow_counter = 0
            self.speed_fast_counter = 0
            self.speed_sudden_slow_counter = 0
            self.speed_sudden_fast_counter = 0
            self.speed_history = []
            return None

        speed = features.get("loading_avg_rate_kn_s")
        if speed is None:
            return None
        force_rate = float(speed)

        if force_rate < self.speed_slow_limit:
            self.speed_slow_counter += 1
            self.speed_fast_counter = 0
        elif force_rate > self.speed_fast_limit:
            self.speed_fast_counter += 1
            self.speed_slow_counter = 0
        else:
            self.speed_slow_counter = 0
            self.speed_fast_counter = 0

        if self.speed_fast_counter >= SPEED_CONSECUTIVE_LIMIT:
            return self._make_result(
                status="alarm",
                level=2,
                type_="speed_too_fast",
                msg=(
                    f"整体张拉速度持续过快（近{LOADING_SPEED_AVERAGE_WINDOW_S:.0f}s平均）：当前速度 {force_rate:.2f}kN/s，"
                    f"高于阈值 {self.speed_fast_limit:.2f}kN/s。"
                    "【常见原因】未分级升压、油泵流量过大或误快进；后果：冲击滑丝/断丝、锚区压碎、读数失真。"
                    "【隐患】冲击与局部应力集中，夹片滑丝、咬丝及端部混凝土劈裂风险升高；双控校核困难。"
                ),
                features=features,
            )

        if self.speed_slow_counter >= SPEED_CONSECUTIVE_LIMIT:
            return self._make_result(
                status="warning",
                level=1,
                type_="speed_too_slow",
                msg=(
                    f"整体张拉速度持续过慢（近{LOADING_SPEED_AVERAGE_WINDOW_S:.0f}s平均）：当前速度 {force_rate:.2f}kN/s，"
                    f"低于阈值 {self.speed_slow_limit:.2f}kN/s。"
                    "【常见原因】供油不足、滤网堵或千斤顶摩阻大；后果：油温升高损密封、松弛与回缩偏大。"
                    "【隐患】长时间高压不利密封与油温稳定，摩阻读数漂移，真实控制力与记录易偏离。"
                ),
                features=features,
            )

        if len(self.speed_history) >= SPEED_JUMP_REFERENCE_COUNT:
            recent = self.speed_history[-SPEED_JUMP_REFERENCE_COUNT:]
            recent_avg = sum(recent) / len(recent)

            if recent_avg > 0:
                if force_rate < recent_avg * SPEED_SUDDEN_SLOW_RATIO:
                    self.speed_sudden_slow_counter += 1
                    self.speed_sudden_fast_counter = 0

                elif force_rate > recent_avg * SPEED_SUDDEN_FAST_RATIO:
                    self.speed_sudden_fast_counter += 1
                    self.speed_sudden_slow_counter = 0

                else:
                    self.speed_sudden_slow_counter = 0
                    self.speed_sudden_fast_counter = 0

                if self.speed_sudden_slow_counter >= SPEED_JUMP_CONSECUTIVE_LIMIT:
                    return self._make_result(
                        status="warning",
                        level=1,
                        type_="speed_sudden_slow",
                        msg=(
                            f"整体张拉速度突然下降（近{LOADING_SPEED_AVERAGE_WINDOW_S:.0f}s平均）：当前速度 {force_rate:.2f}kN/s，"
                            f"近期均值 {recent_avg:.2f}kN/s。"
                            "【常见原因】油路泄漏、密封失效或孔道卡阻突变；后果：力—伸长不对应，易误判或掩盖风险。"
                            "【隐患】油路或摩阻状态突变，力—伸长关系易失真；若伴松脱则控制力不足风险上升。"
                        ),
                        features=features,
                    )

                if self.speed_sudden_fast_counter >= SPEED_JUMP_CONSECUTIVE_LIMIT:
                    return self._make_result(
                        status="alarm",
                        level=2,
                        type_="speed_sudden_fast",
                        msg=(
                            f"整体张拉速度突然升高（近{LOADING_SPEED_AVERAGE_WINDOW_S:.0f}s平均）：当前速度 {force_rate:.2f}kN/s，"
                            f"近期均值 {recent_avg:.2f}kN/s。"
                            "【常见原因】阀门突开、误快进或泵况突变；后果：冲击断丝/超张、锚混凝土压碎。"
                            "【隐患】冲击升载损伤齿口与夹片跟进，滑丝、断丝及局部承压恶化风险显著。"
                        ),
                        features=features,
                    )

        self.speed_history.append(float(force_rate))

        if len(self.speed_history) > 20:
            self.speed_history.pop(0)

        return None

    def _check_force_drop(self, features):
        """
        单步突降：与 RuleEngine 一致，张拉立即告警；持荷经观察窗区分滑丝与卸载性回落。
        """
        tf = self.target_force
        if tf <= 0:
            return None

        phase_fe = features.get("phase", "loading")
        t = float(features.get("time", 0.0))
        curr = float(features.get("force_avg", 0.0) or 0.0)

        if self.unloading_started or phase_fe == "unloading":
            self._mon_holding_drop_watch = None
            self._mon_prev_force_drop = curr
            return None

        if curr / tf < INITIAL_STAGE_RATIO:
            self._mon_holding_drop_watch = None
            self._mon_prev_force_drop = curr
            return None

        if self._mon_holding_drop_watch is not None:
            if phase_fe != "holding":
                self._mon_holding_drop_watch = None
            else:
                w = self._mon_holding_drop_watch
                w["fmin"] = min(w["fmin"], curr)
                w["fmax"] = max(w["fmax"], curr)
                if t - w["t0"] >= FORCE_DROP_POST_CONFIRM_S:
                    anchor = w["anchor"]
                    band = w["fmax"] - w["fmin"]
                    continued = (anchor - w["fmin"]) >= FORCE_DROP_CONTINUE_RATIO * tf
                    stable = band <= FORCE_DROP_STABLE_BAND_RATIO * tf
                    self._mon_holding_drop_watch = None
                    if continued:
                        self._mon_prev_force_drop = curr
                        return None
                    if stable:
                        self._mon_prev_force_drop = curr
                        return self._make_result(
                            status="alarm",
                            level=2,
                            type_="force_drop_alarm",
                            msg=(
                                f"持荷突降后约{FORCE_DROP_POST_CONFIRM_S:.0f}s 力值在突降后水平附近平稳："
                                f"疑似断丝/滑丝（锚点 {anchor:.1f}kN，窗内起伏 {band:.1f}kN）。"
                                "【常见原因】断丝、严重滑丝或锚固异常；后果：有效预应力骤降，须停机确认是否退锚。"
                                "【隐患】该束预应力突然损失，内力重分布，裂缝与挠度控制削弱；"
                                "若继续发展可演变为逐根滑脱。"
                            ),
                            features=features,
                        )
                    self._mon_prev_force_drop = curr
                    return None

        prev = self._mon_prev_force_drop
        if prev is None:
            self._mon_prev_force_drop = curr
            return None

        delta = curr - prev
        if delta >= 0:
            self._mon_prev_force_drop = curr
            return None

        drop_ratio = abs(delta) / tf
        if drop_ratio < BROKEN_WIRE_DROP:
            self._mon_prev_force_drop = curr
            return None

        if phase_fe == "loading":
            self._mon_holding_drop_watch = None
            self._mon_prev_force_drop = curr
            if drop_ratio >= FORCE_DROP_PERCENT:
                return self._make_result(
                    status="alarm",
                    level=2,
                    type_="force_drop_alarm",
                    msg=(
                        f"张拉阶段力值突降：单步下降 {abs(delta):.1f}kN（{drop_ratio*100:.1f}% 目标力），"
                        f"疑似设备或锚固异常，请核查。"
                        "【常见原因】断丝、油管爆裂、工具锚松动或严重内泄；后果：力值急降，须停机泄压排查。"
                        "【隐患】油路失压、内泄或工具锚松动等均可能；不处置则双控失真，最终有效预应力不可控。"
                    ),
                    features=features,
                )
            return self._make_result(
                status="warning",
                level=1,
                type_="force_drop_warning",
                msg=(
                    f"张拉阶段力值明显下降：单步下降 {abs(delta):.1f}kN（{drop_ratio*100:.1f}% 目标力）。"
                    "【常见原因】轻微滑丝、回缩偏大或油路/量测瞬态；后果：记录力高于实际锁定力。"
                    "【隐患】提示轻微滑丝、回缩异常或测量瞬态问题，若属实后期预应力损失偏大。"
                ),
                features=features,
            )

        if phase_fe == "holding":
            self._mon_holding_drop_watch = {
                "t0": t,
                "anchor": curr,
                "fmin": curr,
                "fmax": curr,
            }
        self._mon_prev_force_drop = curr
        return None

    def _check_holding_fluctuation(self, features):
        if self.phase != "holding":
            return None

        force_avg = features["force_avg"]
        if self.target_force > 0:
            lower_ok = self.target_force * (1.0 - float(FORCE_LOWER_THRESHOLD))
            upper_ok = self.target_force * (1.0 + float(FORCE_THRESHOLD))
            if not (lower_ok <= force_avg <= upper_ok):
                return None

        fluctuation = abs(force_avg - self.target_force) / self.target_force

        if fluctuation > FORCE_FLUCTUATION:
            return self._make_result(
                status="alarm",
                level=2,
                type_="hold_fluctuation",
                msg=(
                    f"持荷阶段波动过大：当前波动率 {fluctuation * 100:.1f}%，"
                    f"超过允许范围。"
                    "【常见原因】油泵脉动、油路进气或蓄能器失效；后果：锁定力难认定，锚固评定争议大。"
                    "【隐患】保压不稳使锁定力难以认定，亦可能掩盖缓慢滑丝或泄漏，后期损失不确定。"
                ),
                features=features,
            )

        return None

    def _normalize_features_from_args(
        self,
        current_time_str,
        tension=None,
        displacement=0.0,
        force_left=None,
        force_right=None,
        dis_left=None,
        dis_right=None,
    ):
        if tension is None:
            if force_left is None or force_right is None:
                raise ValueError("tension 为空时必须提供 force_left 和 force_right")
            tension = (float(force_left) + float(force_right)) / 2

        if force_left is None:
            force_left = tension

        if force_right is None:
            force_right = tension

        dis_left = 0.0 if dis_left is None else float(dis_left)
        dis_right = 0.0 if dis_right is None else float(dis_right)

        force_left = float(force_left)
        force_right = float(force_right)
        force_avg = float(tension)
        displacement = float(displacement)

        return {
            "time": current_time_str,
            "time_dt": self._parse_time(current_time_str),

            "force_left": force_left,
            "force_right": force_right,
            "force_avg": force_avg,
            "force_sum": force_left + force_right,
            "force_diff": abs(force_left - force_right),
            "force_diff_ratio": abs(force_left - force_right) / (abs(force_avg) + 1e-6),

            "dis_left": dis_left,
            "dis_right": dis_right,
            "total_delta_dis": displacement,
            "dis_diff": abs(dis_left - dis_right),

            "left_force_rate": 0.0,
            "right_force_rate": 0.0,
            "force_rate": 0.0,
            "force_rate_diff": 0.0,
            "force_rate_ratio": 0.0,

            "left_force_acc": 0.0,
            "right_force_acc": 0.0,
            "force_acc": 0.0,

            "left_dis_rate": 0.0,
            "right_dis_rate": 0.0,
            "dis_rate_diff": 0.0,

            "stiffness_ratio": force_avg / (displacement + 1e-6),
        }

    def _check_point_dict(self, features):
        features = dict(features)

        if "time_dt" not in features:
            features["time_dt"] = self._parse_time(features["time"])

        force_avg = float(features["force_avg"])
        total_delta_dis = float(features.get("total_delta_dis", 0.0))

        self.max_tension = max(self.max_tension, force_avg)
        self.last_displacement = total_delta_dis

        phase_result = self._update_phase_state(features["time_dt"], force_avg)

        checks = [
            self._check_force_limit,
            self._check_under_tension_hold,
            self._check_side_sync,
            self._check_global_speed,
            self._check_force_drop,
            self._check_holding_fluctuation,
        ]

        for check in checks:
            result = check(features)
            if result is not None:
                self._append_history(features)
                return result

        if phase_result is not None:
            self._append_history(features)
            phase_result["features"] = features
            return phase_result

        self._append_history(features)

        return self._make_result(
            status="normal",
            level=0,
            type_="normal",
            msg="运行正常",
            features=features,
        )

    def check_point(
        self,
        current_time_str=None,
        tension=None,
        displacement=0.0,
        force_left: Optional[float] = None,
        force_right: Optional[float] = None,
        dis_left: Optional[float] = None,
        dis_right: Optional[float] = None,
        return_dict: bool = False,
    ):
        """
        对单个数据点进行规则检测。

        支持两种调用形式：
        1. 直接传入特征字典: engine.check_point(features_dict)
        2. 分字段传入: engine.check_point(current_time_str=..., tension=..., ...)

        返回:
            若 return_dict=True，返回检测结果字典；
            否则返回 (status, msg) 元组，兼容旧版接口。
        """
        if isinstance(current_time_str, dict):
            result = self._check_point_dict(current_time_str)
            if return_dict:
                return result
            return self._result_to_old_tuple(result)

        features = self._normalize_features_from_args(
            current_time_str=current_time_str,
            tension=tension,
            displacement=displacement,
            force_left=force_left,
            force_right=force_right,
            dis_left=dis_left,
            dis_right=dis_right,
        )

        result = self._check_point_dict(features)

        if return_dict:
            return result

        return self._result_to_old_tuple(result)

    def final_check(self):
        results = []

        if self.max_tension < self.target_force * HOLD_START_RATIO:
            results.append({
                "status": "alarm",
                "level": 2,
                "type": "under_tension",
                "msg": (
                    f"目标力未达成：全过程最大力值 {self.max_tension:.2f}kN，"
                    f"未达到目标力的 {HOLD_START_RATIO * 100:.0f}%。"
                    "【常见原因】张拉力未加至设计、摩阻过大或设备标定问题；后果：承载与抗裂储备不足，难验收。"
                    "【隐患】本束有效预应力明显低于设计，抗裂与承载储备不足，不满足张拉工艺与验收要求。"
                )
            })

        if not self.holding_started:
            results.append({
                "status": "alarm",
                "level": 2,
                "type": "no_holding_stage",
                "msg": (
                    "未进入有效持荷阶段。"
                    "【常见原因】未稳定在设计力区段即卸载或工序跳步；后果：回缩与锁定吨位不可评定，验收风险大。"
                    "【隐患】缺少规定持荷无法稳定观测回缩与锁定吨位，应力损失与伸长双控结论不可靠，归档数据难以通过验收。"
                )
            })

        if self.holding_started and self.hold_duration < HOLD_TIME_MIN:
            results.append({
                "status": "alarm",
                "level": 2,
                "type": "hold_time_insufficient",
                "msg": (
                    f"持荷时间不足：最终持荷时间 {self.hold_duration:.1f}s，"
                    f"低于要求的 {HOLD_TIME_MIN}s。"
                    "【常见原因】持荷时间人为缩短或提前回油；后果：锁定力偏低，后期开裂与下挠风险上升。"
                    "【隐患】弹性压缩与锚具变形未充分完成即锁定或卸载，有效预应力易偏低，使用阶段裂缝与挠度风险上升。"
                )
            })

        if (
            self.holding_started
            and self.hold_duration >= HOLD_TIME_MIN
            and not self.unloading_finished
        ):
            results.append({
                "status": "warning",
                "level": 1,
                "type": "unload_not_finished",
                "msg": (
                    f"未检测到卸载完成阶段：当前阶段 {self.phase}，"
                    f"卸载完成标准为力值降至目标力的 "
                    f"{UNLOAD_FINISH_RATIO * 100:.0f}% 以下。"
                    "【常见原因】未回油至规定比例或采集中断；后果：工序未闭环，尚余较高张拉力时安全风险与验收争议。"
                    "【隐患】工序未闭环，回缩量与夹片就位状态未处于可评定工况，数据链不完整；"
                    "若现场力值仍偏高，尚存在未安全释放张拉力的施工风险。"
                )
            })

        if not results:
            results.append({
                "status": "normal",
                "level": 0,
                "type": "final_ok",
                "msg": (
                    f"本次张拉过程完成：最大力值 {self.max_tension:.2f}kN，"
                    f"持荷时间 {self.hold_duration:.1f}s。"
                )
            })

        return results