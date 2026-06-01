"""
Flask 主应用
提供 WebSocket 实时推送和 REST API
"""
import os
import time
import threading
import statistics
from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO

from config import (
    FLASK_HOST,
    FLASK_PORT,
    FLASK_DEBUG,
    PUSH_INTERVAL,
    REALTIME_CHART_MAX_POINTS,
    HOLDING_STABLE_MEDIAN_SKIP_S,
    resolve_simulation_csv_path,
)
from database import (
    init_database,
    create_session,
    finish_session,
    insert_data_point,
    insert_anomaly,
    get_all_sessions,
    get_session_detail,
    delete_session,
)
from feature_engine import OnlineFeatureBuilder
from model_detector import IFLOFDetector
from trend_predictor import TrendPredictor
from rule_engine import RuleEngine
from data_simulator import DataSimulator

# 初始化 Flask
app = Flask(__name__)
app.config["SECRET_KEY"] = "tension_monitor_secret"
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

# 全局状态
tension_running = False
tension_thread = None
current_session_id = None


# ==================== 页面路由 ====================

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/realtime")
def realtime():
    return render_template(
        "realtime.html",
        chart_max_points=REALTIME_CHART_MAX_POINTS,
        holding_stable_skip_s=HOLDING_STABLE_MEDIAN_SKIP_S,
    )


@app.route("/history")
def history():
    return render_template("history.html")

@app.route("/history/<int:session_id>")
def session_detail(session_id):
    return render_template("session_detail.html", session_id=session_id)

# ==================== REST API ====================

@app.route("/api/start_tension", methods=["POST"])
def start_tension():
    """开始张拉"""
    global tension_running, tension_thread, current_session_id

    if tension_running:
        return jsonify({"status": "error", "message": "张拉正在进行中"}), 400

    data = request.get_json() or {}
    target_force = float(data.get("target_force", 1000.0))
    csv_path = resolve_simulation_csv_path()
    if csv_path:
        print(f"[App] 模拟数据源: {csv_path}")
    else:
        print("[App] 未找到配置的模拟 CSV，使用合成数据")

    # 创建数据库会话
    current_session_id = create_session(target_force)

    # 启动张拉线程
    tension_running = True
    tension_thread = threading.Thread(
        target=_tension_worker,
        args=(current_session_id, target_force, csv_path),
        daemon=True,
    )
    tension_thread.start()

    return jsonify({
        "status": "ok",
        "session_id": current_session_id,
        "target_force": target_force,
    })


@app.route("/api/stop_tension", methods=["POST"])
def stop_tension():
    """手动停止张拉"""
    global tension_running
    tension_running = False
    return jsonify({"status": "ok", "message": "已发送停止信号"})


@app.route("/api/history/sessions", methods=["GET"])
def api_sessions():
    """获取所有张拉会话列表"""
    sessions = get_all_sessions()
    return jsonify({"status": "ok", "sessions": sessions})


@app.route("/api/history/session/<int:session_id>", methods=["GET"])
def api_session_detail(session_id):
    """获取某次张拉的详细数据"""
    detail = get_session_detail(session_id)
    if detail is None:
        return jsonify({"status": "error", "message": "会话不存在"}), 404
    return jsonify({"status": "ok", "data": detail})


@app.route("/api/history/session/<int:session_id>", methods=["DELETE"])
def api_delete_session(session_id):
    """删除某次张拉会话及其关联数据"""
    n = delete_session(session_id)
    if n == 0:
        return jsonify({"status": "error", "message": "会话不存在"}), 404
    return jsonify({"status": "ok", "message": "已删除"})


# ==================== 张拉工作线程 ====================

def _tension_worker(session_id, target_force, csv_path=None):
    """张拉数据处理工作线程"""
    global tension_running

    # 初始化各模块（csv_path 为绝对路径，由 resolve_simulation_csv_path 解析）
    simulator = DataSimulator(csv_path=csv_path, target_force=target_force)
    feature_builder = OnlineFeatureBuilder(window_size=10, target_force=target_force)
    if_lof_detector = IFLOFDetector()
    trend_predictor = TrendPredictor(
        model_dir="models"
    )
    trend_predictor._target_force = target_force  # 传入目标力值
    rule_monitor = RuleEngine(target_force=target_force)

    total_points = 0
    anomaly_count = 0
    max_force_holding = None
    holding_stable_forces = []
    holding_stable_elongations = []
    ever_holding = False
    critical_anomaly_count = 0

    print(f"[Worker] 张拉开始，会话ID={session_id}，目标力值={target_force}")

    while tension_running:
        # 获取下一个数据点
        raw_point = simulator.next_point()
        if raw_point is None:
            break

        # 计算特征
        features = feature_builder.update(raw_point)
        total_points += 1

        anomalies = []

        # === 规则引擎（含单步突降：张拉立即告警；持荷 2s 确认后区分滑丝与卸载） ===
        rule_result = rule_monitor.check(features)
        if rule_monitor.consume_unloading_after_hold_drop():
            feature_builder.force_transition_to_unloading()
            features["phase"] = feature_builder.phase
        phase = features["phase"]

        if phase == "holding":
            ever_holding = True
            fa = float(features["force_avg"])
            max_force_holding = (
                fa if max_force_holding is None else max(max_force_holding, fa)
            )
            het = float(features.get("holding_elapsed_time", 0.0))
            if het >= HOLDING_STABLE_MEDIAN_SKIP_S:
                holding_stable_forces.append(fa)
                holding_stable_elongations.append(
                    float(features.get("total_delta_dis", 0.0))
                )

        # 存储数据点（阶段可能与规则引擎卸载判定对齐）
        insert_data_point(session_id, {
            "time_offset": features["time"],
            "force_left": features["force_left"],
            "force_right": features["force_right"],
            "force_avg": features["force_avg"],
            "dis_left": features["dis_left"],
            "dis_right": features["dis_right"],
            "total_delta_dis": features["total_delta_dis"],
            "phase": phase,
        })
        if rule_result["has_anomaly"]:
            for alert in rule_result["alerts"]:
                anomalies.append({
                    "time_offset": features["time"],
                    "phase": phase,
                    "source": "rule_engine",
                    "anomaly_type": alert["type"],
                    "severity": alert.get("severity", "warning"),
                    "detail": alert.get("message", ""),
                    "force_avg": features["force_avg"],
                    "force_diff": features["force_diff"],
                })

        # === IF/LOF 检测（仅张拉阶段） ===
        if phase == "loading":
            iflof_result = if_lof_detector.detect(features)
            if iflof_result["is_anomaly"]:
                anomalies.append({
                    "time_offset": features["time"],
                    "phase": phase,
                    "source": "if_lof",
                    "anomaly_type": "statistical_anomaly",
                    "severity": "warning",
                    "detail": (
                        f"IF分数={iflof_result['if_score']:.4f}, "
                        f"LOF分数={iflof_result['lof_score']:.4f}。"
                        "【常见原因】油路/摩阻突变、传感器漂移或隐蔽滑丝未在单点规则显现；后果：双控结论失真，带病锁定致后期预应力不确定。"
                        "【隐患】张拉曲线形态偏离历史正常模式，可能对应油路/摩阻突变、采集异常或隐蔽滑丝等；"
                        "应结合伸长量、油压与现场检查复核，避免带病锁定。"
                    ),
                    "force_avg": features["force_avg"],
                    "force_diff": features["force_diff"],
                })

        # === TCN 趋势预测（仅张拉阶段） ===
        tcn_result = None
        if phase == "loading":
            # 把目标力值塞进 features，供 feed_point 归一化用
            features["_target_force"] = target_force
            trend_predictor.feed_point(features)
            tcn_result = trend_predictor.predict()
            if tcn_result and tcn_result["is_anomaly"]:
                anomalies.append({
                    "time_offset": features["time"],
                    "phase": phase,
                    "source": "tcn",
                    "anomaly_type": "trend_anomaly",
                    "severity": "warning",
                    "detail": (
                        f"残差={tcn_result['residual']:.4f}, "
                        f"预测力值={tcn_result['predicted_force']}。"
                        "【常见原因】设备状态或摩阻与张拉规律不同步变化；后果：力—伸长双控偏离设计，锁定吨位与伸长结论不可靠。"
                        "【隐患】短期趋势与正常张拉规律不一致，若伴随设备或摩阻问题未处理，"
                        "可能导致控制力与伸长双控偏离设计，最终有效预应力不确定。"
                    ),
                    "force_avg": features["force_avg"],
                    "force_diff": features["force_diff"],
                })

        # 存储异常到数据库
        for anomaly in anomalies:
            insert_anomaly(session_id, anomaly)
            anomaly_count += 1
            if (anomaly.get("severity") or "").lower() == "critical":
                critical_anomaly_count += 1

        # 通过 WebSocket 推送实时数据
        emit_data = {
            "time": features["time"],
            "force_left": features["force_left"],
            "force_right": features["force_right"],
            "force_avg": features["force_avg"],
            "dis_left": features["dis_left"],
            "dis_right": features["dis_right"],
            "total_delta_dis": features["total_delta_dis"],
            "phase": phase,
            "target_force": target_force,
            "progress": simulator.progress,
            "anomalies": anomalies,
            "has_anomaly": len(anomalies) > 0,
        }

        # 如果 TCN 有预测结果，附加上
        if phase == "loading" and tcn_result:
            emit_data["tcn_prediction"] = tcn_result["predicted_force"]
            emit_data["tcn_residual"] = tcn_result["residual"]

        socketio.emit("tension_data", emit_data)

        # 控制推送频率 — 使用 PUSH_INTERVAL（已改为1.0秒）
        time.sleep(PUSH_INTERVAL)

    # 张拉结束
    tension_running = False

    holding_median_force_kn = None
    if holding_stable_forces:
        holding_median_force_kn = statistics.median(holding_stable_forces)

    holding_median_elongation_mm = None
    if holding_stable_elongations:
        holding_median_elongation_mm = statistics.median(holding_stable_elongations)

    holding_deviation_pct = None
    if target_force > 1e-9 and holding_median_force_kn is not None:
        holding_deviation_pct = (
            (holding_median_force_kn - target_force) / target_force * 100.0
        )

    finish_session(
        session_id,
        total_points,
        anomaly_count,
        critical_anomaly_count=critical_anomaly_count,
        final_elongation_mm=holding_median_elongation_mm,
        holding_median_force_kn=holding_median_force_kn,
        holding_force_deviation_pct=holding_deviation_pct,
    )

    socketio.emit("tension_complete", {
        "session_id": session_id,
        "total_points": total_points,
        "anomaly_count": anomaly_count,
        "message": "张拉完成",
        "final_elongation_mm": holding_median_elongation_mm,
        "holding_max_force_kn": max_force_holding,
        "holding_median_force_kn": holding_median_force_kn,
        "holding_median_sample_count": len(holding_stable_forces),
        "holding_stable_skip_s": HOLDING_STABLE_MEDIAN_SKIP_S,
        "had_holding_phase": ever_holding,
        "holding_force_deviation_pct": holding_deviation_pct,
        "critical_anomaly_count": critical_anomaly_count,
    })

    print(f"[Worker] 张拉结束，共 {total_points} 点，{anomaly_count} 条异常")


# ==================== 启动 ====================

if __name__ == "__main__":
    # 初始化数据库
    init_database()
    print("[App] 数据库初始化完成")
    print(f"[App] 启动服务 http://{FLASK_HOST}:{FLASK_PORT}")
    socketio.run(app, host=FLASK_HOST, port=FLASK_PORT, debug=FLASK_DEBUG, allow_unsafe_werkzeug=True)