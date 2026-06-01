/**
 * 实时张拉监测页面逻辑
 * 使用 Socket.IO 接收实时数据，Chart.js 绘制图表
 * 包含 TCN 预测曲线显示
 */

function cssVar(name, fallback) {
    var v = getComputedStyle(document.documentElement).getPropertyValue(name);
    v = (v || "").trim();
    return v || fallback;
}

function chartCommonFont() {
    return { size: 11, family: "Noto Sans SC, system-ui, sans-serif" };
}

function setConnStatus(connected) {
    var el = document.getElementById("conn-status");
    if (!el) return;
    el.classList.remove("conn-ok", "conn-bad", "conn-warn");
    if (connected) {
        el.textContent = "数据通道已连接";
        el.classList.add("conn-ok");
    } else {
        el.textContent = "连接已断开";
        el.classList.add("conn-bad");
    }
}

function tickClock() {
    var el = document.getElementById("topbar-clock");
    if (!el) return;
    var d = new Date();
    var pad = function (n) { return n < 10 ? "0" + n : String(n); };
    el.textContent =
        pad(d.getHours()) + ":" + pad(d.getMinutes()) + ":" + pad(d.getSeconds());
}

// Socket.IO 连接
const socket = io();

if (socket.connected) {
    setConnStatus(true);
}

setInterval(tickClock, 1000);
tickClock();

// 图表数据缓冲最大点数（由后端 realtime 页注入，缺省 500）
const MAX_POINTS = (function () {
    var n = (window.APP_CONFIG && window.APP_CONFIG.realtimeChartMaxPoints) || 500;
    n = parseInt(n, 10);
    return !isNaN(n) && n > 0 ? n : 500;
})();

var C_FORCE_L = cssVar("--chart-force-l", "#e11d48");
var C_FORCE_R = cssVar("--chart-force-r", "#2563eb");
var C_FORCE_AVG = cssVar("--chart-force-avg", "#059669");
var C_TCN = cssVar("--chart-tcn", "#d97706");
var C_DIS_TOTAL = cssVar("--chart-dis-total", "#7c3aed");
var C_GRID = "rgba(15, 23, 42, 0.06)";
var C_TEXT = cssVar("--ink-muted", "#64748b");

const chartData = {
    labels: [],
    forceLeft: [],
    forceRight: [],
    forceAvg: [],
    disLeft: [],
    disRight: [],
    totalDis: [],
};

// TCN 预测相关状态（独立管理，不污染主数据）
let tcnPredictionData = {
    startIndex: -1,         // 预测起始点在 labels 中的索引
    startTime: 0,           // 预测起始时间
    values: [],             // 预测力值数组
    stepInterval: 1.0,      // 每步时间间隔（秒）
};

// 异常计数（规则 / IF·LOF 等与 TCN 分列）
let anomalyCount = 0;
let tcnAnomalyCount = 0;

// ==================== 图表初始化 ====================

const forceChart = new Chart(document.getElementById("force-chart"), {
    type: "line",
    data: {
        labels: chartData.labels,
        datasets: [
            {
                label: "左侧力值",
                data: chartData.forceLeft,
                borderColor: C_FORCE_L,
                borderWidth: 1.5,
                pointRadius: 0,
                fill: false,
                tension: 0.12,
            },
            {
                label: "右侧力值",
                data: chartData.forceRight,
                borderColor: C_FORCE_R,
                borderWidth: 1.5,
                pointRadius: 0,
                fill: false,
                tension: 0.12,
            },
            {
                label: "平均力值",
                data: chartData.forceAvg,
                borderColor: C_FORCE_AVG,
                borderWidth: 2,
                pointRadius: 0,
                fill: false,
                tension: 0.12,
            },
            {
                label: "TCN 预测",
                data: [],
                borderColor: C_TCN,
                borderWidth: 2,
                borderDash: [6, 4],
                pointRadius: 0,
                pointHoverRadius: 4,
                pointBackgroundColor: C_TCN,
                fill: false,
                tension: 0.12,
                spanGaps: false,
            },
        ],
    },
    options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: false,
        interaction: {
            intersect: false,
            mode: "index",
        },
        scales: {
            x: {
                title: {
                    display: true,
                    text: "时间 (s)",
                    font: chartCommonFont(),
                    color: C_TEXT,
                },
                ticks: { maxTicksLimit: 20, color: C_TEXT, font: chartCommonFont() },
                grid: { color: C_GRID },
            },
            y: {
                title: {
                    display: true,
                    text: "力值 (kN)",
                    font: chartCommonFont(),
                    color: C_TEXT,
                },
                beginAtZero: true,
                ticks: { color: C_TEXT, font: chartCommonFont() },
                grid: { color: C_GRID },
            },
        },
        plugins: {
            legend: {
                position: "top",
                labels: {
                    usePointStyle: true,
                    padding: 14,
                    font: chartCommonFont(),
                    color: C_TEXT,
                },
            },
            tooltip: { enabled: true },
        },
    },
});

const disChart = new Chart(document.getElementById("displacement-chart"), {
    type: "line",
    data: {
        labels: chartData.labels,
        datasets: [
            {
                label: "左侧位移",
                data: chartData.disLeft,
                borderColor: C_FORCE_L,
                borderWidth: 1.5,
                pointRadius: 0,
                fill: false,
                tension: 0.12,
            },
            {
                label: "右侧位移",
                data: chartData.disRight,
                borderColor: C_FORCE_R,
                borderWidth: 1.5,
                pointRadius: 0,
                fill: false,
                tension: 0.12,
            },
            {
                label: "总位移",
                data: chartData.totalDis,
                borderColor: C_DIS_TOTAL,
                borderWidth: 2,
                pointRadius: 0,
                fill: false,
                tension: 0.12,
            },
        ],
    },
    options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: false,
        interaction: {
            intersect: false,
            mode: "index",
        },
        scales: {
            x: {
                title: {
                    display: true,
                    text: "时间 (s)",
                    font: chartCommonFont(),
                    color: C_TEXT,
                },
                ticks: { maxTicksLimit: 20, color: C_TEXT, font: chartCommonFont() },
                grid: { color: C_GRID },
            },
            y: {
                title: {
                    display: true,
                    text: "位移 (mm)",
                    font: chartCommonFont(),
                    color: C_TEXT,
                },
                beginAtZero: true,
                ticks: { color: C_TEXT, font: chartCommonFont() },
                grid: { color: C_GRID },
            },
        },
        plugins: {
            legend: {
                position: "top",
                labels: {
                    usePointStyle: true,
                    padding: 14,
                    font: chartCommonFont(),
                    color: C_TEXT,
                },
            },
        },
    },
});

// ==================== Socket.IO 事件监听 ====================

socket.on("connect", function () {
    console.log("[Socket] 已连接到服务器");
    setConnStatus(true);
});

socket.on("disconnect", function () {
    console.log("[Socket] 连接断开");
    setConnStatus(false);
});

socket.on("tension_data", function (data) {
    // 更新图表数据
    updateChartData(data);

    // 更新状态栏
    updateStatusBar(data);

    // 处理异常
    if (data.has_anomaly && data.anomalies.length > 0) {
        addAnomalies(data.anomalies);
    }
});

socket.on("tension_complete", function (data) {
    console.log("[Socket] 张拉完成:", data);
    document.getElementById("btn-start").disabled = false;
    document.getElementById("btn-stop").disabled = true;
    document.getElementById("current-phase").textContent = "已完成";
    document.getElementById("current-phase").className = "value phase-complete";

    var skipS = data.holding_stable_skip_s != null ? Number(data.holding_stable_skip_s) : 30;

    var elE = document.getElementById("post-final-elongation");
    if (elE) {
        if (data.final_elongation_mm != null && data.final_elongation_mm !== undefined) {
            elE.textContent = Number(data.final_elongation_mm).toFixed(2) + " mm";
        } else if (data.had_holding_phase && (!data.holding_median_sample_count || data.holding_median_sample_count === 0)) {
            elE.textContent = "持荷不足 " + skipS + " s（无稳定样本）";
        } else {
            elE.textContent = "无持荷数据";
        }
    }
    var elD = document.getElementById("post-actual-tension-kn");
    if (elD) {
        if (data.holding_median_force_kn != null && data.holding_median_force_kn !== undefined) {
            elD.textContent = Number(data.holding_median_force_kn).toFixed(2) + " kN";
        } else if (data.had_holding_phase && (!data.holding_median_sample_count || data.holding_median_sample_count === 0)) {
            elD.textContent = "持荷不足 " + skipS + " s（无稳定样本）";
        } else {
            elD.textContent = "无持荷数据";
        }
    }

    var elDev = document.getElementById("post-target-force-deviation-pct");
    if (elDev) {
        if (data.holding_force_deviation_pct != null && data.holding_force_deviation_pct !== undefined) {
            var pct = Number(data.holding_force_deviation_pct);
            if (!isNaN(pct)) {
                var sign = pct > 0 ? "+" : "";
                elDev.textContent = sign + pct.toFixed(2) + " %";
            } else {
                elDev.textContent = "--";
            }
        } else if (data.had_holding_phase && (!data.holding_median_sample_count || data.holding_median_sample_count === 0)) {
            elDev.textContent = "持荷不足 " + skipS + " s（无稳定样本）";
        } else {
            elDev.textContent = "无持荷数据";
        }
    }

    var elS = document.getElementById("post-critical-anomaly-count");
    if (elS) {
        elS.className = "value";
        var cnt = data.critical_anomaly_count != null ? parseInt(data.critical_anomaly_count, 10) : 0;
        if (isNaN(cnt)) cnt = 0;
        elS.textContent = cnt + " 次";
        if (cnt > 0) {
            elS.classList.add("max-severity--critical");
        } else {
            elS.classList.add("max-severity--none");
        }
    }

    addSystemMessage(
        "张拉完成！共 " + data.total_points + " 个数据点，" + data.anomaly_count + " 条异常记录。"
    );
});

// ==================== 控制函数 ====================

function startTension() {
    const targetForce = parseFloat(document.getElementById("target-force").value);

    if (isNaN(targetForce) || targetForce <= 0) {
        alert("请输入有效的目标力值");
        return;
    }

    resetDisplay();

    fetch("/api/start_tension", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
            target_force: targetForce,
        }),
    })
        .then(function (res) { return res.json(); })
        .then(function (data) {
            if (data.status === "ok") {
                document.getElementById("btn-start").disabled = true;
                document.getElementById("btn-stop").disabled = false;
                document.getElementById("display-target").textContent = targetForce + " kN";
                console.log("[API] 张拉已启动，会话ID:", data.session_id);
            } else {
                alert("启动失败：" + data.message);
            }
        })
        .catch(function (err) {
            alert("请求失败：" + err.message);
        });
}

function stopTension() {
    fetch("/api/stop_tension", { method: "POST" })
        .then(function (res) { return res.json(); })
        .then(function (data) {
            console.log("[API] 停止信号已发送");
            document.getElementById("btn-start").disabled = false;
            document.getElementById("btn-stop").disabled = true;
        })
        .catch(function (err) {
            alert("请求失败：" + err.message);
        });
}

// ==================== 数据更新函数 ====================

function updateChartData(data) {
    var currentTime = parseFloat(data.time);
    var timeLabel = currentTime.toFixed(1);

    // 添加实际数据点到主数据数组
    chartData.labels.push(timeLabel);
    chartData.forceLeft.push(data.force_left);
    chartData.forceRight.push(data.force_right);
    chartData.forceAvg.push(data.force_avg);
    chartData.disLeft.push(data.dis_left);
    chartData.disRight.push(data.dis_right);
    chartData.totalDis.push(data.total_delta_dis);

    // 限制显示点数
    if (chartData.labels.length > MAX_POINTS) {
        chartData.labels.shift();
        chartData.forceLeft.shift();
        chartData.forceRight.shift();
        chartData.forceAvg.shift();
        chartData.disLeft.shift();
        chartData.disRight.shift();
        chartData.totalDis.shift();
    }

    // 处理 TCN 预测数据（不修改主数据数组）
    updateTCNPrediction(data, currentTime);

    // 刷新图表
    forceChart.update();
    disChart.update();
}

function updateTCNPrediction(data, currentTime) {
    var tcnDataset = forceChart.data.datasets[3];
    var numLabels = chartData.labels.length;

    if (data.tcn_prediction && data.tcn_prediction.length > 0) {
        // 保存预测状态
        tcnPredictionData.startIndex = numLabels - 1;
        tcnPredictionData.startTime = currentTime;
        tcnPredictionData.values = data.tcn_prediction.slice();
        tcnPredictionData.stepInterval = 1.0; // 与 PUSH_INTERVAL 一致

        // 构建 TCN 预测线数据：
        // 在当前点之前全部为 null，当前点为实际力值（作为连接点），
        // 之后的预测点也放在 labels 对应位置
        // 但我们不往 labels 里加新标签，只在已有范围内显示
        // 预测点超出当前 labels 范围的部分，暂时添加到末尾

        var predArray = new Array(numLabels).fill(null);

        // 当前点作为预测线起点
        predArray[numLabels - 1] = data.force_avg;

        // 添加未来预测点（临时扩展 labels）
        var predictSteps = data.tcn_prediction.length;
        for (var i = 0; i < predictSteps; i++) {
            var futureTime = (currentTime + (i + 1) * tcnPredictionData.stepInterval).toFixed(1);
            predArray.push(data.tcn_prediction[i]);

            // 同步扩展其他 dataset 的数据长度（用 null 填充）
            // 注意：这些扩展点是临时的，下次更新时会被重建
        }

        // 为了让 Chart.js 正确渲染，需要同步 labels 和所有 dataset 的长度
        // 策略：用临时的 labels 副本，不修改 chartData 的主数组
        var extendedLabels = chartData.labels.slice();
        for (var i = 0; i < predictSteps; i++) {
            var futureTime = (currentTime + (i + 1) * tcnPredictionData.stepInterval).toFixed(1);
            extendedLabels.push(futureTime);
        }

        // 直接设置 chart 的 labels（覆盖引用）
        forceChart.data.labels = extendedLabels;

        // 扩展其他 dataset 的数据（用 null 填充未来部分）
        forceChart.data.datasets[0].data = chartData.forceLeft.concat(
            new Array(predictSteps).fill(null)
        );
        forceChart.data.datasets[1].data = chartData.forceRight.concat(
            new Array(predictSteps).fill(null)
        );
        forceChart.data.datasets[2].data = chartData.forceAvg.concat(
            new Array(predictSteps).fill(null)
        );
        forceChart.data.datasets[3].data = predArray;

    } else {
        // 没有 TCN 预测时，恢复正常显示
        forceChart.data.labels = chartData.labels;
        forceChart.data.datasets[0].data = chartData.forceLeft;
        forceChart.data.datasets[1].data = chartData.forceRight;
        forceChart.data.datasets[2].data = chartData.forceAvg;
        forceChart.data.datasets[3].data = new Array(numLabels).fill(null);
    }
}

function updateStatusBar(data) {
    var phaseMap = {
        loading: "张拉加载",
        holding: "持荷",
        unloading: "卸载",
    };
    var phaseEl = document.getElementById("current-phase");
    phaseEl.textContent = phaseMap[data.phase] || data.phase;
    phaseEl.className = "value phase-" + data.phase;

    document.getElementById("current-force").textContent =
        data.force_avg.toFixed(1) + " kN";

    var resEl = document.getElementById("tcn-residual");
    if (resEl) {
        var setFromAnomaly = false;
        if (data.anomalies && data.anomalies.length > 0) {
            for (var i = 0; i < data.anomalies.length; i++) {
                if (data.anomalies[i].source === "tcn") {
                    var detail = data.anomalies[i].detail || "";
                    var match = detail.match(/残差=([\d.]+)/);
                    if (match) {
                        resEl.textContent = match[1];
                        setFromAnomaly = true;
                    }
                }
            }
        }
        if (
            !setFromAnomaly &&
            data.tcn_residual != null &&
            data.tcn_residual !== undefined &&
            !isNaN(Number(data.tcn_residual))
        ) {
            resEl.textContent = Number(data.tcn_residual).toFixed(4);
        }
    }
}

function buildAnomalyItemDom(anomaly) {
    var item = document.createElement("div");
    item.className = "anomaly-item severity-" + (anomaly.severity || "warning");
    if ((anomaly.source || "") === "tcn") {
        item.classList.add("anomaly-item--tcn");
    }
    item.innerHTML =
        '<span class="anomaly-time">[' + parseFloat(anomaly.time_offset).toFixed(1) + "s]</span>" +
        '<span class="anomaly-source">[' + anomaly.source + "]</span>" +
        '<span class="anomaly-type">' + anomaly.anomaly_type + "</span>: " +
        '<span class="anomaly-detail">' + (anomaly.detail || "") + "</span>";
    return item;
}

function addAnomalies(anomalies) {
    var listEl = document.getElementById("anomaly-list");
    var tcnListEl = document.getElementById("tcn-anomaly-list");

    anomalies.forEach(function (anomaly) {
        var isTcn = (anomaly.source || "") === "tcn";
        var targetList = isTcn ? tcnListEl : listEl;
        var noAnomaly = targetList.querySelector(".no-anomaly");
        if (noAnomaly) {
            noAnomaly.remove();
        }
        var item = buildAnomalyItemDom(anomaly);
        targetList.insertBefore(item, targetList.firstChild);
        if (isTcn) {
            tcnAnomalyCount++;
        } else {
            anomalyCount++;
        }
    });

    document.getElementById("anomaly-count").textContent = anomalyCount;
    var tcnBadge = document.getElementById("tcn-anomaly-count");
    if (tcnBadge) {
        tcnBadge.textContent = tcnAnomalyCount;
    }
}

function addSystemMessage(message) {
    var listEl = document.getElementById("anomaly-list");
    var noAnomaly = listEl.querySelector(".no-anomaly");
    if (noAnomaly) {
        noAnomaly.remove();
    }
    var item = document.createElement("div");
    item.className = "anomaly-item system-message";
    item.innerHTML = "<strong>[系统]</strong> " + message;
    listEl.insertBefore(item, listEl.firstChild);
}

function resetDisplay() {
    chartData.labels.length = 0;
    chartData.forceLeft.length = 0;
    chartData.forceRight.length = 0;
    chartData.forceAvg.length = 0;
    chartData.disLeft.length = 0;
    chartData.disRight.length = 0;
    chartData.totalDis.length = 0;

    // 重置 TCN 预测状态
    tcnPredictionData = {
        startIndex: -1,
        startTime: 0,
        values: [],
        stepInterval: 1.0,
    };

    // 恢复 chart labels 引用
    forceChart.data.labels = chartData.labels;
    forceChart.data.datasets[0].data = chartData.forceLeft;
    forceChart.data.datasets[1].data = chartData.forceRight;
    forceChart.data.datasets[2].data = chartData.forceAvg;
    forceChart.data.datasets[3].data = [];

    forceChart.update();
    disChart.update();

    anomalyCount = 0;
    tcnAnomalyCount = 0;
    document.getElementById("anomaly-count").textContent = "0";
    document.getElementById("anomaly-list").innerHTML =
        '<p class="no-anomaly">暂无预警及警告</p>';
    var tcnBadge = document.getElementById("tcn-anomaly-count");
    if (tcnBadge) {
        tcnBadge.textContent = "0";
    }
    var tcnList = document.getElementById("tcn-anomaly-list");
    if (tcnList) {
        tcnList.innerHTML = '<p class="no-anomaly">暂无 TCN 预测预警</p>';
    }

    var phaseReset = document.getElementById("current-phase");
    phaseReset.textContent = "--";
    phaseReset.className = "value";
    document.getElementById("current-force").textContent = "-- kN";
    document.getElementById("tcn-residual").textContent = "--";

    var elE = document.getElementById("post-final-elongation");
    if (elE) elE.textContent = "--";
    var elD = document.getElementById("post-actual-tension-kn");
    if (elD) elD.textContent = "--";
    var elDev = document.getElementById("post-target-force-deviation-pct");
    if (elDev) elDev.textContent = "--";
    var elS = document.getElementById("post-critical-anomaly-count");
    if (elS) {
        elS.textContent = "--";
        elS.className = "value";
    }
}