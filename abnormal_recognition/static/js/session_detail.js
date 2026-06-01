/**
 * 会话详情页面逻辑
 * 独立页面展示某次张拉的完整数据
 */

function cssVar(name, fallback) {
    var v = getComputedStyle(document.documentElement).getPropertyValue(name);
    v = (v || "").trim();
    return v || fallback;
}

function chartFont() {
    return { size: 11, family: "Noto Sans SC, system-ui, sans-serif" };
}

var SD_C_FORCE_L = cssVar("--chart-force-l", "#e11d48");
var SD_C_FORCE_R = cssVar("--chart-force-r", "#2563eb");
var SD_C_FORCE_AVG = cssVar("--chart-force-avg", "#059669");
var SD_C_DIS_TOTAL = cssVar("--chart-dis-total", "#7c3aed");
var SD_C_GRID = "rgba(15, 23, 42, 0.06)";
var SD_C_TEXT = cssVar("--ink-muted", "#64748b");

let historyForceChart = null;
let historyDisChart = null;

/** 会话详情预警表：全量数据（供筛选） */
let detailAnomaliesAll = [];

document.addEventListener("DOMContentLoaded", function () {
    var phaseSel = document.getElementById("anomaly-filter-phase");
    var srcSel = document.getElementById("anomaly-filter-source");
    var sevSel = document.getElementById("anomaly-filter-severity");
    var typeSel = document.getElementById("anomaly-filter-type");
    if (phaseSel) {
        phaseSel.addEventListener("change", applyAnomalyFilters);
    }
    if (srcSel) {
        srcSel.addEventListener("change", applyAnomalyFilters);
    }
    if (sevSel) {
        sevSel.addEventListener("change", applyAnomalyFilters);
    }
    if (typeSel) {
        typeSel.addEventListener("change", applyAnomalyFilters);
    }
    loadSessionDetail(SESSION_ID);
});

function loadSessionDetail(sessionId) {
    fetch("/api/history/session/" + sessionId)
        .then(function (res) { return res.json(); })
        .then(function (data) {
            if (data.status === "ok") {
                renderSessionInfo(data.data.session);
                renderHistoryCharts(data.data.data_points);
                renderAnomalyTable(data.data.anomalies);
            } else {
                document.getElementById("detail-info").innerHTML =
                    '<p class="error-msg">获取数据失败：' + data.message + "</p>";
            }
        })
        .catch(function (err) {
            console.error("获取会话详情失败:", err);
            document.getElementById("detail-info").innerHTML =
                '<p class="error-msg">网络错误，请稍后重试</p>';
        });
}

function fmtElongationMm(v) {
    if (v == null || v === "") {
        return "--";
    }
    var n = Number(v);
    return isNaN(n) ? "--" : n.toFixed(2) + " mm";
}

function fmtForceKn(v) {
    if (v == null || v === "") {
        return "--";
    }
    var n = Number(v);
    return isNaN(n) ? "--" : n.toFixed(2) + " kN";
}

function fmtDeviationPct(v) {
    if (v == null || v === "") {
        return "--";
    }
    var n = Number(v);
    if (isNaN(n)) {
        return "--";
    }
    return (n > 0 ? "+" : "") + n.toFixed(2) + " %";
}

function renderSessionInfo(session) {
    var infoEl = document.getElementById("detail-info");
    var statusClass = session.status === "running" ? "status-running" : "status-ok";
    var crit = session.critical_anomaly_count != null ? parseInt(session.critical_anomaly_count, 10) : 0;
    if (isNaN(crit)) {
        crit = 0;
    }

    infoEl.innerHTML =
        '<div class="info-grid info-grid--session-detail">' +
        '<div class="info-item"><span class="info-label">开始时间</span><span class="info-value">' + (session.start_time || "--") + "</span></div>" +
        '<div class="info-item"><span class="info-label">结束时间</span><span class="info-value">' + (session.end_time || "--") + "</span></div>" +
        '<div class="info-item"><span class="info-label">目标力值</span><span class="info-value">' + session.target_force + " kN</span></div>" +
        '<div class="info-item"><span class="info-label">严重异常数量</span><span class="info-value anomaly-highlight">' + crit + "</span></div>" +
        '<div class="info-item"><span class="info-label">数据点数</span><span class="info-value">' + (session.total_points || 0) + "</span></div>" +
        '<div class="info-item"><span class="info-label">状态</span><span class="info-value ' + statusClass + '">' + session.status + "</span></div>" +
        '<div class="info-item"><span class="info-label">实际伸长量</span><span class="info-value">' + fmtElongationMm(session.final_elongation_mm) + "</span></div>" +
        '<div class="info-item"><span class="info-label">实际张拉力</span><span class="info-value">' + fmtForceKn(session.holding_median_force_kn) + "</span></div>" +
        '<div class="info-item"><span class="info-label">目标力值偏差率</span><span class="info-value">' + fmtDeviationPct(session.holding_force_deviation_pct) + "</span></div>" +
        "</div>";
}

function renderHistoryCharts(dataPoints) {
    if (!dataPoints || dataPoints.length === 0) {
        return;
    }

    var labels = dataPoints.map(function (p) { return parseFloat(p.time_offset).toFixed(1); });
    var forceLeft = dataPoints.map(function (p) { return p.force_left; });
    var forceRight = dataPoints.map(function (p) { return p.force_right; });
    var forceAvg = dataPoints.map(function (p) { return p.force_avg; });
    var disLeft = dataPoints.map(function (p) { return p.dis_left; });
    var disRight = dataPoints.map(function (p) { return p.dis_right; });
    var totalDis = dataPoints.map(function (p) { return p.total_delta_dis; });

    // 销毁旧图表
    if (historyForceChart) historyForceChart.destroy();
    if (historyDisChart) historyDisChart.destroy();

    // 力值图表
    historyForceChart = new Chart(document.getElementById("history-force-chart"), {
        type: "line",
        data: {
            labels: labels,
            datasets: [
                {
                    label: "左侧力值",
                    data: forceLeft,
                    borderColor: SD_C_FORCE_L,
                    borderWidth: 1.5,
                    pointRadius: 0,
                    fill: false,
                    tension: 0.12,
                },
                {
                    label: "右侧力值",
                    data: forceRight,
                    borderColor: SD_C_FORCE_R,
                    borderWidth: 1.5,
                    pointRadius: 0,
                    fill: false,
                    tension: 0.12,
                },
                {
                    label: "平均力值",
                    data: forceAvg,
                    borderColor: SD_C_FORCE_AVG,
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
            interaction: { intersect: false, mode: "index" },
            scales: {
                x: {
                    title: { display: true, text: "时间 (s)", font: chartFont(), color: SD_C_TEXT },
                    ticks: { color: SD_C_TEXT, font: chartFont() },
                    grid: { color: SD_C_GRID },
                },
                y: {
                    title: { display: true, text: "力值 (kN)", font: chartFont(), color: SD_C_TEXT },
                    ticks: { color: SD_C_TEXT, font: chartFont() },
                    grid: { color: SD_C_GRID },
                },
            },
            plugins: {
                legend: {
                    position: "top",
                    labels: { usePointStyle: true, font: chartFont(), color: SD_C_TEXT, padding: 14 },
                },
            },
        },
    });

    // 位移图表
    historyDisChart = new Chart(document.getElementById("history-dis-chart"), {
        type: "line",
        data: {
            labels: labels,
            datasets: [
                {
                    label: "左侧位移",
                    data: disLeft,
                    borderColor: SD_C_FORCE_L,
                    borderWidth: 1.5,
                    pointRadius: 0,
                    fill: false,
                    tension: 0.12,
                },
                {
                    label: "右侧位移",
                    data: disRight,
                    borderColor: SD_C_FORCE_R,
                    borderWidth: 1.5,
                    pointRadius: 0,
                    fill: false,
                    tension: 0.12,
                },
                {
                    label: "总位移",
                    data: totalDis,
                    borderColor: SD_C_DIS_TOTAL,
                    backgroundColor: "rgba(124, 58, 246, 0.08)",
                    borderWidth: 2,
                    pointRadius: 0,
                    fill: true,
                    tension: 0.12,
                },
            ],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: { intersect: false, mode: "index" },
            scales: {
                x: {
                    title: { display: true, text: "时间 (s)", font: chartFont(), color: SD_C_TEXT },
                    ticks: { color: SD_C_TEXT, font: chartFont() },
                    grid: { color: SD_C_GRID },
                },
                y: {
                    title: { display: true, text: "位移 (mm)", font: chartFont(), color: SD_C_TEXT },
                    ticks: { color: SD_C_TEXT, font: chartFont() },
                    grid: { color: SD_C_GRID },
                },
            },
            plugins: {
                legend: {
                    position: "top",
                    labels: { usePointStyle: true, font: chartFont(), color: SD_C_TEXT, padding: 14 },
                },
            },
        },
    });

    requestAnimationFrame(function () {
        if (historyForceChart) {
            historyForceChart.resize();
        }
        if (historyDisChart) {
            historyDisChart.resize();
        }
    });
}

function phaseLabel(phase) {
    var map = { loading: "张拉加载", holding: "持荷", unloading: "卸载" };
    return map[phase] || phase || "--";
}

function sourceLabel(source) {
    var map = { rule_engine: "规则引擎", if_lof: "IF/LOF", tcn: "TCN" };
    return map[source] || source || "--";
}

/** 根据当前会话异常列表填充「类型」下拉的选项（去重、排序） */
function rebuildAnomalyTypeSelectOptions() {
    var sel = document.getElementById("anomaly-filter-type");
    if (!sel) {
        return;
    }
    sel.innerHTML = "";
    var optAll = document.createElement("option");
    optAll.value = "";
    optAll.textContent = "全部类型";
    sel.appendChild(optAll);
    var seen = {};
    var types = [];
    detailAnomaliesAll.forEach(function (a) {
        var t = (a.anomaly_type != null ? String(a.anomaly_type) : "").trim();
        if (t && !seen[t]) {
            seen[t] = true;
            types.push(t);
        }
    });
    types.sort();
    types.forEach(function (t) {
        var o = document.createElement("option");
        o.value = t;
        o.textContent = t;
        sel.appendChild(o);
    });
}

function filterDetailAnomalies() {
    var phaseSel = document.getElementById("anomaly-filter-phase");
    var srcSel = document.getElementById("anomaly-filter-source");
    var sevSel = document.getElementById("anomaly-filter-severity");
    var typeSel = document.getElementById("anomaly-filter-type");
    var phaseVal = phaseSel ? phaseSel.value : "";
    var srcVal = srcSel ? srcSel.value : "";
    var sevVal = sevSel ? sevSel.value : "";
    var typeVal = typeSel ? typeSel.value : "";
    return detailAnomaliesAll.filter(function (a) {
        if (phaseVal && (a.phase || "") !== phaseVal) {
            return false;
        }
        if (srcVal && (a.source || "") !== srcVal) {
            return false;
        }
        if (typeVal) {
            var ty = (a.anomaly_type != null ? String(a.anomaly_type) : "").trim();
            if (ty !== typeVal) {
                return false;
            }
        }
        if (sevVal) {
            var s = (a.severity || "").toString().toLowerCase();
            if (s !== sevVal.toLowerCase()) {
                return false;
            }
        }
        return true;
    });
}

function updateAnomalyFilterButtonState() {
    var phaseSel = document.getElementById("anomaly-filter-phase");
    var srcSel = document.getElementById("anomaly-filter-source");
    var sevSel = document.getElementById("anomaly-filter-severity");
    var typeSel = document.getElementById("anomaly-filter-type");
    var phaseBtn = phaseSel && phaseSel.closest(".anomaly-th-filter-btn");
    var srcBtn = srcSel && srcSel.closest(".anomaly-th-filter-btn");
    var sevBtn = sevSel && sevSel.closest(".anomaly-th-filter-btn");
    var typeBtn = typeSel && typeSel.closest(".anomaly-th-filter-btn");
    if (phaseBtn) {
        phaseBtn.classList.toggle("is-filtered", !!(phaseSel && phaseSel.value));
    }
    if (srcBtn) {
        srcBtn.classList.toggle("is-filtered", !!(srcSel && srcSel.value));
    }
    if (sevBtn) {
        sevBtn.classList.toggle("is-filtered", !!(sevSel && sevSel.value));
    }
    if (typeBtn) {
        typeBtn.classList.toggle("is-filtered", !!(typeSel && typeSel.value));
    }
}

function applyAnomalyFilters() {
    var tbody = document.getElementById("anomaly-tbody");
    var badge = document.getElementById("anomaly-count-badge");
    if (!tbody) {
        return;
    }

    var filtered = filterDetailAnomalies();

    if (badge) {
        badge.textContent = filtered.length ? String(filtered.length) : "0";
    }

    var phaseSel = document.getElementById("anomaly-filter-phase");
    var srcSel = document.getElementById("anomaly-filter-source");
    var sevSel = document.getElementById("anomaly-filter-severity");
    var typeSel = document.getElementById("anomaly-filter-type");
    if (phaseSel) {
        phaseSel.disabled = !detailAnomaliesAll.length;
    }
    if (srcSel) {
        srcSel.disabled = !detailAnomaliesAll.length;
    }
    if (sevSel) {
        sevSel.disabled = !detailAnomaliesAll.length;
    }
    if (typeSel) {
        typeSel.disabled = !detailAnomaliesAll.length;
    }

    tbody.innerHTML = "";

    if (!detailAnomaliesAll.length) {
        updateAnomalyFilterButtonState();
        tbody.innerHTML =
            '<tr><td colspan="6" class="table-loading-cell">无预警及警告</td></tr>';
        return;
    }

    updateAnomalyFilterButtonState();

    if (!filtered.length) {
        tbody.innerHTML =
            '<tr><td colspan="6" class="table-loading-cell">没有符合当前筛选条件的记录（共 ' +
            detailAnomaliesAll.length +
            " 条）</td></tr>";
        return;
    }

    filtered.forEach(function (a) {
        var tr = document.createElement("tr");
        var severityClass = "severity-row-" + (a.severity || "warning");
        tr.className = severityClass;
        tr.innerHTML =
            "<td>" + parseFloat(a.time_offset).toFixed(1) + "s</td>" +
            "<td><span class='phase-tag phase-" + a.phase + "'>" + phaseLabel(a.phase) + "</span></td>" +
            "<td><span class='source-tag source-" + a.source + "'>" + sourceLabel(a.source) + "</span></td>" +
            "<td>" + a.anomaly_type + "</td>" +
            "<td><span class='severity-tag severity-" + a.severity + "'>" + a.severity + "</span></td>" +
            "<td>" + (a.detail || "") + "</td>";
        tbody.appendChild(tr);
    });
}

function renderAnomalyTable(anomalies) {
    detailAnomaliesAll = anomalies && anomalies.length ? anomalies.slice() : [];

    rebuildAnomalyTypeSelectOptions();

    var phaseSel = document.getElementById("anomaly-filter-phase");
    var srcSel = document.getElementById("anomaly-filter-source");
    var sevSel = document.getElementById("anomaly-filter-severity");
    var typeSel = document.getElementById("anomaly-filter-type");
    if (phaseSel) {
        phaseSel.value = "";
    }
    if (srcSel) {
        srcSel.value = "";
    }
    if (sevSel) {
        sevSel.value = "";
    }
    if (typeSel) {
        typeSel.value = "";
    }

    applyAnomalyFilters();
}