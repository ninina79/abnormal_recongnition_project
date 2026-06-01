/**
 * 历史数据查看页面逻辑
 * 点击查看按钮跳转到独立详情页
 */

// 页面加载时获取会话列表
document.addEventListener("DOMContentLoaded", function () {
    loadSessions();
});

// ==================== 会话列表 ====================

function loadSessions() {
    fetch("/api/history/sessions")
        .then(function (res) { return res.json(); })
        .then(function (data) {
            if (data.status === "ok") {
                renderSessionTable(data.sessions);
            }
        })
        .catch(function (err) {
            console.error("获取会话列表失败:", err);
        });
}

function renderSessionTable(sessions) {
    var tbody = document.getElementById("session-tbody");
    tbody.innerHTML = "";

    if (sessions.length === 0) {
        tbody.innerHTML =
            '<tr><td colspan="8" class="table-loading-cell">暂无张拉记录</td></tr>';
        return;
    }

    sessions.forEach(function (session) {
        var tr = document.createElement("tr");
        var statusClass = session.status === "running" ? "status-running" : "status-ok";
        var anomalyClass = session.anomaly_count > 0 ? "has-anomaly" : "";

        tr.innerHTML =
            "<td><strong>#" + session.id + "</strong></td>" +
            "<td>" + (session.start_time || "--") + "</td>" +
            "<td>" + (session.end_time || "--") + "</td>" +
            "<td>" + session.target_force + " kN</td>" +
            "<td>" + (session.total_points || 0) + "</td>" +
            '<td class="' + anomalyClass + '">' + (session.anomaly_count || 0) + "</td>" +
            '<td class="' + statusClass + '">' + session.status + "</td>" +
            '<td>' +
            '<a href="/history/' + session.id + '" class="btn-view">查看详情</a> ' +
            '<button type="button" class="btn-delete" data-session-id="' + session.id + '">删除</button>' +
            "</td>";
        tbody.appendChild(tr);

        tr.querySelector(".btn-delete").addEventListener("click", function () {
            deleteSessionRow(session.id, tr);
        });
    });
}

function deleteSessionRow(sessionId, tr) {
    if (!window.confirm("确定删除会话 #" + sessionId + "？关联数据点与异常记录将一并删除。")) {
        return;
    }
    fetch("/api/history/session/" + sessionId, { method: "DELETE" })
        .then(function (res) { return res.json(); })
        .then(function (data) {
            if (data.status === "ok") {
                tr.remove();
                var tbody = document.getElementById("session-tbody");
                if (tbody && tbody.children.length === 0) {
                    tbody.innerHTML =
                        '<tr><td colspan="8" class="table-loading-cell">暂无张拉记录</td></tr>';
                }
            } else {
                alert(data.message || "删除失败");
            }
        })
        .catch(function (err) {
            console.error("删除会话失败:", err);
            alert("删除请求失败");
        });
}