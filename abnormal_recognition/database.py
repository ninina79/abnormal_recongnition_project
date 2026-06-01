"""
MySQL 数据库操作模块
使用 pymysql 连接 MySQL (Navicat 管理)
"""
import pymysql
from pymysql.cursors import DictCursor
from datetime import datetime
from contextlib import contextmanager
from config import DB_CONFIG


def init_database():
    """初始化数据库，创建所需表"""
    # 先连接 MySQL 服务器（不指定数据库），创建数据库
    conn = pymysql.connect(
        host=DB_CONFIG["host"],
        port=DB_CONFIG["port"],
        user=DB_CONFIG["user"],
        password=DB_CONFIG["password"],
        charset=DB_CONFIG["charset"],
    )
    try:
        with conn.cursor() as cursor:
            cursor.execute(
                f"CREATE DATABASE IF NOT EXISTS `{DB_CONFIG['database']}` "
                f"DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
            )
        conn.commit()
    finally:
        conn.close()

    # 连接到目标数据库，创建表
    conn = get_connection()
    try:
        with conn.cursor() as cursor:
            # 张拉会话表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS tension_sessions (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    start_time DATETIME NOT NULL,
                    end_time DATETIME DEFAULT NULL,
                    target_force DOUBLE NOT NULL,
                    status VARCHAR(20) DEFAULT 'running',
                    total_points INT DEFAULT 0,
                    anomaly_count INT DEFAULT 0,
                    critical_anomaly_count INT NOT NULL DEFAULT 0,
                    final_elongation_mm DOUBLE NULL,
                    holding_median_force_kn DOUBLE NULL,
                    holding_force_deviation_pct DOUBLE NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            """)
            _migrate_tension_sessions_columns(cursor)

            # 张拉数据点表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS tension_data_points (
                    id BIGINT AUTO_INCREMENT PRIMARY KEY,
                    session_id INT NOT NULL,
                    time_offset DOUBLE NOT NULL,
                    force_left DOUBLE NOT NULL,
                    force_right DOUBLE NOT NULL,
                    force_avg DOUBLE NOT NULL,
                    dis_left DOUBLE NOT NULL,
                    dis_right DOUBLE NOT NULL,
                    total_delta_dis DOUBLE NOT NULL,
                    phase VARCHAR(20) DEFAULT 'loading',
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    INDEX idx_session_id (session_id),
                    FOREIGN KEY (session_id) REFERENCES tension_sessions(id)
                        ON DELETE CASCADE
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            """)

            # 异常记录表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS tension_anomalies (
                    id BIGINT AUTO_INCREMENT PRIMARY KEY,
                    session_id INT NOT NULL,
                    time_offset DOUBLE NOT NULL,
                    phase VARCHAR(20) NOT NULL,
                    source VARCHAR(50) NOT NULL,
                    anomaly_type VARCHAR(100) NOT NULL,
                    severity VARCHAR(20) DEFAULT 'warning',
                    detail TEXT,
                    force_avg DOUBLE,
                    force_diff DOUBLE,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    INDEX idx_session_id (session_id),
                    INDEX idx_source (source),
                    FOREIGN KEY (session_id) REFERENCES tension_sessions(id)
                        ON DELETE CASCADE
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            """)

        conn.commit()
    finally:
        conn.close()


def get_connection():
    """获取数据库连接"""
    return pymysql.connect(
        host=DB_CONFIG["host"],
        port=DB_CONFIG["port"],
        user=DB_CONFIG["user"],
        password=DB_CONFIG["password"],
        database=DB_CONFIG["database"],
        charset=DB_CONFIG["charset"],
        cursorclass=DictCursor,
    )


@contextmanager
def get_db():
    """数据库连接上下文管理器"""
    conn = get_connection()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _migrate_tension_sessions_columns(cursor):
    """已有库表补齐新列（无需清空数据）。"""
    additions = [
        ("critical_anomaly_count", "INT NOT NULL DEFAULT 0"),
        ("final_elongation_mm", "DOUBLE NULL"),
        ("holding_median_force_kn", "DOUBLE NULL"),
        ("holding_force_deviation_pct", "DOUBLE NULL"),
    ]
    db_name = DB_CONFIG["database"]
    for col, ddl in additions:
        cursor.execute(
            "SELECT COUNT(*) AS c FROM information_schema.COLUMNS "
            "WHERE TABLE_SCHEMA = %s AND TABLE_NAME = 'tension_sessions' "
            "AND COLUMN_NAME = %s",
            (db_name, col),
        )
        row = cursor.fetchone()
        cnt = row["c"] if isinstance(row, dict) else row[0]
        if cnt == 0:
            cursor.execute(
                f"ALTER TABLE tension_sessions ADD COLUMN {col} {ddl}"
            )


def create_session(target_force):
    """创建新的张拉会话"""
    with get_db() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                "INSERT INTO tension_sessions (start_time, target_force, status) "
                "VALUES (%s, %s, %s)",
                (datetime.now(), target_force, "running"),
            )
            session_id = cursor.lastrowid
    return session_id


def finish_session(
    session_id,
    total_points,
    anomaly_count,
    *,
    critical_anomaly_count=0,
    final_elongation_mm=None,
    holding_median_force_kn=None,
    holding_force_deviation_pct=None,
):
    """结束张拉会话，写入汇总与完成时摘要（伸长、持荷中位力、偏差率）。"""
    with get_db() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                "UPDATE tension_sessions SET end_time=%s, status=%s, "
                "total_points=%s, anomaly_count=%s, critical_anomaly_count=%s, "
                "final_elongation_mm=%s, holding_median_force_kn=%s, "
                "holding_force_deviation_pct=%s WHERE id=%s",
                (
                    datetime.now(),
                    "completed",
                    total_points,
                    anomaly_count,
                    int(critical_anomaly_count),
                    final_elongation_mm,
                    holding_median_force_kn,
                    holding_force_deviation_pct,
                    session_id,
                ),
            )


def delete_session(session_id):
    """删除一条张拉会话（关联数据点与异常记录由外键级联删除）"""
    with get_db() as conn:
        with conn.cursor() as cursor:
            cursor.execute("DELETE FROM tension_sessions WHERE id=%s", (session_id,))
            return cursor.rowcount


def insert_data_point(session_id, point_data):
    """插入一条数据点"""
    with get_db() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                "INSERT INTO tension_data_points "
                "(session_id, time_offset, force_left, force_right, force_avg, "
                "dis_left, dis_right, total_delta_dis, phase) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)",
                (
                    session_id,
                    point_data["time_offset"],
                    point_data["force_left"],
                    point_data["force_right"],
                    point_data["force_avg"],
                    point_data["dis_left"],
                    point_data["dis_right"],
                    point_data["total_delta_dis"],
                    point_data["phase"],
                ),
            )


def insert_anomaly(session_id, anomaly_data):
    """插入一条异常记录"""
    with get_db() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                "INSERT INTO tension_anomalies "
                "(session_id, time_offset, phase, source, anomaly_type, "
                "severity, detail, force_avg, force_diff) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)",
                (
                    session_id,
                    anomaly_data["time_offset"],
                    anomaly_data["phase"],
                    anomaly_data["source"],
                    anomaly_data["anomaly_type"],
                    anomaly_data.get("severity", "warning"),
                    anomaly_data.get("detail", ""),
                    anomaly_data.get("force_avg"),
                    anomaly_data.get("force_diff"),
                ),
            )


def get_all_sessions():
    """获取所有张拉会话列表"""
    with get_db() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                "SELECT id, start_time, end_time, target_force, status, "
                "total_points, anomaly_count, critical_anomaly_count, "
                "final_elongation_mm, holding_median_force_kn, "
                "holding_force_deviation_pct "
                "FROM tension_sessions ORDER BY start_time DESC"
            )
            rows = cursor.fetchall()
    # 序列化 datetime
    for row in rows:
        for key in ("start_time", "end_time", "created_at"):
            if key in row and isinstance(row[key], datetime):
                row[key] = row[key].strftime("%Y-%m-%d %H:%M:%S")
    return rows


def get_session_detail(session_id):
    """获取某次张拉的完整数据和异常"""
    with get_db() as conn:
        with conn.cursor() as cursor:
            # 会话信息
            cursor.execute(
                "SELECT * FROM tension_sessions WHERE id=%s", (session_id,)
            )
            session = cursor.fetchone()
            if not session:
                return None

            # 数据点
            cursor.execute(
                "SELECT time_offset, force_left, force_right, force_avg, "
                "dis_left, dis_right, total_delta_dis, phase "
                "FROM tension_data_points WHERE session_id=%s "
                "ORDER BY time_offset",
                (session_id,),
            )
            data_points = cursor.fetchall()

            # 异常记录
            cursor.execute(
                "SELECT time_offset, phase, source, anomaly_type, severity, detail "
                "FROM tension_anomalies WHERE session_id=%s "
                "ORDER BY time_offset",
                (session_id,),
            )
            anomalies = cursor.fetchall()

    # 序列化
    for key in ("start_time", "end_time", "created_at"):
        if key in session and isinstance(session[key], datetime):
            session[key] = session[key].strftime("%Y-%m-%d %H:%M:%S")

    return {
        "session": session,
        "data_points": data_points,
        "anomalies": anomalies,
    }