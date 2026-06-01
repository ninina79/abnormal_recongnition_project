from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#滤波器测试，看哪一个效果最好

# =========================
# 实时(因果)滤波器实现
# =========================
def ewma_filter(x: np.ndarray, alpha: float = 0.2) -> np.ndarray:
    """
    EWMA (Exponentially Weighted Moving Average), 因果/实时可用。
    y[t] = alpha * x[t] + (1-alpha) * y[t-1]
    """
    if not (0 < alpha <= 1):
        raise ValueError("alpha must be in (0, 1].")
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return x.copy()
    y = np.empty_like(x, dtype=float)
    y[0] = x[0]
    for i in range(1, len(x)):
        y[i] = alpha * x[i] + (1.0 - alpha) * y[i - 1]
    return y


def sma_filter(x: np.ndarray, window: int = 7) -> np.ndarray:
    """
    简单滑动均值 (SMA)，用过去 window 个点，因果/实时可用。
    """
    if window <= 0:
        raise ValueError("window must be > 0.")
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return x.copy()
    s = pd.Series(x)
    # min_periods=1：避免前段变 NaN
    return s.rolling(window=window, min_periods=1).mean().to_numpy(dtype=float)


def median_filter_causal(x: np.ndarray, window: int = 7) -> np.ndarray:
    """
    因果中值滤波：对尖峰噪声更强，但会带来更大的形状变化。
    """
    if window <= 0:
        raise ValueError("window must be > 0.")
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return x.copy()
    s = pd.Series(x)
    return s.rolling(window=window, min_periods=1).median().to_numpy(dtype=float)


@dataclass
class OneEuroState:
    x_prev: float | None = None
    dx_prev: float = 0.0
    t_prev: float | None = None


def _lowpass_alpha(cutoff_hz: float, dt_s: float) -> float:
    # 1st-order low-pass: alpha = dt / (dt + tau), tau = 1/(2*pi*fc)
    if cutoff_hz <= 0:
        return 1.0
    tau = 1.0 / (2.0 * math.pi * cutoff_hz)
    return dt_s / (dt_s + tau)


def one_euro_filter(
    x: np.ndarray,
    t_s: np.ndarray,
    *,
    min_cutoff_hz: float = 0.8,
    beta: float = 0.02,
    d_cutoff_hz: float = 1.0,
) -> np.ndarray:
    """
    One Euro Filter（实时常用）：在“平滑”和“响应速度”之间自适应权衡。
    参考：G. Casiez, N. Roussel, D. Vogel, 2012.
    """
    x = np.asarray(x, dtype=float)
    t_s = np.asarray(t_s, dtype=float)
    if x.size == 0:
        return x.copy()
    if x.size != t_s.size:
        raise ValueError("x and t_s must have same length.")

    st = OneEuroState()
    y = np.empty_like(x, dtype=float)

    for i in range(len(x)):
        if st.t_prev is None:
            st.t_prev = float(t_s[i])
            st.x_prev = float(x[i])
            st.dx_prev = 0.0
            y[i] = float(x[i])
            continue

        dt = float(t_s[i] - st.t_prev)
        if dt <= 0:
            dt = 1.0

        # 1) derivative
        dx = (float(x[i]) - float(st.x_prev)) / dt
        a_d = _lowpass_alpha(d_cutoff_hz, dt)
        dx_hat = a_d * dx + (1.0 - a_d) * float(st.dx_prev)

        # 2) adaptive cutoff
        cutoff = min_cutoff_hz + beta * abs(dx_hat)
        a = _lowpass_alpha(cutoff, dt)
        x_hat = a * float(x[i]) + (1.0 - a) * float(st.x_prev)

        y[i] = x_hat
        st.x_prev = x_hat
        st.dx_prev = dx_hat
        st.t_prev = float(t_s[i])

    return y


# =========================
# 评价指标（偏“实时规则引擎友好”）
# =========================
def _safe_std(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    if x.size <= 1:
        return 0.0
    return float(np.nanstd(x))


def estimate_lag_by_xcorr(raw: np.ndarray, filt: np.ndarray, max_lag: int = 25) -> int:
    """
    用互相关估计“滤波输出相对原始”的延迟（单位：样本）。
    返回 lag>=0 表示 filt 更“滞后”。
    """
    raw = np.asarray(raw, dtype=float)
    filt = np.asarray(filt, dtype=float)
    n = min(len(raw), len(filt))
    if n < 5:
        return 0

    a = raw[:n] - np.nanmean(raw[:n])
    b = filt[:n] - np.nanmean(filt[:n])
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0

    best_lag = 0
    best_corr = -1e18
    for lag in range(0, max_lag + 1):
        # b 滞后 lag：b[lag:] 对齐 a[:-lag]
        if lag == 0:
            aa, bb = a, b
        else:
            aa, bb = a[:-lag], b[lag:]
        if len(aa) < 5:
            break
        corr = float(np.dot(aa, bb) / (np.linalg.norm(aa) * np.linalg.norm(bb) + 1e-12))
        if corr > best_corr:
            best_corr = corr
            best_lag = lag
    return int(best_lag)


def score_filter(raw: np.ndarray, filt: np.ndarray) -> Dict[str, float]:
    """
    给每个滤波结果计算一组指标（越适合实时规则引擎越好）：
    - smoothness: 一阶差分的标准差（越小越平滑）
    - lag: 互相关估计的延迟（越小越实时）
    - diff_corr: 一阶差分相关（越大越保持变化趋势）
    - amp_ratio: 标准差比值（越接近 1 越“幅值不失真”，但也可能意味着没去噪）
    """
    raw = np.asarray(raw, dtype=float)
    filt = np.asarray(filt, dtype=float)
    n = min(len(raw), len(filt))
    raw = raw[:n]
    filt = filt[:n]

    d_raw = np.diff(raw)
    d_f = np.diff(filt)

    smoothness = _safe_std(d_f)
    lag = float(estimate_lag_by_xcorr(raw, filt, max_lag=25))

    # 变化趋势保持：用差分相关
    if d_raw.size < 3 or _safe_std(d_raw) == 0 or _safe_std(d_f) == 0:
        diff_corr = 0.0
    else:
        diff_corr = float(np.corrcoef(d_raw, d_f)[0, 1])
        if math.isnan(diff_corr):
            diff_corr = 0.0

    amp_ratio = float((_safe_std(filt) + 1e-12) / (_safe_std(raw) + 1e-12))

    return {
        "smoothness": smoothness,
        "lag": lag,
        "diff_corr": diff_corr,
        "amp_ratio": amp_ratio,
    }


def composite_rank(metrics: Dict[str, Dict[str, float]]) -> Tuple[str, Dict[str, float], Dict[str, float]]:
    """
    将不同维度归一化后打综合分（越大越好）。
    设计偏好：强去噪 + 低滞后 + 保持变化趋势。
    """
    names = list(metrics.keys())
    smooth = np.array([metrics[n]["smoothness"] for n in names], dtype=float)
    lag = np.array([metrics[n]["lag"] for n in names], dtype=float)
    corr = np.array([metrics[n]["diff_corr"] for n in names], dtype=float)

    def norm01(v: np.ndarray) -> np.ndarray:
        v = np.asarray(v, dtype=float)
        lo, hi = float(np.nanmin(v)), float(np.nanmax(v))
        if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo < 1e-12:
            return np.zeros_like(v)
        return (v - lo) / (hi - lo)

    # smoothness/lag 越小越好，所以取 1 - norm
    smooth_score = 1.0 - norm01(smooth)
    lag_score = 1.0 - norm01(lag)
    corr_score = norm01(corr)  # 越大越好

    # 权重：更偏实时（lag）与规则稳定性（smoothness），同时要求趋势保持（corr）
    w_smooth, w_lag, w_corr = 0.45, 0.35, 0.20
    total = w_smooth * smooth_score + w_lag * lag_score + w_corr * corr_score

    best_idx = int(np.nanargmax(total))
    best_name = names[best_idx]

    totals = {n: float(total[i]) for i, n in enumerate(names)}
    best_metrics = metrics[best_name]
    return best_name, best_metrics, totals


# =========================
# 主流程：读取CSV、滤波、画图对比、给结论
# =========================
def load_raw_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if df.shape[1] < 3:
        raise ValueError(f"CSV 列数不足(期望>=3): {csv_path}")

    # 你的 CSV 固定三列，但列名可能是中文；这里统一映射
    df = df.copy()
    df.columns = ["time", "force", "displacement"] + [f"extra_{i}" for i in range(df.shape[1] - 3)]
    df["time_dt"] = pd.to_datetime(df["time"], errors="coerce")

    # 用秒级时间戳（OneEuro 需要 t）
    t0 = df["time_dt"].iloc[0]
    df["t_s"] = (df["time_dt"] - t0).dt.total_seconds()

    # 去掉无效行
    df = df.dropna(subset=["time_dt", "t_s", "force"]).reset_index(drop=True)
    return df


def run_filter_benchmark(
    csv_path: Path,
    *,
    force_col: str = "force",
    max_points: int | None = None,
    output_dir: Path | None = None,
) -> Path:
    df = load_raw_csv(csv_path)
    if max_points is not None and len(df) > max_points:
        df = df.iloc[:max_points].reset_index(drop=True)

    x = df[force_col].to_numpy(dtype=float)
    t_s = df["t_s"].to_numpy(dtype=float)

    # 一组“实时友好”的候选滤波器（参数偏保守）
    filters: Dict[str, Callable[[], np.ndarray]] = {
        "RAW(不滤波)": lambda: x.copy(),
        "EWMA(alpha=0.25)": lambda: ewma_filter(x, alpha=0.25),
        "EWMA(alpha=0.15)": lambda: ewma_filter(x, alpha=0.15),
        "SMA(window=7)": lambda: sma_filter(x, window=7),
        "SMA(window=11)": lambda: sma_filter(x, window=11),
        "Median(window=7)": lambda: median_filter_causal(x, window=7),
        "OneEuro(min=0.8,beta=0.02)": lambda: one_euro_filter(
            x, t_s, min_cutoff_hz=0.8, beta=0.02, d_cutoff_hz=1.0
        ),
        "OneEuro(min=0.6,beta=0.04)": lambda: one_euro_filter(
            x, t_s, min_cutoff_hz=0.6, beta=0.04, d_cutoff_hz=1.0
        ),
    }

    series: Dict[str, np.ndarray] = {name: fn() for name, fn in filters.items()}
    metrics: Dict[str, Dict[str, float]] = {name: score_filter(x, y) for name, y in series.items()}

    # 不参与“最佳滤波”竞选：RAW
    metrics_for_rank = {k: v for k, v in metrics.items() if not k.startswith("RAW")}
    best_name, best_metrics, totals = composite_rank(metrics_for_rank)

    # ========= 画图 =========
    if output_dir is None:
        output_dir = csv_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    out_png = output_dir / f"filter_compare_{csv_path.stem}.png"
    out_pair_png = output_dir / f"filter_before_after_{csv_path.stem}.png"

    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    fig = plt.figure(figsize=(14, 9), dpi=120)
    gs = fig.add_gridspec(2, 1, height_ratios=[2.3, 1.0], hspace=0.28)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(df["time_dt"], x, color="black", linewidth=1.0, alpha=0.55, label="RAW")

    # 只画少数曲线避免太乱：把 OneEuro + EWMA + SMA(较好) + Median 画出来
    to_plot = [
        best_name,
        "EWMA(alpha=0.25)",
        "SMA(window=7)",
        "Median(window=7)",
        "OneEuro(min=0.8,beta=0.02)",
    ]
    plotted = set()
    for name in to_plot:
        if name in series and name not in plotted:
            y = series[name]
            ax1.plot(df["time_dt"], y, linewidth=1.6, label=name)
            plotted.add(name)

    ax1.set_title(f"力值曲线滤波对比（最佳：{best_name}）")
    ax1.set_xlabel("时间")
    ax1.set_ylabel("力值")
    ax1.grid(True, alpha=0.25)
    ax1.legend(loc="upper left", ncol=2, fontsize=9, frameon=True)

    ax2 = fig.add_subplot(gs[1, 0])
    names = list(metrics_for_rank.keys())
    scores = np.array([totals[n] for n in names], dtype=float)
    order = np.argsort(scores)[::-1]
    names_sorted = [names[i] for i in order]
    scores_sorted = [float(scores[i]) for i in order]
    colors = ["tab:orange" if n == best_name else "tab:blue" for n in names_sorted]
    ax2.bar(range(len(names_sorted)), scores_sorted, color=colors, alpha=0.9)
    ax2.set_xticks(range(len(names_sorted)))
    ax2.set_xticklabels(names_sorted, rotation=20, ha="right")
    ax2.set_ylim(0, max(scores_sorted) * 1.15 if scores_sorted else 1)
    ax2.set_title("综合评分（越高越适合实时规则引擎：更平滑 + 更低滞后 + 趋势保持）")
    ax2.grid(True, axis="y", alpha=0.25)

    # 注释最佳指标
    txt = (
        f"最佳: {best_name}\n"
        f"smoothness(std(diff))={best_metrics['smoothness']:.3f}\n"
        f"lag(samples)={best_metrics['lag']:.0f}\n"
        f"diff_corr={best_metrics['diff_corr']:.3f}\n"
        f"amp_ratio={best_metrics['amp_ratio']:.3f}"
    )
    ax2.text(
        0.01,
        0.98,
        txt,
        transform=ax2.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.75", alpha=0.95),
    )

    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)

    # ========= 画图2：每种滤波器“前后对比”总图 =========
    compare_names = [n for n in series.keys() if not n.startswith("RAW")]
    n_filters = len(compare_names)
    n_cols = 2
    n_rows = int(math.ceil(n_filters / n_cols))
    fig2, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4.2 * n_rows), dpi=120, sharex=True)
    axes = np.atleast_1d(axes).reshape(n_rows, n_cols)

    for idx, name in enumerate(compare_names):
        r = idx // n_cols
        c = idx % n_cols
        ax = axes[r, c]
        y = series[name]

        # 每个子图都画“滤波前(raw) + 滤波后(filtered)”
        ax.plot(df["time_dt"], x, color="black", linewidth=1.0, alpha=0.45, label="RAW(滤波前)")
        ax.plot(df["time_dt"], y, color="tab:blue", linewidth=1.5, label="Filtered(滤波后)")

        m = metrics[name]
        title = (
            f"{name}"
            f"\nlag={m['lag']:.0f}, smooth={m['smoothness']:.2f}, corr={m['diff_corr']:.2f}"
        )
        if name == best_name:
            title += "  ★最佳"
        ax.set_title(title, fontsize=10)
        ax.grid(True, alpha=0.25)
        if r == n_rows - 1:
            ax.set_xlabel("时间")
        ax.set_ylabel("力值")
        ax.legend(loc="upper left", fontsize=8, frameon=True)

    # 隐藏多余空子图
    total_axes = n_rows * n_cols
    for idx in range(n_filters, total_axes):
        r = idx // n_cols
        c = idx % n_cols
        axes[r, c].axis("off")

    fig2.suptitle("各滤波器：滤波前 vs 滤波后对比（同图逐项查看）", fontsize=14, y=0.995)
    fig2.tight_layout()
    fig2.savefig(out_pair_png)
    plt.close(fig2)

    print(f"[完成] 已生成对比图: {out_png}")
    print(f"[完成] 已生成逐滤波器前后对比图: {out_pair_png}")
    print(f"[结论] 最适合实时规则引擎的滤波器: {best_name}")
    print(f"[指标] {best_metrics}")
    return out_pair_png


if __name__ == "__main__":
    repo_dir = Path(__file__).resolve().parent
    csv_path = repo_dir / "data/1.csv"
    run_filter_benchmark(csv_path, max_points=None, output_dir=repo_dir)
