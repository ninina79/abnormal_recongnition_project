import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Ellipse
import numpy as np

fig, ax = plt.subplots(1, 1, figsize=(14, 11))
ax.set_xlim(0, 16)
ax.set_ylim(0, 14)
ax.axis('off')

# Style parameters
box_ec = '#2C3E50'
box_lw = 1.8
font_size = 20
start_fc = '#D5F5E3'
end_fc = '#FADBD8'
process_fc = '#E8F4FD'
init_fc = '#E8DAEF'
decision_fc = '#FEF9E7'

def draw_ellipse(ax, cx, cy, w, h, text, fc=start_fc, fs=font_size):
    ellipse = Ellipse((cx, cy), w, h, facecolor=fc, edgecolor=box_ec, linewidth=box_lw)
    ax.add_patch(ellipse)
    ax.text(cx, cy, text, ha='center', va='center', fontsize=fs,
            fontfamily='SimHei', linespacing=1.3)

def draw_rounded_box(ax, cx, cy, w, h, text, fc=process_fc, fs=font_size):
    box = FancyBboxPatch((cx - w/2, cy - h/2), w, h,
                         boxstyle="round,pad=0.1",
                         facecolor=fc, edgecolor=box_ec, linewidth=box_lw)
    ax.add_patch(box)
    ax.text(cx, cy, text, ha='center', va='center', fontsize=fs,
            fontfamily='SimHei', linespacing=1.3)

def draw_diamond(ax, cx, cy, w, h, text, fc=decision_fc, fs=font_size):
    diamond = plt.Polygon([[cx, cy+h/2], [cx+w/2, cy], [cx, cy-h/2], [cx-w/2, cy]],
                          facecolor=fc, edgecolor=box_ec, linewidth=box_lw, closed=True)
    ax.add_patch(diamond)
    ax.text(cx, cy, text, ha='center', va='center', fontsize=fs,
            fontfamily='SimHei')

def draw_arrow(ax, x1, y1, x2, y2):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->,head_length=1.2,head_width=0.8', color=box_ec, lw=3.0))

def draw_line(ax, points):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    ax.plot(xs, ys, color=box_ec, lw=2.0)

# Layout: serpentine (3 rows)
r1_y = 12.0
r1_x = [2.5, 6.0, 10.0, 13.5]

r2_y = 9.0
r2_x = [13.5, 10.0, 6.0, 2.5]

r3_y = 6.0
r3_x = [2.5, 6.0, 10.0, 13.5]

r4_y = 3.0

bw = 3.0
bh = 1.5

# === Row 1 ===
draw_ellipse(ax, r1_x[0], r1_y, 3.2, 1.6, '开始张拉', fc=start_fc)
draw_rounded_box(ax, r1_x[1], r1_y, bw, bh, '创建会话记录', fc=process_fc)
draw_rounded_box(ax, r1_x[2], r1_y, bw, bh, '启动工作线程', fc=process_fc)
draw_rounded_box(ax, r1_x[3], r1_y, 3.2, 1.8, '初始化组件\n(5个模块)', fc=init_fc, fs=14)

# Row 1 arrows
draw_arrow(ax, r1_x[0] + 1.6, r1_y, r1_x[1] - bw/2, r1_y)
draw_arrow(ax, r1_x[1] + bw/2, r1_y, r1_x[2] - bw/2, r1_y)
draw_arrow(ax, r1_x[2] + bw/2, r1_y, r1_x[3] - 1.6, r1_y)

# Corner: Row1 -> Row2
draw_arrow(ax, r1_x[3], r1_y - 0.9, r2_x[0], r2_y + bh/2)

# === Row 2 ===
draw_rounded_box(ax, r2_x[0], r2_y, bw, bh, '读取数据点', fc=process_fc)
draw_rounded_box(ax, r2_x[1], r2_y, bw, bh, '特征计算', fc=process_fc)
draw_rounded_box(ax, r2_x[2], r2_y, bw, bh, '阶段判定', fc=process_fc)
draw_rounded_box(ax, r2_x[3], r2_y, bw, bh, '规则检测', fc=process_fc)

# Row 2 arrows (right to left)
draw_arrow(ax, r2_x[0] - bw/2, r2_y, r2_x[1] + bw/2, r2_y)
draw_arrow(ax, r2_x[1] - bw/2, r2_y, r2_x[2] + bw/2, r2_y)
draw_arrow(ax, r2_x[2] - bw/2, r2_y, r2_x[3] + bw/2, r2_y)

# Corner: Row2 -> Row3
draw_arrow(ax, r2_x[3], r2_y - bh/2, r3_x[0], r3_y + bh/2)

# === Row 3 ===
draw_rounded_box(ax, r3_x[0], r3_y, bw, bh, 'IF/LOF检测', fc=process_fc)
draw_rounded_box(ax, r3_x[1], r3_y, bw, bh, 'TCN预测', fc=process_fc)
draw_rounded_box(ax, r3_x[2], r3_y, bw, bh, '写入数据库', fc=process_fc)
draw_rounded_box(ax, r3_x[3], r3_y, 3.2, bh, 'WebSocket推送', fc=process_fc)

# Row 3 arrows (left to right)
draw_arrow(ax, r3_x[0] + bw/2, r3_y, r3_x[1] - bw/2, r3_y)
draw_arrow(ax, r3_x[1] + bw/2, r3_y, r3_x[2] - bw/2, r3_y)
draw_arrow(ax, r3_x[2] + bw/2, r3_y, r3_x[3] - 1.6, r3_y)

# Corner: Row3 -> Decision
draw_arrow(ax, r3_x[3], r3_y - bh/2, r3_x[3], r4_y + 1.0)

# === Decision ===
draw_diamond(ax, r3_x[3], r4_y, 3.4, 2.1, '张拉结束?', fc=decision_fc)

# Yes -> End (left)
draw_ellipse(ax, 8.1, r4_y, 3.2, 1.6, '结束会话', fc=end_fc)
draw_arrow(ax, r3_x[3] - 1.7, r4_y, 8.0 + 1.6, r4_y)
ax.text(10.5, r4_y + 0.35, '是', fontsize=font_size, fontfamily='SimHei', color='#27AE60')

# No -> loop back to "读取数据点"
no_x = 15.8
draw_line(ax, [(r3_x[3] + 1.7, r4_y), (no_x, r4_y)])
draw_line(ax, [(no_x, r4_y), (no_x, r2_y)])
ax.annotate('', xy=(r2_x[0] + bw/2, r2_y), xytext=(no_x, r2_y),
            arrowprops=dict(arrowstyle='->,head_length=1.2,head_width=0.8', color=box_ec, lw=3.0))
ax.text(15.0, r4_y + 0.6, '否', fontsize=font_size, fontfamily='SimHei', color='#E74C3C')

# Title
# ax.text(8.0, 13.5, '图5-1  在线主控线程流程图', ha='center', va='center',
#         fontsize=18, fontfamily='SimHei', fontweight='bold')

plt.tight_layout()
plt.savefig('fig5_1_online_thread_flowchart.png', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.show()