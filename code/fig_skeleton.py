"""骨架与关节角定义示意图（新图 2）。
(a) 正面：21 关节骨架 + 冠状面内可见的角（肩抬高、肘屈曲、膝屈曲、躯干侧倾、髋/肩连线）
(b) 俯视：骨盆/躯干转动与分离角
(c) 侧视：躯干前倾、上臂前后角、髋屈曲、膝屈曲
"""
import sys
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Arc, Circle, FancyArrowPatch, Ellipse  # noqa: E402
from PIL import Image  # noqa: E402

OUT = sys.argv[1] if len(sys.argv) > 1 else "Fig2_skeleton"
# 背景人体图 mannequin.png 与传感器照片 sensor_pn3.png 不随代码公开（版权属于原作者 / 设备厂商），
# 放在第 2 个参数指定的目录里即可（默认当前目录）
ASSETS = sys.argv[2] if len(sys.argv) > 2 else "."
MM = 1 / 25.4
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
    "font.size": 7, "pdf.fonttype": 42, "axes.unicode_minus": True,
})
INK, INK2, MUTED = "#1f1f1e", "#52514e", "#8a8984"
C_ROT, C_POST, C_RARM, C_LARM, C_LEG = "#2a5caa", "#1a9e75", "#e0622a", "#8a4fbf", "#b07d1a"
BONE = "#3a3936"

# ------------------------------------------------------------------ 正面图关节（图片像素坐标，R = 图左）
J = {
    "head": (545, 135), "neck1": (545, 215), "neck": (545, 252),
    "spine2": (545, 370), "spine1": (545, 460), "spine": (545, 540), "pelvis": (545, 612),
    "r_collar": (505, 268), "r_shoulder": (400, 318), "r_elbow": (326, 515), "r_wrist": (293, 690),
    "l_collar": (585, 268), "l_shoulder": (690, 318), "l_elbow": (754, 518), "l_wrist": (786, 684),
    "r_hip": (472, 642), "r_knee": (394, 900), "r_ankle": (357, 1200),
    "l_hip": (618, 642), "l_knee": (694, 900), "l_ankle": (728, 1200),
}
BONES = [("head", "neck1"), ("neck1", "neck"), ("neck", "spine2"), ("spine2", "spine1"), ("spine1", "spine"),
         ("spine", "pelvis"), ("neck", "r_collar"), ("r_collar", "r_shoulder"), ("r_shoulder", "r_elbow"),
         ("r_elbow", "r_wrist"), ("neck", "l_collar"), ("l_collar", "l_shoulder"), ("l_shoulder", "l_elbow"),
         ("l_elbow", "l_wrist"), ("pelvis", "r_hip"), ("r_hip", "r_knee"), ("r_knee", "r_ankle"),
         ("pelvis", "l_hip"), ("l_hip", "l_knee"), ("l_knee", "l_ankle")]
assert len(J) == 21


def P(k):
    return np.array(J[k], float)


def ang(v):
    """数据坐标里向量的方向角（度）。"""
    return np.degrees(np.arctan2(v[1], v[0]))


def arc_between(ax, c, d1, d2, r, color, lw=1.4, z=6):
    """在顶点 c 处画 d1 与 d2 两方向之间的小弧（取小于 180° 的一侧）。返回弧中点方向（单位向量）。"""
    a1, a2 = ang(d1), ang(d2)
    da = (a2 - a1 + 540) % 360 - 180
    t1, t2 = (a1, a1 + da) if da > 0 else (a1 + da, a1)
    ax.add_patch(Arc(c, 2 * r, 2 * r, theta1=t1, theta2=t2, color=color, lw=lw, zorder=z))
    mid = np.radians((t1 + t2) / 2)
    return np.array([np.cos(mid), np.sin(mid)])


def dashed(ax, a, b, color, lw=0.9, z=5):
    ax.plot([a[0], b[0]], [a[1], b[1]], color=color, lw=lw, ls=(0, (3, 2)), zorder=z)


def callout(ax, xy, txt, xytext, color, ha="left"):
    ax.annotate(txt, xy=xy, xytext=xytext, textcoords="data", ha=ha, va="center", fontsize=6.6, color=INK,
                arrowprops=dict(arrowstyle="-", color=color, lw=0.7, shrinkA=1, shrinkB=2), zorder=9,
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=color, lw=0.8))


def panel_label(ax, s, x=0.0, y=1.0):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=9, fontweight="bold", va="top", ha="left")


fig = plt.figure(figsize=(174 * MM, 132 * MM))

# ================================================================== (a) 正面
axA = fig.add_axes([0.0, 0.0, 0.53, 0.95])
img = np.asarray(Image.open(f"{ASSETS}/mannequin.png").convert("RGB")).astype(float) / 255
img = 1 - 0.55 * (1 - img)                      # 人体淡化，让骨架和标注更清楚
axA.imshow(img, zorder=0)
axA.set_xlim(-60, 1146)
axA.set_ylim(1360, 40)
axA.set_aspect("equal")
axA.axis("off")

# 髋连线、肩连线（转动的来源）
for a, b, col in (("r_shoulder", "l_shoulder", C_ROT), ("r_hip", "l_hip", C_ROT)):
    v = P(b) - P(a)
    dashed(axA, P(a) - 0.18 * v, P(b) + 0.18 * v, col, lw=1.1)
# 躯干轴：骨盆 → 颈
dashed(axA, P("pelvis"), P("pelvis") + np.array([0, -430]), MUTED, lw=0.7)
axA.plot(*zip(P("pelvis"), P("neck")), color=C_POST, lw=2.2, zorder=4, solid_capstyle="round", alpha=0.9)

for a, b in BONES:
    axA.plot(*zip(P(a), P(b)), color=BONE, lw=1.3, zorder=3, solid_capstyle="round")
used = {"pelvis", "spine1", "neck", "r_shoulder", "r_elbow", "r_wrist", "l_shoulder", "l_elbow", "l_wrist",
        "r_hip", "r_knee", "r_ankle", "l_hip", "l_knee", "l_ankle"}
for k, (x, y) in J.items():
    axA.scatter([x], [y], s=16 if k in used else 9, color="white" if k in used else BONE,
                edgecolor=BONE, lw=0.9, zorder=7)

# 肩抬高：上臂 vs 躯干向下轴
down = np.array([0, 1.0])
M = {}
for s_, col in (("r", C_RARM), ("l", C_LARM)):
    sh, el = P(f"{s_}_shoulder"), P(f"{s_}_elbow")
    dashed(axA, sh, sh + down * 150, col)
    M[f"sh_{s_}"] = sh + 85 * arc_between(axA, sh, down, el - sh, 85, col)
# 肘屈曲：上臂延长线 vs 前臂
for s_, col in (("r", C_RARM), ("l", C_LARM)):
    sh, el, wr = P(f"{s_}_shoulder"), P(f"{s_}_elbow"), P(f"{s_}_wrist")
    u = (el - sh) / np.linalg.norm(el - sh)
    dashed(axA, el, el + u * 130, col)
    M[f"el_{s_}"] = el + 70 * arc_between(axA, el, u, wr - el, 70, col)
# 膝屈曲：大腿延长线 vs 小腿
for s_ in ("r", "l"):
    hp, kn, an = P(f"{s_}_hip"), P(f"{s_}_knee"), P(f"{s_}_ankle")
    u = (kn - hp) / np.linalg.norm(kn - hp)
    dashed(axA, kn, kn + u * 150, C_LEG)
    M[f"kn_{s_}"] = kn + 90 * arc_between(axA, kn, u, an - kn, 90, C_LEG)

sh_end = P("l_shoulder") + 0.18 * (P("l_shoulder") - P("r_shoulder"))
hp_end = P("l_hip") + 0.18 * (P("l_hip") - P("r_hip"))
callout(axA, M["sh_r"], "Shoulder elevation R\n(upper arm vs\ntrunk down-axis)", (-50, 455), C_RARM, "left")
callout(axA, M["el_r"], "Elbow flexion R\n(0° = straight)", (-50, 640), C_RARM, "left")
callout(axA, M["kn_r"], "Knee flexion R\n(0° = straight)", (-50, 1060), C_LEG, "left")
callout(axA, sh_end, "Shoulder line\n→ thorax rotation", (1140, 175), C_ROT, "right")
callout(axA, M["sh_l"], "Shoulder\nelevation L", (1140, 400), C_LARM, "right")
callout(axA, M["el_l"], "Elbow\nflexion L", (1140, 580), C_LARM, "right")
callout(axA, hp_end, "Hip line\n→ pelvis rotation", (1140, 760), C_ROT, "right")
callout(axA, P("spine2") + np.array([0, 30]), "Trunk axis (pelvis→neck)\n→ lateral tilt, forward lean",
        (1140, 930), C_POST, "right")
callout(axA, M["kn_l"], "Knee flexion L\n(0° = straight)", (1140, 1100), C_LEG, "right")
axA.text(545, 1345, "Front view · 21-joint skeleton (○ joints used for angles; ▬ sensor sites, schematic)", ha="center", va="bottom",
         fontsize=6.6, color=INK2)
axA.text(250, 1240, "R", fontsize=9, fontweight="bold", color=INK2, ha="center")
axA.text(850, 1240, "L", fontsize=9, fontweight="bold", color=INK2, ha="center")
panel_label(axA, "a", 0.0, 1.03)

# ---------- 惯性传感器位置（Perception Neuron 3，17 个；示意位置）+ 放大引出
from matplotlib.patches import FancyBboxPatch as _FB, ConnectionPatch as _CP  # noqa: E402
SENS = {"head": (545, 92), "chest": (545, 318), "hips": (545, 578),
        "r_shoulder": (455, 287), "l_shoulder": (635, 287),
        "r_uarm": (352, 418), "l_uarm": (738, 418), "r_farm": (306, 606), "l_farm": (772, 604),
        "r_hand": (284, 742), "l_hand": (796, 742), "r_thigh": (428, 772), "l_thigh": (662, 772),
        "r_shank": (374, 1052), "l_shank": (713, 1052), "r_foot": (338, 1252), "l_foot": (752, 1252)}
assert len(SENS) == 17
SW, SH = 30, 20
for k_, (x_, y_) in SENS.items():
    axA.add_patch(_FB((x_ - SW / 2, y_ - SH / 2), SW, SH, boxstyle="round,pad=0,rounding_size=7",
                      fc="#3b3b3d", ec="white", lw=0.6, zorder=8))
    axA.scatter([x_], [y_], s=5, color="#d9363e", zorder=9, lw=0)
# 放大：右上臂的传感器 → 左上角的传感器照片
zc = np.array(SENS["r_uarm"], float)
axA.add_patch(Circle(zc, 32, fc="none", ec=INK2, lw=0.8, zorder=9))
bx0, bx1, by0, by1 = -45, 215, 62, 262
sen = np.asarray(Image.open(f"{ASSETS}/sensor_pn3.png").convert("RGB")).astype(float) / 255
axA.add_patch(_FB((bx0, by0), bx1 - bx0, by1 - by0, boxstyle="round,pad=0,rounding_size=10",
                  fc="white", ec=INK2, lw=0.8, zorder=9))
ih = (by1 - by0) - 52
iw = ih * sen.shape[1] / sen.shape[0]
ix0 = (bx0 + bx1) / 2 - iw / 2
axA.imshow(sen, extent=(ix0, ix0 + iw, by0 + 8 + ih, by0 + 8), zorder=10)
axA.text((bx0 + bx1) / 2, by1 - 5, "IMU sensor (×17)", ha="center", va="bottom", fontsize=6.4, color=INK,
         zorder=11)
for cy_ in (by0, by1):
    ang_ = np.arctan2(cy_ - zc[1], bx1 - zc[0])
    axA.plot([zc[0] + 32 * np.cos(ang_), bx1], [zc[1] + 32 * np.sin(ang_), cy_], color=INK2, lw=0.6,
             zorder=8)
axA.set_xlim(-60, 1146)
axA.set_ylim(1360, 40)


# ================================================================== (b) 俯视：转动
axB = fig.add_axes([0.56, 0.50, 0.44, 0.47])
axB.set_xlim(-1.55, 1.55)
axB.set_ylim(-1.30, 1.15)
axB.set_aspect("equal")
axB.axis("off")
panel_label(axB, "b", -0.02, 1.03)
axB.text(0, 1.12, "Top view · trunk rotation", ha="center", va="top", fontsize=7, color=INK2)
ref, th_p, th_t = -40.0, 8.0, 35.0      # t1（引拍、身体关闭）→ 当前时刻（已打开）


def uvec(a_):
    return np.array([np.cos(np.radians(a_)), np.sin(np.radians(a_))])


axB.add_patch(Ellipse((0, 0), 1.30, 0.55, angle=th_t, fc="#f1f0ec", ec="none", zorder=0))
axB.add_patch(Circle((0, 0), 0.15, fc="#e2e1dc", ec=MUTED, lw=0.6, zorder=2))
v = uvec(ref)
dashed(axB, -v * 0.70, v * 0.70, MUTED, lw=0.8)
axB.text(*(-v * 0.70 + np.array([-0.03, 0.02])), "hip & shoulder\nlines at t1", fontsize=6, color=MUTED,
         ha="right", va="bottom")
C_PEL = "#7fa0d6"
for a_, L, col, lab in ((th_t, 1.40, C_ROT, "shoulder line"), (th_p, 1.00, C_PEL, "hip line")):
    v = uvec(a_)
    axB.plot(*zip(-v * L / 2, v * L / 2), color=col, lw=2.4, solid_capstyle="round", zorder=4)
    axB.text(*(-v * L / 2 + np.array([-0.04, 0])), lab, fontsize=6.3, color=col, va="center", ha="right")
axB.text(*(uvec(th_t) * 0.78), "R", fontsize=7.5, fontweight="bold", color=INK2, ha="left", va="bottom")
for r, a1, a2, col, lab in ((0.40, ref, th_p, C_PEL, "θp"), (0.62, ref, th_t, C_ROT, "θt"),
                           (0.92, th_p, th_t, C_POST, "θt − θp")):
    axB.add_patch(Arc((0, 0), 2 * r, 2 * r, theta1=a1, theta2=a2, color=col, lw=1.3, zorder=6))
    e = np.radians(a2)
    axB.annotate("", xy=(r * np.cos(e), r * np.sin(e)), xytext=(r * np.cos(e - 0.1), r * np.sin(e - 0.1)),
                 arrowprops=dict(arrowstyle="-|>", color=col, lw=1.0, mutation_scale=7), zorder=7)
    axB.text(*(uvec((a1 + a2) / 2) * (r + 0.05)), lab, fontsize=6.8, color=col, ha="left", va="center",
             fontweight="bold", zorder=8)
axB.text(-1.5, -1.05, "θp pelvis rotation, θt thorax rotation: horizontal orientation of the hip /\n"
         "shoulder line relative to t1; + = toward the swing (body opening).\n"
         "Thorax–pelvis separation = θt − θp.",
         fontsize=6.3, color=INK, va="center")
axB.annotate("", xy=(1.42, 1.0), xytext=(1.42, 0.45), arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=0.9))
axB.text(1.35, 0.72, "to net", rotation=90, fontsize=6, color=MUTED, ha="right", va="center")


# ================================================================== (c) 侧视：矢状面角
axC = fig.add_axes([0.56, 0.0, 0.44, 0.49])
axC.set_xlim(-1.2, 1.62)
axC.set_ylim(-0.08, 2.05)
axC.set_aspect("equal")
axC.axis("off")
panel_label(axC, "c", -0.02, 1.02)
axC.text(-1.15, 2.04, "Side view (from the right) · facing →", ha="left", va="top", fontsize=7, color=INK2)
S = {"ankle": np.array([0.10, 0.05]), "knee": np.array([0.38, 0.50]), "hip": np.array([0.05, 0.95]),
     "neck": np.array([0.40, 1.62]), "shoulder": np.array([0.38, 1.57]), "elbow": np.array([0.72, 1.36]),
     "wrist": np.array([1.02, 1.46])}
axC.plot([-0.05, 0.35], [0.02, 0.02], color=MUTED, lw=1.0)
for a, b in (("ankle", "knee"), ("knee", "hip"), ("hip", "neck"), ("shoulder", "elbow"), ("elbow", "wrist")):
    axC.plot(*zip(S[a], S[b]), color=BONE, lw=2.0, solid_capstyle="round", zorder=3)
axC.add_patch(Circle(S["neck"] + np.array([0.06, 0.17]), 0.12, fc="#e2e1dc", ec=BONE, lw=1.0, zorder=3))
for k in S:
    axC.scatter(*S[k], s=14, color="white", edgecolor=BONE, lw=0.9, zorder=5)
trunk = S["neck"] - S["hip"]
tu = trunk / np.linalg.norm(trunk)
# 躯干前倾：躯干轴 vs 竖直
dashed(axC, S["hip"], S["hip"] + np.array([0, 0.6]), MUTED, lw=0.8)
m_ = arc_between(axC, S["hip"], np.array([0, 1.0]), trunk, 0.30, C_POST)
callout(axC, S["hip"] + 0.30 * m_, "Trunk forward lean\n(trunk axis vs vertical)", (-1.15, 1.30),
        C_POST, "left")
# 上臂前后角：上臂 vs 躯干冠状面（侧视中与躯干轴平行，过肩）
dashed(axC, S["shoulder"], S["shoulder"] - tu * 0.45, C_RARM)
m_ = arc_between(axC, S["shoulder"], -tu, S["elbow"] - S["shoulder"], 0.22, C_RARM)
callout(axC, S["shoulder"] + 0.22 * m_, "R upper-arm forward angle\n(upper arm vs thorax\n"
        "coronal plane; + = forward)", (1.58, 1.12), C_RARM, "right")
# 髋屈曲：大腿 vs 骨盆向下轴
dashed(axC, S["hip"], S["hip"] - tu * 0.45, C_LEG)
m_ = arc_between(axC, S["hip"], -tu, S["knee"] - S["hip"], 0.25, C_LEG)
callout(axC, S["hip"] + 0.25 * m_, "Hip flexion R / L\n(thigh vs pelvis down-axis)",
        (1.58, 0.72), C_LEG, "right")
# 膝屈曲
u = (S["knee"] - S["hip"]) / np.linalg.norm(S["knee"] - S["hip"])
dashed(axC, S["knee"], S["knee"] + u * 0.32, C_LEG)
m_ = arc_between(axC, S["knee"], u, S["ankle"] - S["knee"], 0.2, C_LEG)
callout(axC, S["knee"] + 0.2 * m_, "Knee flexion R / L", (1.58, 0.22), C_LEG, "right")

fig.savefig(OUT + ".png", dpi=600)
fig.savefig(OUT + ".pdf")
fig.savefig(OUT + ".tif", dpi=600, pil_kwargs={"compression": "tiff_lzw"})
print("saved", OUT)
