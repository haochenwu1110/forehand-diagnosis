"""Paper B 预处理配置。改这里就行，不用动其他代码。"""

# 受试者分组由 convert_csv.py 的 groups.csv 提供（见 groups_template.csv）；这里保留空字典以兼容旧接口
GROUPS = {}
GROUP_ORDER = ["pro", "amateur", "novice"]

# 整人剔除（写明原因，论文里要报告）；本研究无整人剔除
EXCLUDE_SUBJECTS = []

# 左手持拍的受试者（会做左右镜像）。目前数据里 IMU 都在右臂，默认为空
LEFT_HANDED = []

# 统一重采样频率（Hz）。原始约 40 Hz 且不均匀，这里插值到均匀网格
FS = 60.0
# 低通滤波（零相位 Butterworth）
LOWPASS_HZ = 10.0
FILTER_ORDER = 4

# ---------------- 单拍剔除标准（硬性） ----------------
# 核心窗口：t2 前 0.1 s 到 t3 后 0.1 s（前挥 + 击球）
CORE_PAD = (0.10, 0.10)
# 核心窗口内原始相邻帧间隔上限（秒）
MAX_GAP_CORE_S = 0.08
# 核心窗口内被插值修补的原始帧比例上限（骨架瞬移、关节无效或跳变）
MAX_REPAIRED_FRAC = 0.10
# 关节角生理范围（度），归一化窗口内任一点越界即剔除
PLAUSIBLE_RANGES = {
    "pelvis_rot_imu": (-150, 150), "thorax_rot_imu": (-180, 180), "separation_imu": (-100, 100),
    "knee_flex_L": (-15, 130), "knee_flex_R": (-15, 130),
    "elbow_flex_R": (-15, 170), "elbow_flex_L": (-15, 170),
    "shoulder_elev_R": (0, 180), "shoulder_elev_L": (0, 180),
    "trunk_forward_lean": (-45, 70), "trunk_lateral_tilt": (-50, 50),
}
# 每个受试者至少要有这么多拍通过，否则在报告里警告
MIN_STROKES_PER_SUBJECT = 30

# 骨盆 / 胸廓朝向以哪个事件为零点（消除站位和场次坐标系朝向的差异）
YAW_REFERENCE = "t1"

# 分段时间归一化：(起点事件, 终点事件, 该段采样点数)
# 事件可以写 "t1"/"t2"/"t3"，或 "t3+0.4" 这种相对 t3 的偏移（秒）
SEGMENTS = [
    ("t1", "t2", 50),     # 引拍
    ("t2", "t3", 50),     # 前挥到击球
    ("t3", "t3+0.4", 25), # 随挥
]

# 进入归一化矩阵的关节角通道（顺序即矩阵通道顺序）
# v3.1 移出：pelvis_yaw / thorax_yaw / x_factor（骨架髋、肩连线，见 kinematics.imu_yaw 注释）、
#          shoulder_horiz_R（手臂竖直时无定义，改用 uarm_forward_R）。它们仍写进逐拍 CSV。
# 未列入：wrist_dev_R（骨架里手和前臂刚性相连，恒约为 0）、
#         racket_forearm_R（球拍 IMU 与前臂 IMU 的世界坐标系疑似不一致，相对转角可达 180°，待核实）
# 这两个仍会写进 uniform/ 下的逐拍 CSV，方便以后检查
CHANNELS = [
    # 第一类：IMU 直接测量（可信度最高）
    "pelvis_rot_imu", "thorax_rot_imu", "separation_imu",
    # 第二类：右臂，骨架由右上臂、右前臂 IMU 驱动（较可信）
    "shoulder_elev_R", "uarm_forward_R", "elbow_flex_R",
    # 第三类：躯干姿态、下肢、左臂，没有对应 IMU，来自骨架估计（可信度待核实，结论要谨慎）
    "trunk_forward_lean", "trunk_lateral_tilt",
    "hip_flex_L", "hip_flex_R", "knee_flex_L", "knee_flex_R",
    "shoulder_elev_L", "elbow_flex_L",
]
# 各通道的可信度分级，分析脚本据此分开报告
CHANNEL_TIER = {**{c: 1 for c in CHANNELS[:3]}, **{c: 2 for c in CHANNELS[3:6]}, **{c: 3 for c in CHANNELS[6:]},
                **{f"angspeed_{n}": 1 for n in ("pelvis", "thorax", "upper_arm", "forearm", "racket")}}

# 新数据（convert_csv.py 转换的全身骨架）所有关节来自同一套系统，可信度分级不再成立：
# 运行后续脚本时加环境变量 PAPERB_UNIFORM_TIER=1，全部通道按 1 级处理（不再对 3 级通道 ×0.5）
import os as _os
if _os.environ.get("PAPERB_UNIFORM_TIER") == "1":
    CHANNEL_TIER = {c: 1 for c in CHANNEL_TIER}

# 用于近端到远端顺序分析的 IMU（列名前缀 -> 输出名），按近端到远端排列
IMU_CHAIN = [
    ("imu_Pelvis", "pelvis"),
    ("imu_Sternum", "thorax"),
    ("imu_R_UpArm", "upper_arm"),
    ("imu_R_LowArm", "forearm"),
    ("racket", "racket"),
]
# 找角速度峰值的窗口（事件表达式，同 SEGMENTS）
CHAIN_WINDOW = ("t2", "t3+0.15")

# 阶段时长异常检测：同一受试者内鲁棒 z 分数超过该值则标记
PHASE_OUTLIER_Z = 3.5

# 骨架跳变检测：与 5 帧滑动中值的偏差（米）
# 骨盆偏差超过 ROOT_SPIKE_M -> 整帧所有关节当作缺失再插值（整副骨架瞬移）
# 单个关节偏差超过 JOINT_SPIKE_M -> 该关节该帧当作缺失
ROOT_SPIKE_M = 0.20
JOINT_SPIKE_M = 0.35
# IMU 四元数与上一帧完全相同（采样保持/掉包），当作缺失，由球面插值补
DROP_HELD_QUATS = True

# 球拍-前臂相对转角的基准窗口：t1 之前 [0.3 s, 0.05 s] 的准备姿势
BASELINE_WIN = (0.3, 0.05)

# 角度跳变检测：60 Hz 均匀网格上，相邻两点之间任一关节角变化超过该值（度）即剔除该拍。
# 45°/帧 = 2700°/s，超过上肢真实角速度上限，只可能是骨架翻转或角度定义跳变。
MAX_ANGLE_STEP_DEG = 45.0
# 骨架-IMU 一致性检查用的节段：(骨架朝向通道, IMU 名)
SKELETON_IMU_PAIRS = [("pelvis_yaw", "pelvis"), ("thorax_yaw", "thorax")]
