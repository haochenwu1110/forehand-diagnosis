"""关节角与角速度计算。

坐标系：原始数据是 Unity 左手系（x 右、y 上、z 前）。这里把 z 取反变成右手系
（x 右、y 上、z 后，前向 = -z），之后所有叉乘和有符号角都在右手系下算。
四元数只用来算角速度大小和相对转角，这两个量对镜像不敏感，所以不需要转换。
"""
import numpy as np
from scipy.spatial.transform import Rotation as R

UP = np.array([0.0, 1.0, 0.0])

JOINTS = ["pelvis", "left_hip", "right_hip", "spine1", "left_knee", "right_knee",
          "spine2", "left_ankle", "right_ankle", "spine3", "left_foot", "right_foot",
          "neck", "left_collar", "right_collar", "head", "left_shoulder",
          "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist",
          "left_hand", "right_hand"]


# ---------- 基础工具 ----------
def unit(v):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.clip(n, 1e-9, None)


def angle_between(a, b):
    """两向量夹角（度），0–180。"""
    c = np.sum(unit(a) * unit(b), axis=-1)
    return np.degrees(np.arccos(np.clip(c, -1.0, 1.0)))


def joint_angle(p_prox, p_mid, p_dist):
    """三点夹角（度），在 p_mid 处。180 = 伸直。"""
    return angle_between(p_prox - p_mid, p_dist - p_mid)


def horizontal(v):
    h = v.copy()
    h[..., 1] = 0.0
    return h


def yaw_of_lateral(lat):
    """侧向向量在水平面上的偏航角（度），绕 +y 逆时针为正，并做相位展开。"""
    yaw = np.arctan2(-lat[..., 2], lat[..., 0])
    return np.degrees(np.unwrap(yaw))


def wrap180(a):
    return (a + 180.0) % 360.0 - 180.0


# ---------- 坐标系转换与镜像 ----------
def to_right_handed(P):
    """P: dict joint -> (T,3)。z 取反。"""
    out = {}
    for k, v in P.items():
        w = v.copy()
        w[:, 2] = -w[:, 2]
        out[k] = w
    return out


def mirror_left_right(P):
    """左手持拍者镜像：x 取反并交换左右关节。"""
    out = {}
    for k, v in P.items():
        w = v.copy()
        w[:, 0] = -w[:, 0]
        if k.startswith("left_"):
            k2 = "right_" + k[5:]
        elif k.startswith("right_"):
            k2 = "left_" + k[6:]
        else:
            k2 = k
        out[k2] = w
    return out


# ---------- 身体坐标系 ----------
def segment_frame(lateral, up):
    """由侧向（指向身体右侧）和上方向构造正交基：L(右)、U(上)、F(前)。"""
    U = unit(up)
    L = unit(lateral - np.sum(lateral * U, axis=-1, keepdims=True) * U)
    F = np.cross(U, L)  # 右手系 x=右, y=上 => 前 = U × L
    return L, U, F


# ---------- 关节角 ----------
def compute_angles(P):
    """P: dict joint -> (T,3)，右手系。返回 dict 通道名 -> (T,) 度。

    约定：
    - pelvis_yaw / thorax_yaw：髋连线 / 肩连线在水平面的朝向（实验室坐标）
    - x_factor = thorax_yaw - pelvis_yaw（躯干相对骨盆的扭转）
    - trunk_forward_lean / trunk_lateral_tilt：骨盆→颈 向量相对竖直方向的前倾 / 右倾
    - hip_flex：大腿相对骨盆向下轴的矢状面角，前屈为正
    - knee_flex / elbow_flex：180 - 三点夹角，0 = 伸直
    - shoulder_elev：上臂与躯干向下轴夹角，0 = 自然下垂
    - shoulder_horiz_R：上臂在躯干横断面内的方向，0 = 正侧方，+90 = 正前方（水平内收）
    - wrist_dev_R：前臂与手段夹角（不分屈伸方向；手段很短，可靠性低）
    """
    A = {}
    hip_lat = P["right_hip"] - P["left_hip"]
    sh_lat = P["right_shoulder"] - P["left_shoulder"]

    A["pelvis_yaw"] = yaw_of_lateral(horizontal(hip_lat))
    A["thorax_yaw"] = yaw_of_lateral(horizontal(sh_lat))
    A["x_factor"] = wrap180(A["thorax_yaw"] - A["pelvis_yaw"])

    # 骨盆坐标系（水平化）：用于躯干倾斜
    Lh, _, Fh = segment_frame(horizontal(hip_lat), np.broadcast_to(UP, hip_lat.shape))
    trunk = P["neck"] - P["pelvis"]
    A["trunk_forward_lean"] = np.degrees(np.arctan2(np.sum(trunk * Fh, -1), trunk[:, 1]))
    A["trunk_lateral_tilt"] = np.degrees(np.arctan2(np.sum(trunk * Lh, -1), trunk[:, 1]))

    # 骨盆解剖坐标系：髋屈
    Lp, Up, Fp = segment_frame(hip_lat, P["spine1"] - P["pelvis"])
    for s, side in (("L", "left"), ("R", "right")):
        thigh = P[f"{side}_knee"] - P[f"{side}_hip"]
        A[f"hip_flex_{s}"] = np.degrees(np.arctan2(np.sum(thigh * Fp, -1),
                                                   np.sum(thigh * -Up, -1)))
        A[f"knee_flex_{s}"] = 180.0 - joint_angle(P[f"{side}_hip"], P[f"{side}_knee"],
                                                  P[f"{side}_ankle"])

    # 胸廓坐标系：肩
    Lt, Ut, Ft = segment_frame(sh_lat, P["neck"] - P["spine1"])
    for s, side, lat_sign in (("R", "right", 1.0), ("L", "left", -1.0)):
        ua = P[f"{side}_elbow"] - P[f"{side}_shoulder"]
        A[f"shoulder_elev_{s}"] = angle_between(ua, -Ut)
        if s == "R":
            # 【易错-11】shoulder_horiz_R（上臂在水平面内的方向）用 arctan2 算，在 ±180° 处会跳 360°；
            # 更根本的问题是：手臂接近竖直（随挥时举过头顶）时水平方向没有定义，结果在 ±180° 间乱跳，
            # 展开也救不回来。v3 的真实数据里它出现了 ±200° 以上的假包络。
            # 所以它只保留在逐拍 CSV 里供查看，模型通道改用 uarm_forward_R：
            # 上臂与躯干冠状面的夹角（向前为正），范围 ±90°，任何姿势下都有定义。
            A["shoulder_horiz_R"] = np.degrees(np.unwrap(np.arctan2(np.sum(ua * Ft, -1),
                                                                    np.sum(ua * Lt, -1) * lat_sign)))
            A["uarm_forward_R"] = np.degrees(np.arcsin(np.clip(np.sum(unit(ua) * Ft, -1), -1, 1)))
        A[f"elbow_flex_{s}"] = 180.0 - joint_angle(P[f"{side}_shoulder"], P[f"{side}_elbow"],
                                                   P[f"{side}_wrist"])
    A["wrist_dev_R"] = angle_between(P["right_wrist"] - P["right_elbow"],
                                     P["right_hand"] - P["right_wrist"])
    return A


# ---------- IMU 角速度 ----------
def angular_speed(quats_xyzw, t):
    """quats: (T,4) xyzw，t: (T,) 秒（均匀网格）。返回角速度大小 (T,)，deg/s。
    用相邻帧相对旋转的转角 / dt，中心差分。"""
    rot = R.from_quat(quats_xyzw)
    rel = rot[1:] * rot[:-1].inv()
    w = np.degrees(np.linalg.norm(rel.as_rotvec(), axis=1)) / np.diff(t)
    # 前向差分值放在两帧中点，再插回原网格
    tm = 0.5 * (t[1:] + t[:-1])
    return np.interp(t, tm, w)


def relative_angle(q_a, q_b):
    """两个传感器朝向之间的相对转角（度）。含安装偏差，只适合看同一次佩戴内的变化。"""
    ra, rb = R.from_quat(q_a), R.from_quat(q_b)
    return np.degrees(np.linalg.norm((ra.inv() * rb).as_rotvec(), axis=1))


def imu_yaw(q_xyzw, t_ref, tg):
    """IMU 节段绕世界竖直轴（Unity 的 y 轴）相对参考时刻的转角（度，有正负）。
    【重要-9】骨盆、躯干的转动改用 IMU 计算：骨架里的髋连线、肩连线受相机姿态估计和手臂带动影响，
    在样例数据中肩连线的转速达到 IMU 实测躯干角速度的 5–7 倍，不能代表躯干转动；
    试点数据中一名受试者的骨架骨盆几乎不转（约 13°），而他的骨盆 IMU 峰值角速度是全体最高的。
    符号约定：引拍方向为负。"""
    ok = np.isfinite(q_xyzw).all(1)
    out = np.full(len(tg), np.nan)
    if ok.sum() < 3:
        return out
    rot = R.from_quat(q_xyzw[ok])
    tt = tg[ok]
    i = int(np.clip(np.searchsorted(tt, t_ref), 0, len(tt) - 1))
    rv = (rot * rot[i].inv()).as_rotvec()
    out[ok] = -np.degrees(rv[:, 1])
    return out


def relative_change(qa, qb, t_ref, tg):
    """两个 IMU 之间的相对朝向，相对参考时刻变化了多少度（大小，无正负）。
    不需要传感器到骨骼的标定，所以只能看"变化量"，不能当成解剖学关节角。"""
    out = np.full(len(tg), np.nan)
    if qa is None or qb is None:
        return out
    ok = np.isfinite(qa).all(1) & np.isfinite(qb).all(1)
    if ok.sum() < 3:
        return out
    rel = R.from_quat(qa[ok]).inv() * R.from_quat(qb[ok])
    i = int(np.clip(np.searchsorted(tg[ok], t_ref), 0, ok.sum() - 1))
    out[ok] = np.degrees((rel[i].inv() * rel).magnitude())
    return out
