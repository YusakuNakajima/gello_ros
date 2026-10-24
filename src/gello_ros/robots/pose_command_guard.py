"""手先の目標姿勢を、速度と追従誤差の上限で制限する（ROS に依存しない純粋な計算）。

姿勢は [x, y, z, qx, qy, qz, qw]（位置は m、quaternion は scipy と同じ xyzw 順）。
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.spatial.transform import Rotation as R


@dataclass(frozen=True)
class PoseCommandLimits:
    # 指令の目標が 1 秒あたりに動ける量の上限（並進は m/s、回転は rad/s）
    max_linear_speed_m_per_s: float
    max_angular_speed_rad_per_s: float
    # 目標と計測姿勢の差の上限。目標はこの範囲に引き戻す（m、rad）
    max_tracking_error_m: float
    max_tracking_error_rad: float

    def __post_init__(self):
        for name in ('max_linear_speed_m_per_s', 'max_angular_speed_rad_per_s',
                     'max_tracking_error_m', 'max_tracking_error_rad'):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f'{name} must be a finite value > 0, got {value}')


def normalized_pose(pose) -> Optional[np.ndarray]:
    """有限の 7 要素で quaternion のノルムが 0 でなければ、quaternion を正規化して返す。それ以外は None。"""
    pose = np.asarray(pose, dtype=float).reshape(-1)
    if pose.shape != (7,) or not np.all(np.isfinite(pose)):
        return None
    norm = np.linalg.norm(pose[3:])
    if norm < 1e-6:
        return None
    return np.concatenate((pose[:3], pose[3:] / norm))


def limit_toward(origin, target, max_distance_m, max_angle_rad) -> np.ndarray:
    """origin から target へ、並進 max_distance_m、回転 max_angle_rad までだけ進めた姿勢を返す。"""
    position = target[:3]
    delta = target[:3] - origin[:3]
    distance = np.linalg.norm(delta)
    if distance > max_distance_m:
        position = origin[:3] + delta * (max_distance_m / distance)

    origin_rot = R.from_quat(origin[3:])
    relative = R.from_quat(target[3:]) * origin_rot.inv()
    rotvec = relative.as_rotvec()
    angle = np.linalg.norm(rotvec)
    rotation = R.from_quat(target[3:])
    if angle > max_angle_rad:
        rotation = R.from_rotvec(rotvec * (max_angle_rad / angle)) * origin_rot
    return np.concatenate((position, rotation.as_quat()))


def limit_pose_command(target, previous_command, measured, dt_s, limits: PoseCommandLimits):
    """送ってよい目標姿勢を返す。

    1. 前回送った目標（なければ計測姿勢）から、速度の上限 × dt_s までだけ target へ進める
    2. その結果を、計測姿勢から追従誤差の上限の範囲に引き戻す

    2 は 1 より優先する。ロボットが前回の目標から追従誤差の上限以上ずれた（外力で押された、
    追従できなかった）ときは、目標が 1 周期で速度の上限より大きく計測姿勢の側へ動く。
    impedance 制御のばねの力は「目標 − 計測姿勢」に比例するので、引き戻すほうが力は小さくなる。
    前回の目標を保つ・送るのをやめる（controller は前回の目標を保つ）と、上限より大きい差が残る。

    target・measured が無効（NaN・ノルム 0 の quaternion など）なら None を返す（送らない）。
    """
    target = normalized_pose(target)
    measured = normalized_pose(measured)
    if target is None or measured is None:
        return None
    previous = normalized_pose(previous_command) if previous_command is not None else None
    if previous is None:
        previous = measured
    if not np.isfinite(dt_s) or dt_s < 0.0:
        dt_s = 0.0

    stepped = limit_toward(
        previous, target,
        limits.max_linear_speed_m_per_s * dt_s,
        limits.max_angular_speed_rad_per_s * dt_s)
    return limit_toward(
        measured, stepped, limits.max_tracking_error_m, limits.max_tracking_error_rad)
