import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R

from gello_ros.robots.pose_command_guard import (
    PoseCommandLimits,
    limit_pose_command,
    normalized_pose,
)

LIMITS = PoseCommandLimits(
    max_linear_speed_m_per_s=0.1,
    max_angular_speed_rad_per_s=0.5,
    max_tracking_error_m=0.02,
    max_tracking_error_rad=0.1,
)
IDENTITY = np.array([0.0, 0.0, 0.0, 1.0])


def pose(xyz, quat=IDENTITY):
    return np.concatenate((np.asarray(xyz, dtype=float), quat))


def angle_between(a, b):
    return np.linalg.norm((R.from_quat(b[3:]) * R.from_quat(a[3:]).inv()).as_rotvec())


def test_step_is_limited_by_speed():
    measured = pose([0.4, 0.0, 0.3])
    target = pose([0.9, 0.0, 0.3])  # 0.5 m 先
    out = limit_pose_command(target, measured, measured, 0.01, LIMITS)
    assert out[0] == pytest.approx(0.4 + 0.1 * 0.01)


def test_command_is_pulled_back_to_tracking_error():
    measured = pose([0.4, 0.0, 0.3])
    previous = pose([0.45, 0.0, 0.3])  # 計測から 50 mm 先を指していた
    out = limit_pose_command(previous, previous, measured, 0.01, LIMITS)
    assert np.linalg.norm(out[:3] - measured[:3]) == pytest.approx(0.02)


def test_first_command_starts_from_measured_pose():
    measured = pose([0.4, 0.0, 0.3])
    target = pose([0.0, 0.0, 0.0])  # 原点（観測が 0 埋めだった場合など）
    out = limit_pose_command(target, None, measured, 0.0, LIMITS)
    np.testing.assert_allclose(out[:3], measured[:3])


def test_rotation_is_limited():
    measured = pose([0.4, 0.0, 0.3])
    target = pose([0.4, 0.0, 0.3], R.from_euler('z', 1.0).as_quat())
    out = limit_pose_command(target, measured, measured, 0.02, LIMITS)
    assert angle_between(measured, out) == pytest.approx(0.5 * 0.02)
    out = limit_pose_command(target, target, measured, 0.02, LIMITS)
    assert angle_between(measured, out) == pytest.approx(0.1)


def test_negative_dt_does_not_move():
    measured = pose([0.4, 0.0, 0.3])
    out = limit_pose_command(pose([0.5, 0.0, 0.3]), measured, measured, -1.0, LIMITS)
    np.testing.assert_allclose(out[:3], measured[:3])


@pytest.mark.parametrize('bad', [
    pose([np.nan, 0.0, 0.0]),
    pose([0.0, 0.0, 0.0], np.zeros(4)),
    np.zeros(6),
])
def test_invalid_target_is_rejected(bad):
    measured = pose([0.4, 0.0, 0.3])
    assert limit_pose_command(bad, measured, measured, 0.01, LIMITS) is None


def test_invalid_measured_is_rejected():
    assert limit_pose_command(pose([0.4, 0, 0.3]), None, pose([np.inf, 0, 0]), 0.01, LIMITS) is None


def test_quaternion_is_normalized():
    out = normalized_pose(pose([0, 0, 0], np.array([0.0, 0.0, 0.0, 2.0])))
    np.testing.assert_allclose(out[3:], IDENTITY)


@pytest.mark.parametrize('field', ['max_linear_speed_m_per_s', 'max_tracking_error_rad'])
@pytest.mark.parametrize('value', [0.0, -1.0, float('nan')])
def test_limits_must_be_positive(field, value):
    kwargs = dict(max_linear_speed_m_per_s=0.1, max_angular_speed_rad_per_s=0.5,
                  max_tracking_error_m=0.02, max_tracking_error_rad=0.1)
    kwargs[field] = value
    with pytest.raises(ValueError):
        PoseCommandLimits(**kwargs)
