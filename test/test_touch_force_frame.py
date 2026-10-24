import numpy as np
from scipy.spatial.transform import Rotation as R

from gello_ros.agents.touch_agent import rotate_vector


def test_sensor_force_is_rotated_into_base():
    # センサが base に対して Z 軸まわりに +90° 回っているとき（FK が返す姿勢）、
    # センサの +X 方向の力は base の +Y 方向になる
    sensor_in_base = R.from_euler('z', np.pi / 2).as_quat()
    force_base = rotate_vector(R.from_quat(sensor_in_base).as_matrix(), np.array([1.0, 0.0, 0.0]))
    np.testing.assert_allclose(force_base, [0.0, 1.0, 0.0], atol=1e-12)
