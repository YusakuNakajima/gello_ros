from types import SimpleNamespace

import numpy as np
from builtin_interfaces.msg import Time
from scipy.spatial.transform import Rotation as R

from gello_ros.agents.touch_agent import TouchAgent


class FakeClock:
    def now(self):
        return SimpleNamespace(to_msg=Time)


def fake_agent(sensor_quat):
    """transform_wrench_kdl が使う属性だけを持つ、ROS ノードを作らない TouchAgent の代わり。"""
    agent = SimpleNamespace(
        kdl_helper=SimpleNamespace(forward_kinematics=lambda q: [0.0, 0.0, 0.0, *sensor_quat]),
        wrench_static_transform=np.eye(4),  # base -> touch_force を恒等にして、base 座標の力がそのまま出るようにする
        force_scale_to_touch=1.0,
        touch_max_force=10.0,
        robot_base_frame='base',
    )
    agent.get_clock = FakeClock
    agent.get_logger = lambda: SimpleNamespace(warn=print, error=print, debug=print)
    agent._create_zero_wrench_messages = lambda: (None, None)
    return agent


def test_sensor_force_is_rotated_into_base():
    # センサが base に対して Z 軸まわりに +90° 回っているとき（FK が返す姿勢）、
    # センサの +X 方向の力は base の +Y 方向になる（転置すると −Y になって失敗する）
    agent = fake_agent(R.from_euler('z', np.pi / 2).as_quat())
    vis, feedback = TouchAgent.transform_wrench_kdl(agent, np.array([1.0, 0, 0, 0, 0, 0]), np.zeros(7))
    np.testing.assert_allclose(
        [vis.wrench.force.x, vis.wrench.force.y, vis.wrench.force.z], [0.0, 1.0, 0.0], atol=1e-9)
    np.testing.assert_allclose([feedback.force.x, feedback.force.y, feedback.force.z], [0.0, 1.0, 0.0], atol=1e-9)
