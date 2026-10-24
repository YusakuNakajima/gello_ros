"""止める処理の状態遷移（Touch を離す → 停止中）を、ROS ノードを作らずに確かめる。"""
from types import SimpleNamespace

import numpy as np
from builtin_interfaces.msg import Time

from gello_ros.agents.touch_agent import TouchAgent
from gello_ros.env import RobotEnv
from gello_ros.freshness import monotonic_ns

IDENTITY_QUAT = np.array([0.0, 0.0, 0.0, 1.0])


class FakeClock:
    def now(self):
        return SimpleNamespace(nanoseconds=0, to_msg=Time)


def fake_touch_agent():
    logger = SimpleNamespace(debug=lambda *a, **k: None, error=lambda *a, **k: None,
                             warn=lambda *a, **k: None, info=lambda *a, **k: None)
    agent = SimpleNamespace(
        _touch_current_pose=np.array([0.1, 0.0, 0.1, *IDENTITY_QUAT]),
        _touch_received_ns=monotonic_ns(), _touch_stamp_ns=0, touch_timeout_s=10.0,
        _is_teleop_active=False, _was_teleop_active=False, _needs_rearm=False, _is_z_lock_active=False,
        pose_static_transform=np.eye(4), teleop_mode="unilateral",
        _robot_current_pose=None, _pose_initialized=False, _robot_start_pose=None,
        _robot_start_pose_with_touch=None, _last_target_pose=None,
        z_down_quat=IDENTITY_QUAT, robot_base_frame="base",
        touch_debug_action_pose_pub=SimpleNamespace(publish=lambda m: None),
    )
    agent.get_clock = FakeClock
    agent.get_logger = lambda: logger
    agent.calculate_pose_difference = lambda *a: TouchAgent.calculate_pose_difference(agent, *a)
    agent.require_rearm = lambda: TouchAgent.require_rearm(agent)
    return agent


def obs_at(x):
    return {"ee_pos": np.array([x, 0.0, 0.3]), "ee_quat": IDENTITY_QUAT,
            "joint_positions": np.zeros(7), "stale_sensors": []}


def test_release_holds_once_then_sends_nothing():
    agent = fake_touch_agent()
    # 停止中: 何も送らない
    assert not TouchAgent.act(agent, obs_at(0.40))["send_command"]
    # 白ボタンを押す → 送る
    agent._is_teleop_active = True
    assert TouchAgent.act(agent, obs_at(0.40))["send_command"]
    # 離す → その周期だけ計測姿勢で止める
    agent._is_teleop_active = False
    action = TouchAgent.act(agent, obs_at(0.41))
    assert action["hold_measured"] and action["send_command"]
    # 以後は、ロボットが動いても（計測 0.43）古い姿勢へ戻す指令を送らない
    for x in (0.42, 0.43):
        action = TouchAgent.act(agent, obs_at(x))
        assert not action["send_command"]
        assert not action["hold_measured"]


def test_press_again_starts_from_the_current_measured_pose():
    agent = fake_touch_agent()
    agent._is_teleop_active = True
    TouchAgent.act(agent, obs_at(0.40))
    agent._is_teleop_active = False
    TouchAgent.act(agent, obs_at(0.40))
    # 停止中にロボットが 0.43 へ動いたあとで押し直すと、0.43 から始まる（0.40 へは戻さない）
    agent._is_teleop_active = True
    action = TouchAgent.act(agent, obs_at(0.43))
    np.testing.assert_allclose(action["ee_pos"], [0.43, 0.0, 0.3])


class FakeRobot:
    def __init__(self):
        self.commands, self.holds, self.last_sent_pose = [], 0, np.zeros(7)

    def command_pose(self, pose):
        self.commands.append(pose)
        self.last_sent_pose = pose

    def hold_measured_pose(self):
        self.holds += 1
        return True


def step(robot, action):
    env = SimpleNamespace(_robot=robot, _control_mode="cartesian",
                          _rate=SimpleNamespace(sleep=lambda: None), get_obs=lambda: {})
    return RobotEnv.step(env, action)


def test_env_step_sends_nothing_when_told_not_to():
    robot = FakeRobot()
    step(robot, {"ee_pos": np.zeros(3), "ee_quat": IDENTITY_QUAT, "send_command": False})
    assert robot.commands == [] and robot.holds == 0
    assert robot.last_sent_pose is None  # 記録では NaN になる


def test_env_step_hold_bypasses_the_pose_command():
    robot = FakeRobot()
    step(robot, {"ee_pos": np.zeros(3), "ee_quat": IDENTITY_QUAT, "hold_measured": True})
    assert robot.holds == 1 and robot.commands == []


def fake_impedance_robot(samples):
    """hold_measured_pose() が使う属性だけを持つ代わり。samples は spin のたびに届く (stamp_ns, x)。"""
    from gello_ros.robots.ros_cartesian_impedance_control_robot import CartesianImpedanceControlRobot as Robot
    published = []
    robot = SimpleNamespace(
        joint_state_timeout_ns=100_000_000, _joint_positions=np.zeros(7),
        _joint_state_received_ns=None, _joint_state_stamp_ns=None, _x=None,
        _last_command_pose=None, _last_command_ns=None, last_sent_pose=None, hold_pending=False,
    )
    ros_now = {"ns": 10_000_000_000}
    robot.get_clock = lambda: SimpleNamespace(now=lambda: SimpleNamespace(nanoseconds=ros_now["ns"]))
    robot.get_logger = lambda: SimpleNamespace(error=lambda *a, **k: None)
    robot._measured_pose = lambda now_ns: np.array([robot._x, 0, 0.3, *IDENTITY_QUAT])
    robot._publish_pose = lambda pose: published.append(np.array(pose))

    def deliver(*_args, **_kwargs):
        if samples:
            stamp_ns, x = samples.pop(0)
            robot._joint_state_received_ns = monotonic_ns()
            robot._joint_state_stamp_ns = stamp_ns
            robot._x = x
    return Robot, robot, published, deliver, ros_now


def test_hold_uses_a_sample_measured_after_the_stop(monkeypatch):
    import gello_ros.robots.ros_cartesian_impedance_control_robot as mod
    # 1 つ目: 停止後に届いたが 80 ms 前に計測（x=0.40）→ 使わない。2 つ目: 停止後の計測（x=0.43）→ 使う
    Robot, robot, published, deliver, _ = fake_impedance_robot([(9_920_000_000, 0.40), (10_000_000_001, 0.43)])
    monkeypatch.setattr(mod.rclpy, "spin_once", deliver)
    monkeypatch.setattr(mod.rclpy, "ok", lambda: True)
    assert Robot.hold_measured_pose(robot)
    assert len(published) == 1
    np.testing.assert_allclose(published[0][:3], [0.43, 0, 0.3])
    np.testing.assert_allclose(robot._last_command_pose[:3], [0.43, 0, 0.3])


def test_hold_fails_without_a_new_stamped_sample(monkeypatch):
    import gello_ros.robots.ros_cartesian_impedance_control_robot as mod
    # stamp のないサンプルしか来ない → 送らずに False（controller は最後の目標を保つ）
    Robot, robot, published, deliver, _ = fake_impedance_robot([(0, 0.43)])
    monkeypatch.setattr(mod.rclpy, "spin_once", deliver)
    monkeypatch.setattr(mod.rclpy, "ok", lambda: True)
    assert not Robot.hold_measured_pose(robot)
    assert published == []
    # 止められなかったことを覚えておき、あとで止め直す
    assert robot.hold_pending


def test_env_step_retries_a_pending_hold_while_idle():
    robot = FakeRobot()
    robot.hold_pending = True
    step(robot, {"ee_pos": np.zeros(3), "ee_quat": IDENTITY_QUAT, "send_command": False})
    assert robot.holds == 1 and robot.commands == []


def test_stop_zeroes_touch_force_before_holding():
    import importlib.util
    import os
    import sys
    # scripts/run_agent_component.py の AgentNode._stop_and_hold を、ノードを作らずに呼ぶ
    script = os.path.join(os.path.dirname(__file__), "..", "scripts", "run_agent_component.py")
    sys.path.insert(0, os.path.dirname(script))
    spec = importlib.util.spec_from_file_location("run_agent_component", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    calls = []
    node = SimpleNamespace(
        agent=SimpleNamespace(zero_force=lambda: calls.append("zero_force"),
                              require_rearm=lambda: calls.append("require_rearm")),
        robot=SimpleNamespace(hold_measured_pose=lambda: calls.append("hold") or True),
        get_logger=lambda: SimpleNamespace(error=lambda *a, **k: None),
    )
    assert module.AgentNode._stop_and_hold(node)
    assert calls == ["zero_force", "require_rearm", "hold"]


def load_runner():
    import importlib.util
    import os
    import sys
    script = os.path.join(os.path.dirname(__file__), "..", "scripts", "run_agent_component.py")
    sys.path.insert(0, os.path.dirname(script))
    spec = importlib.util.spec_from_file_location("run_agent_component", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cleanup_zeroes_force_first_and_continues_after_a_failing_hold(monkeypatch):
    module = load_runner()
    monkeypatch.setattr(module.rclpy, "ok", lambda: True)
    calls = []

    def failing_hold():
        calls.append("hold")
        raise RuntimeError("hold failed")
    node = SimpleNamespace(
        agent=SimpleNamespace(cleanup=lambda: calls.append("agent.cleanup")),
        robot=SimpleNamespace(hold_measured_pose=failing_hold),
        env=SimpleNamespace(destroy_node=lambda: calls.append("env.destroy")),
        get_logger=lambda: SimpleNamespace(info=lambda *a, **k: None, error=lambda *a, **k: None),
    )
    module.AgentNode.cleanup(node)
    assert calls == ["agent.cleanup", "hold", "env.destroy"]


def test_stale_touch_sends_zero_feedback_even_if_robot_sensors_are_fresh():
    agent = fake_touch_agent()
    agent.teleop_mode = "bilateral"
    agent._is_teleop_active = True
    agent._touch_received_ns = monotonic_ns() - 1_000_000_000  # Touch の姿勢が 1 s 途切れている
    agent.touch_timeout_s = 0.1
    from geometry_msgs.msg import WrenchStamped
    from touch_msgs.msg import TouchFeedback
    zero = (WrenchStamped(), TouchFeedback())
    agent._create_zero_wrench_messages = lambda: zero
    agent.transform_wrench_kdl = lambda *a: (_ for _ in ()).throw(AssertionError("must not send real force"))
    sent = []
    agent.force_feedback_pub = SimpleNamespace(publish=sent.append)
    agent.force_feedback_vis_pub = SimpleNamespace(publish=lambda m: None)
    obs = dict(obs_at(0.4), ee_wrench=np.array([5.0, 0, 0, 0, 0, 0]))
    TouchAgent.act(agent, obs)
    assert len(sent) == 1 and sent[0] is zero[1]


def test_startup_motion_aborts_after_a_shutdown_signal():
    import pytest
    module = load_runner()
    module.SHUTDOWN_SIGNALLED.clear()
    module.abort_if_shutdown_signalled()  # まだ要求なし → 何もしない
    module.SHUTDOWN_SIGNALLED.set()
    try:
        with pytest.raises(RuntimeError):
            module.abort_if_shutdown_signalled()
    finally:
        module.SHUTDOWN_SIGNALLED.clear()
