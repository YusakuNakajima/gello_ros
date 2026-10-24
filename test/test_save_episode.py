from types import SimpleNamespace

import numpy as np
import pytest

from gello_ros.data_utils import save_episode as module


def fake_node(tmp_path, camera_names):
    values = {"camera_names": SimpleNamespace(string_array_value=camera_names),
              "camera_width": SimpleNamespace(integer_value=4), "camera_height": SimpleNamespace(integer_value=4),
              "wrench_dim": SimpleNamespace(integer_value=6),
              "save_episode_dir": SimpleNamespace(string_value=str(tmp_path)),
              "task_name": SimpleNamespace(string_value="t"), "use_FT_sensor": SimpleNamespace(bool_value=True),
              "hardware_label": SimpleNamespace(string_value="fake")}
    return SimpleNamespace(
        get_parameter=lambda name: SimpleNamespace(get_parameter_value=lambda: values[name]),
        get_logger=lambda: SimpleNamespace(info=lambda *a, **k: None, error=lambda *a, **k: None))


def step(image_shape):
    obs = {"joint_positions": np.zeros(7), "joint_velocities": np.zeros(7), "joint_torques": np.zeros(7),
           "ee_pos": np.zeros(3), "ee_quat": np.array([0, 0, 0, 1.0]), "ee_rot_matrix": np.eye(3),
           "ee_euler": np.zeros(3), "ee_wrench": np.zeros(6), "bottom_rgb": np.zeros(image_shape, np.uint8)}
    action = {"joint_positions": np.zeros(7), "ee_pos": np.zeros(3), "ee_quat": np.array([0, 0, 0, 1.0]),
              "ee_rot_matrix": np.eye(3), "ee_euler": np.zeros(3)}
    return obs, action


def files(tmp_path):
    return sorted(p.name for p in tmp_path.rglob("*") if p.is_file())


def test_saved_episode_appears_only_when_complete(tmp_path):
    obs, action = step((4, 4, 3))
    module.save_episode(fake_node(tmp_path, ["bottom"]), 0, [obs, obs], [action, action])
    assert files(tmp_path) == ["episode_0.hdf5"]


def test_failed_save_leaves_no_episode_file(tmp_path):
    # 画像サイズが設定と違う → 書き込みの途中で失敗する。完成したように見えるファイルも一時ファイルも残さない
    obs, action = step((8, 8, 3))
    with pytest.raises(Exception):
        module.save_episode(fake_node(tmp_path, ["bottom"]), 0, [obs], [action])
    assert files(tmp_path) == []
