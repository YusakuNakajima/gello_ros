import os

# gello_ros.policy は import 時に DEVICE を読む（ふだんは scripts/policy_config.py が設定する）
os.environ.setdefault("DEVICE", "cpu")

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader

from gello_ros.policy.utils import EpisodicDataset, get_norm_stats, max_episode_length  # noqa: E402


def write_episode(path, steps):
    """save_episode と同じキーを持つ、小さい画像の hdf5 を作る。"""
    rot = np.tile(np.eye(3), (steps, 1, 1))
    with h5py.File(path, "w") as root:
        root.attrs["sim"] = True
        root.attrs["action_alignment"] = "same_step"
        root["/observations/ee_pos"] = np.random.rand(steps, 3)
        root["/observations/ee_rot_matrix"] = rot
        root["/observations/wrench"] = np.random.rand(steps, 6)
        root["/observations/images/bottom_rgb"] = np.zeros((steps, 4, 4, 3), dtype=np.uint8)
        root["/action/ee_pos"] = np.random.rand(steps, 3)
        root["/action/ee_rot_matrix"] = rot


def test_episodes_of_different_lengths_can_be_batched(tmp_path):
    # quit で途中保存した短いエピソード（3 ステップ）と通常のエピソード（6 ステップ）を同じ batch にする
    write_episode(tmp_path / "episode_0.hdf5", 6)
    write_episode(tmp_path / "episode_1.hdf5", 3)
    stats = get_norm_stats(str(tmp_path), 2, include_ft=True)
    max_len = max_episode_length(str(tmp_path), 2)
    assert max_len == 6
    dataset = EpisodicDataset([0, 1], str(tmp_path), ["bottom_rgb"], stats, True, max_len)
    images, obs, ft, actions, is_pad = next(iter(DataLoader(dataset, batch_size=2, shuffle=False)))
    assert actions.shape == (2, 6, 12)
    assert is_pad.shape == (2, 6)
    # 短いエピソードの残りは padding として印が付く
    assert bool(is_pad[1, 3:].all())
    assert torch.isfinite(actions).all()
