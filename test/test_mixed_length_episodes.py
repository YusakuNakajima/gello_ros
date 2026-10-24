import os

# gello_ros.policy は import 時に DEVICE を読む（ふだんは scripts/policy_config.py が設定する）
os.environ.setdefault("DEVICE", "cpu")

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader

from gello_ros.policy.utils import (  # noqa: E402
    EpisodicDataset, episode_ids, get_norm_stats, max_episode_length, normalize_ft, normalize_observation,
)


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
    ids = episode_ids(str(tmp_path))
    assert ids == [0, 1]
    stats = get_norm_stats(str(tmp_path), ids, include_ft=True)
    max_len = max_episode_length(str(tmp_path), ids)
    assert max_len == 6
    dataset = EpisodicDataset([0, 1], str(tmp_path), ["bottom_rgb"], stats, True, max_len)
    images, obs, ft, actions, is_pad = next(iter(DataLoader(dataset, batch_size=2, shuffle=False)))
    assert actions.shape == (2, 6, 12)
    assert is_pad.shape == (2, 6)
    # 短いエピソードの残りは padding として印が付く
    assert bool(is_pad[1, 3:].all())
    assert torch.isfinite(actions).all()


def test_training_inputs_use_the_same_normalization_as_inference(tmp_path):
    # 学習で方策に渡す姿勢と力は、推論（CompACTAgent）と同じ正規化をしたもの
    write_episode(tmp_path / "episode_0.hdf5", 4)
    stats = get_norm_stats(str(tmp_path), [0], include_ft=True)
    dataset = EpisodicDataset([0], str(tmp_path), ["bottom_rgb"], stats, True, 4)
    with h5py.File(tmp_path / "episode_0.hdf5", "r") as root:
        raw_obs = [np.concatenate((root["/observations/ee_pos"][t], root["/observations/ee_rot_matrix"][t].flatten()))
                   for t in range(4)]
        raw_ft = [root["/observations/wrench"][t] for t in range(4)]
    _, obs, ft, _, _ = dataset[0]
    # 取り出した時刻はランダムなので、どれか 1 つと一致すればよい
    assert any(np.allclose(obs.numpy(), normalize_observation(stats, o), atol=1e-5) for o in raw_obs)
    assert any(np.allclose(ft.numpy(), normalize_ft(stats, f), atol=1e-5) for f in raw_ft)


def test_agents_use_the_shared_normalization():
    import inspect
    from gello_ros.agents import act_agent, comp_act_agent
    assert comp_act_agent.normalize_observation is normalize_observation
    assert act_agent.normalize_observation is normalize_observation
    assert "normalize_ft(self.stats" in inspect.getsource(comp_act_agent.CompACTAgent.act)


def test_unfinished_files_are_not_counted_as_episodes(tmp_path):
    write_episode(tmp_path / "episode_0.hdf5", 3)
    (tmp_path / ".episode_1.hdf5.tmp").write_bytes(b"")
    (tmp_path / "notes.txt").write_text("x")
    assert episode_ids(str(tmp_path)) == [0]
