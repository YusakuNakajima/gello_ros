import os
import h5py
import torch
import numpy as np
from torch.utils.data import DataLoader

from gello_ros.policy.policy import ACTPolicy, CompACTPolicy, CNNMLPPolicy

import IPython

e = IPython.embed


class EpisodicDataset(torch.utils.data.Dataset):
    def __init__(self, episode_ids, dataset_dir, camera_names, norm_stats, include_ft, max_episode_len=None):
        super(EpisodicDataset, self).__init__()
        # action と padding mask をこの長さに揃える（quit で途中保存したエピソードと混ぜても batch にできるよう、
        # データセット中で最も長いエピソードの長さ）。None なら各エピソード自身の長さ（以前の動作）
        self.max_episode_len = max_episode_len
        self.episode_ids = episode_ids
        self.dataset_dir = dataset_dir
        self.camera_names = camera_names
        self.norm_stats = norm_stats
        self.include_ft = include_ft
        self.is_sim = None

    def __len__(self):
        return len(self.episode_ids)

    def __getitem__(self, index):
        sample_full_episode = False  # hardcode
        episode_id = self.episode_ids[index]
        dataset_path = os.path.join(self.dataset_dir, f"episode_{episode_id}.hdf5")

        with h5py.File(dataset_path, "r") as root:
            is_sim = root.attrs["sim"]
            # action と観測の index の対応。新しいファイルは action_alignment、古いファイルは sim で判定する
            same_step = (root.attrs["action_alignment"] == "same_step") if "action_alignment" in root.attrs else bool(is_sim)
            episode_len = root["/action/ee_pos"].shape[0]

            if sample_full_episode:
                start_ts = 0
            else:
                start_ts = np.random.choice(episode_len)

            # Build observation from /observations/ee_pos and /observations/ee_rot_matrix
            obs_ee_pos = root["/observations/ee_pos"][start_ts]
            obs_ee_rot = root["/observations/ee_rot_matrix"][start_ts]
            observation = np.concatenate((obs_ee_pos, obs_ee_rot.flatten()))

            # Collect image data as before
            image_dict = {}
            for cam_name in self.camera_names:
                image_dict[cam_name] = root[f"/observations/images/{cam_name}"][start_ts]

            # If including force-torque data, append it to the observation
            if self.include_ft:
                ft = root["/observations/wrench"][start_ts]
                
            # Build action by concatenating /action/ee_pos and flattened /observations/ee_rot_matrix over timesteps
            if same_step:
                act_ee_pos = root["/action/ee_pos"][start_ts:]
                act_rot = root["/action/ee_rot_matrix"][start_ts:]
                action_len = episode_len - start_ts
            else:
                idx = max(0, start_ts - 1)
                act_ee_pos = root["/action/ee_pos"][idx:]
                act_rot = root["/action/ee_rot_matrix"][idx:]
                action_len = episode_len - idx

        self.is_sim = is_sim
        # Concatenate along the second axis (flatten rot matrix for each timestep)
        action = np.concatenate((act_ee_pos, act_rot.reshape(act_rot.shape[0], -1)), axis=1)

        # Pad action to a common length so episodes of different lengths can be batched
        padded_len = self.max_episode_len if self.max_episode_len is not None else episode_len
        padded_action = np.zeros((padded_len, action.shape[1]), dtype=np.float32)
        padded_action[:action_len] = action

        is_pad = np.zeros(padded_len)
        is_pad[action_len:] = 1

        # Stack images from different cameras
        all_cam_images = [image_dict[cam_name] for cam_name in self.camera_names]
        all_cam_images = np.stack(all_cam_images, axis=0)
        image_data = torch.from_numpy(all_cam_images)
        image_data = torch.einsum("k h w c -> k c h w", image_data)
        image_data = image_data / 255.0

        # Normalize action data
        action_data = (padded_action - self.norm_stats["action_mean"]) / self.norm_stats["action_std"]
        action_data = torch.from_numpy(action_data).float()
        is_pad = torch.from_numpy(is_pad).bool()
        
        if self.include_ft:
            ft_data = torch.from_numpy(ft).float()
            ft_data = (ft_data - self.norm_stats["ft_mean"]) / self.norm_stats["ft_std"]
        else:
            ft_data = []


        # Return tuple: image data, observation, action, and padding mask
        return image_data, torch.from_numpy(observation).float(), ft_data, action_data, is_pad


def get_norm_stats(dataset_dir, num_episodes, include_ft):
    """
    各エピソードのHDF5ファイルから、観測とアクションのデータを読み込み、
    各々の平均と標準偏差を計算する。
    観測は、/observations/ee_pos と /observations/ee_rot_matrix（flattenして連結）として、
    アクションは、/action/ee_pos と /observations/ee_rot_matrix（flattenして連結）として算出する。
    """
    all_obs_data = []
    all_ft_data = []
    all_action_data = []
    
    for episode_idx in range(num_episodes):
        dataset_path = os.path.join(dataset_dir, f"episode_{episode_idx}.hdf5")
        with h5py.File(dataset_path, "r") as root:
            # 読み込み: 観測側
            obs_ee_pos = root["/observations/ee_pos"][()]         # shape: (T, 3)
            obs_rot = root["/observations/ee_rot_matrix"][()]       # shape: (T, 3, 3)
            # flattenして連結: (T, 3) と (T, 9) -> (T, 12)
            obs_data = np.concatenate((obs_ee_pos, obs_rot.reshape(obs_rot.shape[0], -1)), axis=1)
            
            # 読み込み: アクション側
            act_ee_pos = root["/action/ee_pos"][()]                # shape: (T, 3)
            # 注意：アクションの回転については、観測側の回転行列をそのまま利用する（指示に合わせる）
            act_rot = root["/action/ee_rot_matrix"][()]      
            act_data = np.concatenate((act_ee_pos, act_rot.reshape(act_rot.shape[0], -1)), axis=1)
            
            # 読み込み: 力覚センサーデータ（存在する場合のみ）
            if include_ft:
                ft = root["/observations/wrench"][()]                   # shape: (T, 6)
        
        all_obs_data.append(torch.from_numpy(obs_data))
        all_action_data.append(torch.from_numpy(act_data))
        if include_ft:
            all_ft_data.append(torch.from_numpy(ft))
    
    # エピソード間で全てのタイムステップを連結
    all_obs_data = torch.cat(all_obs_data, dim=0)
    all_action_data = torch.cat(all_action_data, dim=0)
    
    # normalize action data
    action_mean = all_action_data.mean(dim=0, keepdim=True)
    action_std = all_action_data.std(dim=0, keepdim=True)
    action_std = torch.clip(action_std, 1e-2, float("inf"))
    
    # normalize observation data
    obs_mean = all_obs_data.mean(dim=0, keepdim=True)
    obs_std = all_obs_data.std(dim=0, keepdim=True)
    obs_std = torch.clip(obs_std, 1e-2, float("inf"))
    
    stats = {
        "action_mean": action_mean.numpy().squeeze(),
        "action_std": action_std.numpy().squeeze(),
        "obs_mean": obs_mean.numpy().squeeze(),
        "obs_std": obs_std.numpy().squeeze(),
        "example_obs": obs_data,  # 最終エピソードのサンプル（任意）
    }
    
    if include_ft:
        all_ft_data = torch.cat(all_ft_data, dim=0)
        ft_mean = all_ft_data.mean(dim=0, keepdim=True)
        ft_std = all_ft_data.std(dim=0, keepdim=True)
        ft_std = torch.clip(ft_std, 1e-2, float("inf"))
        stats["ft_mean"] = ft_mean.numpy().squeeze()
        stats["ft_std"] = ft_std.numpy().squeeze()
    
    return stats


def max_episode_length(dataset_dir, num_episodes):
    """episode_0..num_episodes-1 のうち、最も長いエピソードのステップ数。"""
    lengths = []
    for episode_idx in range(num_episodes):
        with h5py.File(os.path.join(dataset_dir, f"episode_{episode_idx}.hdf5"), "r") as root:
            lengths.append(root["/action/ee_pos"].shape[0])
    return max(lengths)


def load_data(
    dataset_dir, num_episodes, camera_names, include_ft, batch_size_train, batch_size_val
):
    print(f"\nData from: {dataset_dir}\n")
    # obtain train test split
    train_ratio = 0.8
    shuffled_indices = np.random.permutation(num_episodes)
    train_indices = shuffled_indices[: int(train_ratio * num_episodes)]
    val_indices = shuffled_indices[int(train_ratio * num_episodes) :]

    # obtain normalization stats for qpos and action
    norm_stats = get_norm_stats(dataset_dir, num_episodes, include_ft)
    max_episode_len = max_episode_length(dataset_dir, num_episodes)

    # construct dataset and dataloader
    train_dataset = EpisodicDataset(
        train_indices, dataset_dir, camera_names, norm_stats, include_ft, max_episode_len
    )
    val_dataset = EpisodicDataset(
        val_indices, dataset_dir, camera_names, norm_stats, include_ft, max_episode_len
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size_train,
        shuffle=True,
        pin_memory=True,
        num_workers=1,
        prefetch_factor=1,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size_val,
        shuffle=True,
        pin_memory=True,
        num_workers=1,
        prefetch_factor=1,
    )

    return train_dataloader, val_dataloader, norm_stats, train_dataset.is_sim


def make_policy(policy_class, policy_config):
    if policy_class == "ACT":
        policy = ACTPolicy(policy_config)
    elif policy_class == "CompACT":
        policy = CompACTPolicy(policy_config)
    elif policy_class == "CNNMLP":
        policy = CNNMLPPolicy(policy_config)
    else:
        raise ValueError(f"Unknown policy class: {policy_class}")
    return policy


def make_optimizer(policy_class, policy):
    if policy_class == "ACT":
        optimizer = policy.configure_optimizers()
    elif policy_class == "CompACT":
        optimizer = policy.configure_optimizers()
    elif policy_class == "CNNMLP":
        optimizer = policy.configure_optimizers()
    else:
        raise ValueError(f"Unknown policy class: {policy_class}")
    return optimizer


### env utils


def sample_box_pose():
    x_range = [0.0, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    cube_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    cube_quat = np.array([1, 0, 0, 0])
    return np.concatenate([cube_position, cube_quat])


def sample_insertion_pose():
    # Peg
    x_range = [0.1, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    peg_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    peg_quat = np.array([1, 0, 0, 0])
    peg_pose = np.concatenate([peg_position, peg_quat])

    # Socket
    x_range = [-0.2, -0.1]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    socket_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    socket_quat = np.array([1, 0, 0, 0])
    socket_pose = np.concatenate([socket_position, socket_quat])

    return peg_pose, socket_pose


### helper functions


def get_image(images, camera_names, device="cpu"):
    curr_images = []
    for cam_name in camera_names:
        curr_image = images[cam_name].transpose(2, 0, 1)
        curr_images.append(curr_image)
    curr_image = np.stack(curr_images, axis=0)
    curr_image = torch.from_numpy(curr_image / 255.0).float().to(device).unsqueeze(0)
    return curr_image


def compute_dict_mean(epoch_dicts):
    result = {k: None for k in epoch_dicts[0]}
    num_items = len(epoch_dicts)
    for k in result:
        value_sum = 0
        for epoch_dict in epoch_dicts:
            value_sum += epoch_dict[k]
        result[k] = value_sum / num_items
    return result


def detach_dict(d):
    new_d = dict()
    for k, v in d.items():
        new_d[k] = v.detach()
    return new_d


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


def pos2pwm(pos: np.ndarray) -> np.ndarray:
    """
    :param pos: numpy array of joint positions in range [-pi, pi]
    :return: numpy array of pwm values in range [0, 4096]
    """
    return (pos / 3.14 + 1.0) * 2048


def pwm2pos(pwm: np.ndarray) -> np.ndarray:
    """
    :param pwm: numpy array of pwm values in range [0, 4096]
    :return: numpy array of joint positions in range [-pi, pi]
    """
    return (pwm / 2048 - 1) * 3.14


def pwm2vel(pwm: np.ndarray) -> np.ndarray:
    """
    :param pwm: numpy array of pwm/s joint velocities
    :return: numpy array of rad/s joint velocities
    """
    return pwm * 3.14 / 2048


def vel2pwm(vel: np.ndarray) -> np.ndarray:
    """
    :param vel: numpy array of rad/s joint velocities
    :return: numpy array of pwm/s joint velocities
    """
    return vel * 2048 / 3.14


def pwm2norm(x: np.ndarray) -> np.ndarray:
    """
    :param x: numpy array of pwm values in range [0, 4096]
    :return: numpy array of values in range [0, 1]
    """
    return x / 4096


def norm2pwm(x: np.ndarray) -> np.ndarray:
    """
    :param x: numpy array of values in range [0, 1]
    :return: numpy array of pwm values in range [0, 4096]
    """
    return x * 4096
