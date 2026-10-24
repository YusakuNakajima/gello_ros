import os

# fallback to cpu if mps is not available for specific operations
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import torch

# data directory
DATA_DIR = "data/"

# device
device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
# if torch.backends.mps.is_available(): device = 'mps'
os.environ["DEVICE"] = device


# データの置き場所。onolab_ros2 ではリポジトリ直下の gello_data/（git 管理外）に置く。
# 環境変数で上書きできる（GELLO_DATASET_DIR: 学習に使うエピソードのフォルダ、
# GELLO_CHECKPOINT_DIR: チェックポイントの保存先、GELLO_NUM_EPOCHS: 学習エポック数）
GELLO_DATA_ROOT = os.environ.get("GELLO_DATA_ROOT", "/home/ros/onolab_ros2/gello_data")

# task config (you can add new tasks)
TASK_CONFIG = {
    "task_name": "grinding_test",
    "dataset_dir": os.environ.get(
        "GELLO_DATASET_DIR",
        os.path.join(GELLO_DATA_ROOT, "episode_data", "20250901_151810_grinding_test"),
    ),
    "state_dim": 12,
    "action_dim": 12,
    # hdf5 の /observations/images/ 以下のキー。save_episode は「<common.yaml の camera_names>_rgb」で保存する
    "camera_names": ["bottom_rgb"],
}


# policy config
POLICY_CONFIG = {
    "lr": 1e-5,
    "device": device,
    "num_queries": 100,  # get the last 100 actions from the policy
    "kl_weight": 10,
    "hidden_dim": 512,
    "dim_feedforward": 3200,
    "lr_backbone": 1e-5,
    "backbone": "resnet18",
    "enc_layers": 4,
    "dec_layers": 7,
    "nheads": 8,
    "camera_names": ["bottom_rgb"],
    "policy_class": "CompACT",
    "temporal_agg": False,
    "include_ft": True,
    "ft_as_obs": False,
}

# training config
TRAIN_CONFIG = {
    "seed": 42,
    "num_epochs": int(os.environ.get("GELLO_NUM_EPOCHS", "2000")),  # number of training epochs
    "batch_size_val": 8,
    "batch_size_train": 8,
    "checkpoint_dir": os.environ.get(
        "GELLO_CHECKPOINT_DIR", os.path.join(GELLO_DATA_ROOT, "checkpoints")
    ),
}
