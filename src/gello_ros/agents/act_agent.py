from typing import Any, Dict
import numpy as np
import os
import pickle
import rclpy
from rclpy.node import Node
import torch
from gello_ros.agents.agent import Agent
from gello_ros.policy.utils import normalize_ft, normalize_observation
from scipy.spatial.transform import Rotation as R

class ACTAgent(Agent, Node):
    def __init__(
        self,
        policy,
        camera_names: list,
        train_cfg: dict,
        policy_config: dict,
        task_cfg: dict,
        device: str,
    ):
        # Initialize ROS2 node
        Node.__init__(self, 'act_agent')
        """
        Initializes the ACTAgent with policy, camera names, and configurations.

        Args:
            policy: The policy model to be used for action prediction.
            camera_names (list): List of camera names.
            train_cfg (dict): Configuration for training.
            policy_config (dict): Configuration for the policy.
            task_cfg (dict): Task-specific configurations.
            device (str): Device to run the model on (e.g., 'cpu' or 'cuda').
        """
        self.policy = policy
        self.camera_names = camera_names
        self.device = device

        # Declare and get ROS2 parameters
        self.declare_parameter("eval_ckpt_dir", "")
        self.declare_parameter("number_of_steps", 1000)
        
        eval_ckpt_dir = self.get_parameter("eval_ckpt_dir").get_parameter_value().string_value
        
        # Load dataset statistics
        stats_path = os.path.join(
            eval_ckpt_dir,
            "dataset_stats.pkl",
        )

        # Error handling for loading statistics
        try:
            with open(stats_path, "rb") as f:
                self.stats = pickle.load(f)
        except FileNotFoundError:
            self.get_logger().error(f"Statistics file not found at: {stats_path}")
            raise ValueError(f"Statistics file not found at: {stats_path}")

        # Preprocess and postprocess lambdas
        self.pre_process = (
            # 学習（EpisodicDataset）と同じ正規化
            lambda s_qpos: normalize_observation(self.stats, s_qpos)
        )
        self.post_process = (
            lambda a: a * self.stats["action_std"] + self.stats["action_mean"]
        )

        # Query frequency configuration
        self.query_frequency = policy_config["num_queries"]
        num_steps = self.get_parameter("number_of_steps").get_parameter_value().integer_value
        if policy_config["temporal_agg"]:
            self.query_frequency = 1
            self.all_time_actions = torch.zeros(
                [
                    num_steps,
                    num_steps,
                    +policy_config["num_queries"],
                    task_cfg["state_dim"],
                ],
                device=device,
            )

    def _image_converter(self, obs: Dict[str, Any]) -> torch.Tensor:
        """
        Convert images from observations to a tensor format.

        Args:
            obs (dict): Dictionary containing observed data.

        Returns:
            torch.Tensor: A tensor of shape (1, num_images, channels, height, width).
        """
        curr_images = []

        # Iterate through the camera names
        for cam_name in self.camera_names:
            # Retrieve and rearrange the image for the current camera
            curr_image = obs[f"{cam_name}_rgb"].transpose(2, 0, 1)
            curr_images.append(curr_image)

        # Stack images into a single numpy array
        curr_image = np.stack(curr_images, axis=0)

        # Convert to a PyTorch tensor, normalize and move to the specified device
        curr_image = (
            torch.from_numpy(curr_image / 255.0).float().to(self.device).unsqueeze(0)
        )

        return curr_image

    def act(self, obs: Dict[str, Any], t: int) -> np.ndarray:
        """
        Generate actions based on observations and current time step.

        Args:
            obs (dict): Dictionary containing observed data.
            t (int): Current time step.

        Returns:
            np.ndarray: The generated action.
        """

        action_dict={}
        obs_pos_rot_matrix = np.append(obs["ee_pos"],obs["ee_rot_matrix"].flatten())

        with torch.inference_mode():
            # Convert observations to image tensor
            img = self._image_converter(obs)

            # Preprocess joint positions
            processed_pos_rot_matrix = self.pre_process(obs_pos_rot_matrix)

            # Process joint positions
            processed_pos_rot_matrix = (
                torch.from_numpy(processed_pos_rot_matrix)
                .float()
                .to(self.device)
                .unsqueeze(0)
            )

            # Call the policy to get actions at specific time steps
            if t % self.query_frequency == 0:
                self.all_actions = self.policy(processed_pos_rot_matrix, img)

            if self.query_frequency == 1:  # Temporal aggregation
                self.all_time_actions[[t], t : t + self.query_frequency] = (
                    self.all_actions
                )
                actions_for_curr_step = self.all_time_actions[:, t]
                actions_populated = torch.all(actions_for_curr_step != 0, axis=1)
                actions_for_curr_step = actions_for_curr_step[actions_populated]

                # Exponential weights for actions
                k = 0.01
                exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                exp_weights = exp_weights / exp_weights.sum()
                exp_weights = (
                    torch.from_numpy(exp_weights.astype(np.float32))
                    .to(self.device)
                    .unsqueeze(dim=1)
                )

                raw_action = (actions_for_curr_step * exp_weights).sum(
                    dim=0, keepdim=True
                )
            else:
                raw_action = self.all_actions[:, t % self.query_frequency]

            # Post-process actions
            raw_action = raw_action.squeeze(0).cpu().numpy()
            action = self.post_process(raw_action)
            pos=action[:3]
            rot_matrix=action[3:].reshape((3,3))
            # Convert rotation matrix to quaternion and euler angles
            quat = R.from_matrix(rot_matrix).as_quat()  # [x, y, z, w] format
            euler = R.from_matrix(rot_matrix).as_euler('xyz')

            action_dict["joint_positions"]=np.zeros(6)
            action_dict["ee_pos"] = pos
            action_dict["ee_rot_matrix"] = rot_matrix
            action_dict["ee_quat"] = quat
            action_dict["ee_euler"]= euler

            return action_dict
