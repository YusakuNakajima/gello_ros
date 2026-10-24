import os
import datetime
import h5py
import numpy as np
import rclpy
from rclpy.node import Node
from typing import List, Dict, Any

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

def save_episode(node: Node, episode_number: int, obs_replay: List[Dict], action_replay: List[Dict]):
    """
    観測データと行動データをHDF5ファイルに保存します。

    Args:
        node (Node): 呼び出し元のROS2ノードオブジェクト。
        episode_number (int): エピソード番号。
        obs_replay (List[Dict]): 観測データのリスト。
        action_replay (List[Dict]): 行動データのリスト。
    """
    # --- パラメータの取得 ---
    # パラメータは呼び出し元のノードで宣言済みであることを前提とします。
    camera_names = node.get_parameter("camera_names").get_parameter_value().string_array_value
    cam_width = node.get_parameter("camera_width").get_parameter_value().integer_value
    cam_height = node.get_parameter("camera_height").get_parameter_value().integer_value
    wrench_dim = node.get_parameter("wrench_dim").get_parameter_value().integer_value
    save_episode_dir = node.get_parameter("save_episode_dir").get_parameter_value().string_value
    task_name = node.get_parameter("task_name").get_parameter_value().string_value
    use_FT_sensor = node.get_parameter("use_FT_sensor").get_parameter_value().bool_value
    
    # --- データの準備 ---
    node.get_logger().info(f"エピソード {episode_number} を {save_episode_dir} に保存します。")

    # データを格納するための辞書を初期化
    data_dict = {
        "/action/joint_positions": [], "/action/ee_pos": [], "/action/ee_quat": [],
        "/action/ee_rot_matrix": [], "/action/ee_euler": [],
        "/observations/joint_positions": [], "/observations/joint_velocities": [],
        "/observations/joint_torques": [], "/observations/ee_pos": [],
        "/observations/ee_quat": [], "/observations/ee_rot_matrix": [],
        "/observations/ee_euler": [],
        # 指令ガードで制限したあと、実際に controller に送った姿勢 [x,y,z,qx,qy,qz,qw]（送らなかった周期は NaN）。
        # /action/ee_* はエージェントが出した目標（制限前）
        "/action/sent_ee_pose": [],
    }

    # # カメラ画像を辞書に追加
    for cam_name in camera_names:
        data_dict[f"/observations/images/{cam_name}_rgb"] = []

    # # FTセンサーデータを辞書に追加
    if use_FT_sensor:
        data_dict["/observations/wrench"] = []

    # リプレイバッファからデータを辞書に格納
    for o, a in zip(obs_replay, action_replay):
        data_dict["/action/joint_positions"].append(a["joint_positions"])
        data_dict["/action/ee_pos"].append(a["ee_pos"])
        data_dict["/action/ee_quat"].append(a["ee_quat"])
        data_dict["/action/ee_rot_matrix"].append(a["ee_rot_matrix"])
        data_dict["/action/ee_euler"].append(a["ee_euler"])
        data_dict["/action/sent_ee_pose"].append(a.get("sent_ee_pose", np.full(7, np.nan)))
        
        data_dict["/observations/joint_positions"].append(o["joint_positions"])
        data_dict["/observations/joint_velocities"].append(o["joint_velocities"])
        data_dict["/observations/joint_torques"].append(o["joint_torques"])
        data_dict["/observations/ee_pos"].append(o["ee_pos"])
        data_dict["/observations/ee_quat"].append(o["ee_quat"])
        data_dict["/observations/ee_rot_matrix"].append(o["ee_rot_matrix"])
        data_dict["/observations/ee_euler"].append(o["ee_euler"])
        
        if use_FT_sensor:
            data_dict["/observations/wrench"].append(o["ee_wrench"])
            
        for cam_name in camera_names:
            img_name = f"{cam_name}_rgb"
            data_dict[f"/observations/images/{img_name}"].append(o[img_name])

    # --- HDF5ファイルへの保存 ---
    data_dir = os.path.join(save_episode_dir, f"{timestamp}_{task_name}")
    
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    dataset_path = os.path.join(data_dir, f"episode_{episode_number}.hdf5")

    max_timesteps = len(obs_replay)
    print(data_dict.keys())
    with h5py.File(dataset_path, "w", rdcc_nbytes=1024**2 * 2) as root:
        root.attrs["sim"] = True
        obs = root.create_group("observations")
        image = obs.create_group("images")
        action = root.create_group("action")

        # データセットの作成
        for cam_name in camera_names:
            image.create_dataset(
                f"{cam_name}_rgb", (max_timesteps, cam_height, cam_width, 3), 
                dtype="uint8", chunks=(1, cam_height, cam_width, 3)
            )
        
        obs.create_dataset("joint_positions", (max_timesteps, 7))
        obs.create_dataset("joint_velocities", (max_timesteps, 7))
        obs.create_dataset("joint_torques", (max_timesteps, 7))
        obs.create_dataset("ee_pos", (max_timesteps, 3))
        obs.create_dataset("ee_quat", (max_timesteps, 4))
        obs.create_dataset("ee_rot_matrix", (max_timesteps, 3, 3))
        obs.create_dataset("ee_euler", (max_timesteps, 3))
        
        action.create_dataset("joint_positions", (max_timesteps, 7))
        action.create_dataset("ee_pos", (max_timesteps, 3))
        action.create_dataset("ee_quat", (max_timesteps, 4))
        action.create_dataset("ee_rot_matrix", (max_timesteps, 3, 3))
        action.create_dataset("ee_euler", (max_timesteps, 3))
        action.create_dataset("sent_ee_pose", (max_timesteps, 7))

        if use_FT_sensor:
            obs.create_dataset("wrench", (max_timesteps, wrench_dim))

        # データを書き込み
        for name, data in data_dict.items():
            root[name][...] = np.array(data)

    node.get_logger().info(f"データを {dataset_path} に正常に保存しました。")


# --- 使用例 ---
def main(args=None):
    # このmain関数は、save_episode関数の使い方を示すためのサンプルです。
    rclpy.init(args=args)
    
    # ダミーのノードを作成
    dummy_node = Node("episode_saver_test_node")
    
    # ダミーのデータを作成
    num_steps = 10
    obs_replay_dummy = [{
        "joint_positions": np.random.rand(6), "joint_velocities": np.random.rand(6),
        "joint_torques": np.random.rand(6), "ee_pos": np.random.rand(3),
        "ee_quat": np.random.rand(4), "ee_rot_matrix": np.random.rand(3, 3),
        "ee_euler": np.random.rand(3), "ee_wrench": np.random.rand(6),
        "base_rgb": np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
    } for _ in range(num_steps)]
    
    action_replay_dummy = [{
        "joint_positions": np.random.rand(6), "ee_pos": np.random.rand(3),
        "ee_quat": np.random.rand(4), "ee_rot_matrix": np.random.rand(3, 3),
        "ee_euler": np.random.rand(3)
    } for _ in range(num_steps)]

    # 関数を呼び出し
    save_episode(dummy_node, episode_number=0, obs_replay=obs_replay_dummy, action_replay=action_replay_dummy)
    
    dummy_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()