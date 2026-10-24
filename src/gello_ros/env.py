import time
from typing import Any, Dict, List
from functools import partial

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image

from gello_ros.robots.robot import Robot
from gello_ros.freshness import input_age_s, stamp_to_ns, stamp_went_backwards


def image_msg_to_numpy(msg: Image) -> np.ndarray:
    """Convert ROS Image message to numpy array without CvBridge for zero-copy efficiency.
    
    Args:
        msg: ROS Image message
        
    Returns:
        numpy array with shape (height, width, channels)
    """
    if msg is None:
        return None
        
    # Convert image data to numpy array
    dtype_map = {
        'rgb8': (np.uint8, 3),
        'bgr8': (np.uint8, 3), 
        'rgba8': (np.uint8, 4),
        'bgra8': (np.uint8, 4),
        'mono8': (np.uint8, 1),
        'mono16': (np.uint16, 1)
    }
    
    if msg.encoding not in dtype_map:
        raise ValueError(f"Unsupported image encoding: {msg.encoding}")
    
    dtype, channels = dtype_map[msg.encoding]
    
    # Convert bytes to numpy array
    image_array = np.frombuffer(msg.data, dtype=dtype)
    
    if channels == 1:
        # Grayscale image
        image_array = image_array.reshape((msg.height, msg.width))
    else:
        # Multi-channel image
        image_array = image_array.reshape((msg.height, msg.width, channels))
        
        # Convert BGR to RGB if necessary
        if msg.encoding == 'bgr8':
            image_array = image_array[:, :, ::-1]  # BGR -> RGB
        elif msg.encoding == 'bgra8':
            image_array = image_array[:, :, [2, 1, 0, 3]]  # BGRA -> RGBA
    
    return image_array



class Rate:
    def __init__(self, rate: float):
        self.last = time.time()
        self.rate = rate

    def sleep(self) -> None:
        while self.last + (1.0 / self.rate) > time.time():
            time.sleep(0.00001)
        self.last = time.time()


class RobotEnv(Node):
    def __init__(
        self,
        robot: Robot,
        control_rate_hz: float = 100.0,
        camera_names: List[str] = [],
        control_mode: str = "cartesian",
        node_name: str = "robot_env",
    ) -> None:
        super().__init__(node_name)
        assert control_mode in ["joint", "cartesian"]
        self._robot = robot
        self._rate = Rate(control_rate_hz)
        self._control_mode = control_mode
        
        # Camera handling - Zero-copy mode
        self.camera_names = camera_names
        self.camera_images = {}  # Store Image messages directly for zero-copy
        # カメラごとに最後に画像を受信した時刻と、その header.stamp（ROS 時刻、ns）
        self.camera_received_ns = {}
        self.camera_stamp_ns = {}
        # 画像・力センサの値がこれより古ければ、観測の stale_sensors に名前を入れる（ms）
        self.declare_parameter("sensor_timeout_ms", 200.0)
        self.sensor_timeout_s = self.get_parameter("sensor_timeout_ms").get_parameter_value().double_value * 1e-3
        if not self.sensor_timeout_s > 0.0:
            raise ValueError("sensor_timeout_ms must be > 0")
        self.latest_valid_images = {}  # Store latest valid numpy arrays as fallback
        if self.camera_names:
            self.start_camera_subscribers()

    def start_camera_subscribers(self):
        """Start camera subscribers for all camera names"""
        for camera_name in self.camera_names:
            # Try to subscribe to shared memory topic first, fallback to regular topic
            topic_name = f"/{camera_name}/color/image_raw"
            self.create_subscription(
                Image,
                topic_name,
                partial(self._camera_color_callback, camera_name),
                qos_profile_sensor_data,
            )
            self.get_logger().info(f"Subscribed to {topic_name} in RobotEnv")
        
        # Wait briefly for initial camera connection
        self.get_logger().info("カメラ接続を待機中...")
        import time
        time.sleep(1.0)  # Give cameras time to connect
        
        # Try to receive initial camera data
        for _ in range(50):  # Try for up to 5 seconds
            rclpy.spin_once(self, timeout_sec=0.1)
            if len(self.camera_images) == len(self.camera_names):
                self.get_logger().info("全てのカメラが接続されました")
                break
        else:
            self.get_logger().warn(f"カメラ接続タイムアウト。接続済み: {len(self.camera_images)}/{len(self.camera_names)}")

    def _camera_color_callback(self, camera_name, msg: Image):
        """Callback for the color image - Zero-copy mode: store Image message directly."""
        # Store the Image message directly for zero-copy access
        # Image-to-numpy conversion will be done on-demand in get_obs()
        stamp_ns = stamp_to_ns(msg.header.stamp)
        if stamp_went_backwards(self.camera_stamp_ns.get(camera_name), stamp_ns):
            self.get_logger().warn(f"{camera_name}: image stamp went backwards; ignoring it.", throttle_duration_sec=5)
            return
        self.camera_images[camera_name] = msg
        self.camera_received_ns[camera_name] = self.get_clock().now().nanoseconds
        self.camera_stamp_ns[camera_name] = stamp_ns

    def robot(self) -> Robot:
        """Get the robot object.

        Returns:
            robot: the robot object.
        """
        return self._robot

    def __len__(self):
        return 0

    def step(self, action: Dict[str, np.ndarray]) -> Dict[str, Any]:
        if action.get("hold_measured", False):
            # 止めるとき: 速度の上限を掛けずに、いまの計測姿勢を目標にする
            self._robot.hold_measured_pose()
        elif self._control_mode == "cartesian":
            # Ensure keys exist before accessing
            if "ee_pos" in action and "ee_quat" in action:
                pos_quat = np.concatenate([action["ee_pos"], action["ee_quat"]])
                self._robot.command_pose(pos_quat)
        elif self._control_mode == "joint":
            if "joint_positions" in action:
                self._robot.command_joint_state(action["joint_positions"])
        else:
            raise ValueError(f"Invalid control mode: {self._control_mode}")
        
        self._rate.sleep()
        return self.get_obs()
    
    def wait_for_fresh_inputs(self, timeout_s: float = 2.0) -> Dict[str, Any]:
        """溜まっていたコールバックを処理し、入力が新しくなるまで最大 timeout_s 待って観測を返す。

        方策の読み込みなどで spin しない時間があると、最初に処理する画像は溜まっていた古いものになる。
        """
        deadline = time.monotonic() + timeout_s
        obs = self.get_obs()
        while obs["stale_sensors"] and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.01)
            obs = self.get_obs()
        return obs

    def get_obs(self) -> Dict[str, Any]:
        """Get observation from the environment."""
        # Only spin robot node to update robot states
        # Camera updates are handled by the main spin loop in component mode
        rclpy.spin_once(self._robot, timeout_sec=0.001)
        rclpy.spin_once(self, timeout_sec=0.0001) # Process camera callbacks

        observations = {}
        
        # Add camera images to observations - Zero-copy conversion
        for name in self.camera_names:
            image_msg = self.camera_images.get(name)
            if image_msg is not None:
                # 新しい画像が利用可能な場合、numpy配列に変換して保存
                img_array = image_msg_to_numpy(image_msg)
                observations[f"{name}_rgb"] = img_array
                self.latest_valid_images[name] = img_array  # 最新の有効な画像として保存
                self.get_logger().debug(f"カメラ画像を観測に追加: {name}")
            else:
                # 画像が利用できない場合、最新の有効な画像を使用
                if name in self.latest_valid_images:
                    observations[f"{name}_rgb"] = self.latest_valid_images[name]
                    self.get_logger().debug(f"以前のカメラ画像を使用: {name}")
                else:
                    observations[f"{name}_rgb"] = None
                    self.get_logger().warn(f"カメラ画像が利用できません（履歴もありません）: {name}")

        robot_obs = self._robot.get_observations()
        # It's safer to merge dictionaries
        observations.update(robot_obs)

        # 途切れた・遅れて届く入力（古い画像・力・joint state）を、ほかの入力と組み合わせて使わないよう、
        # 古い入力の名前を載せる（受信時刻と header.stamp の両方で判定）
        now_ns = self.get_clock().now().nanoseconds
        stale = [
            name for name in self.camera_names
            if input_age_s(now_ns, self.camera_received_ns.get(name), self.camera_stamp_ns.get(name))
            > self.sensor_timeout_s
        ]
        if observations.get("wrench_age_s", 0.0) > self.sensor_timeout_s:
            stale.append("wrench")
        if not observations.get("joint_state_fresh", True):
            stale.append("joint_states")
        observations["stale_sensors"] = stale
        
        # Ensure essential keys are present
        assert "joint_positions" in observations
        assert "joint_velocities" in observations
        assert "ee_pos" in observations
        
        return observations


def main() -> None:
    pass


if __name__ == "__main__":
    main()