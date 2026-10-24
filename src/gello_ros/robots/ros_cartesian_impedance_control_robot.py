from typing import Dict, Tuple

import numpy as np
from scipy.spatial.transform import Rotation as R

from gello_ros.robots.robot import Robot
from gello_ros.robots.pose_command_guard import PoseCommandLimits, limit_pose_command

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped, WrenchStamped
from std_srvs.srv import Empty
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from kdl_parser_py.kdl_helper import KDLHelper
import time


class CartesianImpedanceControlRobot(Robot, Node):
    """A class representing a UR robot with Cartesian impedance control."""

    def __init__(
        self,
        use_gripper: bool = False,
        robot_description: str = None,
    ):
        # Initialize ROS2 node
        Node.__init__(self, 'cartesian_impedance_control_robot')
        
        if use_gripper:
            self.get_logger().error("Only no gripper configuration supported")
            raise NotImplementedError("Gripper support not implemented")

        # Declare ROS2 parameters
        self._declare_parameters()
        
        # Get parameters
        self.get_parameters()

        # Log topic configuration
        self._log_topic_info()

        # Initialize publishers
        self.cartesian_command_publisher = self.create_publisher(
            PoseStamped,
            self.cartesian_impedance_controller_command_topic,
            1
        )

        # Initialize subscribers
        self.joint_state_subscription = self.create_subscription(
            JointState,
            self.joint_states_topic,
            self.joint_states_callback,
            1
        )
        self.wrench_subscription = self.create_subscription(
            WrenchStamped,
            self.feedback_wrench_topic,
            self.wrench_callback,
            1
        )
        
        # Initialize variables
        self.ros_joint_state = None
        self._wrench = None
        self._joint_positions = None
        self._joint_velocities = None
        self._joint_efforts = None
        # joint state を最後に受信した時刻と、その header.stamp（どちらも ROS 時刻、ns）。
        # 古い計測で指令を出さないために使う。stamp が 0 のメッセージは受信時刻だけで判定する
        self._joint_state_received_ns = None
        self._joint_state_stamp_ns = None
        # 力センサの値を最後に受信した時刻（ROS 時刻、ns）。観測に「何秒前の値か」を載せるために使う
        self._wrench_received_ns = None
        # 最後に送った指令姿勢と、その時刻（ROS 時刻、ns）
        self._last_command_pose = None
        self._last_command_ns = None
        
        # 安全機能: 最後に有効だったjoint state (position, velocity, effort)
        self._last_valid_joint_positions = None
        self._last_valid_joint_velocities = None
        self._last_valid_joint_efforts = None

        # Wait for joint states
        self.get_logger().info(f"Waiting for joint states on topic: {self.joint_states_topic}")
        start_time = time.time()
        while self._joint_positions is None and rclpy.ok():
            if time.time() - start_time > 5:
                self.get_logger().error(f"Timeout waiting for joint_states_topic: {self.joint_states_topic}. システムを終了します。")
                exit(1)
            rclpy.spin_once(self, timeout_sec=0.1)

        # Wait for wrench feedback
        self.get_logger().info(f"Waiting for wrench feedback on topic: {self.feedback_wrench_topic}")
        start_time = time.time()
        while self._wrench is None and rclpy.ok():
            if time.time() - start_time > 5:
                self.get_logger().error(f"Timeout waiting for feedback_wrench_topic: {self.feedback_wrench_topic}. Exiting.")
                raise TimeoutError("Wrench feedback timeout")
            rclpy.spin_once(self, timeout_sec=0.1)

        # Initialize service client for FT sensor zero reset
        self.feedback_wrench_zero_client = self.create_client(
            Empty, 
            self.feedback_wrench_zero_service
        )
        
        # Wait for service
        self.get_logger().info("Waiting for feedback wrench zero service...")
        if not self.feedback_wrench_zero_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().error("Feedback wrench zero service not available")
            raise TimeoutError("Service timeout")
        
        # Call zero reset service
        self.get_logger().info("Zero reset feedback FT sensor offset")
        future = self.feedback_wrench_zero_client.call_async(Empty.Request())
        rclpy.spin_until_future_complete(self, future)
        if future.result() is None:
            self.get_logger().error("Failed to call feedback wrench zero service")
        time.sleep(1)

        
     
        # Initialize kinematics
        try:
            urdf_string = robot_description
            if not urdf_string:
                try:
                    self.get_logger().info(f"Attempting to get URDF from parameter: robot_description")
                    urdf_string = self.get_parameter("robot_description").get_parameter_value().string_value
                    if not urdf_string or len(urdf_string) == 0:
                        self.get_logger().warn("robot_description parameter is empty, trying to get from external parameter server")
                        try:
                            import subprocess
                            result = subprocess.run(['ros2', 'param', 'get', '/robot_state_publisher', 'robot_description'], capture_output=True, text=True, timeout=2.0)
                            if result.returncode == 0:
                                raw_output = result.stdout.strip()
                                xml_start = -1
                                for tag in ['<?xml', '<robot']:
                                    idx = raw_output.find(tag)
                                    if idx != -1:
                                        xml_start = idx
                                        break
                                if xml_start != -1:
                                    urdf_string = raw_output[xml_start:]
                        except Exception as ext_e:
                            self.get_logger().debug(f"Failed to get URDF from external source: {ext_e}")
                except Exception as e:
                    self.get_logger().error(f"Failed to get robot_description parameter: {e}")
                
            if urdf_string and len(urdf_string) > 0:
                self.get_logger().info(f"Initializing KDL with base_link='{self.robot_base_frame}', ee_link='{self.ee_link}'")
                self.kdl_helper = KDLHelper(self.get_logger(), urdf_path=None, urdf_string=urdf_string, base_link=self.robot_base_frame, ee_link=self.ee_link)
                self.get_logger().info("KDL kinematics initialized successfully")
            else:
                self.kdl_helper = None
                self.get_logger().error("No URDF string available or empty string. KDL kinematics disabled.")
                
        except Exception as e:
            self.get_logger().error(f"Failed to initialize KDL kinematics: {e}")
            self.kdl_helper = None

        if self.kdl_helper is None:
            raise RuntimeError("KDL kinematics failed to initialize.")

        # Control parameters
        control_freq = 100
        self._min_traj_dur = 5.0 / control_freq
        self._speed_scale = 1
        self._use_gripper = use_gripper

    def _declare_parameters(self):
        """Declare ROS2 parameters"""
        self.declare_parameter("joint_names_order", [""])
        self.declare_parameter("joint_max_vel", [0.0])
        self.declare_parameter("joint_pos_limits_upper", [0.0])
        self.declare_parameter("joint_pos_limits_lower", [0.0])
        self.declare_parameter("cartesian_impedance_controller_command_topic", "/cartesian_impedance_controller/target_frame")
        self.declare_parameter("joint_states_topic", "/joint_states")
        self.declare_parameter("feedback_wrench_topic", "/ft_sensor/wrench")
        self.declare_parameter("feedback_wrench_zero_service", "/ft_sensor/zero")
        self.declare_parameter("robot_base_frame", "base_link")
        self.declare_parameter("ee_link", "tool0")
        self.declare_parameter("robot_description_name", "/robot_description")
        self.declare_parameter("robot_description", "")
        # 指令ガード（pose_command_guard）の上限。既定値は暫定の推測値（模擬HWでのみ確認）
        self.declare_parameter("max_linear_speed_m_per_s", 0.1)
        self.declare_parameter("max_angular_speed_rad_per_s", 0.5)
        self.declare_parameter("max_tracking_error_m", 0.02)
        self.declare_parameter("max_tracking_error_rad", 0.1)
        # これより古い joint state しかないときは指令を出さない（ms）
        self.declare_parameter("joint_state_timeout_ms", 100.0)
        # 前回の指令からの経過時間として使う上限（s）。間が空いても、1 回に進める量をこれ以上にしない
        self.declare_parameter("max_command_interval_s", 0.02)

    def get_parameters(self):
        """Get ROS2 parameters"""
        self.joint_names_order = self.get_parameter("joint_names_order").get_parameter_value().string_array_value
        self.joint_max_vel = self.get_parameter("joint_max_vel").get_parameter_value().double_array_value
        self.joint_pos_limits_upper = self.get_parameter("joint_pos_limits_upper").get_parameter_value().double_array_value
        self.joint_pos_limits_lower = self.get_parameter("joint_pos_limits_lower").get_parameter_value().double_array_value
        self.cartesian_impedance_controller_command_topic = self.get_parameter("cartesian_impedance_controller_command_topic").get_parameter_value().string_value
        self.joint_states_topic = self.get_parameter("joint_states_topic").get_parameter_value().string_value
        self.feedback_wrench_topic = self.get_parameter("feedback_wrench_topic").get_parameter_value().string_value
        self.feedback_wrench_zero_service = self.get_parameter("feedback_wrench_zero_service").get_parameter_value().string_value
        self.robot_base_frame = self.get_parameter("robot_base_frame").get_parameter_value().string_value
        self.ee_link = self.get_parameter("ee_link").get_parameter_value().string_value
        self.robot_description_name = self.get_parameter("robot_description_name").get_parameter_value().string_value
        self.command_limits = PoseCommandLimits(
            max_linear_speed_m_per_s=self.get_parameter("max_linear_speed_m_per_s").get_parameter_value().double_value,
            max_angular_speed_rad_per_s=self.get_parameter("max_angular_speed_rad_per_s").get_parameter_value().double_value,
            max_tracking_error_m=self.get_parameter("max_tracking_error_m").get_parameter_value().double_value,
            max_tracking_error_rad=self.get_parameter("max_tracking_error_rad").get_parameter_value().double_value,
        )
        self.joint_state_timeout_ns = int(self.get_parameter("joint_state_timeout_ms").get_parameter_value().double_value * 1e6)
        self.max_command_interval_s = self.get_parameter("max_command_interval_s").get_parameter_value().double_value
        if self.joint_state_timeout_ns <= 0 or not self.max_command_interval_s > 0.0:
            raise ValueError("joint_state_timeout_ms and max_command_interval_s must be > 0")

    def _log_topic_info(self) -> None:
        """Log publisher and subscriber topic information."""
        self.get_logger().info("=== CartesianImpedanceControlRobot Topic Configuration ===")
        self.get_logger().info("Publishers:")
        self.get_logger().info(f"  - {self.cartesian_impedance_controller_command_topic}: PoseStamped")
        self.get_logger().info("Subscribers:")
        self.get_logger().info(f"  - {self.joint_states_topic}: JointState")
        self.get_logger().info(f"  - {self.feedback_wrench_topic}: WrenchStamped")
        self.get_logger().info("Service Clients:")
        self.get_logger().info(f"  - {self.feedback_wrench_zero_service}: Empty")
        self.get_logger().info("Key Parameters:")
        self.get_logger().info(f"  - robot_base_frame: {self.robot_base_frame}")
        self.get_logger().info(f"  - ee_link: {self.ee_link}")
        self.get_logger().info(f"  - joint_names_order: {self.joint_names_order}")
        self.get_logger().info(f"  - command limits: {self.command_limits}")
        self.get_logger().info("================================================================")

    def joint_states_callback(self, msg: JointState):
        """Joint states callback to process position, velocity, and effort."""
        self.ros_joint_state = msg

        if not msg.name:
            self.get_logger().warn("Received JointState message with no joint names.", throttle_duration_sec=5)
            return

        pos_dict = dict(zip(msg.name, msg.position))
        vel_dict = dict(zip(msg.name, msg.velocity)) if msg.velocity else {}
        eff_dict = dict(zip(msg.name, msg.effort)) if msg.effort else {}

        stamp_ns = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec
        if stamp_ns != 0 and self._joint_state_stamp_ns and stamp_ns < self._joint_state_stamp_ns:
            # 時刻が戻ったメッセージ（遅れて届いた・再送された）は使わない
            self.get_logger().warn("JointState stamp went backwards; ignoring it.", throttle_duration_sec=5)
            return
        missing = [name for name in self.joint_names_order if name not in pos_dict]
        if missing:
            # 欠けた関節を 0 で埋めると FK が別の姿勢を返すので、この joint state は使わない
            self.get_logger().warn(f"JointState is missing joints {missing}; ignoring it.", throttle_duration_sec=5)
            return
        ordered_pos = np.array([pos_dict[name] for name in self.joint_names_order])
        if not np.all(np.isfinite(ordered_pos)):
            self.get_logger().warn("JointState has a non-finite position; ignoring it.", throttle_duration_sec=5)
            return
        ordered_vel = np.array([vel_dict.get(name, 0.0) for name in self.joint_names_order]) if vel_dict else np.zeros_like(ordered_pos)
        ordered_eff = np.array([eff_dict.get(name, 0.0) for name in self.joint_names_order]) if eff_dict else np.zeros_like(ordered_pos)

        self._joint_positions = ordered_pos
        self._joint_velocities = ordered_vel
        self._joint_efforts = ordered_eff
        self._joint_state_received_ns = self.get_clock().now().nanoseconds
        self._joint_state_stamp_ns = stamp_ns

        self._last_valid_joint_positions = self._joint_positions.copy()
        self._last_valid_joint_velocities = self._joint_velocities.copy()
        self._last_valid_joint_efforts = self._joint_efforts.copy()

    def wrench_callback(self, msg: WrenchStamped):
        """Wrench feedback callback"""
        self._wrench = msg
        self._wrench_received_ns = self.get_clock().now().nanoseconds

    def num_dofs(self) -> int:
        return len(self.joint_names_order)

    def get_joint_state(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get the current state (positions, velocities, efforts) of the robot joints.
        Uses the last valid state as a fallback if the current state is not available.
        """
        # 初回コールバック受信前に呼び出された場合のエラーハンドリング
        if self._joint_positions is None and self._last_valid_joint_positions is None:
            self.get_logger().error("Joint states have not been received yet. Exiting.")
            exit(1)

        num_joints = len(self.joint_names_order)
        
        # 最新のコールバック値があればそれを、なければ最後の有効な値を、それもなければゼロ配列を返す
        joint_pos = self._joint_positions if self._joint_positions is not None else self._last_valid_joint_positions
        joint_vel = self._joint_velocities if self._joint_velocities is not None else self._last_valid_joint_velocities
        joint_eff = self._joint_efforts if self._joint_efforts is not None else self._last_valid_joint_efforts

        # 万が一に備え、Noneの場合はゼロ配列で初期化
        if joint_pos is None: joint_pos = np.zeros(num_joints)
        if joint_vel is None: joint_vel = np.zeros(num_joints)
        if joint_eff is None: joint_eff = np.zeros(num_joints)
        
        return joint_pos.copy(), joint_vel.copy(), joint_eff.copy()


    def command_joint_state(self, joint_state: np.ndarray) -> None:
        """Command the leader robot to a given state."""
        try:
            if self.kdl_helper is None:
                self.get_logger().warn("KDL helper not available. Cannot command joint state.")
                return
            
            pose = self.kdl_helper.forward_kinematics(joint_state.tolist())
            self.command_pose(np.asarray(pose, dtype=float))
        except Exception as e:
            self.get_logger().error(f"Failed to command joint state: {e}")

    def _measured_pose(self, now_ns: int):
        """新しい joint state から FK で計測姿勢を返す。joint state が古い・ない、FK に失敗した場合は None。"""
        if self._joint_state_received_ns is None or self._joint_positions is None:
            return None
        if now_ns - self._joint_state_received_ns > self.joint_state_timeout_ns:
            return None
        # 受信は新しくても、計測した時刻（header.stamp）が古い・未来なら使わない
        if self._joint_state_stamp_ns and abs(now_ns - self._joint_state_stamp_ns) > self.joint_state_timeout_ns:
            return None
        try:
            return np.asarray(self.kdl_helper.forward_kinematics(self._joint_positions.tolist()), dtype=float)
        except Exception as e:
            self.get_logger().error(f"Failed to compute FK: {e}", throttle_duration_sec=1)
            return None

    def command_pose(self, pose: np.ndarray) -> None:
        """目標姿勢を、速度と追従誤差の上限で制限してから controller に送る。

        joint state が古い・FK に失敗した・目標が無効な場合は送らず、次に送るときは計測姿勢から始め直す。
        """
        now_ns = self.get_clock().now().nanoseconds
        measured = self._measured_pose(now_ns)
        if measured is None:
            self.get_logger().warn("No fresh joint state / FK; not sending the pose command.", throttle_duration_sec=1)
            self._last_command_pose = None
            return
        if self._last_command_ns is None:
            dt_s = 0.0
        else:
            dt_s = min((now_ns - self._last_command_ns) * 1e-9, self.max_command_interval_s)
        limited = limit_pose_command(pose, self._last_command_pose, measured, dt_s, self.command_limits)
        if limited is None:
            self.get_logger().warn(f"Invalid pose command {pose}; not sending it.", throttle_duration_sec=1)
            self._last_command_pose = None
            return
        self._last_command_pose = limited
        self._last_command_ns = now_ns
        pose = limited

        pose_stamped = PoseStamped()
        pose_stamped.header.stamp = self.get_clock().now().to_msg()
        pose_stamped.header.frame_id = self.robot_base_frame
        pose_stamped.pose.position.x, pose_stamped.pose.position.y, pose_stamped.pose.position.z = pose[:3]
        pose_stamped.pose.orientation.x, pose_stamped.pose.orientation.y, pose_stamped.pose.orientation.z, pose_stamped.pose.orientation.w = pose[3:]
        self.cartesian_command_publisher.publish(pose_stamped)

    def get_observations(self) -> Dict[str, np.ndarray]:
        """Get robot observations including kinematics and sensor data"""
        # get_joint_stateを使用して、位置、速度、力の情報をまとめて取得
        joint_pos, joint_vel, joint_eff = self.get_joint_state()
        
        try:
            pos_quat = np.array(self.kdl_helper.forward_kinematics(joint_pos.tolist())) if self.kdl_helper else np.zeros(7)
        except Exception as e:
            self.get_logger().error(f"Failed to compute FK: {e}")
            pos_quat = np.zeros(7)
            
        gripper_pos = np.array([joint_pos[-1]]) if len(joint_pos) > 0 else np.array([0.0])

        wrench = np.array([self._wrench.wrench.force.x, self._wrench.wrench.force.y, self._wrench.wrench.force.z, self._wrench.wrench.torque.x, self._wrench.wrench.torque.y, self._wrench.wrench.torque.z]) if self._wrench else np.zeros(6)

        try:
            num_joints = len(self.joint_names_order)
            jacobian = self.kdl_helper.jacobian(joint_pos.tolist()) if self.kdl_helper else np.zeros((6, num_joints))
        except Exception as e:
            self.get_logger().error(f"Failed to compute Jacobian: {e}")
            jacobian = np.zeros((6, num_joints))

        quat, rot_matrix, euler_angles = np.array([0.,0.,0.,1.]), np.eye(3), np.zeros(3)
        if len(pos_quat) >= 7:
            try:
                quat_raw = pos_quat[3:7]
                if np.linalg.norm(quat_raw) > 1e-6:
                    quat = quat_raw / np.linalg.norm(quat_raw)
                    rotation = R.from_quat(quat)
                    rot_matrix = rotation.as_matrix()
                    euler_angles = rotation.as_euler('xyz')
            except Exception as e:
                self.get_logger().error(f"Failed to convert quaternion: {e}")

        return {
            "joint_positions": joint_pos,
            "joint_velocities": joint_vel,
            "joint_torques": joint_eff,
            "gripper_position": gripper_pos,
            "ee_pos": pos_quat[:3] if len(pos_quat) >= 3 else np.zeros(3),
            "ee_quat": quat,
            "ee_rot_matrix": rot_matrix,
            "ee_euler": euler_angles,
            "ee_wrench": wrench,
            # 力センサの値が何秒前に届いたものか（一度も届いていなければ inf）
            "wrench_age_s": (
                (self.get_clock().now().nanoseconds - self._wrench_received_ns) * 1e-9
                if self._wrench_received_ns is not None else float("inf")
            ),
            "jacobian": jacobian,
        }


def main():
    rclpy.init()
    try:
        ros_robot = CartesianImpedanceControlRobot(use_gripper=False)
        rclpy.spin(ros_robot)
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()