import time
from typing import Dict, Tuple

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, WrenchStamped
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from scipy.spatial.transform import Rotation as R
from touch_msgs.msg import TouchButtonEvent, TouchFeedback

import tf2_ros
from gello_ros.agents.agent import Agent
from gello_ros.freshness import input_age_s, monotonic_ns, stamp_to_ns, stamp_went_backwards
from kdl_parser_py.kdl_helper import KDLHelper

# tf2_geometry_msgsはPoseやWrenchの変換に便利ですが、
# 今回は元のロジックを尊重し、numpyとscipyで計算しています。
# import tf2_geometry_msgs


def limit_norm(vector: np.ndarray, max_norm: float) -> np.ndarray:
    """ベクトルの大きさを max_norm 以下にする（向きは変えない）。"""
    vector = np.asarray(vector, dtype=float)
    norm = np.linalg.norm(vector)
    if not np.isfinite(norm):
        return np.zeros_like(vector)
    return vector * (max_norm / norm) if norm > max_norm else vector


def rotate_vector(rotation_matrix: np.ndarray, vector: np.ndarray) -> np.ndarray:
    """rotation_matrix（A 座標の姿勢を B 座標で表したもの）で、A 座標のベクトルを B 座標に移す。"""
    return np.asarray(rotation_matrix) @ np.asarray(vector)


class TouchAgent(Agent, Node):
    """
    3D Systems Touchデバイスからの入力に基づき、ロボットアームを遠隔操作するエージェント。

    このクラスは、Touchデバイスの姿勢とボタン入力を購読し、
    ロボットのエンドエフェクタの目標姿勢を計算します。
    また、ロボットからの力覚フィードバックをTouchデバイスに送信する機能も持ちます。
    """

    def __init__(self, robot_description: str = None):
        """ノードを初期化し、パラメータ、通信、TFを設定します。"""
        Node.__init__(self, 'touch_agent')
        Agent.__init__(self)

        # メンバ変数の初期化
        self._robot_description = robot_description
        self._touch_current_pose: np.ndarray | None = None
        self._robot_start_pose_with_touch: np.ndarray | None = None
        self._robot_start_pose: np.ndarray | None = None
        self._robot_current_pose: np.ndarray | None = None
        self._last_target_pose: np.ndarray | None = None  # 最後に計算された目標姿勢
        self._pose_initialized = False  # 初期姿勢が設定されたかのフラグ

        self.pose_static_transform: np.ndarray | None = None
        self.wrench_static_transform: np.ndarray | None = None

        # テレオペレーションの状態管理フラグ
        self._is_teleop_active = False  # 白ボタンが押されているか
        self._was_teleop_active = False # 前のステップで白ボタンが押されていたか
        self._is_z_lock_active = False  # 灰ボタンが押されているか
        # Touch の姿勢を最後に受信した時刻（monotonic、ns）と header.stamp（ROS 時刻、ns）。ドライバが止まったことを検出する
        self._touch_received_ns: int | None = None
        self._touch_stamp_ns: int | None = None
        # Touch が途切れたあとは、白ボタンをいったん離すまでテレオペを再開しない
        #（ボタンは変化したときしか配信されないので、押したままの状態が残り続けないようにする）
        self._needs_rearm = False
        
        # Z軸下向きの回転（固定値）
        self.z_down_quat = R.from_euler('xyz', [0, np.pi, 0]).as_quat()

        # ROS関連の設定
        self._setup_parameters()
        self._setup_ros_communications()
        self._setup_tf()
        self._setup_kdl()
        
        # Topic一覧を表示
        self._log_topic_info()

        # Touchデバイスからの最初のポーズメッセージを待つ
        self._wait_for_first_pose()

    def _setup_parameters(self) -> None:
        """ROSパラメータを宣言し、読み込みます。"""
        self.declare_parameter("touch_ee_pose_topic", "/touch/tip_pose")
        self.declare_parameter("touch_button_topic", "/touch/button_event")
        self.declare_parameter("touch_force_feedback_topic", "/touch/force_feedback")
        self.declare_parameter("touch_max_force", 1.5)
        self.declare_parameter("force_scale_to_touch", 1.0)
        self.declare_parameter("teleoperation_mode", "bilateral")
        # Touch の姿勢がこれより古ければテレオペを止める（ms）。touch_state は 1 kHz で配信する
        self.declare_parameter("touch_timeout_ms", 100.0)
        self.declare_parameter("robot_base_frame", "base_link")
        self.declare_parameter("touch_base_frame", "touch_base")
        self.declare_parameter("touch_force_frame", "touch_force_frame")
        self.declare_parameter("feedback_wrench_sensor_frame", "tool0")
        self.declare_parameter("robot_description", "")

        self.ee_pose_topic = self.get_parameter("touch_ee_pose_topic").get_parameter_value().string_value
        self.button_topic = self.get_parameter("touch_button_topic").get_parameter_value().string_value
        self.force_feedback_topic = self.get_parameter("touch_force_feedback_topic").get_parameter_value().string_value
        self.touch_max_force = self.get_parameter("touch_max_force").get_parameter_value().double_value
        self.force_scale_to_touch = self.get_parameter("force_scale_to_touch").get_parameter_value().double_value
        self.teleop_mode = self.get_parameter("teleoperation_mode").get_parameter_value().string_value
        self.touch_timeout_s = self.get_parameter("touch_timeout_ms").get_parameter_value().double_value * 1e-3
        if not self.touch_timeout_s > 0.0:
            raise ValueError("touch_timeout_ms must be > 0")
        self.robot_base_frame = self.get_parameter("robot_base_frame").get_parameter_value().string_value
        self.touch_base_frame = self.get_parameter("touch_base_frame").get_parameter_value().string_value
        self.touch_force_frame = self.get_parameter("touch_force_frame").get_parameter_value().string_value
        self.feedback_wrench_sensor_frame = self.get_parameter("feedback_wrench_sensor_frame").get_parameter_value().string_value

        if self.teleop_mode not in ["unilateral", "bilateral"]:
            self.get_logger().error(f"無効な通信モードです: {self.teleop_mode}。終了します。")
            exit()
        self.get_logger().info(f"{self.teleop_mode} モードでテレオペレーションを開始します。")

    def _setup_ros_communications(self) -> None:
        """PublisherとSubscriberを初期化します。"""
        self.force_feedback_vis_pub = self.create_publisher(
            WrenchStamped, "touch_debug_force_feedback_rviz", 10
        )
        self.force_feedback_pub = self.create_publisher(
            TouchFeedback, self.force_feedback_topic, 10
        )
        self.touch_debug_action_pose_pub = self.create_publisher(
            PoseStamped, "touch_debug_action_pose_rviz", 10
        )
        self.pose_sub = self.create_subscription(
            PoseStamped, self.ee_pose_topic, self.pose_callback, qos_profile_sensor_data
        )
        # ボタンは変化したときしか配信されないので、取りこぼさないよう reliable で受ける（touch_state は reliable, depth 100）。
        # best-effort だと「離した」を落としたとき、押したままの扱いが続く
        self.button_sub = self.create_subscription(
            TouchButtonEvent, self.button_topic, self.button_callback,
            QoSProfile(depth=100, reliability=ReliabilityPolicy.RELIABLE)
        )

    def _setup_tf(self) -> None:
        """TF2のBufferとListenerを初期化します。"""
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

    def _setup_kdl(self) -> None:
        """KDLとstatic TF変換を初期化します。"""
        self.get_logger().info("KDL初期化を開始...")
        
        # KDLヘルパーの初期化
        try:
            # URDFの取得
            self.get_logger().info("URDFを取得中...")
            urdf_string = self._get_robot_urdf()
            if not urdf_string:
                self.get_logger().error("URDFの取得に失敗しました。KDLを無効化します。")
                self.kdl_helper = None
                self.pose_static_transform = None
                self.wrench_static_transform = None
                return
            
            # KDLヘルパーの初期化
            self.get_logger().info(f"KDLヘルパーを初期化中... base_link='{self.robot_base_frame}', ee_link='{self.feedback_wrench_sensor_frame}'")
            self.kdl_helper = KDLHelper(
                self.get_logger(),
                urdf_path=None,
                urdf_string=urdf_string,
                base_link=self.robot_base_frame,
                ee_link=self.feedback_wrench_sensor_frame  # tool0
            )
            self.get_logger().info(f"KDL初期化が完了しました: {self.kdl_helper._num_jnts} joints")
            
            # 静的TF変換の取得
            self.get_logger().info("静的TF変換を取得中...")
            self.pose_static_transform = self._get_transform(self.touch_base_frame, self.robot_base_frame)
            self.wrench_static_transform = self._get_transform(self.touch_force_frame, self.robot_base_frame)

            if self.pose_static_transform is not None:
                self.get_logger().info("姿勢計算用の静的TF変換の取得が完了しました")
            else:
                self.get_logger().warn("姿勢計算用の静的TF変換の取得に失敗しました")

            if self.wrench_static_transform is not None:
                self.get_logger().info("力覚計算用の静的TF変換の取得が完了しました")
            else:
                self.get_logger().warn("力覚計算用の静的TF変換の取得に失敗しました")
                
        except Exception as e:
            self.get_logger().error(f"KDL初期化に失敗: {e}")
            import traceback
            self.get_logger().error(f"詳細: {traceback.format_exc()}")
            self.kdl_helper = None
            self.pose_static_transform = None
            self.wrench_static_transform = None
            raise RuntimeError(f"Failed to initialize KDL in TouchAgent: {e}")

    def _log_topic_info(self) -> None:
        """Publisherとsubscriberのtopic一覧をログ出力します。"""
        self.get_logger().info("=== TouchAgent Topic Configuration ===")
        self.get_logger().info("  Parameters:")
        self.get_logger().info(f"  - teleoperation_mode: {self.teleop_mode}")
        self.get_logger().info(f"  - robot_base_frame: {self.robot_base_frame}")
        self.get_logger().info(f"  - touch_base_frame: {self.touch_base_frame}")
        self.get_logger().info(f"  - touch_force_frame: {self.touch_force_frame}")
        self.get_logger().info(f"  - feedback_wrench_sensor_frame: {self.feedback_wrench_sensor_frame}")
        self.get_logger().info("  Subscribers:")
        self.get_logger().info(f"  - Pose: {self.ee_pose_topic} (PoseStamped)")
        self.get_logger().info(f"  - Button: {self.button_topic} (TouchButtonEvent)")
        self.get_logger().info("  Publishers:")
        self.get_logger().info(f"  - Force Feedback: {self.force_feedback_topic} (TouchFeedback)")
        self.get_logger().info("  - Debug Publishers:")
        self.get_logger().info(f"  -  - Action Pose (RViz): /touch_debug_action_pose_rviz (PoseStamped)")
        self.get_logger().info(f"  -  - Force Feedback (RViz): /touch_debug_force_feedback_rviz (WrenchStamped)")
        self.get_logger().info("=======================================")

    def _get_robot_urdf(self) -> str:
        """ロボットのURDFを取得します。"""
        if self._robot_description:
            self.get_logger().info(f"URDF from constructor argument (length: {len(self._robot_description)} chars)")
            return self._robot_description
        try:
            # robot_descriptionパラメータから直接取得を試行
            self.get_logger().info("robot_descriptionパラメータから取得を試行...")
            urdf_string = self.get_parameter("robot_description").get_parameter_value().string_value
            if urdf_string and len(urdf_string) > 0:
                self.get_logger().info(f"URDFを取得しました（長さ: {len(urdf_string)} chars）")
                return urdf_string
            else:
                self.get_logger().warn("robot_descriptionパラメータが空です")
            
            # 外部パラメータから取得を試行
            self.get_logger().info("外部パラメータサーバーから取得を試行...")
            import subprocess
            result = subprocess.run(
                ['ros2', 'param', 'get', '/robot_state_publisher', 'robot_description'],
                capture_output=True, text=True, timeout=3.0
            )
            if result.returncode == 0:
                raw_output = result.stdout.strip()
                xml_start = max(raw_output.find('<?xml'), raw_output.find('<robot'))
                if xml_start != -1:
                    urdf_string = raw_output[xml_start:]
                    self.get_logger().info(f"外部URDFを取得しました（長さ: {len(urdf_string)} chars）")
                    return urdf_string
                else:
                    self.get_logger().warn("外部パラメータにXMLが見つかりませんでした")
            else:
                self.get_logger().warn(f"外部パラメータの取得に失敗: return code {result.returncode}")
        except Exception as e:
            self.get_logger().warn(f"URDF取得エラー: {e}")
        
        self.get_logger().error("すべてのURDF取得方法が失敗しました")
        return ""

    def _get_transform(self, target_frame: str, source_frame: str) -> np.ndarray | None:
        """指定されたフレーム間の静的TF変換を取得します。"""
        self.get_logger().info(f"静的TF変換を取得中: '{source_frame}' -> '{target_frame}'")
        
        start_time = time.time()
        while rclpy.ok():
            if time.time() - start_time > 5.0:
                self.get_logger().error(f"静的TF変換の取得がタイムアウトしました: '{source_frame}' -> '{target_frame}'")
                return None
            
            try:
                transform = self.tf_buffer.lookup_transform(
                    target_frame,
                    source_frame,
                    tf2_ros.Time(),
                    timeout=Duration(seconds=0.1)
                )
                self.get_logger().info(f"静的TF変換を取得しました: '{source_frame}' -> '{target_frame}'")
                return self._transform_to_matrix(transform)
                
            except tf2_ros.TransformException:
                rclpy.spin_once(self, timeout_sec=0.05)
        return None

    def _transform_to_matrix(self, transform: tf2_ros.TransformStamped) -> np.ndarray:
        """TransformStampedを4x4変換行列に変換します。"""
        t = transform.transform
        
        # 回転クォータニオンから回転行列を作成
        quat = [t.rotation.x, t.rotation.y, t.rotation.z, t.rotation.w]
        rotation_matrix = R.from_quat(quat).as_matrix()
        
        # 4x4同次変換行列を作成
        transform_matrix = np.eye(4)
        transform_matrix[:3, :3] = rotation_matrix
        transform_matrix[:3, 3] = [t.translation.x, t.translation.y, t.translation.z]
        
        return transform_matrix
                
    def _wait_for_first_pose(self) -> None:
        """指定したトピックから最初のPoseメッセージが届くまで待機します。"""
        self.get_logger().info(f"トピック '{self.ee_pose_topic}' からのメッセージを待機中...")
        start_time = time.time()
        while self._touch_current_pose is None and rclpy.ok():
            if time.time() - start_time > 5.0:
                self.get_logger().error(f"タイムアウト: トピック '{self.ee_pose_topic}' が利用できません。終了します。")
                exit()
            rclpy.spin_once(self, timeout_sec=0.1)
        self.get_logger().info("Touchデバイスのポーズを正常に受信しました。")

    def pose_callback(self, msg: PoseStamped) -> None:
        """Touchデバイスの姿勢を購読し、numpy配列として保存します。"""
        quat = np.array([msg.pose.orientation.x, msg.pose.orientation.y,
                        msg.pose.orientation.z, msg.pose.orientation.w])
        
        # 無効なquaternionの場合は、このメッセージを無視（コールバックを早期リターン）
        if np.linalg.norm(quat) < 1e-6:
            self.get_logger().warn("Touchデバイスからのquaternionが無効です。このメッセージを無視します。")
            return
        
        stamp_ns = stamp_to_ns(msg.header.stamp)
        if stamp_went_backwards(self._touch_stamp_ns, stamp_ns):
            self.get_logger().warn("Touch pose stamp went backwards; ignoring it.", throttle_duration_sec=5)
            return
        self._touch_received_ns = monotonic_ns()
        self._touch_stamp_ns = stamp_ns

        # 有効なquaternionを正規化
        quat = quat / np.linalg.norm(quat)
        self._touch_current_pose = np.array([
            msg.pose.position.x, msg.pose.position.y, msg.pose.position.z,
            quat[0], quat[1], quat[2], quat[3]
        ])
        self.get_logger().debug(f"Received Touch pose: {np.round(self._touch_current_pose, 3)}")

    def button_callback(self, msg: TouchButtonEvent) -> None:
        """Touchデバイスのボタン状態を購読し、フラグを更新します。"""
        if msg.white_button == 0:
            self._needs_rearm = False
        # Touch が途切れたあとは、いったん離して押し直すまで有効にしない
        self._is_teleop_active = (msg.white_button == 1) and not self._needs_rearm
        self._is_z_lock_active = (msg.grey_button == 1)
        self.get_logger().debug(f"Received Button event: white_button={msg.white_button}, grey_button={msg.grey_button}")
        
    def transform_wrench_kdl(
        self, wrench_array: np.ndarray, joint_positions: np.ndarray
    ) -> Tuple[WrenchStamped, TouchFeedback]:
        """
        KDLを使ってレンチ（力とトルク）をリアルタイムに変換し、フィードバックメッセージを作成します。
        
        変換チェーン: tool0 -> base_link (KDL) -> touch_force_frame (static TF)
        """
        # KDLまたは静的変換が利用できない場合はフォールバック
        if self.kdl_helper is None or self.wrench_static_transform is None:
            self.get_logger().warn("KDLまたはレンチ用のTF変換が利用できません。ゼロレンチを返します。")
            self.get_logger().debug(f"  - kdl_helper: {self.kdl_helper is not None}")
            self.get_logger().debug(f"  - wrench_static_transform: {self.wrench_static_transform is not None}")
            return self._create_zero_wrench_messages()
        
        try:
            # 1. 観測されたレンチをスケーリングし、最大値でクリッピング
            scaled_wrench = wrench_array * self.force_scale_to_touch
            force_in_tool = np.clip(scaled_wrench[:3], -self.touch_max_force, self.touch_max_force)
            torque_in_tool = np.clip(scaled_wrench[3:], -self.touch_max_force, self.touch_max_force)

            # 2. KDLでtool0姿勢を取得（base_link基準）
            tool_pose = self.kdl_helper.forward_kinematics(joint_positions.tolist())  # [x,y,z,qx,qy,qz,qw]
            
            # 3. tool0からbase_linkへの回転行列を取得
            tool_quat = tool_pose[3:]  # [qx, qy, qz, qw]
            tool_pos = tool_pose[:3]
            
            # FK はセンサの姿勢を base 座標で返すので、その回転行列はセンサ座標のベクトルを base 座標に移す
            #（v_base = R @ v_sensor）。転置するとセンサが回転しているときに力の向きが逆になる
            R_tool_to_base = R.from_quat(tool_quat).as_matrix()
            
            # 力覚の座標変換（回転のみ、位置による影響は無視）
            force_in_base = rotate_vector(R_tool_to_base, force_in_tool)
            torque_in_base = rotate_vector(R_tool_to_base, torque_in_tool)
            
            # 4. base_linkからtouch_force_frameへの静的変換を適用
            R_base_to_touch = self.wrench_static_transform[:3, :3]
            t_base_to_touch = self.wrench_static_transform[:3, 3]
            
            force_in_touch_frame = R_base_to_touch @ force_in_base
            # 軸ごとの clip は座標変換で崩れる（45° 回すと一軸が √2 倍になる）ので、
            # Touch に送る直前に合力の大きさを touch_max_force（N）以下にする
            force_in_touch_frame = limit_norm(force_in_touch_frame, self.touch_max_force)
            torque_in_touch_frame = R_base_to_touch @ torque_in_base + np.cross(t_base_to_touch, force_in_touch_frame)

            # 5. RViz可視化用メッセージ (robot_base_frame座標系) を作成
            wrench_vis_msg = WrenchStamped()
            wrench_vis_msg.header.stamp = self.get_clock().now().to_msg()
            wrench_vis_msg.header.frame_id = self.robot_base_frame
            wrench_vis_msg.wrench.force.x = float(force_in_base[0])
            wrench_vis_msg.wrench.force.y = float(force_in_base[1])
            wrench_vis_msg.wrench.force.z = float(force_in_base[2])
            wrench_vis_msg.wrench.torque.x = float(torque_in_base[0])
            wrench_vis_msg.wrench.torque.y = float(torque_in_base[1])
            wrench_vis_msg.wrench.torque.z = float(torque_in_base[2])

            # 6. Touchデバイスへのフィードバック用メッセージ (touch_force_frame座標系) を作成
            feedback_msg = TouchFeedback()
            feedback_msg.force.x = float(force_in_touch_frame[0])
            feedback_msg.force.y = float(force_in_touch_frame[1])
            feedback_msg.force.z = float(force_in_touch_frame[2])

            return wrench_vis_msg, feedback_msg
            
        except Exception as e:
            self.get_logger().error(f"KDL力覚変換エラー: {e}")
            return self._create_zero_wrench_messages()

    def require_rearm(self) -> None:
        """テレオペを止め、白ボタンを離して押し直すまで再開しないようにする（止める処理のあとに呼ぶ）。"""
        if self._is_teleop_active:
            self._needs_rearm = True
        self._is_teleop_active = False

    def zero_force(self) -> None:
        """Touch の力を 0 にする（Touch ドライバは最後に受け取った力を出し続けるため）。"""
        if not rclpy.ok():
            return
        _, zero_feedback = self._create_zero_wrench_messages()
        for _ in range(3):
            self.force_feedback_pub.publish(zero_feedback)
            time.sleep(0.01)

    def cleanup(self) -> None:
        """終了時に Touch の力を 0 にする。"""
        self.zero_force()

    def _create_zero_wrench_messages(self) -> Tuple[WrenchStamped, TouchFeedback]:
        """ゼロレンチのメッセージを作成します。"""
        wrench_vis_msg = WrenchStamped()
        wrench_vis_msg.header.stamp = self.get_clock().now().to_msg()
        wrench_vis_msg.header.frame_id = self.robot_base_frame
        
        feedback_msg = TouchFeedback()
        
        return wrench_vis_msg, feedback_msg
    
    def calculate_pose_difference(
        self, touch_start: np.ndarray, touch_current: np.ndarray, robot_start: np.ndarray
    ) -> np.ndarray:
        """
        Touchデバイスの移動量からロボットの目標姿勢を計算します。
        座標変換：touch_base_frame -> robot_base_frame
        """
        # Touchデバイス座標系での移動量（touchdiff）を計算
        touch_pos_diff = touch_current[:3] - touch_start[:3]
        touch_start_rot = R.from_quat(touch_start[3:])
        touch_current_rot = R.from_quat(touch_current[3:])
        touch_relative_rot = touch_current_rot * touch_start_rot.inv()
        
        # 静的変換行列が利用可能な場合は座標変換を行う
        if self.pose_static_transform is not None:
            # touch_base_frame -> robot_base_frameの変換行列を取得
            # static_transformは robot_base_frame -> touch_base_frame なので逆変換を使用
            robot_to_touch_transform = self.pose_static_transform
            touch_to_robot_transform = np.linalg.inv(robot_to_touch_transform)
            
            # 位置の変換（回転のみ適用、並進は加算しない）
            R_touch_to_robot = touch_to_robot_transform[:3, :3]
            robot_pos_diff = R_touch_to_robot @ touch_pos_diff
            
            # 回転の変換
            # Touch座標系の相対回転をロボット座標系に変換
            touch_relative_rot_matrix = touch_relative_rot.as_matrix()
            robot_relative_rot_matrix = R_touch_to_robot @ touch_relative_rot_matrix @ R_touch_to_robot.T
            robot_relative_rot = R.from_matrix(robot_relative_rot_matrix)
            
        else:
            # 座標変換なしで動かすと、Touch とロボットで軸の向きが違う方向に動くので、開始姿勢のまま止める
            self.get_logger().error(
                "touch_base -> robot_base の静的TFがないため、テレオペで動かしません", throttle_duration_sec=1.0)
            return np.asarray(robot_start, dtype=float).copy()
        
        # ロボットの新しい姿勢を計算
        robot_start_rot = R.from_quat(robot_start[3:])
        new_robot_rot = robot_relative_rot * robot_start_rot
        new_robot_pos = robot_start[:3] + robot_pos_diff
        
        return np.concatenate((new_robot_pos, new_robot_rot.as_quat()))

    def act(self, obs: Dict[str, np.ndarray], force_pose_update: bool = False) -> Dict:
        """
        観測(obs)に基づいて行動(action)を決定します。
        """
        current_ee_pose = np.concatenate((obs["ee_pos"], obs["ee_quat"]))
        released = False
        
        # 危険な入力データの検証
        if self._touch_current_pose is None:
            self.get_logger().warn("Touch device pose not available yet, using current robot pose")
            # Touchデバイスのポーズが利用できない場合は、テレオペを無効にする
            self._is_teleop_active = False
        touch_age_s = input_age_s(
            self.get_clock().now().nanoseconds, self._touch_received_ns, self._touch_stamp_ns)
        if touch_age_s > self.touch_timeout_s and self._is_teleop_active:
            # Touch ドライバが止まった・遅れている。最後の姿勢と押下状態で動かし続けないよう、
            # 離したのと同じ扱いで止め、押し直すまで再開しない
            self.get_logger().error(
                f"Touch pose is {touch_age_s:.3f} s old; stopping teleop until the white button is released and pressed again.")
            self._is_teleop_active = False
            self._needs_rearm = True
        if "joint_states" in obs.get("stale_sensors", ()) and self._is_teleop_active:
            # ロボットの計測が途切れた。戻ったときに押したままの状態で動き出さないよう、押し直すまで止める
            self.get_logger().error("Joint states are stale; stopping teleop until the white button is pressed again.")
            self.require_rearm()
        if self.pose_static_transform is None:
            # touch_base -> robot_base のTFがないと Touch の動きをロボット座標に移せない。
            # Z固定（灰ボタン）の姿勢上書きも含めて、テレオペを無効にする
            self.get_logger().error(
                "touch_base -> robot_base の静的TFがないため、テレオペを無効にしています", throttle_duration_sec=1.0)
            self._is_teleop_active = False
        
        # デバッグ: 入力観測の確認
        self.get_logger().debug(f"TouchAgent act() called:")
        self.get_logger().debug(f"  - current_ee_pose: {np.round(current_ee_pose, 3)}")
        self.get_logger().debug(f"  - joint_positions: {np.round(obs['joint_positions'], 3)}")
        self.get_logger().debug(f"  - teleop_active: {self._is_teleop_active}")
        self.get_logger().debug(f"  - touch_current_pose: {np.round(self._touch_current_pose, 3) if self._touch_current_pose is not None else None}")
        
        if self.teleop_mode == "bilateral":
            # KDLベースのリアルタイム力覚変換を使用
            self.get_logger().debug(f"  - ee_wrench: {np.round(obs['ee_wrench'], 3)}")
            if {"wrench", "joint_states"} & set(obs.get("stale_sensors", ())):
                # 古い力、または古い関節角での座標変換（向きを誤る）で返し続けないよう、途切れている間は 0 を返す
                wrench_vis, wrench_feedback = self._create_zero_wrench_messages()
            else:
                wrench_vis, wrench_feedback = self.transform_wrench_kdl(
                    obs["ee_wrench"],
                    obs["joint_positions"]
                )
            
            self.force_feedback_vis_pub.publish(wrench_vis)
            self.force_feedback_pub.publish(wrench_feedback)
            self.get_logger().debug(f"  - published force feedback: [{wrench_feedback.force.x:.3f}, {wrench_feedback.force.y:.3f}, {wrench_feedback.force.z:.3f}]")

        if self._is_teleop_active and not self._was_teleop_active:
            # ボタンを押した時の開始姿勢は、その時点の計測姿勢（停止中に覚えていた姿勢は古いことがある）
            self._robot_start_pose = current_ee_pose.copy()
            self._robot_start_pose_with_touch = self._touch_current_pose
            # self.get_logger().info("テレオペレーション開始")
        elif not self._is_teleop_active and self._was_teleop_active:
            # ボタンを離したら、その時点の計測姿勢で止める。最後に計算した目標（指令ガードで制限する前）を
            # 保つと、ロボットが追いついていない分だけ、離したあとも動き続ける
            self._robot_current_pose = current_ee_pose.copy()
            released = True
            # self.get_logger().info("テレオペレーション終了")
        
        self._was_teleop_active = self._is_teleop_active

        # 危険なゼロポーズを避けるため、現在の実際のロボット姿勢で初期化
        target_pose = current_ee_pose.copy()
        if self._is_teleop_active:
            self.get_logger().debug("  - Teleoperation ACTIVE")
            if self._robot_start_pose_with_touch is not None and self._robot_start_pose is not None:
                if self._is_z_lock_active:
                    self.get_logger().debug("  - Z-lock mode")
                    target_pose = self.calculate_pose_difference(
                        self._robot_start_pose_with_touch, self._touch_current_pose, self._robot_start_pose
                    )
                    target_pose[3:] = self.z_down_quat
                    self._last_target_pose = target_pose.copy()  # 計算した目標姿勢を保存
                else:
                    self.get_logger().debug("  - Free mode")
                    target_pose = self.calculate_pose_difference(
                        self._robot_start_pose_with_touch, self._touch_current_pose, self._robot_start_pose
                    )
                    self._last_target_pose = target_pose.copy()  # 計算した目標姿勢を保存
            else:
                self.get_logger().debug("  - Missing poses, using current/stored pose")
                target_pose = self._robot_current_pose if self._robot_current_pose is not None else current_ee_pose
        else:
            self.get_logger().debug("  - Teleoperation INACTIVE")
            # テレオペ非アクティブ時の姿勢維持ロジック - 必ず現在のロボットposeで初期化
            if self._robot_current_pose is None or not self._pose_initialized:
                if not np.allclose(current_ee_pose[:3], [0, 0, 0], atol=1e-6):
                    self._robot_current_pose = current_ee_pose.copy()  # 現在のロボットの実際のpose
                    self._pose_initialized = True
                    self.get_logger().debug(f"  - Initialized robot_current_pose with current robot pose: {np.round(self._robot_current_pose[:3], 3)}")
            elif force_pose_update:
                self._robot_current_pose = current_ee_pose.copy()
                self.get_logger().debug(f"  - Force updated robot_current_pose: {np.round(self._robot_current_pose[:3], 3)}")
            
            target_pose = self._robot_current_pose if self._robot_current_pose is not None else current_ee_pose

        target_quat = target_pose[3:]
        action_dict = {
            "joint_positions": np.zeros(len(obs["joint_positions"])),
            "ee_pos": target_pose[:3],
            "ee_quat": target_quat,
            "ee_rot_matrix": R.from_quat(target_quat).as_matrix(),
            "ee_euler": R.from_quat(target_quat).as_euler('xyz'),
            # 離した周期だけ、速度の上限を掛けずに計測姿勢で止める（RobotEnv.step が見る）
            "hold_measured": released,
            # テレオペ停止中は何も送らない。止めたときに送った目標を controller が保つ。
            # 停止中に覚えている姿勢（止めたときの観測、joint state が途切れていれば古い値）を送り続けると、
            # 止めたあとや通信の復帰後に、その姿勢へ戻る動きになる
            "send_command": self._is_teleop_active or released,
        }
        
        # デバッグ: 最終アクション結果を表示
        self.get_logger().debug(f"  - target_pose: pos={np.round(target_pose[:3], 3)}, quat={np.round(target_quat, 3)}")
        self.get_logger().debug(f"  - action_dict ee_pos: {np.round(action_dict['ee_pos'], 3)}")
        self.get_logger().debug(f"  - action_dict ee_quat: {np.round(action_dict['ee_quat'], 3)}")
        
        debug_pose_msg = PoseStamped()
        debug_pose_msg.header.stamp = self.get_clock().now().to_msg()
        debug_pose_msg.header.frame_id = self.robot_base_frame
        debug_pose_msg.pose.position.x = float(target_pose[0])
        debug_pose_msg.pose.position.y = float(target_pose[1])
        debug_pose_msg.pose.position.z = float(target_pose[2])
        debug_pose_msg.pose.orientation.x = float(target_quat[0])
        debug_pose_msg.pose.orientation.y = float(target_quat[1])
        debug_pose_msg.pose.orientation.z = float(target_quat[2])
        debug_pose_msg.pose.orientation.w = float(target_quat[3])
        self.touch_debug_action_pose_pub.publish(debug_pose_msg)
        
        return action_dict

def main(args=None):
    rclpy.init(args=args)
    touch_agent = TouchAgent()
    # このエージェントは外部ループから`act`が呼び出される想定
    touch_agent.get_logger().info("TouchAgent node has been initialized.")
    # rclpy.spin(touch_agent) # 独立して動作させる場合はspinが必要
    touch_agent.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()