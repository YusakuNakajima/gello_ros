
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration, Command, PathJoinSubstitution, FindExecutable
from launch_ros.actions import Node, ComposableNodeContainer
from launch_ros.descriptions import ComposableNode
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare

def launch_setup(context, *args, **kwargs):
    """Launch setup function to handle parameters dynamically"""
    
    # Get launch arguments
    robot_config = LaunchConfiguration('robot_config').perform(context)
    common_config = LaunchConfiguration('common_config').perform(context)
    gello_config = LaunchConfiguration('gello_config').perform(context)
    touch_config = LaunchConfiguration('touch_config').perform(context)
    save_episode = LaunchConfiguration('save_episode').perform(context) == 'true'
    agent_type = LaunchConfiguration('agent_type').perform(context)
    skip_initial_move = LaunchConfiguration('skip_initial_move').perform(context) == 'true'
    controller_type = LaunchConfiguration('controller_type').perform(context)
    
    # Robot description generation for Franka FR3
    # KDL で FK を計算するためのモデル。config/franka_fr3.yaml の ee_link（pestle_tip）と
    # feedback_wrench_sensor_frame（leptrino_frame）を含む、onolab_ros2 のペストル + Leptrino 構成を使う。
    # ロボット側は onolab_robot_bringup の franka_fr3_bringup_with_pestle.launch.py で起動すること
    robot_description_content = Command(
        [
            PathJoinSubstitution([FindExecutable(name="xacro")]),
            " ",
            PathJoinSubstitution([FindPackageShare("onolab_robot_description"), "robots", "fr3", "fr3_with_pestle_and_leptrino.urdf.xacro"]),
        ]
    )
    
    # Create agent node with component mode enabled
    agent_node = Node(
        package='gello_ros',
        executable='run_agent_component.py',
        name='agent_node_franka_component',
        output='screen',
        parameters=[
            robot_config,
            common_config,
            {
                'robot_description': ParameterValue(value=robot_description_content, value_type=str),
                'save_episode': save_episode,
                'agent_type': agent_type,
                'skip_initial_move': skip_initial_move,
                'controller_type': controller_type,
                'component_mode': True,  # Enable component mode
            }
        ]
    )
    
    # Alternative: Use ComposableNodeContainer for true component management
    # This creates a container that can manage multiple components
    component_container = ComposableNodeContainer(
        name='franka_agent_container',
        namespace='',
        package='rclcpp_components',
        executable='component_container',
        composable_node_descriptions=[],
        output='screen',
    )
    
    return [agent_node, component_container]

def generate_launch_description():
    gello_ros_share_dir = get_package_share_directory('gello_ros')

    return LaunchDescription([
        DeclareLaunchArgument(
            'robot_config',
            default_value=os.path.join(gello_ros_share_dir, 'config', 'franka_fr3.yaml'),
            description='Path to the robot config file'),
        DeclareLaunchArgument(
            'common_config',
            default_value=os.path.join(gello_ros_share_dir, 'config', 'common.yaml'),
            description='Path to the common config file'),
        DeclareLaunchArgument(
            'gello_config',
            default_value=os.path.join(gello_ros_share_dir, 'config', 'gello.yaml'),
            description='Path to the gello config file'),
        DeclareLaunchArgument(
            'touch_config',
            default_value=os.path.join(gello_ros_share_dir, 'config', 'touch.yaml'),
            description='Path to the touch config file'),
        DeclareLaunchArgument(
            'save_episode',
            default_value='false',
            description='Whether to save the episode'),
        DeclareLaunchArgument(
            'agent_type',
            default_value='touch',
            description='Type of agent to use'),
        DeclareLaunchArgument(
            'skip_initial_move',
            default_value='true',
            description='Whether to skip the initial move'),
        DeclareLaunchArgument(
            'controller_type',
            default_value='cartesian_impedance_controller',
            description='Controller type'),

        # Launch setup function to handle dynamic parameter loading
        OpaqueFunction(function=launch_setup)
    ])
