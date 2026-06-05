"""Start the custom Gazebo world and bridge services needed by the RL backend."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
from pathlib import Path


def generate_launch_description():
    package_share = Path(get_package_share_directory("uav_px4_rl"))
    world = str(package_share / "worlds" / "wire_training_world.sdf")
    gui = LaunchConfiguration("gui")
    bridge_lidar = LaunchConfiguration("bridge_lidar")
    lidar_topic = LaunchConfiguration("lidar_topic")

    gui_sim = ExecuteProcess(
        cmd=["gz", "sim", "-r", world],
        output="screen",
        condition=IfCondition(gui),
    )
    headless_sim = ExecuteProcess(
        cmd=["gz", "sim", "-s", "-r", world],
        output="screen",
        condition=UnlessCondition(gui),
    )
    bridge = Node(
        package="uav_px4_rl",
        executable="gz_harmonic_bridge",
        name="wire_world_bridge",
        arguments=["--world", "wire_training_world", "--timeout-ms", "1000"],
        output="screen",
    )
    lidar_bridge = Node(
        package="uav_px4_rl",
        executable="gz_pointcloud_bridge",
        name="x500_3d_lidar_bridge",
        arguments=["--gz-topic", "/x500/lidar/points", "--ros-topic", lidar_topic],
        output="screen",
        condition=IfCondition(bridge_lidar),
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "gui",
                default_value="false",
                description="Open Gazebo GUI for online evaluation or video recording.",
            ),
            DeclareLaunchArgument(
                "lidar_topic",
                default_value="/x500/lidar/points",
                description=(
                    "ROS 2 PointCloud2 topic used by the x500_3d_lidar Gazebo bridge."
                ),
            ),
            DeclareLaunchArgument(
                "bridge_lidar",
                default_value="true",
                description="Bridge the x500_3d_lidar Gazebo point cloud to ROS 2.",
            ),
            gui_sim,
            headless_sim,
            bridge,
            lidar_bridge,
        ]
    )
