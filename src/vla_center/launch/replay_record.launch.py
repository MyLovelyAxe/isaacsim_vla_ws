import os
from glob import glob
from pathlib import Path
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


# at runtime, __file__ = ~/isaacsim_vla_ws/install/vla_center/share/vla_center/launch/replay_record.launch.py
DEFAULT_RECORD_OUTPUT_PATH = Path(__file__).parent.parent.parent.parent.parent.parent.resolve() / "record"


def generate_launch_description():

    npy_files = sorted(DEFAULT_RECORD_OUTPUT_PATH.glob("*.npy"))

    if len(npy_files) == 0:
        return LaunchDescription([
            LogInfo(msg="No recorded .npy files found in /record. Replay node not started.")
        ])

    launch_arg = DeclareLaunchArgument(
        "npy_name",
        default_value=str(npy_files[0]),
        description="Path to the recorded .npy file"
    )

    replay_record_node = Node(
        package="vla_center",
        executable="replay_record",
        name="replay_record",
        parameters=[{
            "npy_name": LaunchConfiguration("npy_name")
        }],
    )

    return LaunchDescription([
        launch_arg,
        replay_record_node,
    ])
