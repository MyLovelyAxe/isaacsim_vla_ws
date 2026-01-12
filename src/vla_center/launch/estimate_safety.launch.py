from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():

    send_obs_2cam_node = Node(
        package='vla_center',
        executable='send_observation_2cam',
        name='send_observation_2cam',
    )

    estimate_safety_node = Node(
        package='vla_center',
        executable='estimate_safety',
        name='estimate_safety',
    )

    return LaunchDescription(
        [
            send_obs_2cam_node,
            estimate_safety_node,
        ]
    )
