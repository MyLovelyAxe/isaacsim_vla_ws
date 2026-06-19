from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():

    send_obs_side_wrist_node = Node(
        package='vla_center',
        executable='send_obs_side_wrist',
        name='send_obs_side_wrist',
    )

    get_action_node = Node(
        package='vla_center',
        executable='get_action',
        name='get_action',
    )

    return LaunchDescription(
        [
            send_obs_side_wrist_node,
            get_action_node,
        ]
    )
