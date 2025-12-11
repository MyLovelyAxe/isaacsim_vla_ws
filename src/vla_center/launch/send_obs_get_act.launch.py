from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():

    send_obs_1cam_node = Node(
        package='vla_center',
        executable='send_observation',
        name='send_observation',
    )

    send_obs_2cam_node = Node(
        package='vla_center',
        executable='send_observation_2cam',
        name='send_observation_2cam',
    )

    get_action_node = Node(
        package='vla_center',
        executable='get_action',
        name='get_action',
    )

    return LaunchDescription(
        [
            # send_obs_1cam_node,
            send_obs_2cam_node,
            get_action_node,
        ]
    )
