from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():

    send_joint_state_node = Node(
        package='vla_center',
        executable='send_joint_state',
        name='send_joint_state',
    )

    get_action_node = Node(
        package='vla_center',
        executable='get_action',
        name='get_action',
    )

    return LaunchDescription(
        [
            send_joint_state_node,
            get_action_node,
        ]
    )
