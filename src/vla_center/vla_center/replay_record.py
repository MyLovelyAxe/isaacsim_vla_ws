"""this node replays a recorded trajectory of exectuted actions, to check if the record is correct"""

#!/usr/bin/env python3
from pathlib import Path
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

import numpy as np
from .robot_configs import SO100_JOINT_NAMES

# at runtime, __file__ = ~/isaacsim_vla_ws/src/vla_center/vla_center/replay_record.py
DEFAULT_RECORD_OUTPUT_PATH = Path(__file__).parent.parent.parent.parent.resolve() / "record"

class ReplayRecordedActionsNode(Node):
    def __init__(self):
        super().__init__("replay_recorded_actions_node")

        # get the npy name to replay
        self.declare_parameter("npy_name", "") # PS: param name is defined in replay_record.launch.py
        npy_name = self.get_parameter("npy_name").value

        if not npy_name:
            self.get_logger().error("Parameter 'npy_name' is empty. Nothing to replay.")
            raise RuntimeError("Missing npy_name")

        self.get_logger().info(f"Loading record: {npy_name}")

        # load the recorded action trajectory
        npy_record = np.load(DEFAULT_RECORD_OUTPUT_PATH / npy_name, allow_pickle=True).item()
        if 'executed_actions' not in npy_record:
            self.get_logger().error("key 'executed_actions' is not in recorded .npy. Nothing to replay.")
            raise RuntimeError("Missing executed_actions key")
        self.record = npy_record['executed_actions']
        if len(self.record) == 0:
            self.get_logger().error("The 'executed_actions' is emplty. Nothing to replay.")
            raise RuntimeError("Empty executed_actions")
        self.replay_count = 0

        # /joint_command: the topic that Isaac Sim gives target state to controller
        self.joint_pub = self.create_publisher(JointState, "/joint_command", 10)

        # timer
        self.timer = self.create_timer(0.1, self.replay_action_callback)
        self.get_logger().info("Ready to replay the recorded action commands ......")


    def replay_action_callback(self):
        """Read the recorded npy file for action commands, publish to topic /joint_command."""
        
        if self.replay_count >= len(self.record):
            self.get_logger().info("Replay finished.")
            self.timer.cancel()
            rclpy.shutdown()
            return

        action_array = self.record[self.replay_count]
        if isinstance(action_array, np.ndarray):
            action_array = action_array.tolist()

        # send the action command as target state to /joint_command
        target_joint_states = JointState()
        target_joint_states.header.stamp = self.get_clock().now().to_msg()
        target_joint_states.name = SO100_JOINT_NAMES
        target_joint_states.position = action_array
        self.joint_pub.publish(target_joint_states)

        self.replay_count += 1


def main(args=None):
    rclpy.init(args=args)
    node = ReplayRecordedActionsNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
