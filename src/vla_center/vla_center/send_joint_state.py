#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

import io
import zmq
import numpy as np

from .robot_configs import SO100_JOINT_NAMES


class CurrentJointStateSenderNode(Node):
    def __init__(self):
        super().__init__("current_joint_state_sender_node")

        # to publish observation (images + joint states) to VLA model
        self.obs_context = zmq.Context()
        self.obs_socket = self.obs_context.socket(zmq.PUB)
        self.obs_socket.bind("tcp://127.0.0.1:5556")

        # /joint_states: current joint states of robot arm
        self.joint_sub = self.create_subscription(
            JointState, '/joint_states', self.joint_state_callback, 10,
        )

        # to control the frequency of observation publishing
        self.latest_joint_state_msg = None  # Store latest message
        self.timer = self.create_timer(0.005, self.send_joint_state) # 200Hz
        self.get_logger().info("Start to send joint states ......")


    def joint_state_callback(self, msg):
        """Store the latest joint state."""
        self.latest_joint_state_msg = msg


    def send_joint_state(self):
        """Send the latest joint state to socket."""

        if self.latest_joint_state_msg is not None:

            # timestamp (float seconds) from header
            timestamp = self.latest_joint_state_msg.header.stamp.sec + self.latest_joint_state_msg.header.stamp.nanosec * 1e-9

            # curr_joint_states: shape (6,) float32
            if self.latest_joint_state_msg.name != SO100_JOINT_NAMES:
                self.get_logger().warn("The joint names don't match robot definition!")
                return 
            curr_joint_states = np.array(self.latest_joint_state_msg.position, dtype=np.float32)

            # pack everything into one binary blob using np.savez_compressed
            observation = self.pack_observation(
                joints=curr_joint_states,
                ts=timestamp,
            )

            # send to VLA model through ZMQ socket
            self.obs_socket.send(observation, flags=0)

            self.latest_joint_state_msg = None


    def pack_observation(
        self,
        joints: np.ndarray,
        ts: float,
    ):
        """Pack data into a single bytes object using numpy's .npz format.
        
        :param joints: current joint states of robot arm, shape (6,), float32
        :param ts: current timestamp
        """
        buf = io.BytesIO()
        np.savez_compressed(
            buf,
            ts=np.array([ts], dtype=np.float64),
            joints=joints,
        )
        return buf.getvalue()


def main(args=None):
    rclpy.init(args=args)
    node = CurrentJointStateSenderNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
