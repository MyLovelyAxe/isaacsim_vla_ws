#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

import zmq
import numpy as np

from .robot_configs import (
    SO100_JOINT_NAMES, 
    SO100_SIGNS, 
    SO100_OFFSETS,
)


class VLAActionReceiverNode(Node):
    def __init__(self):
        super().__init__("vla_action_receiver_node")

        # /joint_command: the topic that Isaac Sim gives target state to controller
        self.joint_pub = self.create_publisher(JointState, "/joint_command", 10)

        # to receive returned action from VLA model
        self.act_context = zmq.Context()
        self.act_socket = self.act_context.socket(zmq.SUB)
        self.act_socket.setsockopt(zmq.SUBSCRIBE, b"") # subscribe to all message
        # TODO: only keep the latest one msg, necessary?
        # obs_socket.setsockopt(zmq.CONFLATE, 1)
        self.act_socket.RCVTIMEO = 0  # 0 ms timeout
        self.act_socket.connect("tcp://127.0.0.1:5555")

        # timer
        self.timer = self.create_timer(0.01, self.receive_action_callback)
        self.last_send = self.get_clock().now()
        self.get_logger().info("Ready to receive action commands ......")


    def receive_action_callback(self):
        """Receive returned action commands from VLA model, publish to topic /joint_command."""
        
        # # control how often to send action command
        # now = self.get_clock().now()
        # if (now - self.last_send).nanoseconds < 1e9 / 1:  # 10 Hz
        #     return
        # self.last_send = now

        try:
            msg_bytes = self.act_socket.recv(flags=zmq.NOBLOCK)
        except zmq.Again:
            return

        # decode received action message
        action_array = np.frombuffer(msg_bytes, dtype=np.float32)
        # self.get_logger().info(f"Received action: {action_array}")

        if action_array.shape[0] != len(SO100_JOINT_NAMES):
            self.get_logger().warn(
                f"Received action of size {action_array.shape[0]}, "
                f"expected {len(SO100_JOINT_NAMES)}. Skipping."
            )
            return

        # calibrate directions
        calib_action_array = SO100_SIGNS * action_array + SO100_OFFSETS
        # self.get_logger().info(f"Calibrated action: {calib_action_array}")

        # send the action command as target state to /joint_command
        target_joint_states = JointState()
        target_joint_states.header.stamp = self.get_clock().now().to_msg()
        target_joint_states.name = SO100_JOINT_NAMES
        target_joint_states.position = calib_action_array.tolist()
        self.joint_pub.publish(target_joint_states)
        # self.get_logger().info(f"Published joint command: {target_joint_states.position}")


def main(args=None):
    rclpy.init(args=args)
    node = VLAActionReceiverNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
