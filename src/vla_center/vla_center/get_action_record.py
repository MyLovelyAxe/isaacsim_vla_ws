#!/usr/bin/env python3
import os
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from pathlib import Path
from datetime import datetime

import zmq
import numpy as np

from .robot_configs import (
    SO100_JOINT_NAMES, 
    SO100_SIGNS, 
    SO100_OFFSETS,
)

DEFAULT_RECORD_OUTPUT_PATH = Path(__file__).parent.parent.parent.parent.resolve() / "record"
RECORD_TIME = datetime.now().strftime("%Y%m%d_%H%M%S")

class VLAActionReceiverRecordNode(Node):
    def __init__(self):
        super().__init__("vla_action_receiver_record_node")

        ### publish returned actions from VLA model

        # /joint_command: the topic that Isaac Sim gives target state to controller
        self.executed_action_pub = self.create_publisher(JointState, "/joint_command", 10)

        # to receive returned action from VLA model
        self.act_context = zmq.Context()
        self.act_socket = self.act_context.socket(zmq.SUB)
        self.act_socket.connect("tcp://127.0.0.1:5555")
        self.act_socket.setsockopt(zmq.SUBSCRIBE, b"") # subscribe to all message
        self.act_socket.RCVTIMEO = 0  # 0 ms timeout

        # timer
        self.timer = self.create_timer(0.01, self.receive_action_callback)
        self.last_no_action_log_time = 0.0
        self.last_no_joint_log_time = 0.0
        self.log_interval_sec = 1.0  # print at most once per second
        self.clock_count = 0
        self.last_send = self.get_clock().now()
        self.get_logger().info("Ready to receive action commands ......")

        ### subscribe the latest joint states

        # /joint_states: the topic that contains the current joint states of robot arm
        self.curr_jointstates_sub = self.create_subscription(
            JointState, 
            "/joint_states", 
            self.current_joint_states_callback,
            10,
        )
        # fixed 6 joints for SO100 robot arm
        self.latest_joint_states = np.zeros((6), dtype=np.float32)
        self.latest_joint_states_time = np.zeros((1), dtype=np.float32)
        self.joint_states_exist = False

        ### record container

        self.record = {
            "time_idx": list(),
            "joint_states_time": list(),
            "joint_states": list(),
            "executed_actions_time": list(),
            "proposed_actions": list(),
            "executed_actions": list(),
        }
        os.makedirs(DEFAULT_RECORD_OUTPUT_PATH, exist_ok=True)


    def current_joint_states_callback(self, msg):
        """Always keep the latest joint states."""
        joint_states = np.asarray(msg.position, dtype=np.float32)
        if joint_states.shape[0] != len(SO100_JOINT_NAMES):
            return
        self.latest_joint_states = joint_states
        self.latest_joint_states_time = np.float32(msg.header.stamp.sec + 1e-9 * msg.header.stamp.nanosec)
        self.joint_states_exist = True


    def receive_action_callback(self):
        """Receive returned action commands from VLA model, publish to topic /joint_command."""

        now = self.get_clock().now()

        # only send action when there are actions received from VLA model (i.e. zmq socket)
        try:
            msg_bytes = self.act_socket.recv(flags=zmq.NOBLOCK)
        except zmq.Again:
            if (
                (self.last_no_action_log_time is None)
                and ((now - self.last_no_action_log_time).nanoseconds > self.log_interval_sec * 1e9)
            ):
                self.get_logger().info("No action from VLA model received ......")
                self.last_no_action_log_time = now
            return

        # only continue when the current joint states exist
        if not self.joint_states_exist:
            if (
                (self.last_no_joint_log_time is None)
                and ((now - self.last_no_joint_log_time).nanoseconds > self.log_interval_sec * 1e9)
            ):
                self.get_logger().info("No joint states subscribed yet ......")
                self.last_no_joint_log_time = now
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
        target_joint_states.header.stamp = now.to_msg()
        target_joint_states.name = SO100_JOINT_NAMES
        target_joint_states.position = calib_action_array.tolist()
        self.executed_action_pub.publish(target_joint_states) # TODO: update executed action based on risk score later
        # self.get_logger().info(f"Published joint command: {target_joint_states.position}")

        # update record content
        self.record["time_idx"].append(self.clock_count)
        self.record["joint_states_time"].append(self.latest_joint_states_time)
        self.record["joint_states"].append(self.latest_joint_states.astype(np.float32).copy())
        self.record["executed_actions_time"].append(
            np.float32(target_joint_states.header.stamp.sec + 1e-9 * target_joint_states.header.stamp.nanosec)
        )
        self.record["proposed_actions"].append(calib_action_array.astype(np.float32).copy())
        # TODO: later when triggering safety method, executed actions might be different from proposed action from VLA, but keep them the same for now
        self.record["executed_actions"].append(calib_action_array.astype(np.float32).copy()) 
        self.clock_count += 1

        # store record repetitively
        if self.clock_count % 100 == 0:
            # only store if it is not empty, all key-value pairs should be the same for being empty or not 
            if self.record["time_idx"]: 
                self.store_record()


    def store_record(self):
        """Store the current existing record on hard disk"""
        output_path = DEFAULT_RECORD_OUTPUT_PATH / f"{RECORD_TIME}.npy"
        np.save(output_path, self.record, allow_pickle=True)

def main(args=None):
    rclpy.init(args=args)
    node = VLAActionReceiverRecordNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node.record["time_idx"]: 
            node.store_record()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
