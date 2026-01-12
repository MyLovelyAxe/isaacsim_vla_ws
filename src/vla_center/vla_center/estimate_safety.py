#!/usr/bin/env python3
import os
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from pathlib import Path
from datetime import datetime

import io
import zmq
import numpy as np
from collections import deque

from .robot_configs import SO100_JOINT_NAMES

# PS: must be the same with the trained model of safety estimator
HISTORY_LEN = 10


class VLAActionReceiverRecordNode(Node):
    def __init__(self):
        super().__init__("vla_action_receiver_record_node")

        ### Publish returned actions from VLA model ###

        # /joint_command: the topic that Isaac Sim gives target state to controller
        self.joint_pub = self.create_publisher(JointState, "/joint_command", 10)

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
        self.last_send = self.get_clock().now()
        self.get_logger().info("Ready to receive action commands ......")

        ### Subscribe the latest joint states ###

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

        ### Send history to safety estimator ###

        # deque as buffer to temporarily store the history info
        self.N = HISTORY_LEN
        self.hist_q = deque(maxlen=self.N)
        self.hist_a = deque(maxlen=self.N)

        # to publish history + proposed next action to safety estimator
        self.history_context = zmq.Context()
        self.history_socket = self.history_context.socket(zmq.PUB)
        self.history_socket.bind("tcp://127.0.0.1:5557")


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

        curr_ts = self.get_clock().now()

        ### Get latest messages ###

        # only send action when there are actions received from VLA model (i.e. zmq socket)
        try:
            action_msg_bytes = self.act_socket.recv(flags=zmq.NOBLOCK)
        except zmq.Again:
            if (
                (self.last_no_action_log_time is None)
                and ((curr_ts - self.last_no_action_log_time).nanoseconds > self.log_interval_sec * 1e9)
            ):
                self.get_logger().info("No action from VLA model received ......")
                self.last_no_action_log_time = curr_ts
            return

        # only continue when the current joint states exist
        if not self.joint_states_exist:
            if (
                (self.last_no_joint_log_time is None)
                and ((curr_ts - self.last_no_joint_log_time).nanoseconds > self.log_interval_sec * 1e9)
            ):
                self.get_logger().info("No joint states subscribed yet ......")
                self.last_no_joint_log_time = curr_ts
            return

        # get latest action
        new_a = np.frombuffer(action_msg_bytes, dtype=np.float32)

        ### Send to Isaac Sim controller ###

        if new_a.shape[0] != len(SO100_JOINT_NAMES):
            self.get_logger().warn(
                f"Received action of size {new_a.shape[0]}, "
                f"expected {len(SO100_JOINT_NAMES)}. Skipping."
            )
            return

        # send the action command as target state to /joint_command
        target_joint_states = JointState()
        target_joint_states.header.stamp = curr_ts.to_msg()
        target_joint_states.name = SO100_JOINT_NAMES
        target_joint_states.position = new_a.tolist()
        self.joint_pub.publish(target_joint_states)

        ### Send to safety estimator ###

        q = self.latest_joint_states.copy().astype(np.float32) # shape (6,)
        a = new_a.copy().astype(np.float32) # shape (6,)

        # if the history buffer is full, send it, otherwise only fill without sending
        if len(self.hist_q) == self.N:

            # pack everything into one binary blob using np.savez_compressed
            ts_msg = curr_ts.to_msg()
            timestamp = ts_msg.sec + ts_msg.nanosec * 1e-9
            history = self.pack_history(
                ts=timestamp, 
                q_history=np.stack(self.hist_q, axis=0), # shape (N,6)
                a_history=np.stack(self.hist_a, axis=0), # shape (N,6)
                prop_next_a=a, # shape (6,)
            )
            # send to safety estimator
            self.history_socket.send(history, flags=0)

        # fill the history buffer with new history
        self.hist_q.append(q)
        self.hist_a.append(a)


    def pack_history(
        self, 
        ts: float,
        q_history: np.ndarray,
        a_history: np.ndarray,
        prop_next_a: np.ndarray,
    ):
        """Pack data into a single bytes object using numpy's .npz format.
        
        :param ts: current timestamp, float64
        :param q_history: joint states in history window, shape (N, 6), float32
        :param a_history: the executed actions of robot arm, shape (N, 6), float32
        :param prop_next_a: the proposed next action from VLA, shape (6,), float32
        """

        buf = io.BytesIO()
        np.savez_compressed(
            buf,
            ts=np.array([ts], dtype=np.float64),
            joint_states_history=q_history,
            executed_actions_history=a_history,
            prop_next_actioin=prop_next_a,
        )
        return buf.getvalue()


def main(args=None):
    rclpy.init(args=args)
    node = VLAActionReceiverRecordNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
