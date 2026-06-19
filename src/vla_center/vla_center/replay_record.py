"""this node replays a recorded trajectory of exectuted actions, to check if the record is correct"""

#!/usr/bin/env python3
from pathlib import Path
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

import io
import zmq
import numpy as np
from collections import deque
from .robot_configs import SO100_JOINT_NAMES

# at runtime, __file__ = ~/isaacsim_vla_ws/src/vla_center/vla_center/replay_record.py
DEFAULT_RECORD_OUTPUT_PATH = Path(__file__).parent.parent.parent.parent.resolve() / "record"

# PS: must be the same with the trained model of safety estimator
HISTORY_LEN = 10 


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
        self.record = np.load(DEFAULT_RECORD_OUTPUT_PATH / npy_name, allow_pickle=True).item()
        if 'executed_actions' not in self.record or 'joint_states' not in self.record:
            self.get_logger().error("key 'executed_actions' or 'joint_states' is not in recorded .npy. Nothing to replay.")
            raise RuntimeError("Missing key: 'executed_actions' or 'joint_states'")
        
        self.len_record = len(self.record['executed_actions'])
        if self.len_record == 0:
            self.get_logger().error("The record is emplty. Nothing to replay.")
            raise RuntimeError("Empty record")
        self.replay_count = 0

        # /joint_command: the topic that Isaac Sim gives target state to controller
        self.joint_pub = self.create_publisher(JointState, "/joint_command", 10)

        # deque as buffer to temporarily store the history info
        self.N = HISTORY_LEN
        self.hist_q = deque(maxlen=self.N)
        self.hist_a = deque(maxlen=self.N)

        # to publish history + proposed next action to safety estimator
        self.history_context = zmq.Context()
        self.history_socket = self.history_context.socket(zmq.PUB)
        self.history_socket.bind("tcp://127.0.0.1:5557")

        # timer
        self.timer = self.create_timer(0.1, self.replay_action_callback)
        self.get_logger().info("Ready to replay the recorded action commands ......")


    def replay_action_callback(self):
        """Read the recorded npy file for action commands, publish to topic /joint_command."""
        
        # self.get_logger().info(f"Current q buffer: {self.hist_q}")
        # self.get_logger().info(f"Current a buffer: {self.hist_a}")

        if self.replay_count >= self.len_record:
            self.get_logger().info("Replay finished.")
            self.timer.cancel()
            rclpy.shutdown()
            return

        ### Load recorded history ###

        action_array = self.record['executed_actions'][self.replay_count]
        joints_array = self.record['joint_states'][self.replay_count]
        if isinstance(action_array, np.ndarray):
            action_array_list = action_array.tolist()

        ### Send to Isaac Sim controller ###

        # send the action command as target state to /joint_command
        curr_ts = self.get_clock().now().to_msg()
        target_joint_states = JointState()
        target_joint_states.header.stamp = curr_ts
        target_joint_states.name = SO100_JOINT_NAMES
        target_joint_states.position = action_array_list
        self.joint_pub.publish(target_joint_states)

        ### Send to safety estimator ###

        q = joints_array.copy().astype(np.float32) # shape (6,)
        a = action_array.copy().astype(np.float32) # shape (6,)

        # if the history buffer is full, send it, otherwise only fill without sending
        if len(self.hist_q) == self.N:

            # pack everything into one binary blob using np.savez_compressed
            timestamp = curr_ts.sec + curr_ts.nanosec * 1e-9
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

        self.replay_count += 1


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
        np.savez(
            buf,
            ts=np.array([ts], dtype=np.float64),
            joint_states_history=q_history,
            executed_actions_history=a_history,
            prop_next_actioin=prop_next_a,
        )
        return buf.getvalue()



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
