#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, JointState
from cv_bridge import CvBridge
from message_filters import Subscriber, ApproximateTimeSynchronizer

import io
import cv2
import zmq
import numpy as np

from .robot_configs import SO100_JOINT_NAMES


class VLAObservationSenderNode(Node):
    def __init__(self):
        super().__init__("vla_observation_sender_node")

        # to process images
        self.bridge = CvBridge()

        # to publish observation (images + joint states) to VLA model
        self.obs_context = zmq.Context()
        self.obs_socket = self.obs_context.socket(zmq.PUB)
        self.obs_socket.bind("tcp://127.0.0.1:5556")

        # /camera1_img: images from simulated camera
        self.cam1_sub = Subscriber(self, Image, "/camera1_img")

        # /joint_states: current joint states of robot arm
        self.joint_sub = Subscriber(self, JointState, "/joint_states")

        # to make sure the send observations have the same timestamp
        self.sync = ApproximateTimeSynchronizer(
            [self.cam1_sub, self.joint_sub],
            queue_size=10,
            slop=0.02,
        )
        self.sync.registerCallback(self.callback_observation)

        # to control the frequency of observation publishing
        self.last_send = self.get_clock().now()
        self.get_logger().info("Start to send image and joint states ......")


    # with 1 camera
    def callback_observation(
        self, 
        cam1_msg: Image, 
        joint_msg: JointState,
    ):
        """Receive the synchronized observation messages.
        
        :param cam1_msg: image from simulated camera 1
        :param joint_msg: current joint states massage of robot arm
        """

        now = self.get_clock().now()
        if (now - self.last_send).nanoseconds < 1e9 / 10:  # 10 Hz
            return
        self.last_send = now

        # timestamp (float seconds) from header
        timestamp = cam1_msg.header.stamp.sec + cam1_msg.header.stamp.nanosec * 1e-9

        # image: shape (H, W, 3) uint8
        # attention: here is BGR image, needs to convert to RGB on VLA side
        img1 = self.bridge.imgmsg_to_cv2(cam1_msg, desired_encoding="bgr8")

        # compress image to JPEG before packing
        ok, buf = cv2.imencode(".jpg", img1, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        if not ok:
            self.get_logger().warn("JPEG encoding failed, skipping frame")
            return
        img1_bytes = buf.tobytes()
        encoded = True

        # curr_joint_states: shape (6,) float32
        if joint_msg.name != SO100_JOINT_NAMES:
            self.get_logger().warn("The joint names don't match robot definition!")
            return 
        curr_joint_states = np.array(joint_msg.position, dtype=np.float32)

        # pack everything into one binary blob using np.savez_compressed
        observation = self.pack_observation(
            img1_bytes=img1_bytes, 
            joints=curr_joint_states,
            ts=timestamp, 
            encoded_flag=encoded,
        )

        # send to VLA model through ZMQ socket
        self.obs_socket.send(observation, flags=0)


    def pack_observation(
        self,
        img1_bytes,
        joints: np.ndarray,
        ts: float,
        encoded_flag: bool,
    ):
        """Pack data into a single bytes object using numpy's .npz format.
        
        :param img1_bytes: bytes of camera 1 image
        :param joints: current joint states of robot arm, shape (6,), float32
        :param ts: current timestamp
        :param encoded_flag: bool (True if img1_bytes is JPEG, False if raw)
        """
        buf = io.BytesIO()
        np.savez(
            buf,
            ts=np.array([ts], dtype=np.float64),
            joints=joints,
            img1=np.frombuffer(img1_bytes, dtype=np.uint8),
            img1_encoded=np.array([1 if encoded_flag else 0], dtype=np.int8),
        )
        return buf.getvalue()


def main(args=None):
    rclpy.init(args=args)
    node = VLAObservationSenderNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
