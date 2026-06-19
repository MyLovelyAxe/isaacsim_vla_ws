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


class VLAObservationSenderWristSideNode(Node):
    def __init__(self):
        super().__init__("vla_observation_sender_wrist_side_node")

        # to process images
        self.bridge = CvBridge()

        # to publish observation (images + joint states) to VLA model
        self.obs_context = zmq.Context()
        self.obs_socket = self.obs_context.socket(zmq.PUB)
        self.obs_socket.bind("tcp://127.0.0.1:5556")

        # images from multiple simulated cameras
        self.side_cam_sub = Subscriber(self, Image, "/camera2_img") # side camera
        self.wrist_cam_sub = Subscriber(self, Image, "/camera3_img") # wrist camera

        # /joint_states: current joint states of robot arm
        self.joint_sub = Subscriber(self, JointState, "/joint_states")

        # to make sure the send observations have the same timestamp
        self.sync = ApproximateTimeSynchronizer(
            [self.side_cam_sub, self.wrist_cam_sub, self.joint_sub],
            queue_size=10,
            slop=0.02,
        )
        # self.sync.registerCallback(self.callback_1cam)
        self.sync.registerCallback(self.callback_observation)

        self.get_logger().info("Start to send image and joint states ......")


    # 2 cameras
    def callback_observation(        
        self, 
        side_cam_msg: Image, 
        wrist_cam_msg: Image, 
        joint_msg: JointState,
    ):
        """Receive the synchronized observation messages.
        
        :param side_cam_msg: image from simulated camera 1
        :param wrist_cam_msg: image from simulated camera 2
        :param joint_msg: current joint states massage of robot arm
        """        

        # timestamp (float seconds) from header
        timestamp = side_cam_msg.header.stamp.sec + side_cam_msg.header.stamp.nanosec * 1e-9

        # image: shape (H, W, 3) uint8
        # attention: here is BGR image, needs to convert to RGB on VLA side
        side_img = self.bridge.imgmsg_to_cv2(side_cam_msg, desired_encoding="bgr8")
        wrist_img = self.bridge.imgmsg_to_cv2(wrist_cam_msg, desired_encoding="bgr8")

        # compress image to JPEG before packing
        ok1, side_buf = cv2.imencode(".jpg", side_img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        ok2, wrist_buf = cv2.imencode(".jpg", wrist_img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        if not (ok1 and ok2):
            self.get_logger().warn("JPEG encoding failed, skipping frame")
            return
        side_img_bytes = side_buf.tobytes()
        wrist_img_bytes = wrist_buf.tobytes()
        encoded = True

        # curr_joint_states: shape (6,) float32
        if joint_msg.name != SO100_JOINT_NAMES:
            self.get_logger().warn("The joint names don't match robot definition!")
            return 
        curr_joint_states = np.array(joint_msg.position, dtype=np.float32)

        # pack everything into one binary blob using numpy's .npz format
        observation = self.pack_observation(
            side_img_bytes=side_img_bytes, 
            wrist_img_bytes=wrist_img_bytes, 
            joints=curr_joint_states,
            ts=timestamp, 
            encoded_flag=encoded,
        )

        # send to VLA model through ZMQ socket
        self.obs_socket.send(observation, flags=0)


    def pack_observation(
        self, 
        side_img_bytes, 
        wrist_img_bytes,
        joints: np.ndarray,
        ts: float,
        encoded_flag: bool,
    ):
        """Pack data into a single bytes object using numpy's .npz format.
        
        :param side_img_bytes & side_img_bytes: bytes of side and wrist camera images
        :param joints: current joint states of robot arm, shape (6,), float32
        :param ts: current timestamp
        :param encoded_flag: bool (True if side_img_bytes is JPEG, False if raw)
        """
        buf = io.BytesIO()
        np.savez_compressed(
            buf,
            ts=np.array([ts], dtype=np.float64),
            joints=joints,
            side_img=np.frombuffer(side_img_bytes, dtype=np.uint8),
            side_img_encoded=np.array([1 if encoded_flag else 0], dtype=np.int8),
            wrist_img=np.frombuffer(wrist_img_bytes, dtype=np.uint8),
            wrist_img_encoded=np.array([1 if encoded_flag else 0], dtype=np.int8),
        )
        return buf.getvalue()


def main(args=None):
    rclpy.init(args=args)
    node = VLAObservationSenderWristSideNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
