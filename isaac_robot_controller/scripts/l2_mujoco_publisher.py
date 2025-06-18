#!/usr/bin/env python

import rclpy
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
import numpy as np
from inverse_kinematics import inverse_kinematic  # Your IK function

SEQUENCE = np.array([
    [-0.11, -0.15, 0.15, 0.45, 0.75, 0.08],
    [-0.11, -0.15, 0.00, 0.45, 0.75, 0.08],
    [-0.11, -0.15, 0.00, 0.45, 0.75, 0.0],
    [-0.11, -0.15, 0.15, 0.45, 0.75, 0.0],
    [-0.11, 0.15, 0.15, 0.45, 0.75, 0.0],
])

class SmoothJointPublisher(Node):
    def __init__(self):
        super().__init__('smooth_joint_publisher')

        self.publisher = self.create_publisher(JointTrajectory, '/joint_command', 10)
        self.timer_period = 0.02  # 50 Hz
        self.timer = self.create_timer(self.timer_period, self.timer_callback)

        self.joint_names = [
            'left_hip_joint', 'left_knuckle_joint', 'left_ankle_joint',
            'right_hip_joint', 'right_knuckle_joint', 'right_ankle_joint'
        ]

        self.sequence = SEQUENCE
        self.index = 0
        self.pose_duration = 2.0  # seconds to hold each pose
        self.last_pose_time = self.get_clock().now()

        self.current_solution = inverse_kinematic(self.sequence[self.index], max_iter=100, tol=1e-5, alpha=0.5)

    def timer_callback(self):
        now = self.get_clock().now()
        elapsed = (now - self.last_pose_time).nanoseconds * 1e-9

        if elapsed >= self.pose_duration:
            self.index += 1
            if self.index >= len(self.sequence):
                rclpy.shutdown()
                return

            self.last_pose_time = now
            self.current_solution = inverse_kinematic(self.sequence[self.index], max_iter=100, tol=1e-5, alpha=0.5)

        if self.current_solution is None:
            self.get_logger().warn(f"IK failed for pose index {self.index}")
            return

        msg = JointTrajectory()
        msg.joint_names = self.joint_names

        point = JointTrajectoryPoint()
        point.positions = list(self.current_solution)
        point.time_from_start = rclpy.duration.Duration(seconds=0.0).to_msg()

        msg.points.append(point)
        self.publisher.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = SmoothJointPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
