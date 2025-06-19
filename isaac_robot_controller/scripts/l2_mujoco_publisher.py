#!/usr/bin/env python3

from xml.parsers.expat import model
import rclpy
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from sensor_msgs.msg import JointState
import numpy as np
from inverse_kinematics import inverse_kinematic  # Your IK function
import pinocchio as pin

SEQUENCE = np.array(
    [[0, 0, 0, 0, 0, 0],
     [-0.8, -1.0, -0.6, 0.8, -1.0, 0.6]])

class SmoothJointPublisher(Node):
    def __init__(self):
        super().__init__('smooth_joint_publisher')

        self.model = pin.buildModelFromUrdf("/home/delta/isaacsim_ws/src/isaac_robot_description/urdf/l2_clean.urdf")
        self.data = self.model.createData()
        self.model.gravity.linear = np.array([0, 0, -9.81])  # gravity vector in world frame
        # print(self.model, self.data)

        self.publisher = self.create_publisher(JointState, '/joint_command', 10)
        self.timer_period = 0.02  # 50 Hz
        self.timer = self.create_timer(self.timer_period, self.timer_callback)

        self.joint_names = [
            'left_hip_joint', 'left_knuckle_joint', 'left_ankle_joint',
            'right_hip_joint', 'right_knuckle_joint', 'right_ankle_joint'
        ]

        self.sequence = SEQUENCE
        self.index = 1
        self.pose_duration = 9999999.0  # seconds to hold each pose
        self.last_pose_time = self.get_clock().now()
        self.current_solution = np.zeros(6)

    def timer_callback(self):
        now = self.get_clock().now()
        elapsed = (now - self.last_pose_time).nanoseconds * 1e-9

        if elapsed >= self.pose_duration:
            self.index += 1
            if self.index >= len(self.sequence):
                self.get_logger().info("Sequence completed.")
                rclpy.shutdown()
                return

            self.last_pose_time = now

        q = self.sequence[self.index]
        v = np.zeros(self.model.nv)
        a = np.zeros(self.model.nv)

        tau = pin.rnea(self.model, self.data, q, v, a)
        self.get_logger().info(f"Computed torque (rnea) for pose {self.index}: {tau}")
        self.current_solution = list(tau)

        if self.current_solution is None:
            self.get_logger().warn(f"Torque computation failed for pose index {self.index}")
            return

        # Publish torque as effort via JointState message
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = self.joint_names
        msg.effort = self.current_solution  # torque command

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
