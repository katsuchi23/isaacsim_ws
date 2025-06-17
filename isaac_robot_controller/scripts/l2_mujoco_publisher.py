#!/usr/bin/env python

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
import numpy as np
import time

class JointCommandPublisher(Node):
    def __init__(self):
        super().__init__('joint_command_publisher')

        self.publisher = self.create_publisher(
            Float64MultiArray,
            '/joint_command',
            10
        )

        # Simulation setup
        self.joint_count = 6
        self.target_position = np.array([0.5, -0.6, 0.3, -0.1, 0.0, 0.0])
        self.current_position = np.zeros(self.joint_count)
        self.current_velocity = np.zeros(self.joint_count)

        # PID gains
        self.Kp = 100.0
        self.Kd = 2.0
        self.Ki = 0.0
        self.integral_error = np.zeros(self.joint_count)

        # Timing
        self.time_step = 0.001
        self.last_time = time.time()

        # Timer for publishing at fixed rate
        self.timer = self.create_timer(self.time_step, self.control_loop)

        self.get_logger().info("JointCommandPublisher started.")

    def control_loop(self):
        # Time delta
        now = time.time()
        dt = now - self.last_time
        self.last_time = now
        if dt <= 0.0:
            dt = self.time_step  # fallback

        # Compute error terms
        error = self.target_position - self.current_position
        self.integral_error += error * dt
        derivative_error = -self.current_velocity

        # PID control
        cmd = (
            self.Kp * error +
            self.Ki * self.integral_error +
            self.Kd * derivative_error
        )

        # Integrate to get simulated current position/velocity
        self.current_velocity += cmd * dt
        self.current_position += self.current_velocity * dt

        # Publish command
        msg = Float64MultiArray()
        msg.data = cmd.tolist()
        self.publisher.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = JointCommandPublisher()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down JointCommandPublisher...')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
