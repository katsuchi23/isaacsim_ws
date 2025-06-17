#!/usr/bin/env python

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray

import mujoco
import mujoco.viewer
import numpy as np
import time

class MujocoSimNode(Node):
    def __init__(self):
        super().__init__('mujoco_sim_node')

        # Load MuJoCo model
        self.model = mujoco.MjModel.from_xml_path("/home/rey/isaacsim_ws/src/isaac_robot_description/urdf/l2.mjcf")
        self.data = mujoco.MjData(self.model)

        self.time_step = 0.001

        # Initialize control to zeros
        self.data.ctrl[:] = np.zeros(self.model.nu)

        # Subscriber for joint commands
        self.subscription = self.create_subscription(
            Float64MultiArray,
            '/joint_command',
            self.joint_command_callback,
            10
        )

        # Start the simulation loop
        self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        self.viewer.sync()

        self.get_logger().info("MuJoCo simulation started.")

        # Use timer for periodic stepping
        self.timer = self.create_timer(self.time_step, self.simulation_step)

    def joint_command_callback(self, msg):
        # Update data.ctrl safely
        if len(msg.data) != self.model.nu:
            self.get_logger().warn(f"Expected {self.model.nu} joints, got {len(msg.data)}.")
            return
        self.data.ctrl[:] = msg.data

    def simulation_step(self):
        # Step MuJoCo
        mujoco.mj_step(self.model, self.data)
        self.viewer.sync()

    def destroy_node(self):
        # Clean up viewer
        self.viewer.close()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = MujocoSimNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down MuJoCo simulation...')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
