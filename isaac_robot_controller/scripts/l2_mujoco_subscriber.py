#!/usr/bin/env python

import rclpy
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory

import mujoco
import mujoco.viewer
import numpy as np
import time

class MujocoSimNode(Node):
    def __init__(self):
        super().__init__('mujoco_sim_node')

        # Load MuJoCo model
        self.model = mujoco.MjModel.from_xml_path("/home/rey/isaacsim_ws/src/mujoco_menagerie/unitree_g1/scene.xml")
        self.data = mujoco.MjData(self.model)

        self.time_step = 0.002
        self.data.ctrl[:] = np.zeros(self.model.nu)

        # Define joint name to index mapping
        self.joint_name_to_index = {
            'left_hip_joint': 0,
            'left_knuckle_joint': 1,
            'left_ankle_joint': 2,
            'right_hip_joint': 3,
            'right_knuckle_joint': 4,
            'right_ankle_joint': 5,
        }

        # Subscribe to JointTrajectory
        self.subscription = self.create_subscription(
            JointTrajectory,
            '/joint_command',
            self.trajectory_callback,
            10
        )

        # Start the viewer
        self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        self.viewer.sync()
        self.get_logger().info("MuJoCo simulation started with JointCommand interface.")

        # Timer for stepping
        self.timer = self.create_timer(self.time_step, self.simulation_step)

    def trajectory_callback(self, msg: JointTrajectory):
        if not msg.points:
            self.get_logger().warn("Received empty trajectory message.")
            return

        point = msg.points[0]
        positions = point.positions

        if len(msg.joint_names) != len(positions):
            self.get_logger().warn("Mismatch between joint names and positions.")
            return

        ctrl = np.zeros(self.model.nu)
        for joint_name, position in zip(msg.joint_names, positions):
            if joint_name not in self.joint_name_to_index:
                self.get_logger().warn(f"Unknown joint name: {joint_name}")
                continue
            idx = self.joint_name_to_index[joint_name]
            ctrl[idx] = position

        self.data.ctrl[:] = ctrl

    def simulation_step(self):
        mujoco.mj_step(self.model, self.data)
        self.viewer.sync()

    def destroy_node(self):
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
