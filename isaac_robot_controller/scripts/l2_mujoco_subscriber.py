#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

import mujoco
import mujoco.viewer
import numpy as np
import time

class MujocoSimNode(Node):
    def __init__(self):
        super().__init__('mujoco_sim_node')

        # Load MuJoCo model
        self.model = mujoco.MjModel.from_xml_path("/home/rey/isaacsim_ws/src/isaac_robot_description/urdf/l2.xml")
        self.data = mujoco.MjData(self.model)

        self.time_step = 0.001
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
            JointState,
            '/joint_command',
            self.joint_state_callback,
            10
        )

        # Start the viewer
        self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        self.viewer.sync()
        self.get_logger().info("MuJoCo simulation started with JointState interface.")

        # Timer for stepping
        self.timer = self.create_timer(self.time_step, self.simulation_step)

    def joint_state_callback(self, msg: JointState):
        if not msg:
            self.get_logger().warn("Received empty joint state message.")
            return

        ctrl = np.zeros(self.model.nu)
        for joint_name, effort in zip(msg.name, msg.effort):
            if joint_name not in self.joint_name_to_index:
                self.get_logger().warn(f"Unknown joint name: {joint_name}")
                continue
            idx = self.joint_name_to_index[joint_name]
            ctrl[idx] = effort
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
