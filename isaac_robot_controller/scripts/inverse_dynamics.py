import mujoco
import numpy as np
from inverse_kinematics import inverse_kinematic  # Your IK function

# Load your MJCF model
model = mujoco.MjModel.from_xml_path("/home/rey/isaacsim_ws/src/isaac_robot_description/urdf/l2.xml")
data = mujoco.MjData(model)
print(model.nq, model.nv)

# Set joint position (qpos), velocity (qvel), and acceleration (qacc)
# Example: squat_pose analog for your robot arm — customize this
SEQUENCE = np.array([-0.11, -0.15, 0.0, 0.45, 0.75, 0.08])
arm_pose = inverse_kinematic(
    SEQUENCE, 
    initial_guess=None, 
    max_iter=100, 
    tol=1e-5, 
    alpha=0.5,
    lower_limits=None,
    upper_limits=None
)
print("Inverse Kinematic Solution:", arm_pose)

qvel = np.zeros(model.nv)
qacc = np.zeros(model.nv)

# Assign to MuJoCo data
data.qpos[:len(arm_pose)] = arm_pose
data.qvel[:] = qvel
data.qacc[:] = qacc

# Run inverse dynamics
mujoco.mj_inverse(model, data)

# Print result
print("Joint torques:", data.qfrc_inverse)
