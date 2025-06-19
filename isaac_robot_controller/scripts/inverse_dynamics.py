import pinocchio as pin
import numpy as np

model = pin.buildModelFromUrdf("/home/delta/isaacsim_ws/src/isaac_robot_description/urdf/l2_clean.urdf")
data = model.createData()
print(model)
print("-------------------------------------------------")
print(data)

squat_pose = np.array([
    -0.8,  # left_hip_joint
    -1.0,   # left_knee_joint (knuckle)
    -0.6,  # left_ankle_joint
    -0.8,  # right_hip_joint
    -1.0,   # right_knee_joint
    -0.6,  # right_ankle_joint
])
q = np.array([0, 0, 1.0, 0, 0, 0, 1] + list(squat_pose))  # joint positions
v = np.zeros(model.nv)  # joint velocities
a = np.zeros(model.nv)  # joint accelerations

tau = pin.rnea(model, data, q, v, a)
print("Joint torques for squat pose:", tau)