import mujoco
import mujoco.viewer
import numpy as np
from inverse_kinematics import inverse_kinematic  # Your IK function
import time

# Load your MJCF model
model = mujoco.MjModel.from_xml_path("/home/rey/isaacsim_ws/src/unitree_mujoco/unitree_robots/go2/scene_terrain.xml")
data = mujoco.MjData(model)
viewer = mujoco.viewer.launch_passive(model, data)
print(model.nu, model.nq, model.nv)

while True:
    mujoco.mj_step(model, data)
    viewer.sync()
    time.sleep(0.001)  # Adjust the sleep time as needed

