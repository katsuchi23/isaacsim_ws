import mujoco
import mujoco.viewer
import numpy as np
import time

# Load model and create data
model = mujoco.MjModel.from_xml_path("/home/rey/isaacsim_ws/src/isaac_robot_description/urdf/l2.mjcf")
data = mujoco.MjData(model)

time_step = 0.001
time_per_pose = 10

# PD gains
Kp = 100.0
Kd = 2.0

# Control sequence
sequence = [
    # Stand
    [0, 0, 0, 0, 0, 0],
    # Lift left leg
    [0.3, -0.6, 0.3, 0, 0, 0],
    # Swing left leg forward
    [0.5, -0.6, 0.3, -0.1, 0, 0],
    # Land left leg
    [0.5, 0, 0, -0.1, 0, 0],
]
cur_idx = 0

with mujoco.viewer.launch_passive(model, data) as viewer:
    viewer.sync()

    start_time = time.time()
    while time.time() - start_time < time_per_pose:
        start_time = time.time()


        # print(f"Current sequence index: {cur_idx}")

        # Control Logic
        for i in range(6):
            data.ctrl[i] = sequence[1][i]
        
        mujoco.mj_step(model, data)
        viewer.sync()


        elapsed = time.time() - start_time
        sleep_time = time_step - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)

print("Simulation finished.")
