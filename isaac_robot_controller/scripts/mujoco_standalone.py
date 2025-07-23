import mujoco
import mujoco.viewer
import numpy as np
import time

# --- Setup MuJoCo ---1
SCENE_PATH = "/home/rey/isaacsim_ws/src/mujoco_menagerie/unitree_go2/scene.xml"
model = mujoco.MjModel.from_xml_path(SCENE_PATH)
data = mujoco.MjData(model)
desired_data = mujoco.MjData(model)
viewer = mujoco.viewer.launch_passive(model, data)  # passive mode

# --- Parameters ---
ADD_OBSTACLE = True  # Set to False if you don't want to launch obstacles
DT = 0.001  # simulation timestep
DURATION = 2.0  # total duration seconds
NUM_STEPS = int(DURATION / DT)
free_joint_pos = 7  # number of free joint positions in the robot
free_joint_vel = 6  # number of free joints in the robot
robot_position_dof = 19
robot_velocity_dof = 18
robot_actuated_dof = 12
launch_interval = 1.5
last_launch_time = 0.0  # last time an obstacle was launched
contact_wrench = {"FL": np.zeros(6), # 20
                  "FR": np.zeros(6), # 32
                  "RL": np.zeros(6), # 44
                  "RR": np.zeros(6)} # 56
jacobian = {"FL": np.zeros((6, model.nv)),
            "FR": np.zeros((6, model.nv)),
            "RL": np.zeros((6, model.nv)),
            "RR": np.zeros((6, model.nv))}
# PD gains for feedback control
Kp = 500.0
Kd = 5.0
i = 0


data.qpos[:robot_position_dof] = np.array([
    -8.67742173e-02, -8.81724810e-04,  7.71154443e-02,  9.98647439e-01,
    7.79675628e-05, -5.19777230e-02,  1.26633295e-03,  4.36252548e-02,
    1.26797972e+00, -2.72385334e+00, -4.50877639e-02,  1.26724602e+00,
    -2.72370792e+00,  4.00327695e-01,  1.29830087e+00, -2.72448199e+00,
    -4.00274137e-01,  1.29837385e+00, -2.72449991e+00
])

# --- Generate smooth trajectory ---
def minimum_jerk_trajectory(q_start, q_end, duration, t):
    t = np.clip(t, 0, duration)
    s = t / duration
    
    # Minimum jerk polynomial
    p = 10*s**3 - 15*s**4 + 6*s**5
    v = (30*s**2 - 60*s**3 + 30*s**4) / duration
    a = (60*s - 180*s**2 + 120*s**3) / (duration**2)
    
    # Interpolate positions linearly by p
    q_des = q_start + p * (q_end - q_start)
    
    # Velocity vector (length 18):
    # For free joint position, skip quaternion scalar (index 3)
    # Use simple finite difference for orientation angular velocity approx:
    lin_vel = v * (q_end[0:3] - q_start[0:3])
    ang_vel = v * (q_end[4:7] - q_start[4:7])  # ignoring scalar at index 3
    motor_vel = v * (q_end[7:] - q_start[7:])
    
    v_des = np.zeros(18)
    v_des[0:3] = lin_vel
    v_des[3:6] = ang_vel
    v_des[6:] = motor_vel
    
    # Acceleration vector (length 18)
    lin_acc = a * (q_end[0:3] - q_start[0:3])
    ang_acc = a * (q_end[4:7] - q_start[4:7])
    motor_acc = a * (q_end[7:] - q_start[7:])
    
    a_des = np.zeros(18)
    a_des[0:3] = lin_acc
    a_des[3:6] = ang_acc
    a_des[6:] = motor_acc
    
    return q_des, v_des, a_des

def launch_obstacle(model, data, speed=3.0):
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "flying_obstacle")
    if body_id < 0:
        print("❌ Obstacle body not found.")
        return

    # Get obstacle joint address (for qpos and qvel)
    jnt_adr = model.jnt_qposadr[model.body_jntadr[body_id]]

    # Set starting position (e.g., above the robot, offset in +x)
    obstacle_pos = np.random.uniform([-0.5, -0.5, 0.0], [0.5, 0.5, 1.0])  # starting xyz
    data.qpos[jnt_adr + 0 : jnt_adr + 3] = obstacle_pos  # set xyz

    # Set identity quaternion for orientation (w=1, x=y=z=0)
    data.qpos[jnt_adr + 3 : jnt_adr + 7] = np.array([1.0, 0.0, 0.0, 0.0])

    # Target is the robot base position (first 3 elements of qpos)
    robot_pos = data.qpos[:3]
    
    # Compute direction vector
    direction = robot_pos - obstacle_pos
    direction[2] += 0.2
    direction = direction / np.linalg.norm(direction)

    # Set linear velocity (first 3 dof of qvel for a freejoint)
    dof_adr = model.jnt_dofadr[model.body_jntadr[body_id]]
    data.qvel[dof_adr + 0 : dof_adr + 3] = direction * speed

    # Zero angular velocity (last 3 dof of freejoint)
    data.qvel[dof_adr + 3 : dof_adr + 6] = 0.0

def get_wrench(model, data): # 15, 16, 17, 18 (left), 30, 31, 32 , 33 (right)
    global contact_wrench
    for j in range(data.ncon):

        contact = data.contact[j]
        f_contact = np.zeros(6)  # [fx, fy, fz, tx, ty, tz]
        mujoco.mj_contactForce(model, data, j, f_contact)

        if contact.geom1 == 0:  # Check if geom1 is the ground
            if contact.geom2 == 20:
                contact_wrench['FL'] = f_contact
            elif contact.geom2 == 32:
                contact_wrench['FR'] = f_contact
            elif contact.geom2 == 44:
                contact_wrench['RL'] = f_contact
            elif contact.geom2 == 56:
                contact_wrench['RR'] = f_contact

        # geom1_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1)
        # geom2_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2)  
        # print(f"Contact between {geom1_name}, id: {contact.geom1} and {geom2_name}, id: {contact.geom2}")

def get_foot_jacobian(model, data, site_name):
    # Allocate space for Jacobians (3 x nv)
    Jp = np.zeros((3, model.nv))  # translational part
    Jr = np.zeros((3, model.nv))  # rotational part

    # Correct way to get site id
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    # print(f"Site ID for {site_name}: {site_id}")

    # Now compute the Jacobian for the site
    mujoco.mj_jacSite(model, data, Jp, Jr, site_id)

    # Combine linear and rotational Jacobian (6 x nv)
    return np.vstack((Jp, Jr))

def get_robot_jacobian(model, data):
    global jacobian
    # Get the robot base body ID
    jacobian["FL"] = get_foot_jacobian(model, data, "FL_site")
    jacobian["FR"] = get_foot_jacobian(model, data, "FR_site")
    jacobian["RL"] = get_foot_jacobian(model, data, "RL_site")
    jacobian["RR"] = get_foot_jacobian(model, data, "RR_site")

def calculate_tau_jump(front_left_wrench, front_right_wrench, rear_left_wrench, rear_right_wrench, front_left_jacobian, front_right_jacobian, rear_left_jacobian, rear_right_jacobian, upward_force=100.0):
    # Calculate the total wrench required to achieve the desired upward force
    total_wrench = np.array([0.0, 0.0, upward_force, 0.0, 0.0, 0.0])
    each_leg_wrench = total_wrench / 4 # distribute equally among the four legs

    # front_left_required_wrench = front_left_wrench
    # front_left_required_wrench[2] = each_leg_wrench[2]  # adjust z-force
    # front_right_required_wrench = front_right_wrench
    # front_right_required_wrench[2] = each_leg_wrench[2]  # adjust z-force
    # rear_left_required_wrench = rear_left_wrench
    # rear_left_required_wrench[2] = each_leg_wrench[2] # adjust z-force
    # rear_right_required_wrench = rear_right_wrench
    # rear_right_required_wrench[2] = each_leg_wrench[2]  # adjust z-force

    front_left_required_wrench = each_leg_wrench
    front_right_required_wrench = each_leg_wrench
    rear_left_required_wrench = each_leg_wrench
    rear_right_required_wrench = each_leg_wrench

    # Combine the Jacobians into a single matrix
    front_left_tau = front_left_jacobian.T @ front_left_required_wrench
    front_right_tau = front_right_jacobian.T @ front_right_required_wrench
    rear_left_tau = rear_left_jacobian.T @ rear_left_required_wrench
    rear_right_tau = rear_right_jacobian.T @ rear_right_required_wrench

    return front_left_tau + front_right_tau + rear_left_tau + rear_right_tau

# Initialize the simulation loop
for i in range(50):
    mujoco.mj_step(model, data)
    viewer.sync()
    time.sleep(DT)

    # total_mass = np.sum(model.body_mass)
    # print(total_mass * model.opt.gravity)

    # in_contact = False
    # for c in data.contact[:data.ncon]:
    #     geom1_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, c.geom1)
    #     geom2_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, c.geom2)

    #     if (
    #         ("left_foot_collision" in [geom1_name, geom2_name] or
    #          "right_foot_collision" in [geom1_name, geom2_name]) and
    #         "ground" in [geom1_name, geom2_name]
    #     ):
    #         print(f"✅ CONTACT: {geom1_name} <--> {geom2_name}")
    #         in_contact = True

    # if in_contact:
    #     print("✅ Foot is in contact with ground.")
    # else:
    #     print("❌ No foot-ground contact yet.")

q_start = data.qpos[:robot_position_dof].copy()
q_end = data.qpos[:robot_position_dof].copy()
q_end[:robot_position_dof] = np.array([ 
    0.0, 0.0, 0.38, 1.0, 0.0, 0.0, 0.0,
    0.0,  0.8, -2.0,  # FL
    0.0,  0.8, -2.0,  # FR
    0.0,  0.8, -2.0,  # RL
    0.0,  0.8, -2.0   # RR
])

# Initialize actual_q_before outside the loop
actual_q_before = data.qpos[:robot_position_dof].copy()

while True:
    get_wrench(model, data)
    get_robot_jacobian(model, data)

    if ADD_OBSTACLE and (time.time() - last_launch_time > launch_interval):
        launch_obstacle(model, data)
        last_launch_time = time.time()

    if i == 0:
        q_des = q_start.copy()
        v_des = np.zeros(robot_velocity_dof)
        a_des = np.zeros(robot_velocity_dof)
    elif i > NUM_STEPS:
        q_des = q_end.copy()
        v_des = np.zeros(robot_velocity_dof)
        a_des = np.zeros(robot_velocity_dof)
    else:
        q_des, v_des, a_des = minimum_jerk_trajectory(q_start, q_end, DURATION, i * DT)

    # Save actual state
    actual_q_current = data.qpos[:robot_position_dof].copy()
    actual_v = data.qvel[:robot_velocity_dof].copy()

    # Compute actuated joint velocities manually
    actuated_joint_vel = (actual_q_current[free_joint_pos:] - actual_q_before[free_joint_pos:]) / DT
    actual_v[free_joint_vel:] = actuated_joint_vel
    actual_q_before = actual_q_current.copy()

    # --- PD torque ---
    q_err = q_des[free_joint_pos:] - actual_q_current[free_joint_pos:]
    v_err = v_des[free_joint_vel:] - actual_v[free_joint_vel:]
    tau_pd = Kp * q_err + Kd * v_err

    # --- Contact torque (from measured foot wrenches) ---
    tau_contact = np.zeros(model.nv)
    for key in jacobian:
        J_foot = jacobian[key]           # shape: (6, nv)
        f_foot = contact_wrench[key]     # shape: (6,)
        tau_contact += J_foot.T @ f_foot # accumulate torques

    tau_contact_actuated = tau_contact[free_joint_vel:robot_velocity_dof]  # Only actuated joints

    # --- Combine PD and contact compensation ---
    tau_cmd = tau_pd + tau_contact_actuated
    print(f"jacobian: {jacobian}")

    # Apply torque to actuators
    data.ctrl[:] = tau_cmd

    # Step simulation
    mujoco.mj_step(model, data)
    viewer.sync()
    time.sleep(DT)
    i += 1
