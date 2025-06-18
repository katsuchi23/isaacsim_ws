#!/usr/bin/env python
import numpy as np
import math

# Link lengths for the robot arm
L2 = 0.111
L3 = 0.079
L4 = 0.100
L6 = 0.051

def forward_kinematics(joint_angles):
    """Takes joint angles in radians, returns pose in [x, y, z, gripper_rot (rad), gripper_width (m)]"""
    thetas = np.array(joint_angles)

    x = ((L2 * math.sin(thetas[1])) + 
         (-L3 * math.cos(thetas[1] - thetas[2])) + 
         (-L4 * math.cos(thetas[1] - thetas[2] + thetas[3]))) * math.cos(thetas[0])

    y = ((L2 * math.sin(thetas[1])) + 
         (-L3 * math.cos(thetas[1] - thetas[2])) + 
         (-L4 * math.cos(thetas[1] - thetas[2] + thetas[3]))) * -math.sin(thetas[0]) - 0.03

    z = ((L2 * math.cos(thetas[1])) + 
         (L3 * math.sin(thetas[1] - thetas[2])) + 
         (L4 * math.sin(thetas[1] - thetas[2] + thetas[3]))) + 0.01

    gripper_roll = thetas[4]  # In radians
    gripper_pitch = thetas[3]
    gripper_width = -L6 * thetas[5]  # In meters

    return np.array([x, y, z, gripper_roll, gripper_pitch, gripper_width])

def compute_jacobian(fk, theta, alpha=1e-5):
    """Numerical Jacobian with input in radians"""
    n = len(theta)
    fk0 = fk(theta)
    m = len(fk0)
    J = np.zeros((m, n))
    for i in range(n):
        theta_d = np.copy(theta)
        theta_d[i] += alpha
        fk_d = fk(theta_d)
        J[:, i] = (fk_d - fk0) / alpha
    return J

def inverse_kinematic(
    target_pose, 
    initial_guess=None, 
    max_iter=100, 
    tol=1e-5, 
    alpha=0.5,
    lower_limits=None,
    upper_limits=None
):
    if initial_guess is None:
        theta = np.zeros(6)  # Default guess
    else:
        theta = np.array(initial_guess)

    # Default joint limits: full range for revolute, no limits for prismatic
    if lower_limits is None:
        lower_limits = np.array([-2.2, -1.57, -1.57, -2, -3.14, -1.6])
    if upper_limits is None:
        upper_limits = np.array([2.2,  0.6,  1.45,  2,  3.14,  0.032])

    for i in range(max_iter):
        current_pose = forward_kinematics(theta)
        error = target_pose - current_pose

        # Normalize angle error
        error[3] = (error[3] + np.pi) % (2 * np.pi) - np.pi

        if np.linalg.norm(error) < tol:
            # print(f"Converged in {i} iterations")
            print(f"Final solution: {theta}")
            print(f"Final pose: {current_pose}")
            return theta

        J = compute_jacobian(forward_kinematics, theta)
        delta_theta = alpha * np.linalg.pinv(J) @ error
        theta += delta_theta

        # Apply joint limits
        theta = np.clip(theta, lower_limits, upper_limits)

    print("IK did not converge")
    return None

# target_pose = np.array([-0.1, -0.15, 0.01, -0.66, 0.5, 0.02])  # roll in radians
# initial_guess = np.zeros(6)  # Initial guess in radians

# solution = inverse_kinematic(target_pose, initial_guess, max_iter=100, tol=1e-10, alpha=0.5)

# if solution is not None:
#     print("IK solution (rad):", solution)
#     print("FK result from solution:", forward_kinematics(solution))
# else:
#     print("No solution found")

