#!/usr/bin/env python3
# ------------------------------------------------------------------------------
# Detailed 7-Axis Franka Pick-and-Place Script with Gravity Compensation
# and Explicit Gripper Control
# ------------------------------------------------------------------------------

import os
import time
import numpy as np
import mujoco
import mujoco.viewer

def gripper_config():
    """
    Defines configuration for the single gripper finger actuator[cite: 1].
    """
    return [
        {"actuator_id": 7, "open_cmd": 0.0, "close_cmd": 0.022},
    ]

GRIPPERS = gripper_config()

def compensate_gravity(model, data):
    """
    Applies gravity and Coriolis compensation to all arm actuators while 
    skipping the gripper actuators so they remain under direct control[cite: 1].
    """
    skip = {g["actuator_id"] for g in GRIPPERS}
    for aid in range(model.nu):
        if aid in skip:
            continue
        joint_dof = model.actuator_trnid[aid][0]
        data.ctrl[aid] = data.qfrc_bias[joint_dof]

def main():
    # Resolve paths relative to this script location
    current_dir = os.path.dirname(os.path.abspath(__file__))
    xml_path = os.path.join(current_dir, "scene_franka.xml")

    # Load the MuJoCo model and data
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    N_ARM = 7

    # Set initial home posture for the 7 arm joints
    home_qpos = np.array([0.0, 0.2, 0.0, -1.2, 0.0, 1.2, 0.0])
    data.qpos[:N_ARM] = home_qpos
    mujoco.mj_forward(model, data)

    # Verify site ID safety check to prevent index out of bounds errors
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "ee_site")
    if site_id == -1:
        print("Warning: 'ee_site' not found in model, proceeding with joint-space control.")

    # Keyframes: [j1, j2, j3, j4, j5, j6, j7, gripper_cmd, duration]
    keyframes = [
        [0.00,  0.2,   0.0, -1.2,   0.0, 1.2,   0.0, 0.000, 3.0],  # 1. Home position
        [0.45,  0.25,  0.1, -1.10,  0.0, 1.10,  0.1, 0.000, 3.0],  # 2. Hover higher over cube
        [0.45,  0.65,  0.1, -1.50,  0.0, 1.50,  0.1, 0.022, 2.5],  # 3. Lower safely onto cube (clears table)
        [0.45,  0.65,  0.1, -1.50,  0.0, 1.50,  0.1, 0.010, 2.0],  # 4. Clamp onto cube
        [0.45,  0.25,  0.1, -1.10,  0.0, 1.10,  0.1, 0.010, 2.5],  # 5. Lift up high
        [0.10,  0.35, -0.1, -1.15,  0.0, 1.15, -0.1, 0.010, 3.0],  # 6. Move over black plate safely
        [0.10,  0.62, -0.1, -1.45,  0.0, 1.45, -0.1, 0.010, 2.5],  # 7. Lower onto plate surface
        [0.10,  0.62, -0.1, -1.45,  0.0, 1.45, -0.1, 0.022, 2.0],  # 8. Release cube
        [0.00,  0.2,   0.0, -1.2,   0.0, 1.2,   0.0, 0.022, 3.0],  # 9. Return home
    ]

    current_frame = 0
    start_time = time.time()
    current_qpos = home_qpos.copy()

    # Launch the passive MuJoCo viewer
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.azimuth = 135
        viewer.cam.elevation = -25
        viewer.cam.distance = 1.85
        viewer.cam.lookat[:] = [0.35, 0.45, 0.4]

        while viewer.is_running():
            step_start = time.time()
            elapsed = time.time() - start_time

            target_frame = keyframes[current_frame]
            target_qpos = np.array(target_frame[:7])
            target_gripper = target_frame[7]
            duration = target_frame[8]

            # Smooth cubic interpolation to avoid sudden jerky movements
            alpha = min(1.0, elapsed / duration)
            smooth_alpha = alpha * alpha * (3.0 - 2.0 * alpha)
            desired_qpos = current_qpos + smooth_alpha * (target_qpos - current_qpos)

            if alpha >= 1.0:
                current_qpos = target_qpos.copy()
                current_frame += 1
                start_time = time.time()
                if current_frame >= len(keyframes):
                    current_frame = len(keyframes) - 1

            # 1) Apply gravity compensation to arm joints[cite: 1]
            compensate_gravity(model, data)

            # 2) Add trajectory tracking control torques on top of gravity compensation
            kp = 250.0
            kd = 35.0
            q_err = desired_qpos - data.qpos[:N_ARM]
            dq = data.qvel[:N_ARM]
            
            data.ctrl[:N_ARM] += kp * q_err - kd * dq

            # 3) Drive the gripper actuators explicitly[cite: 1]
            for g in GRIPPERS:
                data.ctrl[g["actuator_id"]] = target_gripper

            # 4) Step simulation forward and sync viewer[cite: 1]
            mujoco.mj_step(model, data)
            viewer.sync()

            # Maintain real-time simulation pacing
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

if __name__ == '__main__':
    main()
