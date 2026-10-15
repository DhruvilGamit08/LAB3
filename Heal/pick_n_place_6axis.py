#!/usr/bin/env python3
import os
import time
import numpy as np
import mujoco
import mujoco.viewer

current_dir = os.path.dirname(os.path.abspath(__file__))
xml_path = os.path.join(current_dir, "scene_heal.xml")

POSITION_THRESHOLD = 0.025 

class TaskSpacePlanner:
    def __init__(self, model, data):
        self.model = model
        self.data = data
        self.site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "ee_site")
        self.n_arm = 6

        mat = self.data.site_xmat[self.site_id].reshape(3, 3)
        self.target_quat = np.zeros(4)
        mujoco.mju_mat2Quat(self.target_quat, mat.flatten())

    def reach_pose(self, target_pos):
        pos = self.data.site_xpos[self.site_id].copy()
        pos_err = target_pos - pos

        mat = self.data.site_xmat[self.site_id].reshape(3, 3)
        curr_quat = np.zeros(4)
        mujoco.mju_mat2Quat(curr_quat, mat.flatten())

        neg_curr_quat = np.zeros(4)
        err_quat = np.zeros(4)
        ori_err = np.zeros(3)
        mujoco.mju_negQuat(neg_curr_quat, curr_quat)
        mujoco.mju_mulQuat(err_quat, self.target_quat, neg_curr_quat)
        mujoco.mju_quat2Vel(ori_err, err_quat, 1.0)

        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self.site_id)

        J_pos = jacp[:, :self.n_arm]
        J_ori = jacr[:, :self.n_arm]
        dq = self.data.qvel[:self.n_arm]

        # INCREASED STIFFNESS: Doubled the P-gains to prevent gravity sag
        f_pos = 800.0 * pos_err - 80.0 * (J_pos @ dq)
        f_ori = 80.0 * ori_err - 8.0 * (J_ori @ dq)

        tau = J_pos.T @ f_pos + J_ori.T @ f_ori + self.data.qfrc_bias[:self.n_arm]
        return tau, np.linalg.norm(pos_err)

def main():
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    # Initial starting pose
    data.qpos[:6] = [0.20, -0.8, 1.2, -0.4, 0.0, 0.0]
    mujoco.mj_forward(model, data)

    gripper_act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "fingers_actuator")

    # RAISED WAYPOINTS: Z-heights bumped up to clear the table completely
    pregrasp_pos   = np.array([0.381, 0.569, 0.70])
    grasp_pos      = np.array([0.381, 0.569, 0.48]) 
    lift_pos       = np.array([0.381, 0.569, 0.70])
    tray_above_pos = np.array([0.6, 0.4, 0.70])
    tray_drop_pos  = np.array([0.6, 0.4, 0.52])

    planner = TaskSpacePlanner(model, data)
    phase = "pregrasp"
    grip_closed = False
    grip_opened = False
    close_time = None
    drop_time = None

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.azimuth = 135
        viewer.cam.elevation = -25
        viewer.cam.distance = 1.65
        viewer.cam.lookat[:] = [0.35, 0.45, 0.4]

        while viewer.is_running():
            step_start = time.time()

            if phase == "pregrasp": tau, err = planner.reach_pose(pregrasp_pos)
            elif phase == "grasp": tau, err = planner.reach_pose(grasp_pos)
            elif phase == "lift": tau, err = planner.reach_pose(lift_pos)
            elif phase == "tray_above": tau, err = planner.reach_pose(tray_above_pos)
            elif phase == "tray_drop" or phase == "done": tau, err = planner.reach_pose(tray_drop_pos)

            data.ctrl[:6] = tau

            if grip_closed and not grip_opened: 
                data.ctrl[gripper_act_id] = 0.022
            else: 
                data.ctrl[gripper_act_id] = 0.0

            if phase == "pregrasp" and err < POSITION_THRESHOLD: 
                phase = "grasp"
            elif phase == "grasp" and err < POSITION_THRESHOLD and not grip_closed:
                grip_closed = True
                close_time = time.time()
            elif phase == "grasp" and grip_closed and (time.time() - close_time > 0.8): 
                phase = "lift"
            elif phase == "lift" and err < POSITION_THRESHOLD: 
                phase = "tray_above"
            elif phase == "tray_above" and err < POSITION_THRESHOLD: 
                phase = "tray_drop"
            elif phase == "tray_drop" and err < POSITION_THRESHOLD and not grip_opened:
                grip_opened = True
                drop_time = time.time()
                phase = "done"
            elif phase == "done" and (time.time() - drop_time > 3.0): 
                break

            mujoco.mj_step(model, data)
            viewer.sync()

            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0: 
                time.sleep(time_until_next_step)

if __name__ == "__main__":
    main()
