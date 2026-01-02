import sys
import os
# Add parent directory to path so we can import utils when running directly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mujoco
import mujoco.viewer
import time
import numpy as np
import matplotlib.pyplot as plt
import scipy.spatial.transform
from scipy.spatial.transform import Rotation as R
from utils.ik_utils import interpolate, slerp

# Simulation timestep in seconds.
dt: float = 0.002

# Integration dt (needs to be begger than dt for some reason)
integration_dt = 0.002

# Damping term for the pseudoinverse. This is used to prevent joint velocities from
# becoming too large when the Jacobian is close to singular.
damping: float = 1e-8

# Number of seconds of a trajectory for now.
numSeconds = 10

#block initial pos (pos, any, 0)
# targetInitialPos = np.array([0, 0.5, 0])
# Table bounds (center + half-size)
table_pos = np.array([0, 0.125])
table_size = np.array([0.2, 0.25])  # half-widths in x/y

# Robot reachable area (example limits, adjust to your robot)
x_min, x_max = -0.5, 0.5
y_min, y_max = -0.2, 0.6

def sample_block_pos():
    while True:
        x = np.random.uniform(x_min, x_max)
        y = np.random.uniform(y_min, y_max)
        # check if outside table
        if not (table_pos[0] - table_size[0] - 0.1 <= x <= table_pos[0] + table_size[0] + 0.1 and
                table_pos[1] - table_size[1] - 0.1 <= y <= table_pos[1] + table_size[1] + 0.1):
            z = 0.02  # table height + half block size (adjust if needed)
            return np.array([x, y, z])

# Maximum gripper width (distance between fingers)
max_gripper_width = 0.03  # meters
safety_margin = 0.005      # 5 mm safety

def sample_block_size():
    """
    Randomize block dimensions (width, depth, height) while ensuring the gripper can grasp.
    Width must be smaller than max_gripper_width.
    """
    width  = np.random.uniform(0.02, max_gripper_width - safety_margin)  # x-axis
    depth  = np.random.uniform(0.02, 0.08)  # y-axis
    height = np.random.uniform(0.02, 0.08)  # z-axis
    return np.array([width, depth, height])  # half-sizes if MuJoCo expects that


# assign random block position outside table
targetInitialPos = sample_block_pos()

blockAngleDeg = 145  # rotation around z-axis in degrees

# Maximum allowable joint velocity in rad/s. Set to 0 to disable.
max_angvel = 0.0

def main() -> None:
    # Load the model and data.
    model = mujoco.MjModel.from_xml_path("franka_emika_panda/pickAndPlace.xml")
    data = mujoco.MjData(model)
    mujoco.mj_kinematics(model, data)

    # Initial joint configuration saved as a keyframe in the XML file.
    key_id = model.key("home").id

    # access to block and end_effector
    block_id = model.body('target').id
    site = model.site('palm_contact_edge_vis').id
    blockGeom_id = model.geom('box').id
    mocap_id = model.body('mocap').mocapid[0]
    joint_names = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7", "finger_joint1", "finger_joint2"][:7]
    #print(joint_names)
    dof_ids = np.array([model.joint(name).id for name in joint_names])
   
    # Note that actuator names are the same as joint names in this case.
    actuator_names = ['actuator1', 'actuator2', 'actuator3', 'actuator4', 'actuator5', 'actuator6', 'actuator7', 'actuator8'][:7]
    actuator_ids = [model.actuator(name).id for name in actuator_names]

    # Find the block's free joint
    block_jnt_adr = model.body(block_id).jntadr[0]
    block_jnt_id = block_jnt_adr  # This is the index into qpos

    # Sample block size (half-sizes for MuJoCo)
    block_size = sample_block_size()

    # Set the size in the MuJoCo model
    blockGeom_id = model.geom('box').id
    model.geom(blockGeom_id).size[:] = block_size



    #---------------------------------------------------------------------
    # choose which local axis is pointing DOWN (face on table)
    axes = np.eye(3)
    face = np.random.randint(6)
    if face < 3:
        local_down = axes[:, face]
    else:
        local_down = -axes[:, face - 3]

    # world down
    world_down = np.array([0.0, 0.0, -1.0])

    # rotation that maps local_down -> world_down
    v = np.cross(local_down, world_down)
    c = np.dot(local_down, world_down)

    if np.linalg.norm(v) < 1e-8:
        R_align = np.eye(3)
    else:
        vx = np.array([
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0],
        ])
        R_align = np.eye(3) + vx + vx @ vx * (1 / (1 + c))

    # random rotation around world down axis
    theta = np.random.uniform(0, 2*np.pi)
    R_yaw = R.from_rotvec(theta * world_down).as_matrix()

    # final block rotation
    Q = R_yaw @ R_align

    # convert to MuJoCo quaternion
    quat_mj = np.zeros(4)
    mujoco.mju_mat2Quat(quat_mj, Q.flatten())

    # set block pose
    data.qpos[block_jnt_id:block_jnt_id+3] = targetInitialPos
    data.qpos[block_jnt_id+3:block_jnt_id+7] = quat_mj



    #---------------------------------------------------------------------

    steps_on_waypoint = 0
    mujoco.mj_forward(model, data)
    def createTrajectory(timesteps = numSeconds//dt):
        blockWPos = data.xpos[block_id]
        blockWMat = data.xmat[block_id]
        blockSize = model.geom(blockGeom_id).size #x_half, y_half, z_half

        site_pos = data.site_xpos[site]  # 3-vector
        site_mat = data.site_xmat[site].reshape(3,3)  # 3x3 rotation
        site_quat = np.zeros(4)
        mujoco.mju_mat2Quat(site_quat, site_mat.flatten())


        #---------------------------------------------------------------------
        # Compute grasp rotation
        blockR = blockWMat.reshape(3, 3)       # block rotation matrix
        world_up = np.array([0.0, 0.0, 1.0])   # table normal

        # Block face normals (±x, ±y, ±z)
        face_normals = np.hstack((blockR, -blockR))
        idx = np.argmax(face_normals.T @ world_up)
        top_face_normal = face_normals[:, idx]

        # Approach vector (from above)
        approach = -top_face_normal
        approach /= np.linalg.norm(approach)

        # block axes
        block_axes = blockR  # local x, y, z

        # project onto plane perpendicular to approach
        hor_axes = [a - np.dot(a, approach)*approach for a in block_axes.T]
        hor_lengths = [np.linalg.norm(a) for a in hor_axes]

        # handle near-zero axes safely
        for i, length in enumerate(hor_lengths):
            if length < 1e-6:
                # pick arbitrary perpendicular vector
                if abs(approach[0]) < 0.9:
                    hor_axes[i] = np.cross(approach, np.array([1,0,0]))
                else:
                    hor_axes[i] = np.cross(approach, np.array([0,1,0]))
                hor_axes[i] /= np.linalg.norm(hor_axes[i])
                hor_lengths[i] = 1.0

        # choose shortest horizontal axis
        shortest_idx = np.argmin(hor_lengths)
        x_axis_candidate = hor_axes[shortest_idx] / hor_lengths[shortest_idx]

        # Two possible grasp directions along this axis
        options = [x_axis_candidate, -x_axis_candidate]

        # Compute rotation difference from current hand orientation
        # Get current gripper x-axis
        current_rot = site_mat  # 3x3 rotation of hand
        current_x = current_rot[:,0]  # assuming x-axis points along gripper

        best_x = None
        min_angle = np.inf
        for opt in options:
            # angle between current gripper x-axis and candidate
            angle = np.arccos(np.clip(np.dot(current_x, opt), -1, 1))
            if angle < min_angle:
                min_angle = angle
                best_x = opt

        x_axis = best_x
        y_axis = np.cross(approach, x_axis)
        y_axis /= np.linalg.norm(y_axis)

        # Final rotation
        wantedRot = np.column_stack((x_axis, y_axis, approach))
        wantedRotQuat = np.zeros(4)
        mujoco.mju_mat2Quat(wantedRotQuat, wantedRot.flatten())



        #---------------------------------------------------------------------

        # locations
        # prepickPos = blockWPos.copy() + np.array([0, 0, blockSize[2] + 0.05])
        # pickPos = blockWPos.copy() + np.array([0, 0, blockSize[2]])
        # placePos = np.array([0, 0.3, 0.24 + blockSize[2]])
        # pick positions along block’s vertical axis (top face normal)
        # Determine height of top face in world coordinates
        top_face_offset = blockWPos + top_face_normal * blockSize[np.argmax(np.abs(top_face_normal))]

        prepickPos = top_face_offset + top_face_normal * 0.05  # a bit above
        pickPos    = top_face_offset + top_face_normal * 0.005  # right at top face
        placePos   = np.array([0, 0.3, 0.24]) + np.array([0,0, blockSize[2]])  # can add safety margin



        #intermediate location
        table_pos = np.array([0, 0.125, 0.18])
        table_size = np.array([0.2, 0.25, 0.02])  # half-sizes
        intermediatePos = np.array([0, 0, 0.3 + blockSize[2]]) #setup

        x_sign = -1 if pickPos[0] == 0 else pickPos[0]/abs(pickPos[0])

        y_safe = table_pos[1] + table_size[1] + 0.10 + blockSize[1]

        if pickPos[1] < -0.125:
            print("behind table!!!")
            intermediatePos[1] = table_pos[1] - table_size[1] - 0.15
            intermediatePos[0] = (table_size[0] + 0.2) * x_sign
        else:
            if pickPos[1] < y_safe:
                if pickPos[1] < 0.375: #tablepos+tablesize
                    print("side of table table -> pushing x-dir")
                    intermediatePos[0] = (table_pos[1] - table_size[1] - 0.15) * x_sign
                else:
                    print("behind table -> pushing forward")
                    intermediatePos[1] = y_safe
            else:
                print("not behind table")
                intermediatePos[1] = pickPos[1]
            intermediatePos[0] = pickPos[0]

        #rotations
        pickQ = wantedRotQuat
        placeQ = np.array([0, 1, 0, 0])

        # how many timesteps per segment
        s1 = int(timesteps // 15)
        s2 = int(timesteps // 40)
        s3 = int(timesteps // 50)
        s4 = int(timesteps // 15)
        s5 = timesteps - (s1+s2+s3+s4)

        trajectory = []
        #go to intermediate
        for i in range(s1):
            newPos = interpolate(site_pos, intermediatePos, i + 1, s1)
            newRot = slerp(site_quat, pickQ, i + 1, s1)
            trajectory.append((newPos, newRot))

        #go to prepick
        for i in range(s1):
            newPos = interpolate(intermediatePos, prepickPos, i + 1, s1)
            newRot = slerp(pickQ, pickQ, i + 1, s1)
            trajectory.append((newPos, newRot))

        # go to pick
        for i in range(s2):
            newPos = interpolate(prepickPos, pickPos, i + 1, s2)
            newRot = slerp(pickQ, pickQ, i + 1, s2)
            trajectory.append((newPos, newRot))

        # stay at pick
        for i in range(int(s3)): 
            newPos  = interpolate(pickPos, pickPos, i, s3)
            newRot = slerp(pickQ, pickQ, i, s3)
            trajectory.append((newPos, newRot))

        # back to prepick
        for i in range(int(s3//2)): 
            newPos  = interpolate(pickPos, prepickPos, i, int(s3//2))
            newRot = slerp(pickQ, pickQ, i, s3)
            trajectory.append((newPos, newRot))

        # transition = 
        for i in range(s3): 
            newPos  = interpolate(prepickPos, intermediatePos, i, s3)
            newRot = slerp(pickQ, pickQ, i, s3)
            trajectory.append((newPos, newRot))

        # go to place
        for i in range(s3): 
            newPos  = interpolate(intermediatePos, placePos, i, s3)
            newRot = slerp(pickQ, placeQ, i, s4)
            trajectory.append((newPos, newRot))
        shutFingers = 2 * s1 + s2 + int(s3//2)
        return trajectory, shutFingers

    def interpolate(start, end, step, numSteps):
        return (1 - step/numSteps) * start + step/numSteps * end
    
    #chatgpt helped with slerp implementation
    def slerp(start, end, step, numSteps):
        r_start = scipy.spatial.transform.Rotation.from_quat(start)
        r_end   = scipy.spatial.transform.Rotation.from_quat(end)
        slerp   = scipy.spatial.transform.Slerp([0, 1], scipy.spatial.transform.Rotation.concatenate([r_start, r_end]))
        return slerp([step/numSteps]).as_quat()[0]
    

    def moveArm(desiredPos, desiredQuat, shutFingers):
        jac = np.zeros((6, model.nv))
        diag = damping * np.eye(6)
        error = np.zeros((6,))

        site_pos = data.site_xpos[site]  # 3-vector
        site_mat = data.site_xmat[site].reshape(3,3)  # 3x3 rotation
        site_quat = np.zeros(4)
        mujoco.mju_mat2Quat(site_quat, site_mat.flatten())

        #pos error
        error[:3] = desiredPos - site_pos
        
        # Compute quaternion difference
        current_quat = site_quat
        quat_diff = np.zeros(4)
        mujoco.mju_negQuat(quat_diff, current_quat)  # Invert current
        mujoco.mju_mulQuat(quat_diff, desiredQuat, quat_diff) # desired * current^-1

        mujoco.mj_jacSite(model, data, jac[:3], jac[3:], site)

        # orientation error (body frame)
        error[3:] = quat_diff[1:] * np.sign(quat_diff[0])

        dq = jac.T @ np.linalg.solve(jac @ jac.T + diag, error)

        # Scale down joint velocities if they exceed maximum.
        if max_angvel > 0:
            dq_abs_max = np.abs(dq).max()
            if dq_abs_max > max_angvel:
                dq *= max_angvel / dq_abs_max
        pass

        q_des = data.qpos.copy()
        q_des[dof_ids] += dq[dof_ids]

        for i, jid in enumerate(dof_ids):
            min_range = model.jnt_range[jid, 0]
            max_range = model.jnt_range[jid, 1]
            
            # Wrap within ±2π until in range
            while q_des[jid] < min_range:
                q_des[jid] += 2 * np.pi
            while q_des[jid] > max_range:
                q_des[jid] -= 2 * np.pi

        # Send to actuators
        data.ctrl[actuator_ids] = q_des[dof_ids]

        # Set finger actuator
        if shutFingers:
            data.ctrl[model.actuator("actuator8").id] = 0        # close fingers
        else:
            data.ctrl[model.actuator("actuator8").id] = 255      # open fingers


        return error


    trajectory, shutFingers = createTrajectory()
    with mujoco.viewer.launch_passive(
        model=model, data=data, show_left_ui=False, show_right_ui=False
    ) as viewer:
        # Reset the simulation to the initial keyframe.
        # mujoco.mj_resetDataKeyframe(model, data, key_id)

        # Initialize the camera view to that of the free camera.
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)

        # Toggle site frame visualization.
        viewer.opt.frame = mujoco.mjtFrame.mjFRAME_SITE

        i = 0
        while viewer.is_running():
            step_start = time.time()

            # for visualization only
            if i < len(trajectory):
                data.mocap_pos[mocap_id] = trajectory[i][0]
                data.mocap_quat[mocap_id] = trajectory[i][1]

                error = moveArm(trajectory[i][0], trajectory[i][1], i > shutFingers)
            else:
                error = moveArm(trajectory[-1][0], trajectory[-1][1], False)

            # Step the simulation.
            mujoco.mj_step(model, data)

            viewer.sync()
            time_until_next_step = dt - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

            error_norm = np.linalg.norm(error)
            if error_norm < 0.03 or steps_on_waypoint > 50:
                i += 1
                steps_on_waypoint = 0
            else:
                steps_on_waypoint += 1


if __name__ == "__main__":
    main()