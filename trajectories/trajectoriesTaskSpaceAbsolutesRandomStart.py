"""
Trajectory generation for task-space pick-and-place with absolute actions and random starting positions.
Same as trajectoriesTaskSpace but with random robot starting positions for each trajectory.
The moveArm function works in task space (takes desiredPos, desiredQuat).
Uses absolute actions: [ee_pos(3), ee_quat(3), gripper(1)] = 7D
"""
import sys
import os
# Add parent directory to path so we can import utils when running directly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mujoco
import mujoco.viewer
import time
import numpy as np
import scipy.spatial.transform
from scipy.spatial.transform import Rotation as R
from utils.ik_utils import computeIKError, moveArm, interpolate, slerp, computeBlockGraspInfo, verifyPlacement

# Import sample_robot_start_position from trajectoriesSimpler
try:
    from trajectories.trajectoriesSimpler import sample_robot_start_position
except ImportError:
    # If running directly, use relative import
    from trajectoriesSimpler import sample_robot_start_position

# Simulation timestep in seconds.
dt: float = 0.002

# Integration dt (needs to be begger than dt for some reason)
integration_dt = 0.002

# Damping term for the pseudoinverse. This is used to prevent joint velocities from
# becoming too large when the Jacobian is close to singular.
damping: float = 1e-10

# Number of seconds of a trajectory for now.
numSeconds = 10

# Table bounds (center + half-size) - UPDATED for bigger table in pickAndPlaceSimpler.xml
# Table is shifted to x=0.3, y=0 so robot is on the side
table_pos = np.array([0.3, 0])
table_size = np.array([0.4, 0.6])  # half-widths in x/y (0.4 x 0.6)

# Placement box bounds (to exclude from block spawning)
# Placement box at (0.1, 0.47) with size (0.15, 0.08) half-widths
placement_box_pos = np.array([0.1, 0.47])
placement_box_size = np.array([0.15, 0.08])
# Safety margin to account for block size (max block half-size is ~0.04m)
placement_box_margin = 0.05  # 5cm margin to ensure block doesn't overlap green box

# Robot reachable area (example limits, adjust to your robot)
x_min, x_max = -0.5, 0.5
y_min, y_max = -0.6, 0.6

def sample_block_pos():
    """
    Sample a block position ON TOP of the table (not on the placement box).
    Table surface is at z=0.2 (table center z=0.18 + table thickness 0.02).
    """
    while True:
        # Sample within table bounds
        x = np.random.uniform(table_pos[0] - np.abs(table_size[0]), table_pos[0] + np.abs(table_size[0]))
        y = np.random.uniform(table_pos[1] - np.abs(table_size[1]), table_pos[1] + np.abs(table_size[1]))
        
        # Check if NOT on placement box (green patch)
        # Add margin to account for block size - ensure block center is far enough away
        # that the block's edges won't overlap the green box
        on_placement_box = (placement_box_pos[0] - np.abs(placement_box_size[0]) - placement_box_margin <= x <= placement_box_pos[0] + np.abs(placement_box_size[0]) + placement_box_margin and
                           placement_box_pos[1] - np.abs(placement_box_size[1]) - placement_box_margin <= y <= placement_box_pos[1] + np.abs(placement_box_size[1]) + placement_box_margin)
        
        if not on_placement_box:
            # Table surface is at z=0.2 (table center z=0.18 + table top thickness 0.02)
            # Block center should be at table surface + half block height
            # For now, use a small offset that will be adjusted when block size is known
            z = 0.203  # Slightly above table surface (will be adjusted based on block size)
            return np.array([x, y, z])

# Maximum gripper width (distance between fingers)
max_gripper_width = 0.04  # meters

def sample_block_size():
    """
    Randomize block dimensions (width, depth, height) while ensuring the gripper can grasp.
    Width must be smaller than max_gripper_width.
    """
    width  = np.random.uniform(0.02, max_gripper_width)  # x-axis
    depth  = np.random.uniform(0.02, max_gripper_width)  # y-axis
    height = np.random.uniform(0.02, 0.1)  # z-axis
    return np.array([width, depth, height])  # half-sizes if MuJoCo expects that


max_angvel = 1.5

# Helper functions (interpolate and slerp are imported from ik_utils at the top)

# sample_robot_start_position is imported from trajectoriesSimpler
# computeBlockGraspInfo is imported from ik_utils

def createTrajectory(model, data, block_id, site, blockGeom_id, timesteps=None):
    if timesteps is None:
        timesteps = int(numSeconds // dt)
    mujoco.mj_forward(model, data)
    
    block_height_world, top_face_center, wantedRotQuat = computeBlockGraspInfo(model, data, block_id, blockGeom_id)

    site_pos = data.site_xpos[site]
    site_mat = data.site_xmat[site].reshape(3,3)
    site_quat = np.zeros(4)
    mujoco.mju_mat2Quat(site_quat, site_mat.flatten())

    # ---------------------------------------------------------------------
    prepickPos = top_face_center.copy()
    prepickPos[2] = top_face_center[2] + 0.05  # 5cm above top face center
    pickPos = top_face_center.copy()
    pickPos[2] = top_face_center[2] + 0.005  # 5mm above top face center
    
    table_surface_z = 0.2  # Table surface height
    block_top_face_z = table_surface_z + block_height_world
    # Updated placement position - place at placement box position (0.1, 0.47)
    placePos = np.array([0.1, 0.47, block_top_face_z + 0.005])  # 5mm above block's top face

    # ---------------------------------------------------------------------
    # Circular transfer parameters
    table_pos_3d = np.array([0.3, 0, 0.18])  # Updated to match shifted table (x=0.3, y=0)
    circle_center = np.array([0.0, 0.0])
    circle_radius = 0.45
    circle_height = table_pos_3d[2] + 0.4

    def circle_point(center, radius, angle, z):
        return np.array([
            center[0] + radius * np.cos(angle),
            center[1] + radius * np.sin(angle),
            z
        ])

    pick_angle  = np.arctan2(pickPos[1],  pickPos[0])
    place_angle = np.arctan2(placePos[1], placePos[0])
    angle_diff = np.unwrap([pick_angle, place_angle])

    circle_pick  = circle_point(circle_center, circle_radius, pick_angle,  circle_height)
    circle_place = circle_point(circle_center, circle_radius, place_angle, circle_height)

    # ---------------------------------------------------------------------
    # Rotations
    pickQ  = wantedRotQuat
    # Place with full alignment - maintain the same orientation as when picked
    # This ensures the block is placed with the same orientation (not just z-axis alignment)
    placeQ = wantedRotQuat.copy()

    # Segment lengths
    s1 = int(timesteps // 15)   # up
    s2 = int(timesteps // 40)   # approach pick
    s3 = int(timesteps // 50)   # swivel
    s4 = int(timesteps // 10)   # down (longer for smoother transition)

    trajectory = []

    # ---------------------------------------------------------------------
    # Go to cricle_pick
    for i in range(s1):
        newPos = interpolate(site_pos, circle_pick, i + 1, s1)
        newRot = slerp(site_quat, pickQ, i + 1, s1)
        trajectory.append((newPos, newRot))

    # Go to prepick
    for i in range(s1):
        newPos = interpolate(circle_pick, prepickPos, i + 1, s1)
        newRot = slerp(pickQ, pickQ, i + 1, s1)  # Maintain pickQ rotation
        trajectory.append((newPos, newRot))

    # Go to pick
    for i in range(s2):
        newPos = interpolate(prepickPos, pickPos, i + 1, s2)
        newRot = slerp(pickQ, pickQ, i + 1, s2)
        trajectory.append((newPos, newRot))

    # Stay at pick
    for i in range(s2):
        trajectory.append((pickPos, pickQ))

    # ---------------------------------------------------------------------
    # Lift to circle
    for i in range(s1):
        newPos = interpolate(pickPos, circle_pick, i + 1, s1)
        newRot = slerp(pickQ, pickQ, i + 1, s1)
        trajectory.append((newPos, newRot))

    # Swivel around circle - start transitioning orientation earlier for smoother transition
    transition_start_idx = int(0.5 * s3)  # Start transitioning at 50% of swivel
    for i in range(s3):
        theta = angle_diff[0] + (angle_diff[1] - angle_diff[0]) * (i + 1) / s3
        newPos = circle_point(circle_center, circle_radius, theta, circle_height)
        if i >= transition_start_idx:
            transition_steps = s3 - transition_start_idx + s4  # Transition over swivel + down
            transition_i = i - transition_start_idx + 1
            newRot = slerp(pickQ, placeQ, transition_i, transition_steps)
        else:
            newRot = slerp(pickQ, pickQ, i + 1, s3)
        trajectory.append((newPos, newRot))

    # Go down to place - continue orientation transition smoothly
    if s3 > transition_start_idx:
        transition_steps = s3 - transition_start_idx + s4
        swivel_transition_complete = s3 - transition_start_idx
        for i in range(s4):
            newPos = interpolate(circle_place, placePos, i + 1, s4)
            transition_i = swivel_transition_complete + i + 1
            newRot = slerp(pickQ, placeQ, transition_i, transition_steps)
            trajectory.append((newPos, newRot))
    else:
        for i in range(s4):
            newPos = interpolate(circle_place, placePos, i + 1, s4)
            newRot = slerp(pickQ, placeQ, i + 1, s4)
            trajectory.append((newPos, newRot))

    # ---------------------------------------------------------------------
    # Wait phase: Stay at place position to allow fingers to extend
    s_wait = int(timesteps // 10)  # Wait phase duration
    for i in range(s_wait):
        trajectory.append((placePos, placeQ))

    # ---------------------------------------------------------------------
    # Lifting phase: Lift above the object
    # Calculate lift height to be above the object (top of block + clearance)
    clearance = 0.05  # 5cm clearance above object
    lift_height = block_top_face_z + block_height_world + clearance
    liftPos = placePos.copy()
    liftPos[2] = lift_height
    
    s_lift = int(timesteps // 20)  # Lifting phase duration
    for i in range(s_lift):
        newPos = interpolate(placePos, liftPos, i + 1, s_lift)
        newRot = slerp(placeQ, placeQ, i + 1, s_lift)  # Maintain orientation during lift
        trajectory.append((newPos, newRot))

    shutFingers = 2 * s1 + s2
    placingPhaseStart = len(trajectory) - s_lift - s_wait - s4  # Start at beginning of "go down to place"
    liftingPhaseStart = len(trajectory) - s_lift - s_wait
    liftingPhaseEnd = len(trajectory)
    return trajectory, shutFingers, placingPhaseStart, liftingPhaseStart, liftingPhaseEnd


def main() -> None:
    # Load the model and data - USING SIMPLER XML
    model = mujoco.MjModel.from_xml_path("franka_emika_panda/pickAndPlaceSimpler.xml")
    data = mujoco.MjData(model)
    mujoco.mj_kinematics(model, data)

    # Initial joint configuration - randomize starting position
    key_id = model.key("home").id
    joint_names = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"][:7]
    dof_ids_temp = np.array([model.joint(name).id for name in joint_names])
    
    # Sample random starting position
    qpos_start = sample_robot_start_position(model, dof_ids_temp, key_id)
    data.qpos[:7] = qpos_start[:7]  # Set initial joint positions
    mujoco.mj_forward(model, data)  # Update forward kinematics

    # access to block and end_effector
    block_id = model.body('target').id
    site = model.site('palm_contact_edge_vis').id
    blockGeom_id = model.geom('box').id
    mocap_id = model.body('mocap').mocapid[0]
    joint_names = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7", "finger_joint1", "finger_joint2"][:7]
    dof_ids = np.array([model.joint(name).id for name in joint_names])
   
    # Note that actuator names are the same as joint names in this case.
    actuator_names = ['actuator1', 'actuator2', 'actuator3', 'actuator4', 'actuator5', 'actuator6', 'actuator7', 'actuator8'][:7]
    actuator_ids = [model.actuator(name).id for name in actuator_names]

    # -------------------------------- Initialize Block --------------------------------
    # ----------------------------------------------------------------------------------

    # Find the block's free joint
    block_jnt_adr = model.body(block_id).jntadr[0]
    block_jnt_id = block_jnt_adr  # This is the index into qpos

    # Sample block size (half-sizes for MuJoCo)
    block_size = sample_block_size()

    # Set the size in the MuJoCo model
    blockGeom_id = model.geom('box').id
    model.geom(blockGeom_id).size[:] = block_size

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
    targetInitialPos = sample_block_pos()
    data.qpos[block_jnt_id:block_jnt_id+3] = targetInitialPos
    data.qpos[block_jnt_id+3:block_jnt_id+7] = quat_mj



    # -------------------------------- Create Trajectory --------------------------------
    # ----------------------------------------------------------------------------------

    steps_on_waypoint = 0
    mujoco.mj_forward(model, data)
    trajectory, shutFingers, placingPhaseStart, liftingPhaseStart, liftingPhaseEnd = createTrajectory(
        model, data, block_id, site, blockGeom_id, timesteps=int(numSeconds//dt)
    )

     # -------------------------------- Visualize Trajectory --------------------------------
    # ----------------------------------------------------------------------------------
    
    with mujoco.viewer.launch_passive(
        model=model, data=data, show_left_ui=False, show_right_ui=False
    ) as viewer:
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)
        viewer.opt.frame = mujoco.mjtFrame.mjFRAME_SITE

        i = 0
        while viewer.is_running():
            step_start = time.time()

            if i < len(trajectory):
                data.mocap_pos[mocap_id] = trajectory[i][0]
                data.mocap_quat[mocap_id] = trajectory[i][1]
                error = moveArm(
                    model, data, site, block_id, dof_ids, actuator_ids,
                    trajectory[i][0], trajectory[i][1], i > shutFingers and i < liftingPhaseStart,
                    trajectory, i, placingPhaseStart, liftingPhaseStart, liftingPhaseEnd,
                    damping=damping, max_angvel=max_angvel
                )
            else:
                error = moveArm(
                    model, data, site, block_id, dof_ids, actuator_ids,
                    trajectory[-1][0], trajectory[-1][1], False,
                    trajectory, len(trajectory) - 1, placingPhaseStart, liftingPhaseStart, liftingPhaseEnd,
                    damping=damping, max_angvel=max_angvel
                )

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

