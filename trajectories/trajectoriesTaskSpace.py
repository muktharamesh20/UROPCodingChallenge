"""
Trajectory generation for task-space pick-and-place.
Same as trajectoriesEvenSimpler but designed for task-space action representation.
The moveArm function works in task space (takes desiredPos, desiredQuat).
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
from utils.ik_utils import computeIKError, moveArm, interpolate, slerp, get_robot_start_position, computeBlockGraspInfo, verifyPlacement

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

# get_robot_start_position and computeBlockGraspInfo are now imported from ik_utils

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




# verifyPlacement is now imported from ik_utils

