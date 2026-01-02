"""
Inverse kinematics utilities for trajectory generation.
Contains shared functions for differential IK with nullspace projection.
"""
import mujoco
import numpy as np
import scipy.spatial.transform
from utils.training_utils import get_joint_range_for_joint


def computeIKError(model, data, site, desiredPos, desiredQuat):
    """
    Compute 6D IK error (position + orientation).
    
    Args:
        model: MuJoCo model
        data: MuJoCo data
        site: Site ID
        desiredPos: Desired position (3,)
        desiredQuat: Desired quaternion (4,)
    
    Returns:
        error: 6D error vector [pos_error(3), orient_error(3)]
    """
    error = np.zeros(6)
    site_pos = data.site_xpos[site]
    site_mat = data.site_xmat[site].reshape(3,3)
    site_quat = np.zeros(4)
    mujoco.mju_mat2Quat(site_quat, site_mat.flatten())
    error[:3] = desiredPos - site_pos
    quat_diff = np.zeros(4)
    mujoco.mju_negQuat(quat_diff, site_quat)
    mujoco.mju_mulQuat(quat_diff, desiredQuat, quat_diff)
    error[3:] = quat_diff[1:] * np.sign(quat_diff[0])
    return error


def moveArm(model, data, site, block_id, dof_ids, actuator_ids, desiredPos, desiredQuat, shutFingers, 
            trajectory=None, current_idx=None, placingPhaseStart=None, liftingPhaseStart=None, liftingPhaseEnd=None,
            damping=1e-10, max_angvel=1.5):
    """
    Move arm to desired end-effector pose using differential IK with nullspace projection.
    
    This function works in task space (takes desiredPos, desiredQuat) and uses:
    - Damped least squares (DLS) for primary task
    - Hierarchical nullspace projection for secondary tasks:
      * Joint limit avoidance
      * Trajectory lookahead (if trajectory provided)
      * Elbow-up bias
    
    During placing/lifting phases or when gripper is closed, only constrains z-axis to point down.
    
    Args:
        model: MuJoCo model
        data: MuJoCo data
        site: End-effector site ID
        block_id: Block body ID
        dof_ids: Joint DOF IDs (array)
        actuator_ids: Actuator IDs (list)
        desiredPos: Desired end-effector position (3,)
        desiredQuat: Desired end-effector quaternion (4,)
        shutFingers: Whether gripper is closed (bool)
        trajectory: Optional trajectory list of (pos, quat) tuples
        current_idx: Optional current trajectory index
        placingPhaseStart: Optional start index of placing phase
        liftingPhaseStart: Optional start index of lifting phase
        liftingPhaseEnd: Optional end index of lifting phase
        damping: Damping term for DLS (default: 1e-10)
        max_angvel: Maximum joint angular velocity in rad/s (default: 1.5, set to 0 to disable)
    
    Returns:
        error: 6D IK error vector
    """
    in_placing_phase = False
    in_lifting_phase = False
    if placingPhaseStart is not None and liftingPhaseStart is not None and current_idx is not None:
        in_placing_phase = placingPhaseStart <= current_idx < liftingPhaseStart
    if liftingPhaseStart is not None and liftingPhaseEnd is not None and current_idx is not None:
        in_lifting_phase = liftingPhaseStart <= current_idx < liftingPhaseEnd
    
    if shutFingers or in_placing_phase or in_lifting_phase:
        site_mat = data.site_xmat[site].reshape(3,3)
        current_z = site_mat[:, 2]
        desired_z = np.array([0.0, 0.0, -1.0])
        
        if np.dot(current_z, desired_z) > 0.9:
            site_quat = np.zeros(4)
            mujoco.mju_mat2Quat(site_quat, site_mat.flatten())
            desiredQuat = site_quat
        else:
            current_x = site_mat[:, 0]
            current_y = site_mat[:, 1]
            new_x = current_x - np.dot(current_x, desired_z) * desired_z
            new_x /= np.linalg.norm(new_x) if np.linalg.norm(new_x) > 1e-6 else 1.0
            new_y = np.cross(desired_z, new_x)
            new_y /= np.linalg.norm(new_y)
            new_rot_mat = np.column_stack((new_x, new_y, desired_z))
            desiredQuat = np.zeros(4)
            mujoco.mju_mat2Quat(desiredQuat, new_rot_mat.flatten())
    
    error = computeIKError(model, data, site, desiredPos, desiredQuat)
    
    if shutFingers or in_placing_phase or in_lifting_phase:
        # Get current z-axis (approach direction)
        site_mat = data.site_xmat[site].reshape(3,3)
        current_z = site_mat[:, 2]  # z-axis is the third column
        
        # Desired z-axis is straight down
        desired_z = np.array([0.0, 0.0, -1.0])
        
        # Compute error as deviation from desired z-axis
        # This is the cross product which gives the axis of rotation needed
        z_error = np.cross(current_z, desired_z)
        
        # Scale by the angle between them (magnitude of cross product = sin(angle))
        # For small angles, this approximates the rotation needed
        error[3:] = z_error
    
    # Compute Jacobian
    jac = np.zeros((6, model.nv))
    jacp = jac[:3]
    jacr = jac[3:]
    mujoco.mj_jacSite(model, data, jacp, jacr, site)
    jac_controlled = jac[:, dof_ids]
    
    # Primary task (DLS)
    JJt = jac_controlled @ jac_controlled.T
    JJt_damped = JJt + damping * np.eye(6)
    dq_task_full = np.zeros(model.nv)
    dq_task_full[dof_ids] = jac_controlled.T @ np.linalg.solve(JJt_damped, error)

    J_pinv = np.linalg.pinv(jac_controlled)
    N1 = np.eye(len(dof_ids)) - J_pinv @ jac_controlled
    
    dq_nullspace_total = np.zeros(model.nv)
    N_current = N1.copy()
    
    # Task 1: Joint limit escape
    limit_escape_threshold = 0.4
    limit_escape_gain = 2
    
    dq_task1 = np.zeros(model.nv)
    for i, jid in enumerate(dof_ids):
        minr, maxr = get_joint_range_for_joint(i, model_mj=model, dof_id=jid, use_default=True)
        if minr >= maxr or not np.isfinite(minr) or not np.isfinite(maxr):
            continue
        
        q_current = data.qpos[jid]
        dist_to_min = q_current - minr
        dist_to_max = maxr - q_current
        
        if dist_to_min < limit_escape_threshold:
            escape_velocity = limit_escape_gain * (limit_escape_threshold - dist_to_min) / limit_escape_threshold
            dq_task1[jid] = escape_velocity
        elif dist_to_max < limit_escape_threshold:
            escape_velocity = -limit_escape_gain * (limit_escape_threshold - dist_to_max) / limit_escape_threshold
            dq_task1[jid] = escape_velocity
    
    dq_task1_projected = np.zeros(model.nv)
    dq_task1_projected[dof_ids] = N_current @ dq_task1[dof_ids]
    dq_nullspace_total += dq_task1_projected
    
    dq_task1_proj_norm_sq = np.dot(dq_task1_projected[dof_ids], dq_task1_projected[dof_ids])
    if dq_task1_proj_norm_sq > 1e-10:
        dq_task1_proj_normalized = dq_task1_projected[dof_ids] / np.sqrt(dq_task1_proj_norm_sq)
        N_task1 = np.eye(len(dof_ids)) - np.outer(dq_task1_proj_normalized, dq_task1_proj_normalized)
        N_current = N_current @ N_task1
    
    # Task 2: Trajectory lookahead
    if trajectory is not None and current_idx is not None:
        lookahead_steps = 5
        future_idx = current_idx + lookahead_steps
        
        if future_idx < len(trajectory):
            future_pos, future_quat = trajectory[future_idx]
            future_error = computeIKError(model, data, site, future_pos, future_quat)
            
            if shutFingers or in_placing_phase or in_lifting_phase:
                site_mat = data.site_xmat[site].reshape(3,3)
                current_z = site_mat[:, 2]
                desired_z = np.array([0.0, 0.0, -1.0])
                z_error = np.cross(current_z, desired_z)
                future_error[3:] = z_error
            
            dq_task2 = np.zeros(model.nv)
            dq_task2[dof_ids] = jac_controlled.T @ np.linalg.solve(JJt_damped, future_error)
            lookahead_gain = 0.1
            dq_task2[dof_ids] *= lookahead_gain
            dq_task2_projected = np.zeros(model.nv)
            dq_task2_projected[dof_ids] = N_current @ dq_task2[dof_ids]
            dq_nullspace_total += dq_task2_projected
            
            dq_task2_proj_norm_sq = np.dot(dq_task2_projected[dof_ids], dq_task2_projected[dof_ids])
            if dq_task2_proj_norm_sq > 1e-10:
                dq_task2_proj_normalized = dq_task2_projected[dof_ids] / np.sqrt(dq_task2_proj_norm_sq)
                N_task2 = np.eye(len(dof_ids)) - np.outer(dq_task2_proj_normalized, dq_task2_proj_normalized)
                N_current = N_current @ N_task2

    # Task 3: Elbow-up bias
    elbow_qpos_id = model.joint("joint1").qposadr
    elbow_dof_id  = model.joint("joint1").dofadr
    base_pos = data.xpos[model.body("link0").id]
    obj_pos  = data.xpos[block_id]
    q_elbow_des = np.arctan2(obj_pos[1] - base_pos[1], obj_pos[0] - base_pos[0])
    dq_task3 = np.zeros(model.nv)
    dq_task3[elbow_dof_id] = 0.1 * (q_elbow_des - data.qpos[elbow_qpos_id])
    
    dq_task3_projected = np.zeros(model.nv)
    dq_task3_projected[dof_ids] = N_current @ dq_task3[dof_ids]
    dq_nullspace_total += dq_task3_projected

    dq = dq_task_full + dq_nullspace_total

    if max_angvel > 0:
        dq_abs_max = np.abs(dq).max()
        if dq_abs_max > max_angvel:
            dq *= max_angvel / dq_abs_max

    q_des = data.qpos.copy()
    q_des[dof_ids] += dq[dof_ids]

    for i, jid in enumerate(dof_ids):
        minr, maxr = get_joint_range_for_joint(i, model_mj=model, dof_id=jid, use_default=True)
        if minr >= maxr or not np.isfinite(minr) or not np.isfinite(maxr):
            continue
        while q_des[jid] < minr:
            q_des[jid] += 2*np.pi
        while q_des[jid] > maxr:
            q_des[jid] -= 2*np.pi

    data.ctrl[actuator_ids] = q_des[dof_ids]
    data.ctrl[model.actuator("actuator8").id] = 0 if shutFingers else 255
    return error


# ============================================================================
# Trajectory Interpolation Utilities
# ============================================================================

def interpolate(start, end, step, numSteps):
    """
    Linear interpolation between start and end positions.
    
    Args:
        start: Starting position/vector
        end: Ending position/vector
        step: Current step (0 to numSteps)
        numSteps: Total number of steps
    
    Returns:
        Interpolated position/vector
    """
    return (1 - step/numSteps) * start + step/numSteps * end


def slerp(start, end, step, numSteps):
    """
    Spherical linear interpolation (SLERP) between two quaternions.
    
    Args:
        start: Starting quaternion (4,)
        end: Ending quaternion (4,)
        step: Current step (0 to numSteps)
        numSteps: Total number of steps
    
    Returns:
        Interpolated quaternion (4,)
    """
    start = start / np.linalg.norm(start)
    end = end / np.linalg.norm(end)
    if np.dot(start, end) < 0:
        end = -end
    r_start = scipy.spatial.transform.Rotation.from_quat(start)
    r_end = scipy.spatial.transform.Rotation.from_quat(end)
    slerp_interp = scipy.spatial.transform.Slerp([0, 1], scipy.spatial.transform.Rotation.concatenate([r_start, r_end]))
    result = slerp_interp([step/numSteps]).as_quat()[0]
    return result / np.linalg.norm(result)


# ============================================================================
# Robot Start Position Utilities
# ============================================================================

def get_robot_start_position(model, dof_ids, home_key_id=None):
    """
    Get a fixed starting position for the robot using the home keyframe.
    No randomization - always uses the home position for a convenient starting location above the table.
    
    Args:
        model: MuJoCo model
        dof_ids: Array of joint DOF IDs
        home_key_id: Optional keyframe ID to use as base (if None, uses "home" key)
    
    Returns:
        qpos_start: Starting joint positions (always home position)
    """
    if home_key_id is None:
        home_key_id = model.key("home").id
    
    qpos_home = model.key(home_key_id).qpos.copy()
    return qpos_home.copy()


# ============================================================================
# Block Grasp Computation
# ============================================================================

def computeBlockGraspInfo(model, data, block_id, blockGeom_id):
    """
    Compute block height, top face center, and desired grasp orientation.
    
    Args:
        model: MuJoCo model
        data: MuJoCo data
        block_id: Block body ID
        blockGeom_id: Block geometry ID
    
    Returns:
        block_height_world: Height of the block in world coordinates
        top_face_center: Center point of the top face of the block
        wantedRotQuat: Desired grasp orientation as a quaternion
    """
    blockWPos = data.xpos[block_id]
    blockWMat = data.xmat[block_id]
    blockSize = model.geom(blockGeom_id).size

    blockR = blockWMat.reshape(3, 3)
    corners_local = np.array([
        [-blockSize[0], -blockSize[1], -blockSize[2]],
        [ blockSize[0], -blockSize[1], -blockSize[2]],
        [-blockSize[0],  blockSize[1], -blockSize[2]],
        [ blockSize[0],  blockSize[1], -blockSize[2]],
        [-blockSize[0], -blockSize[1],  blockSize[2]],
        [ blockSize[0], -blockSize[1],  blockSize[2]],
        [-blockSize[0],  blockSize[1],  blockSize[2]],
        [ blockSize[0],  blockSize[1],  blockSize[2]],
    ])
    
    corners_world = (blockR @ corners_local.T).T + blockWPos
    max_z = np.max(corners_world[:, 2])
    min_z = np.min(corners_world[:, 2])
    block_height_world = max_z - min_z
    tolerance = max(1e-5, block_height_world * 1e-4)
    top_face_corners = corners_world[corners_world[:, 2] >= (max_z - tolerance)]
    top_face_center = np.mean(top_face_corners, axis=0)
    
    top_face_horiz = top_face_corners.copy()
    top_face_horiz[:, 2] = 0.0
    if len(top_face_horiz) >= 4:
        # Compute vectors from first corner to others, get 2 shortest edges
        ref = top_face_horiz[0]
        vecs = top_face_horiz[1:] - ref
        dists = np.linalg.norm(vecs, axis=1)
        valid = dists > 1e-6
        
        sorted_idx = np.argsort(dists[valid])[:2]
        dirs = vecs[valid][sorted_idx]
        dirs[:, 2] = 0.0
        dirs = dirs / np.linalg.norm(dirs, axis=1, keepdims=True)
        
        extents = dists[valid][sorted_idx]
        idx = np.argsort(extents)
        shortest_horiz_dir = dirs[idx[0]]
        longest_horiz_dir = dirs[idx[1]]

    else:
        shortest_horiz_dir = np.array([1.0, 0.0, 0.0])
        longest_horiz_dir = np.array([0.0, 1.0, 0.0])
    
    # y-axis is the opening/closing direction of the gripper
    # Set y-axis to shortest, and x-axis will be perpendicular
    y_axis_target = shortest_horiz_dir.copy()
    y_axis_target[2] = 0.0
    y_axis_target /= np.linalg.norm(y_axis_target)
    
    x_axis = longest_horiz_dir.copy()
    x_axis[2] = 0.0
    x_axis /= np.linalg.norm(x_axis)
    
    shortest_horiz_dir_aligned = shortest_horiz_dir.copy()
    shortest_horiz_dir_aligned[2] = 0.0
    shortest_horiz_dir_aligned /= np.linalg.norm(shortest_horiz_dir_aligned)
    
    y_axis = shortest_horiz_dir_aligned - np.dot(shortest_horiz_dir_aligned, x_axis) * x_axis
    y_axis_norm = np.linalg.norm(y_axis)
    
    if y_axis_norm < 1e-6:
        # shortest_horiz_dir is parallel to x_axis, use perpendicular direction
        y_axis = np.cross(np.array([0.0, 0.0, 1.0]), x_axis)
        y_axis /= np.linalg.norm(y_axis)
    else:
        y_axis /= y_axis_norm
    
    # Choose direction that aligns with shortest_horiz_dir
    if np.dot(y_axis, shortest_horiz_dir_aligned) < 0:
        y_axis = -y_axis

    # Compute approach from x and y (right-handed: approach = x × y)
    # This ensures approach points in the correct direction
    approach_final = np.cross(x_axis, y_axis)
    approach_final /= np.linalg.norm(approach_final)
    
    # If approach points up, we need to flip y_axis
    # This maintains alignment with shortest dimension (just flips direction)
    if approach_final[2] > 0:
        y_axis = -y_axis
        approach_final = np.cross(x_axis, y_axis)
        approach_final /= np.linalg.norm(approach_final)

    wantedRot = np.column_stack((x_axis, y_axis, approach_final))
    wantedRotQuat = np.zeros(4)
    mujoco.mju_mat2Quat(wantedRotQuat, wantedRot.flatten())
    
    return block_height_world, top_face_center, wantedRotQuat


# ============================================================================
# Placement Verification
# ============================================================================

def verifyPlacement(model, data, block_id, table_surface_z=0.2, tolerance=0.01):
    """
    Verify if block is placed above the table surface.
    
    Args:
        model: MuJoCo model
        data: MuJoCo data
        block_id: Block body ID
        table_surface_z: Z-coordinate of table surface (default: 0.2)
        tolerance: Position tolerance in meters (default: 0.01)
    
    Returns:
        success: True if block is above table surface
        block_z: Block's z-coordinate
        message: Status message
    """
    mujoco.mj_forward(model, data)
    block_pos = data.xpos[block_id]
    block_z = block_pos[2]
    if block_z > table_surface_z + tolerance:
        return True, block_z, f"Success: Block at z={block_z:.4f}m (table at {table_surface_z}m)"
    else:
        return False, block_z, f"Failed: Block at z={block_z:.4f}m (table at {table_surface_z}m, need >{table_surface_z+tolerance}m)"

