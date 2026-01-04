"""
Shared utility functions for training pipelines.
Contains common functions used across different training pipelines.
"""
import os
import glob
import pickle
import numpy as np
import pandas as pd
import mujoco
from scipy.spatial.transform import Rotation as R
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

from .load_dataset_csv import load_dataset_csv, get_state_action_pairs
from .path_utils import (
    get_variation_path,
    get_model_path,
    get_dataset_path,
    get_scaler_path,
    get_variation_from_base_filename,
    find_model_file,
    find_scaler_file,
    find_dataset_files,
    get_model_path_for_file,
    get_scaler_path_for_model
)
try:
    from trajectories.trajectoriesSimpler import placement_box_pos as default_placement_box_pos, placement_box_size as default_placement_box_size
except ImportError:
    default_placement_box_pos = None
    default_placement_box_size = None

# ============================================================================
# Constants
# ============================================================================

JOINT_NAMES = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"]
ACTUATOR_NAMES = ['actuator1', 'actuator2', 'actuator3', 'actuator4', 'actuator5', 'actuator6', 'actuator7']
GRIPPER_ACTUATOR_NAME = "actuator8"
BLOCK_BODY_NAME = "target"
BLOCK_GEOM_NAME = "box"
SITE_NAME = "palm_contact_edge_vis"

FRANKA_PANDA_JOINT_RANGES = np.array([
    (-2.8973, 2.8973),    # joint1
    (-1.7628, 1.7628),    # joint2
    (-2.8973, 2.8973),    # joint3
    (-3.0718, -0.0698),   # joint4
    (-2.8973, 2.8973),    # joint5
    (-0.0175, 3.7525),    # joint6
    (-2.8973, 2.8973),    # joint7
])

# ============================================================================
# State and Action Extraction
# ============================================================================

def extract_arm_state(qpos, qvel, ee_pos, ee_quat, use_sin_cos_encoding=False):
    """
    Extract kinematic state of arm.
    
    Args:
        qpos: Joint positions
        qvel: Joint velocities
        ee_pos: End-effector position
        ee_quat: End-effector quaternion
        use_sin_cos_encoding: If True, uses sin/cos encoding for joint angles (28D),
                             otherwise uses raw joint positions (21D)
    
    Returns:
        arm_state: 28D if use_sin_cos_encoding=True, 21D otherwise
    """
    if use_sin_cos_encoding:
        joint_sin = np.sin(qpos[:7])
        joint_cos = np.cos(qpos[:7])
        return np.concatenate([joint_sin, joint_cos, qvel[:7], ee_pos, ee_quat])
    else:
        return np.concatenate([qpos[:7], qvel[:7], ee_pos, ee_quat])

def extract_block_state(block_pos, block_quat, block_size):
    """Extract kinematic state of block: [pos(3), quat(4), size(3)] = 10D"""
    return np.concatenate([block_pos, block_quat, block_size])

def convert_state_dimension(state, target_dim):
    """
    Convert state vector between 31D and 38D representations.
    
    State structure:
    - 31D: [joint_pos(7), joint_vel(7), ee_pos(3), ee_quat(4), block_pos(3), block_quat(4), block_size(3)]
    - 38D: [sin(joint_pos)(7), cos(joint_pos)(7), joint_vel(7), ee_pos(3), ee_quat(4), block_pos(3), block_quat(4), block_size(3)]
    
    Args:
        state: State vector (numpy array) - can be 31D or 38D
        target_dim: Desired output dimension (31 or 38)
    
    Returns:
        converted_state: State vector in the desired format
    
    Raises:
        ValueError: If input state dimension is not 31 or 38, or target_dim is not 31 or 38
    """
    state = np.asarray(state)
    input_dim = len(state)
    
    if input_dim not in [31, 38]:
        raise ValueError(f"Input state must be 31D or 38D, got {input_dim}D")
    if target_dim not in [31, 38]:
        raise ValueError(f"Target dimension must be 31 or 38, got {target_dim}")
    
    if input_dim == target_dim:
        return state.copy()
    
    block_state = state[-10:]
    
    if input_dim == 31:
        joint_pos = state[0:7]
        middle = state[7:21]
    else:
        joint_sin = state[0:7]
        joint_cos = state[7:14]
        middle = state[14:28]
        joint_pos = np.arctan2(joint_sin, joint_cos)
    
    if target_dim == 38:
        joint_sin = np.sin(joint_pos)
        joint_cos = np.cos(joint_pos)
        return np.concatenate([joint_sin, joint_cos, middle, block_state])
    else:
        return np.concatenate([joint_pos, middle, block_state])

def extract_action(ctrl, actuator8_id):
    """Extract action: [joint_targets(7), gripper(1)] = 8D"""
    gripper_cmd = 1.0 if ctrl[actuator8_id] < 127.5 else 0.0
    return np.concatenate([ctrl[:7], [gripper_cmd]])

def extract_action_delta(current_qpos, desired_qpos, ctrl, actuator8_id):
    """
    Extract delta action: [joint_deltas(7), gripper(1)] = 8D
    
    Args:
        current_qpos: Current joint positions (7,)
        desired_qpos: Desired joint positions from controller (7,)
        ctrl: Control array
        actuator8_id: Gripper actuator ID
    
    Returns:
        delta_action: [joint_deltas(7), gripper(1)] = 8D
    """
    delta_qpos = desired_qpos - current_qpos
    gripper_cmd = 1.0 if ctrl[actuator8_id] < 127.5 else 0.0
    
    return np.concatenate([delta_qpos, [gripper_cmd]])

def extract_action_taskspace_delta(current_ee_pos, current_ee_quat, desired_ee_pos, desired_ee_quat, ctrl, actuator8_id):
    """
    Extract task-space delta action: [delta_ee_pos(3), delta_ee_quat(3), gripper(1)] = 7D
    
    Args:
        current_ee_pos: Current end-effector position (3,)
        current_ee_quat: Current end-effector quaternion (4,)
        desired_ee_pos: Desired end-effector position (3,)
        desired_ee_quat: Desired end-effector quaternion (4,)
        ctrl: Control array
        actuator8_id: Gripper actuator ID
    
    Returns:
        delta_action: [delta_ee_pos(3), delta_ee_quat(3), gripper(1)] = 7D
    """
    delta_ee_pos = desired_ee_pos - current_ee_pos
    
    quat_diff = np.zeros(4)
    mujoco.mju_negQuat(quat_diff, current_ee_quat)
    mujoco.mju_mulQuat(quat_diff, desired_ee_quat, quat_diff)
    delta_ee_quat = quat_diff[1:] * np.sign(quat_diff[0])
    
    gripper_cmd = 1.0 if ctrl[actuator8_id] < 127.5 else 0.0
    
    return np.concatenate([delta_ee_pos, delta_ee_quat, [gripper_cmd]])


def extract_action_taskspace_absolute(desired_ee_pos, desired_ee_quat, ctrl, actuator8_id):
    """
    Extract task-space absolute action: [ee_pos(3), ee_quat(3), gripper(1)] = 7D
    Uses 3D quaternion representation for consistency with delta version.
    
    Args:
        desired_ee_pos: Desired end-effector position (3,)
        desired_ee_quat: Desired end-effector quaternion (4,)
        ctrl: Control array
        actuator8_id: Gripper actuator ID
    
    Returns:
        action: [ee_pos(3), ee_quat(3), gripper(1)] = 7D
        Note: quat is 3D representation (quat[1:] * sign(quat[0])) for consistency
    """
    quat_3d = desired_ee_quat[1:] * np.sign(desired_ee_quat[0])
    gripper_cmd = 1.0 if ctrl[actuator8_id] < 127.5 else 0.0
    
    return np.concatenate([desired_ee_pos, quat_3d, [gripper_cmd]])


def convert_delta_action_to_absolute_action(state, delta_action, use_sin_cos_encoding=True):
    """
    Convert task-space delta action to absolute action using current state.
    
    Args:
        state: State vector (38D if use_sin_cos_encoding=True, 31D otherwise)
        delta_action: Delta action [delta_ee_pos(3), delta_ee_quat(3), gripper(1)] = 7D
        use_sin_cos_encoding: Whether state uses sin/cos encoding
    
    Returns:
        absolute_action: [ee_pos(3), ee_quat(3), gripper(1)] = 7D
    """
    state_31d = convert_state_dimension(state, 31) if use_sin_cos_encoding else state
    
    current_ee_pos = state_31d[14:17]
    current_ee_quat = state_31d[17:21]
    
    delta_ee_pos = delta_action[:3]
    delta_ee_quat_3d = delta_action[3:6]
    gripper_cmd = delta_action[6]
    
    absolute_ee_pos = current_ee_pos + delta_ee_pos
    quat_norm_sq = np.dot(delta_ee_quat_3d, delta_ee_quat_3d)
    if quat_norm_sq > 1.0:
        delta_ee_quat_3d = delta_ee_quat_3d / np.sqrt(quat_norm_sq) * 0.999
        quat_norm_sq = 0.998001
    
    quat_diff_w = np.sqrt(max(0, 1.0 - quat_norm_sq))
    quat_diff = np.concatenate([[quat_diff_w], delta_ee_quat_3d])
    quat_diff = quat_diff / (np.linalg.norm(quat_diff) + 1e-8)
    
    absolute_ee_quat = np.zeros(4)
    mujoco.mju_mulQuat(absolute_ee_quat, quat_diff, current_ee_quat)
    absolute_ee_quat = absolute_ee_quat / (np.linalg.norm(absolute_ee_quat) + 1e-8)
    
    absolute_ee_quat_3d = absolute_ee_quat[1:] * np.sign(absolute_ee_quat[0])
    
    return np.concatenate([absolute_ee_pos, absolute_ee_quat_3d, [gripper_cmd]])


def compute_desired_pose_from_absolute(absolute_ee_pos, absolute_ee_quat_3d):
    """
    Convert task-space absolute action to desired pose (position and quaternion).
    
    Args:
        absolute_ee_pos: Absolute end-effector position (3,)
        absolute_ee_quat_3d: Absolute end-effector quaternion (3,) - 3D representation
    
    Returns:
        desired_pos: Desired end-effector position (3,)
        desired_quat: Desired end-effector quaternion (4,)
    """
    quat_norm_sq = np.dot(absolute_ee_quat_3d, absolute_ee_quat_3d)
    if quat_norm_sq > 1.0:
        absolute_ee_quat_3d = absolute_ee_quat_3d / np.sqrt(quat_norm_sq) * 0.999
        quat_norm_sq = 0.998001
    
    quat_w = np.sqrt(max(0, 1.0 - quat_norm_sq))
    desired_quat = np.concatenate([[quat_w], absolute_ee_quat_3d])
    desired_quat = desired_quat / (np.linalg.norm(desired_quat) + 1e-8)
    
    return absolute_ee_pos.copy(), desired_quat

# ============================================================================
# Joint Ranges and Action Normalization
# ============================================================================

def get_joint_ranges(model_mj=None, dof_ids=None, use_default=True):
    """
    Get joint ranges for normalization.
    
    Args:
        model_mj: MuJoCo model (optional). If provided, extracts ranges from model.
        dof_ids: List of joint DOF IDs (optional, required if model_mj is provided).
        use_default: If True and model extraction fails, uses FRANKA_PANDA_JOINT_RANGES.
                    If False and model extraction fails, uses (-pi, pi) for each joint.
    
    Returns:
        joint_ranges: Array of (min, max) tuples for each joint, shape (7, 2)
    """
    if model_mj is not None and dof_ids is not None:
        joint_ranges = []
        for jid in dof_ids:
            minr = model_mj.jnt_range[jid, 0]
            maxr = model_mj.jnt_range[jid, 1]
            if minr >= maxr or not np.isfinite(minr) or not np.isfinite(maxr):
                if use_default:
                    joint_idx = len(joint_ranges)
                    if joint_idx < len(FRANKA_PANDA_JOINT_RANGES):
                        minr, maxr = FRANKA_PANDA_JOINT_RANGES[joint_idx]
                    else:
                        minr, maxr = -np.pi, np.pi
                else:
                    minr, maxr = -np.pi, np.pi
            joint_ranges.append((minr, maxr))
        return np.array(joint_ranges)
    else:
        return FRANKA_PANDA_JOINT_RANGES.copy()

def get_joint_range_for_joint(joint_id, model_mj=None, dof_id=None, use_default=True):
    """
    Get joint range for a single joint.
    
    Args:
        joint_id: Joint index (0-6) or joint name
        model_mj: MuJoCo model (optional)
        dof_id: DOF ID in model (optional, required if model_mj is provided)
        use_default: If True and model extraction fails, uses FRANKA_PANDA_JOINT_RANGES
    
    Returns:
        (min, max): Tuple of (min, max) joint limits
    """
    if model_mj is not None and dof_id is not None:
        minr = model_mj.jnt_range[dof_id, 0]
        maxr = model_mj.jnt_range[dof_id, 1]
        if minr >= maxr or not np.isfinite(minr) or not np.isfinite(maxr):
            if use_default:
                if isinstance(joint_id, int) and 0 <= joint_id < len(FRANKA_PANDA_JOINT_RANGES):
                    return FRANKA_PANDA_JOINT_RANGES[joint_id]
                else:
                    return (-np.pi, np.pi)
            else:
                return (-np.pi, np.pi)
        return (minr, maxr)
    else:
        # No model, use default
        if isinstance(joint_id, int) and 0 <= joint_id < len(FRANKA_PANDA_JOINT_RANGES):
            return FRANKA_PANDA_JOINT_RANGES[joint_id]
        elif isinstance(joint_id, str):
            try:
                idx = JOINT_NAMES.index(joint_id)
                return FRANKA_PANDA_JOINT_RANGES[idx]
            except ValueError:
                return (-np.pi, np.pi)
        else:
            return (-np.pi, np.pi)

def normalize_action(action, joint_ranges, clamp_input=True):
    """
    Normalize action to [0, 1] range.
    
    Args:
        action: [joint_targets(7), gripper(1)] = 8D
        joint_ranges: Array of (min, max) tuples for each joint
        clamp_input: If True, clamp action to joint ranges before normalizing
    
    Returns:
        Normalized action in [0, 1] range
    """
    normalized = np.zeros_like(action)
    for i in range(7):
        minr, maxr = joint_ranges[i]
        if clamp_input:
            # Clamp to range first to handle out-of-range values
            clamped = np.clip(action[i], minr, maxr)
        else:
            clamped = action[i]
        # Normalize to [0, 1]
        normalized[i] = (clamped - minr) / (maxr - minr) if (maxr - minr) > 1e-6 else 0.5
    # Gripper is already in [0, 1] range, preserve it
    normalized[7] = action[7]
    return normalized

def denormalize_action(normalized_action, joint_ranges, clamp_output=True):
    """
    Denormalize action from [0, 1] range back to joint positions.
    
    Args:
        normalized_action: Normalized action in [0, 1] range
        joint_ranges: Array of (min, max) tuples for each joint
        clamp_output: If True, clamp normalized value to [0, 1] before denormalizing
    
    Returns:
        Denormalized action with actual joint positions
    """
    denormalized = np.zeros_like(normalized_action)
    for i in range(7):
        minr, maxr = joint_ranges[i]
        if clamp_output:
            # Clamp normalized value to [0, 1] to handle model outputs outside range
            clamped_norm = np.clip(normalized_action[i], 0.0, 1.0)
        else:
            clamped_norm = normalized_action[i]
        # Denormalize from [0, 1] to [minr, maxr]
        denormalized[i] = minr + clamped_norm * (maxr - minr)
    # Gripper stays in [0, 1] range (preserve it)
    if clamp_output:
        denormalized[7] = np.clip(normalized_action[7], 0.0, 1.0)
    else:
        denormalized[7] = normalized_action[7]
    return denormalized

# ============================================================================
# Scaler Save/Load
# ============================================================================

def save_action_scaler(joint_ranges, target_trajectories, base_name='default'):
    """Save joint ranges for action normalization."""
    scaler_dir = get_scaler_path(base_name)
    os.makedirs(scaler_dir, exist_ok=True)
    scaler_file = os.path.join(scaler_dir, f"action_scaler_{base_name}_{target_trajectories}traj.pkl")
    with open(scaler_file, 'wb') as f:
        pickle.dump(joint_ranges, f)
    print(f"  Saved action scaler to {scaler_file}")

def load_action_scaler(target_trajectories, base_name='default'):
    """Load joint ranges for action normalization."""
    scaler_file = os.path.join(get_scaler_path(base_name), f"action_scaler_{base_name}_{target_trajectories}traj.pkl")
    if os.path.exists(scaler_file):
        with open(scaler_file, 'rb') as f:
            return pickle.load(f)
    return None

def load_state_scaler(target_trajectories, base_name='default'):
    """Load state scaler from file if it exists."""
    scaler_file = os.path.join(get_scaler_path(base_name), f"state_scaler_{base_name}_{target_trajectories}traj.pkl")
    if os.path.exists(scaler_file):
        with open(scaler_file, 'rb') as f:
            return pickle.load(f)
    return None

def save_state_scaler(scaler, target_trajectories, base_name='default'):
    """Save state scaler to file."""
    scaler_dir = get_scaler_path(base_name)
    os.makedirs(scaler_dir, exist_ok=True)
    scaler_file = os.path.join(scaler_dir, f"state_scaler_{base_name}_{target_trajectories}traj.pkl")
    with open(scaler_file, 'wb') as f:
        pickle.dump(scaler, f)

# ============================================================================
# Dataset Utilities
# ============================================================================

def count_trajectories(base_filename):
    """Count total trajectories and successes from CSV files."""
    # Determine variation folder for datasets
    variation = get_variation_from_base_filename(base_filename)
    dataset_dir = get_dataset_path(variation)
    
    # Look in variation folder first
    pattern = os.path.join(dataset_dir, f"{base_filename}_*.csv")
    files = sorted(glob.glob(pattern))
    old_filename = os.path.join(dataset_dir, f"{base_filename}.csv")
    if os.path.exists(old_filename):
        files.append(old_filename)
    
    # Fallback: check current directory (for backward compatibility)
    if not files:
        pattern = f"{base_filename}_*.csv"
        files = sorted(glob.glob(pattern))
        if os.path.exists(f"{base_filename}.csv"):
            files.append(f"{base_filename}.csv")
    
    max_id = -1
    successes = 0
    for filename in files:
        if os.path.getsize(filename) == 0:
            continue
        try:
            for chunk in pd.read_csv(filename, chunksize=100000, usecols=['trajectory_id', 'isDone']):
                if len(chunk) > 0:
                    max_id = max(max_id, chunk['trajectory_id'].max())
                    successes += chunk.groupby('trajectory_id')['isDone'].any().sum()
        except (pd.errors.EmptyDataError, ValueError, KeyError):
            continue
    return max_id + 1 if max_id >= 0 else 0, successes

# ============================================================================
# MuJoCo Setup Utilities
# ============================================================================

def setup_mujoco_model(xml_path, dt=0.002):
    """
    Setup MuJoCo model and data, return model, data, and IDs.
    
    Args:
        xml_path: Path to XML file
        dt: Simulation timestep
    
    Returns:
        model: MuJoCo model
        data: MuJoCo data
        block_id: Block body ID
        site_id: End-effector site ID
        block_geom_id: Block geometry ID
        dof_ids: Joint DOF IDs
        actuator_ids: Actuator IDs
        gripper_actuator_id: Gripper actuator ID
        block_jnt_id: Block joint ID
        key_id: Home keyframe ID
    """
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    model.opt.timestep = dt
    
    block_id = model.body(BLOCK_BODY_NAME).id
    site_id = model.site(SITE_NAME).id
    block_geom_id = model.geom(BLOCK_GEOM_NAME).id
    dof_ids = np.array([model.joint(name).id for name in JOINT_NAMES])
    actuator_ids = [model.actuator(name).id for name in ACTUATOR_NAMES]
    gripper_actuator_id = model.actuator(GRIPPER_ACTUATOR_NAME).id
    block_jnt_id = model.body(block_id).jntadr[0]
    key_id = model.key("home").id
    
    return model, data, block_id, site_id, block_geom_id, dof_ids, actuator_ids, gripper_actuator_id, block_jnt_id, key_id

def initialize_block_random_orientation(model, data, block_geom_id, block_jnt_id, sample_block_pos, sample_block_size):
    """
    Initialize block with random size and orientation on the table.
    
    Args:
        model: MuJoCo model
        data: MuJoCo data
        block_id: Block body ID
        block_geom_id: Block geometry ID
        block_jnt_id: Block joint ID
        sample_block_pos: Function to sample block position
        sample_block_size: Function to sample block size
    
    Returns:
        block_size: Block size array
    """
    block_size = sample_block_size()
    model.geom(block_geom_id).size[:] = block_size
    target_initial_pos = sample_block_pos()
    
    axes = np.eye(3)
    face = np.random.randint(6)
    if face < 3:
        local_down = axes[:, face]
    else:
        local_down = -axes[:, face - 3]
    
    world_down = np.array([0.0, 0.0, -1.0])
    v = np.cross(local_down, world_down)
    c = np.dot(local_down, world_down)
    
    if np.linalg.norm(v) < 1e-8:
        R_align = np.eye(3)
    else:
        vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        R_align = np.eye(3) + vx + vx @ vx * (1 / (1 + c))
    
    theta = np.random.uniform(0, 2*np.pi)
    R_yaw = R.from_rotvec(theta * world_down).as_matrix()
    Q = R_yaw @ R_align
    quat_mj = np.zeros(4)
    mujoco.mju_mat2Quat(quat_mj, Q.flatten())
    
    data.qpos[block_jnt_id:block_jnt_id+3] = target_initial_pos
    data.qpos[block_jnt_id+3:block_jnt_id+7] = quat_mj
    mujoco.mj_forward(model, data)
    
    return block_size

def extract_state_from_simulation(data, block_id, site_id, block_size, use_sin_cos_encoding=False):
    """
    Extract state vector from MuJoCo simulation.
    
    Args:
        data: MuJoCo data
        block_id: Block body ID
        site_id: End-effector site ID
        block_size: Block size array
        use_sin_cos_encoding: If True, uses sin/cos encoding for joint angles
    
    Returns:
        state: State vector (38D if use_sin_cos_encoding=True, 31D otherwise)
    """
    qpos = data.qpos[:7]
    qvel = data.qvel[:7]
    ee_pos = data.site_xpos[site_id]
    site_mat = data.site_xmat[site_id].reshape(3, 3)
    ee_quat = np.zeros(4)
    mujoco.mju_mat2Quat(ee_quat, site_mat.flatten())
    ee_quat = ee_quat / (np.linalg.norm(ee_quat) + 1e-8)
    
    block_pos = data.xpos[block_id]
    block_quat = data.xquat[block_id]
    block_quat = block_quat / (np.linalg.norm(block_quat) + 1e-8)
    
    arm_state = extract_arm_state(qpos, qvel, ee_pos, ee_quat, use_sin_cos_encoding=False)
    block_state = extract_block_state(block_pos, block_quat, block_size)
    state = np.concatenate([arm_state, block_state])
    
    if use_sin_cos_encoding:
        state = convert_state_dimension(state, 38)
    
    return state

# ============================================================================
# Batch Size Utilities
# ============================================================================

def get_batch_size(num_samples, device_type):
    """Get appropriate batch size based on dataset size and device."""
    if device_type == 'mps':
        if num_samples > 1000000:
            return 16384
        elif num_samples > 500000:
            return 8192
        elif num_samples > 100000:
            return 4096
        elif num_samples > 50000:
            return 2048
        elif num_samples > 20000:
            return 1024
        else:
            return 512
    else:
        if num_samples > 1000000:
            return 4096
        elif num_samples > 500000:
            return 2048
        elif num_samples > 100000:
            return 1024
        elif num_samples > 50000:
            return 512
        elif num_samples > 20000:
            return 384
        else:
            return 256

# ============================================================================
# Dataset Loading and Filtering
# ============================================================================

def load_and_filter_dataset(base_filename, target_trajectories, filter_top_pct=0.2, 
                            recovery_filename=None, use_recovery=True, use_sin_cos_encoding=False):
    """
    Load dataset, filter longest trajectories, return states/actions.
    If recovery_filename is provided and use_recovery is True, mixes 50% normal + 50% recovery data.
    
    Args:
        base_filename: Base filename for dataset (e.g., 'dataset_even_simpler')
        target_trajectories: Target number of trajectories to load
        filter_top_pct: Percentage of longest trajectories to filter out (default: 0.2 = 20%)
        recovery_filename: Optional base filename for recovery dataset
        use_recovery: Whether to use recovery dataset if provided
        use_sin_cos_encoding: If True, converts 21D arm states to 28D with sin/cos encoding
    
    Returns:
        states: State vectors (38D if use_sin_cos_encoding=True, 31D otherwise)
        actions: Action vectors
        traj_ids: Trajectory IDs
        steps: Optional step indices
    """
    # Check for preprocessed file first (much faster)
    variation = get_variation_from_base_filename(base_filename)
    dataset_dir = get_dataset_path(variation)
    preprocessed_file = os.path.join(dataset_dir, f"{base_filename}_preprocessed.pkl")
    
    if os.path.exists(preprocessed_file):
        print(f"  Loading from preprocessed file: {os.path.basename(preprocessed_file)}")
        with open(preprocessed_file, 'rb') as f:
            preprocessed_data = pickle.load(f)
        
        # Extract data from preprocessed file
        states_preprocessed = preprocessed_data['states']
        actions_preprocessed = preprocessed_data['actions']
        traj_ids_preprocessed = preprocessed_data['trajectory_ids']
        steps_preprocessed = preprocessed_data.get('steps', None)
        
        # Limit to target_trajectories if needed
        unique_traj_ids_preprocessed = np.unique(traj_ids_preprocessed)
        if len(unique_traj_ids_preprocessed) > target_trajectories:
            # Take first target_trajectories unique trajectory IDs
            traj_ids_to_use = unique_traj_ids_preprocessed[:target_trajectories]
            mask = np.isin(traj_ids_preprocessed, traj_ids_to_use)
            states_preprocessed = states_preprocessed[mask]
            actions_preprocessed = actions_preprocessed[mask]
            traj_ids_preprocessed = traj_ids_preprocessed[mask]
            if steps_preprocessed is not None:
                steps_preprocessed = steps_preprocessed[mask]
        
        # Convert to dict format for compatibility
        data = {
            'arm_states': states_preprocessed[:, :21],  # First 21 dims are arm state
            'block_states': states_preprocessed[:, 21:],  # Remaining 10 dims are block state
            'actions': actions_preprocessed,
            'trajectory_ids': traj_ids_preprocessed,
            'isDone': np.zeros(len(traj_ids_preprocessed), dtype=bool)  # Not used for filtering
        }
        if steps_preprocessed is not None:
            data['steps'] = steps_preprocessed
    else:
        # Fall back to loading from CSV
        data = load_dataset_csv(filename=None, base_filename=base_filename, 
                                max_trajectories=target_trajectories)
    
    # Get states and actions
    # Note: get_state_action_pairs returns 31D states (21D arm + 10D block)
    states, actions = get_state_action_pairs(data)
    
    # Convert to 38D if using sin/cos encoding
    if use_sin_cos_encoding:
        # Convert all states from 31D to 38D using convert_state_dimension
        states = np.array([convert_state_dimension(state, 38) for state in states])
        print(f"  Converted states from 31D to 38D (sin/cos encoding)")
    
    actions = data['actions']
    traj_ids = data['trajectory_ids']
    
    # Filter longest trajectories
    unique_ids = np.unique(traj_ids)
    traj_lengths = {tid: np.sum(traj_ids == tid) for tid in unique_ids}
    sorted_ids = sorted(unique_ids, key=lambda tid: traj_lengths[tid], reverse=True)
    num_remove = max(1, int(len(sorted_ids) * filter_top_pct))
    
    mask = ~np.isin(traj_ids, sorted_ids[:num_remove])
    states_normal = states[mask]
    actions_normal = actions[mask]
    traj_ids_normal = traj_ids[mask]
    steps_normal = data.get('steps', None)
    if steps_normal is not None:
        steps_normal = steps_normal[mask]
    
    # Load recovery dataset if requested
    if use_recovery and recovery_filename:
        try:
            print(f"  Loading recovery dataset from {recovery_filename}...")
            data_recovery = load_dataset_csv(filename=None, base_filename=recovery_filename,
                                            max_trajectories=target_trajectories)
            
            # Get recovery states and actions
            states_recovery, actions_recovery = get_state_action_pairs(data_recovery)
            
            # Convert to 38D if using sin/cos encoding
            if use_sin_cos_encoding:
                states_recovery = np.array([convert_state_dimension(state, 38) for state in states_recovery])
                print(f"  Converted recovery states from 31D to 38D (sin/cos encoding)")
            
            actions_recovery = data_recovery['actions']
            
            traj_ids_recovery = data_recovery['trajectory_ids']
            steps_recovery = data_recovery.get('steps', None)
            
            unique_ids_recovery = np.unique(traj_ids_recovery)
            
            traj_lengths_recovery = {tid: np.sum(traj_ids_recovery == tid) for tid in unique_ids_recovery}
            sorted_ids_recovery = sorted(unique_ids_recovery, key=lambda tid: traj_lengths_recovery[tid], reverse=True)
            num_remove_recovery = max(1, int(len(sorted_ids_recovery) * filter_top_pct))
            
            mask_recovery = ~np.isin(traj_ids_recovery, sorted_ids_recovery[:num_remove_recovery])
            states_recovery = states_recovery[mask_recovery]
            actions_recovery = actions_recovery[mask_recovery]
            traj_ids_recovery = traj_ids_recovery[mask_recovery]
            if steps_recovery is not None:
                steps_recovery = steps_recovery[mask_recovery]
            
            num_normal_samples = len(states_normal)
            num_recovery_samples = len(states_recovery)
            samples_per_source = min(num_normal_samples, num_recovery_samples)
            
            if num_normal_samples > samples_per_source:
                indices = np.random.choice(num_normal_samples, samples_per_source, replace=False)
                states_normal = states_normal[indices]
                actions_normal = actions_normal[indices]
                traj_ids_normal = traj_ids_normal[indices]
                if steps_normal is not None:
                    steps_normal = steps_normal[indices]
            
            if num_recovery_samples > samples_per_source:
                indices = np.random.choice(num_recovery_samples, samples_per_source, replace=False)
                states_recovery = states_recovery[indices]
                actions_recovery = actions_recovery[indices]
                traj_ids_recovery = traj_ids_recovery[indices]
                if steps_recovery is not None:
                    steps_recovery = steps_recovery[indices]
            
            max_normal_traj_id = traj_ids_normal.max() if len(traj_ids_normal) > 0 else -1
            traj_ids_recovery = traj_ids_recovery + max_normal_traj_id + 1
            
            # Concatenate
            states = np.concatenate([states_normal, states_recovery], axis=0)
            actions = np.concatenate([actions_normal, actions_recovery], axis=0)
            traj_ids = np.concatenate([traj_ids_normal, traj_ids_recovery], axis=0)
            
            if steps_normal is not None and steps_recovery is not None:
                steps = np.concatenate([steps_normal, steps_recovery], axis=0)
            elif steps_normal is not None:
                steps = steps_normal
            elif steps_recovery is not None:
                steps = steps_recovery
            else:
                steps = None
            
            print(f"  ✓ Mixed dataset: {len(states_normal)} normal + {len(states_recovery)} recovery = {len(states)} total samples")
        except (FileNotFoundError, ValueError, KeyError) as e:
            print(f"  ⚠️  Could not load recovery dataset: {e}")
            print("  Using normal dataset only")
            states = states_normal
            actions = actions_normal
            traj_ids = traj_ids_normal
            steps = steps_normal
    else:
        states = states_normal
        actions = actions_normal
        traj_ids = traj_ids_normal
        steps = steps_normal
    
    return states, actions, traj_ids, steps

# ============================================================================
# Model Loading
# ============================================================================

def load_or_create_model(target_trajectories, states_shape, actions_shape, device, model_name_prefix='default', MLP_class=None):
    """
    Load existing model if available and compatible, otherwise return None.
    
    Args:
        target_trajectories: Number of trajectories used for this model
        states_shape: Shape of states array
        actions_shape: Shape of actions array
        device: Device to load model on
        model_name_prefix: Prefix for model filename (e.g., 'even_simpler', 'direct')
        MLP_class: MLP class to instantiate (required)
    
    Returns:
        model: Loaded model or None
        model_file: Model filename or None
        success_rate: Success rate parsed from filename or None
    """
    if MLP_class is None:
        raise ValueError("MLP_class must be provided to load_or_create_model")
    
    model_dir = get_model_path(model_name_prefix)
    os.makedirs(model_dir, exist_ok=True)
    pattern = os.path.join(model_dir, f"model_{model_name_prefix}_{target_trajectories}traj_*success.pth")
    existing = glob.glob(pattern)
    if existing:
        # Check if the model's input size matches the current state size
        checkpoint = torch.load(existing[0], map_location=device, weights_only=False)
        if 'fc1.weight' in checkpoint:
            model_input_size = checkpoint['fc1.weight'].shape[1]
            current_input_size = states_shape[1]
            
            if model_input_size != current_input_size:
                print(f"  ⚠️  Model architecture mismatch detected!")
                print(f"  Model expects: {model_input_size} input features")
                print(f"  Current state representation: {current_input_size} features")
                print(f"  This mismatch is likely due to a state representation change (e.g., adding sin/cos encoding).")
                print(f"  Will train a new model with the correct input size instead of using the existing one.")
                return None, None, None
        
        model = MLP_class(states_shape[1], actions_shape[1])
        model.load_state_dict(checkpoint)
        model.to(device)
        # Parse success rate from filename
        try:
            success_rate = int(existing[0].split('_')[-1].replace('success.pth', '')) / 100.0
        except (ValueError, IndexError):
            success_rate = None
        return model, existing[0], success_rate
    return None, None, None

# ============================================================================
# Training Pipeline Utilities
# ============================================================================

def train_and_evaluate_step(target_trajectories, states, actions, traj_ids, steps,
                            batch_size, model_name_prefix, MLP_class, train_model_fn,
                            pretrained_model_path=None, joint_ranges=None, use_action_normalization=False,
                            evaluate_model_fn=None):
    """
    Single training/evaluation step - unified version for all pipelines.
    
    Args:
        target_trajectories: Number of trajectories for this model
        states: State vectors
        actions: Action vectors
        traj_ids: Trajectory IDs
        steps: Step indices (optional)
        batch_size: Batch size for training
        model_name_prefix: Prefix for model/scaler filenames (e.g., 'delta', 'taskspace', 'even_simpler', 'direct')
        MLP_class: MLP class to use
        train_model_fn: Function to call for training: train_model_fn(states, actions_normalized, traj_ids, steps, ...) -> (model, scaler)
        pretrained_model_path: Optional path to pretrained model for fine-tuning
        joint_ranges: Optional joint ranges for action normalization (if use_action_normalization=True)
        use_action_normalization: If True, normalizes actions using joint_ranges
        evaluate_model_fn: Function to call for evaluation: evaluate_model_fn(model, ...) -> (success_rate, results)
                          If None, uses default evaluation (20 objects)
    
    Returns:
        model: Trained/loaded model
        scaler: State scaler
        action_scaler: Action scaler (joint_ranges if normalized, None otherwise)
        success_rate: Success rate from evaluation
        model_file: Model filename
    """
    # Auto-detect device (defaults to MPS > CUDA > CPU)
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    model, model_file, success_rate = load_or_create_model(
        target_trajectories, states.shape, actions.shape, device,
        model_name_prefix=model_name_prefix, MLP_class=MLP_class)
    
    if model is not None and success_rate is not None:
        scaler = load_state_scaler(target_trajectories, base_name=model_name_prefix)
        if scaler is not None:
            expected_features = scaler.n_features_in_ if hasattr(scaler, 'n_features_in_') else scaler.mean_.shape[0] if hasattr(scaler, 'mean_') and len(scaler.mean_.shape) > 0 else None
            if expected_features is not None and expected_features != states.shape[1]:
                print(f"  ⚠️  Loaded state scaler expects {expected_features} features but states have {states.shape[1]} features.")
                print(f"  State scaler is incompatible with new state representation (sin/cos encoding).")
                print(f"  Will refit scaler during evaluation if needed.")
                scaler = None  # Set to None so it gets refit during evaluation
        
        if use_action_normalization:
            action_scaler = load_action_scaler(target_trajectories, base_name=model_name_prefix)
        else:
            action_scaler = None
        return model, scaler, action_scaler, success_rate, model_file
    
    if use_action_normalization:
        if joint_ranges is not None:
            print(f"  Normalizing actions to [0, 1] range...")
            actions_normalized = np.array([normalize_action(action, joint_ranges, clamp_input=False) for action in actions])
            save_action_scaler(joint_ranges, target_trajectories, base_name=model_name_prefix)
            action_scaler = joint_ranges
        else:
            # Try to load existing action scaler
            action_scaler = load_action_scaler(target_trajectories, base_name=model_name_prefix)
            if action_scaler is not None:
                print(f"  Loading action scaler and normalizing actions...")
                actions_normalized = np.array([normalize_action(action, action_scaler, clamp_input=False) for action in actions])
                joint_ranges = action_scaler
            else:
                print(f"  ⚠️  No joint ranges provided, training without action normalization")
                actions_normalized = actions
                action_scaler = None
    else:
        # No normalization for delta/task-space actions
        print(f"  Training on {model_name_prefix} actions (no normalization)")
        actions_normalized = actions  # Actions are already in natural units
        action_scaler = None
    
    # Train new model (with fine-tuning if pretrained_model_path provided)
    model, scaler = train_model_fn(states, actions_normalized, traj_ids, steps,
                                    epochs=None, batch_size=batch_size,
                                    patience=25, min_epochs=100, pretrained_model_path=pretrained_model_path)
    save_state_scaler(scaler, target_trajectories, base_name=model_name_prefix)
    
    # Save model before evaluation (in case evaluation hangs)
    model_dir = get_model_path(model_name_prefix)
    os.makedirs(model_dir, exist_ok=True)
    model_filename_temp = os.path.join(model_dir, f"model_{model_name_prefix}_{target_trajectories}traj_temp.pth")
    torch.save(model.state_dict(), model_filename_temp)
    
    if evaluate_model_fn is not None:
        success_rate, _ = evaluate_model_fn(model, num_objects=20, state_scaler=scaler, action_scaler=action_scaler)
    else:
        # Default evaluation - this should not happen if properly configured
        raise ValueError("evaluate_model_fn must be provided")
    
    model_file = os.path.join(model_dir, f"model_{model_name_prefix}_{target_trajectories}traj_{int(success_rate*100)}success.pth")
    torch.save(model.state_dict(), model_file)
    
    if os.path.exists(model_filename_temp):
        try:
            os.remove(model_filename_temp)
        except:
            pass
    
    return model, scaler, action_scaler, success_rate, model_file

# ============================================================================
# Unified Training Function
# ============================================================================

def train_model_unified(states, actions, trajectory_ids=None, steps=None, epochs=None, batch_size=256, lr=0.001, 
                        patience=25, min_delta=1e-6, state_scaler=None, max_epochs=1000, min_epochs=100, 
                        pretrained_model_path=None, MLP_class=None, finetune_lr_scale=0.5,
                        lr_scheduler='plateau', lr_decay_factor=0.5, lr_patience=10, lr_step_size=30, lr_gamma=0.1,
                        weight_decay=1e-5):
    """
    Unified train_model function for all pipelines.
    Train MLP model with early stopping based on validation loss.
    
    Args:
        states: State vectors (N, state_dim)
        actions: Action vectors (N, action_dim)
        trajectory_ids: Optional trajectory IDs for weighted sampling
        steps: Optional step indices
        epochs: Number of training epochs
        batch_size: Batch size for training
        lr: Initial learning rate
        patience: Early stopping patience (number of validation checks with no improvement, default: 25)
        min_delta: Minimum improvement for early stopping
        state_scaler: Optional StandardScaler for state normalization (if None, will fit one)
        max_epochs: Maximum number of epochs to train
        min_epochs: Minimum number of epochs to train before early stopping can trigger (default: 100)
        pretrained_model_path: Optional path to pretrained model for fine-tuning
        MLP_class: MLP class to instantiate (required)
        finetune_lr_scale: Learning rate scaling factor for fine-tuning (default: 0.5)
        lr_scheduler: Learning rate scheduler type ('plateau', 'step', 'exponential', 'cosine', or None)
                     - 'plateau': ReduceLROnPlateau (reduces LR when validation loss plateaus, recommended)
                     - 'step': StepLR (reduces LR every N epochs)
                     - 'exponential': ExponentialLR (exponential decay)
                     - 'cosine': CosineAnnealingLR (cosine annealing)
                     - None: No learning rate decay
        lr_decay_factor: Factor by which to reduce learning rate (for plateau/step/exponential, default: 0.5)
        lr_patience: Patience for ReduceLROnPlateau (number of validation checks with no improvement, default: 10)
        lr_step_size: Step size for StepLR (epochs between LR reductions, default: 30)
        lr_gamma: Multiplicative factor for StepLR/ExponentialLR (default: 0.1)
    
    Returns:
        model: Trained model
        state_scaler: Fitted state scaler (for use during evaluation)
    """
    if MLP_class is None:
        raise ValueError("MLP_class must be provided")
    
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    if state_scaler is None:
        print(f"  Fitting state scaler to normalize states...")
        state_scaler = StandardScaler()
        states_normalized = state_scaler.fit_transform(states)
        print(f"    State means: {state_scaler.mean_[:5]}... (showing first 5)")
        print(f"    State stds: {state_scaler.scale_[:5]}... (showing first 5)")
    else:
        expected_features = state_scaler.n_features_in_ if hasattr(state_scaler, 'n_features_in_') else state_scaler.mean_.shape[0] if hasattr(state_scaler, 'mean_') and len(state_scaler.mean_.shape) > 0 else None
        if expected_features is not None and expected_features != states.shape[1]:
            print(f"  ⚠️  State scaler expects {expected_features} features but states have {states.shape[1]} features.")
            print(f"  Refitting state scaler to match new state representation (sin/cos encoding)...")
            state_scaler = StandardScaler()
            states_normalized = state_scaler.fit_transform(states)
            print(f"    State means: {state_scaler.mean_[:5]}... (showing first 5)")
            print(f"    State stds: {state_scaler.scale_[:5]}... (showing first 5)")
        else:
            print(f"  Using provided state scaler...")
            states_normalized = state_scaler.transform(states)
    
    train_indices = None
    train_steps = None
    if trajectory_ids is not None:
        # CRITICAL: Split by trajectory ID to avoid data leakage (samples from same trajectory must stay together)
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        train_idx, val_idx = next(gss.split(states_normalized, actions, groups=trajectory_ids))
        
        X_train = states_normalized[train_idx]
        X_val = states_normalized[val_idx]
        y_train = actions[train_idx]
        y_val = actions[val_idx]
        train_traj_ids = trajectory_ids[train_idx]
        val_traj_ids = trajectory_ids[val_idx]
        train_indices = train_idx
        val_indices = val_idx
        
        if steps is not None:
            train_steps = steps[train_idx]
            val_steps = steps[val_idx]
        else:
            train_steps = None
            val_steps = None
        
        train_traj_set = set(train_traj_ids)
        val_traj_set = set(val_traj_ids)
        overlap = train_traj_set & val_traj_set
        if overlap:
            print(f"  ⚠️  WARNING: {len(overlap)} trajectories appear in both train and validation sets!")
            print(f"  This indicates a bug in trajectory-aware splitting.")
        else:
            print(f"  ✓ Trajectory-aware split: {len(train_traj_set)} train trajectories, {len(val_traj_set)} validation trajectories (no overlap)")
            # Diagnostic: Check if there are differences in action magnitudes
            train_action_magnitude = np.mean(np.linalg.norm(y_train, axis=1))
            val_action_magnitude = np.mean(np.linalg.norm(y_val, axis=1))
            print(f"    Train action magnitude (mean): {train_action_magnitude:.6f}")
            print(f"    Val action magnitude (mean): {val_action_magnitude:.6f}")
            if abs(train_action_magnitude - val_action_magnitude) / train_action_magnitude > 0.1:
                print(f"    ⚠️  Warning: Significant difference in action magnitudes between train/val sets")
    else:
        # No trajectory IDs available - use standard random split
        # This is acceptable if data doesn't have trajectory structure
        if steps is not None:
            X_train, X_val, y_train, y_val, train_steps, val_steps = train_test_split(
                states_normalized, actions, steps, test_size=0.2, random_state=42
            )
        else:
            X_train, X_val, y_train, y_val = train_test_split(
                states_normalized, actions, test_size=0.2, random_state=42
            )
        train_traj_ids = None
        val_traj_ids = None
    
    X_train_tensor = torch.FloatTensor(X_train).to(device)
    y_train_tensor = torch.FloatTensor(y_train).to(device)
    X_val_tensor = torch.FloatTensor(X_val).to(device)
    y_val_tensor = torch.FloatTensor(y_val).to(device)
    
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    
    val_batch_size = min(batch_size * 4, 32768)
    
    # PyTorch's WeightedRandomSampler has a limit of 2^24 samples (16,777,216)
    # For larger datasets, fall back to regular shuffle
    MAX_SAMPLER_SIZE = 2**24 - 1000000  # Use 15M as safe threshold (leave some margin)
    
    if train_traj_ids is not None and len(train_traj_ids) == len(X_train) and len(X_train) < MAX_SAMPLER_SIZE:
        unique_trajs, counts = np.unique(train_traj_ids, return_counts=True)
        
        traj_weights = 1.0 / (counts + 1e-8)
        traj_weights = traj_weights / traj_weights.sum() * len(unique_trajs)
        
        sample_weights = np.zeros(len(X_train))
        for traj_id, weight in zip(unique_trajs, traj_weights):
            mask = train_traj_ids == traj_id
            sample_weights[mask] = weight
        
        sample_weights_tensor = torch.FloatTensor(sample_weights)
        
        sampler = WeightedRandomSampler(
            weights=sample_weights_tensor,
            num_samples=len(sample_weights_tensor),
            replacement=True
        )
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler,
                                 num_workers=0, pin_memory=False)  # MPS doesn't support pin_memory or prefetch_factor with num_workers=0
        
        print(f"    Using weighted sampling: {len(unique_trajs)} trajectories, weight range: [{traj_weights.min():.4f}, {traj_weights.max():.4f}]")
        print(f"    Each trajectory contributes equally regardless of length")
    else:
        if train_traj_ids is not None and len(train_traj_ids) == len(X_train) and len(X_train) >= MAX_SAMPLER_SIZE:
            print(f"    Dataset too large ({len(X_train):,} samples) for WeightedRandomSampler (limit: {MAX_SAMPLER_SIZE:,})")
            print(f"    Falling back to regular shuffle (weighted sampling disabled for large datasets)")
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                 num_workers=0, pin_memory=False)  # MPS doesn't support pin_memory or prefetch_factor with num_workers=0
    
    val_loader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False, 
                           num_workers=0, pin_memory=False)  # MPS doesn't support pin_memory or prefetch_factor with num_workers=0
    
    model = MLP_class(input_size=states_normalized.shape[1], output_size=actions.shape[1])
    
    if pretrained_model_path and os.path.exists(pretrained_model_path):
        try:
            print(f"  Loading pretrained model from {pretrained_model_path}...")
            pretrained_state = torch.load(pretrained_model_path, map_location=device, weights_only=False)
            model.load_state_dict(pretrained_state, strict=False)
            print(f"  ✓ Loaded pretrained weights (fine-tuning)")
            original_lr = lr
            lr = lr * finetune_lr_scale
            print(f"  Using reduced learning rate for fine-tuning: {original_lr} -> {lr} (x{finetune_lr_scale})")
        except Exception as e:
            print(f"  ⚠️  Could not load pretrained model: {e}")
            print(f"  Training from scratch instead...")
    else:
        print(f"  Created new model (training from scratch)")
    
    print(f"  Moving model to {device}...")
    model.to(device)
    
    print(f"  Setting up optimizer (initial lr={lr}, weight_decay={weight_decay})...")
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    max_epochs_to_use = epochs if epochs is not None else max_epochs
    
    scheduler = None
    if lr_scheduler is not None and lr_scheduler.lower() != 'none':
        if lr_scheduler.lower() == 'plateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=lr_decay_factor, patience=lr_patience,
                min_lr=1e-7
            )
            print(f"  Using ReduceLROnPlateau scheduler (factor={lr_decay_factor}, patience={lr_patience} validation checks)")
        elif lr_scheduler.lower() == 'step':
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=lr_step_size, gamma=lr_gamma
            )
            print(f"  Using StepLR scheduler (step_size={lr_step_size}, gamma={lr_gamma})")
        elif lr_scheduler.lower() == 'exponential':
            scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer, gamma=lr_gamma
            )
            print(f"  Using ExponentialLR scheduler (gamma={lr_gamma})")
        elif lr_scheduler.lower() == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max_epochs_to_use, eta_min=lr * 0.01
            )
            print(f"  Using CosineAnnealingLR scheduler (T_max={max_epochs_to_use}, eta_min={lr * 0.01})")
        else:
            print(f"  ⚠️  Unknown scheduler type '{lr_scheduler}', using no scheduler")
            scheduler = None
    else:
        print(f"  No learning rate scheduler (using constant lr={lr})")
    
    use_amp = device.type in ['mps', 'cuda']
    if use_amp:
        try:
            if device.type == 'mps':
                scaler = torch.amp.GradScaler('mps')
            else:
                scaler = torch.cuda.amp.GradScaler()
        except:
            use_amp = False
            scaler = None
    else:
        scaler = None
    
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    val_frequency = 2
    
    total_batches = len(train_loader)
    
    if epochs is None:
        print(f"  Starting training: minimum {min_epochs} epochs, up to {max_epochs_to_use} epochs (early stopping enabled), {total_batches} total batches per epoch")
    else:
        print(f"  Starting training: {epochs} epochs, {total_batches} total batches per epoch")
    print(f"  Validation every {val_frequency} epoch(s)")
    print(f"  Early stopping patience: {patience} validation checks (stops if no improvement for {patience} checks = {patience * val_frequency} epochs, but only after {min_epochs} epochs minimum)")
    if scheduler is not None:
        print(f"  Learning rate scheduler: {type(scheduler).__name__}")
    print(f"  Press Ctrl-C to stop training early and save model")
    try:
        epoch = 0
        while epoch < max_epochs_to_use:
            model.train()
            train_loss = 0.0
            train_samples = 0
            
            for batch_X, batch_y in train_loader:
                batch_size = batch_X.size(0)
                
                optimizer.zero_grad()
                
                if use_amp:
                    with torch.amp.autocast(device_type='mps' if device.type == 'mps' else 'cuda'):
                        outputs = model(batch_X)
                        loss = criterion(outputs, batch_y)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
                
                train_loss += loss.item() * batch_size
                train_samples += batch_size
            
            train_loss /= train_samples if train_samples > 0 else 1
            
            if (epoch + 1) % 1 == 0:
                if epochs is None:
                    print(f"Epoch [{epoch+1}], Train Loss: {train_loss:.6f}", end='')
                else:
                    print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.6f}", end='')
            
            if (epoch + 1) % val_frequency == 0 or epoch == 0:
                model.eval()
                val_loss = 0.0
                val_samples = 0
                with torch.no_grad():
                    for batch_X, batch_y in val_loader:
                        batch_size = batch_X.size(0)
                        if use_amp:
                            with torch.amp.autocast(device_type='mps' if device.type == 'mps' else 'cuda'):
                                outputs = model(batch_X)
                                loss = criterion(outputs, batch_y)
                        else:
                            outputs = model(batch_X)
                            loss = criterion(outputs, batch_y)
                        val_loss += loss.item() * batch_size
                        val_samples += batch_size
                val_loss /= val_samples if val_samples > 0 else 1
                if (epoch + 1) % 1 == 0:
                    current_lr = optimizer.param_groups[0]['lr']
                    if scheduler is not None and isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        print(f", Val Loss: {val_loss:.6f}, LR: {current_lr:.2e}")
                    else:
                        print(f", Val Loss: {val_loss:.6f}")
            else:
                val_loss = best_val_loss if best_val_loss != float('inf') else train_loss
                if (epoch + 1) % 1 == 0:
                    print()
            
            if (epoch + 1) % val_frequency == 0 or epoch == 0:
                if val_loss < best_val_loss - min_delta:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_model_state = model.state_dict().copy()
                else:
                    patience_counter += 1
            
                if scheduler is not None and isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_loss)
            
            if scheduler is not None and not isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step()
                current_lr = optimizer.param_groups[0]['lr']
                if (epoch + 1) % val_frequency == 0 and current_lr < lr * 0.99:
                    print(f"    LR: {current_lr:.2e}")
            
            if epoch >= min_epochs and patience_counter >= patience:
                effective_epochs_waited = patience_counter * val_frequency
                print(f"\n  ✓ Early stopping triggered: no improvement for {patience_counter} validation checks ({effective_epochs_waited} epochs, after minimum {min_epochs} epochs)")
                break
            
            epoch += 1
    except KeyboardInterrupt:
        if epochs is None:
            print(f"\n  ⚠️  Training interrupted by user at epoch {epoch+1}")
        else:
            print(f"\n  ⚠️  Training interrupted by user at epoch {epoch+1}/{epochs}")
        print(f"  Saving current model state...")
    
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"  ✓ Loaded best model state (val_loss: {best_val_loss:.6f})")
    else:
        print(f"  ✓ Using current model state")
    
    return model, state_scaler

# ============================================================================
# Unified Evaluation Helpers
# ============================================================================

def evaluate_model_unified(model, num_objects, max_steps, state_scaler, 
                           xml_path, dt, num_seconds, get_robot_start_fn, initialize_block_fn,
                           use_sin_cos_encoding, apply_action_fn, placement_box_pos, placement_box_size):
    """
    Unified evaluate_model function for all pipelines.
    Evaluate model on unseen objects with random starting positions.
    Success is defined as block placed on the green placement box.
    
    Args:
        model: Trained model
        num_objects: Number of random objects to test
        max_steps: Maximum steps per trial. If None, calculates from expert trajectory length.
        state_scaler: Optional StandardScaler for state normalization (must match training)
        xml_path: Path to MuJoCo XML file
        dt: Simulation timestep
        num_seconds: Number of seconds for expert trajectory (used to calculate max_steps if None)
        get_robot_start_fn: Function to get robot start position
        initialize_block_fn: Function to initialize block
        use_sin_cos_encoding: Whether to use sin/cos encoding for states
        apply_action_fn: Callback function(model_output, data, model_mj, site_id, block_id, dof_ids, actuator_ids, gripper_actuator_id) -> None
        placement_box_pos: Placement box position
        placement_box_size: Placement box size
    
    Returns:
        success_rate: Success rate (0.0 to 1.0)
        results: List of result dicts
    """
    # Auto-detect device (defaults to MPS > CUDA > CPU)
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    # Calculate reasonable max_steps based on expert trajectory length
    if max_steps is None:
        expert_trajectory_steps = int(num_seconds / dt)  # ~20000 steps for 40 seconds
        # For evaluation, use a reasonable multiplier (10x expert time) with a cap
        # Model should succeed or fail much sooner than data collection
        max_steps = min(expert_trajectory_steps * 10, 200000)  # 10x expert time, cap at 200k steps (~6.7 minutes per trial)
    
    model.eval()
    successes = 0
    results = []
    
    for obj_idx in range(num_objects):
        # Setup simulation
        model_mj, data, block_id, site_id, block_geom_id, dof_ids, actuator_ids, gripper_actuator_id, block_jnt_id, key_id = setup_mujoco_model(xml_path, dt)
        
        # Use fixed robot starting position (home position, no randomization)
        qpos_start = get_robot_start_fn(model_mj, dof_ids, key_id)
        data.qpos[:7] = qpos_start[:7]
        mujoco.mj_forward(model_mj, data)
        block_size = initialize_block_fn(model_mj, data, block_geom_id, block_jnt_id)
        
        # Rollout policy
        success = False
        for step in range(max_steps):
            # Extract state
            state = extract_state_from_simulation(data, block_id, site_id, block_size, use_sin_cos_encoding=use_sin_cos_encoding)
            
            # Normalize state if scaler is provided
            if state_scaler is not None:
                # Check if scaler matches current state size (might be old scaler from before sin/cos encoding)
                expected_features = state_scaler.n_features_in_ if hasattr(state_scaler, 'n_features_in_') else state_scaler.mean_.shape[0] if hasattr(state_scaler, 'mean_') and len(state_scaler.mean_.shape) > 0 else None
                if expected_features is not None and expected_features != len(state):
                    if expected_features in [31, 38] and len(state) in [31, 38]:
                        if obj_idx == 0:  # Only print info once
                            print(f"  Converting state from {len(state)}D to {expected_features}D to match scaler")
                        state = convert_state_dimension(state, expected_features)
                    else:
                        if obj_idx == 0:  # Only print warning once
                            print(f"  ⚠️  Warning: State scaler expects {expected_features} features but state has {len(state)} features.")
                            print(f"  Skipping state normalization (using raw state). This may cause poor performance.")
                else:
                    state = state_scaler.transform(state.reshape(1, -1)).flatten()
            
            # Get action from policy
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                model_output = model(state_tensor).squeeze(0).cpu().numpy()
            
            apply_action_fn(model_output, data, model_mj, site_id, block_id, dof_ids, actuator_ids, gripper_actuator_id)
            
            mujoco.mj_step(model_mj, data)
            
            # Check Success
            if step % 100 == 0:
                mujoco.mj_forward(model_mj, data)
                success_check, block_pos_check, _ = verify_placement_on_green(
                    model_mj, data, block_id, placement_box_pos, placement_box_size)
                if success_check:
                    success = True
                    break
        
        # Let block settle before final check
        settling_steps = int(1.0 / dt)
        for _ in range(settling_steps):
            mujoco.mj_step(model_mj, data)
        
        # Final check
        mujoco.mj_forward(model_mj, data)
        success_final, block_pos_final, message = verify_placement_on_green(
            model_mj, data, block_id, placement_box_pos, placement_box_size)
        
        if success_final:
            successes += 1
        
        results.append({
            'object_idx': obj_idx,
            'success': success_final,
            'block_size': block_size.tolist(),
            'block_final_pos': block_pos_final.tolist(),
            'message': message
        })
    
    success_rate = successes / num_objects
    return success_rate, results

# ============================================================================
# Placement Verification
# ============================================================================

def verify_placement_on_green(model, data, block_id, placement_box_pos=None, placement_box_size=None, 
                              table_surface_z=0.2, tolerance=0.02):
    """
    Check if block is placed on the green placement box.
    
    Args:
        model: MuJoCo model
        data: MuJoCo data
        block_id: Block body ID
        placement_box_pos: Position of placement box [x, y] (optional, falls back to trajectoriesSimpler if None)
        placement_box_size: Size of placement box [half_width_x, half_width_y] (optional, falls back to trajectoriesSimpler if None)
        table_surface_z: Z-coordinate of table surface
        tolerance: Position tolerance in meters (for z-check only; box bounds use fp_tolerance=0.001)
    
    Returns:
        success: True if block is on green box
        block_pos: Block position
        message: Status message
    """
    if placement_box_pos is None or placement_box_size is None:
        if default_placement_box_pos is not None and default_placement_box_size is not None:
            placement_box_pos = placement_box_pos if placement_box_pos is not None else default_placement_box_pos
            placement_box_size = placement_box_size if placement_box_size is not None else default_placement_box_size
        else:
            raise ValueError("placement_box_pos and placement_box_size must be provided if trajectoriesSimpler cannot be imported")
    
    mujoco.mj_forward(model, data)
    block_pos = data.xpos[block_id]
    
    if block_pos[2] < table_surface_z + tolerance:
        return False, block_pos, f"Block below table: z={block_pos[2]:.4f}m"
    
    fp_tolerance = 0.001
    x_min = placement_box_pos[0] - placement_box_size[0]
    x_max = placement_box_pos[0] + placement_box_size[0]
    y_min = placement_box_pos[1] - placement_box_size[1]
    y_max = placement_box_pos[1] + placement_box_size[1]
    
    x_in_box = (x_min - fp_tolerance <= block_pos[0] <= x_max + fp_tolerance)
    y_in_box = (y_min - fp_tolerance <= block_pos[1] <= y_max + fp_tolerance)
    
    if x_in_box and y_in_box:
        return True, block_pos, f"Success: Block on green box at ({block_pos[0]:.4f}, {block_pos[1]:.4f}, {block_pos[2]:.4f})"
    else:
        return False, block_pos, f"Failed: Block at ({block_pos[0]:.4f}, {block_pos[1]:.4f}, {block_pos[2]:.4f}), not on green box"


# ============================================================================
# Task-Space Utilities
# ============================================================================

def compute_desired_pose_from_delta(data, site_id, delta_ee_pos, delta_ee_quat, delta_scale=0.1):
    """
    Convert task-space delta to desired pose (position and quaternion).
    
    Args:
        data: MuJoCo data
        site_id: End-effector site ID
        delta_ee_pos: Position delta (3,)
        delta_ee_quat: Quaternion delta (3,) - 3D representation from extract_action_taskspace_delta
        delta_scale: Scale factor for delta (default: 0.1)
    
    Returns:
        desired_pos: Desired end-effector position (3,)
        desired_quat: Desired end-effector quaternion (4,)
    """
    current_ee_pos = data.site_xpos[site_id]
    site_mat = data.site_xmat[site_id].reshape(3, 3)
    current_ee_quat = np.zeros(4)
    mujoco.mju_mat2Quat(current_ee_quat, site_mat.flatten())
    current_ee_quat = current_ee_quat / (np.linalg.norm(current_ee_quat) + 1e-8)
    
    desired_ee_pos = current_ee_pos + delta_ee_pos * delta_scale
    
    delta_quat_norm_sq = np.dot(delta_ee_quat, delta_ee_quat)
    if delta_quat_norm_sq > 1.0:
        delta_ee_quat = delta_ee_quat / np.sqrt(delta_quat_norm_sq) * 0.999
        delta_quat_norm_sq = 0.998001
    
    quat_diff_w = np.sqrt(max(0, 1.0 - delta_quat_norm_sq))
    quat_diff = np.concatenate([[quat_diff_w], delta_ee_quat])
    quat_diff = quat_diff / (np.linalg.norm(quat_diff) + 1e-8)
    
    delta_quat_norm = np.linalg.norm(delta_ee_quat)
    if delta_quat_norm < 1e-6:
        quat_diff_scaled = np.array([1.0, 0.0, 0.0, 0.0])
    else:
        theta_half = np.arcsin(np.clip(delta_quat_norm, 0, 1.0))
        theta_half_scaled = theta_half * delta_scale
        axis = delta_ee_quat / (delta_quat_norm + 1e-8)
        quat_diff_scaled = np.zeros(4)
        quat_diff_scaled[0] = np.cos(theta_half_scaled)
        quat_diff_scaled[1:] = np.sin(theta_half_scaled) * axis
        quat_diff_scaled = quat_diff_scaled / (np.linalg.norm(quat_diff_scaled) + 1e-8)
    
    # Multiply: desired_quat = quat_diff_scaled * current_quat
    # This applies the scaled rotation delta to the current quaternion
    desired_ee_quat = np.zeros(4)
    mujoco.mju_mulQuat(desired_ee_quat, quat_diff_scaled, current_ee_quat)
    desired_ee_quat = desired_ee_quat / (np.linalg.norm(desired_ee_quat) + 1e-8)
    
    return desired_ee_pos, desired_ee_quat


# ============================================================================
# CSV Saving Utilities
# ============================================================================

def save_trajectory_rows_to_csv(pending_rows, current_filename, successes, verbose=True):
    """
    Save pending trajectory rows to CSV file, appending to existing data if file exists.
    
    Args:
        pending_rows: List of dictionaries with trajectory data
        current_filename: Path to CSV file
        successes: Current number of successes (for logging)
        verbose: Whether to print status messages
    
    Returns:
        True if successful, False otherwise
    """
    if not pending_rows:
        return True
    
    if os.path.exists(current_filename) and os.path.getsize(current_filename) > 0:
        try:
            df = pd.read_csv(current_filename)
        except (pd.errors.EmptyDataError, ValueError):
            df = pd.DataFrame()
    else:
        df = pd.DataFrame()
    
    df_new = pd.DataFrame(pending_rows)
    df = pd.concat([df, df_new], ignore_index=True)
    
    try:
        df.to_csv(current_filename, index=False)
        if verbose:
            print(f"  ✓ Saved to {current_filename} ({len(df)} total rows, {successes} successes)")
        return True
    except (IOError, OSError, pd.errors.ParserError) as e:
        if verbose:
            print(f"  Error: Failed to save to {current_filename}: {e}")
        return False


# ============================================================================
# Unified Trajectory Collection
# ============================================================================

def collect_trajectory_data_unified(
    target_successes,
    base_filename,
    trajectories_per_file,
    save_frequency,
    xml_path,
    dt,
    num_seconds,
    # Callbacks for pipeline-specific behavior
    get_robot_start_fn,
    initialize_block_fn,
    create_trajectory_fn,
    extract_action_fn,
    verify_placement_fn,
    # Optional parameters
    setup_model_fn=None,  # If None, uses setup_mujoco_model
    move_arm_fn=None,  # If None, imports from ik_utils
    move_arm_kwargs=None,  # Extra kwargs for moveArm (e.g., damping, max_angvel)
    placement_box_pos=None,
    placement_box_size=None,
    use_sin_cos_encoding=False,
    # Constants (hardcoded, not parameters):
    # error_threshold=0.03
    # max_steps_per_waypoint=50
    # settle_steps=int(1.0 / dt)
    # placement_tolerance=0.02
    # table_surface_z=0.2
):
    """
    Unified function to collect trajectory data across all pipeline variants.
    
    This function handles the common trajectory collection loop, with callbacks
    for pipeline-specific differences (robot start, block init, action extraction, etc.).
    
    Args:
        target_successes: Target number of successful trajectories
        base_filename: Base filename for CSV files
        trajectories_per_file: Number of trajectories per CSV file
        save_frequency: Save every N successful trajectories
        xml_path: Path to MuJoCo XML file
        dt: Simulation timestep
        num_seconds: Trajectory duration in seconds
        get_robot_start_fn: Function(model, dof_ids, key_id) -> qpos_start
        initialize_block_fn: Function(model, data, block_geom_id, block_jnt_id, ...) -> block_size
        create_trajectory_fn: Function(model, data, block_id, site_id, block_geom_id, timesteps) -> (trajectory, shutFingers, placingPhaseStart, liftingPhaseStart, liftingPhaseEnd)
        extract_action_fn: Function(data, model, site_id, block_id, dof_ids, actuator_ids, gripper_actuator_id, ...) -> action
        verify_placement_fn: Function(model, data, block_id, ...) -> (success, block_pos, message)
        setup_model_fn: Optional function to setup model (if None, uses setup_mujoco_model)
        move_arm_kwargs: Optional dict of extra kwargs for moveArm
        placement_box_pos: Placement box position (optional)
        placement_box_size: Placement box size (optional)
        use_sin_cos_encoding: Whether to use sin/cos encoding for state (for direct pipeline)
        
    Note: The following constants are hardcoded (always the same):
        error_threshold=0.03, max_steps_per_waypoint=50, settle_steps=int(1.0/dt),
        placement_tolerance=0.02, table_surface_z=0.2
    
    Returns:
        None (saves data to CSV files)
    """
    if setup_model_fn is None:
        setup_model_fn = setup_mujoco_model
    
    if move_arm_fn is None:
        # Lazy import to avoid circular dependency (ik_utils imports from training_utils)
        from .ik_utils import moveArm
        move_arm_fn = moveArm
    
    # Constants (hardcoded)
    error_threshold = 0.03
    max_steps_per_waypoint = 50
    settle_steps = int(1.0 / dt)
    placement_tolerance = 0.02
    table_surface_z = 0.2
    
    if move_arm_kwargs is None:
        move_arm_kwargs = {}
    
    # Count existing trajectories
    num_trajectories, successes = count_trajectories(base_filename)
    trajectory_id_counter = num_trajectories
    
    print(f"Starting data collection...")
    print(f"  Existing trajectories: {num_trajectories}")
    print(f"  Existing successes: {successes}")
    print(f"  Target successes: {target_successes}")
    print(f"  Need {max(0, target_successes - successes)} more successful trajectories")
    
    # Determine variation folder for datasets
    variation = get_variation_from_base_filename(base_filename)
    # get_dataset_path expects base_name (which maps to variation), so we can pass variation directly
    # since variation names match the folder names
    dataset_dir = get_dataset_path(variation)
    os.makedirs(dataset_dir, exist_ok=True)
    
    total_collected = 0
    pending_rows = []
    current_file_index = num_trajectories // trajectories_per_file
    current_filename = os.path.join(dataset_dir, f"{base_filename}_{current_file_index * trajectories_per_file}_{(current_file_index + 1) * trajectories_per_file}.csv")
    
    while successes < target_successes:
        total_collected += 1
        print(f"\nCollecting trajectory {total_collected} (Successes: {successes}/{target_successes})...")
        
        # Setup simulation
        model, data, block_id, site_id, block_geom_id, dof_ids, actuator_ids, gripper_actuator_id, block_jnt_id, key_id = setup_model_fn(xml_path, dt)
        
        # Get robot start position
        qpos_start = get_robot_start_fn(model, dof_ids, key_id)
        data.qpos[:7] = qpos_start[:7]
        mujoco.mj_forward(model, data)
        
        # Initialize block
        block_size = initialize_block_fn(model, data, block_geom_id, block_jnt_id)
        mujoco.mj_forward(model, data)
        
        # Create trajectory
        trajectory, shutFingers, placingPhaseStart, liftingPhaseStart, liftingPhaseEnd = create_trajectory_fn(
            model, data, block_id, site_id, block_geom_id, timesteps=int(num_seconds // dt)
        )
        
        trajectory_rows = []
        steps_on_waypoint = 0
        i = 0
        max_steps = len(trajectory) * 100
        
        for step in range(max_steps):
            if i < len(trajectory):
                desiredPos, desiredQuat = trajectory[i]
                shut = i > shutFingers and i < liftingPhaseStart
            else:
                desiredPos, desiredQuat = trajectory[-1]
                shut = False
            
            state = extract_state_from_simulation(data, block_id, site_id, block_size, use_sin_cos_encoding=False)
            arm_state = state[:21]
            block_state = state[21:]
            
            current_qpos_before = data.qpos[:7].copy()
            current_ee_pos_before = data.site_xpos[site_id].copy()
            current_site_mat_before = data.site_xmat[site_id].reshape(3, 3)
            current_ee_quat_before = np.zeros(4)
            mujoco.mju_mat2Quat(current_ee_quat_before, current_site_mat_before.flatten())
            current_ee_quat_before = current_ee_quat_before / (np.linalg.norm(current_ee_quat_before) + 1e-8)
            
            error = move_arm_fn(
                model, data, site_id, block_id, dof_ids, actuator_ids,
                desiredPos, desiredQuat, shut, trajectory, i, placingPhaseStart, liftingPhaseStart, liftingPhaseEnd,
                **move_arm_kwargs
            )
            
            action = extract_action_fn(
                data, model, site_id, block_id, dof_ids, actuator_ids, gripper_actuator_id,
                desiredPos, desiredQuat, trajectory, i,
                current_qpos_before=current_qpos_before,
                current_ee_pos_before=current_ee_pos_before,
                current_ee_quat_before=current_ee_quat_before
            )
            
            mujoco.mj_step(model, data)
            
            isDone = (i >= len(trajectory) - 1)
            
            trajectory_rows.append({
                'trajectory_id': trajectory_id_counter,
                'step': step,
                'kinematic_state_arm': ','.join(map(str, arm_state)),
                'kinematic_state_block': ','.join(map(str, block_state)),
                'action': ','.join(map(str, action)),
                'isDone': isDone
            })
            
            error_norm = np.linalg.norm(error)
            if error_norm < error_threshold or steps_on_waypoint > max_steps_per_waypoint:
                i += 1
                steps_on_waypoint = 0
                if i >= len(trajectory):
                    break
            else:
                steps_on_waypoint += 1
        
        # Let block settle
        for _ in range(settle_steps):
            mujoco.mj_step(model, data)
        
        # Check success
        mujoco.mj_forward(model, data)
        success, block_pos_final, message = verify_placement_fn(
            model, data, block_id, placement_box_pos, placement_box_size,
            table_surface_z=table_surface_z, tolerance=placement_tolerance
        )
        print(f"  {message}")
        
        if success:
            successes += 1
            trajectory_id_counter += 1
            
            # Update trajectory_id for all rows
            for row in trajectory_rows:
                row['trajectory_id'] = trajectory_id_counter - 1
            
            pending_rows.extend(trajectory_rows)
            
            # Check if we need to start a new file
            if trajectory_id_counter % trajectories_per_file == 0:
                current_file_index = trajectory_id_counter // trajectories_per_file
                current_filename = os.path.join(dataset_dir, f"{base_filename}_{current_file_index * trajectories_per_file}_{(current_file_index + 1) * trajectories_per_file}.csv")
            
            # Save periodically
            if successes % save_frequency == 0:
                if save_trajectory_rows_to_csv(pending_rows, current_filename, successes):
                    pending_rows = []
        else:
            print(f"  ⚠️  Skipping failed trajectory (not saving to dataset)")
    
    # Final save
    if pending_rows:
        save_trajectory_rows_to_csv(pending_rows, current_filename, successes, verbose=True)
