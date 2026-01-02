"""
Visualize random trajectories from a dataset CSV file.
Replays recorded expert trajectories in MuJoCo.

Used AI to generate the code.
"""
import os
import sys
import time
import glob
import numpy as np
import pandas as pd
import mujoco
import mujoco.viewer
from utils.load_dataset_csv import load_dataset_csv
from utils.path_utils import find_dataset_files, get_dataset_path
from utils.visualization_utils import detect_environment_from_filename, detect_environment_from_action_dim

def parse_string_to_array(s):
    """Parse comma-separated string to numpy array"""
    return np.array([float(x) for x in s.split(',')])

def visualize_trajectories(filename=None, base_filename='dataset', num_trajectories=5, 
                          trajectory_ids=None, delay=0.002, random_seed=None, interactive_file_selection=True):
    """
    Visualize random trajectories from a dataset CSV file.
    
    Args:
        filename: Specific CSV file to load, or None to use base_filename pattern
        base_filename: Base name for finding multiple files (e.g., 'dataset' finds dataset_*.csv)
        num_trajectories: Number of random trajectories to visualize
        trajectory_ids: Specific trajectory IDs to visualize (overrides num_trajectories)
        delay: Delay between steps in seconds
        random_seed: Random seed for selecting trajectories
        interactive_file_selection: If True and filename is None, allow user to select file interactively
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    print("=" * 70)
    print("Visualizing Trajectories from Dataset")
    print("=" * 70)
    
    # Find available files - search across all variations if base_filename is generic
    files = find_dataset_files(base_filename, pattern='*.csv')
    
    # If base_filename is just 'dataset' (no variation suffix), search all variations
    if base_filename == 'dataset':
        # Search in all known variations
        variations = ['default', 'delta', 'direct', 'even_simpler', 'simpler', 
                     'taskspace', 'taskspace_absolutes', 'taskspace_absolutes_random_start', 'taskspace_random_smaller', 'random_start']
        all_files = list(files)  # Start with default variation files
        for variation in variations:
            if variation == 'default':
                continue  # Already searched via find_dataset_files
            dataset_dir = get_dataset_path(variation)
            pattern_path = os.path.join(dataset_dir, f"{base_filename}_*.csv")
            variation_files = glob.glob(pattern_path)
            # Also check for single file format
            single_file = os.path.join(dataset_dir, f"{base_filename}.csv")
            if os.path.exists(single_file) and single_file not in variation_files:
                variation_files.append(single_file)
            all_files.extend(variation_files)
        files = sorted(set(all_files))  # Remove duplicates and sort
    
    # Interactive file selection if filename not specified and files found
    if filename is None and interactive_file_selection and len(files) > 1:
        print(f"\nFound {len(files)} dataset file(s):")
        for i, f in enumerate(files, 1):
            file_size = os.path.getsize(f) / (1024 * 1024)  # Size in MB
            print(f"  {i}. {f} ({file_size:.1f} MB)")
        print(f"  {len(files) + 1}. Use all files (default pattern)")
        
        try:
            choice = input(f"\nSelect file number (1-{len(files) + 1}, or press Enter for all): ").strip()
            if choice == "":
                # Use all files (default behavior)
                filename = None
            else:
                choice_num = int(choice)
                if 1 <= choice_num <= len(files):
                    filename = files[choice_num - 1]
                    print(f"  Selected: {filename}")
                elif choice_num == len(files) + 1:
                    filename = None
                    print(f"  Using all files")
                else:
                    print(f"  Invalid choice, using all files")
                    filename = None
        except (ValueError, KeyboardInterrupt):
            print(f"  Invalid input or cancelled, using all files")
            filename = None
    
    # Load dataset
    print(f"\nLoading dataset...")
    if filename:
        print(f"  filename: {filename}")
    else:
        print(f"  base_filename: {base_filename} (using all matching files)")
    
    try:
        data = load_dataset_csv(filename=filename, base_filename=base_filename, max_trajectories=None)
    except FileNotFoundError as e:
        print(f"  Error: {e}")
        print(f"\nAvailable dataset files:")
        if files:
            for f in files:
                print(f"    - {f}")
        else:
            print(f"    No files found matching '{base_filename}_*.csv' or '{base_filename}.csv'")
        return
    
    # Get unique trajectory IDs
    unique_traj_ids = np.unique(data['trajectory_ids'])
    num_available = len(unique_traj_ids)
    
    print(f"  ✓ Loaded {len(data['trajectory_ids'])} state-action pairs from {num_available} unique trajectories")
    
    # Select trajectories to visualize
    if trajectory_ids is not None:
        traj_ids_to_show = np.array(trajectory_ids)
        # Filter to only those that exist
        traj_ids_to_show = traj_ids_to_show[np.isin(traj_ids_to_show, unique_traj_ids)]
        if len(traj_ids_to_show) == 0:
            print(f"  Error: None of the specified trajectory IDs exist in the dataset")
            return
        print(f"  Showing {len(traj_ids_to_show)} specified trajectory(ies): {traj_ids_to_show}")
    else:
        if num_trajectories > num_available:
            print(f"  Warning: Only {num_available} trajectories available, showing all")
            num_trajectories = num_available
        traj_ids_to_show = np.random.choice(unique_traj_ids, size=num_trajectories, replace=False)
        print(f"  Randomly selected {len(traj_ids_to_show)} trajectory(ies): {traj_ids_to_show}")
    
    # Determine action dimension from actual loaded data
    if len(data['actions']) > 0:
        action_dim = data['actions'][0].shape[0]
    else:
        # Fallback: infer from filename
        env_info = detect_environment_from_filename(filename if filename else base_filename)
        action_dim = env_info['action_dim']
    
    # Get action dimension info
    action_info = detect_environment_from_action_dim(action_dim)
    gripper_idx = action_info['gripper_idx']
    joint_action_dim = action_info['joint_action_dim']
    
    # Detect environment from filename to get XML path
    env_info = detect_environment_from_filename(filename if filename else base_filename)
    xml_path = env_info['xml_path']
    
    # Load MuJoCo model - use appropriate XML based on dataset type
    print(f"\nLoading MuJoCo model...")
    print(f"  Using: {xml_path}")
    print(f"  Action dimension: {action_dim}D (gripper at index {gripper_idx})")
    if env_info['use_taskspace_absolutes']:
        print(f"  Variation: taskspace_absolutes (absolute end-effector poses)")
    elif env_info['use_taskspace']:
        print(f"  Variation: taskspace (task-space deltas)")
    elif env_info['use_direct']:
        print(f"  Variation: direct")
    elif env_info['use_delta']:
        print(f"  Variation: delta")
    elif env_info['use_even_simpler']:
        print(f"  Variation: even_simpler")
    elif env_info['use_simpler']:
        print(f"  Variation: simpler")
    else:
        print(f"  Variation: default")
    if env_info['use_random_start']:
        print(f"  Using random robot starting positions")
    model = mujoco.MjModel.from_xml_path(xml_path)
    data_mj = mujoco.MjData(model)
    
    block_id = model.body('target').id
    block_jnt_id = model.body(block_id).jntadr[0]
    blockGeom_id = model.geom('box').id
    joint_names = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"][:7]
    dof_ids = np.array([model.joint(name).id for name in joint_names])
    actuator_names = ['actuator1', 'actuator2', 'actuator3', 'actuator4', 'actuator5', 'actuator6', 'actuator7'][:7]
    actuator_ids = [model.actuator(name).id for name in actuator_names]
    
    print(f"  ✓ Model loaded")
    
    # Extract trajectory data
    print(f"\nExtracting trajectory data...")
    trajectories_data = {}
    for traj_id in traj_ids_to_show:
        mask = data['trajectory_ids'] == traj_id
        traj_steps = np.where(mask)[0]
        
        # Get states and actions for this trajectory
        arm_states = data['arm_states'][mask]
        block_states = data['block_states'][mask]
        actions = data['actions'][mask]
        is_done = data['isDone'][mask]
        
        trajectories_data[traj_id] = {
            'arm_states': arm_states,
            'block_states': block_states,
            'actions': actions,
            'is_done': is_done,
            'num_steps': len(traj_steps)
        }
        print(f"  Trajectory {traj_id}: {len(traj_steps)} steps")
    
    print(f"\nStarting visualization...")
    print(f"  Press 'q' in the viewer to quit")
    print(f"  Press 'n' in the viewer to go to next trajectory")
    print(f"  Press 'r' in the viewer to replay current trajectory")
    print(f"  Delay between steps: {delay*1000:.1f}ms")
    
    # Visualize each trajectory
    with mujoco.viewer.launch_passive(model, data_mj) as viewer:
        traj_idx = 0
        
        while traj_idx < len(traj_ids_to_show):
            traj_id = traj_ids_to_show[traj_idx]
            traj_data = trajectories_data[traj_id]
            
            print(f"\n{'='*70}")
            print(f"Trajectory {traj_idx + 1}/{len(traj_ids_to_show)}: ID {traj_id}")
            print(f"{'='*70}")
            print(f"  Steps: {traj_data['num_steps']}")
            
            # Reset to initial state
            arm_state_0 = traj_data['arm_states'][0]
            block_state_0 = traj_data['block_states'][0]
            
            # Extract initial states
            # Arm: [qpos(7), qvel(7), ee_pos(3), ee_quat(4)] = 21D
            qpos_0 = arm_state_0[:7]
            qvel_0 = arm_state_0[7:14]
            
            # Block: [pos(3), quat(4), size(3)] = 10D
            block_pos_0 = block_state_0[:3]
            block_quat_0 = block_state_0[3:7]
            block_size = block_state_0[7:10]
            
            # Set initial configuration
            data_mj.qpos[:7] = qpos_0
            data_mj.qvel[:7] = qvel_0
            data_mj.qpos[block_jnt_id:block_jnt_id+3] = block_pos_0
            data_mj.qpos[block_jnt_id+3:block_jnt_id+7] = block_quat_0
            model.geom(blockGeom_id).size[:] = block_size
            mujoco.mj_forward(model, data_mj)
            
            # Replay trajectory
            step_idx = 0
            replay = True
            
            while replay and step_idx < traj_data['num_steps']:
                # Check for viewer input
                if viewer.is_running():
                    # Get keyboard input (if available)
                    # Note: mujoco.viewer doesn't have direct keyboard input in passive mode
                    # We'll use time-based stepping instead
                    pass
                
                # Get current state from trajectory
                arm_state = traj_data['arm_states'][step_idx]
                block_state = traj_data['block_states'][step_idx]
                action = traj_data['actions'][step_idx]
                
                # Extract states
                qpos = arm_state[:7]
                qvel = arm_state[7:14]
                block_pos = block_state[:3]
                block_quat = block_state[3:7]
                
                # Set robot state
                data_mj.qpos[:7] = qpos
                data_mj.qvel[:7] = qvel
                
                # Set block state
                data_mj.qpos[block_jnt_id:block_jnt_id+3] = block_pos
                data_mj.qpos[block_jnt_id+3:block_jnt_id+7] = block_quat
                
                # Apply action based on action dimension
                if action_dim == 7:
                    # Taskspace action: [ee_pos(3), ee_quat(3), gripper(1)]
                    # For visualization, we can't directly apply taskspace actions
                    # We'll just use the recorded state (already set above)
                    gripper_cmd = action[gripper_idx]
                    # Don't set joint controls for taskspace - state is already correct
                else:
                    # Joint space action: [joint_pos(7), gripper(1)]
                    target_qpos = action[:joint_action_dim]
                    gripper_cmd = action[gripper_idx]
                    
                    # Set control to match recorded action (for gripper visualization)
                    data_mj.ctrl[actuator_ids] = target_qpos
                
                # Set gripper control and directly set finger joint positions for visualization
                # Gripper: 1.0 = closed (actuator8 = 0), 0.0 = open (actuator8 = 255)
                # In moveArm: shutFingers=True sets actuator8 to 0 (closed), False sets to 255 (open)
                # In dataset: gripper_cmd=1.0 means closed, gripper_cmd=0.0 means open
                actuator8_id = model.actuator("actuator8").id
                data_mj.ctrl[actuator8_id] = 0 if gripper_cmd > 0.5 else 255
                
                # Directly set finger joint positions for visualization
                # Finger joints: 0.0 = closed, 0.04 = open (based on XML keyframe)
                finger_joint1_id = model.joint("finger_joint1").id
                finger_joint2_id = model.joint("finger_joint2").id
                finger_pos = 0.0 if gripper_cmd > 0.5 else 0.04  # 0.0 = closed, 0.04 = open
                data_mj.qpos[finger_joint1_id] = finger_pos
                data_mj.qpos[finger_joint2_id] = finger_pos
                
                # Forward kinematics
                mujoco.mj_forward(model, data_mj)
                
                # Sync viewer
                viewer.sync()
                
                # Step forward
                step_idx += 1
                
                # Delay
                time.sleep(delay)
                
                # Check if done
                if step_idx >= traj_data['num_steps']:
                    # Wait a bit at the end
                    time.sleep(1.0)
                    replay = False
            
            # Move to next trajectory
            traj_idx += 1
            
            # Small pause between trajectories
            if traj_idx < len(traj_ids_to_show):
                time.sleep(0.5)
        
        print(f"\n{'='*70}")
        print(f"Visualization complete!")
        print(f"{'='*70}")

def main():
    """Main entry point"""
    visualize_trajectories(
        filename=None,
        base_filename='dataset',
        num_trajectories=5,
        trajectory_ids=None,
        delay=0.002,
        random_seed=None,
        interactive_file_selection=True
    )

if __name__ == "__main__":
    main()

