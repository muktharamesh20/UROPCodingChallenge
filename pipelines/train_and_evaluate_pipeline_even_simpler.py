"""
Training and evaluation pipeline for the even simpler pick-and-place task.
Uses pickAndPlaceSimpler.xml and trajectoriesEvenSimpler.py.
Incrementally collects data and trains models from scratch (no fine-tuning).
Success is defined as block placed on the green placement box.
"""
import sys
import os
# Add parent directory to path so we can import utils when running directly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import json
import os
from utils.load_dataset_csv import load_dataset_csv, get_state_action_pairs
import mujoco
from trajectories.trajectoriesEvenSimpler import (
    sample_block_pos, sample_block_size, get_robot_start_position,
    dt, numSeconds, placement_box_pos, placement_box_size
)
import pickle
from utils.training_utils import (
    extract_arm_state, extract_block_state, extract_action, extract_action_delta,
    get_joint_ranges, normalize_action, denormalize_action,
    save_action_scaler, load_action_scaler,
    save_state_scaler, load_state_scaler,
    count_trajectories, setup_mujoco_model, initialize_block_random_orientation,
    extract_state_from_simulation, get_batch_size,
    load_and_filter_dataset, load_or_create_model, verify_placement_on_green,
    train_and_evaluate_step as train_and_evaluate_step_base,
    save_trajectory_rows_to_csv,
    collect_trajectory_data_unified,
    train_model_unified, evaluate_model_unified,
    get_joint_range_for_joint
)
from utils.path_utils import (
    get_dataset_path,
    get_variation_path,
    get_variation_from_base_filename,
    find_dataset_files
)
from trajectories.trajectoriesEvenSimpler import (
    createTrajectory,
    sample_block_pos,
    sample_block_size,
    get_robot_start_position
)

def train_and_evaluate_step(target_trajectories, states, actions, traj_ids, steps,
                           batch_size, pretrained_model_path=None, joint_ranges=None):
    """Single training/evaluation step - returns model, scaler, action_scaler, success_rate, model_file."""
    return train_and_evaluate_step_base(
        target_trajectories, states, actions, traj_ids, steps,
        batch_size, model_name_prefix='even_simpler', MLP_class=MLP,
        train_model_fn=train_model, evaluate_model_fn=evaluate_model,
        pretrained_model_path=pretrained_model_path, joint_ranges=joint_ranges,
        use_action_normalization=False
    )

# ============================================================================
# Constants
# ============================================================================

XML_PATH = "franka_emika_panda/pickAndPlaceSimpler.xml"

# Evaluation and collection constants
# Constants moved to training_utils.py (hardcoded in collect_trajectory_data_unified)

# State encoding flag for even_simpler (uses sin/cos encoding)
USE_SIN_COS_ENCODING = True

# load_and_filter_dataset and get_batch_size are now imported from training_utils

# train_and_evaluate_step is now imported from training_utils

# ============================================================================
# Model Definition
# ============================================================================

class MLP(nn.Module):
    """MLP for even simpler pick-and-place task - larger architecture"""
    def __init__(self, input_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, 1024)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(1024, 1024)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(1024, 512)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(512, 256)
        self.relu4 = nn.ReLU()
        self.fc5 = nn.Linear(256, output_size)

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.relu3(self.fc3(x))
        x = self.relu4(self.fc4(x))
        x = self.fc5(x)
        return x

def train_model(states, actions, trajectory_ids=None, steps=None, epochs=None, batch_size=256, lr=0.001, 
                patience=25, min_delta=1e-6, state_scaler=None, max_epochs=10000, min_epochs=100, pretrained_model_path=None):
    """
    Train MLP model with early stopping based on validation loss.
    Wrapper around unified train_model_unified function.
    """
    return train_model_unified(states, actions, trajectory_ids, steps, epochs, batch_size, lr,
                              patience, min_delta, state_scaler, max_epochs, min_epochs, pretrained_model_path, MLP_class=MLP)

# verifyPlacementOnGreen is now imported from training_utils as verify_placement_on_green

def evaluate_model(model, num_objects=20, max_steps=None, state_scaler=None, action_scaler=None):
    """
    Evaluate model on unseen objects with random starting positions.
    Success is defined as block placed on the green placement box.
    Wrapper around unified evaluate_model_unified function.
    """
    # Define action application callback for delta actions (even_simpler uses delta, same as delta pipeline)
    def apply_delta_action(model_output, data, model_mj, site_id, block_id, dof_ids, actuator_ids, gripper_actuator_id):
        delta_action = model_output
        current_qpos = data.qpos[:7].copy()
        
        # Scale delta for stability (10% of predicted delta per step)
        delta_scale = 0.1
        target_qpos = current_qpos + delta_action[:7] * delta_scale
        
        # Enforce joint limits
        for i, jid in enumerate(dof_ids):
            minr, maxr = get_joint_range_for_joint(i, model_mj=model_mj, dof_id=jid, use_default=True)
            if minr < maxr and np.isfinite(minr) and np.isfinite(maxr):
                target_qpos[i] = np.clip(target_qpos[i], minr, maxr)
        
        # Apply controls
        data.ctrl[actuator_ids] = target_qpos
        
        # Gripper: delta_action[7] is binary (0 or 1)
        gripper_cmd = delta_action[7]
        data.ctrl[gripper_actuator_id] = 0 if gripper_cmd > 0.5 else 255
    
    return evaluate_model_unified(
        model, num_objects, max_steps, state_scaler,
        XML_PATH, dt, numSeconds, get_robot_start_position,
        lambda model, data, block_geom_id, block_jnt_id: initialize_block_random_orientation(
            model, data, block_geom_id, block_jnt_id, sample_block_pos, sample_block_size
        ),
        USE_SIN_COS_ENCODING, apply_delta_action, placement_box_pos, placement_box_size
    )

def collectDataCSV_simpler(target_successes=500, base_filename='dataset_even_simpler', trajectories_per_file=200, save_frequency=20):
    """
    Collect trajectories using trajectoriesEvenSimpler.py and save to CSV.
    Uses unified trajectory collection function.
    """
    # Define action extraction callback for absolute actions (even_simpler uses absolute, not delta)
    def extract_action_absolute_callback(data, model, site_id, block_id, dof_ids, actuator_ids, gripper_actuator_id,
                                         desiredPos, desiredQuat, trajectory, i,
                                         current_qpos_before=None, current_ee_pos_before=None, current_ee_quat_before=None):
        return extract_action(data.ctrl, gripper_actuator_id)
    
    # Use unified collection function
    collect_trajectory_data_unified(
        target_successes=target_successes,
        base_filename=base_filename,
        trajectories_per_file=trajectories_per_file,
        save_frequency=save_frequency,
        xml_path=XML_PATH,
        dt=dt,
        num_seconds=numSeconds,
        get_robot_start_fn=get_robot_start_position,
        initialize_block_fn=lambda model, data, block_geom_id, block_jnt_id: initialize_block_random_orientation(
            model, data, block_geom_id, block_jnt_id, sample_block_pos, sample_block_size
        ),
        create_trajectory_fn=createTrajectory,
        extract_action_fn=extract_action_absolute_callback,
        verify_placement_fn=lambda model, data, block_id, placement_box_pos, placement_box_size, table_surface_z, tolerance: verify_placement_on_green(
            model, data, block_id, placement_box_pos, placement_box_size, table_surface_z, tolerance
        ),
        placement_box_pos=placement_box_pos,
        placement_box_size=placement_box_size,
        use_sin_cos_encoding=False
    )

def main():
    # Use different filenames for even simpler version
    base_filename = 'dataset_even_simpler'
    variation = get_variation_from_base_filename(base_filename)
    dataset_dir = get_dataset_path(variation)
    os.makedirs(dataset_dir, exist_ok=True)
    preprocessed_file = os.path.join(dataset_dir, 'dataset_even_simpler_preprocessed.pkl')
    variation_dir = get_variation_path(variation)
    os.makedirs(variation_dir, exist_ok=True)
    results_file = os.path.join(variation_dir, 'training_results_simpler.json')
    results_csv = os.path.join(variation_dir, 'training_results_simpler.csv')
    
    # Training configurations: start at 500, then increase by 200 up to 1500
    data_sizes = [500, 700, 900, 1100, 1300, 1500]
    
    # Check for GPU availability
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
        print(f"✓ Apple Silicon GPU (MPS) available")
    elif torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"✓ GPU available: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("⚠ GPU not available, using CPU")
    print(f"Using device: {device}")
    
    # Load existing results if available
    results = []
    if os.path.exists(results_file):
        print("\nLoading existing results...")
        with open(results_file, 'r') as f:
            results = json.load(f)
        print(f"  Found {len(results)} previous results")
    
    # Track the best model so far
    best_model_path = None
    best_success_rate = 0.0
    
    # Process each data size incrementally
    for target_trajectories in data_sizes:
        print(f"\n{'='*70}")
        print(f"Training with {target_trajectories} trajectories")
        print(f"{'='*70}\n")
        
        # Check current dataset size (look in datasets folder)
        files = find_dataset_files(base_filename, pattern='*.csv')
        
        num_trajectories_total = 0
        if files:
            for filename in files:
                try:
                    if os.path.getsize(filename) == 0:
                        continue
                    chunk_size = 100000
                    traj_ids_seen = set()
                    for chunk in pd.read_csv(filename, chunksize=chunk_size, usecols=['trajectory_id']):
                        traj_ids_seen.update(chunk['trajectory_id'].unique())
                    num_trajectories_total = max(num_trajectories_total, len(traj_ids_seen))
                except (pd.errors.EmptyDataError, ValueError):
                    continue
        
        # Collect more data if needed (incremental collection)
        if num_trajectories_total < target_trajectories:
            needed = target_trajectories - num_trajectories_total
            print(f"Need {needed} more successful trajectories (have {num_trajectories_total}, need {target_trajectories})")
            print(f"Collecting additional trajectories...")
            
            collectDataCSV_simpler(
                target_successes=target_trajectories,
                base_filename=base_filename,
                trajectories_per_file=200,
                save_frequency=20
            )
            
            # Reload to get accurate count
            num_trajectories_total, _ = count_trajectories(base_filename)
            print(f"Now have {num_trajectories_total} trajectories")
        
        # Load and filter dataset (50% normal + 50% recovery data)
        print(f"\nLoading and filtering dataset...")
        recovery_filename = 'dataset_even_simpler_recovery'
        states_subset, actions_subset, trajectory_ids_subset, steps_subset = load_and_filter_dataset(
            base_filename, target_trajectories, filter_top_pct=0.2,
            recovery_filename=recovery_filename, use_recovery=True,
            use_sin_cos_encoding=USE_SIN_COS_ENCODING)
        
        # Print filtering info
        unique_traj_ids_subset = np.unique(trajectory_ids_subset)
        print(f"  ✓ Loaded {len(states_subset)} state-action pairs from {len(unique_traj_ids_subset)} unique trajectories")
        
        # Determine pretrained model path (use previous best model for fine-tuning)
        pretrained_model_path = best_model_path
        num_prev_trajectories = 0
        if pretrained_model_path and os.path.exists(pretrained_model_path):
            # Count how many trajectories were used in the previous model
            for result in results:
                if result.get('model_filename') == os.path.basename(pretrained_model_path):
                    num_prev_trajectories = result.get('num_trajectories', 0)
                    break
            print(f"  Using pretrained model: {os.path.basename(pretrained_model_path)} ({num_prev_trajectories} trajectories)")
        else:
            print(f"  Training from scratch (no pretrained model)")
        
        # Get batch size and train/evaluate (auto-detect device for batch size)
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device_type = 'mps'
        elif torch.cuda.is_available():
            device_type = 'cuda'
        else:
            device_type = 'cpu'
        batch_size = get_batch_size(len(states_subset), device_type)
        model, state_scaler, action_scaler, success_rate, model_filename = train_and_evaluate_step(
            target_trajectories, states_subset, actions_subset, 
            trajectory_ids_subset, steps_subset, batch_size,
            pretrained_model_path=pretrained_model_path, joint_ranges=None)
        
        # Update best model
        if success_rate > best_success_rate:
            best_success_rate = success_rate
            best_model_path = model_filename
        
        # Store results
        result_entry = {
            'num_trajectories': target_trajectories,
            'num_state_action_pairs': len(states_subset),
            'success_rate': float(success_rate),
            'num_successes': int(success_rate * 20),
            'num_evaluations': 20,
            'model_filename': model_filename,
            'pretrained_model': os.path.basename(pretrained_model_path) if pretrained_model_path else None
        }
        results.append(result_entry)
        
        # Save results
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save as CSV too
        df_results = pd.DataFrame(results)
        df_results.to_csv(results_csv, index=False)
        
        print(f"\n  Success rate: {success_rate:.2%} ({int(success_rate * 20)}/20)")
        print(f"  Best model so far: {best_model_path} ({best_success_rate:.2%})")
        
        # Stop if we reach 50% success rate
        if success_rate >= 0.5:
            print(f"\n{'='*70}")
            print(f"✓ Reached target success rate of 50%!")
            print(f"  Final model: {model_filename}")
            print(f"  Success rate: {success_rate:.2%}")
            print(f"{'='*70}")
            break

if __name__ == "__main__":
    main()

