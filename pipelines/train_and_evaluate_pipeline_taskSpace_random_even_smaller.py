"""
Training and evaluation pipeline for pick-and-place using task-space absolute actions with random starting positions.
Uses pickAndPlaceSimpler.xml and trajectoriesTaskspaceRandomSmaller.py.
Incrementally collects data and trains models with fine-tuning.
Success is defined as block placed on the green placement box.
Uses task-space absolute actions (absolute end-effector poses) instead of deltas.
Actions: [ee_pos(3), ee_quat(3), gripper(1)] = 7D
Random robot starting positions for each trajectory.
Uses even smaller model architecture (256→128→64) to prevent overfitting.
"""
import sys
import os
# Add parent directory to path so we can import utils when running directly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import mujoco
from utils.load_dataset_csv import load_dataset_csv, get_state_action_pairs
from utils.training_utils import (
    extract_arm_state, extract_block_state, extract_action, extract_action_delta,
    extract_action_taskspace_absolute, compute_desired_pose_from_absolute,
    convert_delta_action_to_absolute_action,
    get_joint_ranges, normalize_action, denormalize_action,
    save_action_scaler, load_action_scaler,
    save_state_scaler, load_state_scaler,
    count_trajectories, setup_mujoco_model, initialize_block_random_orientation,
    extract_state_from_simulation, get_batch_size,
    load_and_filter_dataset, load_or_create_model, verify_placement_on_green,
    train_and_evaluate_step as train_and_evaluate_step_base,
    save_trajectory_rows_to_csv,
    train_model_unified, evaluate_model_unified,
    collect_trajectory_data_unified
)
from utils.path_utils import (
    get_dataset_path,
    get_variation_path,
    get_variation_from_base_filename,
    find_dataset_files,
    find_model_file
)
from utils.ik_utils import moveArm, computeIKError
from trajectories.trajectoriesTaskspaceRandomSmaller import (
    createTrajectory,
    sample_block_pos,
    sample_block_size,
    sample_robot_start_position,
    dt,
    numSeconds,
    placement_box_pos,
    placement_box_size
)

def train_and_evaluate_step(target_trajectories, states, actions, traj_ids, steps,
                           batch_size, pretrained_model_path=None, joint_ranges=None, finetune_lr_scale=0.5):
    """Single training/evaluation step - returns model, scaler, action_scaler, success_rate, model_file."""
    # Create a wrapper that passes finetune_lr_scale to train_model
    def train_model_wrapper(*args, **kwargs):
        kwargs['finetune_lr_scale'] = finetune_lr_scale
        return train_model(*args, **kwargs)
    
    return train_and_evaluate_step_base(
        target_trajectories, states, actions, traj_ids, steps,
        batch_size, model_name_prefix='taskspace_random_even_smaller', MLP_class=MLP,
        train_model_fn=train_model_wrapper, evaluate_model_fn=evaluate_model,
        pretrained_model_path=pretrained_model_path, joint_ranges=joint_ranges,
        use_action_normalization=True  # Enable for absolute actions (values can vary widely)
    )

# ============================================================================
# Constants
# ============================================================================

XML_PATH = "franka_emika_panda/pickAndPlaceSimpler.xml"

# Evaluation and collection constants
# Constants moved to training_utils.py (hardcoded in collect_trajectory_data_unified)

# State encoding flag for task-space pipeline (uses sin/cos encoding)
USE_SIN_COS_ENCODING = True

# Damping for IK (same as trajectoriesTaskSpace)
DAMPING = 1e-10

# ============================================================================
# Model Definition
# ============================================================================

class MLP(nn.Module):
    """MLP for task-space pick-and-place task with random starts - even smaller architecture to prevent overfitting"""
    def __init__(self, input_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(256, 128)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(128, 64)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(64, output_size)

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.relu3(self.fc3(x))
        x = self.fc4(x)
        # No clamping - task-space absolutes can be any valid pose
        return x

def train_model(states, actions, trajectory_ids=None, steps=None, epochs=None, batch_size=256, lr=0.001, 
                patience=25, min_delta=1e-6, state_scaler=None, max_epochs=10000, min_epochs=200, pretrained_model_path=None,
                finetune_lr_scale=0.5):
    """
    Train MLP model with early stopping based on validation loss.
    Wrapper around unified train_model_unified function.
    """
    return train_model_unified(states, actions, trajectory_ids, steps, epochs, batch_size, lr,
                              patience, min_delta, state_scaler, max_epochs, min_epochs, pretrained_model_path, 
                              MLP_class=MLP, finetune_lr_scale=finetune_lr_scale)

# verifyPlacementOnGreen is now imported from training_utils as verify_placement_on_green

def evaluate_model(model, num_objects=20, max_steps=None, state_scaler=None, action_scaler=None):
    """
    Evaluate model on unseen objects with random starting positions.
    Success is defined as block placed on the green placement box.
    Wrapper around unified evaluate_model_unified function.
    """
    # Define action application callback for task-space absolute actions
    def apply_taskspace_absolute_action(model_output, data, model_mj, site_id, block_id, dof_ids, actuator_ids, gripper_actuator_id):
        absolute_action = model_output
        
        # Extract absolute pose components
        absolute_ee_pos = absolute_action[:3]
        absolute_ee_quat_3d = absolute_action[3:6]
        
        # Convert 3D quaternion representation to full 4D quaternion
        desired_pos, desired_quat = compute_desired_pose_from_absolute(
            absolute_ee_pos, absolute_ee_quat_3d
        )
        
        # Determine gripper state
        gripper_cmd = absolute_action[6]
        shut_fingers = gripper_cmd > 0.5
        
        # Use moveArm to convert desired pose to joint commands (same as expert controller)
        # Pass None for trajectory/phase info since we don't have it during evaluation
        # moveArm will handle IK, joint limits, nullspace, and gripper control.
        # NOTE: When shut_fingers=True, moveArm will override desiredQuat with z-axis-only
        # constraint (z-axis pointing down). This matches expert behavior during placing/lifting.
        moveArm(
            model_mj, data, site_id, block_id, dof_ids, actuator_ids,
            desired_pos, desired_quat, shut_fingers,
            trajectory=None, current_idx=None, 
            placingPhaseStart=None, liftingPhaseStart=None, liftingPhaseEnd=None,
            damping=DAMPING, max_angvel=1.5
        )
    
    return evaluate_model_unified(
        model, num_objects, max_steps, state_scaler,
        XML_PATH, dt, numSeconds, sample_robot_start_position,
        lambda model, data, block_geom_id, block_jnt_id: initialize_block_random_orientation(
            model, data, block_geom_id, block_jnt_id, sample_block_pos, sample_block_size
        ),
        USE_SIN_COS_ENCODING, apply_taskspace_absolute_action, placement_box_pos, placement_box_size
    )

def collectDataCSV_taskspace_random_smaller(target_successes=500, base_filename='dataset_taskspace_random_smaller', trajectories_per_file=200, save_frequency=20):
    """
    Collect trajectories using trajectoriesTaskspaceRandomSmaller.py and save to CSV.
    Actions are task-space absolutes: [ee_pos(3), ee_quat(3), gripper(1)] = 7D
    Uses random robot starting positions for each trajectory.
    """
    # Define action extraction callback for absolute actions
    def extract_action_taskspace_absolute_callback(data, model, site_id, block_id, dof_ids, actuator_ids, gripper_actuator_id,
                                                   desiredPos, desiredQuat, trajectory, i,
                                                   current_qpos_before=None, current_ee_pos_before=None, current_ee_quat_before=None):
        # Extract absolute desired pose directly (no delta computation)
        return extract_action_taskspace_absolute(
            desiredPos, desiredQuat,
            data.ctrl, gripper_actuator_id
        )
    
    # Use unified collection function
    collect_trajectory_data_unified(
        target_successes=target_successes,
        base_filename=base_filename,
        trajectories_per_file=trajectories_per_file,
        save_frequency=save_frequency,
        xml_path=XML_PATH,
        dt=dt,
        num_seconds=numSeconds,
        get_robot_start_fn=sample_robot_start_position,
        initialize_block_fn=lambda model, data, block_geom_id, block_jnt_id: initialize_block_random_orientation(
            model, data, block_geom_id, block_jnt_id, sample_block_pos, sample_block_size
        ),
        create_trajectory_fn=createTrajectory,
        extract_action_fn=extract_action_taskspace_absolute_callback,
        verify_placement_fn=lambda model, data, block_id, placement_box_pos, placement_box_size, table_surface_z, tolerance: verify_placement_on_green(
            model, data, block_id, placement_box_pos, placement_box_size, table_surface_z, tolerance
        ),
        move_arm_kwargs={'damping': 1e-10, 'max_angvel': 1.5},
        placement_box_pos=placement_box_pos,
        placement_box_size=placement_box_size,
        use_sin_cos_encoding=USE_SIN_COS_ENCODING
    )

def main():
    # Use taskspace_random_even_smaller variation
    base_filename = 'dataset_taskspace_random_smaller'
    variation = get_variation_from_base_filename(base_filename)
    dataset_dir = get_dataset_path(variation)
    os.makedirs(dataset_dir, exist_ok=True)
    preprocessed_file = os.path.join(dataset_dir, 'dataset_taskspace_random_smaller_preprocessed.pkl')
    variation_dir = get_variation_path(variation)
    os.makedirs(variation_dir, exist_ok=True)
    results_file = os.path.join(variation_dir, 'training_results_taskspace_random_even_smaller.json')
    results_csv = os.path.join(variation_dir, 'training_results_taskspace_random_even_smaller.csv')
    
    # Training configurations - double data size each time
    data_sizes = [500, 1000, 2000, 4000]  # Starting at 4000 (500, 1000, 2000 already completed)
    use_finetuning = True
    finetune_lr_scale = 0.5
    explicit_pretrained_model = None
    
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
        
        # Check current dataset size
        num_trajectories_total, _ = count_trajectories(base_filename)
        
        # Collect more data if needed (incremental collection)
        if num_trajectories_total < target_trajectories:
            needed = target_trajectories - num_trajectories_total
            print(f"Need {needed} more successful trajectories (have {num_trajectories_total}, need {target_trajectories})")
            print(f"Collecting additional trajectories...")
            
            collectDataCSV_taskspace_random_smaller(
                target_successes=target_trajectories,
                base_filename=base_filename,
                trajectories_per_file=200,
                save_frequency=20
            )
            
            # Reload to get accurate count
            num_trajectories_total, _ = count_trajectories(base_filename)
        
        # Load and filter dataset
        print(f"\nLoading and filtering dataset...")
        states_subset, actions_subset, trajectory_ids_subset, steps_subset = load_and_filter_dataset(
            base_filename, target_trajectories, filter_top_pct=0.05,
            recovery_filename=None, use_recovery=False,
            use_sin_cos_encoding=USE_SIN_COS_ENCODING)
        
        # Print filtering info
        unique_traj_ids_subset = np.unique(trajectory_ids_subset)
        print(f"  ✓ Using {len(states_subset)} state-action pairs from {len(unique_traj_ids_subset)} unique trajectories")
        
        # Determine pretrained model path for fine-tuning
        pretrained_model_path = None
        num_prev_trajectories = 0
        
        if use_finetuning:
            # Priority: explicit model > previous best model
            if explicit_pretrained_model and os.path.exists(explicit_pretrained_model):
                pretrained_model_path = explicit_pretrained_model
                print(f"  Using explicit pretrained model: {os.path.basename(pretrained_model_path)}")
            else:
                # Find the largest completed size that's smaller than current
                prev_sizes = [r['num_trajectories'] for r in results if r['num_trajectories'] < target_trajectories]
                if prev_sizes:
                    prev_size = max(prev_sizes)  # Use the largest previous size
                    # Look for previous model
                    prev_models = find_model_file(f"model_taskspace_random_even_smaller_{prev_size}traj_*.pth", base_name=variation)
                    if prev_models:
                        pretrained_model_path = prev_models[0]  # Use first match
                        num_prev_trajectories = prev_size
                        print(f"  Found previous model for fine-tuning: {os.path.basename(pretrained_model_path)} ({num_prev_trajectories} trajectories)")
                    elif best_model_path and os.path.exists(best_model_path):
                        pretrained_model_path = best_model_path
                        # Count how many trajectories were used in the previous model
                        for result in results:
                            if result.get('model_filename') == os.path.basename(pretrained_model_path):
                                num_prev_trajectories = result.get('num_trajectories', 0)
                                break
                        print(f"  Using previous best model for fine-tuning: {os.path.basename(pretrained_model_path)} ({num_prev_trajectories} trajectories)")
                    else:
                        print(f"  Training from scratch (no suitable pretrained model found)")
                elif best_model_path and os.path.exists(best_model_path):
                    pretrained_model_path = best_model_path
                    # Count how many trajectories were used in the previous model
                    for result in results:
                        if result.get('model_filename') == os.path.basename(pretrained_model_path):
                            num_prev_trajectories = result.get('num_trajectories', 0)
                            break
                    print(f"  Using previous best model for fine-tuning: {os.path.basename(pretrained_model_path)} ({num_prev_trajectories} trajectories)")
                else:
                    print(f"  Training from scratch (no pretrained model available)")
        else:
            print(f"  Training from scratch (fine-tuning disabled)")
        
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
            pretrained_model_path=pretrained_model_path, joint_ranges=None,
            finetune_lr_scale=finetune_lr_scale)
        
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
            'model_filename': os.path.basename(model_filename), # Store only basename
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

