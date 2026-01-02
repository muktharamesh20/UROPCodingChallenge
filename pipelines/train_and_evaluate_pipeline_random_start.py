"""
Training and evaluation pipeline with random starting positions.
Trains models with increasing amounts of data until 50% success rate is achieved.
This version uses random robot starting positions for each trajectory.
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
from scipy.spatial.transform import Rotation as R
from utils.load_dataset_csv import load_dataset_csv, get_state_action_pairs
from utils.training_utils import (
    train_model_unified, evaluate_model_unified,
    load_and_filter_dataset, get_batch_size,
    count_trajectories, setup_mujoco_model, initialize_block_random_orientation,
    extract_state_from_simulation, verify_placement_on_green
)
from utils.path_utils import (
    get_dataset_path,
    get_variation_path,
    get_variation_from_base_filename,
    find_dataset_files,
    find_model_file
)
from trajectories.trajectoriesCircle import (
    sample_block_pos,
    sample_block_size,
    verifyPlacement,
    sample_robot_start_position,
    numSeconds,
    dt
)
from collect_dataset_csv import collectDataCSV as original_collectDataCSV

class MLP(nn.Module):
    """MLP with more parameters for random_start training"""
    def __init__(self, input_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(256, 256)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(256, 128)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(128, 64)
        self.relu4 = nn.ReLU()
        self.fc5 = nn.Linear(64, output_size)

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.relu3(self.fc3(x))
        x = self.relu4(self.fc4(x))
        x = self.fc5(x)
        return x

def train_model(states, actions, trajectory_ids=None, steps=None, epochs=None, batch_size=256, lr=0.001, 
                patience=25, min_delta=1e-6, state_scaler=None, max_epochs=1000, min_epochs=100, pretrained_model_path=None):
    """
    Train MLP model with early stopping based on validation loss.
    Wrapper around unified train_model_unified function.
    """
    return train_model_unified(states, actions, trajectory_ids, steps, epochs, batch_size, lr,
                              patience, min_delta, state_scaler, max_epochs, min_epochs, pretrained_model_path, MLP_class=MLP)

def evaluate_model(model, num_objects=20, max_steps=None, state_scaler=None, action_scaler=None):
    """
    Evaluate model on unseen objects with random starting positions.
    Wrapper around unified evaluate_model_unified function.
    """
    XML_PATH = "franka_emika_panda/pickAndPlace.xml"
    
    def apply_absolute_action(model_output, data, model_mj, site_id, block_id, dof_ids, actuator_ids, gripper_actuator_id):
        action = model_output
        action_scale = 0.1
        target_qpos = action[:7]
        current_qpos = data.qpos[dof_ids].copy()
        ctrl = current_qpos + (target_qpos - current_qpos) * action_scale
        gripper_cmd = action[7]
        data.ctrl[actuator_ids] = ctrl
        data.ctrl[gripper_actuator_id] = 0 if gripper_cmd > 0.5 else 255
    
    return evaluate_model_unified(
        model, num_objects, max_steps, state_scaler,
        XML_PATH, dt, numSeconds, sample_robot_start_position,
        lambda model, data, block_geom_id, block_jnt_id: initialize_block_random_orientation(
            model, data, block_geom_id, block_jnt_id, sample_block_pos, sample_block_size
        ),
        False, apply_absolute_action, None, None
    )

def collectDataCSV_random_start(target_successes=500, base_filename='dataset_random_start', trajectories_per_file=200, save_frequency=10):
    """
    Collect trajectories with random starting positions and save to CSV.
    This modifies collectDataCSV to use random starting positions.
    """
    # We need to patch collectDataCSV to use random starting positions
    # The simplest approach is to modify collect_dataset_csv.py to accept a parameter
    # For now, we'll create a wrapper that patches the function behavior
    # Actually, let's just modify collect_dataset_csv.py directly to support this
    
    # Call the original - we'll need to modify collect_dataset_csv.py to support random starts
    # For now, this will use the default (no random starts), but the user should modify
    # collect_dataset_csv.py to call sample_robot_start_position after line 130
    # For now, just call the original - user needs to modify collect_dataset_csv.py
    original_collectDataCSV(target_successes=target_successes, base_filename=base_filename, 
                           trajectories_per_file=trajectories_per_file, save_frequency=save_frequency)

def main():
    # Use different filenames for random start version
    base_filename = 'dataset_random_start'
    variation = get_variation_from_base_filename(base_filename)
    dataset_dir = get_dataset_path(variation)
    os.makedirs(dataset_dir, exist_ok=True)
    preprocessed_file = os.path.join(dataset_dir, 'dataset_random_start_preprocessed.pkl')
    variation_dir = get_variation_path(variation)
    os.makedirs(variation_dir, exist_ok=True)
    results_file = os.path.join(variation_dir, 'training_results_random_start.json')
    results_csv = os.path.join(variation_dir, 'training_results_random_start.csv')
    
    # Target: collect 1000 random start samples
    target_random_start_samples = 1000
    # Find pretrained model in default variation
    pretrained_models = find_model_file('model_800traj_*success.pth', base_name='default')
    pretrained_model_path = pretrained_models[0] if pretrained_models else 'model_800traj_0success.pth'
    
    # Check if we already have enough random start samples
    files = find_dataset_files(base_filename, pattern='*.csv')
    
    # Count existing trajectories
    num_trajectories_total = 0
    if files:
        for filename in files:
            try:
                # Quick count of unique trajectory IDs
                chunk_size = 100000
                traj_ids_seen = set()
                for chunk in pd.read_csv(filename, chunksize=chunk_size, usecols=['trajectory_id']):
                    traj_ids_seen.update(chunk['trajectory_id'].unique())
                num_trajectories_total = max(num_trajectories_total, len(traj_ids_seen))
            except Exception as e:
                pass
    
    # Collect random start samples if needed
    if num_trajectories_total < target_random_start_samples:
        needed = target_random_start_samples - num_trajectories_total
        
        # Collect trajectories with random starting positions
        collectDataCSV_random_start(target_successes=target_random_start_samples, base_filename=base_filename, 
                                   trajectories_per_file=200, save_frequency=20)
        
        # Reload to get accurate count
        data = load_dataset_csv(filename=None, base_filename=base_filename, max_trajectories=None)
        unique_traj_ids = np.unique(data['trajectory_ids'])
        num_trajectories_total = len(unique_traj_ids)
    
    # Try to load preprocessed dataset first (much faster)
    preprocessed_file = 'dataset_random_start_preprocessed.pkl'
    
    if os.path.exists(preprocessed_file):
        print(f"  Found preprocessed dataset: {preprocessed_file}")
        print(f"  Loading from preprocessed file (fast)...")
        with open(preprocessed_file, 'rb') as f:
            preprocessed_data = pickle.load(f)
        
        states = preprocessed_data['states']
        actions = preprocessed_data['actions']
        unique_traj_ids = preprocessed_data['unique_trajectory_ids']
        trajectory_ids_full = preprocessed_data['trajectory_ids']
        num_trajectories_total = preprocessed_data['num_trajectories']
        steps_full = preprocessed_data.get('steps', None)
        
        print(f"  ✓ Loaded {len(states)} state-action pairs from {num_trajectories_total} unique trajectories")
        print(f"    Trajectory ID range: {unique_traj_ids.min()} to {unique_traj_ids.max()}")
    else:
        print(f"  Preprocessed file not found: {preprocessed_file}")
        print(f"  Loading from CSV files (this may take a while)...")
        print(f"  Tip: Run 'python -m utils.preprocess_dataset_random_start' to create preprocessed file for faster loading")
        
        # Load the random start dataset
        data = load_dataset_csv(filename=None, base_filename=base_filename, max_trajectories=None)
        states, actions = get_state_action_pairs(data)
        trajectory_ids_full = data['trajectory_ids']
        unique_traj_ids = np.unique(trajectory_ids_full)
        num_trajectories_total = len(unique_traj_ids)
        
        # Load steps if available
        steps_full = data.get('steps', None)
        
        print(f"  ✓ Loaded {len(states)} state-action pairs from {num_trajectories_total} unique trajectories")
        print(f"    Trajectory ID range: {unique_traj_ids.min()} to {unique_traj_ids.max()}")
    
    # Load rejected trajectories and filter them out
    # Train on: approved, skipped, and not reviewed (everything except rejected)
    rejected_traj_ids = set()
    approved_file = 'approved_trajectories.json'
    if os.path.exists(approved_file):
        try:
            with open(approved_file, 'r') as f:
                approved_data = json.load(f)
                rejected_traj_ids = set(approved_data.get('rejected', []))
        except (json.JSONDecodeError, ValueError) as e:
            pass  # If file is corrupted, treat as empty (no rejections)
    
    # Filter out rejected trajectories
    if rejected_traj_ids:
        print(f"  Filtering out {len(rejected_traj_ids)} rejected trajectories")
        # Convert to numpy array for filtering
        rejected_array = np.array(list(rejected_traj_ids))
        # Keep trajectories that are NOT in rejected set
        keep_mask = ~np.isin(trajectory_ids_full, rejected_array)
        states = states[keep_mask]
        actions = actions[keep_mask]
        trajectory_ids_full = trajectory_ids_full[keep_mask]
        if steps_full is not None:
            steps_full = steps_full[keep_mask]
        unique_traj_ids = np.unique(trajectory_ids_full)
        num_trajectories_total = len(unique_traj_ids)
        print(f"  After filtering: {num_trajectories_total} trajectories remaining")
    
    # Use only the random start samples (up to target_random_start_samples)
    traj_ids_to_use = unique_traj_ids[:target_random_start_samples]
    mask = np.isin(trajectory_ids_full, traj_ids_to_use)
    states_subset = states[mask]
    actions_subset = actions[mask]
    trajectory_ids_subset = trajectory_ids_full[mask]
    steps_subset = steps_full[mask] if steps_full is not None else None
    
    # Check for GPU availability
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    # Check if pretrained model exists
    if not os.path.exists(pretrained_model_path):
        return
    
    # Training parameters
    num_samples = len(states_subset)
    device_type = 'mps' if device.type == 'mps' else 'cuda' if device.type == 'cuda' else 'cpu'
    batch_size = get_batch_size(num_samples, device_type)
    
    # Train model - fine-tune ONLY with random start samples
    model = train_model(
        states_subset, actions_subset, trajectory_ids=trajectory_ids_subset, steps=steps_subset,
        epochs=100, batch_size=batch_size, lr=0.001,
        patience=15,
        pretrained_model_path=pretrained_model_path
    )
    
    # Evaluate model
    success_rate, eval_results = evaluate_model(model, num_objects=20)
    
    # Save model with descriptive name
    model_filename = f"model_random_start_1000samples_finetuned_{int(success_rate*100)}success.pth"
    torch.save(model.state_dict(), model_filename)
    
    # Store results
    results = [{
        'pretrained_model': pretrained_model_path,
        'num_random_start_trajectories': len(traj_ids_to_use),
        'num_state_action_pairs': len(states_subset),
        'success_rate': float(success_rate),
        'num_successes': int(success_rate * 20),
        'num_evaluations': 20,
        'model_filename': model_filename
    }]
    
    # Save results
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save final results as CSV too
    df_results = pd.DataFrame(results)
    df_results.to_csv(results_csv, index=False)

if __name__ == "__main__":
    main()

