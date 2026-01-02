"""
Preprocess the random_start dataset CSV files into a pickle file for faster loading.
This significantly speeds up training by avoiding repeated CSV parsing.

Used AI to generate the code.
"""
import os
import pickle
import numpy as np
from utils.load_dataset_csv import load_dataset_csv, get_state_action_pairs
from utils.path_utils import get_dataset_path, get_variation_from_base_filename

def preprocess_random_start_dataset(base_filename='dataset_random_start', output_file=None):
    """
    Load all random_start dataset CSV files, process them, and save to a pickle file.
    
    Args:
        base_filename: Base filename for dataset CSV files (e.g., 'dataset_random_start')
        output_file: Output pickle file path. If None, saves to variation's datasets folder.
    """
    
    # Determine output file location
    if output_file is None:
        variation = get_variation_from_base_filename(base_filename)
        dataset_dir = get_dataset_path(variation)
        os.makedirs(dataset_dir, exist_ok=True)
        output_file = os.path.join(dataset_dir, f"{base_filename}_preprocessed.pkl")
    
    print(f"Preprocessing {base_filename} dataset...")
    print(f"  Loading from CSV files (this may take a while)...")
    
    # Load all data from CSV files
    data = load_dataset_csv(filename=None, base_filename=base_filename, max_trajectories=None)
    
    # Get states and actions
    states, actions = get_state_action_pairs(data)
    
    # Get unique trajectory IDs
    unique_traj_ids = np.unique(data['trajectory_ids'])
    num_trajectories_total = len(unique_traj_ids)
    
    # Create mapping from trajectory_id to indices
    trajectory_id_to_indices = {}
    trajectory_ids_full = data['trajectory_ids']
    for traj_id in unique_traj_ids:
        trajectory_id_to_indices[traj_id] = np.where(trajectory_ids_full == traj_id)[0]
    
    # Load steps if available
    steps = data.get('steps', None)
    
    # Prepare preprocessed data
    preprocessed_data = {
        'states': states,
        'actions': actions,
        'trajectory_ids': trajectory_ids_full,
        'unique_trajectory_ids': unique_traj_ids,
        'trajectory_id_to_indices': trajectory_id_to_indices,
        'num_trajectories': num_trajectories_total,
        'steps': steps  # Include steps if available
    }
    
    print(f"  ✓ Processed {len(states)} state-action pairs from {num_trajectories_total} unique trajectories")
    print(f"    Trajectory ID range: {unique_traj_ids.min()} to {unique_traj_ids.max()}")
    if steps is not None:
        print(f"    Steps included: {len(steps)} samples")
    
    # Save to pickle file
    print(f"\n  Saving to {output_file}...")
    with open(output_file, 'wb') as f:
        pickle.dump(preprocessed_data, f)
    
    file_size_mb = os.path.getsize(output_file) / (1024 * 1024)
    print(f"  ✓ Saved preprocessed dataset ({file_size_mb:.2f} MB)")
    print(f"  You can now use this file for faster loading in train_and_evaluate_pipeline_random_start.py")

if __name__ == "__main__":
    preprocess_random_start_dataset()

