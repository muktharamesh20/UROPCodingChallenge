"""
Preprocess dataset from CSV files and save to a binary format for faster loading.

This script loads all trajectories from CSV files, processes them into state-action pairs,
and saves them to a pickle file. This avoids reprocessing the dataset every time.

Used AI to generate the code.
"""

import os
import sys
import numpy as np
import pickle
from utils.load_dataset_csv import load_dataset_csv, get_state_action_pairs
from utils.path_utils import get_dataset_path, get_variation_from_base_filename

def preprocess_and_save_dataset(output_file=None, base_filename='dataset'):
    """
    Load dataset from CSV files, process into state-action pairs, and save to pickle.
    
    Args:
        output_file: Path to save the preprocessed dataset. If None, saves to variation's datasets folder.
        base_filename: Base filename for CSV files (e.g., 'dataset')
    """
    
    # Determine output file location
    if output_file is None:
        variation = get_variation_from_base_filename(base_filename)
        dataset_dir = get_dataset_path(variation)
        os.makedirs(dataset_dir, exist_ok=True)
        output_file = os.path.join(dataset_dir, f"{base_filename}_preprocessed.pkl")
    
    print("=" * 70)
    print("Preprocessing Dataset")
    print("=" * 70)
    print(f"\nLoading dataset from CSV files...")
    print(f"  Output file: {output_file}")
    
    # Load full dataset (no limit)
    data = load_dataset_csv(filename=None, base_filename=base_filename, max_trajectories=None)
    states, actions = get_state_action_pairs(data)
    
    # Get unique trajectory IDs
    unique_traj_ids = np.unique(data['trajectory_ids'])
    num_trajectories_total = len(unique_traj_ids)
    
    print(f"\nProcessed dataset:")
    print(f"  Total state-action pairs: {len(states)}")
    print(f"  Total unique trajectories: {num_trajectories_total}")
    print(f"  Trajectory ID range: {unique_traj_ids.min()} to {unique_traj_ids.max()}")
    print(f"  State shape: {states.shape}")
    print(f"  Action shape: {actions.shape}")
    
    # Create mapping from trajectory_id to indices
    trajectory_id_to_indices = {}
    for traj_id in unique_traj_ids:
        trajectory_id_to_indices[traj_id] = np.where(data['trajectory_ids'] == traj_id)[0]
    
    # Save everything
    preprocessed_data = {
        'states': states,
        'actions': actions,
        'trajectory_ids': data['trajectory_ids'],
        'unique_trajectory_ids': unique_traj_ids,
        'trajectory_id_to_indices': trajectory_id_to_indices,
        'num_trajectories': num_trajectories_total,
        'num_samples': len(states)
    }
    
    print(f"\nSaving to {output_file}...")
    with open(output_file, 'wb') as f:
        pickle.dump(preprocessed_data, f)
    
    file_size_mb = os.path.getsize(output_file) / (1024 * 1024)
    print(f"✓ Saved preprocessed dataset ({file_size_mb:.2f} MB)")
    print(f"  Contains {num_trajectories_total} trajectories with {len(states)} samples")
    
    return preprocessed_data

if __name__ == '__main__':
    output_file = 'dataset_preprocessed.pkl'
    if len(sys.argv) > 1:
        output_file = sys.argv[1]
    
    preprocess_and_save_dataset(output_file=output_file)

