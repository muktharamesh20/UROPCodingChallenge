"""
Helper script to load and parse the CSV dataset.
Converts string columns back to numpy arrays.

Used AI to generate the code.
"""
import pandas as pd
import numpy as np
import os
import glob
import time
from .path_utils import get_variation_from_base_filename, get_dataset_path

def load_dataset_csv(filename='dataset.csv', base_filename='dataset', max_trajectories=None):
    """
    Load CSV dataset(s) and convert string columns to numpy arrays.
    If filename is a pattern or base_filename is provided, loads all matching files.
    
    Args:
        filename: Single CSV file path, or if None, uses base_filename pattern
        base_filename: Base name for finding multiple files (e.g., 'dataset' finds dataset_*.csv)
        max_trajectories: If provided, only load up to this many trajectories (for memory efficiency)
    
    Returns:
        data: dict with keys:
            - 'arm_states': (N, 21) array
            - 'block_states': (N, 10) array  
            - 'actions': (N, 8) array
            - 'isDone': (N,) boolean array
            - 'trajectory_ids': (N,) array
    """
    # Determine variation folder
    variation = get_variation_from_base_filename(base_filename)
    dataset_dir = get_dataset_path(variation)
    
    # Find all dataset files
    if filename and os.path.exists(filename):
        # Single file specified and exists (absolute path or in current directory)
        files = [filename]
    else:
        # First try to find in variation folder
        pattern = os.path.join(dataset_dir, f"{base_filename}_*.csv")
        files = sorted(glob.glob(pattern))
        
        # Also check for old format single file in variation folder
        old_filename = os.path.join(dataset_dir, f"{base_filename}.csv")
        if os.path.exists(old_filename) and old_filename not in files:
            files.append(old_filename)
            print(f"  Found old format file: {old_filename}")
        
        # Fallback: check current directory (for backward compatibility)
        if not files:
            pattern = f"{base_filename}_*.csv"
            files = sorted(glob.glob(pattern))
            old_filename = f"{base_filename}.csv"
            if os.path.exists(old_filename) and old_filename not in files:
                files.append(old_filename)
                print(f"  Found dataset in current directory (legacy location)")
        
        # If still nothing and filename was specified, try that
        if not files and filename:
            if os.path.exists(filename):
                files = [filename]
    
    if not files:
        raise FileNotFoundError(f"No dataset files found. Tried: {dataset_dir}/{base_filename}_*.csv, {dataset_dir}/{base_filename}.csv, {base_filename}_*.csv, {base_filename}.csv")
    
    print(f"Loading dataset from {len(files)} file(s)...")
    if max_trajectories:
        print(f"  Limiting to first {max_trajectories} trajectories for memory efficiency")
    
    # Process files and collect only needed trajectories
    # Process new format files first, then old format (to avoid duplicates)
    # New format files take precedence over old format for overlapping IDs
    old_format_files = [f for f in files if os.path.basename(f) == f"{base_filename}.csv"]
    new_format_files = [f for f in files if f not in old_format_files]
    # Reorder: new format first (sorted), then old format
    files_ordered = sorted(new_format_files) + old_format_files
    
    print(f"  Processing order: {len(new_format_files)} new format files, {len(old_format_files)} old format files")
    
    all_rows = []
    trajectory_count = 0
    max_traj_id_needed = max_trajectories - 1 if max_trajectories else None
    seen_trajectory_ids = set()  # Track which trajectory IDs we've already loaded
    
    for f in files_ordered:
        # Always process all files, but filter by max_trajectories inside
        # Don't break early - old format file might have unique trajectories
        file_basename = os.path.basename(f)
        
        # Check if file is empty
        if os.path.getsize(f) == 0:
            print(f"  Skipping {file_basename} (empty file)")
            continue
        
        print(f"  Processing {file_basename}...")
        # Read in chunks to reduce memory usage
        chunk_size = 100000  # Read 100k rows at a time
        file_traj_count = 0
        skipped_duplicates = 0
        file_rows_added = 0
        
        # Retry logic for concurrent file access
        max_read_retries = 3
        read_retry_delay = 0.2
        chunk_iterator = None
        
        for read_attempt in range(max_read_retries):
            try:
                # Skip temporary files (being written)
                if f.endswith('.tmp'):
                    continue
                
                # Try reading with error handling - skip bad lines
                # on_bad_lines='skip' is available in pandas 1.3.0+
                try:
                    chunk_iterator = pd.read_csv(f, chunksize=chunk_size, on_bad_lines='skip', engine='python')
                    break  # Success, exit retry loop
                except TypeError:
                    # Fallback for older pandas versions
                    chunk_iterator = pd.read_csv(f, chunksize=chunk_size, error_bad_lines=False, warn_bad_lines=False, engine='python')
                    break  # Success
            except pd.errors.EmptyDataError:
                if read_attempt < max_read_retries - 1:
                    # File might be empty temporarily, wait and retry
                    time.sleep(read_retry_delay)
                    continue
                else:
                    print(f"  Skipping {file_basename} (empty or invalid CSV)")
                    break
            except (IOError, OSError, pd.errors.ParserError) as e:
                if read_attempt < max_read_retries - 1:
                    # File might be locked or being written, wait and retry
                    print(f"  Warning: File read conflict for {file_basename} (attempt {read_attempt + 1}/{max_read_retries}), retrying...")
                    time.sleep(read_retry_delay * (read_attempt + 1))
                    continue
                else:
                    print(f"  Warning: Error reading {file_basename} after {max_read_retries} attempts: {e}")
                    print(f"  Skipping this file...")
                    break
            except Exception as e:
                print(f"  Warning: Unexpected error reading {file_basename}: {e}")
                print(f"  Skipping this file...")
                break
        
        if chunk_iterator is None:
            continue
            try:
                # Try with on_bad_lines first
                try:
                    chunk_iterator = pd.read_csv(f, chunksize=chunk_size, on_bad_lines='skip', engine='python')
                except TypeError:
                    # Fallback for older pandas
                    chunk_iterator = pd.read_csv(f, chunksize=chunk_size, error_bad_lines=False, warn_bad_lines=False, engine='python')
            except Exception as e2:
                print(f"  ✗ Could not read {file_basename}: {e2}")
                print(f"  Skipping this file...")
                continue
        
        for chunk in chunk_iterator:
            # Don't filter by trajectory_id - we want to count unique trajectories, not filter by ID
            # The max_trajectories limit is handled by counting unique trajectories loaded
            
            # Skip trajectories we've already seen (from previous files)
            # This handles duplicates: new format files take precedence
            chunk = chunk[~chunk['trajectory_id'].isin(seen_trajectory_ids)]
            if len(chunk) == 0:
                continue
            
            # Track which trajectory IDs we're adding (before filtering)
            chunk_traj_ids_original = chunk['trajectory_id'].values
            
            # Convert string columns to numpy arrays on the fly (more memory efficient)
            # Handle potential inconsistencies in data format
            chunk_arm_list = []
            chunk_block_list = []
            chunk_action_list = []
            valid_indices = []
            
            for idx, (arm_str, block_str, action_str) in enumerate(zip(
                chunk['kinematic_state_arm'], 
                chunk['kinematic_state_block'], 
                chunk['action']
            )):
                try:
                    arm_arr = np.fromstring(arm_str, sep=',')
                    block_arr = np.fromstring(block_str, sep=',')
                    action_arr = np.fromstring(action_str, sep=',')
                    
                    # Expected dimensions: arm (21 or 28), block (10), action (7 or 8)
                    # Filter out rows with unexpected dimensions
                    if len(block_arr) == 10 and len(action_arr) in [7, 8]:
                        chunk_arm_list.append(arm_arr)
                        chunk_block_list.append(block_arr)
                        chunk_action_list.append(action_arr)
                        valid_indices.append(idx)
                except Exception as e:
                    # Skip malformed rows
                    continue
            
            if len(valid_indices) == 0:
                print(f"  ⚠️  Warning: No valid rows in chunk, skipping...")
                continue
            
            # Convert to numpy arrays (now all have consistent shapes)
            chunk_arm = np.array(chunk_arm_list)
            chunk_block = np.array(chunk_block_list)
            chunk_action = np.array(chunk_action_list)
            
            # Filter other columns to match valid indices
            chunk_isDone = chunk['isDone'].values[valid_indices].astype(bool)
            chunk_traj_ids = chunk_traj_ids_original[valid_indices]
            
            # Update trajectory tracking with valid indices only
            unique_traj_ids_in_chunk = np.unique(chunk_traj_ids)
            seen_trajectory_ids.update(unique_traj_ids_in_chunk)
            
            # Load step information if available (filter to valid indices)
            chunk_steps = None
            if 'step' in chunk.columns:
                chunk_steps = chunk['step'].values[valid_indices].astype(np.int64)
            
            # Store as dict to avoid keeping pandas DataFrame in memory
            row_dict = {
                'arm': chunk_arm,
                'block': chunk_block,
                'action': chunk_action,
                'isDone': chunk_isDone,
                'trajectory_ids': chunk_traj_ids
            }
            if chunk_steps is not None:
                row_dict['steps'] = chunk_steps
            all_rows.append(row_dict)
            
            file_rows_added += len(chunk_traj_ids)
            
            # Track max trajectory ID seen (for reporting)
            if len(unique_traj_ids_in_chunk) > 0:
                file_traj_count = max(file_traj_count, unique_traj_ids_in_chunk.max() + 1)
            
            # Stop if we've loaded enough unique trajectories
            if max_trajectories and len(seen_trajectory_ids) >= max_trajectories:
                # We've loaded enough, but finish processing current chunk
                break
        
        if skipped_duplicates > 0:
            print(f"    Skipped {skipped_duplicates} duplicate rows")
        print(f"    Added {file_rows_added} rows, {len(seen_trajectory_ids)} total unique trajectories so far")
        
        # Stop if we've loaded enough unique trajectories
        if max_trajectories and len(seen_trajectory_ids) >= max_trajectories:
            print(f"    Reached {max_trajectories} unique trajectories, stopping file processing")
            break
    
    # Combine all chunks into final arrays
    print("  Combining chunks into arrays...")
    if not all_rows:
        # Return empty data structure instead of raising error
        print("  Warning: No data found in any files. Returning empty dataset.")
        return {
            'arm_states': np.empty((0, 21), dtype=np.float32),
            'block_states': np.empty((0, 10), dtype=np.float32),
            'actions': np.empty((0, 8), dtype=np.float32),
            'isDone': np.empty((0,), dtype=bool),
            'trajectory_ids': np.empty((0,), dtype=np.int64)
        }
    
    arm_states = np.concatenate([r['arm'] for r in all_rows], axis=0)
    block_states = np.concatenate([r['block'] for r in all_rows], axis=0)
    actions = np.concatenate([r['action'] for r in all_rows], axis=0)
    isDone = np.concatenate([r['isDone'] for r in all_rows], axis=0)
    trajectory_ids = np.concatenate([r['trajectory_ids'] for r in all_rows], axis=0)
    
    # Load step information if available
    steps = None
    if all_rows and 'steps' in all_rows[0]:
        steps = np.concatenate([r['steps'] for r in all_rows], axis=0)
    
    # Clear intermediate data
    del all_rows
    
    num_unique_trajs = len(np.unique(trajectory_ids))
    print(f"Combined: {len(arm_states)} rows from {num_unique_trajs} trajectories")
    print(f"  Arm state shape: {arm_states.shape}")
    print(f"  Block state shape: {block_states.shape}")
    print(f"  Action shape: {actions.shape}")
    if steps is not None:
        print(f"  Steps shape: {steps.shape}")
    
    result = {
        'arm_states': arm_states,
        'block_states': block_states,
        'actions': actions,
        'isDone': isDone,
        'trajectory_ids': trajectory_ids
    }
    
    if steps is not None:
        result['steps'] = steps
    
    return result

def get_state_action_pairs(data):
    """
    Combine arm and block states into single state vector.
    
    Returns:
        states: (N, 31) array [arm_state(21), block_state(10)]
        actions: (N, 8) array
    """
    states = np.concatenate([data['arm_states'], data['block_states']], axis=1)
    return states, data['actions']

if __name__ == "__main__":
    # Example usage
    data = load_dataset_csv('dataset.csv')
    states, actions = get_state_action_pairs(data)
    print(f"\nCombined state shape: {states.shape}")
    print(f"Action shape: {actions.shape}")

