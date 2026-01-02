"""
Utility functions for getting paths in the variation folder structure.
Provides a unified way to access models, scalers, datasets, and results.

Used AI to generate the code.
"""
import os
import glob

def get_variation_from_base_filename(base_filename):
    """
    Map base_filename to variation name for folder structure.
    
    Args:
        base_filename: Base filename (e.g., 'dataset_even_simpler', 'dataset', 'dataset_direct')
    
    Returns:
        Variation name (e.g., 'even_simpler', 'default', 'direct')
    """
    # Remove 'dataset_' prefix if present
    if base_filename.startswith('dataset_'):
        variation = base_filename.replace('dataset_', '')
    else:
        variation = 'default'
    
    # Map known variations
    variation_map = {
        'even_simpler': 'even_simpler',
        'simpler': 'simpler',
        'direct': 'direct',
        'delta': 'delta',
        'taskspace': 'taskspace',
        'taskspace_absolutes': 'taskspace_absolutes',
        'taskspace_absolutes_random_start': 'taskspace_absolutes_random_start',
        'random_start': 'random_start',
        'dataset': 'default',
        '': 'default'
    }
    
    return variation_map.get(variation, 'default')

def get_variation_from_model_filename(model_filename):
    """
    Extract variation name from model filename.
    
    Args:
        model_filename: Model filename (e.g., 'model_delta_500traj_0success.pth')
    
    Returns:
        Variation name (e.g., 'delta', 'default')
    """
    basename = os.path.basename(model_filename)
    if basename.startswith("model_taskspace_absolutes_random_start_"):
        return "taskspace_absolutes_random_start"
    elif basename.startswith("model_taskspace_absolutes_"):
        return "taskspace_absolutes"
    elif basename.startswith("model_delta_"):
        return "delta"
    elif basename.startswith("model_direct_"):
        return "direct"
    elif basename.startswith("model_even_simpler_"):
        return "even_simpler"
    elif basename.startswith("model_simpler_"):
        return "simpler"
    elif basename.startswith("model_taskspace_"):
        return "taskspace"
    elif basename.startswith("model_random_start_"):
        return "random_start"
    else:
        return "default"

def get_variation_path(base_name='default'):
    """
    Get the base path for a variation.
    
    Args:
        base_name: Variation name (e.g., 'even_simpler', 'simpler', 'direct', 'delta', 
                   'taskspace', 'taskspace_absolutes', 'taskspace_absolutes_random_start', 'random_start', 'default')
    
    Returns:
        Path to variation folder (e.g., 'data/even_simpler')
    """
    # Map base_name to variation folder name
    variation_map = {
        'even_simpler': 'even_simpler',
        'simpler': 'simpler',
        'direct': 'direct',
        'delta': 'delta',
        'taskspace': 'taskspace',
        'taskspace_absolutes': 'taskspace_absolutes',
        'taskspace_absolutes_random_start': 'taskspace_absolutes_random_start',
        'random_start': 'random_start',
        'default': 'default'
    }
    
    variation_name = variation_map.get(base_name, 'default')
    return os.path.join('data', variation_name)

def get_model_path(base_name='default'):
    """Get path to models folder for a variation."""
    return os.path.join(get_variation_path(base_name), 'models')

def get_dataset_path(base_name='default'):
    """Get path to datasets folder for a variation."""
    return os.path.join(get_variation_path(base_name), 'datasets')

def get_scaler_path(base_name='default'):
    """Get path to scalers folder for a variation."""
    return os.path.join(get_variation_path(base_name), 'scalers')

def find_model_file(pattern, base_name=None):
    """
    Find model files matching a pattern.
    
    Args:
        pattern: Pattern to match (e.g., 'model_*traj_*success.pth', 'model_delta_500traj_*.pth')
        base_name: Optional variation name. If None, searches all variations.
    
    Returns:
        List of matching model file paths
    """
    if base_name:
        # Search in specific variation
        model_dir = get_model_path(base_name)
        pattern_path = os.path.join(model_dir, pattern)
        return glob.glob(pattern_path)
    else:
        # Search in all variations
        all_models = []
        variations = ['default', 'delta', 'direct', 'even_simpler', 'simpler', 
                     'taskspace', 'taskspace_absolutes', 'taskspace_absolutes_random_start', 'random_start']
        for variation in variations:
            model_dir = get_model_path(variation)
            pattern_path = os.path.join(model_dir, pattern)
            all_models.extend(glob.glob(pattern_path))
        return all_models

def find_scaler_file(scaler_type, base_name, traj_count):
    """
    Find a scaler file by type, variation, and trajectory count.
    
    Args:
        scaler_type: 'state' or 'action'
        base_name: Variation name (e.g., 'delta', 'direct', 'even_simpler')
        traj_count: Number of trajectories
    
    Returns:
        Path to scaler file if found, None otherwise
    """
    scaler_dir = get_scaler_path(base_name)
    scaler_file = os.path.join(scaler_dir, f"{scaler_type}_scaler_{base_name}_{traj_count}traj.pkl")
    if os.path.exists(scaler_file):
        return scaler_file
    return None

def find_dataset_files(base_filename, pattern='*.csv'):
    """
    Find dataset files for a given base_filename.
    
    Args:
        base_filename: Base filename (e.g., 'dataset_even_simpler', 'dataset')
        pattern: File pattern to match (default: '*.csv')
    
    Returns:
        List of matching dataset file paths
    """
    variation = get_variation_from_base_filename(base_filename)
    dataset_dir = get_dataset_path(variation)
    pattern_path = os.path.join(dataset_dir, f"{base_filename}_{pattern}")
    files = glob.glob(pattern_path)
    
    # Also check for old format single file
    old_file = os.path.join(dataset_dir, f"{base_filename}.csv")
    if os.path.exists(old_file) and old_file not in files:
        files.append(old_file)
    
    # Fallback to current directory for backward compatibility
    if not files:
        pattern_path = f"{base_filename}_{pattern}"
        files = glob.glob(pattern_path)
        old_file = f"{base_filename}.csv"
        if os.path.exists(old_file) and old_file not in files:
            files.append(old_file)
    
    return sorted(files)

def get_model_path_for_file(model_filename):
    """
    Get the full path to a model file, searching in the appropriate variation folder.
    
    Args:
        model_filename: Model filename (e.g., 'model_delta_500traj_0success.pth')
    
    Returns:
        Full path to model file if found, None otherwise
    """
    variation = get_variation_from_model_filename(model_filename)
    model_dir = get_model_path(variation)
    model_path = os.path.join(model_dir, os.path.basename(model_filename))
    
    if os.path.exists(model_path):
        return model_path
    
    # Fallback: check if file exists in current directory (legacy)
    if os.path.exists(model_filename):
        return model_filename
    
    return None

def get_scaler_path_for_model(model_filename, scaler_type='state'):
    """
    Get the path to a scaler file associated with a model.
    
    Args:
        model_filename: Model filename (e.g., 'model_delta_500traj_0success.pth')
        scaler_type: 'state' or 'action'
    
    Returns:
        Path to scaler file if found, None otherwise
    """
    variation = get_variation_from_model_filename(model_filename)
    basename = os.path.basename(model_filename)
    
    # Extract trajectory count from filename
    parts = basename.split('_')
    traj_count = None
    for part in parts:
        if 'traj' in part:
            try:
                traj_count = int(part.replace('traj', ''))
                break
            except:
                pass
    
    if traj_count is None:
        return None
    
    return find_scaler_file(scaler_type, variation, traj_count)

