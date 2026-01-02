"""
Visualize a trained model running in MuJoCo.
Can load any model from the training pipeline.

Used AI to generate the code.
"""
import mujoco
import mujoco.viewer
import numpy as np
import torch
import torch.nn as nn
import time
import sys
import glob
import pickle
import os
from scipy.spatial.transform import Rotation as R
from sklearn.preprocessing import StandardScaler
from utils.training_utils import (
    verify_placement_on_green as verifyPlacementOnGreen,
    extract_arm_state,
    extract_block_state,
    compute_desired_pose_from_delta,
    compute_desired_pose_from_absolute,
    convert_state_dimension,
    get_joint_range_for_joint
)
from utils.ik_utils import moveArm
from utils.path_utils import (
    find_model_file,
    get_scaler_path_for_model,
    get_model_path_for_file,
    get_variation_from_model_filename
)
from utils.visualization_utils import detect_environment_from_filename
from trajectories.trajectoriesCircle import sample_block_pos, sample_block_size, verifyPlacement
from trajectories.trajectoriesSimpler import (
    sample_block_pos as sample_block_pos_simpler,
    sample_block_size as sample_block_size_simpler,
    sample_robot_start_position,
    placement_box_pos, placement_box_size
)
from trajectories.trajectoriesEvenSimpler import (
    sample_block_pos as sample_block_pos_even_simpler,
    sample_block_size as sample_block_size_even_simpler,
    get_robot_start_position,
    placement_box_pos as placement_box_pos_even_simpler,
    placement_box_size as placement_box_size_even_simpler
)
from trajectories.trajectoriesDirect import (
    sample_block_pos as sample_block_pos_direct,
    sample_block_size as sample_block_size_direct,
    get_robot_start_position as get_robot_start_position_direct,
    placement_box_pos as placement_box_pos_direct,
    placement_box_size as placement_box_size_direct
)
from trajectories.trajectoriesTaskSpace import (
    placement_box_pos as placement_box_pos_taskspace,
    placement_box_size as placement_box_size_taskspace
)

# verifyPlacementOnGreen is now imported from training_utils

class MLP(nn.Module):
    """MLP matching training code structure"""
    def __init__(self, input_size=31, output_size=8, architecture='new'):
        super(MLP, self).__init__()
        if architecture == 'direct':
            # Direct architecture: 1024 -> 1024 -> 512 -> 256 -> output (largest)
            self.fc1 = nn.Linear(input_size, 1024)
            self.relu1 = nn.ReLU()
            self.fc2 = nn.Linear(1024, 1024)
            self.relu2 = nn.ReLU()
            self.fc3 = nn.Linear(1024, 512)
            self.relu3 = nn.ReLU()
            self.fc4 = nn.Linear(512, 256)
            self.relu4 = nn.ReLU()
            self.fc5 = nn.Linear(256, output_size)
        elif architecture == 'old_direct':
            # Old direct architecture: 512 -> 512 -> 128 -> output
            self.fc1 = nn.Linear(input_size, 512)
            self.relu1 = nn.ReLU()
            self.fc2 = nn.Linear(512, 512)
            self.relu2 = nn.ReLU()
            self.fc3 = nn.Linear(512, 128)
            self.relu3 = nn.ReLU()
            self.fc4 = nn.Linear(128, output_size)
        elif architecture == 'even_simpler':
            # Even simpler architecture: 1024 -> 1024 -> 512 -> 256 -> output (largest)
            self.fc1 = nn.Linear(input_size, 1024)
            self.relu1 = nn.ReLU()
            self.fc2 = nn.Linear(1024, 1024)
            self.relu2 = nn.ReLU()
            self.fc3 = nn.Linear(1024, 512)
            self.relu3 = nn.ReLU()
            self.fc4 = nn.Linear(512, 256)
            self.relu4 = nn.ReLU()
            self.fc5 = nn.Linear(256, output_size)
        elif architecture == 'old_even_simpler':
            # Old even simpler architecture: 512 -> 512 -> 128 -> output
            self.fc1 = nn.Linear(input_size, 512)
            self.relu1 = nn.ReLU()
            self.fc2 = nn.Linear(512, 512)
            self.relu2 = nn.ReLU()
            self.fc3 = nn.Linear(512, 128)
            self.relu3 = nn.ReLU()
            self.fc4 = nn.Linear(128, output_size)
        elif architecture == 'new' or architecture == 'simpler':
            # New architecture: 256 -> 256 -> 64 -> output
            self.fc1 = nn.Linear(input_size, 256)
            self.relu1 = nn.ReLU()
            self.fc2 = nn.Linear(256, 256)
            self.relu2 = nn.ReLU()
            self.fc3 = nn.Linear(256, 64)
            self.relu3 = nn.ReLU()
            self.fc4 = nn.Linear(64, output_size)
        else:
            # Old architecture: 128 -> 64 -> output
            self.fc1 = nn.Linear(input_size, 128)
            self.relu1 = nn.ReLU()
            self.fc2 = nn.Linear(128, 64)
            self.relu2 = nn.ReLU()
            self.fc3 = nn.Linear(64, output_size)

    def forward(self, x):
        if hasattr(self, 'fc5'):
            # Direct or even_simpler architecture: 5 layers (1024->1024->512->256->output)
            x = self.relu1(self.fc1(x))
            x = self.relu2(self.fc2(x))
            x = self.relu3(self.fc3(x))
            x = self.relu4(self.fc4(x))
            x = self.fc5(x)
        elif hasattr(self, 'fc4'):
            # New architecture (simpler or old even_simpler): 4 layers
            x = self.relu1(self.fc1(x))
            x = self.relu2(self.fc2(x))
            x = self.relu3(self.fc3(x))
            x = self.fc4(x)
        else:
            # Old architecture: 3 layers
            x = self.relu1(self.fc1(x))
            x = self.relu2(self.fc2(x))
            x = self.fc3(x)
        return x

# extract_arm_state and extract_block_state are now imported from training_utils

def list_available_models():
    """List all available trained models"""
    # Use path_utils to search for all models across all variations
    # Single pattern matches all model files: model_*traj_*success.pth
    models = find_model_file("model_*traj_*success.pth")
    models = sorted(set(models))  # Remove duplicates
    if not models:
        print("No trained models found. Run train_and_evaluate_pipeline.py first.")
        return []
    
    print("Available models:")
    for i, model_path in enumerate(sorted(models), 1):
        # Get just the filename for display
        model_basename = os.path.basename(model_path)
        # Use path_utils to get variation name
        model_type = get_variation_from_model_filename(model_basename)
        
        # Extract trajectory count and success rate from filename
        # Format: model_{variation}_{num_traj}traj_{success_rate}success.pth
        parts = model_basename.replace('.pth', '').split('_')
        num_traj = None
        success = None
        for part in parts:
            if 'traj' in part:
                try:
                    num_traj = part.replace('traj', '')
                except:
                    pass
            if 'success' in part:
                try:
                    success = part.replace('success', '')
                except:
                    pass
        
        if num_traj and success:
            print(f"  {i}. {model_basename} ({model_type}, {num_traj} trajectories, {success}% success)")
        else:
            print(f"  {i}. {model_basename} ({model_type})")
    
    return sorted(models)

def load_state_scaler(model_path):
    """Load state scaler for delta, even_simpler, or direct models if available."""
    scaler_file = get_scaler_path_for_model(model_path, scaler_type='state')
    if scaler_file and os.path.exists(scaler_file):
        print(f"  Loading state scaler from {scaler_file}...")
        with open(scaler_file, 'rb') as f:
            return pickle.load(f)
    return None

def load_action_scaler(model_path):
    """Load action scaler (joint ranges) for direct or even_simpler models if available."""
    scaler_file = get_scaler_path_for_model(model_path, scaler_type='action')
    if scaler_file and os.path.exists(scaler_file):
        print(f"  Loading action scaler from {scaler_file}...")
        with open(scaler_file, 'rb') as f:
            return pickle.load(f)
    return None

def normalize_action(action, joint_ranges):
    """Normalize action to [0, 1] range."""
    normalized = np.zeros_like(action)
    for i in range(7):
        minr, maxr = joint_ranges[i]
        clamped = np.clip(action[i], minr, maxr)
        normalized[i] = (clamped - minr) / (maxr - minr) if (maxr - minr) > 1e-6 else 0.5
    normalized[7] = action[7]  # Gripper already in [0, 1]
    return normalized

def denormalize_action(normalized_action, joint_ranges):
    """Denormalize action from [0, 1] range back to joint positions."""
    denormalized = np.zeros_like(normalized_action)
    for i in range(7):
        minr, maxr = joint_ranges[i]
        denormalized[i] = minr + normalized_action[i] * (maxr - minr)
    denormalized[7] = normalized_action[7]  # Gripper stays in [0, 1]
    return denormalized

def load_model(model_path, device=None):
    """
    Load a trained model and infer architecture from checkpoint weights.
    
    Args:
        model_path: Path to model file
        device: Device to load model on (if None, auto-detects: MPS > CUDA > CPU)
    
    Returns:
        model: Loaded model
        input_size: Input size of the model
    """
    # Auto-detect device if not provided
    if device is None:
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device('mps')
        elif torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
    
    # Ensure we have the full path to the model file
    full_model_path = get_model_path_for_file(model_path)
    if full_model_path is None:
        if os.path.exists(model_path):
            full_model_path = model_path
        else:
            raise FileNotFoundError(f"Model file not found: {model_path}")
    
    print(f"Loading model from {full_model_path}...")
    checkpoint = torch.load(full_model_path, map_location=device, weights_only=False)
    
    # Extract layer sizes from checkpoint
    input_size = checkpoint['fc1.weight'].shape[1]
    layer_sizes = []
    for i in range(1, 6):  # Check fc1 through fc5
        key = f'fc{i}.weight'
        if key in checkpoint:
            layer_sizes.append(checkpoint[key].shape[0])
        else:
            break
    
    # Determine output size from last layer
    output_size = layer_sizes[-1] if layer_sizes else checkpoint['fc3.weight'].shape[0]
    
    # Match architecture pattern from layer sizes
    # Architecture patterns: (fc1, fc2, fc3, fc4) - fc5 output size varies
    architecture_patterns = {
        (1024, 1024, 512, 256): 'direct',      # New direct/even_simpler (with fc5)
        (512, 512, 128): 'old_direct',         # Old direct/even_simpler: 512→512→128→output
        (256, 256, 64): 'new',                 # Standard: 256→256→64→output
        (128, 64): 'old',                      # Old: 128→64→output
    }
    
    # Try to match pattern (check first N layers where N matches pattern length)
    architecture = None
    for pattern, arch in architecture_patterns.items():
        if len(layer_sizes) >= len(pattern) and layer_sizes[:len(pattern)] == list(pattern):
            architecture = arch
            break
    
    # Fallback: use layer count and filename if pattern doesn't match
    if architecture is None:
        model_basename = os.path.basename(full_model_path).lower()
        if len(layer_sizes) == 5:  # Has fc5
            architecture = 'direct' if 'direct' in model_basename else 'even_simpler'
        elif len(layer_sizes) == 4:  # Has fc4
            if layer_sizes[0] == 1024:
                architecture = 'direct' if 'direct' in model_basename else 'even_simpler'
            else:
                architecture = 'new'
        else:
            architecture = 'old'
    
    # Special case: distinguish old_direct vs old_even_simpler (both have 512→512→128)
    if architecture == 'old_direct':
        model_basename = os.path.basename(full_model_path).lower()
        if 'direct' not in model_basename:
            architecture = 'old_even_simpler'
    
    # Special case: distinguish direct vs even_simpler for 5-layer models (both have 1024→1024→512→256)
    if architecture == 'direct' and len(layer_sizes) == 5:
        model_basename = os.path.basename(full_model_path).lower()
        if 'direct' not in model_basename and ('delta' in model_basename or 'even' in model_basename):
            architecture = 'even_simpler'
    
    model = MLP(input_size=input_size, output_size=output_size, architecture=architecture)
    model.load_state_dict(checkpoint, strict=False)
    model.eval()
    
    print(f"✓ Model loaded: input_size={input_size}, output_size={output_size}, architecture={architecture}")
    return model, input_size

def visualize_model(model_path=None, num_trials=5, action_scale=0.1, delay=0.002, use_standardized=None, max_angvel=0.3, action_smoothing=None, pose_update_frequency=None):
    """
    Visualize a trained model running in MuJoCo.
    
    Args:
        model_path: Path to model file. If None, will list available models.
        num_trials: Number of random blocks to try
        action_scale: Scale factor for actions (lower = slower movement)
        delay: Delay between steps in seconds
        use_standardized: Whether to use standardized input (for even_simpler models). 
                         If None, will prompt user to choose.
        max_angvel: Maximum angular velocity for moveArm (rad/s, default: 0.3 for slower, smoother movement)
        action_smoothing: Exponential moving average factor for action smoothing (0-1, higher = more smoothing). 
                         If None, will prompt user to choose.
        pose_update_frequency: How often to update desired pose (1 = every step, 2 = every 2 steps, etc.).
                              Only applies to taskspace models. If None, will prompt user to choose.
    """
    # List models if none specified
    if model_path is None:
        models = list_available_models()
        if not models:
            return
        print("\nEnter model number to visualize (or press Enter for first): ", end="")
        try:
            choice = input().strip()
            if choice == "":
                model_path = models[0]
            else:
                model_path = models[int(choice) - 1]
        except (ValueError, IndexError):
            print("Invalid choice, using first model")
            model_path = models[0]
    
    # Load model (device auto-detected)
    model, model_input_size = load_model(model_path)
    # Get device from model to use for inference
    device = next(model.parameters()).device
    print(f"Using device: {device}")
    
    # Determine if model uses sin/cos encoding based on input size
    # 31D = no sin/cos encoding (7 joint_pos + 7 joint_vel + 3 ee_pos + 4 ee_quat + 3 block_pos + 4 block_quat + 3 block_size)
    # 38D = sin/cos encoding (7 sin + 7 cos + 7 joint_vel + 3 ee_pos + 4 ee_quat + 3 block_pos + 4 block_quat + 3 block_size)
    use_sin_cos_encoding = (model_input_size == 38)
    if use_sin_cos_encoding:
        print(f"  Model uses sin/cos encoding (input_size={model_input_size})")
    else:
        print(f"  Model does NOT use sin/cos encoding (input_size={model_input_size})")
    
    # Detect environment settings from model filename
    env_info = detect_environment_from_filename(model_path)
    use_taskspace_absolutes = env_info['use_taskspace_absolutes']
    use_taskspace = env_info['use_taskspace']
    use_direct = env_info['use_direct']
    use_delta = env_info['use_delta']
    use_even_simpler = env_info['use_even_simpler']
    use_simpler = env_info['use_simpler']
    state_scaler = None
    use_standardized_input = False
    
    if use_taskspace_absolutes:
        print("  Detected taskspace_absolutes model - using pickAndPlaceSimpler.xml with fixed starting position")
        print(f"  Model path: {model_path}")
        # Ask user if they want 10x speedup
        print("  Run at 10x speed? (y/n, default: n): ", end="")
        try:
            choice = input().strip().lower()
            if choice == 'y' or choice == 'yes':
                delay = delay / 10.0
                print(f"  Running at 10x speed (delay: {delay:.6f}s)")
            else:
                print(f"  Running at normal speed (delay: {delay:.6f}s)")
        except:
            print(f"  Running at normal speed (delay: {delay:.6f}s)")
        
        # Try to load state scaler (taskspace models always use state normalization)
        state_scaler = load_state_scaler(model_path)
        
        # Taskspace_absolutes models don't use action_scaler (actions are raw absolute poses)
        action_scaler = None
        print(f"  Taskspace_absolutes model: actions are absolute poses (7D: ee_pos(3) + ee_quat(3) + gripper(1))")
        
        # Taskspace models always use standardized input (state normalization)
        if state_scaler is not None:
            use_standardized_input = True
            print(f"  State scaler found - using normalized 38D states (sin/cos encoding)")
        else:
            print("  ⚠ Warning: No state scaler found for taskspace_absolutes model!")
            print("  This model was likely trained with state normalization - performance may be poor")
            use_standardized_input = False
    elif use_taskspace:
        print("  Detected taskspace (delta) model - using pickAndPlaceSimpler.xml with fixed starting position")
        print(f"  Model path: {model_path}")
        # Ask user if they want 10x speedup
        print("  Run at 10x speed? (y/n, default: n): ", end="")
        try:
            choice = input().strip().lower()
            if choice == 'y' or choice == 'yes':
                delay = delay / 10.0
                print(f"  Running at 10x speed (delay: {delay:.6f}s)")
            else:
                print(f"  Running at normal speed (delay: {delay:.6f}s)")
        except:
            print(f"  Running at normal speed (delay: {delay:.6f}s)")
        
        # Try to load state scaler (taskspace models always use state normalization)
        state_scaler = load_state_scaler(model_path)
        
        # Taskspace models don't use action_scaler (actions are task-space deltas)
        action_scaler = None
        print(f"  Taskspace model: actions are task-space deltas (7D: delta_pos(3) + delta_quat(3) + gripper(1))")
        
        # Taskspace models always use standardized input (state normalization)
        if state_scaler is not None:
            use_standardized_input = True
            print(f"  State scaler found - using normalized 38D states (sin/cos encoding)")
        else:
            print("  ⚠ Warning: No state scaler found for taskspace model!")
            print("  This model was likely trained with state normalization - performance may be poor")
            use_standardized_input = False
    elif use_direct:
        print("  Detected direct model - using pickAndPlaceDirect.xml with fixed starting position")
        print(f"  Model path: {model_path}")
        # Ask user if they want 10x speedup
        print("  Run at 10x speed? (y/n, default: n): ", end="")
        try:
            choice = input().strip().lower()
            if choice == 'y' or choice == 'yes':
                delay = delay / 10.0
                print(f"  Running at 10x speed (delay: {delay:.6f}s)")
            else:
                print(f"  Running at normal speed (delay: {delay:.6f}s)")
        except:
            print(f"  Running at normal speed (delay: {delay:.6f}s)")
        
        # Try to load state scaler
        state_scaler = load_state_scaler(model_path)
        
        # Try to load action scaler (joint ranges for denormalization)
        action_scaler = load_action_scaler(model_path)
        if action_scaler is not None:
            print(f"  Action scaler found - will denormalize actions from [0, 1] range")
        
        # Ask user if they want to use standardized input
        if use_standardized is None:
            if state_scaler is not None:
                print("\n  State scaler found for this model.")
                print("  Use standardized (normalized) input? (y/n, default: y): ", end="")
                try:
                    choice = input().strip().lower()
                    use_standardized_input = (choice == '' or choice == 'y' or choice == 'yes')
                except:
                    use_standardized_input = True
                print(f"  {'Using' if use_standardized_input else 'Not using'} standardized input")
            else:
                print("  No state scaler found - using non-standardized input")
                use_standardized_input = False
        else:
            use_standardized_input = use_standardized
            if use_standardized_input and state_scaler is None:
                print("  ⚠ Warning: Standardized input requested but no scaler found. Using non-standardized.")
                use_standardized_input = False
    elif use_delta:
        print("  Detected delta model - using pickAndPlaceSimpler.xml with fixed starting position")
        print(f"  Model path: {model_path}")
        # Ask user if they want 10x speedup
        print("  Run at 10x speed? (y/n, default: n): ", end="")
        try:
            choice = input().strip().lower()
            if choice == 'y' or choice == 'yes':
                delay = delay / 10.0
                print(f"  Running at 10x speed (delay: {delay:.6f}s)")
            else:
                print(f"  Running at normal speed (delay: {delay:.6f}s)")
        except:
            print(f"  Running at normal speed (delay: {delay:.6f}s)")
        
        # Try to load state scaler
        state_scaler = load_state_scaler(model_path)
        
        # Delta models don't use action_scaler (actions are not normalized)
        action_scaler = None
        print(f"  Delta model: actions are not normalized (delta actions)")
        
        # Ask user if they want to use standardized input
        if use_standardized is None:
            if state_scaler is not None:
                print("\n  State scaler found for this model.")
                print("  Use standardized (normalized) input? (y/n, default: y): ", end="")
                try:
                    choice = input().strip().lower()
                    use_standardized_input = (choice == '' or choice == 'y' or choice == 'yes')
                except:
                    use_standardized_input = True
                print(f"  {'Using' if use_standardized_input else 'Not using'} standardized input")
            else:
                print("  No state scaler found - using non-standardized input")
                use_standardized_input = False
        else:
            use_standardized_input = use_standardized
            if use_standardized_input and state_scaler is None:
                print("  ⚠ Warning: Standardized input requested but no scaler found. Using non-standardized.")
                use_standardized_input = False
    elif use_even_simpler:
        print("  Detected even simpler model - using pickAndPlaceSimpler.xml with fixed starting position")
        # Ask user if they want 10x speedup
        print("  Run at 10x speed? (y/n, default: n): ", end="")
        try:
            choice = input().strip().lower()
            if choice == 'y' or choice == 'yes':
                delay = delay / 10.0
                print(f"  Running at 10x speed (delay: {delay:.6f}s)")
            else:
                print(f"  Running at normal speed (delay: {delay:.6f}s)")
        except:
            print(f"  Running at normal speed (delay: {delay:.6f}s)")
        
        # Try to load state scaler
        state_scaler = load_state_scaler(model_path)
        
        # Try to load action scaler (joint ranges for denormalization)
        action_scaler = load_action_scaler(model_path)
        if action_scaler is not None:
            print(f"  Action scaler found - will denormalize actions from [0, 1] range")
        
        # Ask user if they want to use standardized input
        if use_standardized is None:
            if state_scaler is not None:
                print("\n  State scaler found for this model.")
                print("  Use standardized (normalized) input? (y/n, default: y): ", end="")
                try:
                    choice = input().strip().lower()
                    use_standardized_input = (choice == '' or choice == 'y' or choice == 'yes')
                except:
                    use_standardized_input = True
                print(f"  {'Using' if use_standardized_input else 'Not using'} standardized input")
            else:
                print("  No state scaler found - using non-standardized input")
                use_standardized_input = False
        else:
            use_standardized_input = use_standardized
            if use_standardized_input and state_scaler is None:
                print("  ⚠ Warning: Standardized input requested but no scaler found. Using non-standardized.")
                use_standardized_input = False
    elif use_simpler:
        action_scaler = None
        print("  Detected simpler model - using pickAndPlaceSimpler.xml")
    
    # Ask user if they want action smoothing
    if action_smoothing is None:
        print("\n  Enable action smoothing to reduce jerkiness? (y/n, default: y): ", end="")
        try:
            choice = input().strip().lower()
            if choice == '' or choice == 'y' or choice == 'yes':
                action_smoothing = 0.7
                print(f"  Action smoothing enabled (factor: {action_smoothing})")
            else:
                action_smoothing = 0.0  # Disabled
                print("  Action smoothing disabled")
        except:
            action_smoothing = 0.7  # Default to enabled
            print(f"  Action smoothing enabled (factor: {action_smoothing})")
    elif action_smoothing > 0:
        print(f"  Action smoothing enabled (factor: {action_smoothing})")
    else:
        print("  Action smoothing disabled")
    
    # Ask user about pose update frequency (only relevant for taskspace models)
    use_pose_update_frequency = (use_taskspace or use_taskspace_absolutes)
    if use_pose_update_frequency and pose_update_frequency is None:
        print("\n  How often to update desired pose? (1=every step, 2=every 2 steps, etc., default: 1): ", end="")
        try:
            choice = input().strip()
            if choice == '':
                pose_update_frequency = 1
            else:
                pose_update_frequency = max(1, int(choice))
            print(f"  Updating desired pose every {pose_update_frequency} step(s)")
        except:
            pose_update_frequency = 1
            print(f"  Updating desired pose every {pose_update_frequency} step(s)")
    elif use_pose_update_frequency:
        print(f"  Updating desired pose every {pose_update_frequency} step(s)")
    else:
        pose_update_frequency = 1  # Not used for non-taskspace models
    
    successes = 0
    
    for trial in range(num_trials):
        print(f"\n{'='*70}")
        print(f"Trial {trial + 1}/{num_trials}")
        print(f"{'='*70}")
        
        # Setup simulation - use appropriate XML (from environment detection)
        xml_path = env_info['xml_path']
        print(f"  Loading XML from: {xml_path}")
        model_mj = mujoco.MjModel.from_xml_path(xml_path)
        data = mujoco.MjData(model_mj)
        
        block_id = model_mj.body('target').id
        site = model_mj.site('palm_contact_edge_vis').id
        blockGeom_id = model_mj.geom('box').id
        joint_names = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"][:7]
        dof_ids = np.array([model_mj.joint(name).id for name in joint_names])
        actuator_names = ['actuator1', 'actuator2', 'actuator3', 'actuator4', 'actuator5', 'actuator6', 'actuator7'][:7]
        actuator_ids = [model_mj.actuator(name).id for name in actuator_names]
        
        block_jnt_id = model_mj.body(block_id).jntadr[0]
        
        # Get mocap body for visualization (if it exists)
        mocap_id = None
        try:
            mocap_body = model_mj.body('mocap')
            if mocap_body.mocapid[0] >= 0:
                mocap_id = mocap_body.mocapid[0]
        except:
            pass  # Mocap body doesn't exist in this XML
        
        # Sample random block - use appropriate function
        if use_direct:
            block_size = sample_block_size_direct()
            # Use fixed robot starting position (home position, no randomization)
            key_id = model_mj.key("home").id
            qpos_start = get_robot_start_position_direct(model_mj, dof_ids, key_id)
            data.qpos[:7] = qpos_start[:7]
            mujoco.mj_forward(model_mj, data)
        elif use_delta or use_even_simpler or use_taskspace or use_taskspace_absolutes:
            block_size = sample_block_size_even_simpler()
            targetInitialPos = sample_block_pos_even_simpler()
            # Use fixed robot starting position (home position, no randomization)
            key_id = model_mj.key("home").id
            qpos_start = get_robot_start_position(model_mj, dof_ids, key_id)
            data.qpos[:7] = qpos_start[:7]
            mujoco.mj_forward(model_mj, data)
        elif use_simpler:
            block_size = sample_block_size_simpler()
            targetInitialPos = sample_block_pos_simpler()
            # Randomize robot starting position for simpler version
            key_id = model_mj.key("home").id
            qpos_start = sample_robot_start_position(model_mj, dof_ids, key_id)
            data.qpos[:7] = qpos_start[:7]
            mujoco.mj_forward(model_mj, data)
        else:
            block_size = sample_block_size()
            targetInitialPos = sample_block_pos()
        
        # Random orientation
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
        
        # For direct models, calculate block position with rotation to get correct vertical half-height
        if use_direct:
            targetInitialPos = sample_block_pos_direct(block_size=block_size, block_rotation=Q)
        
        quat_mj = np.zeros(4)
        mujoco.mju_mat2Quat(quat_mj, Q.flatten())
        
        data.qpos[block_jnt_id:block_jnt_id+3] = targetInitialPos
        data.qpos[block_jnt_id+3:block_jnt_id+7] = quat_mj
        mujoco.mj_forward(model_mj, data)
        
        # Target position for reference (not used in state, just for display)
        table_surface_z = 0.2
        block_height_world = block_size[2] * 2
        place_pos = np.array([0, 0.3, table_surface_z + block_height_world + 0.005])
        
        print(f"Block size: {block_size}")
        print(f"Block position: {targetInitialPos}")
        print("Generating trajectory (without viewer)...")
        
        # Save initial state for replay
        initial_qpos = data.qpos.copy()
        initial_qvel = data.qvel.copy()
        
        # Generate trajectory first (without viewer)
        trajectory_data = []  # Store (qpos, qvel, ctrl, mocap_pos, mocap_quat) for each step
        step_count = 0
        max_steps = 100000  # Allow much longer runs for visualization
        success_achieved = False
        
        # Reset action smoothing for each trial
        smoothed_action = None
        
        # Reset pose update tracking for each trial
        last_desired_pos = None
        last_desired_quat = None
        pose_update_counter = 0
        
        while step_count < max_steps and not success_achieved:
            # Ensure forward kinematics is up-to-date before reading state
            mujoco.mj_forward(model_mj, data)
            
            # Extract current state
            qpos = data.qpos[:7]
            qvel = data.qvel[:7]
            ee_pos = data.site_xpos[site]
            site_mat = data.site_xmat[site].reshape(3, 3)
            ee_quat = np.zeros(4)
            mujoco.mju_mat2Quat(ee_quat, site_mat.flatten())
            ee_quat = ee_quat / (np.linalg.norm(ee_quat) + 1e-8)
            
            block_pos = data.xpos[block_id]
            block_quat = data.xquat[block_id]
            block_quat = block_quat / (np.linalg.norm(block_quat) + 1e-8)
            
            # Construct state (always extract as 31D, then convert if needed)
            # Note: Timestep is NOT included (only kinematic information)
            arm_state = extract_arm_state(qpos, qvel, ee_pos, ee_quat, use_sin_cos_encoding=False)
            block_state = extract_block_state(block_pos, block_quat, block_size)
            state = np.concatenate([arm_state, block_state])  # 31D state
            
            # Convert to 38D if model expects sin/cos encoding
            if use_sin_cos_encoding:
                state = convert_state_dimension(state, 38)
            
            # Normalize state if using standardized input
            if use_standardized_input and state_scaler is not None:
                state = state_scaler.transform(state.reshape(1, -1)).flatten()
            
            # Get action from policy
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                raw_action = model(state_tensor).squeeze(0).cpu().numpy()
            
            # Apply exponential moving average smoothing to reduce jerkiness (if enabled)
            if action_smoothing > 0:
                if smoothed_action is None:
                    smoothed_action = raw_action.copy()
                else:
                    smoothed_action = action_smoothing * smoothed_action + (1 - action_smoothing) * raw_action
                action_normalized = smoothed_action
            else:
                action_normalized = raw_action
            
            # Handle different action types
            if use_taskspace_absolutes:
                # Taskspace_absolutes models output 7D: [absolute_ee_pos(3), absolute_ee_quat(3), gripper(1)]
                absolute_action = action_normalized
                absolute_ee_pos = absolute_action[:3]
                absolute_ee_quat_3d = absolute_action[3:6]
                gripper_cmd = absolute_action[6]  # Gripper is at index 6 for taskspace
                
                # Update desired pose only every N steps (if pose_update_frequency > 1)
                pose_update_counter += 1
                if pose_update_counter >= pose_update_frequency or last_desired_pos is None:
                    # Convert 3D quaternion representation to full 4D quaternion
                    desired_pos, desired_quat = compute_desired_pose_from_absolute(
                        absolute_ee_pos, absolute_ee_quat_3d
                    )
                    last_desired_pos = desired_pos.copy()
                    last_desired_quat = desired_quat.copy()
                    pose_update_counter = 0
                else:
                    # Reuse last desired pose
                    desired_pos = last_desired_pos.copy()
                    desired_quat = last_desired_quat.copy()
                
                # Store desired pose for mocap (if mocap exists)
                mocap_pos_step = desired_pos.copy() if mocap_id is not None else None
                mocap_quat_step = desired_quat.copy() if mocap_id is not None else None
                
                # Update mocap to show desired pose (if mocap exists)
                if mocap_id is not None:
                    data.mocap_pos[mocap_id] = desired_pos
                    data.mocap_quat[mocap_id] = desired_quat
                
                # Use moveArm to convert desired pose to joint commands
                shut_fingers = gripper_cmd > 0.5
                moveArm(
                    model_mj, data, site, block_id, dof_ids, actuator_ids,
                    desired_pos, desired_quat, shut_fingers,
                    trajectory=None, current_idx=None,
                    placingPhaseStart=None, liftingPhaseStart=None, liftingPhaseEnd=None,
                    damping=1e-10, max_angvel=max_angvel
                )
                # moveArm already set data.ctrl, so we just store it
                ctrl = data.ctrl[actuator_ids].copy()
            elif use_taskspace:
                # Taskspace models output 7D: [delta_ee_pos(3), delta_ee_quat(3), gripper(1)]
                delta_action = action_normalized
                delta_ee_pos = delta_action[:3]
                delta_ee_quat = delta_action[3:6]
                gripper_cmd = delta_action[6]  # Gripper is at index 6 for taskspace
                
                # Update desired pose only every N steps (if pose_update_frequency > 1)
                pose_update_counter += 1
                if pose_update_counter >= pose_update_frequency or last_desired_pos is None:
                    # Clip position delta to prevent jerky movements (reduced for smoother motion)
                    max_pos_delta = 0.05  # Reduced from 0.1 to 0.05 for smoother movement
                    pos_delta_norm = np.linalg.norm(delta_ee_pos)
                    if pos_delta_norm > max_pos_delta:
                        delta_ee_pos = delta_ee_pos / pos_delta_norm * max_pos_delta
                    
                    # Clip quaternion delta magnitude to prevent jerky rotations (reduced for smoother motion)
                    max_quat_delta = 0.1  # Reduced from 0.2 to 0.1 for smoother rotation
                    quat_delta_norm = np.linalg.norm(delta_ee_quat)
                    if quat_delta_norm > max_quat_delta:
                        delta_ee_quat = delta_ee_quat / quat_delta_norm * max_quat_delta
                    
                    # Scale delta for stability (reduced from 10% to 5% for smoother movement)
                    delta_scale = 0.05  # Reduced from 0.1 to 0.05 for smoother movement
                    desired_pos, desired_quat = compute_desired_pose_from_delta(
                        data, site, delta_ee_pos, delta_ee_quat, delta_scale=delta_scale
                    )
                    last_desired_pos = desired_pos.copy()
                    last_desired_quat = desired_quat.copy()
                    pose_update_counter = 0
                else:
                    # Reuse last desired pose
                    desired_pos = last_desired_pos.copy()
                    desired_quat = last_desired_quat.copy()
                
                # Store desired pose for mocap (if mocap exists)
                mocap_pos_step = desired_pos.copy() if mocap_id is not None else None
                mocap_quat_step = desired_quat.copy() if mocap_id is not None else None
                
                # Update mocap to show desired pose (if mocap exists)
                if mocap_id is not None:
                    data.mocap_pos[mocap_id] = desired_pos
                    data.mocap_quat[mocap_id] = desired_quat
                
                # Use moveArm to convert desired pose to joint commands
                shut_fingers = gripper_cmd > 0.5
                moveArm(
                    model_mj, data, site, block_id, dof_ids, actuator_ids,
                    desired_pos, desired_quat, shut_fingers,
                    trajectory=None, current_idx=None,
                    placingPhaseStart=None, liftingPhaseStart=None, liftingPhaseEnd=None,
                    damping=1e-10, max_angvel=max_angvel
                )
                # moveArm already set data.ctrl, so we just store it
                ctrl = data.ctrl[actuator_ids].copy()
            elif use_delta:
                # Delta models output 8D delta actions directly (no normalization)
                delta_action = action_normalized
                
                # Apply delta action
                current_qpos = data.qpos[:7].copy()
                
                # Scale delta for stability (reduced from 10% to 5% for smoother movement)
                delta_scale = 0.05  # Reduced from 0.1 to 0.05 for smoother movement
                target_qpos = current_qpos + delta_action[:7] * delta_scale
                
                # Enforce joint limits
                for i, jid in enumerate(dof_ids):
                    minr, maxr = get_joint_range_for_joint(i, model_mj=model_mj, dof_id=jid, use_default=True)
                    if minr < maxr and np.isfinite(minr) and np.isfinite(maxr):
                        target_qpos[i] = np.clip(target_qpos[i], minr, maxr)
                
                ctrl = target_qpos
                gripper_cmd = delta_action[7]  # Gripper is at index 7 for delta
                data.ctrl[actuator_ids] = ctrl
                
                # Compute and store desired end-effector pose for mocap (if mocap exists)
                if mocap_id is not None:
                    # Temporarily set qpos to target to compute forward kinematics
                    temp_qpos = data.qpos.copy()
                    data.qpos[dof_ids] = target_qpos
                    mujoco.mj_forward(model_mj, data)
                    # Read desired end-effector pose
                    desired_pos = data.site_xpos[site].copy()
                    site_mat = data.site_xmat[site].reshape(3, 3)
                    desired_quat = np.zeros(4)
                    mujoco.mju_mat2Quat(desired_quat, site_mat.flatten())
                    desired_quat = desired_quat / (np.linalg.norm(desired_quat) + 1e-8)
                    # Restore qpos
                    data.qpos[:] = temp_qpos
                    mujoco.mj_forward(model_mj, data)
                    # Store for trajectory
                    mocap_pos_step = desired_pos.copy()
                    mocap_quat_step = desired_quat.copy()
                    # Set mocap
                    data.mocap_pos[mocap_id] = desired_pos
                    data.mocap_quat[mocap_id] = desired_quat
                else:
                    mocap_pos_step = None
                    mocap_quat_step = None
            else:
                # Denormalize action from [0, 1] to actual joint positions (for direct or even_simpler models)
                if (use_direct or use_even_simpler) and action_scaler is not None:
                    action = denormalize_action(action_normalized, action_scaler)
                else:
                    action = action_normalized
                
                # Apply action with interpolation and joint limit enforcement (matching training pipeline)
                target_qpos = action[:7]
                current_qpos = data.qpos[dof_ids].copy()
                # Use smaller action_scale for smoother movement (reduced from default 0.1)
                smooth_action_scale = min(action_scale, 0.05)  # Cap at 5% per step for smoother motion
                ctrl = current_qpos + (target_qpos - current_qpos) * smooth_action_scale
                
                # Enforce joint limits to prevent invalid configurations
                for i, jid in enumerate(dof_ids):
                    minr, maxr = get_joint_range_for_joint(i, model_mj=model_mj, dof_id=jid, use_default=True)
                    if minr < maxr and np.isfinite(minr) and np.isfinite(maxr):
                        ctrl[i] = np.clip(ctrl[i], minr, maxr)
                
                gripper_cmd = action[7]  # Gripper is at index 7 for absolute actions
                data.ctrl[actuator_ids] = ctrl
                
                # Compute and store desired end-effector pose for mocap (if mocap exists)
                if mocap_id is not None:
                    # Temporarily set qpos to target to compute forward kinematics
                    temp_qpos = data.qpos.copy()
                    data.qpos[dof_ids] = target_qpos
                    mujoco.mj_forward(model_mj, data)
                    # Read desired end-effector pose
                    desired_pos = data.site_xpos[site].copy()
                    site_mat = data.site_xmat[site].reshape(3, 3)
                    desired_quat = np.zeros(4)
                    mujoco.mju_mat2Quat(desired_quat, site_mat.flatten())
                    desired_quat = desired_quat / (np.linalg.norm(desired_quat) + 1e-8)
                    # Restore qpos
                    data.qpos[:] = temp_qpos
                    mujoco.mj_forward(model_mj, data)
                    # Store for trajectory
                    mocap_pos_step = desired_pos.copy()
                    mocap_quat_step = desired_quat.copy()
                    # Set mocap
                    data.mocap_pos[mocap_id] = desired_pos
                    data.mocap_quat[mocap_id] = desired_quat
                else:
                    mocap_pos_step = None
                    mocap_quat_step = None
            
            # Gripper control mapping:
            # Dataset stores: 1.0 = closed, 0.0 = open
            # MuJoCo expects: 0 = closed, 255 = open
            actuator8_id = model_mj.actuator("actuator8").id
            if gripper_cmd > 0.5:  # Closed (1.0)
                data.ctrl[actuator8_id] = 0  # Closed
            else:  # Open (0.0)
                data.ctrl[actuator8_id] = 255  # Open
            
            # Store trajectory data (including mocap if it exists)
            if mocap_id is not None:
                trajectory_data.append((data.qpos.copy(), data.qvel.copy(), data.ctrl.copy(), 
                                       mocap_pos_step.copy(), mocap_quat_step.copy()))
            else:
                trajectory_data.append((data.qpos.copy(), data.qvel.copy(), data.ctrl.copy(), None, None))
            
            mujoco.mj_step(model_mj, data)
            step_count += 1
            
            # Check if task complete
            if step_count % 100 == 0:
                mujoco.mj_forward(model_mj, data)
                if use_simpler or use_even_simpler or use_direct or use_delta or use_taskspace or use_taskspace_absolutes:
                    # Use appropriate placement box position/size for each model type
                    if use_direct:
                        pb_pos = placement_box_pos_direct
                        pb_size = placement_box_size_direct
                    elif use_taskspace or use_taskspace_absolutes:
                        pb_pos = placement_box_pos_taskspace
                        pb_size = placement_box_size_taskspace
                    elif use_delta or use_even_simpler:
                        pb_pos = placement_box_pos_even_simpler
                        pb_size = placement_box_size_even_simpler
                    else:
                        pb_pos = placement_box_pos
                        pb_size = placement_box_size
                    success_check, block_pos_check, message = verifyPlacementOnGreen(model_mj, data, block_id, tolerance=0.02, placement_box_pos=pb_pos, placement_box_size=pb_size)
                else:
                    table_surface_z = 0.2
                    success_check, block_z, message = verifyPlacement(model_mj, data, block_id, table_surface_z=table_surface_z)
                if success_check:
                    print(f"  ✓ Success at step {step_count}! {message}")
                    successes += 1
                    success_achieved = True
        
        print(f"  Generated {len(trajectory_data)} steps")
        print("Starting visualization...")
        
        # Reset to initial state
        data.qpos[:] = initial_qpos
        data.qvel[:] = initial_qvel
        mujoco.mj_forward(model_mj, data)
        
        # Visualize the trajectory
        try:
            with mujoco.viewer.launch_passive(model_mj, data) as viewer:
                print("  Viewer opened (close to exit)")
                for step_idx, step_data in enumerate(trajectory_data):
                    if not viewer.is_running():
                        break
                    
                    # Unpack trajectory data (with or without mocap)
                    if len(step_data) == 5:
                        qpos_step, qvel_step, ctrl_step, mocap_pos_step, mocap_quat_step = step_data
                    else:
                        # Backward compatibility
                        qpos_step, qvel_step, ctrl_step = step_data[:3]
                        mocap_pos_step, mocap_quat_step = None, None
                    
                    # Set state
                    data.qpos[:] = qpos_step
                    data.qvel[:] = qvel_step
                    data.ctrl[:] = ctrl_step
                    
                    # Update mocap if it exists and we have stored values
                    if mocap_id is not None and mocap_pos_step is not None and mocap_quat_step is not None:
                        data.mocap_pos[mocap_id] = mocap_pos_step
                        data.mocap_quat[mocap_id] = mocap_quat_step
                    
                    mujoco.mj_forward(model_mj, data)
                    
                    viewer.sync()
                    time.sleep(delay)
                    
                    if step_idx % 1000 == 0:
                        print(f"  Visualizing step {step_idx}/{len(trajectory_data)}")
        except KeyboardInterrupt:
            print("\n  Interrupted by user")
        except Exception as e:
            print(f"  Viewer error: {e}")
        
        # Final check (use last state from trajectory)
        if trajectory_data:
            if len(trajectory_data[-1]) == 5:
                final_qpos, final_qvel, final_ctrl, _, _ = trajectory_data[-1]
            else:
                final_qpos, final_qvel, final_ctrl = trajectory_data[-1][:3]
            data.qpos[:] = final_qpos
            data.qvel[:] = final_qvel
            data.ctrl[:] = final_ctrl
            mujoco.mj_forward(model_mj, data)
            
            if use_simpler or use_even_simpler or use_direct or use_delta or use_taskspace or use_taskspace_absolutes:
                # Use appropriate placement box position/size for each model type
                if use_direct:
                    pb_pos = placement_box_pos_direct
                    pb_size = placement_box_size_direct
                elif use_taskspace or use_taskspace_absolutes:
                    pb_pos = placement_box_pos_taskspace
                    pb_size = placement_box_size_taskspace
                elif use_delta or use_even_simpler:
                    pb_pos = placement_box_pos_even_simpler
                    pb_size = placement_box_size_even_simpler
                else:
                    pb_pos = placement_box_pos
                    pb_size = placement_box_size
                success_final, block_pos_final, message = verifyPlacementOnGreen(model_mj, data, block_id, tolerance=0.02, placement_box_pos=pb_pos, placement_box_size=pb_size)
            else:
                table_surface_z = 0.2
                success_final, block_z, message = verifyPlacement(model_mj, data, block_id, table_surface_z=table_surface_z)
            print(f"Final: {message}")
    
    print(f"\n{'='*70}")
    print(f"Summary: {successes}/{num_trials} successful ({100*successes/num_trials:.1f}%)")
    print(f"{'='*70}")

def main():
    visualize_model(
        model_path=None,
        num_trials=5,
        action_scale=0.1,
        delay=0.0002,
        use_standardized=None,
        max_angvel=0.3,
        action_smoothing=None,
        pose_update_frequency=None
    )

if __name__ == "__main__":
    main()

