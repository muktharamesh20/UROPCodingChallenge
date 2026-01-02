"""
Utility functions for visualization scripts (visualize_model.py and visualize_trajectories.py).
Provides shared logic for determining XML paths, action dimensions, and environment settings.

Used AI to generate the code.
"""
import os

def detect_environment_from_filename(filename_or_path):
    """
    Detect environment settings (XML path, action dimension, etc.) from a filename or path.
    
    Args:
        filename_or_path: Model filename, dataset filename, or path containing variation name
    
    Returns:
        dict with keys:
            - xml_path: Path to MuJoCo XML file
            - action_dim: Action dimension (7 for taskspace, 8 for joint space)
            - gripper_idx: Index of gripper in action array (6 for 7D, 7 for 8D)
            - joint_action_dim: Number of joint positions in action (0 for taskspace, 7 for joint space)
            - use_taskspace_absolutes: Whether this is a taskspace_absolutes variation
            - use_taskspace: Whether this is a taskspace (delta) variation
            - use_direct: Whether this is a direct variation
            - use_delta: Whether this is a delta variation
            - use_even_simpler: Whether this is an even_simpler variation
            - use_simpler: Whether this is a simpler variation
            - use_random_start: Whether this uses random starting positions
    """
    # Get basename and convert to lowercase for matching
    if isinstance(filename_or_path, str):
        basename = os.path.basename(filename_or_path).lower()
    else:
        basename = str(filename_or_path).lower()
    
    # Detect variation flags (order matters - check more specific first)
    use_taskspace_absolutes = "taskspace_absolutes" in basename
    use_taskspace = "taskspace" in basename and not use_taskspace_absolutes
    use_direct = "direct" in basename and not use_taskspace and not use_taskspace_absolutes
    use_delta = "delta" in basename and not use_direct and not use_taskspace and not use_taskspace_absolutes
    use_even_simpler = "even_simpler" in basename and not use_direct and not use_delta and not use_taskspace and not use_taskspace_absolutes
    use_simpler = "simpler" in basename and not use_even_simpler and not use_direct and not use_delta and not use_taskspace and not use_taskspace_absolutes
    use_random_start = "random_start" in basename
    
    # Determine XML path
    if use_direct:
        xml_path = "franka_emika_panda/pickAndPlaceDirect.xml"
    elif use_simpler or use_even_simpler or use_delta or use_taskspace or use_taskspace_absolutes:
        xml_path = "franka_emika_panda/pickAndPlaceSimpler.xml"
    else:
        xml_path = "franka_emika_panda/pickAndPlace.xml"
    
    # Determine action dimension
    # Taskspace datasets: 7D (ee_pos(3), ee_quat(3), gripper(1))
    # Joint space datasets: 8D (joint_pos(7), gripper(1))
    if use_taskspace or use_taskspace_absolutes:
        action_dim = 7
        gripper_idx = 6
        joint_action_dim = 0  # No joint positions in taskspace actions
    else:
        action_dim = 8
        gripper_idx = 7
        joint_action_dim = 7  # 7 joint positions
    
    return {
        'xml_path': xml_path,
        'action_dim': action_dim,
        'gripper_idx': gripper_idx,
        'joint_action_dim': joint_action_dim,
        'use_taskspace_absolutes': use_taskspace_absolutes,
        'use_taskspace': use_taskspace,
        'use_direct': use_direct,
        'use_delta': use_delta,
        'use_even_simpler': use_even_simpler,
        'use_simpler': use_simpler,
        'use_random_start': use_random_start
    }

def detect_environment_from_action_dim(action_dim):
    """
    Detect environment settings from actual action dimension in loaded data.
    Useful when filename doesn't clearly indicate the variation.
    
    Args:
        action_dim: Actual action dimension from loaded data (7 or 8)
    
    Returns:
        dict with keys: action_dim, gripper_idx, joint_action_dim
    """
    if action_dim == 7:
        # Taskspace: [ee_pos(3), ee_quat(3), gripper(1)]
        return {
            'action_dim': 7,
            'gripper_idx': 6,
            'joint_action_dim': 0
        }
    else:
        # Joint space: [joint_pos(7), gripper(1)]
        return {
            'action_dim': 8,
            'gripper_idx': 7,
            'joint_action_dim': 7
        }

