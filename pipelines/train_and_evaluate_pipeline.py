"""
Training and evaluation pipeline.
Trains models with increasing amounts of data until 50% success rate is achieved.
"""
import sys
import os
# Add parent directory to path so we can import utils when running directly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import glob
import json
import pickle
import re
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.model_selection import train_test_split
import mujoco
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from utils.load_dataset_csv import load_dataset_csv, get_state_action_pairs
from utils.path_utils import (
    get_dataset_path,
    get_variation_path,
    get_variation_from_base_filename,
    find_model_file,
    get_model_path
)
from trajectories.trajectoriesCircle import (
    sample_block_pos,
    sample_block_size,
    verifyPlacement,
    numSeconds,
    dt
)
from collect_dataset_csv import collectDataCSV

class MLP(nn.Module):
    """MLP matching training code structure"""
    def __init__(self, input_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(128, 64)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(64, output_size)

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.fc3(x)
        return x

def train_model(states, actions, trajectory_ids=None, epochs=150, batch_size=256, lr=0.001, 
                patience=15, min_delta=1e-6, pretrained_model_path=None, num_prev_trajectories=0,
                max_batches_per_epoch=None):
    """
    Train MLP model with early stopping based on validation loss.
    
    Args:
        states: Training states
        actions: Training actions
        epochs: Maximum number of epochs
        batch_size: Number of samples processed before updating weights.
                   - Larger batch_size: Fewer updates per epoch, more stable gradients, faster per epoch
                   - Smaller batch_size: More updates per epoch, noisier gradients, slower per epoch
                   - For larger datasets, can use larger batch sizes (512-1024)
        lr: Learning rate
        patience: Number of epochs to wait before early stopping (increased for larger datasets)
        min_delta: Minimum change to qualify as an improvement
    
    Returns:
        model: Trained model
    """
    # Split data
    train_indices = None
    if trajectory_ids is not None:
        # Split while preserving trajectory structure
        # train_test_split returns arrays in the order passed: states, actions, trajectory_ids, indices
        X_train, X_val, y_train, y_val, train_traj_ids, val_traj_ids, train_indices, val_indices = train_test_split(
            states, actions, trajectory_ids, np.arange(len(states)),
            test_size=0.2, random_state=42
        )
    else:
        X_train, X_val, y_train, y_val = train_test_split(
            states, actions, test_size=0.2, random_state=42
        )
        train_traj_ids = None
    
    # Auto-detect device (defaults to MPS > CUDA > CPU)
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train).to(device)
    y_train_tensor = torch.FloatTensor(y_train).to(device)
    X_val_tensor = torch.FloatTensor(X_val).to(device)
    y_val_tensor = torch.FloatTensor(y_val).to(device)
    
    # Create DataLoaders for batching
    # Optimize DataLoader settings for speed
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    
    # Use larger batch size for validation (no gradients, can handle more)
    val_batch_size = min(batch_size * 4, 32768)  # 4x larger or max 32768 for very large batches
    
    # Calculate sample weights based on inverse of trajectory percentage contribution
    # If doing transfer learning, also weight newer trajectories more heavily
    if train_traj_ids is not None and len(train_traj_ids) == len(X_train):
        # Count samples per trajectory
        unique_trajs, counts = np.unique(train_traj_ids, return_counts=True)
        total_samples = len(X_train)
        
        # Calculate percentage contribution per trajectory
        traj_percentages = counts / total_samples
        
        # Base weight by inverse of percentage (so smaller contributions are weighted more)
        traj_weights = 1.0 / (traj_percentages + 1e-8)  # Add small epsilon to avoid division by zero
        
        # If doing transfer learning, use 60% new / 40% random sampling strategy
        if pretrained_model_path and num_prev_trajectories > 0:
            # Identify old vs new trajectories
            old_traj_mask = unique_trajs < num_prev_trajectories
            new_traj_mask = unique_trajs >= num_prev_trajectories
            
            if np.any(new_traj_mask) and np.any(old_traj_mask):
                # Get new trajectory IDs
                new_traj_ids = unique_trajs[new_traj_mask]
                
                # Find which training samples belong to new trajectories
                new_sample_mask = np.isin(train_traj_ids, new_traj_ids)
                new_sample_indices = np.where(new_sample_mask)[0]
                all_sample_indices = np.arange(len(X_train))
                
                # Create sample weights: 60% for new trajectories, 40% for all (uniform)
                sample_weights = np.ones(len(X_train))
                
                # Weight new trajectory samples so they get 60% of sampling probability
                num_new_samples = len(new_sample_indices)
                num_all_samples = len(all_sample_indices)
                
                if num_new_samples > 0:
                    # For 60% new / 40% random:
                    # P(new) = 0.6 = (weight_new * num_new) / (weight_new * num_new + weight_all * num_all)
                    # P(all) = 0.4 = (weight_all * num_all) / (weight_new * num_new + weight_all * num_all)
                    # Solving: weight_new / weight_all = (0.6 * num_all) / (0.4 * num_new)
                    weight_ratio = (0.6 * num_all_samples) / (0.4 * num_new_samples)
                    
                    sample_weights[new_sample_indices] = weight_ratio
                    
                    num_old = np.sum(old_traj_mask)
                    num_new = np.sum(new_traj_mask)
                    old_samples = counts[old_traj_mask].sum()
                    new_samples = counts[new_traj_mask].sum()
                    
                    print(f"    Transfer learning: {num_old} old trajectories ({old_samples} samples), "
                          f"{num_new} new trajectories ({new_samples} samples)")
                    print(f"    Using 60% new / 40% random sampling strategy")
                
                # Convert to tensor (keep on CPU for sampler, use float32)
                sample_weights_tensor = torch.FloatTensor(sample_weights)
                
                # Create weighted sampler (sampler must be on CPU)
                sampler = WeightedRandomSampler(
                    weights=sample_weights_tensor,
                    num_samples=len(sample_weights_tensor),
                    replacement=True
                )
                
                train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler,
                                         num_workers=0, pin_memory=False)  # MPS doesn't support pin_memory or prefetch_factor with num_workers=0
            else:
                # No new trajectories, use standard inverse percentage weighting
                traj_weights = traj_weights / traj_weights.sum() * len(unique_trajs)
                sample_weights = np.zeros(len(X_train))
                for traj_id, weight in zip(unique_trajs, traj_weights):
                    mask = train_traj_ids == traj_id
                    sample_weights[mask] = weight
                sample_weights_tensor = torch.FloatTensor(sample_weights)  # Keep on CPU
                sampler = WeightedRandomSampler(
                    weights=sample_weights_tensor,
                    num_samples=len(sample_weights_tensor),
                    replacement=True
                )
            train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler,
                                     num_workers=0, pin_memory=False)  # MPS doesn't support pin_memory or prefetch_factor with num_workers=0
            print(f"    Using weighted sampling: {len(unique_trajs)} trajectories, weight range: [{traj_weights.min():.2f}, {traj_weights.max():.2f}]")
        else:
            # Not doing transfer learning, use standard inverse percentage weighting
            traj_weights = traj_weights / traj_weights.sum() * len(unique_trajs)
            sample_weights = np.zeros(len(X_train))
            for traj_id, weight in zip(unique_trajs, traj_weights):
                mask = train_traj_ids == traj_id
                sample_weights[mask] = weight
            sample_weights_tensor = torch.FloatTensor(sample_weights)  # Keep on CPU
            sampler = WeightedRandomSampler(
                weights=sample_weights_tensor,
                num_samples=len(sample_weights_tensor),
                replacement=True
            )
            train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler,
                                     num_workers=0, pin_memory=False)  # MPS doesn't support pin_memory or prefetch_factor with num_workers=0
            print(f"    Using weighted sampling: {len(unique_trajs)} trajectories, weight range: [{traj_weights.min():.2f}, {traj_weights.max():.2f}]")
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                 num_workers=0, pin_memory=False)  # MPS doesn't support pin_memory or prefetch_factor with num_workers=0
    
    val_loader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False, 
                           num_workers=0, pin_memory=False)  # MPS doesn't support pin_memory or prefetch_factor with num_workers=0
    
    # Create model
    model = MLP(input_size=states.shape[1], output_size=actions.shape[1])
    
    # Load pretrained model if available (transfer learning)
    if pretrained_model_path and os.path.exists(pretrained_model_path):
        try:
            print(f"    Loading pretrained weights from {os.path.basename(pretrained_model_path)}...")
            pretrained_state = torch.load(pretrained_model_path, map_location=device, weights_only=False)
            model.load_state_dict(pretrained_state, strict=False)
            print(f"    ✓ Loaded pretrained weights (fine-tuning)")
            # Use slightly lower learning rate for fine-tuning
            lr = lr * 0.5  # Half the learning rate when fine-tuning
            print(f"    Using reduced learning rate for fine-tuning: {lr}")
        except Exception as e:
            print(f"    ⚠ Could not load pretrained model: {e}")
            print(f"    Training from scratch instead...")
    
    model.to(device)
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Use mixed precision training for speed (MPS supports it in PyTorch 2.0+)
    use_amp = device.type in ['mps', 'cuda']
    if use_amp:
        try:
            if device.type == 'mps':
                scaler = torch.amp.GradScaler('mps')
            else:
                scaler = torch.cuda.amp.GradScaler()
        except:
            # Fallback if mixed precision not available
            use_amp = False
            scaler = None
    else:
        scaler = None
    
    # Early stopping variables
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    # Validate less frequently for speed (every N epochs)
    # For large datasets, validate less often since it's expensive
    val_frequency = 5 if len(states) > 1000000 else 3 if len(states) > 100000 else 1
    
    # Mini-batch SGD: process only a subset of batches per epoch for speed
    # This makes epochs much faster while still covering the full dataset over multiple epochs
    total_batches = len(train_loader)
    if total_batches > 200:  # If more than 1000 batches, use mini-batch SGD
        max_batches_per_epoch = 200  # Process 500 batches per epoch
        samples_per_epoch = max_batches_per_epoch * batch_size
        print(f"    Using mini-batch SGD: {max_batches_per_epoch} batches per epoch (out of {total_batches} total)")
        print(f"    Samples per epoch: {samples_per_epoch:,} ({100.0 * samples_per_epoch / len(X_train):.1f}% of training data)")
    else:
        max_batches_per_epoch = total_batches  # Process all batches for smaller datasets
        samples_per_epoch = total_batches * batch_size
        print(f"    Processing all {total_batches} batches per epoch")
        print(f"    Samples per epoch: {samples_per_epoch:,} (100% of training data)")
    
    # Training loop
    for epoch in range(epochs):
        # Training phase - process subset of batches (mini-batch SGD)
        model.train()
        train_loss = 0.0
        num_batches = 0
        
        # Limit to max_batches_per_epoch for speed
        batch_iter = iter(train_loader)
        batches_to_process = min(max_batches_per_epoch, total_batches)
        
        for batch_idx in range(batches_to_process):
            try:
                batch_X, batch_y = next(batch_iter)
            except StopIteration:
                # Restart iterator if we run out (shouldn't happen, but safe)
                batch_iter = iter(train_loader)
                batch_X, batch_y = next(batch_iter)
            
            optimizer.zero_grad()
            
            if use_amp:
                # Mixed precision training
                with torch.amp.autocast(device_type='mps' if device.type == 'mps' else 'cuda'):
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
            
            train_loss += loss.item()
            num_batches += 1
        
        train_loss /= num_batches if num_batches > 0 else 1
        
        # Validation phase (only every N epochs for speed)
        if (epoch + 1) % val_frequency == 0 or epoch == 0:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    if use_amp:
                        with torch.amp.autocast(device_type='mps' if device.type == 'mps' else 'cuda'):
                            outputs = model(batch_X)
                            loss = criterion(outputs, batch_y)
                    else:
                        outputs = model(batch_X)
                        loss = criterion(outputs, batch_y)
                    val_loss += loss.item()
            val_loss /= len(val_loader)
        else:
            # Use last validation loss if not validating this epoch
            val_loss = best_val_loss if best_val_loss != float('inf') else train_loss
        
        # Check for improvement (only when we actually validated)
        if (epoch + 1) % val_frequency == 0 or epoch == 0:
            if val_loss < best_val_loss - min_delta:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model state
                best_model_state = model.state_dict().copy()
            else:
                patience_counter += 1
        
        # Print progress (more frequent for larger datasets)
        print_freq = 20 if len(states) < 50000 else 10
        if (epoch + 1) % print_freq == 0 or (epoch + 1) % val_frequency == 0 or epoch == 0:
            val_str = f"Val Loss: {val_loss:.6f}" if (epoch + 1) % val_frequency == 0 or epoch == 0 else "(skipped)"
            print(f"  Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.6f}, "
                  f"{val_str}, Best Val: {best_val_loss:.6f}")
        
        # Early stopping (adjust patience based on validation frequency)
        effective_patience = patience * val_frequency  # Account for less frequent validation
        if patience_counter * val_frequency >= patience:
            print(f"  Early stopping at epoch {epoch+1} (no improvement for {patience} validation checks)")
            print(f"  Best validation loss: {best_val_loss:.6f}")
            break
    
    # Load best model state
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"  Loaded best model (val loss: {best_val_loss:.6f})")
    
    return model

def evaluate_model(model, num_objects=20, max_steps=None):
    """
    Evaluate model on unseen objects.
    
    Args:
        model: Trained model
        num_objects: Number of random objects to test
        max_steps: Maximum steps per trial. If None, calculates from expert trajectory length.
                   Expert trajectories are 10 seconds = 5000 steps at dt=0.002.
                   We use 3x that (15000 steps = 30 seconds) to account for slower policy execution.
        device: Device to run model on
    """
    # Calculate reasonable max_steps based on expert trajectory length
    if max_steps is None:
        expert_trajectory_steps = int(numSeconds / dt)  # ~5000 steps for 10 seconds
        # Expert execution can take up to len(trajectory) * 100 steps (from collect_dataset_csv.py)
        # Use 4x expert trajectory time to account for slower/less efficient policy execution
        # This ensures we give the policy enough time even if it's slower than expert
        max_steps = expert_trajectory_steps * 4  # ~20000 steps = 40 seconds
        print(f"  Using max_steps={max_steps} ({max_steps * dt:.1f}s) based on expert trajectory")
        print(f"    Expert: {expert_trajectory_steps} steps ({numSeconds}s), allowing 4x for policy execution")
    model.eval()
    successes = 0
    results = []
    
    for obj_idx in range(num_objects):
        # Setup simulation
        model_mj = mujoco.MjModel.from_xml_path("franka_emika_panda/pickAndPlace.xml")
        data = mujoco.MjData(model_mj)
        
        block_id = model_mj.body('target').id
        site = model_mj.site('palm_contact_edge_vis').id
        blockGeom_id = model_mj.geom('box').id
        joint_names = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"][:7]
        dof_ids = np.array([model_mj.joint(name).id for name in joint_names])
        actuator_names = ['actuator1', 'actuator2', 'actuator3', 'actuator4', 'actuator5', 'actuator6', 'actuator7'][:7]
        actuator_ids = [model_mj.actuator(name).id for name in actuator_names]
        
        block_jnt_id = model_mj.body(block_id).jntadr[0]
        
        # Sample random block
        block_size = sample_block_size()
        model_mj.geom(blockGeom_id).size[:] = block_size
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
        quat_mj = np.zeros(4)
        mujoco.mju_mat2Quat(quat_mj, Q.flatten())
        
        data.qpos[block_jnt_id:block_jnt_id+3] = targetInitialPos
        data.qpos[block_jnt_id+3:block_jnt_id+7] = quat_mj
        mujoco.mj_forward(model_mj, data)
        
        # Rollout policy
        success = False
        for step in range(max_steps):
            # Extract state
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
            
            # Construct state (31D: arm 21D + block 10D) - matching training format
            # Arm: [qpos(7), qvel(7), ee_pos(3), ee_quat(4)] = 21D
            # Block: [pos(3), quat(4), size(3)] = 10D
            # Note: Timestep is NOT included in state (only kinematic information)
            arm_state = np.concatenate([qpos, qvel, ee_pos, ee_quat])
            block_state = np.concatenate([block_pos, block_quat, block_size])
            state = np.concatenate([arm_state, block_state])
            
            # Ensure state is correct size (31D, no timestep)
            assert len(state) == 31, f"State size mismatch: {len(state)} != 31"
            
            # Get action from policy
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                action = model(state_tensor).squeeze(0).cpu().numpy()
            
            # Apply action (scale down for stability)
            action_scale = 0.1
            target_qpos = action[:7]
            current_qpos = data.qpos[dof_ids].copy()
            ctrl = current_qpos + (target_qpos - current_qpos) * action_scale
            gripper_cmd = action[7]
            
            data.ctrl[actuator_ids] = ctrl
            # Gripper: 1.0 = closed (0), 0.0 = open (255)
            # In moveArm: shutFingers=True sets actuator8 to 0 (closed), False sets to 255 (open)
            data.ctrl[model_mj.actuator("actuator8").id] = 0 if gripper_cmd > 0.5 else 255
            
            mujoco.mj_step(model_mj, data)
            
            # Check success
            if step % 100 == 0:
                mujoco.mj_forward(model_mj, data)
                table_surface_z = 0.2
                success_check, block_z, _ = verifyPlacement(model_mj, data, block_id, table_surface_z=table_surface_z)
                if success_check:
                    success = True
                    break
        
        # Final check
        mujoco.mj_forward(model_mj, data)
        table_surface_z = 0.2
        success_final, block_z, _ = verifyPlacement(model_mj, data, block_id, table_surface_z=table_surface_z)
        
        if success_final:
            successes += 1
        
        results.append({
            'object_idx': obj_idx,
            'success': success_final,
            'block_size': block_size.tolist(),
            'block_final_z': float(block_z)
        })
    
    success_rate = successes / num_objects
    return success_rate, results

def main():
    print("=" * 70)
    print("Training and Evaluation Pipeline")
    print("=" * 70)
    
    # Try to load preprocessed dataset first (much faster)
    variation = get_variation_from_base_filename('dataset')
    dataset_dir = get_dataset_path(variation)
    os.makedirs(dataset_dir, exist_ok=True)
    preprocessed_file = os.path.join(dataset_dir, 'dataset_preprocessed.pkl')
    print("\nLoading dataset...")
    
    if os.path.exists(preprocessed_file):
        print(f"  Found preprocessed dataset: {preprocessed_file}")
        print(f"  Loading from preprocessed file (fast)...")
        with open(preprocessed_file, 'rb') as f:
            preprocessed_data = pickle.load(f)
        
        states = preprocessed_data['states']
        actions = preprocessed_data['actions']
        unique_traj_ids = preprocessed_data['unique_trajectory_ids']
        trajectory_id_to_indices = preprocessed_data['trajectory_id_to_indices']
        num_trajectories_total = preprocessed_data['num_trajectories']
        
        # Reconstruct trajectory_ids array for compatibility
        trajectory_ids_full = preprocessed_data['trajectory_ids']
        
        print(f"  ✓ Loaded {len(states)} state-action pairs from {num_trajectories_total} unique trajectories")
        print(f"    Trajectory ID range: {unique_traj_ids.min()} to {unique_traj_ids.max()}")
    else:
        print(f"  Preprocessed file not found: {preprocessed_file}")
        print(f"  Loading from CSV files (this may take a while)...")
        print(f"  Tip: Run 'python -m utils.preprocess_dataset' to create preprocessed file for faster loading")
        
        data = load_dataset_csv(filename=None, base_filename='dataset', max_trajectories=None)
        states, actions = get_state_action_pairs(data)
        
        # Get unique trajectory IDs (this is the correct count)
        unique_traj_ids = np.unique(data['trajectory_ids'])
        num_trajectories_total = len(unique_traj_ids)
        
        # Create mapping from trajectory_id to indices
        trajectory_id_to_indices = {}
        trajectory_ids_full = data['trajectory_ids']
        for traj_id in unique_traj_ids:
            trajectory_id_to_indices[traj_id] = np.where(trajectory_ids_full == traj_id)[0]
        
        print(f"  ✓ Loaded {len(states)} state-action pairs from {num_trajectories_total} unique trajectories")
        print(f"    Trajectory ID range: {unique_traj_ids.min()} to {unique_traj_ids.max()}")
    
    # Training configurations: start with 50, then increase
    data_sizes = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500]
    data_sizes.extend([600, 700, 800, 900, 1000])
    data_sizes.extend([1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000])
    #data_sizes.extend([2200, 2400, 2600, 2800, 3000])
    
    # Check if we need to collect more data for any training size
    max_needed = max(data_sizes)
    if num_trajectories_total < max_needed:
        needed = max_needed - num_trajectories_total
        target_total = max_needed  # Total number of successes we want
        print(f"\n{'='*70}")
        print(f"Need {needed} more successful trajectories to reach {max_needed} total")
        print(f"Currently have: {num_trajectories_total} trajectories")
        print(f"Collecting until we have {target_total} total successful trajectories...")
        print(f"{'='*70}\n")
        
        # Collect additional trajectories
        # Note: collectDataCSV expects total number of successes (not additional)
        # It will load existing data and collect more until reaching the target
        # NEVER deletes existing files - only appends or creates new ones
        collectDataCSV(target_successes=target_total, base_filename='dataset', trajectories_per_file=200, save_frequency=10)
        
        # Reload dataset after collection (from all files)
        print("\nReloading dataset after collection...")
        print("  Note: Preprocessed file will be out of date. Re-run 'python -m utils.preprocess_dataset' to update it.")
        data = load_dataset_csv(filename=None, base_filename='dataset', max_trajectories=None)
        states, actions = get_state_action_pairs(data)
        unique_traj_ids = np.unique(data['trajectory_ids'])
        num_trajectories_total = len(unique_traj_ids)
        trajectory_ids_full = data['trajectory_ids']
        
        # Update trajectory_id_to_indices
        trajectory_id_to_indices = {}
        for traj_id in unique_traj_ids:
            trajectory_id_to_indices[traj_id] = np.where(trajectory_ids_full == traj_id)[0]
        
        print(f"Now have {num_trajectories_total} total trajectories")
    
    # Filter to only sizes we actually have (or will have after collection)
    data_sizes = [s for s in data_sizes if s <= num_trajectories_total]
    
    # Check for GPU availability (prioritize MPS for Apple Silicon, then CUDA for NVIDIA)
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
        print(f"✓ Apple Silicon GPU (MPS) available")
    elif torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"✓ GPU available: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA version: {torch.version.cuda}")
        print(f"  GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        device = torch.device('cpu')
        print("⚠ GPU not available, using CPU")
    print(f"Using device: {device}")
    
    # Load existing results if available
    variation_dir = get_variation_path('default')
    os.makedirs(variation_dir, exist_ok=True)
    results_file = os.path.join(variation_dir, 'training_results.json')
    results_csv = os.path.join(variation_dir, 'training_results.csv')
    
    results = []
    if os.path.exists(results_file):
        print("\nLoading existing results...")
        with open(results_file, 'r') as f:
            results = json.load(f)
        print(f"  Found {len(results)} previously completed training runs")
        # Show what's already done
        completed_sizes = [r['num_trajectories'] for r in results]
        print(f"  Completed trajectory sizes: {sorted(completed_sizes)}")
    else:
        print("\nNo existing results found, starting from scratch")
    
    target_success_rate = 0.5
    
    # Filter out already completed sizes
    completed_sizes = set([r['num_trajectories'] for r in results])
    remaining_sizes = [s for s in data_sizes if s not in completed_sizes]
    
    if not remaining_sizes:
        print("\n" + "="*70)
        print("All trajectory sizes have already been trained!")
        print("="*70)
        # Still print summary and create plot
    else:
        print(f"\nResuming from: {remaining_sizes[0]} trajectories")
        print(f"  Remaining sizes to train: {remaining_sizes}")
    
    for num_traj in remaining_sizes:
        print(f"\n{'='*70}")
        print(f"Training with {num_traj} trajectories...")
        print(f"{'='*70}")
        
        # Get subset of trajectories
        traj_ids_subset = unique_traj_ids[:num_traj]
        mask = np.isin(trajectory_ids_full, traj_ids_subset)
        states_subset = states[mask]
        actions_subset = actions[mask]
        
        print(f"  Training data: {len(states_subset)} state-action pairs")
        
        # Check if model already exists (from previous run that was interrupted)
        existing_models = find_model_file(f"model_{num_traj}traj_*.pth", base_name='default')
        # Also check legacy location for backward compatibility
        if not existing_models:
        existing_models = glob.glob(f"model_{num_traj}traj_*.pth")
        model_was_loaded = False
        old_model_path = None
        if existing_models:
            old_model_path = existing_models[0]
            print(f"  Found existing model: {old_model_path}")
            print("  Loading existing model for evaluation (skipping training)...")
            model = MLP(input_size=states.shape[1], output_size=actions.shape[1])
            model.load_state_dict(torch.load(old_model_path, map_location=device, weights_only=False))
            model.to(device)
            model.eval()
            model_was_loaded = True
            
            # Try to extract success rate from filename or use 0
            match = re.search(r'(\d+)success', existing_models[0])
            if match:
                existing_success = int(match.group(1)) / 100.0
                print(f"  Model had {existing_success:.1%} success rate (from filename)")
        else:
            # Train model
            print("  Training model...")
            print(f"    Device: {device}")
            if device.type == 'mps':
                print(f"    Using Apple Silicon GPU (MPS)")
            elif device.type == 'cuda':
                print(f"    GPU: {torch.cuda.get_device_name(0)}")
                print(f"    GPU Memory: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB allocated")
            # Adjust training parameters based on dataset size
            # Larger datasets may need more epochs or different batch sizes
            num_samples = len(states_subset)
            
            # Adaptive batch size: larger datasets can handle much larger batches
            # This reduces the number of batches per epoch significantly
            # For MPS, we can use even larger batches to maximize GPU utilization
            if device.type == 'mps':
                # MPS can handle larger batches efficiently - maximize GPU utilization
                if num_samples > 1000000:  # > 1M samples
                    batch_size = 16384  # Very large batches for maximum GPU utilization
                elif num_samples > 500000:  # > 500K samples
                    batch_size = 8192  # Larger batches for better GPU utilization
                elif num_samples > 100000:  # > 100K samples
                    batch_size = 4096
                elif num_samples > 50000:
                    batch_size = 2048
                elif num_samples > 20000:
                    batch_size = 1024
                else:
                    batch_size = 512
            else:
                # For CUDA/CPU, use more conservative batch sizes
                if num_samples > 1000000:  # > 1M samples
                    batch_size = 4096
                elif num_samples > 500000:  # > 500K samples
                    batch_size = 2048
                elif num_samples > 100000:  # > 100K samples
                    batch_size = 1024
                elif num_samples > 50000:
                    batch_size = 512
                elif num_samples > 20000:
                    batch_size = 384
                else:
                    batch_size = 256
            
            # More epochs for larger datasets (they have more to learn)
            # But use early stopping to prevent overfitting
            # Reduced from 150 to 100 for speed (early stopping will handle it)
            epochs = 100
            
            print(f"    Training config: {epochs} epochs, batch_size={batch_size}, samples={num_samples}")
            
            # Try to load previous model for transfer learning
            prev_model_path = None
            num_prev_trajectories = 0
            # Find the largest completed size that's smaller than current
            prev_sizes = [r['num_trajectories'] for r in results if r['num_trajectories'] < num_traj]
            if prev_sizes:
                prev_size = max(prev_sizes)  # Use the largest previous size
                # Look for previous model
                from utils.path_utils import find_model_file
                prev_models = find_model_file(f"model_{prev_size}traj_*.pth", base_name='default')
                # Also check legacy location
                if not prev_models:
                prev_models = glob.glob(f"model_{prev_size}traj_*.pth")
                if prev_models:
                    prev_model_path = prev_models[0]  # Use first match
                    num_prev_trajectories = prev_size
                    print(f"    Found previous model: {prev_model_path}")
                    print(f"    Loading and fine-tuning from {prev_size} trajectories...")
            
            # Get trajectory IDs for the subset
            traj_ids_subset = trajectory_ids_full[mask]
            
            # Train model (with transfer learning if previous model exists)
            model = train_model(
                states_subset, actions_subset, trajectory_ids=traj_ids_subset,
                epochs=epochs, batch_size=batch_size, lr=0.001,
                patience=15,  # More patience for larger datasets
                pretrained_model_path=prev_model_path,  # Pass previous model for transfer learning
                num_prev_trajectories=num_prev_trajectories  # Pass number of previous trajectories for weighting
            )
        # Verify model is on GPU after training
        if device.type in ['mps', 'cuda']:
            print(f"    Model on device: {next(model.parameters()).device}")
            if device.type == 'cuda':
                print(f"    GPU Memory after training: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
        
        # Evaluate model
        print(f"  Evaluating on 20 unseen objects...")
        success_rate, eval_results = evaluate_model(model, num_objects=20)
        
        print(f"\n  Results: {success_rate:.1%} success rate ({success_rate*20:.0f}/20)")
        
        # Save model with descriptive name (or update if we loaded an existing one)
        model_dir = get_model_path('default')
        os.makedirs(model_dir, exist_ok=True)
        model_filename = os.path.join(model_dir, f"model_{num_traj}traj_{int(success_rate*100)}success.pth")
        if model_was_loaded and old_model_path and old_model_path != model_filename:
            # Remove old model file if name changed
            if os.path.exists(old_model_path):
                os.remove(old_model_path)
                print(f"  Removed old model file: {old_model_path}")
        torch.save(model.state_dict(), model_filename)
        print(f"  Saved model to {model_filename}")
        
        # Store results
        results.append({
            'num_trajectories': num_traj,
            'num_state_action_pairs': len(states_subset),
            'success_rate': float(success_rate),
            'num_successes': int(success_rate * 20),
            'num_evaluations': 20,
            'model_filename': model_filename
        })
        
        # Save results incrementally
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Check if we've reached target
        if success_rate >= target_success_rate:
            print(f"\n{'='*70}")
            print(f"✓ Target success rate ({target_success_rate:.0%}) achieved!")
            print(f"  Required {num_traj} trajectories")
            print(f"{'='*70}")
            break
    
    # Print summary
    print(f"\n{'='*70}")
    print("Training Pipeline Summary")
    print(f"{'='*70}")
    print(f"{'Trajectories':<15} {'Success Rate':<15} {'Model File'}")
    print("-" * 70)
    for r in results:
        print(f"{r['num_trajectories']:<15} {r['success_rate']:<15.1%} {r['model_filename']}")
    
    # Save final results as CSV too
    df_results = pd.DataFrame(results)
    df_results.to_csv(results_csv, index=False)
    print(f"\n✓ Results saved to {results_file} and {results_csv}")
    
    # Create plot
    try:
        plt.figure(figsize=(10, 6))
        plt.plot(df_results['num_trajectories'], df_results['success_rate'], 'o-', linewidth=2, markersize=8)
        plt.axhline(y=target_success_rate, color='r', linestyle='--', label=f'{target_success_rate:.0%} Target')
        plt.xlabel('Number of Training Trajectories')
        plt.ylabel('Success Rate on Unseen Objects')
        plt.title('Effect of Data Diversity on Policy Performance')
        plt.grid(True)
        plt.ylim([0, 1])
        plt.legend()
        plt.savefig('data_diversity_effect.png', dpi=150)
        print("✓ Saved plot to data_diversity_effect.png")
    except ImportError:
        print("  (matplotlib not available, skipping plot)")

if __name__ == "__main__":
    main()

