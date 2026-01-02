"""
Training and evaluation pipeline for the simpler pick-and-place task.
Uses pickAndPlaceSimpler.xml and trajectoriesSimpler.py.
Incrementally collects data and trains models, fine-tuning from previous models.
Success is defined as block placed on the green placement box.
"""
import sys
import os
# Add parent directory to path so we can import utils when running directly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import glob
import json
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.model_selection import train_test_split
import mujoco
from scipy.spatial.transform import Rotation as R
from utils.load_dataset_csv import load_dataset_csv, get_state_action_pairs
from utils.training_utils import (
    verify_placement_on_green as verifyPlacementOnGreen,
    extract_arm_state, extract_block_state, extract_action,
    save_trajectory_rows_to_csv
)
from utils.path_utils import (
    get_dataset_path,
    get_variation_path,
    get_variation_from_base_filename,
    find_dataset_files
)
from trajectories.trajectoriesSimpler import (
    sample_block_pos,
    sample_block_size,
    sample_robot_start_position,
    createTrajectory,
    moveArm,
    dt,
    numSeconds,
    placement_box_pos,
    placement_box_size
)
from collect_dataset_csv import collectDataCSV

class MLP(nn.Module):
    """MLP for simpler pick-and-place task"""
    def __init__(self, input_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(256, 256)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(256, 64)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(64, output_size)

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.relu3(self.fc3(x))
        x = self.fc4(x)
        return x

def train_model(states, actions, trajectory_ids=None, steps=None, epochs=150, batch_size=256, lr=0.001, 
                patience=15, min_delta=1e-6, pretrained_model_path=None, num_prev_trajectories=0,
                max_batches_per_epoch=None, weight_early_steps=False, early_step_threshold=4000):
    """
    Train MLP model with early stopping based on validation loss.
    """
    # Split data
    train_indices = None
    train_steps = None
    if trajectory_ids is not None:
        # Split while preserving trajectory structure
        if steps is not None:
            # Include steps in the split
            X_train, X_val, y_train, y_val, train_traj_ids, val_traj_ids, train_indices, val_indices, train_steps, val_steps = train_test_split(
                states, actions, trajectory_ids, np.arange(len(states)), steps,
                test_size=0.2, random_state=42
            )
        else:
            # train_test_split returns arrays in the order passed: states, actions, trajectory_ids, indices
            X_train, X_val, y_train, y_val, train_traj_ids, val_traj_ids, train_indices, val_indices = train_test_split(
                states, actions, trajectory_ids, np.arange(len(states)),
                test_size=0.2, random_state=42
            )
    else:
        if steps is not None:
            X_train, X_val, y_train, y_val, train_steps, val_steps = train_test_split(
                states, actions, steps, test_size=0.2, random_state=42
            )
        else:
            X_train, X_val, y_train, y_val = train_test_split(
                states, actions, test_size=0.2, random_state=42
            )
        train_traj_ids = None
    
    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train).to(device)
    y_train_tensor = torch.FloatTensor(y_train).to(device)
    X_val_tensor = torch.FloatTensor(X_val).to(device)
    y_val_tensor = torch.FloatTensor(y_val).to(device)
    
    # Create DataLoaders for batching
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    
    # Use larger batch size for validation (no gradients, can handle more)
    val_batch_size = min(batch_size * 4, 32768)  # 4x larger or max 32768 for very large batches
    
    # Calculate sample weights based on inverse of trajectory percentage contribution
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
                    
                    # If fine-tuning and steps available, weight early steps more heavily
                    if weight_early_steps and train_steps is not None and pretrained_model_path:
                        early_step_mask = train_steps < early_step_threshold
                        early_step_indices = np.where(early_step_mask)[0]
                        if len(early_step_indices) > 0:
                            # Double the weight for early steps (within new trajectory samples)
                            early_new_mask = np.isin(early_step_indices, new_sample_indices)
                            early_new_indices = early_step_indices[early_new_mask]
                            if len(early_new_indices) > 0:
                                sample_weights[early_new_indices] *= 2.0  # Double weight for early steps in new trajectories
                
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
                
                # If fine-tuning and steps available, weight early steps more heavily
                if weight_early_steps and train_steps is not None and pretrained_model_path:
                    early_step_mask = train_steps < early_step_threshold
                    early_step_indices = np.where(early_step_mask)[0]
                    if len(early_step_indices) > 0:
                        # Double the weight for early steps
                        sample_weights[early_step_indices] *= 2.0
                
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
            
            # If fine-tuning and steps available, weight early steps more heavily
            if weight_early_steps and train_steps is not None and pretrained_model_path:
                early_step_mask = train_steps < early_step_threshold
                early_step_indices = np.where(early_step_mask)[0]
                if len(early_step_indices) > 0:
                    # Double the weight for early steps
                    sample_weights[early_step_indices] *= 2.0
            
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
            pretrained_state = torch.load(pretrained_model_path, map_location=device, weights_only=False)
            
            # Check if old model has different architecture (fc1: 128 vs new: 256)
            old_fc1_size = pretrained_state.get('fc1.weight', torch.empty(0)).shape[0] if 'fc1.weight' in pretrained_state else None
            
            if old_fc1_size == 128:
                # Old architecture: Input → 128 → 64 → Output
                # New architecture: Input → 256 → 256 → 128 → 64 → Output
                # Map compatible layers and initialize new ones
                new_state = model.state_dict()
                
                # fc1: Old (31→128) to New (31→256) - use projection or pad
                if 'fc1.weight' in pretrained_state and 'fc1.weight' in new_state:
                    old_fc1_w = pretrained_state['fc1.weight']  # [128, 31]
                    old_fc1_b = pretrained_state['fc1.bias']    # [128]
                    new_fc1_w = new_state['fc1.weight']  # [256, 31]
                    new_fc1_b = new_state['fc1.bias']    # [256]
                    # Copy first 128 neurons, leave rest randomly initialized
                    new_fc1_w[:128, :] = old_fc1_w
                    new_fc1_b[:128] = old_fc1_b
                    new_state['fc1.weight'] = new_fc1_w
                    new_state['fc1.bias'] = new_fc1_b
                
                # fc2: Old (128→64) to New (256→256) - can't directly map, skip
                # fc3: Old (64→8) to New (256→128) - can't directly map, skip
                # But we can map old fc3 to new fc5 (both are 64→8)
                if 'fc3.weight' in pretrained_state and 'fc5.weight' in new_state:
                    # Old fc3 is the output layer (64→8), new fc5 is also (64→8)
                    new_state['fc5.weight'] = pretrained_state['fc3.weight']
                    new_state['fc5.bias'] = pretrained_state['fc3.bias']
                
                # Load the partially initialized state
                model.load_state_dict(new_state)
                print(f"    ✓ Loaded pretrained weights (architecture adapted: old 128→64→8, new 256→256→128→64→8)")
            else:
                # Same architecture, load directly
                model.load_state_dict(pretrained_state, strict=False)
                print(f"    ✓ Loaded pretrained weights (same architecture)")
            
            # Use slightly lower learning rate for fine-tuning
            lr = lr * 0.5  # Half the learning rate when fine-tuning
        except Exception as e:
            print(f"    ⚠ Could not load pretrained model: {e}")
            print(f"    Training from scratch instead...")
    
    print(f"  Moving model to {device}...")
    model.to(device)
    
    # Loss and optimizer
    print(f"  Setting up optimizer (lr={lr})...")
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
    
    # Calculate total batches for logging
    if hasattr(train_loader.sampler, 'num_samples'):
        total_samples = train_loader.sampler.num_samples
        total_batches = (total_samples + batch_size - 1) // batch_size  # Ceiling division
    else:
        total_batches = len(train_loader)
    
    # Training loop - process all batches each epoch (full pass)
    print(f"  Starting training: {epochs} epochs, {total_batches} total batches per epoch")
    print(f"  Validation every {val_frequency} epoch(s)")
    for epoch in range(epochs):
        # Training phase - process all batches (full pass)
        model.train()
        train_loss = 0.0
        num_batches = 0
        
        # Process all batches in the epoch
        for batch_X, batch_y in train_loader:
            
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
        
        # Print progress every epoch
        if (epoch + 1) % 1 == 0:  # Print every epoch
            print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.6f}", end='')
        
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
            if (epoch + 1) % 1 == 0:  # Print every epoch
                print(f", Val Loss: {val_loss:.6f}")
        else:
            # Use last validation loss if not validating this epoch
            val_loss = best_val_loss if best_val_loss != float('inf') else train_loss
            if (epoch + 1) % 1 == 0:  # Print every epoch
                print()  # New line for epochs without validation
        
        # Check for improvement (only when we actually validated)
        if (epoch + 1) % val_frequency == 0 or epoch == 0:
            if val_loss < best_val_loss - min_delta:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model state
                best_model_state = model.state_dict().copy()
            else:
                patience_counter += 1
        
        # Early stopping (adjust patience based on validation frequency)
        effective_patience = patience * val_frequency  # Account for less frequent validation
        if patience_counter * val_frequency >= patience:
            break
    
    # Load best model state
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model

# verifyPlacementOnGreen is now imported from training_utils

def evaluate_model(model, num_objects=20, max_steps=None):
    """
    Evaluate model on unseen objects with random starting positions.
    Success is defined as block placed on the green placement box.
    
    Args:
        model: Trained model
        num_objects: Number of random objects to test
        max_steps: Maximum steps per trial. If None, calculates from expert trajectory length.
        device: Device to run model on
    """
    # Calculate reasonable max_steps based on expert trajectory length
    if max_steps is None:
        expert_trajectory_steps = int(numSeconds / dt)  # ~20000 steps for 40 seconds
        # For evaluation, use a reasonable multiplier (10x expert time) with a cap
        # Model should succeed or fail much sooner than data collection
        max_steps = min(expert_trajectory_steps * 10, 200000)  # 10x expert time, cap at 200k steps (~6.7 minutes per trial)
    model.eval()
    successes = 0
    results = []
    
    for obj_idx in range(num_objects):
        # Setup simulation
        model_mj = mujoco.MjModel.from_xml_path("franka_emika_panda/pickAndPlaceSimpler.xml")
        data = mujoco.MjData(model_mj)
        
        block_id = model_mj.body('target').id
        site = model_mj.site('palm_contact_edge_vis').id
        blockGeom_id = model_mj.geom('box').id
        joint_names = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"][:7]
        dof_ids = np.array([model_mj.joint(name).id for name in joint_names])
        actuator_names = ['actuator1', 'actuator2', 'actuator3', 'actuator4', 'actuator5', 'actuator6', 'actuator7'][:7]
        actuator_ids = [model_mj.actuator(name).id for name in actuator_names]
        
        block_jnt_id = model_mj.body(block_id).jntadr[0]
        
        # Randomize robot starting position
        qpos_start = sample_robot_start_position(model_mj, dof_ids)
        data.qpos[:7] = qpos_start[:7]
        mujoco.mj_forward(model_mj, data)  # Update forward kinematics
        
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
            data.ctrl[model_mj.actuator("actuator8").id] = 0 if gripper_cmd > 0.5 else 255
            
            mujoco.mj_step(model_mj, data)
            
            # Check success (on green box)
            if step % 100 == 0:
                mujoco.mj_forward(model_mj, data)
                success_check, block_pos_check, _ = verifyPlacementOnGreen(model_mj, data, block_id, placement_box_pos, placement_box_size, tolerance=0.02)
                if success_check:
                    success = True
                    break
        
        # Let block settle before final check
        settling_steps = int(1.0 / dt)  # 1 second of settling time
        for _ in range(settling_steps):
            mujoco.mj_step(model_mj, data)
        
        # Final check
        mujoco.mj_forward(model_mj, data)
        success_final, block_pos_final, message = verifyPlacementOnGreen(model_mj, data, block_id, placement_box_pos, placement_box_size, tolerance=0.02)
        
        if success_final:
            successes += 1
        
        results.append({
            'object_idx': obj_idx,
            'success': success_final,
            'block_size': block_size.tolist(),
            'block_final_pos': block_pos_final.tolist(),
            'message': message
        })
    
    success_rate = successes / num_objects
    return success_rate, results

def collectDataCSV_simpler(target_successes=500, base_filename='dataset_simpler', trajectories_per_file=200, save_frequency=20):
    """
    Collect trajectories using trajectoriesSimpler.py and save to CSV.
    """
    # extract_arm_state, extract_block_state, and extract_action are now imported from training_utils
    
    # Find existing files to determine starting trajectory ID
    pattern = f"{base_filename}_*.csv"
    files = sorted(glob.glob(pattern))
    old_filename = f"{base_filename}.csv"
    if os.path.exists(old_filename):
        files.append(old_filename)
    
    # Count existing trajectories
    max_traj_id = -1
    successes = 0
    if files:
        for filename in files:
            try:
                if os.path.getsize(filename) == 0:
                    continue
                chunk_size = 100000
                for chunk in pd.read_csv(filename, chunksize=chunk_size, usecols=['trajectory_id', 'isDone']):
                    if len(chunk) > 0:
                        max_traj_id = max(max_traj_id, chunk['trajectory_id'].max())
                        # Count successes (trajectories where isDone is True)
                        traj_done = chunk.groupby('trajectory_id')['isDone'].any()
                        successes += traj_done.sum()
            except (pd.errors.EmptyDataError, ValueError):
                continue
    
    # Initialize trajectory ID counter
    trajectory_id_counter = max_traj_id + 1 if max_traj_id != -1 else 0
    
    print(f"Starting data collection for simpler task...")
    print(f"  Existing trajectories: {max_traj_id + 1 if max_traj_id != -1 else 0}")
    print(f"  Existing successes: {successes}")
    print(f"  Target successes: {target_successes}")
    print(f"  Need {max(0, target_successes - successes)} more successful trajectories")
    
    total_collected = 0
    pending_rows = []
    current_file_index = (max_traj_id + 1) // trajectories_per_file if max_traj_id != -1 else 0
    current_filename = f"{base_filename}_{current_file_index * trajectories_per_file}_{(current_file_index + 1) * trajectories_per_file}.csv"
    
    while successes < target_successes:
        total_collected += 1
        print(f"\nCollecting trajectory {total_collected} (Successes: {successes}/{target_successes})...")
        
        # Setup simulation
        model = mujoco.MjModel.from_xml_path("franka_emika_panda/pickAndPlaceSimpler.xml")
        data = mujoco.MjData(model)
        mujoco.mj_kinematics(model, data)
        
        key_id = model.key("home").id
        block_id = model.body('target').id
        site = model.site('palm_contact_edge_vis').id
        blockGeom_id = model.geom('box').id
        joint_names = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"][:7]
        dof_ids = np.array([model.joint(name).id for name in joint_names])
        actuator_names = ['actuator1', 'actuator2', 'actuator3', 'actuator4', 'actuator5', 'actuator6', 'actuator7'][:7]
        actuator_ids = [model.actuator(name).id for name in actuator_names]
        actuator8_id = model.actuator("actuator8").id
        
        # Randomize robot starting position
        qpos_start = sample_robot_start_position(model, dof_ids, key_id)
        data.qpos[:7] = qpos_start[:7]
        mujoco.mj_forward(model, data)
        
        block_jnt_adr = model.body(block_id).jntadr[0]
        block_jnt_id = block_jnt_adr
        block_size = sample_block_size()
        model.geom(blockGeom_id).size[:] = block_size
        targetInitialPos = sample_block_pos()
        
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
        
        trajectory, shutFingers, placingPhaseStart, liftingPhaseStart, liftingPhaseEnd = createTrajectory(
            model, data, block_id, site, blockGeom_id, timesteps=int(numSeconds//dt)
        )
        
        trajectory_rows = []
        steps_on_waypoint = 0
        i = 0
        # Allow enough steps per waypoint (100 steps per waypoint, no cap)
        max_steps = len(trajectory) * 100
        
        for step in range(max_steps):
            if i < len(trajectory):
                desiredPos, desiredQuat = trajectory[i]
                shut = i > shutFingers and i < liftingPhaseStart
            else:
                desiredPos, desiredQuat = trajectory[-1]
                shut = False
            
            # Extract state BEFORE applying action
            qpos = data.qpos.copy()
            qvel = data.qvel.copy()
            ee_pos = data.site_xpos[site].copy()
            site_mat = data.site_xmat[site].reshape(3, 3)
            ee_quat = np.zeros(4)
            mujoco.mju_mat2Quat(ee_quat, site_mat.flatten())
            ee_quat = ee_quat / (np.linalg.norm(ee_quat) + 1e-8)
            
            block_pos = data.xpos[block_id].copy()
            block_quat = data.xquat[block_id].copy()
            block_quat = block_quat / (np.linalg.norm(block_quat) + 1e-8)
            
            arm_state = extract_arm_state(qpos[:7], qvel[:7], ee_pos, ee_quat)
            block_state = extract_block_state(block_pos, block_quat, block_size)
            
            error = moveArm(
                model, data, site, block_id, dof_ids, actuator_ids,
                desiredPos, desiredQuat, shut, trajectory, i, placingPhaseStart, liftingPhaseStart, liftingPhaseEnd
            )
            
            # Extract action AFTER control is set
            action = extract_action(data.ctrl, actuator8_id)
            
            mujoco.mj_step(model, data)
            
            # Determine if trajectory is done
            isDone = (i >= len(trajectory) - 1)
            
            trajectory_rows.append({
                'trajectory_id': trajectory_id_counter,  # Will be updated on success
                'step': step,
                'kinematic_state_arm': ','.join(map(str, arm_state)),
                'kinematic_state_block': ','.join(map(str, block_state)),
                'action': ','.join(map(str, action)),
                'isDone': isDone
            })
            
            error_norm = np.linalg.norm(error)
            if error_norm < 0.03 or steps_on_waypoint > 50:
                i += 1
                steps_on_waypoint = 0
                if i >= len(trajectory):
                    break
            else:
                steps_on_waypoint += 1
        
        # Let block settle after trajectory completes (run physics for a bit)
        # This gives the block time to settle on the table/placement box
        settling_steps = int(1.0 / dt)  # 1 second of settling time
        for _ in range(settling_steps):
            mujoco.mj_step(model, data)
        
        # Check success: block on green placement box
        mujoco.mj_forward(model, data)
        success, block_pos_final, message = verifyPlacementOnGreen(model, data, block_id, placement_box_pos, placement_box_size, tolerance=0.02)
        print(f"  {message}")
        
        if success:
            successes += 1
            # Only increment trajectory ID on success
            trajectory_id_counter += 1
            
            # Update trajectory_id for all rows in this trajectory
            for row in trajectory_rows:
                row['trajectory_id'] = trajectory_id_counter - 1  # Use 0-indexed IDs
            
            pending_rows.extend(trajectory_rows)
            
            # Check if we need to start a new file
            if trajectory_id_counter % trajectories_per_file == 0:
                current_file_index = trajectory_id_counter // trajectories_per_file
                current_filename = f"{base_filename}_{current_file_index * trajectories_per_file}_{(current_file_index + 1) * trajectories_per_file}.csv"
            
            # Save periodically - every save_frequency successful trajectories
            if successes % save_frequency == 0:
                if save_trajectory_rows_to_csv(pending_rows, current_filename, successes):
                    pending_rows = []
        else:
            print(f"  ⚠️  Skipping failed trajectory (not saving to dataset)")
    
    # Final save of any remaining pending rows
    if pending_rows:
        save_trajectory_rows_to_csv(pending_rows, current_filename, successes, verbose=True)

def main():
    # Use different filenames for simpler version
    base_filename = 'dataset_simpler'
    variation = get_variation_from_base_filename(base_filename)
    dataset_dir = get_dataset_path(variation)
    os.makedirs(dataset_dir, exist_ok=True)
    preprocessed_file = os.path.join(dataset_dir, 'dataset_simpler_preprocessed.pkl')
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
        
        # Check current dataset size
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
            data = load_dataset_csv(filename=None, base_filename=base_filename, max_trajectories=None)
            unique_traj_ids = np.unique(data['trajectory_ids'])
            num_trajectories_total = len(unique_traj_ids)
            print(f"Now have {num_trajectories_total} trajectories")
        
        # Load dataset (50% normal + 50% recovery if recovery dataset exists)
        print(f"\nLoading dataset...")
        recovery_filename = 'dataset_simpler_recovery'
        
        # Load normal dataset
        data = load_dataset_csv(filename=None, base_filename=base_filename, max_trajectories=target_trajectories)
        states, actions = get_state_action_pairs(data)
        trajectory_ids_full = data['trajectory_ids']
        unique_traj_ids = np.unique(trajectory_ids_full)
        num_trajectories_loaded = len(unique_traj_ids)
        steps_full = data.get('steps', None)
        
        # Use only the trajectories we need
        traj_ids_to_use = unique_traj_ids[:target_trajectories]
        mask = np.isin(trajectory_ids_full, traj_ids_to_use)
        states_normal = states[mask]
        actions_normal = actions[mask]
        trajectory_ids_normal = trajectory_ids_full[mask]
        steps_normal = steps_full[mask] if steps_full is not None else None
        
        # Try to load recovery dataset
        states_recovery = None
        actions_recovery = None
        trajectory_ids_recovery = None
        steps_recovery = None
        
        try:
            print(f"  Attempting to load recovery dataset from {recovery_filename}...")
            data_recovery = load_dataset_csv(filename=None, base_filename=recovery_filename, 
                                           max_trajectories=target_trajectories)
            states_recovery, actions_recovery = get_state_action_pairs(data_recovery)
            trajectory_ids_recovery_full = data_recovery['trajectory_ids']
            unique_traj_ids_recovery = np.unique(trajectory_ids_recovery_full)
            
            # Use only the trajectories we need
            traj_ids_to_use_recovery = unique_traj_ids_recovery[:target_trajectories]
            mask_recovery = np.isin(trajectory_ids_recovery_full, traj_ids_to_use_recovery)
            states_recovery = states_recovery[mask_recovery]
            actions_recovery = actions_recovery[mask_recovery]
            trajectory_ids_recovery = trajectory_ids_recovery_full[mask_recovery]
            steps_recovery = data_recovery.get('steps', None)
            if steps_recovery is not None:
                steps_recovery = steps_recovery[mask_recovery]
            
            print(f"  ✓ Loaded recovery dataset: {len(states_recovery)} samples from {len(unique_traj_ids_recovery)} trajectories")
        except Exception as e:
            print(f"  ⚠️  Could not load recovery dataset: {e}")
            print(f"  Using normal dataset only")
        
        # Mix 50% normal + 50% recovery if recovery data is available
        if states_recovery is not None and len(states_recovery) > 0:
            # Calculate samples per source for 50/50 split
            num_normal = len(states_normal)
            num_recovery = len(states_recovery)
            samples_per_source = min(num_normal, num_recovery)
            
            # Randomly sample from each if needed
            if num_normal > samples_per_source:
                indices = np.random.choice(num_normal, samples_per_source, replace=False)
                states_normal = states_normal[indices]
                actions_normal = actions_normal[indices]
                trajectory_ids_normal = trajectory_ids_normal[indices]
                if steps_normal is not None:
                    steps_normal = steps_normal[indices]
            
            if num_recovery > samples_per_source:
                indices = np.random.choice(num_recovery, samples_per_source, replace=False)
                states_recovery = states_recovery[indices]
                actions_recovery = actions_recovery[indices]
                trajectory_ids_recovery = trajectory_ids_recovery[indices]
                if steps_recovery is not None:
                    steps_recovery = steps_recovery[indices]
            
            # Offset recovery trajectory IDs
            max_normal_traj_id = trajectory_ids_normal.max() if len(trajectory_ids_normal) > 0 else -1
            trajectory_ids_recovery = trajectory_ids_recovery + max_normal_traj_id + 1
            
            # Concatenate
            states_subset = np.concatenate([states_normal, states_recovery], axis=0)
            actions_subset = np.concatenate([actions_normal, actions_recovery], axis=0)
            trajectory_ids_subset = np.concatenate([trajectory_ids_normal, trajectory_ids_recovery], axis=0)
            
            if steps_normal is not None and steps_recovery is not None:
                steps_subset = np.concatenate([steps_normal, steps_recovery], axis=0)
            elif steps_normal is not None:
                steps_subset = steps_normal
            elif steps_recovery is not None:
                steps_subset = steps_recovery
            else:
                steps_subset = None
            
            print(f"  ✓ Mixed dataset: {len(states_normal)} normal + {len(states_recovery)} recovery = {len(states_subset)} total samples")
        else:
            # Use normal dataset only
            states_subset = states_normal
            actions_subset = actions_normal
            trajectory_ids_subset = trajectory_ids_normal
            steps_subset = steps_normal
            print(f"  ✓ Loaded {len(states_subset)} state-action pairs from {num_trajectories_loaded} unique trajectories")
        
        # Training parameters
        num_samples = len(states_subset)
        
        # Adaptive batch size
        if device.type == 'mps':
            if num_samples > 1000000:
                batch_size = 16384
            elif num_samples > 500000:
                batch_size = 8192
            elif num_samples > 100000:
                batch_size = 4096
            elif num_samples > 50000:
                batch_size = 2048
            elif num_samples > 20000:
                batch_size = 1024
            else:
                batch_size = 512
        else:
            if num_samples > 1000000:
                batch_size = 4096
            elif num_samples > 500000:
                batch_size = 2048
            elif num_samples > 100000:
                batch_size = 1024
            elif num_samples > 50000:
                batch_size = 512
            elif num_samples > 20000:
                batch_size = 384
            else:
                batch_size = 256
        
        epochs = 100
        
        # Determine pretrained model path (use previous best model)
        pretrained_model_path = best_model_path
        num_prev_trajectories = 0
        if pretrained_model_path and os.path.exists(pretrained_model_path):
            # Count how many trajectories were used in the previous model
            for result in results:
                if result.get('model_filename') == os.path.basename(pretrained_model_path):
                    num_prev_trajectories = result.get('num_trajectories', 0)
                    break
        
        # Train model
        print(f"\nTraining model...")
        model = train_model(
            states_subset, actions_subset, trajectory_ids=trajectory_ids_subset, steps=steps_subset,
            epochs=epochs, batch_size=batch_size, lr=0.001,
            patience=15,
            pretrained_model_path=pretrained_model_path,
            num_prev_trajectories=num_prev_trajectories,
            weight_early_steps=True,
            early_step_threshold=4000
        )
        
        # Save model immediately after training (before evaluation, in case evaluation hangs)
        model_filename_temp = f"model_simpler_{target_trajectories}traj_temp.pth"
        torch.save(model.state_dict(), model_filename_temp)
        print(f"  ✓ Saved trained model to {model_filename_temp} (before evaluation)")
        
        # Evaluate model
        print(f"\nEvaluating model...")
        success_rate, eval_results = evaluate_model(model, num_objects=20)
        
        # Save model with descriptive name (includes success rate)
        model_filename = f"model_simpler_{target_trajectories}traj_{int(success_rate*100)}success.pth"
        torch.save(model.state_dict(), model_filename)
        print(f"  ✓ Saved model to {model_filename}")
        
        # Remove temporary file if final save succeeded
        if os.path.exists(model_filename_temp):
            try:
                os.remove(model_filename_temp)
                print(f"  ✓ Removed temporary model file")
            except:
                pass
        
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

