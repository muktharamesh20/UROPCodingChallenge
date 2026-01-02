# Franka Panda Pick-and-Place Learning

A machine learning pipeline for training neural network controllers to perform pick-and-place tasks with a Franka Emika Panda robot arm in MuJoCo simulation. This project implements various action space representations and training strategies to learn manipulation policies.

## Overview

This project trains MLP (Multi-Layer Perceptron) models to control a 7-DOF Franka Panda robot arm for pick-and-place tasks. The system supports multiple action space representations including:

- **Joint-space actions**: Direct joint position commands
- **Delta actions**: Joint position deltas from current state
- **Task-space actions**: End-effector pose deltas (position + quaternion)
- **Task-space absolute actions**: Absolute end-effector poses

Models are trained using imitation learning from expert demonstrations, with incremental data collection and progressive training strategies.

## Project Structure

```
├── pipelines/              # Training and evaluation pipelines for each variation
│   ├── train_and_evaluate_pipeline.py          # Default pipeline
│   ├── train_and_evaluate_pipeline_delta.py    # Delta actions
│   ├── train_and_evaluate_pipeline_direct.py   # Direct joint-space
│   ├── train_and_evaluate_pipeline_taskSpace.py # Task-space deltas
│   └── ...
├── trajectories/           # Expert trajectory generation scripts
│   ├── trajectories.py
│   ├── trajectoriesTaskSpace.py
│   └── ...
├── utils/                  # Shared utilities
│   ├── training_utils.py   # Unified training and evaluation functions
│   ├── ik_utils.py         # Inverse kinematics utilities
│   ├── path_utils.py       # File path management
│   └── ...
├── data/                   # Generated datasets, models, and results (gitignored)
│   └── {variation}/
│       ├── datasets/       # CSV trajectory datasets
│       ├── models/         # Trained PyTorch models (.pth)
│       └── scalers/        # State/action scalers (.pkl)
├── franka_emika_panda/     # MuJoCo robot model and assets
│   ├── pickAndPlace.xml
│   ├── pickAndPlaceSimpler.xml
│   └── pickAndPlaceDirect.xml
├── visualize_model.py      # Interactive model visualization
└── visualize_trajectories.py # Dataset trajectory visualization
```

## Features

- **Multiple Action Spaces**: Support for joint-space, delta, and task-space action representations
- **Progressive Training**: Incremental data collection with fine-tuning from previous models
- **Early Stopping**: Validation-based early stopping to prevent overfitting
- **State Normalization**: Optional state and action scaling for improved training
- **Visualization Tools**: Interactive visualization of trained models and recorded trajectories
- **Organized Data Storage**: Variation-specific folders for datasets, models, and scalers

## Installation

### Requirements

```bash
pip install mujoco>=3.1.0
pip install torch
pip install numpy scipy scikit-learn pandas matplotlib
```

### MuJoCo Setup

The project uses MuJoCo for physics simulation. Make sure MuJoCo is properly installed and licensed.

## Usage

### Training a Model

Each variation has its own training pipeline. For example, to train a task-space model:

```bash
python pipelines/train_and_evaluate_pipeline_taskSpace.py
```

The pipeline will:
1. Collect expert trajectory data incrementally
2. Train models with increasing dataset sizes
3. Evaluate success rates on test scenarios
4. Save models and training results

### Visualizing a Trained Model

```bash
python visualize_model.py
```

This will:
- List all available trained models
- Allow you to select a model to visualize
- Run the model in simulation with interactive controls

### Visualizing Recorded Trajectories

```bash
python visualize_trajectories.py
```

## Variations/Iterations

The project includes several variations of the pick-and-place task:

| Variation | Action Space | Description |
|-----------|-------------|-------------|
| `default` | Joint-space absolute | Original implementation |
| `delta` | Joint-space delta | Joint position deltas |
| `direct` | Joint-space absolute | Direct trajectory to pick location (no intermediate waypoint) with sin/cos state encoding |
| `simpler` | Joint-space absolute | Simplified task with random robot arm starting positions |
| `even_simpler` | Joint-space absolute | Simplified task without random start positions |
| `taskspace` | Task-space delta | End-effector pose deltas (3D pos + 3D quat + gripper) |
| `taskspace_absolutes` | Task-space absolute | Absolute end-effector poses |
| `random_start` | Joint-space absolute | Random robot arm starting configurations |
| `taskspace_absolutes_random_start` | Task-space absolute | Absolute poses with random robot arm starting positions |

## Training Details

- **Model Architecture**: MLP (Multi-Layer Perceptron) with ReLU activations
- **Loss Function**: MSE (Mean Squared Error)
- **Optimizer**: Adam
- **Early Stopping**: Patience-based with validation loss monitoring
- **Learning Rate Scheduling**: Optional (ReduceLROnPlateau, StepLR, ExponentialLR, CosineAnnealingLR)
- **Batch Size**: Adaptive based on available memory
- **State Representation**: 31D (raw) or 38D (sin/cos encoded joint positions)

## Data Organization

All generated data is stored in `data/{variation}/` folders:
- `datasets/`: CSV files with trajectory data
- `models/`: Trained PyTorch model checkpoints
- `scalers/`: State and action normalization scalers

These directories are gitignored to keep the repository size manageable.

## Acknowledgments

- Robot model adapted from [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie)
- Franka Emika Panda model derived from [franka_ros](https://github.com/frankaemika/franka_ros)

## License

See individual LICENSE files in the `franka_emika_panda/` directory for robot model licensing.
