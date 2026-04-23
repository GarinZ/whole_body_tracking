# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

BeyondMimic is a humanoid motion tracking framework using Isaac Lab for training PPO policies. It tracks motion from reference data and produces control policies for humanoid robots (G1, Unitree).

## Common Commands

### Installation
```bash
pip install -e source/whole_body_tracking
```

### Motion Preprocessing
```bash
python scripts/csv_to_npz.py --input_file {motion_name}.csv --input_fps 30 --output_name {motion_name} --headless
```

### Replay Motion (debugging)
```bash
python scripts/replay_npz.py --registry_name={org}-org/wandb-registry-motions/{motion_name}
```

### Policy Training
```bash
python scripts/rsl_rl/train.py --task=Tracking-Flat-G1-v0 \
  --registry_name {org}-org/wandb-registry-motions/{motion_name} \
  --headless --logger wandb --log_project_name {project_name} --run_name {run_name}
```

### Policy Evaluation
```bash
python scripts/rsl_rl/play.py --task=Tracking-Flat-G1-v0 --num_envs=2 --wandb_path={wandb-run-path}
```

### Linting (pre-commit)
```bash
pre-commit run --all-files
```

## Architecture

### MDP Modules (`tasks/tracking/mdp/`)
The MDP (Markov Decision Process) is decomposed into atomic modules:
- **commands.py** - Motion commands, pose/velocity error computation, adaptive sampling, initial state randomization
- **rewards.py** - DeepMimic reward functions and smoothing terms
- **observations.py** - Policy and critic observation terms
- **events.py** - Domain randomization (physics material, joint defaults, push forces)
- **terminations.py** - Early termination conditions and timeouts

### Environment Configuration (`tracking_env_cfg.py`)
Orchestrates all MDP components via Isaac Lab's ManagerBasedRLEnvCfg. Defines scene (terrain, robot, lights, contact sensors), observations, actions, commands, rewards, terminations, and events.

### Robot Configs (`robots/`)
- **g1.py** - G1 humanoid armatures, joint stiffness/damping, action scaling
- **actuator.py** - Actuator model configuration
- **smpl.py** - SMPL body model parameters

### Training Scripts (`scripts/rsl_rl/`)
- **train.py** - PPO training entry point using Isaac Lab's RL infrastructure
- **play.py** - Policy evaluation and playback
- **cli_args.py** - Shared CLI argument handling

### Agent Configs (`config/{g1,humanoid}/agents/rsl_rl_ppo_cfg.py`)
PPO hyperparameters (learning rates, batch sizes, entropy coefficients, etc.)

## Key Patterns

- MDP functions follow Isaac Lab's `ObsTerm`, `RewTerm`, `EventTerm`, `DoneTerm` config pattern
- Motion commands use `MotionCommandCfg` with pose/velocity ranges and resampling time
- Robot assets are defined via `ArticulationCfg` with joint and body configurations
- All observation terms can optionally add `Unoise` for stochasticity

## Environment Variables
- `WANDB_ENTITY` - Required organization name for wandb registry access
