# rsl-rl >= 4.0.0 Compatibility Fixes

## Overview

This document describes the modifications made to support **rsl-rl >= 4.0.0** in the whole_body_tracking project. The new rsl-rl version introduced breaking API changes that required updates across multiple files.

## API Changes in rsl-rl >= 4.0.0

| Old API (rsl-rl < 4.0.0) | New API (rsl-rl >= 4.0.0) |
|---------------------------|----------------------------|
| `self.alg.policy` | `self.alg.actor` |
| `self.obs_normalizer` (on runner) | `self.alg.actor.obs_normalizer` (on model) |
| `policy.actor` (wrapper) | `actor.mlp` (actual network) |
| `policy.is_recurrent` | `actor.is_recurrent` |
| Config: `policy` | Config: `actor` + `critic` |

## Modified Files

### 1. `isaaclab_rl/rsl_rl/exporter.py` (Isaac Lab, external)

**Change:** Added support for MLPModel direct export in `_OnnxPolicyExporter` and `_TorchPolicyExporter`.

```python
# Added elif branch for rsl-rl >= 4.0.0
if hasattr(policy, "actor"):
    self.actor = copy.deepcopy(policy.actor)
elif hasattr(policy, "student"):
    self.actor = copy.deepcopy(policy.student)
elif hasattr(policy, "mlp"):  # NEW: rsl-rl >= 4.0.0
    self.actor = copy.deepcopy(policy.mlp)
else:
    raise ValueError("Policy does not have an actor/student module.")
```

**Reason:** In rsl-rl >= 4.0.0, `self.alg.actor` is an MLPModel (not a wrapper), and the actual neural network is in `.mlp`. The exporter now correctly detects this and extracts `policy.mlp`.

---

### 2. `scripts/rsl_rl/train.py`

**Changes:**
1. Added `handle_deprecated_rsl_rl_cfg` import and call to convert old `policy` config to new `actor`/`critic` format
2. Added rsl-rl version checking
3. Fixed pickle import (replaced deprecated `dump_pickle` with standard `pickle.dump`)

```python
from importlib.metadata import version as get_rsl_rl_version
from packaging import version
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg

# Check minimum supported rsl-rl version
RSL_RL_VERSION = "3.0.1"
installed_version = get_rsl_rl_version("rsl-rl-lib")
if version.parse(installed_version) < version.parse(RSL_RL_VERSION):
    print(f"Please install the correct version of RSL-RL...")

# Inside main():
agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)
```

**Reason:** The `handle_deprecated_rsl_rl_cfg` function converts the old `policy` configuration field to the new `actor`/`critic` format required by rsl-rl >= 4.0.0.

---

### 3. `scripts/rsl_rl/play.py`

**Changes:**
1. Added `handle_deprecated_rsl_rl_cfg` import and call
2. Added `get_rsl_rl_version` import
3. Updated ONNX export to use new API

```python
from importlib.metadata import version as get_rsl_rl_version
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg

# Inside main():
installed_version = get_rsl_rl_version("rsl-rl-lib")
agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)

# ONNX export (old → new API):
export_motion_policy_as_onnx(
    env.unwrapped,
    ppo_runner.alg.actor,  # was: ppo_runner.alg.policy
    normalizer=getattr(ppo_runner.alg.actor, "obs_normalizer", None),  # was: ppo_runner.obs_normalizer
    path=export_model_dir,
    filename="policy.onnx",
)
```

**Reason:** Same as train.py - need to handle the deprecated config format and use the new actor/critic API.

---

### 4. `source/whole_body_tracking/whole_body_tracking/utils/my_on_policy_runner.py`

**Changes:** Updated both `MyOnPolicyRunner.save()` and `MotionOnPolicyRunner.save()` methods.

```python
# Old (broken):
if self.logger_type in ["wandb"]:  # WRONG: logger_type is on self.logger
    ...
    export_policy_as_onnx(self.alg.policy, normalizer=self.obs_normalizer, ...)

# New (fixed):
if self.logger.logger_type in ["wandb"]:  # CORRECT: logger_type is on self.logger
    ...
    export_policy_as_onnx(self.alg.actor, normalizer=getattr(self.alg.actor, "obs_normalizer", None), ...)
```

**Changes in detail:**
| Location | Old | New |
|----------|-----|-----|
| `self.logger_type` | `self.logger.logger_type` | Access logger_type from Logger object |
| `self.alg.policy` | `self.alg.actor` | New algorithm API |
| `self.obs_normalizer` | `getattr(self.alg.actor, "obs_normalizer", None)` | Normalizer on MLPModel |

---

### 5. `scripts/replay_npz.py`

**Change:** Added `--robot` CLI argument for multi-robot support.

```python
parser.add_argument("--robot", type=str, default="g1", choices=["g1", "pm01_edu"],
                    help="Robot to use for motion replay.")

ROBOT_CONFIGS = {
    "g1": G1_CYLINDER_CFG,
    "pm01_edu": PM01_EDU_CFG,
}

# In main():
scene_cfg.robot = ROBOT_CONFIGS[args_cli.robot].replace(prim_path="{ENV_REGEX_NS}/Robot")
```

**Reason:** The replay script previously hardcoded G1 robot. Now it supports selecting between G1 and pm01_edu.

---

## pm01_edu Environment Configuration

The pm01_edu environment was also configured with the following overrides in `flat_env_cfg.py`:

```python
def __post_init__(self):
    super().__post_init__()
    self.scene.robot = PM01_EDU_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    self.actions.joint_pos.scale = PM01_EDU_ACTION_SCALE
    self.commands.motion.anchor_body_name = "LINK_TORSO_YAW"
    self.commands.motion.body_names = [
        "LINK_HIP_YAW_L",       # pelvis root (left leg)
        "LINK_HIP_ROLL_L",      # left hip
        "LINK_KNEE_PITCH_L",    # left knee
        "LINK_ANKLE_ROLL_L",    # left ankle
        "LINK_HIP_ROLL_R",      # right hip
        "LINK_KNEE_PITCH_R",    # right knee
        "LINK_ANKLE_ROLL_R",    # right ankle
        "LINK_TORSO_YAW",       # torso anchor
        "LINK_SHOULDER_ROLL_L", # left shoulder
        "LINK_ELBOW_YAW_L",     # left elbow end-effector
        "LINK_SHOULDER_ROLL_R", # right shoulder
        "LINK_ELBOW_YAW_R",     # right elbow end-effector
    ]
    # Override base_com to use LINK_TORSO_YAW instead of torso_link
    self.events.base_com.params["asset_cfg"].body_names = "LINK_TORSO_YAW"
```

**Reason:** The base `TrackingEnvCfg` references `torso_link` which doesn't exist in pm01_edu. The pm01_edu uses `LINK_TORSO_YAW` as its torso anchor.

---

## Testing

To verify the fixes work:

```bash
# Training with pm01_edu
isaaclab -p scripts/rsl_rl/train.py \
  --task=Tracking-Flat-PM01Edu-v0 \
  --registry_name garinzh-mimic-org/wandb-registry-Motions/Taichi \
  --headless --logger wandb \
  --log_project_name whole-body-tracking \
  --run_name pm01_taichi_run1 \
  --max_iterations 50

# Replay with pm01_edu
python scripts/replay_npz.py \
  --registry_name garinzh-mimic-org/wandb-registry-Motions/Taichi \
  --robot pm01_edu
```

---

## Summary of All Modified Files

| File | Changes |
|------|---------|
| `isaaclab_rl/rsl_rl/exporter.py` | Added `elif hasattr(policy, "mlp")` branch in exporters |
| `scripts/rsl_rl/train.py` | Added handle_deprecated_rsl_rl_cfg, version check, fixed pickle |
| `scripts/rsl_rl/play.py` | Added handle_deprecated_rsl_rl_cfg, updated ONNX export API |
| `scripts/replay_npz.py` | Added --robot argument for multi-robot support |
| `my_on_policy_runner.py` | Fixed logger_type access, updated to new actor/normalizer API |
| `flat_env_cfg.py` (pm01_edu) | Added body_names, anchor_body_name, base_com overrides |
