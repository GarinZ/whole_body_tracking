isaaclab -p scripts/rsl_rl/train.py \
--task=Tracking-Flat-PM01Edu-v0 \
--registry_name garinzh-mimic-org/wandb-registry-motions/Taichi \
--headless \
--logger wandb \
--log_project_name whole-body-tracking \
--run_name pm01_taichi_run2_8192 \
--max_iterations 3000 \
--num_envs 8192