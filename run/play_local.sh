isaaclab -p scripts/rsl_rl/play.py \
    --task=Tracking-Flat-PM01Edu-v0 \
    --load_run=2026-04-24_15-01-32_pm01_taichi_continue_2000 \
    --num_envs=2 \
    --checkpoint=model_4998.pt \
    --motion_file=/home/engineai/Project/mimic/whole_body_tracking/artifacts/Taichi:v0/motion.npz