@echo off
set ROOT=C:\Users\SS\Projects\metacog-rl\metacog-rl
set EXP=%ROOT%\version5.0\metacog-lite-v0\experiment_FetchPush\PER-Full-baseline
set PYTHONPATH=%EXP%\code\src
C:\Python312\python.exe "%EXP%\code\src\train_fetch_push_lite_v0.py" --run_name fetch_push_per_full_s1_100k_cuda_smoke --output_root "%EXP%\smoke_results" --method per_full_baseline --reward_type sparse --total_steps 100000 --start_steps 10000 --warmup_steps 5000 --updates_per_step 1 --batch_size 256 --buffer_capacity 300000 --her_ratio 0.8 --eval_freq 10000 --eval_episodes 20 --checkpoint_every_steps 50000 --log_every_steps 1000 --weighting_mode batch_norm --weight_scope critic_only --per_alpha 0.6 --per_beta0 0.4 --per_beta1 1.0 --eps_PER 1e-6 --U_PER_max 2.0 --device cuda --env_seed 1 --torch_seed 1 --numpy_seed 1 --action_noise_seed 1 --weight_seed 20001 --eval_seed 10001 --no_save_step_checkpoints

