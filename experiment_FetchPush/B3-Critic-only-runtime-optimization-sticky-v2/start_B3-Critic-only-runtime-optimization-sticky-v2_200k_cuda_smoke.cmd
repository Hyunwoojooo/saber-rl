@echo off
setlocal

cd /d C:\Users\SS\Projects\metacog-rl\metacog-rl

set EXP_ROOT=C:\Users\SS\Projects\metacog-rl\metacog-rl\version5.0\metacog-lite-v0\experiment_FetchPush\B3-Critic-only-runtime-optimization-sticky-v2
set CODE_ROOT=%EXP_ROOT%\code\src
set SCRIPT=%CODE_ROOT%\train_fetch_push_lite_v0.py
set OUTPUT_ROOT=%EXP_ROOT%\smoke_results
set RUN_NAME=fetch_push_b3_critic_only_runtime_opt_sticky_random_v2_s1_200000_ls512_cuda_smoke

set PYTHONPATH=%CODE_ROOT%

python "%SCRIPT%" ^
  --run_name "%RUN_NAME%" ^
  --output_root "%OUTPUT_ROOT%" ^
  --method b3_support_only ^
  --weight_scope critic_only ^
  --total_steps 200000 ^
  --local_search_max 512 ^
  --device cuda ^
  --support_lookup_optimization ^
  --support_scan_mode sticky_random_v1 ^
  --support_scan_seed 30001 ^
  --support_cache_max_entries 500000 ^
  --support_local_scan_cache_max_entries 200000 ^
  --eval_freq 10000 ^
  --eval_episodes 20 ^
  --checkpoint_every_steps 50000 ^
  --log_every_steps 1000 ^
  --env_seed 1 ^
  --torch_seed 1 ^
  --numpy_seed 1 ^
  --action_noise_seed 1 ^
  --weight_seed 31 ^
  --eval_seed 10001

exit /b %ERRORLEVEL%
