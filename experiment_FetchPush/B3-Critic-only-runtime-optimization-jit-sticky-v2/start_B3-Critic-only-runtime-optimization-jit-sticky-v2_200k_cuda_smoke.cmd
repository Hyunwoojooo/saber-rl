@echo off
setlocal

cd /d C:\Users\SS\Projects\metacog-rl\metacog-rl

set EXP_ROOT=C:\Users\SS\Projects\metacog-rl\metacog-rl\version5.0\metacog-lite-v0\experiment_FetchPush\B3-Critic-only-runtime-optimization-jit-sticky-v2
set CODE_ROOT=%EXP_ROOT%\code\src
set SCRIPT=%CODE_ROOT%\train_fetch_push_lite_v0.py
set OUTPUT_ROOT=%EXP_ROOT%\smoke_results
set LOG_ROOT=%EXP_ROOT%\logs
set RUN_NAME=fetch_push_b3_critic_only_runtime_opt_jit_sticky_random_v2_s1_200000_ls512_cuda_smoke

set PYTHONPATH=%CODE_ROOT%
set OMP_NUM_THREADS=1
set MKL_NUM_THREADS=1
set OPENBLAS_NUM_THREADS=1
set NUMEXPR_NUM_THREADS=1

if not exist "%OUTPUT_ROOT%" mkdir "%OUTPUT_ROOT%"
if not exist "%LOG_ROOT%" mkdir "%LOG_ROOT%"

C:\Python312\python.exe "%SCRIPT%" ^
  --run_name "%RUN_NAME%" ^
  --output_root "%OUTPUT_ROOT%" ^
  --method b3_support_only ^
  --reward_type sparse ^
  --total_steps 200000 ^
  --start_steps 10000 ^
  --warmup_steps 5000 ^
  --updates_per_step 1 ^
  --batch_size 256 ^
  --buffer_capacity 300000 ^
  --her_ratio 0.8 ^
  --M_B 32 ^
  --M_Q_min 8 ^
  --N_local_min 8 ^
  --local_search_max 512 ^
  --eval_freq 10000 ^
  --eval_episodes 20 ^
  --checkpoint_every_steps 50000 ^
  --log_every_steps 1000 ^
  --N_gap_min 256 ^
  --weighting_mode batch_norm ^
  --weight_scope critic_only ^
  --device cuda ^
  --env_seed 1 ^
  --torch_seed 1 ^
  --numpy_seed 1 ^
  --action_noise_seed 1 ^
  --weight_seed 20001 ^
  --eval_seed 10001 ^
  --support_lookup_optimization ^
  --support_scan_mode sticky_random_v1 ^
  --support_scan_seed 30001 ^
  --support_local_scan_cache_enabled ^
  --support_cache_max_entries 500000 ^
  --support_local_scan_cache_max_entries 200000 ^
  --support_numba_scan_enabled ^
  --no_support_batch_query_coalescing ^
  --no_save_step_checkpoints ^
  > "%LOG_ROOT%\%RUN_NAME%.out.log" 2> "%LOG_ROOT%\%RUN_NAME%.err.log"

exit /b %ERRORLEVEL%
