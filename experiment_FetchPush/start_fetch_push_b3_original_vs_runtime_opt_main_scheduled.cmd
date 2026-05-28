@echo off
cd /d C:\Users\SS\Projects\metacog-rl\metacog-rl
C:\Python312\python.exe "C:\Users\SS\Projects\metacog-rl\metacog-rl\version5.0\metacog-lite-v0\experiment\run_fetch_push_b3_original_vs_runtime_opt_parallel4.py" --max_parallel 4 --device cpu --total_steps 1000000 > "C:\Users\SS\Projects\metacog-rl\metacog-rl\version5.0\metacog-lite-v0\experiment\scheduler_launcher.out.log" 2> "C:\Users\SS\Projects\metacog-rl\metacog-rl\version5.0\metacog-lite-v0\experiment\scheduler_launcher.err.log"
