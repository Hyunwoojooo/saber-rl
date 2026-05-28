@echo off
cd /d C:\Users\SS\Projects\metacog-rl\metacog-rl
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "C:\Users\SS\Projects\metacog-rl\metacog-rl\version5.0\metacog-lite-v0\experiment\run_fetch_push_b3_original_vs_runtime_opt_parallel4.ps1" -MaxParallel 4 -Device cpu -TotalSteps 1000000 >> "C:\Users\SS\Projects\metacog-rl\metacog-rl\version5.0\metacog-lite-v0\experiment\launcher_wrapper.out.log" 2>> "C:\Users\SS\Projects\metacog-rl\metacog-rl\version5.0\metacog-lite-v0\experiment\launcher_wrapper.err.log"
