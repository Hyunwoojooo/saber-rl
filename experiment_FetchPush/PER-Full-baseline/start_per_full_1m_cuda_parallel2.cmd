@echo off
powershell -ExecutionPolicy Bypass -File "%~dp0run_per_full_1m_cuda_parallel2.ps1" -MaxParallel 2 -Device cuda

