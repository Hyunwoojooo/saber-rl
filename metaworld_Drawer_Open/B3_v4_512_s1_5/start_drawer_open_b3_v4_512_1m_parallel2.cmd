@echo off
setlocal
cd /d "%~dp0"
powershell -ExecutionPolicy Bypass -File "%~dp0run_drawer_open_b3_v4_512_1m_parallel2.ps1"
endlocal
