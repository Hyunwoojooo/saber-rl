@echo off
setlocal
cd /d "%~dp0"
python run.py --seed 1 --steps 100000 --device cuda
endlocal
