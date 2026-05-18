@echo off
call conda activate gdrl
tensorboard --logdir=logs --port=6099
pause
