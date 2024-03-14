#!/bin/bash

srun --pty --partition=gpu --gres=gpu:1 -t 4:00:00 /bin/bash

source /usr/local/linux/mambaforge-3.11/bin/activate ucsf_env
cd /accounts/campus/austin.zane/ucsf_fast