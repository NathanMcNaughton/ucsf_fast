#!/bin/bash
#SBATCH --job-name=unet_test          # Job name
#SBATCH -o ./out/unet_test.%j.log   # define stdout filename; %j expands to jobid; to redirect stderr elsewhere, duplicate this line with -e instead
#
#SBATCH --mail-user=austin.zane@berkeley.edu
#SBATCH --mail-type=FAIL,TIME_LIMIT # get notified via email on job failure or time limit reached
#
#SBATCH -p yugroup
#SBATCH --gres=gpu:1
#SBATCH -t 02:00:00       # set maximum run time in H:M:S


source /usr/local/linux/mambaforge-3.11/bin/activate ucsf_env
cd /accounts/campus/austin.zane/ucsf_fast

python \
    -m experiments.train_segmentation \

