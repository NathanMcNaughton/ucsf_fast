#!/bin/bash
#SBATCH --job-name=resnet_test          # Job name
#SBATCH -o ./out/resnet_test.%j.log   # define stdout filename; %j expands to jobid; to redirect stderr elsewhere, duplicate this line with -e instead
#
#SBATCH --mail-user=austin.zane@berkeley.edu
#SBATCH --mail-type=FAIL,TIME_LIMIT # get notified via email on job failure or time limit reached
#
#SBATCH -p yugroup
#SBATCH --gres=gpu:1
#SBATCH -t 24:00:00       # set maximum run time in H:M:S


source /usr/local/linux/mambaforge-3.11/bin/activate ucsf_env
cd /accounts/campus/austin.zane/ucsf_fast

python \
    -m experiments.train_classification \
    --proj resnet_test_05_20_24 \
    --seg_proj unet_test_05_18_24 \
    --seg_run scarlet-water-11-2l4vc76j \
    --k 10 \
    --n_epochs 200 \
    --batch_size 3 \
    --learning_rate 0.0003 \
    --cutoff 0.5 \
    --output_saving \
    --loss_weight 30.0 \

