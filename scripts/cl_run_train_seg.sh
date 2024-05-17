# This script is meant to be run in the command line, not submitted via sbatch

source /usr/local/linux/mambaforge-3.11/bin/activate ucsf_env
cd /accounts/campus/austin.zane/ucsf_fast

python \
    -m experiments.train_segmentation \
    --proj unet_test_05_16_24 \
    --k 1 \
    --output_saving \
    --n_epochs 3 \
    --batch_size 10 \
    --loss_weight 10 \
    --learning_rate 0.0003 \
