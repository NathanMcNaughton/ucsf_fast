# This script is meant to be run in the command line, not submitted via sbatch

source /usr/local/linux/mambaforge-3.11/bin/activate ucsf_env
cd /accounts/campus/austin.zane/ucsf_fast

python \
    -m experiments.train_classification \
    --proj resnet_test_05_19_24 \
    --seg_proj unet_test_05_18_24 \
    --seg_run scarlet-water-11-2l4vc76j \
    --k 1 \
    --n_epochs 2 \
    --batch_size 2 \
    --learning_rate 0.0003 \
    --cutoff 0.5 \
    --output_saving \
    --loss_weight 10.0 \
